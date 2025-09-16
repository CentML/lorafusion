# ruff: noqa: N806
"""Test triton Fused Multi-LoRA xw + sb."""

from functools import partial
from typing import Any

import torch
from loguru import logger

from lorafusion.ops.multi_lora import prepare_multi_lora_batch_info
from lorafusion.ops.triton_ops.config import get_lora_kernel_config
from lorafusion.ops.triton_ops.fused_multi_lora_dyw_dsa import (
    construct_ds_and_a_ptrs_list,
    fused_multi_lora_dyw_dsa,
)
from lorafusion.ops.triton_ops.fused_multi_lora_xw_sb import construct_s_and_b_ptrs_list
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds


def prepare_func(
    seq_len_list: list[int],
    lora_idx_list: list[int],
    lora_rank_list: list[int],
    dropout_p_list: list[float],
    alpha_list: list[float],
    n: int,
    k: int,
    block_size_m: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_bias: bool = False,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA xw + sb kernel."""
    multi_lora_batch_info = prepare_multi_lora_batch_info(
        seq_len_list=seq_len_list,
        lora_idx_list=lora_idx_list,
        lora_rank_list=lora_rank_list,
        dropout_p_list=dropout_p_list,
        alpha_list=alpha_list,
        block_size_m=block_size_m,
        output_dtype=dtype,
    )
    padded_seq_len_list = multi_lora_batch_info.padded_seq_len_list
    block_to_lookup_table = multi_lora_batch_info.block_to_lookup_table
    block_to_dropout_p = multi_lora_batch_info.block_to_dropout_p
    block_to_alpha = multi_lora_batch_info.block_to_alpha
    enable_dropout = multi_lora_batch_info.enable_dropout
    same_dropout_p_value = multi_lora_batch_info.same_dropout_p_value
    max_r = multi_lora_batch_info.max_r

    padded_x = torch.rand(sum(padded_seq_len_list), k, device="cuda", dtype=dtype) / 10
    linear_w = torch.rand(n, k, device="cuda", dtype=dtype) / 10
    bias = torch.rand(n, device="cuda", dtype=dtype) / 10 if with_bias else None

    lora_a_list = [
        torch.rand(
            lora_rank,
            k,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        for lora_rank in multi_lora_batch_info.lora_rank_list
    ]
    lora_b_list = [
        torch.rand(
            n,
            lora_rank,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        for lora_rank in multi_lora_batch_info.lora_rank_list
    ]

    # Calculate the s list for each block
    full_s = padded_x @ (torch.cat(lora_a_list, dim=0).T)
    curr_m, curr_r = 0, 0
    s_list = []
    for padded_seq_len, lora_a_tensor in zip(
        padded_seq_len_list, lora_a_list, strict=True
    ):
        lora_rank = lora_a_tensor.shape[0]
        s_list.append(
            full_s[curr_m : curr_m + padded_seq_len, curr_r : curr_r + lora_rank]
        )
        curr_m += padded_seq_len
        curr_r += lora_rank

    # s stride / total r dim
    total_r = full_s.shape[1]

    # Construct the s and b ptrs list
    s_ptrs_list, b_ptrs_list, _, _ = construct_s_and_b_ptrs_list(
        raw_s_list=s_list,
        raw_b_list=lora_b_list,
        block_size_m=block_size_m,
    )

    # Construct dy
    dy = torch.rand(sum(padded_seq_len_list), n, device="cuda", dtype=dtype) / 10

    # Construct ds
    raw_ds_list = [torch.rand_like(s_tensor) for s_tensor in s_list]

    # Dropout mask
    # In this test, we want to make sure the dropout_p is the same for all the adapters.
    dropout_p_set = set(dropout_p_list)
    if len(dropout_p_set) != 1:
        msg = f"Incompatible dropout_p_list: {dropout_p_list} in the test."
        raise ValueError(msg)
    dropout_p = dropout_p_set.pop()

    dropout_mask = (
        torch.randn(sum(padded_seq_len_list), k, device="cuda", dtype=torch.float32)
        > dropout_p
    )

    return {
        "x": padded_x,
        "w": linear_w,
        "dy": dy,
        "raw_ds_list": raw_ds_list,
        "raw_a_list": lora_a_list,
        "s_ptrs_list": s_ptrs_list,
        "b_ptrs_list": b_ptrs_list,
        "raw_s_list": s_list,
        "raw_b_list": lora_b_list,
        "dropout_mask": dropout_mask,
        "enable_dropout": enable_dropout,
        "block_to_lookup_table": block_to_lookup_table,
        "block_to_dropout_p": block_to_dropout_p,
        "block_to_alpha": block_to_alpha,
        "bias": bias,
        "max_r": max_r,
        "total_r": total_r,
        "block_size_m": block_size_m,
        "seq_len_list": padded_seq_len_list,
        "padded_seq_len_list": padded_seq_len_list,
        "lora_idx_list": lora_idx_list,
        "lora_rank_list": lora_rank_list,
        "dropout_p_list": dropout_p_list,
        "alpha_list": alpha_list,
    }


def torch_dyw_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of dy @ w."""
    return dy @ w


def torch_multi_lora_dyw_dsa_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    raw_ds_list: list[torch.Tensor],
    raw_a_list: list[torch.Tensor],
    block_to_lookup_table: torch.Tensor,
    block_to_alpha: torch.Tensor,
    block_size_m: int,
    seq_len_list: list[int],
    padded_seq_len_list: list[int],
    dropout_p_list: list[float],
    alpha_list: list[float],
    dropout_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA dyw + dsa."""
    M, N = dy.shape
    N, K = w.shape
    out = torch.zeros((M, K), device=dy.device, dtype=dy.dtype)

    num_loras = len(seq_len_list)

    curr_start = 0
    for i in range(num_loras):
        seq_len = seq_len_list[i]
        padded_seq_len = padded_seq_len_list[i]
        raw_ds = raw_ds_list[i]
        raw_a = raw_a_list[i]
        dropout_p = dropout_p_list[i]
        alpha = alpha_list[i]

        # For x
        curr_dy = dy[curr_start : curr_start + seq_len, :]
        curr_dyw = curr_dy @ w

        # For s and w
        dropout_mask_curr = dropout_mask[curr_start : curr_start + seq_len, :]
        curr_ds = raw_ds[:seq_len, :]
        curr_dsa = curr_ds @ raw_a
        curr_dsa = torch.where(dropout_mask_curr, curr_dsa / (1 - dropout_p), 0.0)
        curr_dyw += curr_dsa

        # Store the result
        out[curr_start : curr_start + seq_len, :] = curr_dyw

        # Update the current start
        curr_start += padded_seq_len

    return out


def torch_multi_lora_dyw_dsa_ref_2(
    dy: torch.Tensor,
    w: torch.Tensor,
    raw_ds_list: list[torch.Tensor],
    raw_a_list: list[torch.Tensor],
    block_to_lookup_table: torch.Tensor,
    block_to_dropout_p: torch.Tensor,
    block_to_alpha: torch.Tensor,
    block_size_m: int,
    dropout_p_list: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA dyw + dsa."""
    M, N = dy.shape
    N, K = w.shape
    out = torch.zeros((M, K), device=dy.device, dtype=dy.dtype)

    # Construct the ds and a ptrs list
    _, _, ds_list, a_list = construct_ds_and_a_ptrs_list(
        raw_ds_list=raw_ds_list,
        raw_a_list=raw_a_list,
        block_size_m=block_size_m,
    )

    num_blocks = block_to_lookup_table.shape[0]
    for i in range(num_blocks):
        valid_size = block_to_lookup_table[i, 1]
        s_offset_m = block_to_lookup_table[i, 2]

        # For dy
        start_m = i * block_size_m
        curr_dy = dy[start_m : start_m + valid_size, :]
        curr_dyw = curr_dy @ w

        # For ds and a
        curr_ds = ds_list[i]
        curr_a = a_list[i]
        start_m_s = (i - s_offset_m) * block_size_m
        curr_dsa = curr_ds[start_m_s : start_m_s + valid_size, :] @ curr_a
        curr_dropout_mask = dropout_mask[start_m : start_m + valid_size, :]
        curr_dsa = torch.where(
            curr_dropout_mask, curr_dsa / (1 - block_to_dropout_p[i]), 0.0
        )
        curr_dyw += curr_dsa

        # Store the result
        out[start_m : start_m + valid_size, :] = curr_dyw

    return out


def verify_kernel_correctness(
    seq_len_list_choices: list[list[int]],
    lora_idx_list_choices: list[list[int]],
    lora_rank_list_choices: list[list[int]],
    dropout_p_list_choices: list[list[float]],
    alpha_list_choices: list[list[float]],
    n: int,
    k: int,
    block_size_m: int,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions.

    This is particularly important when M is not divisible by 128.
    """
    for seq_len_list, lora_idx_list, lora_rank_list, dropout_p_list, alpha_list in zip(
        seq_len_list_choices,
        lora_idx_list_choices,
        lora_rank_list_choices,
        dropout_p_list_choices,
        alpha_list_choices,
        strict=True,
    ):
        logger.info(
            f"Verifying kernel correctness for seq_len_list={seq_len_list} and "
            f"lora_idx_list={lora_idx_list}..."
        )

        # Prepare tensors
        inputs = prepare_func(
            seq_len_list=seq_len_list,
            lora_idx_list=lora_idx_list,
            lora_rank_list=lora_rank_list,
            dropout_p_list=dropout_p_list,
            alpha_list=alpha_list,
            n=n,
            k=k,
            block_size_m=block_size_m,
            dtype=dtype,
        )

        # # Run triton kernel
        triton_output = fused_multi_lora_dyw_dsa(init_zeros=True, **inputs)

        # Compute reference result
        ref_output = torch_multi_lora_dyw_dsa_ref(**inputs)
        ref_output_2 = torch_multi_lora_dyw_dsa_ref_2(**inputs)
        assert_verbose_allclose_two_rounds(ref_output, ref_output_2, atol=5e-3)

        # # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(triton_output, ref_output, atol=5e-3)
            logger.success(
                f"Verification passed for seq_len_list={seq_len_list} and "
                f"lora_idx_list={lora_idx_list}"
            )
        except AssertionError as e:
            logger.error(
                f"Verification failed for seq_len_list={seq_len_list} and "
                f"lora_idx_list={lora_idx_list}"
            )
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(200, 200)

    seq_len_list_choices = [[2048, 2048]]
    lora_idx_list_choices = [[0, 1]]
    lora_rank_list_choices = [[16, 16] for _ in range(len(seq_len_list_choices))]
    dropout_p_list_choices = [[0.1, 0.1] for _ in range(len(seq_len_list_choices))]
    alpha_list_choices = [[2.0, 2.0] for _ in range(len(seq_len_list_choices))]

    # Test with various M values, including those not divisible by 128
    n = 4096
    k = 4096
    block_size_m = get_lora_kernel_config("fused_multi_lora_dyw_dsa").block_size_m
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(
        seq_len_list_choices,
        lora_idx_list_choices,
        lora_rank_list_choices,
        dropout_p_list_choices,
        alpha_list_choices,
        n,
        k,
        block_size_m,
        dtype,
    )

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for seq_len_list, lora_idx_list, lora_rank_list, dropout_p_list, alpha_list in zip(
        seq_len_list_choices,
        lora_idx_list_choices,
        lora_rank_list_choices,
        dropout_p_list_choices,
        alpha_list_choices,
        strict=True,
    ):
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking fused_multi_lora_dyw_dsa with seq_len_list={seq_len_list} "
            f"and lora_idx_list={lora_idx_list}"
        )
        curr_prepare_func = partial(
            prepare_func,
            seq_len_list=seq_len_list,
            lora_idx_list=lora_idx_list,
            lora_rank_list=lora_rank_list,
            dropout_p_list=dropout_p_list,
            alpha_list=alpha_list,
            n=n,
            k=k,
            block_size_m=block_size_m,
            dtype=dtype,
        )

        benchmark(
            fused_multi_lora_dyw_dsa,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_multi_lora_dyw_dsa_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}_lora_rank_list_{lora_rank_list}_dropout_p_list_{dropout_p_list}_alpha_list_{alpha_list}",
        )

        benchmark(
            torch_dyw_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dyw_ref_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}_lora_rank_list_{lora_rank_list}_dropout_p_list_{dropout_p_list}_alpha_list_{alpha_list}",
        )

        benchmark(
            torch_multi_lora_dyw_dsa_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_multi_lora_dyw_dsa_ref_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}_lora_rank_list_{lora_rank_list}_dropout_p_list_{dropout_p_list}_alpha_list_{alpha_list}",
        )
