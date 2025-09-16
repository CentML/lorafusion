# ruff: noqa: N806
"""Test triton Fused Multi-LoRA xw + sb."""

from functools import partial
from typing import Any

import torch
from loguru import logger

from lorafusion.ops.multi_lora import prepare_multi_lora_batch_info
from lorafusion.ops.triton_ops.config import get_lora_kernel_config
from lorafusion.ops.triton_ops.fused_multi_lora_dys_dyb import (
    construct_s_and_b_ptrs_list,
    fused_multi_lora_dys_dyb,
)
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
        ) / 10
        for lora_rank in multi_lora_batch_info.lora_rank_list
    ]
    lora_b_list = [
        torch.rand(
            n,
            lora_rank,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        ) / 10
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

    return {
        "x": padded_x,
        "w": linear_w,
        "dy": dy,
        "s_ptrs_list": s_ptrs_list,
        "b_ptrs_list": b_ptrs_list,
        "raw_s_list": s_list,
        "raw_b_list": lora_b_list,
        "block_to_lookup_table": block_to_lookup_table,
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


def torch_dys_ref(
    dy: torch.Tensor,
    s: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of dy @ s."""
    return dy.T @ s


def torch_dyb_ref(
    dy: torch.Tensor,
    b: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of dy @ b."""
    return dy @ b


def torch_lora_dys_dyb_ref(
    dy: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference implementation of dy @ s + dy @ b * alpha."""
    return (dy.T @ s) * alpha, (dy @ b) * alpha


def torch_multi_lora_dys_dyb_ref(
    dy: torch.Tensor,
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    block_to_lookup_table: torch.Tensor,
    block_to_alpha: torch.Tensor,
    block_size_m: int,
    seq_len_list: list[int],
    padded_seq_len_list: list[int],
    alpha_list: list[float],
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Torch reference implementation of multi-lora dy @ s + dy @ b * alpha."""
    # Initialize the db_list and ds_list
    db_list = []
    ds_list = []

    curr_start = 0
    for i in range(len(seq_len_list)):
        seq_len = seq_len_list[i]
        padded_seq_len = padded_seq_len_list[i]
        alpha = alpha_list[i]
        s = raw_s_list[i]
        b = raw_b_list[i]

        dy_slice = dy[curr_start:curr_start + seq_len]
        db_slice = (dy_slice.T @ s) * alpha
        ds_slice = (dy_slice @ b) * alpha
        db_list.append(db_slice)
        ds_list.append(ds_slice)
        curr_start += padded_seq_len

    return db_list, ds_list

def verify_kernel_correctness(
    seq_len_list_choices: list[list[int]],
    lora_idx_list_choices: list[list[int]],
    lora_rank_list_choices: list[list[int]],
    dropout_p_list_choices: list[list[float]],
    alpha_list_choices: list[list[float]],
    n: int,
    block_size_m: int,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions."""
    for seq_len_list, lora_idx_list, lora_rank_list, dropout_p_list, alpha_list in zip(
        seq_len_list_choices,
        lora_idx_list_choices,
        lora_rank_list_choices,
        dropout_p_list_choices,
        alpha_list_choices,
        strict=True,
    ):
        logger.info(f"Verifying kernel correctness for seq_len_list={seq_len_list}...")

        inputs = prepare_func(
            seq_len_list=seq_len_list,
            lora_idx_list=lora_idx_list,
            lora_rank_list=lora_rank_list,
            dropout_p_list=dropout_p_list,
            alpha_list=alpha_list,
            n=n,
            k=n,
            block_size_m=block_size_m,
            dtype=dtype,
        )

        # Run triton kernel
        db_list_triton, ds_list_triton = fused_multi_lora_dys_dyb(
            **inputs,
        )

        # Compute reference result
        db_list_ref, ds_list_ref = torch_multi_lora_dys_dyb_ref(
            **inputs,
        )

        # Compare results
        try:
            for i in range(len(db_list_triton)):
                logger.info(f"Verifying adapter {i}...")
                # Use higher tolerance for bfloat16
                atol = 5e-3
                logger.info(f"Verifying db for adapter {i}...")
                assert_verbose_allclose_two_rounds(
                    db_list_triton[i], db_list_ref[i], atol=atol
                )
                logger.info(f"Verifying ds for adapter {i}...")
                assert_verbose_allclose_two_rounds(
                    ds_list_triton[i], ds_list_ref[i], atol=atol
                )
                logger.success(f"Verification passed for adapter {i}")
            logger.success(f"Verification passed for seq_len_list={seq_len_list}")
        except AssertionError as e:
            logger.error(f"Verification failed for seq_len_list={seq_len_list}")
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # Test with various M values, including those not divisible by block sizes
    seq_len_list_choices = [[2048, 2048], [4096, 4096]]
    lora_idx_list_choices = [[0, 1] for _ in range(len(seq_len_list_choices))]
    lora_rank_list_choices = [[16, 16] for _ in range(len(seq_len_list_choices))]
    dropout_p_list_choices = [[0.1, 0.1] for _ in range(len(seq_len_list_choices))]
    alpha_list_choices = [[2.0, 2.0] for _ in range(len(seq_len_list_choices))]
    n = 4096  # Keep N as multiple of 256
    block_size_m = get_lora_kernel_config("fused_multi_lora_block_size_m")
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(
        seq_len_list_choices=seq_len_list_choices,
        lora_idx_list_choices=lora_idx_list_choices,
        lora_rank_list_choices=lora_rank_list_choices,
        dropout_p_list_choices=dropout_p_list_choices,
        alpha_list_choices=alpha_list_choices,
        n=n,
        block_size_m=block_size_m,
        dtype=dtype,
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
        total_m = sum(seq_len_list)
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking fused_multi_lora_dys_dyb with seq_len_list={seq_len_list}, total_m={total_m}"
        )

        curr_prepare_func = partial(
            prepare_func,
            seq_len_list=seq_len_list,
            lora_idx_list=lora_idx_list,
            lora_rank_list=lora_rank_list,
            dropout_p_list=dropout_p_list,
            alpha_list=alpha_list,
            n=n,
            k=n,
            block_size_m=block_size_m,
            dtype=dtype,
        )

        benchmark(
            fused_multi_lora_dys_dyb,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_multi_lora_dys_dyb_seq_len_list_{seq_len_list}",
        )
