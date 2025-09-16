# ruff: noqa: N806
"""Test triton Fused Multi-LoRA xw + sb."""

from functools import partial
from typing import Any

import torch
from loguru import logger

from lorafusion.ops.multi_lora import prepare_multi_lora_batch_info
from lorafusion.ops.triton_ops.config import get_lora_kernel_config
from lorafusion.ops.triton_ops.fused_multi_lora_xw_sb import (
    construct_s_and_b_ptrs_list,
    fused_multi_lora_xw_sb,
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

    return {
        "x": padded_x,
        "w": linear_w,
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


def torch_xw_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    return x @ w.T


def torch_multi_lora_xw_sb_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    seq_len_list: list[int],
    padded_seq_len_list: list[int],
    lora_idx_list: list[int],
    lora_rank_list: list[int],
    dropout_p_list: list[float],
    alpha_list: list[float],
    bias: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    M, K = x.shape
    N, K = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=x.dtype)

    num_loras = len(seq_len_list)

    curr_start = 0
    for i in range(num_loras):
        seq_len = seq_len_list[i]
        padded_seq_len = padded_seq_len_list[i]
        raw_s = raw_s_list[i]
        raw_b = raw_b_list[i]
        alpha = alpha_list[i]

        # For x
        curr_x = x[curr_start : curr_start + seq_len, :]
        curr_xw = curr_x @ w.T

        # For s and w
        curr_s = raw_s[:seq_len, :]
        curr_sb = curr_s @ raw_b.T * alpha
        curr_xw += curr_sb

        # Apply bias if provided
        if bias is not None:
            curr_xw += bias

        # Store the result
        out[curr_start : curr_start + seq_len, :] = curr_xw

        # Update the current start
        curr_start += padded_seq_len

    return out


def torch_multi_lora_xw_sb_ref_2(
    x: torch.Tensor,
    w: torch.Tensor,
    block_to_lookup_table: torch.Tensor,
    block_to_alpha: torch.Tensor,
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    block_size_m: int,
    bias: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    M, K = x.shape
    N, K = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=x.dtype)

    # Construct the s and b ptrs list
    _, _, s_list, b_list = construct_s_and_b_ptrs_list(
        raw_s_list=raw_s_list,
        raw_b_list=raw_b_list,
        block_size_m=block_size_m,
        return_tensor_list=True,
    )

    num_blocks = block_to_lookup_table.shape[0]
    for i in range(num_blocks):
        valid_size = block_to_lookup_table[i, 1]
        s_offset_m = block_to_lookup_table[i, 2]
        alpha = block_to_alpha[i]

        # For x
        start_m = i * block_size_m
        curr_x = x[start_m : start_m + valid_size, :]
        curr_xw = curr_x @ w.T

        # For s and b
        curr_s = s_list[i]
        curr_b = b_list[i]
        start_m_s = (i - s_offset_m) * block_size_m
        curr_sb = curr_s[start_m_s : start_m_s + valid_size, :] @ curr_b.T * alpha
        curr_xw += curr_sb

        # Apply bias if provided
        if bias is not None:
            curr_xw += bias

        # Store the result
        out[start_m : start_m + valid_size, :] = curr_xw

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

        # Test without bias
        logger.info("Testing without bias...")
        inputs_no_bias = prepare_func(
            seq_len_list=seq_len_list,
            lora_idx_list=lora_idx_list,
            lora_rank_list=lora_rank_list,
            dropout_p_list=dropout_p_list,
            alpha_list=alpha_list,
            n=n,
            k=k,
            block_size_m=block_size_m,
            dtype=dtype,
            with_bias=False,
        )

        # Run triton kernel
        triton_output_no_bias = fused_multi_lora_xw_sb(
            init_zeros=True, **inputs_no_bias
        )

        # # Compute reference result
        ref_output_no_bias = torch_multi_lora_xw_sb_ref(**inputs_no_bias)
        ref_output_2_no_bias = torch_multi_lora_xw_sb_ref_2(**inputs_no_bias)
        assert_verbose_allclose_two_rounds(
            ref_output_no_bias, ref_output_2_no_bias, atol=5e-3
        )

        # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(
                triton_output_no_bias, ref_output_no_bias, atol=5e-3
            )
            logger.success(
                f"Verification passed for seq_len_list={seq_len_list} and "
                f"lora_idx_list={lora_idx_list} without bias"
            )
        except AssertionError as e:
            logger.error(
                f"Verification failed for seq_len_list={seq_len_list} and "
                f"lora_idx_list={lora_idx_list} without bias"
            )
            logger.error(e)

        # Test with bias
        logger.info("Testing with bias...")
        inputs_with_bias = prepare_func(
            seq_len_list=seq_len_list,
            lora_idx_list=lora_idx_list,
            lora_rank_list=lora_rank_list,
            dropout_p_list=dropout_p_list,
            alpha_list=alpha_list,
            n=n,
            k=k,
            block_size_m=block_size_m,
            dtype=dtype,
            with_bias=True,
        )

        # Run triton kernel
        triton_output_with_bias = fused_multi_lora_xw_sb(
            init_zeros=True, **inputs_with_bias
        )

        # Compute reference result
        ref_output_with_bias = torch_multi_lora_xw_sb_ref(**inputs_with_bias)
        ref_output_2_with_bias = torch_multi_lora_xw_sb_ref_2(**inputs_with_bias)
        assert_verbose_allclose_two_rounds(
            ref_output_with_bias, ref_output_2_with_bias, atol=5e-3
        )

        # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(
                triton_output_with_bias, ref_output_with_bias, atol=5e-3
            )
            logger.success(
                f"Verification passed for seq_len_list={seq_len_list} and "
                f"lora_idx_list={lora_idx_list} with bias"
            )
        except AssertionError as e:
            logger.error(
                f"Verification failed for seq_len_list={seq_len_list} and "
                f"lora_idx_list={lora_idx_list} with bias"
            )
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(200, 200)

    # TODO(zhanda): Right now, if seq_len_list is not divisible by block_size_m,
    # the correctness check will be failed. This is a fake error, because our test
    # does not discard the padding part. So you may need to modify the test to
    # discard the padding part.

    # Test inputs candidates
    seq_len_list_choices = [[2176, 2048]]
    lora_idx_list_choices = [[0, 1]]
    lora_rank_list_choices = [[16, 16]]
    dropout_p_list_choices = [[0.1, 0.1]]
    alpha_list_choices = [[16.0, 16.0]]

    # Test parameters
    n = 4096
    k = 4096
    block_size_m = get_lora_kernel_config("fused_multi_lora_xw_sb").block_size_m
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
        k=k,
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
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking fused_lora_xw_sb with seq_len_list={seq_len_list} and "
            f"lora_idx_list={lora_idx_list}"
        )

        # Benchmark without bias
        logger.info("Benchmarking without bias...")
        curr_prepare_func_no_bias = partial(
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
            with_bias=False,
        )

        benchmark(
            fused_multi_lora_xw_sb,
            prepare_func=curr_prepare_func_no_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_multi_lora_xw_sb_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}_no_bias",
        )

        benchmark(
            torch_xw_ref,
            prepare_func=curr_prepare_func_no_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_xw_ref_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}",
        )

        benchmark(
            torch_multi_lora_xw_sb_ref,
            prepare_func=curr_prepare_func_no_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_multi_lora_xw_sb_ref_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}_no_bias",
        )

        # Benchmark with bias
        logger.info("Benchmarking with bias...")
        curr_prepare_func_with_bias = partial(
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
            with_bias=True,
        )

        benchmark(
            fused_multi_lora_xw_sb,
            prepare_func=curr_prepare_func_with_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_multi_lora_xw_sb_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}_with_bias",
        )

        benchmark(
            torch_multi_lora_xw_sb_ref,
            prepare_func=curr_prepare_func_with_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_multi_lora_xw_sb_ref_seq_len_list_{seq_len_list}_lora_idx_list_{lora_idx_list}_with_bias",
        )
