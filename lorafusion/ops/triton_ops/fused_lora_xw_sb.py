# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused LoRA xw + sb."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.config import (
    KERNEL_SPILL_VERBOSE,
    LoRATritonConfig,
    get_lora_kernel_config,
)
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds


def prepare_func(
    m: int,
    n: int,
    k: int,
    r: int,
    alpha: float,
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_bias: bool = False,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA xw + sb kernel."""
    x = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    w = torch.randn(n, k, device="cuda", dtype=dtype) / 10
    s = torch.randn(m, r, device="cuda", dtype=dtype) / 10
    b = torch.randn(n, r, device="cuda", dtype=dtype) / 10
    bias = torch.randn(n, device="cuda", dtype=dtype) / 10 if with_bias else None

    return {
        "x": x,
        "w": w,
        "s": s,
        "b": b,
        "alpha": alpha,
        "bias": bias,
    }


def torch_xw_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    return x @ w.T


def torch_lora_xw_sb_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    result = x @ w.T + s @ b.T * alpha
    if bias is not None:
        result = result + bias
    return result


@triton.jit
def fused_lora_xw_sb_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    alpha: tl.constexpr,
    has_bias: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_sm,
    stride_sr,
    stride_br: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused LoRA xw + sb kernel.

    Compute Out = XW + S @ B * alpha + bias

    Dimensions:
      X: [M, K]
      W: [K, N]
      S: [M, R]
      B: [R, N]
      bias: [N]
      Out: [M, N]
    """
    # ------------------------------------------------------------
    #  1. Program ID / block mapping
    # ------------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_m_in_group % group_size_m)
    pid_n = pid_m_in_group // group_size_m

    # ------------------------------------------------------------
    #  2. Compute the tile indices for M, N
    # ------------------------------------------------------------
    # Generate offsets for the current block
    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_r = tl.arange(0, R)

    # Create masks for boundary checking
    m_mask = offs_xm < M
    n_mask = offs_wn < N

    # Replace out-of-bounds indices with safe values to avoid memory access errors
    offs_xm_safe = tl.where(m_mask, offs_xm, 0)
    offs_wn_safe = tl.where(n_mask, offs_wn, 0)

    # Compute pointers with safe offsets
    x_ptrs = x_ptr + (offs_xm_safe[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn_safe[None, :] * stride_wn)
    s_ptrs = s_ptr + (offs_xm_safe[:, None] * stride_sm + offs_r[None, :] * stride_sr)
    b_ptrs = b_ptr + (offs_r[:, None] * stride_br + offs_wn_safe[None, :] * stride_bn)

    # ------------------------------------------------------------
    # 3. Compute the LoRA part
    # ------------------------------------------------------------
    accum_main = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load with proper masks for boundary conditions
    s = tl.load(s_ptrs, mask=m_mask[:, None], other=0.0)
    b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)

    # Compute LoRA contribution
    accum_main = tl.dot(s, b * alpha, accum_main)

    # ------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute k-dimension masks for this iteration
        k_offset = k * BLOCK_SIZE_K
        # Combine masks for this iteration
        x_mask = m_mask[:, None] & (offs_k[None, :] < K - k_offset)
        w_mask = (offs_k[:, None] < K - k_offset) & n_mask[None, :]

        # Load with proper masks
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        accum_main = tl.dot(x, w, accum_main)

        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # ------------------------------------------------------------
    # 5. Add bias if available
    # ------------------------------------------------------------
    if has_bias:
        # Load bias with proper mask
        bias_ptrs = bias_ptr + offs_wn_safe * stride_bias
        bias_values = tl.load(bias_ptrs, mask=n_mask, other=0.0)

        # Add bias to each row
        accum_main += bias_values[None, :]

    accum_main = accum_main.to(OUTPUT_DTYPE)

    # ------------------------------------------------------------
    # 6. Store the result
    # ------------------------------------------------------------
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, accum_main, mask=out_mask)


def fused_lora_xw_sb(
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    bias: torch.Tensor | None = None,
    *,
    config: LoRATritonConfig | None = None,
) -> torch.Tensor:
    """Triton Fused LoRA xw + sb."""
    # Check constraints.
    if x.shape[1] != w.shape[1]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {w.shape[1]}. "
            f"x: {x.shape}, w: {w.shape}"
        )
        raise ValueError(msg)
    if s.shape[1] != b.shape[1]:
        msg = (
            f"Incompatible dimensions: {s.shape[1]} != {b.shape[1]}. "
            f"s: {s.shape}, b: {b.shape}"
        )
        raise ValueError(msg)
    if x.dtype != w.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {w.dtype}"
        raise ValueError(msg)
    if x.dtype != s.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {s.dtype}"
        raise ValueError(msg)
    if x.dtype != b.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {b.dtype}"
        raise ValueError(msg)
    if bias is not None and x.dtype != bias.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {bias.dtype}"
        raise ValueError(msg)
    if not x.is_contiguous():
        msg = "Matrix A must be contiguous"
        raise ValueError(msg)

    # Transpose w and b to match the kernel's dimensions.
    w = w.T
    b = b.T

    M, K = x.shape
    K, N = w.shape
    R = s.shape[1]

    # Check bias dimensions if provided
    has_bias = bias is not None

    # Use empty tensor with stride 0 if bias is None to avoid conditional logic in
    # kernel
    if not has_bias:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
        bias_stride = 0
    else:
        bias_stride = bias.stride(0)

    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Get configs
    if config is None:
        lora_kernel_config = get_lora_kernel_config("fused_lora_xw_sb")
    else:
        lora_kernel_config = config
    triton_config = lora_kernel_config.to_triton_config()

    compiled_kernel = fused_lora_xw_sb_kernel[grid](
        x,
        w,
        s,
        b,
        bias,
        out,
        M,
        N,
        K,
        R,
        alpha,
        has_bias,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        s.stride(0),
        s.stride(1),
        b.stride(0),
        b.stride(1),
        bias_stride,
        out.stride(0),
        out.stride(1),
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(out.dtype),
        **triton_config.all_kwargs(),
    )
    if KERNEL_SPILL_VERBOSE and compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return out


def verify_kernel_correctness(
    m_values: list[int],
    n: int,
    k: int,
    r: int,
    alpha: float = 16.0,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions."""
    for m in m_values:
        logger.info(f"Verifying kernel correctness for m={m}...")

        # Test without bias
        inputs_no_bias = prepare_func(
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=False,
        )

        # Run triton kernel
        triton_output_no_bias = fused_lora_xw_sb(**inputs_no_bias)

        # Compute reference result
        ref_output_no_bias = torch_lora_xw_sb_ref(**inputs_no_bias)

        # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(
                triton_output_no_bias, ref_output_no_bias, atol=5e-3
            )
            logger.success(f"Verification passed for m={m} without bias")
        except AssertionError as e:
            logger.error(f"Verification failed for m={m} without bias")
            logger.error(e)

        # Test with bias
        inputs_with_bias = prepare_func(
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=True,
        )

        # Run triton kernel
        triton_output_with_bias = fused_lora_xw_sb(**inputs_with_bias)

        # Compute reference result
        ref_output_with_bias = torch_lora_xw_sb_ref(**inputs_with_bias)

        # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(
                triton_output_with_bias, ref_output_with_bias, atol=5e-3
            )
            logger.success(f"Verification passed for m={m} with bias")
        except AssertionError as e:
            logger.error(f"Verification failed for m={m} with bias")
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # Test with various M values, including those not divisible by 128
    m_choices = [4096]
    n = 4096
    k = 4096
    r = 16
    alpha = 16
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(m_choices, n, k, r, alpha, dtype)

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(f"Benchmarking fused_lora_xw_sb with m={m}, n={n}, k={k}, r={r}")

        # Without bias
        curr_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=False,
        )

        benchmark(
            fused_lora_xw_sb,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_xw_sb_m_{m}_no_bias",
        )

        benchmark(
            torch_xw_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_xw_ref_m_{m}",
        )

        benchmark(
            torch_lora_xw_sb_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_lora_xw_sb_ref_m_{m}_no_bias",
        )

        # With bias
        curr_prepare_func_with_bias = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=True,
        )

        benchmark(
            fused_lora_xw_sb,
            prepare_func=curr_prepare_func_with_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_xw_sb_m_{m}_with_bias",
        )

        benchmark(
            torch_lora_xw_sb_ref,
            prepare_func=curr_prepare_func_with_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_lora_xw_sb_ref_m_{m}_with_bias",
        )
