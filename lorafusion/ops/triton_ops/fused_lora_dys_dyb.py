# ruff: noqa: ANN001, N803, N806, E731
# ruff: noqa: ANN001, N803, N806, E731: ANN001, N803, N806, E731
"""Triton Fused LoRA dy @ s + dy @ b * dropout_scale."""

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
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds


def prepare_func(
    m: int,
    n: int,
    r: int,
    alpha: float,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA dy @ s + dy @ b * dropout_scale kernel."""
    dy = torch.randn(m, n, device="cuda", dtype=dtype) / 10
    s = torch.randn(m, r, device="cuda", dtype=dtype) / 10
    b = torch.randn(n, r, device="cuda", dtype=dtype) / 10
    return {
        "dy": dy,
        "s": s,
        "b": b,
        "alpha": alpha,
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


@triton.jit
def fused_lora_dys_dyb_kernel(
    dy_ptr,
    b_ptr,
    s_ptr,
    db_ptr,
    ds_ptr,
    alpha,
    M,
    K: tl.constexpr,
    R: tl.constexpr,
    stride_dym,
    stride_dyk,
    stride_bk: tl.constexpr,
    stride_br: tl.constexpr,
    stride_sm,
    stride_sr,
    stride_dbk: tl.constexpr,
    stride_dbr: tl.constexpr,
    stride_dsm,
    stride_dsr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    """Triton Fused LoRA dy @ s + dy @ b * alpha kernel.

    We use split-k strategy to compute the db and ds since the number of
    the dimension of r is usually quite small, e.g. 16, 32, etc.

    Compute:
    - db = dy.T @ s * alpha
    - ds = dy @ b * alpha

    To make the kernel more readable as the low level implementation,
    the shape definition of the input tensors are not as the same as the original
    ones:
    - dy: [M, K]
    - b: [K, R]
    - s: [M, R]
    - db: [K, R]
    - ds: [M, R]
    """
    # ------------------------------------------------------------
    #  1. Program ID / block mapping
    # ------------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_m_in_group % group_size_m)
    pid_k = pid_m_in_group // group_size_m

    # ------------------------------------------------------------
    #  2. Compute the tile indices for M, N
    # ------------------------------------------------------------
    # Generate proper offsets without modulo for M (allowing arbitrary M)
    offs_dym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # K is assumed to be a multiple of BLOCK_SIZE_K
    offs_dyk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_r = tl.arange(0, R)

    dy_ptrs = dy_ptr + (offs_dym[:, None] * stride_dym + offs_dyk[None, :] * stride_dyk)
    s_ptrs = s_ptr + (offs_dym[:, None] * stride_sm + offs_r[None, :] * stride_sr)
    b_ptrs = b_ptr + (offs_dyk[:, None] * stride_bk + offs_r[None, :] * stride_br)
    mask_dy = (offs_dym[:, None] < M) & (offs_dyk[None, :] < K)
    mask_s = (offs_dym[:, None] < M) & (offs_r[None, :] < R)
    mask_b = (offs_dyk[:, None] < K) & (offs_r[None, :] < R)

    # ------------------------------------------------------------
    # 3. Initialize accumulators
    # ------------------------------------------------------------
    accum_db = tl.zeros((BLOCK_SIZE_K, R), dtype=tl.float32)
    accum_ds = tl.zeros((BLOCK_SIZE_M, R), dtype=tl.float32)

    # ------------------------------------------------------------
    # 4. Main computation
    # ------------------------------------------------------------
    # Becuase Triton doesn't support atomic add for bfloat16, we need to
    # use float32 as the intermediate dtype.
    # Load the data
    dy = tl.load(dy_ptrs, mask=mask_dy, other=0.0)
    s = tl.load(s_ptrs, mask=mask_s, other=0.0)
    b = tl.load(b_ptrs, mask=mask_b, other=0.0)

    accum_db = tl.dot(dy.T, s, accum_db)
    accum_db = accum_db * alpha

    accum_ds = tl.dot(dy, b, accum_ds)
    accum_ds = accum_ds * alpha

    # ------------------------------------------------------------
    # 5. Store results
    # ------------------------------------------------------------
    offs_dbk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_dsm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    db_ptrs = db_ptr + (offs_dbk[:, None] * stride_dbk + offs_r[None, :] * stride_dbr)
    ds_ptrs = ds_ptr + (offs_dsm[:, None] * stride_dsm + offs_r[None, :] * stride_dsr)

    # Create masks to handle boundary conditions
    db_mask = (offs_dbk[:, None] < K) & (offs_r[None, :] < R)
    ds_mask = (offs_dsm[:, None] < M) & (offs_r[None, :] < R)

    # Store results
    tl.atomic_add(db_ptrs, accum_db, mask=db_mask)
    tl.atomic_add(ds_ptrs, accum_ds, mask=ds_mask)


def fused_lora_dys_dyb(
    dy: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    alpha: float,
    *,
    config: LoRATritonConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused LoRA dy @ s + dy @ b * alpha.

    Compute db = dy.T @ s * alpha
    Compute ds = dy @ b * alpha

    Args:
        dy: Gradient of the loss with respect to the output, shape [M, K]
        b: LoRA weight tensor, shape [K, R]
        s: LoRA output tensor, shape [M, R]
        alpha: Scaling factor for the gradients
        config: Optional LoRA Triton configuration. If None, uses default config.

    Returns:
        A tuple containing:
            db: Gradient of the loss with respect to b, shape [K, R]
            ds: Gradient of the loss with respect to s, shape [M, R]

    Shape definitions:
        - dy: [M, N]
        - b: [N, R]
        - s: [M, R]
        - db: [N, R]
        - ds: [M, R]
    """
    # Validate input shapes
    m_y, n_y = dy.shape
    n_b, r_b = b.shape
    m_s, r_s = s.shape

    if m_y != m_s:
        msg = (
            f"Incompatible dimensions for m of dy and s: {m_y} != {m_s}. "
            f"dy: {dy.shape}, s: {s.shape}"
        )
        raise ValueError(msg)
    if n_y != n_b:
        msg = (
            f"Incompatible dimensions for n of dy and b: {n_y} != {n_b}. "
            f"dy: {dy.shape}, b: {b.shape}"
        )
        raise ValueError(msg)
    if r_s != r_b:
        msg = (
            f"Incompatible dimensions for r of s and b: {r_s} != {r_b}. "
            f"s: {s.shape}, b: {b.shape}"
        )
        raise ValueError(msg)

    # Check that N and R are multiples of expected block sizes
    if n_y % 256 != 0:
        logger.warning(
            f"N dimension {n_y} is not a multiple of 256, which may cause performance issues"
        )
    if r_b % 16 != 0:
        logger.warning(
            f"R dimension {r_b} is not a multiple of 16, which may cause performance issues"
        )

    # Allocate output tensors
    curr_db = torch.zeros((n_b, r_b), device=dy.device, dtype=torch.float32)
    curr_ds = torch.zeros((m_s, r_s), device=dy.device, dtype=torch.float32)

    # Change the representation of tensors to map the low-level implementation.
    M = m_y
    K = n_y
    R = r_s

    # Get configs
    if config is None:
        lora_kernel_config = get_lora_kernel_config("fused_lora_dys_dyb")
    else:
        lora_kernel_config = config
    triton_config = lora_kernel_config.to_triton_config()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )
    compiled_kernel = fused_lora_dys_dyb_kernel[grid](
        dy,
        b,
        s,
        curr_db,
        curr_ds,
        alpha,
        M,
        K,
        R,
        dy.stride(0),
        dy.stride(1),
        b.stride(0),
        b.stride(1),
        s.stride(0),
        s.stride(1),
        curr_db.stride(0),
        curr_db.stride(1),
        curr_ds.stride(0),
        curr_ds.stride(1),
        **triton_config.all_kwargs(),
    )

    db = curr_db.to(dy.dtype)
    ds = curr_ds.to(dy.dtype)

    if KERNEL_SPILL_VERBOSE and compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")

    return db, ds


def verify_kernel_correctness(
    m_values: list[int],
    n: int,
    r: int,
    alpha: float = 16.0,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions.

    This is particularly important when M is not divisible by block sizes.
    """
    for m in m_values:
        logger.info(f"Verifying kernel correctness for m={m}...")

        # Prepare tensors
        inputs = prepare_func(
            m=m,
            n=n,
            r=r,
            alpha=alpha,
            dtype=dtype,
        )

        # Run triton kernel
        db_triton, ds_triton = fused_lora_dys_dyb(**inputs)

        # Compute reference result
        db_ref, ds_ref = torch_lora_dys_dyb_ref(**inputs)

        # Check for correctness
        try:
            # Use higher tolerance for bfloat16
            atol = 1e-4
            logger.info("Verifying db...")
            assert_verbose_allclose_two_rounds(db_triton, db_ref, atol=atol)
            logger.info("Verifying ds...")
            assert_verbose_allclose_two_rounds(ds_triton, ds_ref, atol=atol)
            logger.success(f"Verification passed for m={m}")
        except AssertionError as e:
            logger.error(f"Verification failed for m={m}")
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # Test with various M values, including those not divisible by block sizes
    m_choices = [2048, 2837, 4095, 4096, 4097, 8192]
    n = 4096  # Keep N as multiple of 256
    r = 16  # Keep R as multiple of 16
    alpha = 16.0
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(m_choices, n, r, alpha, dtype)

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(f"Benchmarking fused_lora_dys_dyb with m={m}, n={n}, r={r}")
        curr_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            r=r,
            alpha=alpha,
            dtype=dtype,
        )

        benchmark(
            fused_lora_dys_dyb,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_dys_dyb_m_{m}",
        )

        benchmark(
            torch_dys_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dys_ref_m_{m}",
        )

        benchmark(
            torch_dyb_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dyb_ref_m_{m}",
        )

        benchmark(
            torch_lora_dys_dyb_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_lora_dys_dyb_ref_m_{m}",
        )
