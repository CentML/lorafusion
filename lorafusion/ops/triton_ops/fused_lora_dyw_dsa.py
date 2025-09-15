# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused LoRA dy @ w + ds @ a * dropout_scale."""

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
    dropout_p: float,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA dyw + dsa kernel."""
    dy = torch.randn(m, n, device="cuda", dtype=dtype) / 10
    w = torch.randn(n, k, device="cuda", dtype=dtype) / 10
    ds = torch.randn(m, r, device="cuda", dtype=dtype) / 10
    a = torch.randn(r, k, device="cuda", dtype=dtype) / 10
    dropout_mask = torch.randn(m, k, device="cuda", dtype=torch.float32) > dropout_p

    return {
        "dy": dy,
        "w": w,
        "ds": ds,
        "a": a,
        "dropout_p": dropout_p,
        "dropout_mask": dropout_mask,
    }


def torch_dyw_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of dy @ w."""
    return dy @ w


def torch_lora_dyw_dsa_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    ds: torch.Tensor,
    a: torch.Tensor,
    dropout_p: float,
    dropout_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA dyw + dsa."""
    return dy @ w + torch.where(dropout_mask, ds @ a / (1 - dropout_p), 0.0)


@triton.jit
def fused_lora_dyw_dsa_kernel(
    dy_ptr,
    w_ptr,
    ds_ptr,
    a_ptr,
    dropout_mask_ptr,
    dx_ptr,
    dropout_p,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    stride_dym,
    stride_dyk,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_dsm,
    stride_dsr,
    stride_ar: tl.constexpr,
    stride_an: tl.constexpr,
    stride_mask_m,
    stride_mask_n,
    stride_dxm,
    stride_dxn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused LoRA dyw + dsa kernel.

    Compute dx = dy @ w + where(dropout_mask, ds @ a / (1 - dropout_p), 0.0)

    To make the kernel more readable as the low level implementation,
    the shape definition of the input tensors are not as the same as the original
      dy: [M, K]
      w: [K, N]
      ds: [M, R]
      a: [R, N]
      mask: [M, N]
      dx: [M, N]
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
    offs_dym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_r = tl.arange(0, R)

    # ------------------------------------------------------------
    # 3. Compute the LoRA part
    # ------------------------------------------------------------
    ds_ptrs = ds_ptr + (offs_dym[:, None] * stride_dsm + offs_r[None, :] * stride_dsr)
    a_ptrs = a_ptr + (offs_r[:, None] * stride_ar + offs_wn[None, :] * stride_an)
    mask_ds = (offs_dym[:, None] < M) & (offs_r[None, :] < R)

    ds = tl.load(ds_ptrs, mask=mask_ds, other=0.0)
    a = tl.load(a_ptrs)

    accum_main = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accum_main = tl.dot(ds, a, accum_main)

    if ENABLE_DROPOUT:
        dropout_mask_ptrs = dropout_mask_ptr + (
            offs_dym[:, None] * stride_mask_m + offs_wn[None, :] * stride_mask_n
        )
        mask_dropout_mask = (offs_dym[:, None] < M) & (offs_wn[None, :] < N)
        dropout_mask = tl.load(dropout_mask_ptrs, mask=mask_dropout_mask, other=0.0)
        accum_main = tl.where(dropout_mask, accum_main / (1 - dropout_p), 0.0)

    # ------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dy_ptrs = dy_ptr + (offs_dym[:, None] * stride_dym + offs_k[None, :] * stride_dyk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        mask_dy = (offs_k[None, :] < K - k * BLOCK_SIZE_K) & (offs_dym[:, None] < M)
        dy = tl.load(dy_ptrs, mask=mask_dy, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accum_main = tl.dot(dy, w, accum_main)

        # Advance the ptrs to the next K block.
        dy_ptrs += BLOCK_SIZE_K * stride_dyk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    accum_main = accum_main.to(OUTPUT_DTYPE)

    # ------------------------------------------------------------
    # 5. Store the result
    # ------------------------------------------------------------
    offs_dxm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dxn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dx_ptrs = dx_ptr + stride_dxm * offs_dxm[:, None] + stride_dxn * offs_dxn[None, :]
    dx_mask = (offs_dxm[:, None] < M) & (offs_dxn[None, :] < N)
    tl.store(dx_ptrs, accum_main, mask=dx_mask)


def fused_lora_dyw_dsa(
    dy: torch.Tensor,
    w: torch.Tensor,
    ds: torch.Tensor,
    a: torch.Tensor,
    dropout_p: float,
    dropout_mask: torch.Tensor | None = None,
    *,
    config: LoRATritonConfig | None = None,
) -> torch.Tensor:
    """Triton Fused LoRA dyw + dsa.

    Compute dx = dy @ w + where(dropout_mask, ds @ a / (1 - dropout_p), 0.0)

    To make the kernel more readable as the low level implementation,
    the shape definition of the input tensors are not as the same as the original

      dy: [M, N]
      w: input [N, K]
      ds: [M, R]
      a: input [R, K]
      mask: [M, N]
      dx: [M, N]
    """
    # Check constraints.
    m_y, n_y = dy.shape
    n_w, k_w = w.shape
    m_s, r_s = ds.shape
    r_a, k_a = a.shape
    m_mask, k_mask = dropout_mask.shape

    if m_y != m_s or m_y != m_mask:
        msg = (
            f"Incompatible dimensions for m of dy, ds, and mask: "
            f"{m_y} != {m_s} != {m_mask}"
        )
        raise ValueError(msg)
    if n_y != n_w:
        msg = f"Incompatible dimensions for n of dy and w: {n_y} != {n_w}"
        raise ValueError(msg)
    if k_w != k_a or k_w != k_mask:
        msg = (
            f"Incompatible dimensions for k of w, a, and mask: "
            f"{k_w} != {k_a} != {k_mask}"
        )
        raise ValueError(msg)
    if r_s != r_a:
        msg = f"Incompatible dimensions for r of ds and a: {r_s} != {r_a}"
        raise ValueError(msg)

    if dropout_p != 0 and dropout_mask is None:
        msg = "dropout_mask must be provided if dropout_p != 0"
        raise ValueError(msg)

    dx = torch.empty((m_y, k_w), device=dy.device, dtype=dy.dtype)

    # Change the representation of tensors to map the low-level implementation.
    M = m_y
    N = k_w
    K = n_w
    R = r_s
    # Allocates output.
    # 1D launch kernel where each block gets its own program.

    # Get configs
    if config is None:
        lora_kernel_config = get_lora_kernel_config("fused_lora_dyw_dsa")
    else:
        lora_kernel_config = config
    triton_config = lora_kernel_config.to_triton_config()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    compiled_kernel = fused_lora_dyw_dsa_kernel[grid](
        dy,
        w,
        ds,
        a,
        dropout_mask,
        dx,
        dropout_p,
        M,
        N,
        K,
        R,
        dy.stride(0),
        dy.stride(1),
        w.stride(0),
        w.stride(1),
        ds.stride(0),
        ds.stride(1),
        a.stride(0),
        a.stride(1),
        dropout_mask.stride(0) if dropout_mask is not None else 0,
        dropout_mask.stride(1) if dropout_mask is not None else 0,
        dx.stride(0),
        dx.stride(1),
        ENABLE_DROPOUT=dropout_p != 0,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(dx.dtype),
        **triton_config.all_kwargs(),
    )
    if (
        KERNEL_SPILL_VERBOSE
        and compiled_kernel is not None
        and compiled_kernel.n_spills > 0
    ):
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return dx


def verify_kernel_correctness(
    m_values: list[int],
    n: int,
    k: int,
    r: int,
    dropout_p: float = 0.1,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions.

    This is particularly important when M is not divisible by block sizes.

    Args:
        m_values: List of M dimensions to test
        n: Size of N dimension
        k: Size of K dimension
        r: Size of R dimension
        dropout_p: Dropout probability
        dtype: Data type for tensors
    """
    for m in m_values:
        logger.info(f"Verifying kernel correctness for m={m}...")

        # Prepare tensors
        inputs = prepare_func(
            m=m,
            n=n,
            k=k,
            r=r,
            dropout_p=dropout_p,
            dtype=dtype,
        )

        # Run triton kernel
        dx_triton = fused_lora_dyw_dsa(**inputs)

        # Compute reference result
        dx_ref = torch_lora_dyw_dsa_ref(**inputs)

        # Check shapes
        if dx_triton.shape != dx_ref.shape:
            logger.error(
                f"Shape mismatch for m={m}: {dx_triton.shape} vs {dx_ref.shape}"
            )
            continue

        # Check for NaN values
        if torch.isnan(dx_triton).any():
            logger.error(f"NaN values detected in output for m={m}")
            continue

        # Check for correctness
        try:
            # Use higher tolerance for bfloat16
            atol = 5e-4
            logger.info("Verifying dx...")
            assert_verbose_allclose_two_rounds(dx_triton, dx_ref, atol=atol)
            logger.success(f"Verification passed for m={m}")
        except AssertionError as e:
            logger.error(f"Verification failed for m={m}")
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    m_choices = [4096, 4097]
    n = 4096
    k = 4096
    r = 16
    dropout_p = 0.1
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(m_choices, n, k, r, dropout_p, dtype)

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(f"Benchmarking fused_lora_dyw_dsa with m={m}, n={n}, k={k}, r={r}")
        curr_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            dropout_p=dropout_p,
            dtype=dtype,
        )

        benchmark(
            fused_lora_dyw_dsa,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_dyw_dsa_m_{m}",
        )

        benchmark(
            torch_dyw_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dyw_ref_m_{m}",
        )

        benchmark(
            torch_lora_dyw_dsa_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_lora_dyw_dsa_ref_m_{m}",
        )
