# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused LoRA xw + sb."""

from functools import partial
import math
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger
from bitsandbytes import functional as bnb_functional
from bitsandbytes.functional import QuantState

from lorafusion.ops.triton_ops.dequantize_nf4 import dequantize_16bit_nf4_tiled_kernel
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number


def prepare_func(
    m: int,
    n: int,
    k: int,
    r: int,
    alpha: float,
    dtype: torch.dtype = torch.bfloat16,
    *,
    blocksize: int = 256,
    quant_type: str = "nf4",
    compress_statistics: bool = False,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA xw + sb kernel."""
    x = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    w = torch.randn(n, k, device="cuda", dtype=dtype) / 10
    s = torch.randn(m, r, device="cuda", dtype=dtype) / 10
    b = torch.randn(n, r, device="cuda", dtype=dtype) / 10

    qw, quant_state = bnb_functional.quantize_4bit(
        w,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )

    return {
        "x": x,
        "qw": qw,
        "quant_state": quant_state,
        "s": s,
        "b": b,
        "alpha": alpha,
    }


def torch_quantized_nf4_xw_ref(
    x: torch.Tensor,
    qw: torch.Tensor,
    quant_state: QuantState,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    w = bnb_functional.dequantize_4bit(qw, quant_state)
    return x @ w.T


def torch_quantized_nf4_lora_xw_sb_ref(
    x: torch.Tensor,
    qw: torch.Tensor,
    quant_state: QuantState,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    w = bnb_functional.dequantize_4bit(qw, quant_state)
    return x @ w.T + s @ b.T * alpha


def fused_quantized_nf4_lora_xw_sb_kernel_get_configs() -> list[triton.Config]:
    """Get the configurations for the fused LoRA xw + sb kernel."""
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128]
        for BK in [64]
        for s in ([4])
        for w in [8]
    ]


@triton.autotune(
    configs=fused_quantized_nf4_lora_xw_sb_kernel_get_configs(),
    key=["M", "N", "K", "OUTPUT_DTYPE"],
)
@triton.jit
def fused_quantized_nf4_lora_xw_sb_kernel(
    x_ptr,
    qw_ptr, # uint8 [K // 2, N]
    absmax_ptr,
    nested_code_ptr,
    nested_absmax_ptr,
    offset_ptr,
    s_ptr,
    b_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    alpha: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_qwk,
    stride_qwn,
    stride_sm,
    stride_sr,
    stride_br: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om,
    stride_on,
    IS_NESTED: tl.constexpr,
    ABSMAX_SHIFT: tl.constexpr,
    NESTED_ABSMAX_SHIFT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused LoRA xw + sb kernel.

    Compute Out = XW + S @ B * alpha

    Dimensions:
      X: [M, K]
      W: [K, N]
      S: [M, R]
      B: [R, N]
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
    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    qw_ptrs = qw_ptr + ((tl.arange(0, BLOCK_SIZE_K // 2)[:, None]) * stride_qwk + offs_wn[None, :] * stride_qwn)

    # ------------------------------------------------------------
    # 3. Compute the LoRA part
    # ------------------------------------------------------------
    accum_main = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

        # Direct load
        qw = tl.load(qw_ptrs).to(tl.bfloat16)
        qw = tl.join(qw, qw).reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)

        accum_main = tl.dot(x, qw, accum_main)

        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        qw_ptrs += (BLOCK_SIZE_K // 2) * stride_qwk

    accum_main = accum_main.to(OUTPUT_DTYPE)

    # ------------------------------------------------------------
    # 5. Store the result
    # ------------------------------------------------------------
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, accum_main, mask=out_mask)


def fused_lora_xw_sb(
    x: torch.Tensor,
    qw: torch.Tensor,
    quant_state: QuantState,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Triton Fused LoRA xw + sb."""
    # Check constraints.
    qw_shape = quant_state.shape
    qw_dtype = quant_state.dtype
    if x.shape[1] != qw_shape[1]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {qw_shape[1]}. "
            f"x: {x.shape}, qw: {qw_shape}"
        )
        raise ValueError(msg)
    if s.shape[1] != b.shape[1]:
        msg = (
            f"Incompatible dimensions: {s.shape[1]} != {b.shape[1]}. "
            f"s: {s.shape}, b: {b.shape}"
        )
        raise ValueError(msg)
    if x.dtype != qw_dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {qw_dtype}"
        raise ValueError(msg)
    if x.dtype != s.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {s.dtype}"
        raise ValueError(msg)
    if x.dtype != b.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {b.dtype}"
        raise ValueError(msg)
    if not x.is_contiguous():
        msg = "Matrix A must be contiguous"
        raise ValueError(msg)

    # Transpose w and b to match the kernel's dimensions.
    # qw is not transposable.
    b = b.T

    M, K = x.shape
    N, K = qw_shape
    R = s.shape[1]

    qw = qw.reshape(N, K // 2).T

    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Determine if nested quantization is used
    is_nested = hasattr(quant_state, "state2") and quant_state.state2 is not None

    # Calculate log values for blocksize (for efficiency in bit shifting)
    # -1 is because every uint8 contains 2 values
    quant_blocksize = quant_state.blocksize  # Default blocksize in bitsandbytes
    ABSMAX_SHIFT = int(math.log2(quant_blocksize)) - 1
    if is_nested:
        quant_nested_blocksize = quant_state.state2.blocksize
        # Further shift the absmax by the nested blocksize
        NESTED_ABSMAX_SHIFT = ABSMAX_SHIFT + int(math.log2(quant_nested_blocksize))
    else:
        NESTED_ABSMAX_SHIFT = None

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    compiled_kernel = fused_quantized_nf4_lora_xw_sb_kernel[grid](
        x,
        qw,
        quant_state.absmax,
        quant_state.state2.code if is_nested else None,
        quant_state.state2.absmax if is_nested else None,
        quant_state.offset,
        s,
        b,
        out,
        M,
        N,
        K,
        R,
        alpha,
        x.stride(0),
        x.stride(1),
        qw.stride(0),
        qw.stride(1),
        s.stride(0),
        s.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        IS_NESTED=is_nested,
        ABSMAX_SHIFT=ABSMAX_SHIFT,
        NESTED_ABSMAX_SHIFT=NESTED_ABSMAX_SHIFT,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(out.dtype),
    )
    if compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return out


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    m_choices = [8192]
    n = 4096
    k = 4096
    r = 16
    alpha = 16
    dtype = torch.bfloat16

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(f"Benchmarking fused_lora_xw_sb with m={m}, n={n}, k={k}, r={r}")
        curr_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
        )

        benchmark(
            fused_lora_xw_sb,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_xw_sb_m_{m}",
        )

        benchmark(
            torch_quantized_nf4_xw_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_quantized_nf4_xw_ref_m_{m}",
        )

        benchmark(
            torch_quantized_nf4_lora_xw_sb_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_quantized_nf4_lora_xw_sb_ref_m_{m}",
        )
