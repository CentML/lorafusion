# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused LoRA xw + sb using TMA."""

from functools import partial
import math
from typing import Any

import torch
import triton
import triton.language as tl
from bitsandbytes import functional as bnb_functional
from bitsandbytes.functional import QuantState
from loguru import logger

from lorafusion.ops.triton_ops.dequantize_nf4 import dequantize_16bit_nf4_tiled_kernel
from lorafusion.ops.triton_ops.tma_utils import (
    TmaAutoTuneHelper,
    _compute_pid,
    tl_experimental_descriptor_load,
    tl_experimental_descriptor_store,
)
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
    compress_statistics: bool = True,
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


def fused_quantized_nf4_lora_xw_sb_tma_kernel_get_configs() -> list[triton.Config]:
    """Get the configurations for the fused LoRA xw + sb kernel using TMA.

    Note:
        - Best config in H100 for 4096x4096:
            - BM = 128, BN = 256, BK = 64, s = 3, w = 8,
            - SUBTILE = False, LUF = None, FLAT = False
        - While for plain matmul, FLAT should be set to True.
            we find that current implementation of Flatten makes the performance
            quite bad for this kernel.
    """
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE,
            },
            num_stages=s,
            num_warps=w,
        )
        # # For tuning:
        # for BM in [128]
        # for BN in [128, 256]
        # for BK in [32, 64, 128]
        # for s in ([3, 4, 5, 6])
        # for w in [4, 8]
        # for SUBTILE in [True, False]
        # for LUF in [None, 1, 2, 8]
        # for FLAT in [True, False]
        # # Pre-tuned config:
        for BM in [64]
        for BN in [256]
        for BK in [64]
        for s in ([3])
        for w in [8]
        for SUBTILE in [False]
        if BM * BK < 256 * 256
    ]


@triton.autotune(
    configs=fused_quantized_nf4_lora_xw_sb_tma_kernel_get_configs(),
    key=["M", "N", "K", "OUTPUT_DTYPE"],
)
@triton.jit
def fused_quantized_nf4_lora_xw_sb_tma_kernel(
    x_desc_ptr,
    qw_desc_ptr,
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
    stride_sm,
    stride_sr,
    stride_bn: tl.constexpr,
    stride_br: tl.constexpr,
    IS_NESTED: tl.constexpr,
    ABSMAX_SHIFT: tl.constexpr,
    NESTED_ABSMAX_SHIFT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused LoRA xw + sb kernel using TMA.

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
    dtype = OUTPUT_DTYPE
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # ------------------------------------------------------------

    for tile_id in range(
        start_pid,
        num_tiles,
        NUM_SMS,
    ):
        # ------------------------------------------------------------
        # 2. Compute the tile indices for M, N
        # ------------------------------------------------------------
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_xm = pid_m * BLOCK_SIZE_M
        offs_wn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # ------------------------------------------------------------
        # 3. Compute the LoRA part
        # ------------------------------------------------------------
        v_offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        v_offs_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        v_offs_r = tl.arange(0, R)
        s_ptrs = s_ptr + (
            v_offs_xm[:, None] * stride_sm + v_offs_r[None, :] * stride_sr
        )
        b_ptrs = b_ptr + (
            v_offs_wn[:, None] * stride_bn + v_offs_r[None, :] * stride_br
        )
        s = tl.load(s_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(s, b.T * alpha, accumulator)

        # ------------------------------------------------------------
        # 4. Main loop
        # ------------------------------------------------------------
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            x = tl_experimental_descriptor_load(
                x_desc_ptr, [offs_xm, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
            )
            w_higher, w_lower = dequantize_16bit_nf4_tiled_kernel(
                qw_desc_ptr,
                absmax_ptr,
                nested_code_ptr,
                nested_absmax_ptr,
                offset_ptr,
                offs_wn,
                offs_k,
                N,
                K,
                IS_NESTED,
                ABSMAX_SHIFT,
                NESTED_ABSMAX_SHIFT,
                True,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )
            w = tl.interleave(w_higher, w_lower)
            w = w.to(dtype)

            accumulator = tl.dot(x, w.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_xm_c = pid_m * BLOCK_SIZE_M
        offs_wn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into
        # multiple pieces. By subtiling we can reduce shared memory consumption by the
        # epilogue and instead use that memory to increase our stage count.
        # In this case we partition the accumulator into
        # 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors.
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            tl_experimental_descriptor_store(out_ptr, c0, [offs_xm_c, offs_wn_c])
            c1 = acc1.to(dtype)
            tl_experimental_descriptor_store(
                out_ptr, c1, [offs_xm_c, offs_wn_c + BLOCK_SIZE_N // 2]
            )
        else:
            accumulator = accumulator.to(dtype)
            tl_experimental_descriptor_store(
                out_ptr, accumulator, [offs_xm_c, offs_wn_c]
            )


def fused_quantized_nf4_lora_xw_sb_tma(
    x: torch.Tensor,
    qw: torch.Tensor,
    quant_state: QuantState,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Triton Fused LoRA xw + sb using TMA."""
    # Check constraints.
    w_shape = quant_state.shape
    w_dtype = quant_state.dtype
    if x.shape[1] != w_shape[1]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {w_shape[1]}. "
            f"x: {x.shape}, w: {w_shape}"
        )
        raise ValueError(msg)
    if s.shape[1] != b.shape[1]:
        msg = (
            f"Incompatible dimensions: {s.shape[1]} != {b.shape[1]}. "
            f"s: {s.shape}, b: {b.shape}"
        )
        raise ValueError(msg)
    if x.dtype != w_dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {w_dtype}"
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

    M, K = x.shape
    N, K = w_shape
    R = s.shape[1]
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

    # Initialize TMA descriptors.
    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("x")
    desc_helper.init_tma_descriptor("qw")
    desc_helper.init_tma_descriptor("s")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("out")

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META: dict[str, Any]) -> int:
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "x",
            x.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            x.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "qw",
            qw.data_ptr(),
            N,
            K // 2,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"] // 2,
            qw.element_size(),
        )

        store_block_n = META["BLOCK_SIZE_N"]

        if META["EPILOGUE_SUBTILE"]:
            store_block_n = store_block_n // 2

        desc_helper.fill_2d_tma_descriptor(
            "out",
            out.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"],
            store_block_n,
            out.element_size(),
        )

        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
    desc_qw = desc_helper.get_tma_descriptor_kernel_param("qw")
    desc_out = desc_helper.get_tma_descriptor_kernel_param("out")

    compiled_kernel = fused_quantized_nf4_lora_xw_sb_tma_kernel[grid](
        desc_x,
        desc_qw,
        quant_state.absmax,
        quant_state.state2.code if is_nested else None,
        quant_state.state2.absmax if is_nested else None,
        quant_state.offset,
        s,
        b,
        desc_out,
        M,
        N,
        K,
        R,
        alpha,
        s.stride(0),
        s.stride(1),
        b.stride(0),
        b.stride(1),
        IS_NESTED=is_nested,
        ABSMAX_SHIFT=ABSMAX_SHIFT,
        NESTED_ABSMAX_SHIFT=NESTED_ABSMAX_SHIFT,
        NUM_SMS=NUM_SMS,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(out.dtype),
    )
    if compiled_kernel is not None and compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return out


if __name__ == "__main__":
    set_warmup_and_number(1000, 1000)

    m_choices = [8192]
    n = 4096
    k = 4096
    r = 16
    alpha = 16
    dtype = torch.bfloat16
    blocksize = 256
    quant_type = "nf4"
    compress_statistics = True

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
            blocksize=blocksize,
            quant_type=quant_type,
            compress_statistics=compress_statistics,
        )

        benchmark(
            fused_quantized_nf4_lora_xw_sb_tma,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_quantized_nf4_lora_xw_sb_tma_m_{m}",
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
