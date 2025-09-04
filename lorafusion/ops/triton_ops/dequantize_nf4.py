# ruff: noqa: PLR2004, ANN001, ANN201, N803, N806, E731
"""Dequantize nf4 triton kernel."""

import math

import torch
import triton
import triton.language as tl
from bitsandbytes.functional import QuantState

from lorafusion.ops.triton_ops.tma_utils import (
    tl_experimental_descriptor_load,
    tl_experimental_descriptor_store,
)

NF4_LUT: tl.constexpr = [
    -1.0,  # 0000
    -0.6961928009986877,  # 0001
    -0.5250730514526367,  # 0010
    -0.39491748809814453,  # 0011
    -0.28444138169288635,  # 0100
    -0.18477343022823334,  # 0101
    -0.09105003625154495,  # 0110
    0.0,  # 0111
    0.07958029955625534,  # 1000
    0.16093020141124725,  # 1001
    0.24611230194568634,  # 1010
    0.33791524171829224,  # 1011
    0.44070982933044434,  # 1100
    0.5626170039176941,  # 1101
    0.7229568362236023,  # 1110
    1.0,  # 1111
]


@triton.jit
def nf4_lut_direct(val):
    """Direct lookup table for NF4 dequantization."""
    return tl.where(
        (val & 0b1000) == 8,
        tl.where(
            (val & 0b0100) == 4,
            tl.where(
                (val & 0b0010) == 2,
                tl.where((val & 0b0001) == 1, NF4_LUT[15], NF4_LUT[14]),
                tl.where((val & 0b0001) == 1, NF4_LUT[13], NF4_LUT[12]),
            ),
            tl.where(
                (val & 0b0010) == 2,
                tl.where((val & 0b0001) == 1, NF4_LUT[11], NF4_LUT[10]),
                tl.where((val & 0b0001) == 1, NF4_LUT[9], NF4_LUT[8]),
            ),
        ),
        tl.where(
            (val & 0b0100) == 4,
            tl.where(
                (val & 0b0010) == 2,
                tl.where((val & 0b0001) == 1, NF4_LUT[7], NF4_LUT[6]),
                tl.where((val & 0b0001) == 1, NF4_LUT[5], NF4_LUT[4]),
            ),
            tl.where(
                (val & 0b0010) == 2,
                tl.where((val & 0b0001) == 1, NF4_LUT[3], NF4_LUT[2]),
                tl.where((val & 0b0001) == 1, NF4_LUT[1], NF4_LUT[0]),
            ),
        ),
    )


@triton.jit
def nf4_lut_load(val, lut_ptr):
    """Load the lookup table for NF4 dequantization."""
    return tl.load(lut_ptr + val)


@triton.jit
def dequantize_16bit_nf4_tiled_kernel(
    weight_ptr,
    absmax_ptr,
    nested_code_ptr,
    nested_absmax_ptr,
    offset_ptr,
    block_idx_m,
    block_idx_n,
    M,
    N,
    IS_NESTED: tl.constexpr,
    ABSMAX_SHIFT: tl.constexpr,
    NESTED_ABSMAX_SHIFT: tl.constexpr,
    USE_TMA_DESCRIPTOR: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Tiled kernel for NF4 dequantization.

    This kernel processes a tile of the weight matrix at coordinates
    (block_idx_m, block_idx_n). It's designed to be used with matrix multiplications
    where we can leverage 2D blocks efficiently.

    Args:
        weight_ptr: Pointer to quantized weights [M*N//2] in uint8 format
        absmax_ptr: Pointer to absmax values [M*N // QBS] in uint8 format
        nested_code_ptr: Pointer to nested code [256] in float32 format
        nested_absmax_ptr:
            Pointer to nested absmax [M*N // QBS // QNBS] in float32 format
        offset_ptr: Pointer to offset [1] in float32 format
        block_idx_m: M-dimension block index
        block_idx_n: N-dimension block index
        M: Logical matrix height
        N: Logical matrix width
        IS_NESTED: Whether nested quantization is used
        ABSMAX_SHIFT: Log2 of quantization block size
        NESTED_ABSMAX_SHIFT: Log2 of nested quantization block size
        USE_TMA_DESCRIPTOR: Whether to use TMA descriptor
        BLOCK_SIZE_M: Block size in M dimension
        BLOCK_SIZE_N: Block size in N dimension

    Returns:
        Tuple of (dequantized_higher_bits, dequantized_lower_bits) where each is
        scaled by absmax
    """
    # Calculate the offsets (input offsets)
    BLOCK_SIZE_N_4BITS: tl.constexpr = BLOCK_SIZE_N >> 1
    offsets_m = block_idx_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_idx_n * BLOCK_SIZE_N_4BITS + tl.arange(0, BLOCK_SIZE_N_4BITS)
    offsets = offsets_m[:, None] * (N >> 1) + offsets_n[None, :]

    # Deal with the absmax
    if IS_NESTED:
        # Load the quantized absmax (uint8, another indices matrix for the code book)
        quant_absmax = tl.load(absmax_ptr + (offsets >> ABSMAX_SHIFT))
        # Load the dequantized absmax (float32)
        dequant_absmax = tl.load(nested_code_ptr + quant_absmax)
        # Load the nested absmax to scale the dequantized absmax
        nested_absmax = tl.load(nested_absmax_ptr + (offsets >> NESTED_ABSMAX_SHIFT))
        # Scale the dequantized absmax
        offset = tl.load(offset_ptr)
        absmax = tl.fma(dequant_absmax, nested_absmax, offset)
    else:
        absmax = tl.load(absmax_ptr + (offsets >> ABSMAX_SHIFT))

    # Load the weight and decode
    if USE_TMA_DESCRIPTOR:
        weights = tl_experimental_descriptor_load(
            weight_ptr,
            [block_idx_m * BLOCK_SIZE_M, block_idx_n * BLOCK_SIZE_N_4BITS],
            [BLOCK_SIZE_M, BLOCK_SIZE_N_4BITS],
            tl.uint8,
        )
    else:
        weights = tl.load(weight_ptr + offsets)

    # Split the weights into two 4-bit values
    higher_4bits = weights >> 4
    lower_4bits = weights & 0x0F
    # Dequantize the two 4-bit values
    dequant_higher = nf4_lut_direct(higher_4bits) * absmax
    dequant_lower = nf4_lut_direct(lower_4bits) * absmax

    # Construct the output
    return dequant_higher, dequant_lower


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
            },
            num_warps=w,
        )
        for BM in [1]
        for BN in [1024]
        for w in [4, 8]
    ],
    key=["M", "N", "IS_NESTED", "ABSMAX_SHIFT", "OUTPUT_DTYPE"],
)
@triton.jit
def dequantize_16bit_nf4_kernel(
    weight_ptr,
    output_ptr,
    absmax_ptr,
    nested_code_ptr,
    nested_absmax_ptr,
    offset_ptr,
    M,
    N,
    IS_NESTED: tl.constexpr,
    ABSMAX_SHIFT: tl.constexpr,
    NESTED_ABSMAX_SHIFT: tl.constexpr,
    USE_TMA_DESCRIPTOR: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Dequantize nf4 triton kernel with 2D grid support for future GEMM integration.

    This kernel handles dequantization of NF4 weights by calling the tiled kernel and
    storing the results in the output buffer using an interleaved pattern.

    Args:
        weight_ptr: Pointer to quantized weights [M*N//2] in uint8 format
        output_ptr: Pointer to output buffer [M*N] in bfloat16/float16 format
        absmax_ptr: Pointer to absmax values [M*N // QBS] in uint8 format
        nested_code_ptr: Pointer to nested code [256] in float32 format
        nested_absmax_ptr:
            Pointer to nested absmax [M*N // QBS // QNBS] in float32 format
        offset_ptr: Pointer to offset [1] in float32 format
        M: Logical matrix height
        N: Logical matrix width
        IS_NESTED: Whether nested quantization is used
        ABSMAX_SHIFT: Log2 of quantization block size - 1
        NESTED_ABSMAX_SHIFT: Log2 of nested quantization block size - 1
        USE_TMA_DESCRIPTOR: Whether to use TMA descriptor
        OUTPUT_DTYPE: Output data type (tl.float16 or tl.bfloat16)
        BLOCK_SIZE_M: Block size in M dimension
        BLOCK_SIZE_N: Block size in N dimension

    If nested quantization is used:
    1. absmax is not the real absmax, but quantized indices
    2. these indices are dequantized using nested_code_ptr
    3. the resulting values are scaled by nested_absmax
    4. offset is added to produce the final absmax values
    """
    # pim_m and pim_n are the output block indices
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Call the tiled kernel to get the dequantized values
    dequant_higher, dequant_lower = dequantize_16bit_nf4_tiled_kernel(
        weight_ptr,
        absmax_ptr,
        nested_code_ptr,
        nested_absmax_ptr,
        offset_ptr,
        pid_m,
        pid_n,
        M,
        N,
        IS_NESTED,
        ABSMAX_SHIFT,
        NESTED_ABSMAX_SHIFT,
        USE_TMA_DESCRIPTOR,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    dequant = tl.interleave(dequant_higher, dequant_lower)

    if USE_TMA_DESCRIPTOR:
        tl_experimental_descriptor_store(
            output_ptr,
            dequant.to(OUTPUT_DTYPE),
            [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        )
    else:
        # Calculate block offsets and ranges
        offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offsets = offsets_m[:, None] * N + offsets_n[None, :]
        # Interleave and store dequantized values with proper masking
        tl.store(
            output_ptr + offsets,
            dequant.to(OUTPUT_DTYPE),
        )


def dequantize_nf4_triton(
    A: torch.Tensor,
    quant_state: QuantState,
) -> torch.Tensor:
    """Dequantize NF4 weights using Triton with support for future GEMM integration.

    Args:
        A: Quantized weight tensor [M*N//2] in uint8 format
        quant_state: Quantization state containing absmax, shape and other metadata

    Returns:
        Dequantized weight tensor [M, N] in output_dtype format
    """
    # We use weight to represent the quantized tensor
    weight = A

    # Extract the shape and other metadata from quant_state
    shape = quant_state.shape
    total_numel = shape.numel()

    # Calculate logical matrix dimensions (M rows, N columns)
    if len(shape) >= 2:
        # If shape has at least 2 dimensions, use the last two dimensions as M and N
        M, N = shape[-2], shape[-1]
    else:
        # If shape is 1D, treat it as a row vector
        M, N = 1, shape[-1]

    # Create output tensor
    output_dtype = quant_state.dtype
    output = torch.empty(total_numel, dtype=output_dtype, device=weight.device)

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

    if output_dtype == torch.bfloat16:
        tl_output_dtype = tl.bfloat16
    elif output_dtype == torch.float16:
        tl_output_dtype = tl.float16
    else:
        msg = f"Unsupported output dtype: {output_dtype}"
        raise ValueError(msg)

    # Call the kernel with 1D grid that will be converted to 2D inside kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    dequantize_16bit_nf4_kernel[grid](
        weight,
        output,
        quant_state.absmax,
        quant_state.state2.code if is_nested else None,
        quant_state.state2.absmax if is_nested else None,
        quant_state.offset,
        M,  # Logical matrix height
        N,  # Logical matrix width
        IS_NESTED=is_nested,
        ABSMAX_SHIFT=ABSMAX_SHIFT,
        NESTED_ABSMAX_SHIFT=NESTED_ABSMAX_SHIFT,
        USE_TMA_DESCRIPTOR=False,
        OUTPUT_DTYPE=tl_output_dtype,
    )

    # Reshape output to the original shape
    return output.view(shape)
