# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused Multi-LoRA xw + sb."""

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.config import LoRATritonConfig, get_lora_kernel_config
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype


def get_multi_lora_global_info():
    pass


def prepare_inputs_for_multi_lora():
    pass


def set_multi_lora_global_info():
    pass


MAX_NUM_BLOCK_M_SIZE = 128  # max: MAX_NUM_BLOCK_M_SIZE * BLOCK_SIZE_M tokens
GLOBAL_S_PTR_LIST = None
GLOBAL_B_PTR_LIST = None


def construct_s_and_b_ptrs_list(
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    block_size_m: int,
    *,
    return_tensor_list: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None
]:
    """Construct the s and b ptrs list."""
    GLOBAL_S_PTR_LIST = torch.empty(
        MAX_NUM_BLOCK_M_SIZE, device="cuda", dtype=torch.int64
    )
    GLOBAL_B_PTR_LIST = torch.empty(
        MAX_NUM_BLOCK_M_SIZE, device="cuda", dtype=torch.int64
    )

    curr_block_start = 0
    for raw_s, raw_b in zip(raw_s_list, raw_b_list, strict=True):
        num_blocks = (raw_s.shape[0] + block_size_m - 1) // block_size_m
        GLOBAL_S_PTR_LIST[curr_block_start : curr_block_start + num_blocks] = (
            raw_s.data_ptr()
        )
        GLOBAL_B_PTR_LIST[curr_block_start : curr_block_start + num_blocks] = (
            raw_b.data_ptr()
        )
        curr_block_start += num_blocks

    s_list, b_list = None, None
    if return_tensor_list:
        s_list, b_list = [], []
        for raw_s, raw_b in zip(raw_s_list, raw_b_list, strict=True):
            num_blocks = (raw_s.shape[0] + block_size_m - 1) // block_size_m
            s_list.extend([raw_s] * num_blocks)
            b_list.extend([raw_b] * num_blocks)

    return (
        GLOBAL_S_PTR_LIST,
        GLOBAL_B_PTR_LIST,
        s_list,
        b_list,
    )


@triton.jit
def fused_multi_lora_xw_sb_kernel(  # noqa: PLR0915
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    s_ptrs_list_ptr,
    b_ptrs_list_ptr,
    alpha_list_ptr,
    lookup_table_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    has_bias: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_bias,
    stride_om,
    stride_on,
    total_r: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused Multi-LoRA xw + sb kernel.

    Compute Out = XW + Cat[S[i] @ B[i] * alpha[i] for i in range(num_adapters)] + bias

    In the lookup table, we store the following information for each m-th block:
    - r (int): LoRA rank
    - valid_size (int): Number of active rows in this block
    - s_offset_m (int): Start m for s tensor
    In the s_ptrs and b_ptrs, we store the following information for each m-th block:
    - s_ptr (int64): s tensor pointer
    - b_ptr (int64): b tensor pointer
    In the block_to_alpha, we store the following information for each m-th block:
    - alpha (float): LoRA alpha

    Dimensions:
      X: [M, K]
      W: [K, N]
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

    stride_sm = total_r

    # ------------------------------------------------------------
    #  2. Compute the tile indices for M, N
    # ------------------------------------------------------------
    # Generate offsets for the current block
    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute pointers with safe offsets
    r = tl.load(lookup_table_ptr + pid_m * 3 + 0)
    valid_size = tl.load(lookup_table_ptr + pid_m * 3 + 1)
    s_offset_pid_m = tl.load(lookup_table_ptr + pid_m * 3 + 2)

    # Load alpha
    alpha = tl.load(alpha_list_ptr + pid_m).to(OUTPUT_DTYPE)
    # Load s_ptrs
    s_ptr = tl.load(s_ptrs_list_ptr + pid_m).to(tl.pointer_type(OUTPUT_DTYPE))
    b_ptr = tl.load(b_ptrs_list_ptr + pid_m).to(tl.pointer_type(OUTPUT_DTYPE))

    # Compute pointers
    offs_r = tl.arange(0, MAX_R)
    offs_sm = (pid_m - s_offset_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    s_ptrs = s_ptr + (offs_sm[:, None] * stride_sm + offs_r[None, :] * 1)
    b_ptrs = b_ptr + (offs_r[:, None] * 1 + offs_wn[None, :] * r)
    s_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < valid_size) & (offs_r[None, :] < r)
    b_mask = offs_r[:, None] < r

    # ------------------------------------------------------------
    # 3. Compute the LoRA part
    # ------------------------------------------------------------
    accum_main = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load with proper masks for boundary conditions
    s = tl.load(s_ptrs, mask=s_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Compute LoRA contribution
    accum_main = tl.dot(s, b * alpha, accum_main)

    # ------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute k-dimension masks for this iteration
        k_offset = k * BLOCK_SIZE_K
        # Combine masks for this iteration
        x_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < valid_size) & (
            offs_k[None, :] < K - k_offset
        )
        w_mask = offs_k[:, None] < K - k_offset

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
        bias_ptrs = bias_ptr + offs_wn * stride_bias
        n_mask = offs_wn < N
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
    out_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < valid_size) & (
        offs_on[None, :] < N
    )
    tl.store(out_ptrs, accum_main, mask=out_mask)


def fused_multi_lora_xw_sb(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    block_to_lookup_table: torch.Tensor,
    block_to_alpha: torch.Tensor,
    max_r: int,
    total_r: int,
    s_ptrs_list: torch.Tensor | None = None,
    b_ptrs_list: torch.Tensor | None = None,
    raw_s_list: list[torch.Tensor] | None = None,
    raw_b_list: list[torch.Tensor] | None = None,
    bias: torch.Tensor | None = None,
    init_zeros: bool = False,
    block_size_m: int | None = None,
    config: LoRATritonConfig | None = None,
    **kwargs,
) -> torch.Tensor:
    """Triton Fused LoRA xw + sb."""
    # Check constraints.
    if x.shape[1] != w.shape[1]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {w.shape[1]}. "
            f"x: {x.shape}, w: {w.shape}"
        )
        raise ValueError(msg)

    if config is None:
        lora_kernel_config = get_lora_kernel_config("fused_multi_lora_xw_sb")
    else:
        lora_kernel_config = config

    if block_size_m is not None and lora_kernel_config.block_size_m != block_size_m:
        raise ValueError(
            f"block_size_m for fused_multi_lora_xw_sb is not set and "
            f"lora_kernel_config.block_size_m != input block_size_m. "
            f"lora_kernel_config.block_size_m: {lora_kernel_config.block_size_m}, "
            f"block_size_m: {block_size_m}."
        )

    if block_size_m is None:
        block_size_m = lora_kernel_config.block_size_m

    # Construct the s and b ptrs list
    if s_ptrs_list is None or b_ptrs_list is None:
        if raw_s_list is None or raw_b_list is None:
            msg = (
                "Either raw_s_list and raw_b_list or s_ptrs_list and b_ptrs_list "
                "must be provided."
            )
            raise ValueError(msg)
        s_ptrs_list, b_ptrs_list, _, _ = construct_s_and_b_ptrs_list(
            raw_s_list=raw_s_list,
            raw_b_list=raw_b_list,
            block_size_m=block_size_m,
        )

    # Transpose w and b to match the kernel's dimensions.
    w = w.T

    M, K = x.shape
    K, N = w.shape

    # Check and prepare bias
    has_bias = bias is not None
    if not has_bias:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
        bias_stride = 0
    else:
        bias_stride = bias.stride(0)

    # Allocates output.
    if init_zeros:
        # This is only for the testing purpose.
        out = torch.zeros((M, N), device=x.device, dtype=x.dtype)
    else:
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    triton_config = lora_kernel_config.to_triton_config()

    # Multi-LoRA kernels use @triton.autotune, so we can't override configs manually
    # The config parameter is kept for API compatibility but ignored
    compiled_kernel = fused_multi_lora_xw_sb_kernel[grid](
        x,
        w,
        bias,
        out,
        s_ptrs_list,
        b_ptrs_list,
        block_to_alpha,
        block_to_lookup_table,
        M,
        N,
        K,
        has_bias,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        bias_stride,
        out.stride(0),
        out.stride(1),
        total_r=total_r,
        MAX_R=max_r,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(out.dtype),
        **triton_config.all_kwargs(),
    )
    if compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return out
