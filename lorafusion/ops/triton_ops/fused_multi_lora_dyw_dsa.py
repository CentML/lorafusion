# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused LoRA dy @ w + ds @ a * dropout_scale."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.config import LoRATritonConfig, get_lora_kernel_config
from lorafusion.ops.triton_ops.fused_multi_lora_xw_sb import (
    get_multi_lora_global_info,
    prepare_inputs_for_multi_lora,
    set_multi_lora_global_info,
)
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds

MAX_NUM_BLOCK_M_SIZE = 128  # 8192 tokens
GLOBAL_DS_PTR_LIST = None
GLOBAL_A_PTR_LIST = None


def construct_ds_and_a_ptrs_list(
    raw_ds_list: list[torch.Tensor],
    raw_a_list: list[torch.Tensor],
    block_size_m: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Construct the ds and a ptrs list."""
    global GLOBAL_DS_PTR_LIST, GLOBAL_A_PTR_LIST

    if GLOBAL_DS_PTR_LIST is None:
        GLOBAL_DS_PTR_LIST = torch.zeros(
            MAX_NUM_BLOCK_M_SIZE, device="cuda", dtype=torch.int64
        )
        GLOBAL_A_PTR_LIST = torch.zeros(
            MAX_NUM_BLOCK_M_SIZE, device="cuda", dtype=torch.int64
        )

    ds_list = []
    a_list = []
    curr_block_start = 0
    for raw_ds, raw_a in zip(raw_ds_list, raw_a_list, strict=True):
        num_blocks = (raw_ds.shape[0] + block_size_m - 1) // block_size_m
        GLOBAL_DS_PTR_LIST[curr_block_start : curr_block_start + num_blocks] = (
            raw_ds.data_ptr()
        )
        GLOBAL_A_PTR_LIST[curr_block_start : curr_block_start + num_blocks] = (
            raw_a.data_ptr()
        )
        ds_list.extend([raw_ds] * num_blocks)
        a_list.extend([raw_a] * num_blocks)
        curr_block_start += num_blocks
    return (
        GLOBAL_DS_PTR_LIST,
        GLOBAL_A_PTR_LIST,
        ds_list,
        a_list,
    )

@triton.jit
def fused_multi_lora_dyw_dsa_kernel(
    dy_ptr,
    w_ptr,
    ds_ptrs_list_ptr,
    a_ptrs_list_ptr,
    block_to_dropout_p_ptr,
    block_to_lookup_table_ptr,
    dropout_mask_ptr,
    dx_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_dym,
    stride_dyk,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_mask_m,
    stride_mask_n,
    stride_dxm,
    stride_dxn,
    MAX_R: tl.constexpr,
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

    r = tl.load(block_to_lookup_table_ptr + pid_m * 3 + 0)
    valid_size = tl.load(block_to_lookup_table_ptr + pid_m * 3 + 1)
    s_offset_pid_m = tl.load(block_to_lookup_table_ptr + pid_m * 3 + 2)
    dropout_p = tl.load(block_to_dropout_p_ptr + pid_m).to(OUTPUT_DTYPE)
    ds_ptr = tl.load(ds_ptrs_list_ptr + pid_m).to(tl.pointer_type(OUTPUT_DTYPE))
    a_ptr = tl.load(a_ptrs_list_ptr + pid_m).to(tl.pointer_type(OUTPUT_DTYPE))

    offs_r = tl.arange(0, MAX_R)
    offs_sm = (pid_m - s_offset_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ds_ptrs = ds_ptr + (offs_sm[:, None] * r + offs_r[None, :] * 1)
    a_ptrs = a_ptr + (offs_r[:, None] * N + offs_wn[None, :] * 1)
    ds_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < valid_size) & (offs_r[None, :] < r)
    a_mask = offs_r[:, None] < r

    # ------------------------------------------------------------
    # 3. Compute the LoRA part
    # ------------------------------------------------------------
    accum_main = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    ds = tl.load(ds_ptrs, mask=ds_mask, other=0.0)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    accum_main = tl.dot(ds, a, accum_main)

    if ENABLE_DROPOUT:
        dropout_mask_ptrs = dropout_mask_ptr + (
            offs_dym[:, None] * stride_mask_m + offs_wn[None, :] * stride_mask_n
        )
        mask_dropout_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < valid_size) & (
            offs_wn[None, :] < N
        )
        dropout_mask = tl.load(dropout_mask_ptrs, mask=mask_dropout_mask, other=0.0)
        accum_main = tl.where(dropout_mask, accum_main / (1 - dropout_p), 0.0)

    # ------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dy_ptrs = dy_ptr + (offs_dym[:, None] * stride_dym + offs_k[None, :] * stride_dyk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        mask_dy = (tl.arange(0, BLOCK_SIZE_M)[:, None] < valid_size) & (
            offs_k[None, :] < K - k * BLOCK_SIZE_K
        )
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
    dx_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < valid_size) & (
        offs_dxn[None, :] < N
    )
    tl.store(dx_ptrs, accum_main, mask=dx_mask)


def fused_multi_lora_dyw_dsa(
    dy: torch.Tensor,
    w: torch.Tensor,
    *,
    raw_ds_list: list[torch.Tensor],
    raw_a_list: list[torch.Tensor],
    block_to_dropout_p: torch.Tensor,
    block_to_lookup_table: torch.Tensor,
    max_r: int,
    init_zeros: bool = False,
    enable_dropout: bool = False,
    dropout_mask: torch.Tensor | None = None,
    block_size_m: int | None = None,
    config: LoRATritonConfig | None = None,
    **kwargs,
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
    if config is None:
        lora_kernel_config = get_lora_kernel_config("fused_multi_lora_dyw_dsa")
    else:
        lora_kernel_config = config

    if block_size_m is not None and lora_kernel_config.block_size_m != block_size_m:
        raise ValueError(
            f"block_size_m for fused_multi_lora_dyw_dsa is not set and "
            f"lora_kernel_config.block_size_m != input block_size_m. "
            f"lora_kernel_config.block_size_m: {lora_kernel_config.block_size_m}, "
            f"block_size_m: {block_size_m}."
        )

    if block_size_m is None:
        block_size_m = lora_kernel_config.block_size_m

    # Check constraints.
    m_y, n_y = dy.shape
    n_w, k_w = w.shape

    if n_y != n_w:
        msg = f"Incompatible dimensions for n of dy and w: {n_y} != {n_w}"
        raise ValueError(msg)

    # Construct the ds and a ptrs list
    ds_ptrs_list, a_ptrs_list, _, _ = construct_ds_and_a_ptrs_list(
        raw_ds_list=raw_ds_list,
        raw_a_list=raw_a_list,
        block_size_m=block_size_m,
    )

    if init_zeros:
        # This is only for the testing purpose.
        dx = torch.zeros((m_y, k_w), device=dy.device, dtype=dy.dtype)
    else:
        dx = torch.empty((m_y, k_w), device=dy.device, dtype=dy.dtype)

    # Change the representation of tensors to map the low-level implementation.
    M = m_y
    N = k_w
    K = n_w
    # Allocates output.
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    triton_config = lora_kernel_config.to_triton_config()

    # Multi-LoRA kernels use @triton.autotune, so we can't override configs manually
    # The config parameter is kept for API compatibility but ignored
    compiled_kernel = fused_multi_lora_dyw_dsa_kernel[grid](
        dy,
        w,
        ds_ptrs_list,
        a_ptrs_list,
        block_to_dropout_p,
        block_to_lookup_table,
        dropout_mask,
        dx,
        M,
        N,
        K,
        dy.stride(0),
        dy.stride(1),
        w.stride(0),
        w.stride(1),
        dropout_mask.stride(0) if dropout_mask is not None else 0,
        dropout_mask.stride(1) if dropout_mask is not None else 0,
        dx.stride(0),
        dx.stride(1),
        MAX_R=max_r,
        ENABLE_DROPOUT=enable_dropout,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(dx.dtype),
        **triton_config.all_kwargs(),
    )
    if compiled_kernel is not None and compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return dx

