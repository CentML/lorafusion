# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused Multi-LoRA xw + sb."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds

# Given a pid_m
# To calculate the correct output, we need to compute
# tile_x @ tile_w.T + tile_s @ tile_b.T * alpha
# 1. tile_x: easy to index: according to the pid_m
# 2. tile_w: easy to index: according to the pid_n
# 3. tile_s: not easy.
#   because s is a list of tensors. we also need to get the pid_offset_m.
# 4. tile_b: easy to index. because it is fixed.

MULTI_LORA_GLOBAL_INFO = None


def set_multi_lora_global_info(
    seq_len_list: list[int],
    adapter_idx: list[int],
    adapter_info: dict[int, tuple[int, float, float]],
    padded_seq_len_list: list[int],
    lookup_table: torch.Tensor,
    alpha_list: torch.Tensor,
    block_size_m: int,
    dropout_p_list: torch.Tensor = None,
) -> None:
    """Set the global info for the Multi-LoRA xw + sb kernel."""
    global MULTI_LORA_GLOBAL_INFO
    max_r = lookup_table[:, 0].max().item()
    enable_dropout = torch.any(dropout_p_list > 0.0).item()
    MULTI_LORA_GLOBAL_INFO = {
        "seq_len_list": seq_len_list,
        "adapter_idx": adapter_idx,
        "adapter_info": adapter_info,
        "padded_seq_len_list": padded_seq_len_list,
        "lookup_table": lookup_table,
        "alpha_list": alpha_list,
        "dropout_p_list": dropout_p_list,
        "enable_dropout": enable_dropout,
        "max_r": max_r,
        "block_size_m": block_size_m,
    }


def get_multi_lora_global_info() -> dict[str, Any]:
    """Get the global info for the Multi-LoRA xw + sb kernel."""
    return MULTI_LORA_GLOBAL_INFO


def prepare_inputs_for_multi_lora(
    seq_len_list: list[int],
    adapter_idx: list[int],
    adapter_info: dict[int, tuple[int, float, float]],
    block_size_m: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Prepare the input tensors for the fused Multi-LoRA xw + sb kernel."""
    # Return:
    # 1. total_seq_len: total sequence length
    # 2. lookup table: (contains the r, and mask_size for each m-th block)
    # 3. alpha_list: (contains the alpha for each m-th block)

    # First, pad the seq_len_list to the multiple of block_size_m, also
    # calculate the strides for each m-th block.
    padded_seq_len_list = [
        ((seq_len - 1) // block_size_m + 1) * block_size_m for seq_len in seq_len_list
    ]
    # Because it's already padded, we can directly calculate the num_pid_m.
    num_blocks = len(padded_seq_len_list) // block_size_m
    # Masks for each m-th block.
    # For example, if seq_len_list = [129, 130, 128],
    # then padded_seq_len_list = [256, 256, 128],
    # and the valid_size_list = [128, 1, 128, 2, 128]

    # Calculate the total number of blocks needed
    total_blocks = sum(padded_seq_len_list) // block_size_m

    # Initialize lookup table, and alpha list
    # For each block, we need to store:
    # > [R, valid_size, s_offset_m, a_stride_k, a_stride_r,
    #                               s_stride_m, s_stride_k, b_stride_r, b_stride_n]
    # >>[R, valid_size, s_offset_m, 1, K, R, 1, 1, R]
    # Therefore, we actually only need to store R and valid_size
    lookup_table = torch.zeros((total_blocks, 3), dtype=torch.int32, device="cpu")
    alpha_list = torch.zeros(total_blocks, dtype=output_dtype, device="cpu")
    dropout_p_list = torch.zeros(total_blocks, dtype=output_dtype, device="cpu")

    # Fill in lookup table and alpha list
    block_idx = 0
    for seq_idx, (seq_len, adapter_id) in enumerate(
        zip(seq_len_list, adapter_idx, strict=True)
    ):
        if adapter_id not in adapter_info:
            msg = f"Adapter ID {adapter_id} not found in adapter_info"
            raise ValueError(msg)

        # Get the rank and alpha for this adapter
        r, alpha, dropout_p = adapter_info[adapter_id]

        # Calculate number of blocks for this sequence
        num_blocks = padded_seq_len_list[seq_idx] // block_size_m
        start_m = block_idx

        # For each block in this sequence
        for i in range(num_blocks):
            # Calculate the actual mask size for this block (handle padding)
            valid_size = min(block_size_m, seq_len - i * block_size_m)

            # Store information in lookup table
            lookup_table[block_idx, 0] = r  # LoRA rank
            lookup_table[block_idx, 1] = valid_size  # Actual number of active rows
            lookup_table[block_idx, 2] = start_m  # Start m

            # Store alpha value
            alpha_list[block_idx] = alpha
            dropout_p_list[block_idx] = dropout_p

            block_idx += 1

    # Verify we've filled all blocks
    if block_idx != total_blocks:
        msg = f"Expected {total_blocks} blocks, but filled {block_idx}"
        logger.warning(msg)
        raise ValueError(msg)

    return padded_seq_len_list, lookup_table, alpha_list, dropout_p_list


def prepare_func(
    seq_len_list: list[int],
    adapter_idx: list[int],
    adapter_info: dict[int, tuple[int, float]],
    n: int,
    k: int,
    block_size_m: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_bias: bool = False,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA xw + sb kernel."""
    if len(seq_len_list) != len(adapter_idx):
        msg = f"Incompatible dimensions: {len(seq_len_list)} != {len(adapter_idx)}"
        raise ValueError(msg)

    padded_seq_len_list, lookup_table, alpha_list, _ = prepare_inputs_for_multi_lora(
        seq_len_list=seq_len_list,
        adapter_idx=adapter_idx,
        adapter_info=adapter_info,
        block_size_m=block_size_m,
    )
    lookup_table = lookup_table.to(device="cuda")
    alpha_list = alpha_list.to(device="cuda")

    x = torch.rand(sum(padded_seq_len_list), k, device="cuda", dtype=dtype) / 10
    w = torch.rand(n, k, device="cuda", dtype=dtype) / 10
    bias = torch.rand(n, device="cuda", dtype=dtype) / 10 if with_bias else None

    raw_s_list, raw_b_list = [], []
    for seq_len, adapter_id in zip(padded_seq_len_list, adapter_idx, strict=True):
        r, alpha, dropout_p = adapter_info[adapter_id]
        s = torch.rand(seq_len, r, device="cuda", dtype=dtype) / 10
        b = torch.rand(n, r, device="cuda", dtype=dtype) / 10
        raw_s_list.append(s)
        raw_b_list.append(b)

    set_multi_lora_global_info(
        seq_len_list=seq_len_list,
        adapter_idx=adapter_idx,
        adapter_info=adapter_info,
        padded_seq_len_list=padded_seq_len_list,
        lookup_table=lookup_table,
        alpha_list=alpha_list,
        block_size_m=block_size_m,
    )

    return {
        "x": x,
        "w": w,
        "raw_s_list": raw_s_list,
        "raw_b_list": raw_b_list,
        "lookup_table": lookup_table,
        "alpha_list": alpha_list,
        "block_size_m": block_size_m,
        "padded_seq_len_list": padded_seq_len_list,
        "adapter_idx": adapter_idx,
        "adapter_info": adapter_info,
        "bias": bias,
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
    bias: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    M, K = x.shape
    N, K = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=x.dtype)

    multi_lora_global_info = get_multi_lora_global_info()
    seq_len_list = multi_lora_global_info["seq_len_list"]
    adapter_idx = multi_lora_global_info["adapter_idx"]
    adapter_info = multi_lora_global_info["adapter_info"]
    padded_seq_len_list = multi_lora_global_info["padded_seq_len_list"]

    num_loras = len(seq_len_list)

    curr_start = 0
    for i in range(num_loras):
        seq_len = seq_len_list[i]
        padded_seq_len = padded_seq_len_list[i]
        raw_s = raw_s_list[i]
        raw_b = raw_b_list[i]
        alpha = adapter_info[adapter_idx[i]][1]

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

    multi_lora_global_info = get_multi_lora_global_info()
    lookup_table = multi_lora_global_info["lookup_table"]
    alpha_list = multi_lora_global_info["alpha_list"]

    # Construct the s and b ptrs list
    _, _, s_list, b_list = construct_s_and_b_ptrs_list(
        raw_s_list=raw_s_list,
        raw_b_list=raw_b_list,
    )

    num_blocks = lookup_table.shape[0]
    for i in range(num_blocks):
        valid_size = lookup_table[i, 1]
        s_offset_m = lookup_table[i, 2]
        alpha = alpha_list[i]

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


def fused_multi_lora_xw_sb_kernel_get_configs() -> list[triton.Config]:
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
        for BN in [256]
        for BK in [64]
        for s in ([4])
        for w in [8]
        if BM * BK < 256 * 256
    ]


@triton.autotune(
    configs=fused_multi_lora_xw_sb_kernel_get_configs(),
    key=["M", "N", "K", "OUTPUT_DTYPE"],
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
    In the alpha_list, we store the following information for each m-th block:
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


MAX_NUM_BLOCK_M_SIZE = 128  # 8192 tokens
GLOBAL_S_PTR_LIST = None
GLOBAL_B_PTR_LIST = None


def construct_s_and_b_ptrs_list(
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    block_size_m: int = 128,
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
) -> torch.Tensor:
    """Triton Fused LoRA xw + sb."""
    # Check constraints.
    if x.shape[1] != w.shape[1]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {w.shape[1]}. "
            f"x: {x.shape}, w: {w.shape}"
        )
        raise ValueError(msg)

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
    )
    if compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return out


def verify_kernel_correctness(
    seq_len_list_choices: list[list[int]],
    adapter_idx_choices: list[list[int]],
    adapter_info: dict[int, tuple[int, float]],
    n: int,
    k: int,
    block_size_m: int,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions.

    This is particularly important when M is not divisible by 128.
    """
    for seq_len_list, adapter_idx in zip(
        seq_len_list_choices, adapter_idx_choices, strict=True
    ):
        logger.info(
            f"Verifying kernel correctness for seq_len_list={seq_len_list} and "
            f"adapter_idx={adapter_idx}..."
        )

        # Test without bias
        logger.info("Testing without bias...")
        inputs_no_bias = prepare_func(
            seq_len_list=seq_len_list,
            adapter_idx=adapter_idx,
            adapter_info=adapter_info,
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

        # Compute reference result
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
                f"adapter_idx={adapter_idx} without bias"
            )
        except AssertionError as e:
            logger.error(
                f"Verification failed for seq_len_list={seq_len_list} and "
                f"adapter_idx={adapter_idx} without bias"
            )
            logger.error(e)

        # Test with bias
        logger.info("Testing with bias...")
        inputs_with_bias = prepare_func(
            seq_len_list=seq_len_list,
            adapter_idx=adapter_idx,
            adapter_info=adapter_info,
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
                f"adapter_idx={adapter_idx} with bias"
            )
        except AssertionError as e:
            logger.error(
                f"Verification failed for seq_len_list={seq_len_list} and "
                f"adapter_idx={adapter_idx} with bias"
            )
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    adapter_info = {
        0: (16, 16.0, 0.1),
        1: (16, 16.0, 0.1),
        2: (16, 16.0, 0.1),
        3: (16, 16.0, 0.1),
    }

    seq_len_list_choices = [[3380, 640]]
    adapter_idx_choices = [[0, 1]]

    # Test with various M values, including those not divisible by 128
    n = 4096
    k = 4096
    max_r = 16
    block_size_m = 128
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(
        seq_len_list_choices=seq_len_list_choices,
        adapter_idx_choices=adapter_idx_choices,
        adapter_info=adapter_info,
        n=n,
        k=k,
        block_size_m=block_size_m,
        dtype=dtype,
    )

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for seq_len_list, adapter_idx in zip(
        seq_len_list_choices, adapter_idx_choices, strict=True
    ):
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking fused_lora_xw_sb with seq_len_list={seq_len_list} and "
            f"adapter_idx={adapter_idx}"
        )

        # Benchmark without bias
        logger.info("Benchmarking without bias...")
        curr_prepare_func_no_bias = partial(
            prepare_func,
            seq_len_list=seq_len_list,
            adapter_idx=adapter_idx,
            adapter_info=adapter_info,
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
            msg=f"fused_multi_lora_xw_sb_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}_no_bias",
        )

        benchmark(
            torch_xw_ref,
            prepare_func=curr_prepare_func_no_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_xw_ref_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}",
        )

        benchmark(
            torch_multi_lora_xw_sb_ref,
            prepare_func=curr_prepare_func_no_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_multi_lora_xw_sb_ref_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}_no_bias",
        )

        # Benchmark with bias
        logger.info("Benchmarking with bias...")
        curr_prepare_func_with_bias = partial(
            prepare_func,
            seq_len_list=seq_len_list,
            adapter_idx=adapter_idx,
            adapter_info=adapter_info,
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
            msg=f"fused_multi_lora_xw_sb_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}_with_bias",
        )

        benchmark(
            torch_multi_lora_xw_sb_ref,
            prepare_func=curr_prepare_func_with_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_multi_lora_xw_sb_ref_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}_with_bias",
        )
