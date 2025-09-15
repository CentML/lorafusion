# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused Multi-LoRA dy @ s + dy @ b * alpha."""

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.config import LoRATritonConfig, get_lora_kernel_config
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype

MAX_NUM_BLOCK_M_SIZE = 128  # max: MAX_NUM_BLOCK_M_SIZE * BLOCK_SIZE_M tokens
GLOBAL_S_PTR_LIST = None
GLOBAL_B_PTR_LIST = None


def construct_s_and_b_ptrs_list(
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    block_size_m: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Construct the s and b ptrs list."""
    global GLOBAL_S_PTR_LIST, GLOBAL_B_PTR_LIST

    if GLOBAL_S_PTR_LIST is None:
        GLOBAL_S_PTR_LIST = torch.zeros(
            MAX_NUM_BLOCK_M_SIZE, device="cuda", dtype=torch.int64
        )
        GLOBAL_B_PTR_LIST = torch.zeros(
            MAX_NUM_BLOCK_M_SIZE, device="cuda", dtype=torch.int64
        )

    s_list = []
    b_list = []
    curr_block_start = 0
    for raw_s, raw_b in zip(raw_s_list, raw_b_list, strict=True):
        num_blocks = (raw_s.shape[0] + block_size_m - 1) // block_size_m
        GLOBAL_S_PTR_LIST[curr_block_start : curr_block_start + num_blocks] = (
            raw_s.data_ptr()
        )
        GLOBAL_B_PTR_LIST[curr_block_start : curr_block_start + num_blocks] = (
            raw_b.data_ptr()
        )
        s_list.extend([raw_s] * num_blocks)
        b_list.extend([raw_b] * num_blocks)
        curr_block_start += num_blocks

    return (
        GLOBAL_S_PTR_LIST,
        GLOBAL_B_PTR_LIST,
        s_list,
        b_list,
    )


@triton.jit
def fused_multi_lora_dys_dyb_kernel(
    dy_ptr,
    s_ptrs_list_ptr,
    b_ptrs_list_ptr,
    db_ptrs_list_ptr,
    ds_ptrs_list_ptr,
    block_to_alpha_ptr,
    block_to_lookup_table_ptr,
    M,
    K: tl.constexpr,
    total_r,
    MAX_R: tl.constexpr,
    stride_dym,
    stride_dyk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused Multi-LoRA dy @ s + dy @ b * alpha kernel.

    We use split-k strategy to compute the db and ds since the number of
    the dimension of r is usually quite small, e.g. 16, 32, etc.

    Compute:
    - db = dy.T @ s * alpha
    - ds = dy @ b * alpha

    The shape definition:
    - dy: [M, K]
    - s: from s_ptrs_list_ptr
    - b: from b_ptrs_list_ptr
    - db: to db_ptrs_list_ptr
    - ds: to ds_ptrs_list_ptr
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

    stride_sm = total_r

    # ------------------------------------------------------------
    #  2. Load metadata from lookup table
    # ------------------------------------------------------------
    r = tl.load(block_to_lookup_table_ptr + pid_m * 3 + 0)
    valid_size = tl.load(block_to_lookup_table_ptr + pid_m * 3 + 1)
    s_offset_pid_m = tl.load(block_to_lookup_table_ptr + pid_m * 3 + 2)
    alpha = tl.load(block_to_alpha_ptr + pid_m)

    # Load tensor pointers
    s_ptr = tl.load(s_ptrs_list_ptr + pid_m).to(tl.pointer_type(OUTPUT_DTYPE))
    b_ptr = tl.load(b_ptrs_list_ptr + pid_m).to(tl.pointer_type(OUTPUT_DTYPE))
    db_ptr = tl.load(db_ptrs_list_ptr + pid_m).to(tl.pointer_type(tl.float32))
    ds_ptr = tl.load(ds_ptrs_list_ptr + pid_m).to(tl.pointer_type(tl.float32))

    # ------------------------------------------------------------
    #  3. Compute the tile indices for M, K, R
    # ------------------------------------------------------------
    # Generate proper offsets
    offs_dym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dyk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_r = tl.arange(0, MAX_R)
    offs_sm = (pid_m - s_offset_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Compute pointers and masks
    dy_ptrs = dy_ptr + (offs_dym[:, None] * stride_dym + offs_dyk[None, :] * stride_dyk)
    s_ptrs = s_ptr + (offs_sm[:, None] * stride_sm + offs_r[None, :] * 1)
    b_ptrs = b_ptr + (offs_dyk[:, None] * r + offs_r[None, :] * 1)

    # Row indices local to the current block
    rows_in_block = tl.arange(0, BLOCK_SIZE_M)[:, None]

    # Masks - use local row indices (0..BLOCK_SIZE_M-1) for valid_size comparison
    mask_dy = (rows_in_block < valid_size) & (offs_dyk[None, :] < K)
    mask_s = (rows_in_block < valid_size) & (offs_r[None, :] < r)
    mask_b = (offs_dyk[:, None] < K) & (offs_r[None, :] < r)

    # ------------------------------------------------------------
    # 4. Initialize accumulators
    # ------------------------------------------------------------
    accum_db = tl.zeros((BLOCK_SIZE_K, MAX_R), dtype=tl.float32)
    accum_ds = tl.zeros((BLOCK_SIZE_M, MAX_R), dtype=tl.float32)

    # ------------------------------------------------------------
    # 5. Main computation
    # ------------------------------------------------------------
    # Load the data
    dy = tl.load(dy_ptrs, mask=mask_dy, other=0.0)
    s = tl.load(s_ptrs, mask=mask_s, other=0.0)
    b = tl.load(b_ptrs, mask=mask_b, other=0.0)

    # Compute db = dy.T @ s * alpha
    accum_db = tl.dot(dy.T, s, accum_db)
    accum_db = accum_db * alpha

    # Compute ds = dy @ b * alpha
    accum_ds = tl.dot(dy, b, accum_ds)
    accum_ds = accum_ds * alpha

    # ------------------------------------------------------------
    # 6. Store results
    # ------------------------------------------------------------
    offs_dbk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_dsm = (pid_m - s_offset_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    db_ptrs = db_ptr + (offs_dbk[:, None] * r + offs_r[None, :] * 1)
    ds_ptrs = ds_ptr + (offs_dsm[:, None] * r + offs_r[None, :] * 1)

    # Create masks to handle boundary conditions
    db_mask = (offs_dbk[:, None] < K) & (offs_r[None, :] < r)
    ds_mask = (rows_in_block < valid_size) & (offs_r[None, :] < r)

    # Store results
    tl.atomic_add(db_ptrs, accum_db, mask=db_mask)
    tl.atomic_add(ds_ptrs, accum_ds, mask=ds_mask)


def fused_multi_lora_dys_dyb(
    dy: torch.Tensor,
    block_to_lookup_table: torch.Tensor,
    block_to_alpha: torch.Tensor,
    max_r: int,
    total_r: int,
    block_size_m,
    s_ptrs_list: torch.Tensor | None = None,
    b_ptrs_list: torch.Tensor | None = None,
    raw_s_list: list[torch.Tensor] | None = None,
    raw_b_list: list[torch.Tensor] | None = None,
    *,
    config: LoRATritonConfig | None = None,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Fused multi-LoRA dy @ s + dy @ b * alpha.

    Compute db_list = [dy.T @ s * alpha for s in raw_s_list]
    Compute ds_list = [dy @ b * alpha for b in raw_b_list]

    Args:
        dy: Gradient of the loss with respect to the output, shape [M, K]
        raw_s_list: List of LoRA output tensors, each with shape [seq_len, R]
        raw_b_list: List of LoRA weight tensors, each with shape [K, R]
        block_to_lookup_table: Lookup table for each block, shape [num_blocks, 3]
        block_to_alpha: Alpha scaling factor for each block, shape [num_blocks]
        block_size_m: Block size for the m dimension
        config: Optional LoRA Triton configuration. If None, uses default config.

    Returns:
        A tuple containing:
            db_list: List of gradients with respect to each b
            ds_list: List of gradients with respect to each s
    """
    # Construct the pointers for s and b tensors
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

    # Create result tensors for each adapter
    db_list = []
    ds_list = []
    db_ptrs_list = torch.zeros_like(s_ptrs_list)
    ds_ptrs_list = torch.zeros_like(s_ptrs_list)

    # Initialize outputs and set up pointers
    for i, (raw_s, raw_b) in enumerate(zip(raw_s_list, raw_b_list, strict=True)):
        curr_db = torch.zeros(
            (raw_b.shape[0], raw_b.shape[1]), device=dy.device, dtype=torch.float32
        )
        curr_ds = torch.zeros(
            (raw_s.shape[0], raw_s.shape[1]), device=dy.device, dtype=torch.float32
        )

        db_list.append(curr_db)
        ds_list.append(curr_ds)

        # Set up pointers for blocks of this adapter
        num_blocks = (raw_s.shape[0] + block_size_m - 1) // block_size_m
        start_block = sum(
            [(s.shape[0] + block_size_m - 1) // block_size_m for s in raw_s_list[:i]]
        )
        db_ptrs_list[start_block : start_block + num_blocks] = curr_db.data_ptr()
        ds_ptrs_list[start_block : start_block + num_blocks] = curr_ds.data_ptr()

    # Set up dimensions for kernel
    M = dy.shape[0]
    K = dy.shape[1]

    # Launch the kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )

    if config is None:
        lora_kernel_config = get_lora_kernel_config("fused_multi_lora_dys_dyb")
    else:
        lora_kernel_config = config

    if block_size_m is not None and lora_kernel_config.block_size_m != block_size_m:
        raise ValueError(
            f"block_size_m for fused_multi_lora_dys_dyb is not set and "
            f"lora_kernel_config.block_size_m != input block_size_m. "
            f"lora_kernel_config.block_size_m: {lora_kernel_config.block_size_m}, "
            f"block_size_m: {block_size_m}."
        )

    triton_config = lora_kernel_config.to_triton_config()

    # Multi-LoRA kernels use @triton.autotune, so we can't override configs manually
    # The config parameter is kept for API compatibility but ignored
    compiled_kernel = fused_multi_lora_dys_dyb_kernel[grid](
        dy,
        s_ptrs_list,
        b_ptrs_list,
        db_ptrs_list,
        ds_ptrs_list,
        block_to_alpha,
        block_to_lookup_table,
        M,
        K,
        total_r=total_r,
        MAX_R=max_r,
        stride_dym=dy.stride(0),
        stride_dyk=dy.stride(1),
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(dy.dtype),
        **triton_config.all_kwargs(),
    )

    # Convert outputs to the same dtype as inputs
    db_list = [db.to(dy.dtype) for db in db_list]
    ds_list = [ds.to(dy.dtype) for ds in ds_list]

    if compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")

    return db_list, ds_list
