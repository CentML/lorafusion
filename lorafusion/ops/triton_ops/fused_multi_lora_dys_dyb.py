# ruff: noqa
# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused Multi-LoRA dy @ s + dy @ b * alpha."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
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


def torch_multi_lora_dys_dyb_ref(
    dy: torch.Tensor,
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    block_to_lookup_table: torch.Tensor,
    block_to_alpha: torch.Tensor,
    block_size_m: int = 128,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Torch reference implementation of multi-lora dy @ s + dy @ b * alpha."""
    # Initialize the db_list and ds_list
    db_list = []
    ds_list = []

    for i, (s, b) in enumerate(zip(raw_s_list, raw_b_list, strict=True)):
        # Calculate number of blocks for this adapter
        num_blocks = (s.shape[0] + block_size_m - 1) // block_size_m

        # Extract sequence length from s
        seq_len = s.shape[0]

        # Initialize db and ds for this adapter
        db = torch.zeros_like(b)
        ds = torch.zeros_like(s)

        # Process each block
        for j in range(num_blocks):
            block_idx = (
                sum(
                    [
                        (raw_s.shape[0] + block_size_m - 1) // block_size_m
                        for raw_s in raw_s_list[:i]
                    ]
                )
                + j
            )
            r = block_to_lookup_table[block_idx, 0].item()
            valid_size = block_to_lookup_table[block_idx, 1].item()
            alpha = block_to_alpha[block_idx].item()

            start_m = j * block_size_m
            end_m = min(start_m + valid_size, seq_len)

            # Extract slices
            dy_slice = dy[start_m:end_m]
            s_slice = s[start_m:end_m]

            # Compute the gradients
            db_slice = (dy_slice.T @ s_slice) * alpha
            ds_slice = (dy_slice @ b) * alpha

            # Accumulate
            db += db_slice
            ds[start_m:end_m] += ds_slice[start_m:end_m]

        db_list.append(db)
        ds_list.append(ds)

    return db_list, ds_list


MAX_NUM_BLOCK_M_SIZE = 128  # 8192 tokens
GLOBAL_S_PTR_LIST = None
GLOBAL_B_PTR_LIST = None


def construct_s_and_b_ptrs_list(
    raw_s_list: list[torch.Tensor],
    raw_b_list: list[torch.Tensor],
    block_size_m: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Construct the s and b ptrs list."""
    global GLOBAL_S_PTR_LIST, GLOBAL_B_PTR_LIST  # noqa: PLW0603

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


def fused_multi_lora_dys_dyb_kernel_get_configs() -> list[triton.Config]:
    """Get the configurations for the fused LoRA dy @ s + dy @ b * alpha kernel."""
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": GM,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BK in [128]
        for GM in [8]
        for s in ([5])
        for w in [8]
        if BM * BK < 256 * 256
    ]


@triton.autotune(
    configs=fused_multi_lora_dys_dyb_kernel_get_configs(),
    key=["M", "K", "MAX_R"],
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

    # Masks – use local row indices (0..BLOCK_SIZE_M-1) for valid_size comparison
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
    s_ptrs_list: torch.Tensor | None = None,
    b_ptrs_list: torch.Tensor | None = None,
    raw_s_list: list[torch.Tensor] | None = None,
    raw_b_list: list[torch.Tensor] | None = None,
    block_size_m: int = 128,
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

    # ================================================================
    # V1
    # ================================================================
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


def verify_kernel_correctness(
    seq_len_list_choices: list[list[int]],
    lora_rank_list: list[int],
    alpha_list: list[float],
    n: int,
    block_size_m: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions."""
    for seq_len_list in seq_len_list_choices:
        logger.info(f"Verifying kernel correctness for seq_len_list={seq_len_list}...")

        # Prepare inputs
        M = sum(seq_len_list)
        dy = torch.randn(M, n, device="cuda", dtype=dtype) / 10

        raw_s_list = []
        raw_b_list = []

        # Create lookup table
        total_blocks = sum(
            (seq_len + block_size_m - 1) // block_size_m for seq_len in seq_len_list
        )
        block_to_lookup_table = torch.zeros(
            (total_blocks, 3), dtype=torch.int32, device="cuda"
        )
        block_to_alpha = torch.zeros(total_blocks, dtype=dtype, device="cuda")

        block_idx = 0
        for i, seq_len in enumerate(seq_len_list):
            r = lora_rank_list[i] if i < len(lora_rank_list) else lora_rank_list[-1]
            alpha = alpha_list[i] if i < len(alpha_list) else alpha_list[-1]

            # Create tensors for this adapter
            s = torch.randn(seq_len, r, device="cuda", dtype=dtype) / 10
            b = torch.randn(n, r, device="cuda", dtype=dtype) / 10
            raw_s_list.append(s)
            raw_b_list.append(b)

            # Fill lookup table
            num_blocks = (seq_len + block_size_m - 1) // block_size_m
            start_m = block_idx

            for j in range(num_blocks):
                valid_size = min(block_size_m, seq_len - j * block_size_m)

                block_to_lookup_table[block_idx, 0] = r  # LoRA rank
                block_to_lookup_table[block_idx, 1] = valid_size  # Valid size in block
                block_to_lookup_table[block_idx, 2] = start_m  # Start m offset
                block_to_alpha[block_idx] = alpha

                block_idx += 1

        # Run triton kernel
        db_list_triton, ds_list_triton = fused_multi_lora_dys_dyb(
            dy=dy,
            raw_s_list=raw_s_list,
            raw_b_list=raw_b_list,
            block_to_lookup_table=block_to_lookup_table,
            block_to_alpha=block_to_alpha,
            max_r=max(lora_rank_list),
            total_r=max(lora_rank_list),
            block_size_m=block_size_m,
        )

        # Compute reference result
        db_list_ref, ds_list_ref = torch_multi_lora_dys_dyb_ref(
            dy=dy,
            raw_s_list=raw_s_list,
            raw_b_list=raw_b_list,
            block_to_lookup_table=block_to_lookup_table,
            block_to_alpha=block_to_alpha,
            block_size_m=block_size_m,
        )

        # Compare results
        try:
            for i in range(len(db_list_triton)):
                logger.info(f"Verifying adapter {i}...")
                # Use higher tolerance for bfloat16
                atol = 1e-4
                logger.info(f"Verifying db for adapter {i}...")
                assert_verbose_allclose_two_rounds(
                    db_list_triton[i], db_list_ref[i], atol=atol
                )
                logger.info(f"Verifying ds for adapter {i}...")
                assert_verbose_allclose_two_rounds(
                    ds_list_triton[i], ds_list_ref[i], atol=atol
                )
            logger.success(f"Verification passed for seq_len_list={seq_len_list}")
        except AssertionError as e:
            logger.error(f"Verification failed for seq_len_list={seq_len_list}")
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # Test with various M values, including those not divisible by block sizes
    seq_len_list_choices = [[2048, 2048], [2837, 1258], [4096, 4097], [3071, 1023]]
    lora_rank_list = [16, 16]
    alpha_list = [16.0, 16.0]
    n = 4096  # Keep N as multiple of 256
    block_size_m = 128
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(
        seq_len_list_choices=seq_len_list_choices,
        lora_rank_list=lora_rank_list,
        alpha_list=alpha_list,
        n=n,
        block_size_m=block_size_m,
        dtype=dtype,
    )

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for seq_len_list in seq_len_list_choices:
        total_m = sum(seq_len_list)
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking fused_multi_lora_dys_dyb with seq_len_list={seq_len_list}, total_m={total_m}"
        )

        # Create input tensors and lookup tables for benchmarking
        dy = torch.randn(total_m, n, device="cuda", dtype=dtype) / 10
        raw_s_list = []
        raw_b_list = []

        total_blocks = sum(
            (seq_len + block_size_m - 1) // block_size_m for seq_len in seq_len_list
        )
        block_to_lookup_table = torch.zeros(
            (total_blocks, 3), dtype=torch.int32, device="cuda"
        )
        block_to_alpha = torch.zeros(total_blocks, dtype=dtype, device="cuda")

        block_idx = 0
        for i, seq_len in enumerate(seq_len_list):
            r = lora_rank_list[i] if i < len(lora_rank_list) else lora_rank_list[-1]
            alpha = alpha_list[i] if i < len(alpha_list) else alpha_list[-1]

            s = torch.randn(seq_len, r, device="cuda", dtype=dtype) / 10
            b = torch.randn(n, r, device="cuda", dtype=dtype) / 10
            raw_s_list.append(s)
            raw_b_list.append(b)

            num_blocks = (seq_len + block_size_m - 1) // block_size_m
            start_m = block_idx

            for j in range(num_blocks):
                valid_size = min(block_size_m, seq_len - j * block_size_m)

                block_to_lookup_table[block_idx, 0] = r
                block_to_lookup_table[block_idx, 1] = valid_size
                block_to_lookup_table[block_idx, 2] = start_m
                block_to_alpha[block_idx] = alpha

                block_idx += 1

        benchmark_inputs = {
            "dy": dy,
            "raw_s_list": raw_s_list,
            "raw_b_list": raw_b_list,
            "block_to_lookup_table": block_to_lookup_table,
            "block_to_alpha": block_to_alpha,
            "max_r": max(lora_rank_list),
            "total_r": max(lora_rank_list),
            "block_size_m": block_size_m,
        }

        benchmark(
            fused_multi_lora_dys_dyb,
            prepare_func=lambda: benchmark_inputs,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_multi_lora_dys_dyb_seq_len_list_{seq_len_list}",
        )
