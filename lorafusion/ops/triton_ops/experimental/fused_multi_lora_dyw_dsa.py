# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused LoRA dy @ w + ds @ a * dropout_scale."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.experimental.fused_multi_lora_xw_sb import (
    get_multi_lora_global_info,
    prepare_inputs_for_multi_lora,
    set_multi_lora_global_info,
)
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds


def prepare_func(
    seq_len_list: list[int],
    adapter_idx: list[int],
    adapter_info: dict[int, tuple[int, float]],
    n: int,
    k: int,
    block_size_m: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA dyw + dsa kernel."""
    if len(seq_len_list) != len(adapter_idx):
        msg = f"Incompatible dimensions: {len(seq_len_list)} != {len(adapter_idx)}"
        raise ValueError(msg)

    padded_seq_len_list, lookup_table, alpha_list, dropout_p_list = (
        prepare_inputs_for_multi_lora(
            seq_len_list=seq_len_list,
            adapter_idx=adapter_idx,
            adapter_info=adapter_info,
            block_size_m=block_size_m,
        )
    )
    lookup_table = lookup_table.to(device="cuda")
    alpha_list = alpha_list.to(device="cuda")
    dropout_p_list = dropout_p_list.to(device="cuda")

    dy = torch.rand(sum(padded_seq_len_list), n, device="cuda", dtype=dtype) / 10
    w = torch.rand(n, k, device="cuda", dtype=dtype) / 10

    raw_ds_list, raw_a_list = [], []
    for seq_len, adapter_id in zip(padded_seq_len_list, adapter_idx, strict=True):
        r, alpha, dropout_p = adapter_info[adapter_id]
        ds = torch.rand(seq_len, r, device="cuda", dtype=dtype) / 10
        a = torch.rand(r, k, device="cuda", dtype=dtype) / 10
        raw_ds_list.append(ds)
        raw_a_list.append(a)

    set_multi_lora_global_info(
        seq_len_list=seq_len_list,
        adapter_idx=adapter_idx,
        adapter_info=adapter_info,
        padded_seq_len_list=padded_seq_len_list,
        lookup_table=lookup_table,
        alpha_list=alpha_list,
        dropout_p_list=dropout_p_list,
        block_size_m=block_size_m,
    )

    # We randomly generate the dropout mask for the testing purpose.
    dropout_mask = (
        torch.randn(sum(padded_seq_len_list), k, device="cuda", dtype=torch.float32)
        > dropout_p
    )

    return {
        "dy": dy,
        "w": w,
        "raw_ds_list": raw_ds_list,
        "raw_a_list": raw_a_list,
        "dropout_p": dropout_p,
        "dropout_mask": dropout_mask,
        "lookup_table": lookup_table,
        "alpha_list": alpha_list,
        "dropout_p_list": dropout_p_list,
        "block_size_m": block_size_m,
        "padded_seq_len_list": padded_seq_len_list,
        "adapter_idx": adapter_idx,
        "adapter_info": adapter_info,
    }


def torch_dyw_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of dy @ w."""
    return dy @ w


def torch_multi_lora_dyw_dsa_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    raw_ds_list: list[torch.Tensor],
    raw_a_list: list[torch.Tensor],
    dropout_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA dyw + dsa."""
    M, N = dy.shape
    N, K = w.shape
    out = torch.zeros((M, K), device=dy.device, dtype=dy.dtype)

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
        raw_ds = raw_ds_list[i]
        raw_a = raw_a_list[i]
        dropout_p = adapter_info[adapter_idx[i]][2]

        # For x
        curr_dy = dy[curr_start : curr_start + seq_len, :]
        curr_dyw = curr_dy @ w

        # For s and w
        dropout_mask_curr = dropout_mask[curr_start : curr_start + seq_len, :]
        curr_ds = raw_ds[:seq_len, :]
        curr_dsa = curr_ds @ raw_a
        curr_dsa = torch.where(dropout_mask_curr, curr_dsa / (1 - dropout_p), 0.0)
        curr_dyw += curr_dsa

        # Store the result
        out[curr_start : curr_start + seq_len, :] = curr_dyw

        # Update the current start
        curr_start += padded_seq_len

    return out


MAX_NUM_BLOCK_M_SIZE = 128  # 8192 tokens
GLOBAL_DS_PTR_LIST = None
GLOBAL_A_PTR_LIST = None


def construct_ds_and_a_ptrs_list(
    raw_ds_list: list[torch.Tensor],
    raw_a_list: list[torch.Tensor],
    block_size_m: int = 128,
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


def torch_multi_lora_dyw_dsa_ref_2(
    dy: torch.Tensor,
    w: torch.Tensor,
    raw_ds_list: list[torch.Tensor],
    raw_a_list: list[torch.Tensor],
    block_size_m: int,
    dropout_p_list: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA dyw + dsa."""
    M, N = dy.shape
    N, K = w.shape
    out = torch.zeros((M, K), device=dy.device, dtype=dy.dtype)

    multi_lora_global_info = get_multi_lora_global_info()
    lookup_table = multi_lora_global_info["lookup_table"]
    dropout_p_list = multi_lora_global_info["dropout_p_list"]

    # Construct the ds and a ptrs list
    _, _, ds_list, a_list = construct_ds_and_a_ptrs_list(
        raw_ds_list=raw_ds_list,
        raw_a_list=raw_a_list,
        block_size_m=block_size_m,
    )

    num_blocks = lookup_table.shape[0]
    for i in range(num_blocks):
        valid_size = lookup_table[i, 1]
        s_offset_m = lookup_table[i, 2]

        # For dy
        start_m = i * block_size_m
        curr_dy = dy[start_m : start_m + valid_size, :]
        curr_dyw = curr_dy @ w

        # For ds and a
        curr_ds = ds_list[i]
        curr_a = a_list[i]
        start_m_s = (i - s_offset_m) * block_size_m
        curr_dsa = curr_ds[start_m_s : start_m_s + valid_size, :] @ curr_a
        curr_dropout_mask = dropout_mask[start_m : start_m + valid_size, :]
        curr_dsa = torch.where(
            curr_dropout_mask, curr_dsa / (1 - dropout_p_list[i]), 0.0
        )
        curr_dyw += curr_dsa

        # Store the result
        out[start_m : start_m + valid_size, :] = curr_dyw

    return out


def fused_multi_lora_dyw_dsa_kernel_get_configs() -> list[triton.Config]:
    """Get the configurations for the fused LoRA dyw + dsa kernel."""
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
    configs=fused_multi_lora_dyw_dsa_kernel_get_configs(),
    key=["M", "N", "K", "OUTPUT_DTYPE"],
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
    )
    if compiled_kernel is not None and compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return dx


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

        # Prepare tensors
        inputs = prepare_func(
            seq_len_list=seq_len_list,
            adapter_idx=adapter_idx,
            adapter_info=adapter_info,
            n=n,
            k=k,
            block_size_m=block_size_m,
            dtype=dtype,
        )

        # # Run triton kernel
        triton_output = fused_multi_lora_dyw_dsa(init_zeros=True, **inputs)

        # Compute reference result
        ref_output = torch_multi_lora_dyw_dsa_ref(**inputs)
        ref_output_2 = torch_multi_lora_dyw_dsa_ref_2(**inputs)
        assert_verbose_allclose_two_rounds(ref_output, ref_output_2, atol=5e-3)

        # # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(triton_output, ref_output, atol=5e-3)
            logger.success(
                f"Verification passed for seq_len_list={seq_len_list} and "
                f"adapter_idx={adapter_idx}"
            )
        except AssertionError as e:
            logger.error(
                f"Verification failed for seq_len_list={seq_len_list} and "
                f"adapter_idx={adapter_idx}"
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
        seq_len_list_choices,
        adapter_idx_choices,
        adapter_info,
        n,
        k,
        block_size_m,
        dtype,
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
            f"Benchmarking fused_multi_lora_dyw_dsa with seq_len_list={seq_len_list} "
            f"and adapter_idx={adapter_idx}"
        )
        curr_prepare_func = partial(
            prepare_func,
            seq_len_list=seq_len_list,
            adapter_idx=adapter_idx,
            adapter_info=adapter_info,
            n=n,
            k=k,
            block_size_m=block_size_m,
            dtype=dtype,
        )

        benchmark(
            fused_multi_lora_dyw_dsa,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_multi_lora_dyw_dsa_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}",
        )

        benchmark(
            torch_dyw_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dyw_ref_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}",
        )

        benchmark(
            torch_multi_lora_dyw_dsa_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_multi_lora_dyw_dsa_ref_seq_len_list_{seq_len_list}_adapter_idx_{adapter_idx}",
        )
