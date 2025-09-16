# ruff: noqa: ANN001, N803, N806, E731, PLR0915, PIE808, PLR2004
"""Triton Fused LoRA implementation for XW + (XA)B computation."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.config import KERNEL_SPILL_VERBOSE
from lorafusion.ops.triton_ops.fused_dropout_matmul import rand4x_kernel
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number


def prepare_func(
    m: int,
    n: int,
    k: int,
    r: int,
    dropout_p: float = 0.0,
    seed: int = 42,
    store_mask: bool = False,
    store_masked_scaled_x: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA computation.

    Args:
        m: First dimension of input tensor
        n: Second dimension of weight tensor
        k: Common dimension for input and weight tensors
        r: LoRA rank
        dropout_p: Dropout probability
        seed: Random seed for dropout
        store_mask: Whether to store dropout mask
        store_masked_scaled_x: Whether to store masked and scaled input
        dtype: Data type of the tensors

    Returns:
        Dictionary containing the input tensors
    """
    x = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    w = torch.randn(k, n, device="cuda", dtype=dtype) / 10
    a = torch.randn(k, r, device="cuda", dtype=dtype) / 10
    b = torch.randn(r, n, device="cuda", dtype=dtype) / 10

    return {
        "x": x,
        "w": w,
        "a": a,
        "b": b,
        "dropout_p": dropout_p,
        "seed": seed,
        "store_mask": store_mask,
        "store_masked_scaled_x": store_masked_scaled_x,
    }


def torch_func(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dropout_p: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA computation.

    Computes XW + (XA)B with optional dropout

    Args:
        x: Input tensor of shape [M, K]
        w: Weight tensor of shape [K, N]
        a: LoRA A matrix of shape [K, R]
        b: LoRA B matrix of shape [R, N]
        dropout_p: Dropout probability
        **kwargs: Additional arguments (not used)

    Returns:
        Output tensor of shape [M, N]
    """
    if dropout_p > 0.0:
        # Apply dropout to the input before LoRA computation
        x_dropped = torch.nn.functional.dropout(x, p=dropout_p, training=True)
        return x @ w + x_dropped @ a @ b
    return x @ w + x @ a @ b


def lora_matmul_get_configs() -> list[triton.Config]:
    """Get the configurations for the fused LoRA kernel."""
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
    configs=lora_matmul_get_configs(),
    key=["M", "N", "K", "OUTPUT_DTYPE"],
)
@triton.jit
def lora_matmul_kernel_v1_single_sync(
    x_ptr,
    w_ptr,
    a_ptr,
    b_ptr,
    s_ptr,
    out_ptr,
    mask_ptr,
    masked_scaled_x_ptr,
    dropout_p,
    seed,
    flag_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    F: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_ar: tl.constexpr,
    stride_br: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_sm,
    stride_sr,
    stride_om,
    stride_on,
    stride_fp: tl.constexpr,
    stride_mask_m,
    stride_mask_k: tl.constexpr,
    stride_masked_scaled_x_m,
    stride_masked_scaled_x_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    STORE_MASK: tl.constexpr,
    STORE_MASKED_SCALED_X: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton LoRA MatMul kernel.

    Compute Out = XW + (X A)B in one pass with optional dropout.

    Dimensions:
      X: [M, K]
      W: [K, N]
      A: [K, R]
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

    #  0 → unlocked
    #  1 → locked by master
    #  2 → partial_lora is stored, go ahead
    flag_old = tl.atomic_cas(flag_ptr + pid_m * stride_fp, 0, 1, sem="relaxed")
    is_master = flag_old == 0

    # ------------------------------------------------------------
    #  2. Compute the tile indices for M, N
    # ------------------------------------------------------------
    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_r = tl.arange(0, R)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)
    b_ptrs = b_ptr + (offs_r[:, None] * stride_br + offs_wn[None, :] * stride_bn)
    s_ptrs = s_ptr + (offs_xm[:, None] * stride_sm + offs_r[None, :] * stride_sr)

    # ------------------------------------------------------------
    # 3. Initialize accumulators
    # ------------------------------------------------------------
    accum_main = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------
    if is_master:
        partial_lora = tl.zeros((BLOCK_SIZE_M, R), dtype=tl.float32)

        # Path for lora matmul.
        BLOCK_SIZE_K_FOR_LORA: tl.constexpr = BLOCK_SIZE_K * 2
        offs_k_for_lora = tl.arange(0, BLOCK_SIZE_K_FOR_LORA)
        x_ptrs_for_lora = x_ptr + (
            offs_xm[:, None] * stride_xm + offs_k_for_lora[None, :] * stride_xk
        )
        a_ptrs_for_lora = a_ptr + (
            offs_k_for_lora[:, None] * stride_ak + offs_r[None, :] * stride_ar
        )
        # Dropout
        SUB_BLOCK_SIZE_K_FOR_LORA: tl.constexpr = BLOCK_SIZE_K_FOR_LORA // 4
        dropout_offsets = (
            offs_xm[:, None] * K + tl.arange(0, SUB_BLOCK_SIZE_K_FOR_LORA)[None, :]
        )
        if STORE_MASK:
            mask_ptrs = mask_ptr + (
                offs_xm[:, None] * stride_mask_m
                + offs_k_for_lora[None, :] * stride_mask_k
            )
        if STORE_MASKED_SCALED_X:
            masked_scaled_x_ptrs = masked_scaled_x_ptr + (
                offs_xm[:, None] * stride_masked_scaled_x_m
                + offs_k_for_lora[None, :] * stride_masked_scaled_x_k
            )
        dropout_scaling = tl.cast(1.0 / (1 - dropout_p), OUTPUT_DTYPE)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K_FOR_LORA)):
            x = tl.load(
                x_ptrs_for_lora,
                mask=offs_k_for_lora[None, :] < K - k * BLOCK_SIZE_K_FOR_LORA,
                other=0.0,
            )
            a = tl.load(
                a_ptrs_for_lora,
                mask=offs_k_for_lora[:, None] < K - k * BLOCK_SIZE_K_FOR_LORA,
                other=0.0,
            )

            # Apply dropout if dropout_p > 0
            if dropout_p > 0.0:
                mask = (
                    rand4x_kernel(
                        seed, dropout_offsets, BLOCK_SIZE_M, BLOCK_SIZE_K_FOR_LORA
                    )
                    > dropout_p
                )
                x = tl.where(mask, x * dropout_scaling, 0.0)
                if STORE_MASK:
                    tl.store(mask_ptrs, mask)
                if STORE_MASKED_SCALED_X:
                    tl.store(masked_scaled_x_ptrs, x)

            partial_lora = tl.dot(x, a, partial_lora)

            # Advance the ptrs to the next K block.
            x_ptrs_for_lora += BLOCK_SIZE_K_FOR_LORA * stride_xk
            a_ptrs_for_lora += BLOCK_SIZE_K_FOR_LORA * stride_ak
            if dropout_p > 0.0:
                dropout_offsets += BLOCK_SIZE_K_FOR_LORA
                if STORE_MASK:
                    mask_ptrs += BLOCK_SIZE_K_FOR_LORA * stride_mask_k
                if STORE_MASKED_SCALED_X:
                    masked_scaled_x_ptrs += (
                        BLOCK_SIZE_K_FOR_LORA * stride_masked_scaled_x_k
                    )

        partial_lora = partial_lora.to(tl.bfloat16)
        tl.store(s_ptrs, partial_lora)

        # Control flow for other non-master thread-blocks.
        tl.atomic_add(flag_ptr + pid_m * stride_fp, 1, sem="release")

        # Path for main matmul.
        x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accum_main = tl.dot(x, w, accum_main)

            # Advance the ptrs to the next K block.
            x_ptrs += BLOCK_SIZE_K * stride_xk
            w_ptrs += BLOCK_SIZE_K * stride_wk

        b = tl.load(b_ptrs)

    else:
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accum_main = tl.dot(x, w, accum_main)

            # Advance the ptrs to the next K block.
            x_ptrs += BLOCK_SIZE_K * stride_xk
            w_ptrs += BLOCK_SIZE_K * stride_wk

        b = tl.load(b_ptrs)

        # wait for partial_lora to be written (state == 2)
        while tl.atomic_cas(flag_ptr + pid_m * stride_fp, 2, 2, sem="acquire") != 2:
            pass
        partial_lora = tl.load(s_ptrs)

    # ------------------------------------------------------------
    # 5. Compute the final output
    # ------------------------------------------------------------
    accum_main = tl.dot(partial_lora, b, accum_main)
    accum_main = accum_main.to(OUTPUT_DTYPE)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, accum_main, mask=out_mask)


def lora_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dropout_p: float = 0.0,
    seed: int = 42,
    store_mask: bool = False,
    store_masked_scaled_x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Triton LoRA MatMul.

    Compute XW + (XA)B using a fused kernel for better performance, with optional dropout.

    Args:
        x: Input tensor of shape [M, K]
        w: Weight tensor of shape [K, N]
        a: LoRA A matrix of shape [K, R]
        b: LoRA B matrix of shape [R, N]
        dropout_p: Dropout probability (default: 0.0)
        seed: Random seed for dropout (default: 42)
        store_mask: Whether to store dropout mask (default: False)
        store_masked_scaled_x: Whether to store masked and scaled input (default: False)

    Returns:
        Tuple containing:
            - Output tensor of shape [M, N]
            - Dropout mask if store_mask is True, otherwise None
            - Masked and scaled input if store_masked_scaled_x is True, otherwise None

    Raises:
        ValueError: If tensor dimensions or dtypes are incompatible
    """
    # Check constraints.
    if x.shape[1] != w.shape[0]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {w.shape[0]}. "
            f"x: {x.shape}, w: {w.shape}"
        )
        raise ValueError(msg)
    if a.shape[1] != b.shape[0]:
        msg = (
            f"Incompatible dimensions: {a.shape[1]} != {b.shape[0]}. "
            f"a: {a.shape}, b: {b.shape}"
        )
        raise ValueError(msg)
    if x.dtype != w.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {w.dtype}"
        raise ValueError(msg)
    if x.dtype != a.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {a.dtype}"
        raise ValueError(msg)
    if x.dtype != b.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {b.dtype}"
        raise ValueError(msg)
    if not x.is_contiguous():
        msg = "Matrix X must be contiguous"
        raise ValueError(msg)

    M, K = x.shape
    K, N = w.shape
    R = a.shape[1]
    F = 256

    # Allocates output.
    s = torch.empty((M, R), device=a.device, dtype=x.dtype)
    out = torch.empty((M, N), device=a.device, dtype=x.dtype)
    flag = torch.zeros((F,), device=a.device, dtype=torch.int32)

    # Dropout-related tensors
    mask = (
        torch.empty((M, K), device=x.device, dtype=torch.bool) if store_mask else None
    )
    masked_scaled_x = (
        torch.empty((M, K), device=x.device, dtype=x.dtype)
        if store_masked_scaled_x
        else None
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    compiled_kernel = lora_matmul_kernel_v1_single_sync[grid](
        x,
        w,
        a,
        b,
        s,
        out,
        mask if mask is not None else x,  # Use x as a dummy if mask is None
        masked_scaled_x
        if masked_scaled_x is not None
        else x,  # Use x as a dummy if masked_scaled_x is None
        dropout_p,
        seed,
        flag,
        M,
        N,
        K,
        R,
        F,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        s.stride(0),
        s.stride(1),
        out.stride(0),
        out.stride(1),
        flag.stride(0),
        mask.stride(0) if mask is not None else 0,
        mask.stride(1) if mask is not None else 0,
        masked_scaled_x.stride(0) if masked_scaled_x is not None else 0,
        masked_scaled_x.stride(1) if masked_scaled_x is not None else 0,
        STORE_MASK=store_mask,
        STORE_MASKED_SCALED_X=store_masked_scaled_x,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(out.dtype),
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

    return (out, mask, masked_scaled_x)


if __name__ == "__main__":
    set_warmup_and_number(1000, 1000)

    # Test parameters
    m_choices = [4096]  # bsz * seq
    n = 4096  # hidden_dim
    k = 4096  # hidden_dim
    r = 16  # lora_dim
    dropout_p = 0.1  # dropout probability
    dtype = torch.bfloat16
    store_mask = False
    store_masked_scaled_x = False
    profile = False

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking lora_matmul with m={m}, n={n}, k={k}, r={r}, dropout_p={dropout_p}"
        )

        curr_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            dropout_p=dropout_p,
            dtype=dtype,
            store_mask=store_mask,
            store_masked_scaled_x=store_masked_scaled_x,
        )

        # Validate the correctness of the kernel.
        inputs = curr_prepare_func()
        out, _, _ = lora_matmul(**inputs)
        torch_out = torch_func(**inputs)
        try:
            torch.testing.assert_close(out, torch_out, rtol=6e-2, atol=6e-2)
            logger.info("Correctness check passed")
        except AssertionError as e:
            logger.error(f"Correctness check failed:\n{e}")

        # Run benchmarks
        benchmark(
            torch_func,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            profile=profile,
            output_dir=f"./profiling-results/v1/torch_func_m_{m}_dropout_{dropout_p}",
            msg=f"torch_func_m_{m}_dropout_{dropout_p}",
        )

        # Define a wrapper function to match the benchmark API
        def lora_matmul_wrapper(**kwargs):
            result, _, _ = lora_matmul(**kwargs)
            return result

        benchmark(
            lora_matmul_wrapper,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            profile=profile,
            output_dir=f"./profiling-results/v1/lora_matmul_m_{m}_dropout_{dropout_p}",
            msg=f"lora_matmul_m_{m}_dropout_{dropout_p}",
        )

        # Also benchmark without dropout for comparison
        no_dropout_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            dropout_p=0.0,
            dtype=dtype,
        )

        benchmark(
            torch_func,
            prepare_func=no_dropout_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            profile=profile,
            output_dir=f"./profiling-results/v1/torch_func_m_{m}_no_dropout",
            msg=f"torch_func_m_{m}_no_dropout",
        )

        benchmark(
            lora_matmul_wrapper,
            prepare_func=no_dropout_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            profile=profile,
            output_dir=f"./profiling-results/v1/lora_matmul_m_{m}_no_dropout",
            msg=f"lora_matmul_m_{m}_no_dropout",
        )
