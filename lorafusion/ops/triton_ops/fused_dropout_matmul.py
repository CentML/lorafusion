# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused Dropout Matmul."""

from functools import partial

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.dropout import seeded_dropout
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.common import get_device_short_name
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds

device_name = get_device_short_name()
USE_CUSTOM_KERNEL_MAP_HEURISTIC = True
DISABLE_CUSTOM_KERNEL_MAP = {
    # the shapes are inclusive of the boundaries.
    "h100-80gb-hbm3": ((0, 3072), (4225, 6143))
}
if device_name not in DISABLE_CUSTOM_KERNEL_MAP:
    USE_CUSTOM_KERNEL_MAP_HEURISTIC = False
else:
    CURR_DEVICE_DISABLE_CUSTOM_KERNEL_MAP = DISABLE_CUSTOM_KERNEL_MAP[device_name]


def prepare_func(
    m: int,
    k: int,
    r: int,
    dropout_p: float,
    seed: int = 42,
    *,
    store_mask: bool = False,
    store_masked_scaled_x: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Prepare the input tensors for the fused dropout matmul."""
    x = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    a = torch.randn(r, k, device="cuda", dtype=dtype) / 10
    return {
        "x": x,
        "a": a,
        "dropout_p": dropout_p,
        "seed": seed,
        "store_mask": store_mask,
        "store_masked_scaled_x": store_masked_scaled_x,
    }


def torch_xa_ref(x: torch.Tensor, a: torch.Tensor, **kwargs) -> torch.Tensor:
    """Torch reference implementation of the fused dropout matmul."""
    return x @ a.T


def torch_dropout_ref(
    x: torch.Tensor, a: torch.Tensor, dropout_p: float, **kwargs
) -> torch.Tensor:
    """Torch reference implementation of the dropout."""
    return torch.dropout(x, p=dropout_p, train=True)


def torch_dropout_matmul_ref(
    x: torch.Tensor,
    a: torch.Tensor,
    dropout_p: float,
    dropout_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused dropout matmul.

    Args:
        x: The input tensor.
        a: The input tensor.
        dropout_p: The dropout probability.
        dropout_mask: The dropout mask. The dropout is used for verification.
        **kwargs: Additional keyword arguments.
    """
    if dropout_mask is None:
        mask_scaled_x = torch.dropout(x, p=dropout_p, train=True)
    else:
        mask_scaled_x = torch.where(dropout_mask, x / (1 - dropout_p), 0.0)
    return mask_scaled_x @ a.T


def triton_separate_kernel_dropout_matmul(
    x: torch.Tensor,
    a: torch.Tensor,
    dropout_p: float,
    seed: int = 42,
    *,
    store_mask: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Triton separate kernel dropout matmul."""
    masked_scaled_x, dropout_mask = seeded_dropout(
        x, dropout_p, seed, store_mask=store_mask
    )
    return masked_scaled_x @ a.T


def fused_dropout_matmul_kernel_get_configs() -> list[triton.Config]:
    """Get the configurations for the fused dropout matmul kernel."""
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
        # for BM in [32, 64]
        # for BN in [16, 32, 64]
        # for BK in [32, 64]
        # for s in ([5, 6])
        # for w in [8]
        # if BM * BK < 256 * 256
        for BM in [32]
        for BN in [16]
        for BK in [64]
        for s in ([6])
        for w in [8]
    ]


@triton.jit
def rand4x_kernel(
    seed,
    dropout_offsets_1_in_4,
    SIZE_M: tl.constexpr,
    SIZE_K: tl.constexpr,
) -> tl.tensor:
    """Generate random numbers efficiently with rand4x."""
    r0, r1, r2, r3 = tl.random.rand4x(seed, dropout_offsets_1_in_4, n_rounds=7)
    r_01 = tl.join(r0, r1)
    r_23 = tl.join(r2, r3)
    r = tl.join(r_01, r_23)
    return tl.reshape(r, (SIZE_M, SIZE_K))


@triton.autotune(
    configs=fused_dropout_matmul_kernel_get_configs(),
    key=["M", "N", "K", "OUTPUT_DTYPE"],
)
@triton.jit
def fused_dropout_matmul_kernel(  # noqa: PLR0915
    x_ptr,
    a_ptr,
    out_ptr,
    mask_ptr,
    masked_scaled_x_ptr,
    dropout_p,
    seed,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_ak: tl.constexpr,
    stride_an: tl.constexpr,
    stride_cm,
    stride_cn,
    stride_mask_m,
    stride_mask_k,
    stride_masked_scaled_x_m,
    stride_masked_scaled_x_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    STORE_MASK: tl.constexpr,
    STORE_MASKED_SCALED_X: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused Dropout Matmul kernel.

    Compute Out = Dropout(X) @ A

    Dimensions:
      X: [M, K]
      A: [K, N]
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
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create masks for boundary checking
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Compute pointers with proper offsets
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    a_ptrs = a_ptr + (offs_k[:, None] * stride_ak + offs_n[None, :] * stride_an)
    SUB_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // 4
    dropout_offsets = offs_m[:, None] * K + tl.arange(0, SUB_BLOCK_SIZE_K)[None, :]
    if STORE_MASK:
        mask_ptrs = mask_ptr + (
            offs_m[:, None] * stride_mask_m + offs_k[None, :] * stride_mask_k
        )
    if STORE_MASKED_SCALED_X:
        masked_scaled_x_ptrs = masked_scaled_x_ptr + (
            offs_m[:, None] * stride_masked_scaled_x_m
            + offs_k[None, :] * stride_masked_scaled_x_k
        )

    # ------------------------------------------------------------
    #  3. Main loop
    # ------------------------------------------------------------
    dropout_scaling = tl.cast(1.0 / (1 - dropout_p), OUTPUT_DTYPE)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Create combined masks for loading
        x_mask = m_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a_mask = n_mask[None, :] & (offs_k[:, None] < K - k * BLOCK_SIZE_K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Dropout over x
        mask = (
            rand4x_kernel(seed, dropout_offsets, BLOCK_SIZE_M, BLOCK_SIZE_K) > dropout_p
        )
        # Apply m_mask to ensure we don't store out of bounds
        mask = mask & x_mask

        x = tl.where(mask, x * dropout_scaling, 0.0)
        if STORE_MASK:
            tl.store(mask_ptrs, mask, mask=x_mask)
        if STORE_MASKED_SCALED_X:
            tl.store(masked_scaled_x_ptrs, x, mask=x_mask)

        # Accumulate
        accumulator = tl.dot(x, a, accumulator)

        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        a_ptrs += BLOCK_SIZE_K * stride_ak
        dropout_offsets += BLOCK_SIZE_K
        if STORE_MASK:
            mask_ptrs += BLOCK_SIZE_K * stride_mask_k
        if STORE_MASKED_SCALED_X:
            masked_scaled_x_ptrs += BLOCK_SIZE_K * stride_masked_scaled_x_k

    accumulator = accumulator.to(OUTPUT_DTYPE)

    # ------------------------------------------------------------
    #  4. Store the output
    # ------------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def fused_dropout_matmul(
    x: torch.Tensor,
    a: torch.Tensor,
    dropout_p: float,
    seed: int = 42,
    *,
    store_mask: bool = False,
    store_masked_scaled_x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Triton Fused Dropout Matmul."""
    # Check constraints.
    if x.shape[1] != a.shape[1]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {a.shape[1]}. "
            f"x: {x.shape}, a: {a.shape}"
        )
        raise ValueError(msg)
    if x.dtype != a.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {a.dtype}"
        raise ValueError(msg)

    # Transpose a to match the kernel's dimensions.
    a = a.T

    M, K = x.shape
    K, N = a.shape
    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    mask = torch.empty((M, K), device=x.device, dtype=torch.bool)
    masked_scaled_x = torch.empty((M, K), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    compiled_kernel = fused_dropout_matmul_kernel[grid](
        x,
        a,
        out,
        mask,
        masked_scaled_x,
        dropout_p,
        seed,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        out.stride(0),
        out.stride(1),
        mask.stride(0),
        mask.stride(1),
        masked_scaled_x.stride(0),
        masked_scaled_x.stride(1),
        STORE_MASK=store_mask,
        STORE_MASKED_SCALED_X=store_masked_scaled_x,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(out.dtype),
    )
    if compiled_kernel is not None and compiled_kernel.n_spills > 0:
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return (
        out,
        mask if store_mask else None,
        masked_scaled_x if store_masked_scaled_x else None,
    )


def verify_kernel_correctness(
    m_values: list[int],
    k: int,
    r: int,
    dropout_p: float = 0.1,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions.

    This is particularly important when M is not divisible by the block sizes.
    """
    for m in m_values:
        logger.info(f"Verifying kernel correctness for m={m}...")

        # Prepare tensors
        inputs = prepare_func(
            m=m,
            k=k,
            r=r,
            dropout_p=dropout_p,
            seed=seed,
            dtype=dtype,
            store_mask=True,
            store_masked_scaled_x=True,
        )

        # Run triton kernel
        triton_output, mask, masked_scaled_x = fused_dropout_matmul(**inputs)

        # Recalculate
        dropout_p_scaling = torch.tensor(1.0 / (1 - dropout_p), dtype=dtype)
        ref_masked_scaled_x = torch.where(mask, inputs["x"] * dropout_p_scaling, 0.0)
        ref_output = ref_masked_scaled_x @ inputs["a"].T

        # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(ref_masked_scaled_x, masked_scaled_x)
            assert_verbose_allclose_two_rounds(ref_output, triton_output, atol=5e-3)
        except AssertionError as e:
            logger.error(f"Verification failed for m={m}")
            logger.error(e)
        else:
            logger.success(f"Verification passed for m={m}")


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # 3072 -> 4224 -> 6144 -> 8576

    m_choices = [
        2048,
        2560,
        3072,
        4096,
        4224,
        4225,
        5555,
        6144,
        6332,
        7168,
        7169,
        8192,
        8195,
        8576,
    ]
    k = 6144
    r = 16
    dropout_p = 0.1
    dtype = torch.bfloat16
    store_mask = True
    store_masked_scaled_x = True

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(m_choices, k, r, dropout_p, 42, dtype)

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking fused_dropout_matmul with m={m}, k={k}, r={r}. "
            f"Dropout p={dropout_p}, dtype={dtype}"
        )
        curr_prepare_func = partial(
            prepare_func,
            m=m,
            k=k,
            r=r,
            dropout_p=dropout_p,
            dtype=dtype,
            store_mask=store_mask,
            store_masked_scaled_x=store_masked_scaled_x,
        )

        fused_dropout_matmul_time = benchmark(
            fused_dropout_matmul,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_dropout_matmul_m_{m}",
        )

        benchmark(
            torch_dropout_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dropout_ref_m_{m}",
        )

        benchmark(
            torch_xa_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_xa_ref_m_{m}",
        )

        torch_dropout_matmul_time = benchmark(
            torch_dropout_matmul_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dropout_matmul_ref_m_{m}",
        )

        triton_separate_kernel_dropout_matmul_time = benchmark(
            triton_separate_kernel_dropout_matmul,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"triton_separate_kernel_dropout_matmul_m_{m}",
        )

        if fused_dropout_matmul_time > torch_dropout_matmul_time:
            logger.error(
                f"Slower for fused dropout matmul [m={m}, k={k}]: "
                f"{fused_dropout_matmul_time / torch_dropout_matmul_time:.2f}x"
            )
        else:
            logger.success(
                f"Faster for fused dropout matmul [k={k}, r={r}, m={m}]: "
                f"{fused_dropout_matmul_time / torch_dropout_matmul_time:.2f}x"
            )
