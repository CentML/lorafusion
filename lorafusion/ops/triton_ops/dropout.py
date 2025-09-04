# ruff: noqa: ANN001, N803, N806, E731, B023
"""Triton Dropout."""

import random

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds

BLOCK_SIZE = 1024


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
    STORE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
) -> None:
    """Triton Dropout kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4

    off0 = block_start + BLOCK_SIZE * 0 + tl.arange(0, BLOCK_SIZE)
    off1 = block_start + BLOCK_SIZE * 1 + tl.arange(0, BLOCK_SIZE)
    off2 = block_start + BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE)
    off3 = block_start + BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE)

    mask0 = off0 < n_elements
    mask1 = off1 < n_elements
    mask2 = off2 < n_elements
    mask3 = off3 < n_elements

    x0 = tl.load(x_ptr + off0, mask=mask0)
    x1 = tl.load(x_ptr + off1, mask=mask1)
    x2 = tl.load(x_ptr + off2, mask=mask2)
    x3 = tl.load(x_ptr + off3, mask=mask3)

    r0, r1, r2, r3 = tl.random.rand4x(seed, off0, n_rounds=7)
    keep0, keep1, keep2, keep3 = r0 > p, r1 > p, r2 > p, r3 > p

    if STORE_MASK:
        tl.store(mask_ptr + off0, keep0, mask=mask0)
        tl.store(mask_ptr + off1, keep1, mask=mask1)
        tl.store(mask_ptr + off2, keep2, mask=mask2)
        tl.store(mask_ptr + off3, keep3, mask=mask3)

    dropout_scaling = tl.cast(1.0 / (1 - p), DTYPE)

    o0 = tl.where(keep0, x0 * dropout_scaling, 0.0).to(DTYPE)
    o1 = tl.where(keep1, x1 * dropout_scaling, 0.0).to(DTYPE)
    o2 = tl.where(keep2, x2 * dropout_scaling, 0.0).to(DTYPE)
    o3 = tl.where(keep3, x3 * dropout_scaling, 0.0).to(DTYPE)

    tl.store(output_ptr + off0, o0, mask=mask0)
    tl.store(output_ptr + off1, o1, mask=mask1)
    tl.store(output_ptr + off2, o2, mask=mask2)
    tl.store(output_ptr + off3, o3, mask=mask3)


def seeded_dropout(
    x: torch.Tensor, p: float, seed: int | None = None, *, store_mask: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Seeded Dropout."""
    if seed is None:
        seed = random.randrange(int(1e6))  # noqa: S311

    output = torch.empty_like(x)
    mask = torch.empty_like(x, dtype=torch.bool) if store_mask else None
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"] * 4),)
    _seeded_dropout[grid](
        x,
        output,
        mask,
        n_elements,
        p,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
        STORE_MASK=store_mask,
        DTYPE=torch_dtype_to_triton_dtype(x.dtype),
    )
    return output, mask


def verify_dropout_correctness(
    m_values: list[int],
    k: int,
    dropout_p: float = 0.1,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the dropout kernel produces correct results."""
    for m in m_values:
        logger.info(f"Verifying dropout kernel correctness for m={m}...")

        # Create input tensor
        x = torch.randn(m, k, device="cuda", dtype=dtype) / 10

        # Run triton kernel
        triton_output, mask = seeded_dropout(
            x=x,
            p=dropout_p,
            seed=seed,
            store_mask=True,
        )

        # Calculate reference
        dropout_p_scaling = torch.tensor(1.0 / (1 - dropout_p), dtype=dtype)
        ref_output = torch.where(mask, x * dropout_p_scaling, 0.0)

        # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(ref_output, triton_output)
        except AssertionError as e:
            logger.error(f"Verification failed for m={m}")
            logger.error(e)
        else:
            logger.success(f"Verification passed for m={m}")


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # Test parameters
    m_choices = [2048, 3072, 4096, 6144, 8192]
    k = 6144
    dropout_p = 0.1
    dtype = torch.bfloat16
    store_mask = True
    seed = 42

    # Verify kernel correctness
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_dropout_correctness(m_choices, k, dropout_p, seed, dtype)

    # Run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(
            f"Benchmarking dropout with m={m}, k={k}. "
            f"Dropout p={dropout_p}, dtype={dtype}"
        )

        # Create input tensor
        x = torch.randn(m, k, device="cuda", dtype=dtype) / 10

        # Benchmark triton dropout
        triton_time = benchmark(
            lambda x, p, seed, store_mask: seeded_dropout(
                x, p, seed, store_mask=store_mask
            ),
            prepare_func=lambda: {
                "x": x,
                "p": dropout_p,
                "seed": seed,
                "store_mask": store_mask,
            },
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"triton_dropout_m_{m}",
        )

        # Benchmark torch dropout
        torch_time = benchmark(
            lambda x, dp: torch.nn.functional.dropout(x, p=dp, training=True),
            prepare_func=lambda: {"x": x, "dp": dropout_p},
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dropout_m_{m}",
        )

        # Report results
        if triton_time > torch_time:
            logger.error(
                f"Slower for dropout [m={m}, k={k}]: {triton_time / torch_time:.2f}x"
            )
        else:
            logger.success(
                f"Faster for dropout [m={m}, k={k}]: {torch_time / triton_time:.2f}x"
            )
