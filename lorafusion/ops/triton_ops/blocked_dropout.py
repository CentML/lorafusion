# ruff: noqa: ANN001, N803, N806, E731, B023
"""Triton Blocked Seeded Dropout."""

import random

import torch
import triton
import triton.language as tl

from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype

BLOCK_SIZE = 1024


@triton.jit
def _blocked_seeded_dropout(
    x_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    block_to_dropout_p,
    num_pids_in_each_block_m,
    seed,
    BLOCK_SIZE: tl.constexpr,
    STORE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
) -> None:
    """Triton Dropout kernel."""
    pid = tl.program_id(axis=0)

    dropout_p_block_idx = pid // num_pids_in_each_block_m
    dropout_p = tl.load(block_to_dropout_p + dropout_p_block_idx)

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
    keep0, keep1, keep2, keep3 = (
        r0 > dropout_p,
        r1 > dropout_p,
        r2 > dropout_p,
        r3 > dropout_p,
    )

    if STORE_MASK:
        tl.store(mask_ptr + off0, keep0, mask=mask0)
        tl.store(mask_ptr + off1, keep1, mask=mask1)
        tl.store(mask_ptr + off2, keep2, mask=mask2)
        tl.store(mask_ptr + off3, keep3, mask=mask3)

    dropout_scaling = tl.cast(1.0 / (1 - dropout_p), DTYPE)

    o0 = tl.where(keep0, x0 * dropout_scaling, 0.0).to(DTYPE)
    o1 = tl.where(keep1, x1 * dropout_scaling, 0.0).to(DTYPE)
    o2 = tl.where(keep2, x2 * dropout_scaling, 0.0).to(DTYPE)
    o3 = tl.where(keep3, x3 * dropout_scaling, 0.0).to(DTYPE)

    tl.store(output_ptr + off0, o0, mask=mask0)
    tl.store(output_ptr + off1, o1, mask=mask1)
    tl.store(output_ptr + off2, o2, mask=mask2)
    tl.store(output_ptr + off3, o3, mask=mask3)


def blocked_seeded_dropout(
    x: torch.Tensor,
    block_to_dropout_p: torch.Tensor,
    block_size_m: int,
    seed: int | None = None,
    *,
    store_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Blocked Seeded Dropout.

    Assume BLOCK_SIZE for the kernel is 1024.

    Inside the dropout kernel, each tile is processing [4 * 1024] elements
    and the block_size_m means that for x, each [block_size_m * k] elements
    use the corresponding dropout probability from block_to_dropout_p.

    Therefore, each block_size_m has [block_size_m * k / (4 * 1024)] tiles.

    So for tile with pid, the dropout probability is
    block_to_dropout_p[
        pid // (block_size_m * k / (4 * BLOCK_SIZE))
    ]
    """
    if seed is None:
        seed = random.randrange(int(1e6))  # noqa: S311

    if x.ndim != 2:  # noqa: PLR2004
        msg = "x must be a 2D tensor"
        raise ValueError(msg)

    m, k = x.shape

    if (block_size_m * k) % (4 * BLOCK_SIZE) != 0:
        msg = (
            f"block_size_m * k = {block_size_m * k} must be divisible by "
            f"4 * BLOCK_SIZE = {4 * BLOCK_SIZE}"
        )
        raise ValueError(msg)

    num_pids_in_each_block_m = (block_size_m * k) // (4 * BLOCK_SIZE)

    output = torch.empty_like(x)
    mask = torch.empty_like(x, dtype=torch.bool) if store_mask else None
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"] * 4),)
    _blocked_seeded_dropout[grid](
        x,
        output,
        mask,
        n_elements,
        block_to_dropout_p,
        num_pids_in_each_block_m,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
        STORE_MASK=store_mask,
        DTYPE=torch_dtype_to_triton_dtype(x.dtype),
    )
    return output, mask
