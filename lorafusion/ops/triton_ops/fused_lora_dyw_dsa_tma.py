# ruff: noqa: ANN001, N803, N806, E731
"""Triton Fused LoRA dy @ w + ds @ a * dropout_scale using TMA."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.tma_utils import (
    TmaAutoTuneHelper,
    _compute_pid,
    tl_experimental_descriptor_load,
    tl_experimental_descriptor_store,
)
from lorafusion.ops.triton_ops.utils import torch_dtype_to_triton_dtype
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.testing import assert_verbose_allclose_two_rounds


def prepare_func(
    m: int,
    n: int,
    k: int,
    r: int,
    dropout_p: float,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA dyw + dsa kernel."""
    dy = torch.randn(m, n, device="cuda", dtype=dtype) / 10
    w = torch.randn(n, k, device="cuda", dtype=dtype) / 10
    ds = torch.randn(m, r, device="cuda", dtype=dtype) / 10
    a = torch.randn(r, k, device="cuda", dtype=dtype) / 10
    dropout_mask = torch.randn(m, k, device="cuda", dtype=torch.float32) > dropout_p

    return {
        "dy": dy,
        "w": w,
        "ds": ds,
        "a": a,
        "dropout_p": dropout_p,
        "dropout_mask": dropout_mask,
    }


def torch_dyw_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of dy @ w."""
    return dy @ w


def torch_lora_dyw_dsa_ref(
    dy: torch.Tensor,
    w: torch.Tensor,
    ds: torch.Tensor,
    a: torch.Tensor,
    dropout_p: float,
    dropout_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA dyw + dsa."""
    return dy @ w + torch.where(dropout_mask, ds @ a / (1 - dropout_p), 0.0)


def fused_lora_dyw_dsa_tma_kernel_get_configs() -> list[triton.Config]:
    """Get the configurations for the fused LoRA dyw + dsa kernel using TMA.

    Note:
        - Best config in H100 for 4096x4096:
            - BM = 128, BN = 256, BK = 64, s = 3, w = 8,
            - SUBTILE = False, LUF = None, FLAT = False
        - While for plain matmul, FLAT should be set to True.
            we find that current implementation of Flatten makes the performance
            quite bad for this kernel.
    """
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE,
                "LOOP_UNROLL_FACTOR": LUF,
                "FLATTEN": FLAT,
            },
            num_stages=s,
            num_warps=w,
        )
        # # For tuning:
        # for BM in [128]
        # for BN in [128, 256]
        # for BK in [32, 64, 128]
        # for s in ([3, 4, 5, 6])
        # for w in [4, 8]
        # for SUBTILE in [True, False]
        # for LUF in [None, 1, 2, 8]
        # for FLAT in [True, False]
        # # Pre-tuned config:
        for BM in [128]
        for BN in [256]
        for BK in [64]
        for s in ([3])
        for w in [8]
        for SUBTILE in [False]
        for LUF in [None]
        for FLAT in [False]
        if BM * BK < 256 * 256
    ]


@triton.autotune(
    configs=fused_lora_dyw_dsa_tma_kernel_get_configs(),
    key=["M", "N", "K", "OUTPUT_DTYPE"],
)
@triton.jit
def fused_lora_dyw_dsa_tma_kernel(
    dy_desc_ptr,
    w_desc_ptr,
    ds_ptr,
    a_ptr,
    dropout_mask_ptr,
    dx_desc_ptr,
    dropout_p,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    stride_dsm,
    stride_dsr,
    stride_an: tl.constexpr,
    stride_ar: tl.constexpr,
    stride_mask_m,
    stride_mask_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr,
    FLATTEN: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused LoRA dyw + dsa kernel using TMA.

    Compute dx = dy @ w + where(dropout_mask, ds @ a / (1 - dropout_p), 0.0)

    Dimensions:
      dy: [M, K]
      w: [N, K] (transposed from [K, N])
      ds: [M, R]
      a: [N, R] (transposed from [R, N])
      dropout_mask: [M, N]
      dx: [M, N]
    """
    # ------------------------------------------------------------
    #  1. Program ID / block mapping
    # ------------------------------------------------------------
    dtype = OUTPUT_DTYPE
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # ------------------------------------------------------------

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        # ------------------------------------------------------------
        # 2. Compute the tile indices for M, N
        # ------------------------------------------------------------
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_m = pid_m * BLOCK_SIZE_M
        offs_n = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # ------------------------------------------------------------
        # 3. Compute the LoRA part with dropout
        # ------------------------------------------------------------
        v_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        v_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        v_offs_r = tl.arange(0, R)

        ds_ptrs = ds_ptr + (
            v_offs_m[:, None] * stride_dsm + v_offs_r[None, :] * stride_dsr
        )
        a_ptrs = a_ptr + (v_offs_n[:, None] * stride_an + v_offs_r[None, :] * stride_ar)
        ds = tl.load(ds_ptrs)
        a = tl.load(a_ptrs)

        accumulator = tl.dot(ds, a.T, accumulator)

        if ENABLE_DROPOUT:
            dropout_mask_ptrs = dropout_mask_ptr + (
                v_offs_m[:, None] * stride_mask_m + v_offs_n[None, :] * stride_mask_n
            )
            dropout_mask = tl.load(dropout_mask_ptrs)
            accumulator = tl.where(dropout_mask, accumulator / (1 - dropout_p), 0.0)

        # ------------------------------------------------------------
        # 4. Main loop for dy @ w.T
        # ------------------------------------------------------------
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            dy = tl_experimental_descriptor_load(
                dy_desc_ptr, [offs_m, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
            )
            w = tl_experimental_descriptor_load(
                w_desc_ptr, [offs_n, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
            )
            accumulator = tl.dot(dy, w.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_m_c = pid_m * BLOCK_SIZE_M
        offs_n_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into
        # multiple pieces. By subtiling we can reduce shared memory consumption by the
        # epilogue and instead use that memory to increase our stage count.
        # In this case we partition the accumulator into
        # 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors.
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            tl_experimental_descriptor_store(dx_desc_ptr, c0, [offs_m_c, offs_n_c])
            c1 = acc1.to(dtype)
            tl_experimental_descriptor_store(
                dx_desc_ptr, c1, [offs_m_c, offs_n_c + BLOCK_SIZE_N // 2]
            )
        else:
            accumulator = accumulator.to(dtype)
            tl_experimental_descriptor_store(
                dx_desc_ptr, accumulator, [offs_m_c, offs_n_c]
            )


# Initialize TMA descriptors.
desc_helper = TmaAutoTuneHelper()
desc_helper.init_tma_descriptor("dy")
desc_helper.init_tma_descriptor("w")
desc_helper.init_tma_descriptor("dx")

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

desc_dy = desc_helper.get_tma_descriptor_kernel_param("dy")
desc_w = desc_helper.get_tma_descriptor_kernel_param("w")
desc_dx = desc_helper.get_tma_descriptor_kernel_param("dx")


def fused_lora_dyw_dsa_tma(  # noqa: C901
    dy: torch.Tensor,
    w: torch.Tensor,
    ds: torch.Tensor,
    a: torch.Tensor,
    dropout_p: float,
    dropout_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton Fused LoRA dyw + dsa using TMA.

    Compute dx = dy @ w + where(dropout_mask, ds @ a / (1 - dropout_p), 0.0)

    Shape definition of the input tensors:
      dy: [M, N]
      w: [N, K]
      ds: [M, R]
      a: [R, N]
      mask: [M, K]
      dx: [M, K]
    """
    # Check constraints.
    m_y, n_y = dy.shape
    n_w, k_w = w.shape
    m_s, r_s = ds.shape
    r_a, k_a = a.shape
    if dropout_mask is not None:
        m_mask, k_mask = dropout_mask.shape
        if m_y != m_mask or k_w != k_mask:
            msg = (
                f"Incompatible dimensions for dropout_mask: {dropout_mask.shape}, "
                f"expected [{m_y}, {k_w}]"
            )
            raise ValueError(msg)

    if m_y != m_s:
        msg = (
            f"Incompatible dimensions for m of dy and ds: {m_y} != {m_s}. "
            f"{dy.shape=}, {w.shape=}, {ds.shape=}, {a.shape=}"
        )
        raise ValueError(msg)
    if n_w != n_y:
        msg = (
            f"Incompatible dimensions for n of w and a: {n_w} != {n_y}. "
            f"{dy.shape=}, {w.shape=}, {ds.shape=}, {a.shape=}"
        )
        raise ValueError(msg)
    if k_w != k_a:
        msg = (
            f"Incompatible dimensions for k of dy and w: {k_w} != {k_a}. "
            f"{dy.shape=}, {w.shape=}, {ds.shape=}, {a.shape=}"
        )
        raise ValueError(msg)
    if r_s != r_a:
        msg = (
            f"Incompatible dimensions for r of ds and a: {r_s} != {r_a}. "
            f"{dy.shape=}, {w.shape=}, {ds.shape=}, {a.shape=}"
        )
        raise ValueError(msg)
    if dy.dtype != w.dtype or dy.dtype != ds.dtype or dy.dtype != a.dtype:
        msg = (
            f"Incompatible dtypes: {dy.dtype} != {w.dtype} != {ds.dtype} != {a.dtype}. "
            f"{dy.shape=}, {w.shape=}, {ds.shape=}, {a.shape=}"
        )
        raise ValueError(msg)

    # Then we change the representation of tensors to map the low-level implementation.
    w = w.T
    a = a.T

    M = m_y
    N = k_w
    K = n_y
    R = r_s
    # Allocates output.
    dx = torch.empty((M, N), device=dy.device, dtype=dy.dtype)

    def grid(META: dict[str, Any]) -> int:
        desc_helper.fill_2d_tma_descriptor(
            "dy",
            dy.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            dy.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "w",
            w.data_ptr(),
            N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w.element_size(),
        )

        store_block_n = META["BLOCK_SIZE_N"]

        if META["EPILOGUE_SUBTILE"]:
            store_block_n = store_block_n // 2

        desc_helper.fill_2d_tma_descriptor(
            "dx",
            dx.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"],
            store_block_n,
            dx.element_size(),
        )

        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    compiled_kernel = fused_lora_dyw_dsa_tma_kernel[grid](
        desc_dy,
        desc_w,
        ds,
        a,
        dropout_mask if dropout_mask is not None else torch.empty(0, device=dy.device),
        desc_dx,
        dropout_p,
        M,
        N,
        K,
        R,
        ds.stride(0),
        ds.stride(1),
        a.stride(0),
        a.stride(1),
        dropout_mask.stride(0) if dropout_mask is not None else 0,
        dropout_mask.stride(1) if dropout_mask is not None else 0,
        NUM_SMS=NUM_SMS,
        ENABLE_DROPOUT=dropout_p != 0,
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
    m_values: list[int],
    n: int,
    k: int,
    r: int,
    dropout_p: float = 0.1,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions.

    This is particularly important when M is not divisible by block sizes.

    Args:
        m_values: List of M dimensions to test
        n: Size of N dimension
        k: Size of K dimension
        r: Size of R dimension
        dropout_p: Dropout probability
        dtype: Data type for tensors
    """
    for m in m_values:
        logger.info(f"Verifying kernel correctness for m={m}...")

        # Prepare tensors
        inputs = prepare_func(
            m=m,
            n=n,
            k=k,
            r=r,
            dropout_p=dropout_p,
            dtype=dtype,
        )

        # Run triton kernel
        dx_triton = fused_lora_dyw_dsa_tma(**inputs)

        # Compute reference result
        dx_ref = torch_lora_dyw_dsa_ref(**inputs)

        # Check shapes
        if dx_triton.shape != dx_ref.shape:
            logger.error(
                f"Shape mismatch for m={m}: {dx_triton.shape} vs {dx_ref.shape}"
            )
            continue

        # Check for NaN values
        if torch.isnan(dx_triton).any():
            logger.error(f"NaN values detected in output for m={m}")
            continue

        # Check for correctness
        try:
            # Use higher tolerance for bfloat16
            atol = 1e-5
            logger.info("Verifying dx...")
            assert_verbose_allclose_two_rounds(dx_triton, dx_ref, atol=atol)
            logger.success(f"Verification passed for m={m}")
        except AssertionError as e:
            logger.error(f"Verification failed for m={m}")
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # Test with various M values, including those not divisible by block sizes
    m_choices = [4096]
    n = 14336
    k = 4096
    r = 16
    dropout_p = 0.1
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(m_choices, n, k, r, dropout_p, dtype)

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(f"Benchmarking fused_lora_dyw_dsa with m={m}, n={n}, k={k}, r={r}")
        curr_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            dropout_p=dropout_p,
            dtype=dtype,
        )

        benchmark(
            fused_lora_dyw_dsa_tma,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_dyw_dsa_tma_m_{m}",
        )

        benchmark(
            torch_dyw_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_dyw_ref_m_{m}",
        )

        benchmark(
            torch_lora_dyw_dsa_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_lora_dyw_dsa_ref_m_{m}",
        )
