# ruff: noqa: ANN001, N803, N806
"""Triton Fused LoRA xw + sb using TMA."""

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from loguru import logger

from lorafusion.ops.triton_ops.config import LoRATritonConfig, get_lora_kernel_config
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
    alpha: float,
    dtype: torch.dtype = torch.bfloat16,
    *,
    with_bias: bool = False,
) -> dict[str, Any]:
    """Prepare the input tensors for the fused LoRA xw + sb kernel."""
    x = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    w = torch.randn(n, k, device="cuda", dtype=dtype) / 10
    s = torch.randn(m, r, device="cuda", dtype=dtype) / 10
    b = torch.randn(n, r, device="cuda", dtype=dtype) / 10
    bias = torch.randn(n, device="cuda", dtype=dtype) / 10 if with_bias else None

    return {
        "x": x,
        "w": w,
        "s": s,
        "b": b,
        "alpha": alpha,
        "bias": bias,
    }


def torch_xw_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    return x @ w.T


def torch_lora_xw_sb_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch reference implementation of the fused LoRA xw + sb."""
    result = x @ w.T + s @ b.T * alpha
    if bias is not None:
        result = result + bias
    return result


@triton.jit
def fused_lora_xw_sb_tma_kernel(
    x_desc_ptr,
    w_desc_ptr,
    s_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    alpha: tl.constexpr,
    has_bias: tl.constexpr,
    stride_sm,
    stride_sr,
    stride_bn: tl.constexpr,
    stride_br: tl.constexpr,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr,
    FLATTEN: tl.constexpr,
    NUM_SMS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
) -> None:
    """Triton Fused LoRA xw + sb kernel using TMA.

    Compute Out = XW + S @ B * alpha + bias

    Dimensions:
      X: [M, K]
      W: [K, N]
      S: [M, R]
      B: [R, N]
      bias: [N]
      Out: [M, N]
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

    for tile_id in range(
        start_pid,
        num_tiles,
        NUM_SMS,
    ):
        # ------------------------------------------------------------
        # 2. Compute the tile indices for M, N
        # ------------------------------------------------------------
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_xm = pid_m * BLOCK_SIZE_M
        offs_wn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # ------------------------------------------------------------
        # 3. Compute the LoRA part
        # ------------------------------------------------------------
        # Create offsets with boundary checks
        v_offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        v_offs_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        v_offs_r = tl.arange(0, R)

        # Apply boundary checks
        mask_xm = v_offs_xm < M
        mask_wn = v_offs_wn < N

        # Compute actual offsets for memory access
        v_offs_xm = tl.where(mask_xm, v_offs_xm, 0)
        v_offs_wn = tl.where(mask_wn, v_offs_wn, 0)

        # Compute pointers with proper strides
        s_ptrs = s_ptr + (
            v_offs_xm[:, None] * stride_sm + v_offs_r[None, :] * stride_sr
        )
        b_ptrs = b_ptr + (
            v_offs_r[:, None] * stride_br + v_offs_wn[None, :] * stride_bn
        )

        # Load with masks to avoid out-of-bounds access
        s = tl.load(s_ptrs, mask=mask_xm[:, None])
        b = tl.load(b_ptrs, mask=mask_wn[None, :])

        # Compute the LoRA contribution
        accumulator = tl.dot(s, b * alpha, accumulator)

        # ------------------------------------------------------------
        # 4. Main loop
        # ------------------------------------------------------------
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            x = tl_experimental_descriptor_load(
                x_desc_ptr, [offs_xm, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
            )
            w = tl_experimental_descriptor_load(
                w_desc_ptr, [offs_wn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
            )
            accumulator = tl.dot(x, w.T, accumulator)

        # ------------------------------------------------------------
        # 5. Add bias if available
        # ------------------------------------------------------------
        if has_bias:
            # Load bias with proper mask
            v_offs_wn_safe = tl.where(mask_wn, v_offs_wn, 0)
            bias_ptrs = bias_ptr + v_offs_wn_safe * stride_bias
            bias_values = tl.load(bias_ptrs, mask=mask_wn, other=0.0)

            # Add bias to each row
            accumulator += bias_values[None, :]

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_xm_c = pid_m * BLOCK_SIZE_M
        offs_wn_c = pid_n * BLOCK_SIZE_N

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
            tl_experimental_descriptor_store(out_ptr, c0, [offs_xm_c, offs_wn_c])
            c1 = acc1.to(dtype)
            tl_experimental_descriptor_store(
                out_ptr, c1, [offs_xm_c, offs_wn_c + BLOCK_SIZE_N // 2]
            )
        else:
            accumulator = accumulator.to(dtype)
            tl_experimental_descriptor_store(
                out_ptr, accumulator, [offs_xm_c, offs_wn_c]
            )


desc_helper = TmaAutoTuneHelper()
desc_helper.init_tma_descriptor("x")
desc_helper.init_tma_descriptor("w")
desc_helper.init_tma_descriptor("s")
desc_helper.init_tma_descriptor("b")
desc_helper.init_tma_descriptor("out")

desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
desc_w = desc_helper.get_tma_descriptor_kernel_param("w")
desc_out = desc_helper.get_tma_descriptor_kernel_param("out")


def fused_lora_xw_sb_tma(  # noqa: C901
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    *,
    bias: torch.Tensor | None = None,
    config: LoRATritonConfig | None = None,
) -> torch.Tensor:
    """Triton Fused LoRA xw + sb using TMA."""
    # Check constraints.
    if x.shape[1] != w.shape[1]:
        msg = (
            f"Incompatible dimensions: {x.shape[1]} != {w.shape[1]}. "
            f"x: {x.shape}, w: {w.shape}"
        )
        raise ValueError(msg)
    if s.shape[1] != b.shape[1]:
        msg = (
            f"Incompatible dimensions: {s.shape[1]} != {b.shape[1]}. "
            f"s: {s.shape}, b: {b.shape}"
        )
        raise ValueError(msg)
    if x.dtype != w.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {w.dtype}"
        raise ValueError(msg)
    if x.dtype != s.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {s.dtype}"
        raise ValueError(msg)
    if x.dtype != b.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {b.dtype}"
        raise ValueError(msg)
    if bias is not None and x.dtype != bias.dtype:
        msg = f"Incompatible dtypes: {x.dtype} != {bias.dtype}"
        raise ValueError(msg)
    if not x.is_contiguous():
        msg = "Matrix A must be contiguous"
        raise ValueError(msg)

    M, K = x.shape
    N, K = w.shape
    R = s.shape[1]

    # Check bias dimensions if provided
    has_bias = bias is not None

    # Use empty tensor with stride 0 if bias is None to avoid conditional logic in
    # kernel
    if not has_bias:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
        bias_stride = 0
    else:
        bias_stride = bias.stride(0)

    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Get configs
    if config is None:
        lora_kernel_config = get_lora_kernel_config("fused_lora_xw_sb_tma")
    else:
        lora_kernel_config = config
    triton_config = lora_kernel_config.to_triton_config()
    # Set TMA-specific config parameters
    triton_config.kwargs["EPILOGUE_SUBTILE"] = False
    triton_config.kwargs["LOOP_UNROLL_FACTOR"] = None
    triton_config.kwargs["FLATTEN"] = False

    # Initialize TMA descriptors.
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META: dict[str, Any]) -> int:
        desc_helper.fill_2d_tma_descriptor(
            "x",
            x.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            x.element_size(),
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
            "out",
            out.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"],
            store_block_n,
            out.element_size(),
        )

        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    compiled_kernel = fused_lora_xw_sb_tma_kernel[grid](
        desc_x,
        desc_w,
        s,
        b,
        bias,
        desc_out,
        M,
        N,
        K,
        R,
        alpha,
        has_bias,
        s.stride(0),
        s.stride(1),
        b.stride(0),
        b.stride(1),
        bias_stride,
        NUM_SMS=NUM_SMS,
        OUTPUT_DTYPE=torch_dtype_to_triton_dtype(out.dtype),
        **triton_config.all_kwargs(),
    )
    if (
        not torch.compiler.is_compiling()
        and compiled_kernel is not None
        and compiled_kernel.n_spills > 0
    ):
        logger.warning(
            f"Compiled kernel: {compiled_kernel}, "
            f"n_regs: {compiled_kernel.n_regs}, "
            f"n_spills: {compiled_kernel.n_spills}"
        )
        logger.warning(f"Compiled kernel.metadata: {compiled_kernel.metadata}")
    return out


def verify_kernel_correctness(
    m_values: list[int],
    n: int,
    k: int,
    r: int,
    alpha: float = 16.0,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Verify that the kernel produces correct results for all tested dimensions."""
    for m in m_values:
        logger.info(f"Verifying kernel correctness for m={m}...")

        # Test without bias
        inputs_no_bias = prepare_func(
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=False,
        )

        # Run triton kernel
        triton_output_no_bias = fused_lora_xw_sb_tma(**inputs_no_bias)

        # Compute reference result
        ref_output_no_bias = torch_lora_xw_sb_ref(**inputs_no_bias)

        # Check for correctness
        try:
            # Here we used a looser tolerance because the computation is not exactly the
            # same because of the summation of the dtype.
            # We use fp32 + fp32 accumulation for the addition of lora part and base
            # layer, which should be more accurate than fp16 + fp16 accumulation.
            assert_verbose_allclose_two_rounds(
                triton_output_no_bias, ref_output_no_bias, atol=5e-3
            )
            logger.success(f"Verification passed for m={m} without bias")
        except AssertionError as e:
            logger.error(f"Verification failed for m={m} without bias")
            logger.error(e)

        # Test with bias
        inputs_with_bias = prepare_func(
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=True,
        )

        # Run triton kernel
        triton_output_with_bias = fused_lora_xw_sb_tma(**inputs_with_bias)

        # Compute reference result
        ref_output_with_bias = torch_lora_xw_sb_ref(**inputs_with_bias)

        # Check for correctness
        try:
            assert_verbose_allclose_two_rounds(
                triton_output_with_bias, ref_output_with_bias, atol=5e-3
            )
            logger.success(f"Verification passed for m={m} with bias")
        except AssertionError as e:
            logger.error(f"Verification failed for m={m} with bias")
            logger.error(e)


if __name__ == "__main__":
    set_warmup_and_number(2000, 1000)

    # Test with various M values, including those not divisible by 128
    m_choices = [4020]
    n = 4096
    k = 4096
    r = 16
    alpha = 16
    dtype = torch.bfloat16

    # First verify kernel correctness for various M values
    logger.info("=" * 80)
    logger.info("VERIFYING KERNEL CORRECTNESS")
    logger.info("=" * 80)
    verify_kernel_correctness(m_choices, n, k, r, alpha, dtype)

    # Then run benchmarks
    logger.info("=" * 80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 80)

    for m in m_choices:
        logger.info("-" * 60)
        logger.info(f"Benchmarking fused_lora_xw_sb with m={m}, n={n}, k={k}, r={r}")

        # Without bias
        curr_prepare_func = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=False,
        )

        benchmark(
            fused_lora_xw_sb_tma,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_xw_sb_tma_m_{m}_no_bias",
        )

        benchmark(
            torch_xw_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_xw_ref_m_{m}",
        )

        benchmark(
            torch_lora_xw_sb_ref,
            prepare_func=curr_prepare_func,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_lora_xw_sb_ref_m_{m}_no_bias",
        )

        # With bias
        curr_prepare_func_with_bias = partial(
            prepare_func,
            m=m,
            n=n,
            k=k,
            r=r,
            alpha=alpha,
            dtype=dtype,
            with_bias=True,
        )

        benchmark(
            fused_lora_xw_sb_tma,
            prepare_func=curr_prepare_func_with_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"fused_lora_xw_sb_tma_m_{m}_with_bias",
        )

        benchmark(
            torch_lora_xw_sb_ref,
            prepare_func=curr_prepare_func_with_bias,
            use_cuda_graph=True,
            use_cuda_event=True,
            msg=f"torch_lora_xw_sb_ref_m_{m}_with_bias",
        )
