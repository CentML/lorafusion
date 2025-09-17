"""Tune Triton kernel configurations for different hardware."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import product
from pathlib import Path

import click
import torch
from loguru import logger

from lorafusion.ops.lora_v1 import HARDWARE_USE_TMA
from lorafusion.ops.tests.test_fused_multi_lora_dys_dyb import (
    prepare_func as prepare_multi_lora_dys_dyb,
)
from lorafusion.ops.tests.test_fused_multi_lora_dyw_dsa import (
    prepare_func as prepare_multi_lora_dyw_dsa,
)
from lorafusion.ops.tests.test_fused_multi_lora_xw_sb import (
    prepare_func as prepare_multi_lora_xw_sb,
)
from lorafusion.ops.triton_ops.config import LoRATritonConfig
from lorafusion.ops.triton_ops.fused_lora_dys_dyb import (
    fused_lora_dys_dyb,
)
from lorafusion.ops.triton_ops.fused_lora_dys_dyb import (
    prepare_func as prepare_dys_dyb,
)
from lorafusion.ops.triton_ops.fused_lora_dyw_dsa import (
    fused_lora_dyw_dsa,
)
from lorafusion.ops.triton_ops.fused_lora_dyw_dsa import (
    prepare_func as prepare_dyw_dsa,
)
from lorafusion.ops.triton_ops.fused_lora_dyw_dsa_tma import fused_lora_dyw_dsa_tma
from lorafusion.ops.triton_ops.fused_lora_xw_sb import (
    fused_lora_xw_sb,
)
from lorafusion.ops.triton_ops.fused_lora_xw_sb import (
    prepare_func as prepare_xw_sb,
)
from lorafusion.ops.triton_ops.fused_lora_xw_sb_tma import fused_lora_xw_sb_tma
from lorafusion.ops.triton_ops.fused_multi_lora_dys_dyb import (
    fused_multi_lora_dys_dyb,
)
from lorafusion.ops.triton_ops.fused_multi_lora_dyw_dsa import (
    fused_multi_lora_dyw_dsa,
)
from lorafusion.ops.triton_ops.fused_multi_lora_xw_sb import (
    fused_multi_lora_xw_sb,
)
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.common import get_device_short_name


@dataclass
class BenchmarkResult:
    """Result of a kernel benchmark."""

    kernel_name: str
    config: LoRATritonConfig
    time_ms: float
    torch_gemm_ms: float | None = None  # PyTorch baseline time for comparison


class KernelTuner:
    """Tuner for Triton kernel configurations."""

    def __init__(self, warmup: int = 50, number: int = 50) -> None:
        """Initialize the kernel tuner.

        Args:
            warmup: Number of warmup iterations.
            number: Number of benchmark iterations.
        """
        self.warmup = warmup
        self.number = number
        set_warmup_and_number(warmup, number)

    def generate_config_candidates(self, kernel_name: str) -> list[LoRATritonConfig]:
        """Generate configuration candidates for a kernel.

        Args:
            kernel_name: Name of the kernel to generate candidates for.

        Returns:
            List of configuration candidates.
        """
        # Check if this kernel doesn't use BLOCK_SIZE_N
        no_block_size_n = "dys_dyb" in kernel_name

        # Generate candidates based on different strategies
        block_size_m_candidates = [64, 128]
        block_size_n_candidates = [128, 256]
        block_size_k_candidates = [64, 128]
        group_size_m_candidates = [8]
        num_stages_wraps_pair_candidates = [(4, 8), (3, 8), (4, 4)]
        if no_block_size_n:
            num_stages_wraps_pair_candidates += [(5, 8)]

        candidates = [
            LoRATritonConfig(
                block_size_m=bm,
                block_size_n=bn,
                block_size_k=bk,
                group_size_m=gsm,
                num_stages=ns,
                num_warps=nw,
            )
            for bm, bn, bk, gsm, (ns, nw) in product(
                block_size_m_candidates,
                block_size_n_candidates,
                block_size_k_candidates,
                group_size_m_candidates,
                num_stages_wraps_pair_candidates,
            )
        ]

        # Pre-tuned configs for bk=32
        candidates += [
            # bm, bn, bk, gsm, ns, nw
            LoRATritonConfig(64, 256, 32, 8, 4, 4),
            LoRATritonConfig(128, 128, 32, 8, 4, 4),
            LoRATritonConfig(128, 64, 32, 8, 4, 4),
            LoRATritonConfig(64, 128, 32, 8, 4, 4),
            LoRATritonConfig(128, 32, 32, 8, 4, 4),
            LoRATritonConfig(64, 32, 32, 8, 5, 2),
            LoRATritonConfig(32, 64, 32, 8, 5, 2),
        ]

        # If there is no block size n, set block size n to None
        if no_block_size_n:
            candidates = [
                replace(candidate, block_size_n=None) for candidate in candidates
            ]

        # Remove duplicates by converting to set and back to list
        # This handles the case where no_block_size_n creates duplicate configs
        return list(set(candidates))

    def benchmark_torch_gemm(
        self,
        kernel_name: str,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
    ) -> float | None:
        """Calculate PyTorch baseline time for GEMM operations.

        Args:
            kernel_name: Name of the kernel.
            m: Batch size dimension.
            n: Output dimension.
            k: Input dimension.
            r: LoRA rank.
            alpha: LoRA alpha.
            dtype: Data type.

        Returns:
            PyTorch baseline time in milliseconds, or None if not applicable.
        """
        if "xw" not in kernel_name and "dyw" not in kernel_name:
            return None  # Only calculate baseline for GEMM operations

        # Prepare inputs
        def prepare_func() -> dict[str, torch.Tensor]:
            x = torch.randn(m, k, device="cuda", dtype=dtype) / 10
            w = torch.randn(n, k, device="cuda", dtype=dtype) / 10
            return {"x": x, "w": w}

        time_result = benchmark(
            lambda x, w: x @ w.T,
            prepare_func=prepare_func,
            use_cuda_graph=False,
            use_cuda_event=True,
        )

        return time_result * 1000  # Convert to milliseconds

    def benchmark_kernel_with_config(
        self,
        kernel_func: callable,
        prepare_func: callable,
        config: LoRATritonConfig,
        kernel_name: str,
        torch_gemm_ms: float | None = None,
    ) -> BenchmarkResult:
        """Benchmark a kernel with a specific configuration.

        Args:
            kernel_func: The kernel function to benchmark.
            prepare_func: Function to prepare inputs.
            config: The configuration to test.
            kernel_name: Name of the kernel.
            torch_gemm_ms: PyTorch baseline time for GEMM operations in milliseconds.

        Returns:
            Benchmark result.
        """
        try:
            # Create a wrapper function that passes the config to the kernel
            def kernel_with_config(*args, **kwargs) -> torch.Tensor:
                # Add config parameter to kwargs only if config is not None
                if config is not None:
                    kwargs["config"] = config
                return kernel_func(*args, **kwargs)

            # Run benchmark with the specified config
            time_result = benchmark(
                kernel_with_config,
                prepare_func=prepare_func,
                use_cuda_graph=False,
                use_cuda_event=True,
            )

            return BenchmarkResult(
                kernel_name=kernel_name,
                config=config,
                time_ms=time_result * 1000,  # Convert to milliseconds
                torch_gemm_ms=torch_gemm_ms,
            )

        except Exception as e:
            logger.warning(
                f"Failed to benchmark {kernel_name} with config {config}: {e}"
            )
            # raise e
            return BenchmarkResult(
                kernel_name=kernel_name,
                config=config,
                time_ms=float("inf"),
                torch_gemm_ms=torch_gemm_ms,
            )

    def tune_kernel(
        self,
        kernel_name: str,
        kernel_func: callable,
        prepare_func: callable,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        block_size_m: int | None = None,
    ) -> BenchmarkResult:
        """Tune a single kernel by testing different configurations.

        Args:
            kernel_name: Name of the kernel.
            kernel_func: The kernel function to tune.
            prepare_func: Function to prepare inputs.
            m: Batch size dimension.
            n: Output dimension.
            k: Input dimension.
            dtype: Data type.
            block_size_m: Block size for the M dimension.

        Returns:
            Best benchmark result.
        """
        logger.info(f"Tuning {kernel_name}...")

        # Calculate PyTorch baseline for GEMM operations
        torch_gemm_ms = self.benchmark_torch_gemm(kernel_name, m, n, k, dtype)
        if torch_gemm_ms is not None:
            logger.info(f"PyTorch baseline: {torch_gemm_ms:.3f}ms")

        candidates = self.generate_config_candidates(kernel_name)
        logger.info(f"Generated {len(candidates)} configuration candidates")

        best_result = None
        best_time = float("inf")

        for i, config in enumerate(candidates):
            logger.info(f"Testing config {i + 1}/{len(candidates)}: {config}")

            if block_size_m is not None and config.block_size_m != block_size_m:
                logger.info(
                    f" > Skipping config {config} because it has block size m "
                    f"{config.block_size_m} != {block_size_m}"
                )
                continue

            result = self.benchmark_kernel_with_config(
                kernel_func, prepare_func, config, kernel_name, torch_gemm_ms
            )

            if result.time_ms < best_time:
                best_time = result.time_ms
                best_result = result
                logger.info(f" > New best time: {best_time:.3f}ms")

        logger.info(f"Best configuration for {kernel_name}: {best_result.config}")
        logger.info(f"Best time: {best_result.time_ms:.3f}ms")

        return best_result

    def tune_all_kernels(
        self,
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
        r: int = 16,
        alpha: float = 16.0,
        dtype: torch.dtype = torch.bfloat16,
        *,
        include_multi_lora: bool = True,
    ) -> dict[str, BenchmarkResult]:
        """Tune all kernels by testing different configurations.

        Args:
            m: Batch size dimension.
            n: Output dimension.
            k: Input dimension.
            r: LoRA rank.
            alpha: LoRA alpha.
            dtype: Data type.
            include_multi_lora: Whether to include multi-LoRA kernels.

        Returns:
            Dictionary of best results for each kernel.
        """
        kernel_configs = {
            "fused_lora_xw_sb": {
                "kernel_func": fused_lora_xw_sb,
                "prepare_func": lambda: prepare_xw_sb(
                    m=m, n=n, k=k, r=r, alpha=alpha, dtype=dtype, with_bias=False
                ),
            },
            "fused_lora_dyw_dsa": {
                "kernel_func": fused_lora_dyw_dsa,
                "prepare_func": lambda: prepare_dyw_dsa(
                    m=m, n=n, k=k, r=r, dropout_p=0.0, dtype=dtype
                ),
            },
            "fused_lora_dys_dyb": {
                "kernel_func": fused_lora_dys_dyb,
                "prepare_func": lambda: prepare_dys_dyb(
                    m=m, n=n, r=r, alpha=alpha, dtype=dtype
                ),
            },
        }

        tma_kernel_configs = {
            "fused_lora_xw_sb_tma": {
                "kernel_func": fused_lora_xw_sb_tma,
                "prepare_func": lambda: prepare_xw_sb(
                    m=m, n=n, k=k, r=r, alpha=alpha, dtype=dtype, with_bias=False
                ),
            },
            "fused_lora_dyw_dsa_tma": {
                "kernel_func": fused_lora_dyw_dsa_tma,
                "prepare_func": lambda: prepare_dyw_dsa(
                    m=m, n=n, k=k, r=r, dropout_p=0.0, dtype=dtype
                ),
            },
        }

        # Add TMA kernels if hardware supports it
        if HARDWARE_USE_TMA:
            kernel_configs.update(tma_kernel_configs)

        results = {}

        # Tune each kernel
        for kernel_name, config in kernel_configs.items():
            logger.info("=" * 80)
            logger.info(f"TUNING {kernel_name.upper()}")
            logger.info("=" * 80)

            results[kernel_name] = self.tune_kernel(
                kernel_name,
                config["kernel_func"],
                config["prepare_func"],
                m,
                n,
                k,
                dtype,
            )

        # Tune multi-LoRA kernels if requested
        if include_multi_lora:
            block_size_m = results["fused_lora_xw_sb"].config.block_size_m
            seq_len_list = [2048, 2048]
            lora_idx_list = [0, 1]
            lora_rank_list = [16, 16]
            dropout_p_list = [0.1, 0.1]
            alpha_list = [16.0, 16.0]
            n = k = 4096
            def _multi_lora_prep_partial(prepare_func: callable) -> callable:
                return partial(
                    prepare_func,
                    seq_len_list=seq_len_list,
                    lora_idx_list=lora_idx_list,
                    lora_rank_list=lora_rank_list,
                    dropout_p_list=dropout_p_list,
                    alpha_list=alpha_list,
                    n=n,
                    k=k,
                    block_size_m=block_size_m,
                    dtype=dtype,
                )

            multi_lora_kernel_configs = {
                "fused_multi_lora_xw_sb": {
                    "kernel_func": fused_multi_lora_xw_sb,
                    "prepare_func": _multi_lora_prep_partial(prepare_multi_lora_xw_sb),
                },
                "fused_multi_lora_dyw_dsa": {
                    "kernel_func": fused_multi_lora_dyw_dsa,
                    "prepare_func": _multi_lora_prep_partial(prepare_multi_lora_dyw_dsa),
                },
                "fused_multi_lora_dys_dyb": {
                    "kernel_func": fused_multi_lora_dys_dyb,
                    "prepare_func": _multi_lora_prep_partial(prepare_multi_lora_dys_dyb),
                },
            }

            for kernel_name, config in multi_lora_kernel_configs.items():
                logger.info("=" * 80)
                logger.info(f"TUNING {kernel_name.upper()}")
                logger.info("=" * 80)

                results[kernel_name] = self.tune_kernel(
                    kernel_name,
                    config["kernel_func"],
                    config["prepare_func"],
                    m,
                    n,
                    k,
                    dtype,
                    block_size_m=block_size_m,
                )

        return results

    def save_results(
        self,
        results: dict[str, BenchmarkResult],
        output_path: str | Path,
    ) -> None:
        """Save tuning results to a file.

        Args:
            results: Dictionary of benchmark results.
            output_path: Path to save the results.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to a serializable format
        serializable_results = {}
        for kernel_name, result in results.items():
            serializable_results[kernel_name] = {
                "kernel_name": result.kernel_name,
                "config": asdict(result.config),
                "time_ms": result.time_ms,
                "torch_gemm_ms": result.torch_gemm_ms,
            }

        # Add metadata
        metadata = {
            "gpu_name": get_device_short_name(),
            "timestamp": time.time(),
            "tuning_parameters": {
                "warmup": self.warmup,
                "number": self.number,
            },
        }

        output_data = {
            "metadata": metadata,
            "results": serializable_results,
        }

        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")


@click.command()
@click.option("--m", type=int, default=4096, help="Batch size dimension.")
@click.option("--n", type=int, default=4096, help="Output dimension.")
@click.option("--k", type=int, default=4096, help="Input dimension.")
@click.option("--r", type=int, default=16, help="LoRA rank.")
@click.option("--alpha", type=float, default=16.0, help="LoRA alpha.")
@click.option("--dtype", type=str, default="bfloat16", help="Data type.")
@click.option("--warmup", type=int, default=100, help="Number of warmup iterations.")
@click.option("--number", type=int, default=50, help="Number of benchmark iterations.")
@click.option(
    "--output",
    type=str,
    default="./results/tuning_results.json",
    help="Output file path.",
)
@click.option(
    "--include-multi-lora",
    is_flag=True,
    default=True,
    help="Include multi-LoRA kernels in tuning.",
)
def main(
    m: int,
    n: int,
    k: int,
    r: int,
    alpha: float,
    dtype: str,
    warmup: int,
    number: int,
    output: str,
    *,
    include_multi_lora: bool,
) -> None:
    """Tune Triton kernel configurations."""
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    logger.info(f"Starting kernel tuning on {get_device_short_name()}")
    logger.info(f"Dimensions: M={m}, N={n}, K={k}, R={r}")
    logger.info(f"Data type: {dtype}")
    logger.info(f"Warmup: {warmup}, Number: {number}")

    tuner = KernelTuner(warmup=warmup, number=number)

    results = tuner.tune_all_kernels(
        m=m,
        n=n,
        k=k,
        r=r,
        alpha=alpha,
        dtype=torch_dtype,
        include_multi_lora=include_multi_lora,
    )

    tuner.save_results(results, output)

    # Print summary
    logger.info("=" * 80)
    logger.info("TUNING SUMMARY")
    logger.info("=" * 80)

    logger.info(f"Device short name: {get_device_short_name()}")
    for kernel_name, result in results.items():
        logger.info(f"{kernel_name}: {result.time_ms:.3f}ms")
        logger.info(f"  Config: {result.config}")
    logger.info(
        "Please update the lorafusion/ops/triton_ops/config.py file with the "
        "tuned configurations."
    )


if __name__ == "__main__":
    main()
