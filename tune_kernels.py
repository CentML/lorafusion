"""Tune Triton kernel configurations for different hardware."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import torch
import triton
from loguru import logger

from lorafusion.ops.triton_ops.config import HardwareConfig, TritonConfig, get_gpu_name
from lorafusion.ops.triton_ops.fused_lora_dys_dyb import fused_lora_dys_dyb
from lorafusion.ops.triton_ops.fused_lora_dyw_dsa import fused_lora_dyw_dsa
from lorafusion.ops.triton_ops.fused_lora_dyw_dsa_tma import fused_lora_dyw_dsa_tma
from lorafusion.ops.triton_ops.fused_lora_xw_sb import fused_lora_xw_sb
from lorafusion.ops.triton_ops.fused_lora_xw_sb_tma import fused_lora_xw_sb_tma
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number


@dataclass
class BenchmarkResult:
    """Result of a kernel benchmark."""

    kernel_name: str
    config: TritonConfig
    time_ms: float
    throughput_gflops: float | None = None


class KernelTuner:
    """Tuner for Triton kernel configurations."""

    def __init__(self, warmup: int = 100, number: int = 50) -> None:
        """Initialize the kernel tuner.

        Args:
            warmup: Number of warmup iterations.
            number: Number of benchmark iterations.
        """
        self.warmup = warmup
        self.number = number
        set_warmup_and_number(warmup, number)

    def generate_config_candidates(
        self, 
        kernel_name: str,
        base_config: TritonConfig | None = None
    ) -> list[TritonConfig]:
        """Generate configuration candidates for a kernel.

        Args:
            kernel_name: Name of the kernel.
            base_config: Base configuration to start from.

        Returns:
            List of configuration candidates.
        """
        if base_config is None:
            # Default configuration
            base_config = TritonConfig(
                block_size_m=64,
                block_size_n=128,
                block_size_k=32,
                group_size_m=8,
                num_stages=2,
                num_warps=4,
            )

        candidates = [
            TritonConfig(block_size_m=128, block_size_n=256, block_size_k=64, group_size_m=8, num_stages=3, num_warps=8),
            TritonConfig(block_size_m=64, block_size_n=256, block_size_k=32, group_size_m=8, num_stages=4, num_warps=4),
            TritonConfig(block_size_m=128, block_size_n=128, block_size_k=32, group_size_m=8, num_stages=4, num_warps=4),
            TritonConfig(block_size_m=128, block_size_n=64, block_size_k=32, group_size_m=8, num_stages=4, num_warps=4),
            TritonConfig(block_size_m=64, block_size_n=128, block_size_k=32, group_size_m=8, num_stages=4, num_warps=4),
            TritonConfig(block_size_m=128, block_size_n=32, block_size_k=32, group_size_m=8, num_stages=4, num_warps=4),
            TritonConfig(block_size_m=64, block_size_n=32, block_size_k=32, group_size_m=8, num_stages=5, num_warps=2),
            TritonConfig(block_size_m=32, block_size_n=64, block_size_k=32, group_size_m=8, num_stages=5, num_warps=2),
        ]


        # Generate candidates based on different strategies
        block_size_m_candidates = [32, 64, 128]
        block_size_n_candidates = [64, 128, 256]
        block_size_k_candidates = [32, 64]
        group_size_m_candidates = [8]
        num_stages_wraps_pair_candidates = [(3, 8), (4, 8), (4, 4), (5, 4), (5, 2)]

        # Filter candidates based on hardware constraints
        for bm in block_size_m_candidates:
            for bn in block_size_n_candidates:
                for bk in block_size_k_candidates:
                    for gsm in group_size_m_candidates:
                        for ns in num_stages_candidates:
                            for nw in num_warps_candidates:
                                # Basic constraints
                                if bm * bk > 256 * 256:  # Memory constraint
                                    continue
                                if bn * bk > 256 * 256:  # Memory constraint
                                    continue
                                
                                # Warp constraints
                                if nw > 32:  # Max warps per block
                                    continue
                                
                                # Stage constraints
                                if ns > 6:  # Max stages
                                    continue
                                
                                candidates.append(TritonConfig(
                                    block_size_m=bm,
                                    block_size_n=bn,
                                    block_size_k=bk,
                                    group_size_m=gsm,
                                    num_stages=ns,
                                    num_warps=nw,
                                ))

        # Limit the number of candidates to avoid excessive tuning time
        max_candidates = 50
        if len(candidates) > max_candidates:
            # Sample candidates evenly
            step = len(candidates) // max_candidates
            candidates = candidates[::step][:max_candidates]

        return candidates

    def prepare_fused_lora_xw_sb_inputs(
        self,
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
        r: int = 16,
        alpha: float = 16.0,
        dtype: torch.dtype = torch.bfloat16,
        with_bias: bool = False,
    ) -> dict[str, Any]:
        """Prepare inputs for fused_lora_xw_sb kernel."""
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

    def prepare_fused_lora_dyw_dsa_inputs(
        self,
        m: int = 4096,
        n: int = 4096,
        k: int = 4096,
        r: int = 16,
        alpha: float = 16.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, Any]:
        """Prepare inputs for fused_lora_dyw_dsa kernel."""
        dy = torch.randn(m, n, device="cuda", dtype=dtype) / 10
        w = torch.randn(n, k, device="cuda", dtype=dtype) / 10
        ds = torch.randn(m, r, device="cuda", dtype=dtype) / 10
        a = torch.randn(r, k, device="cuda", dtype=dtype) / 10

        return {
            "dy": dy,
            "w": w,
            "ds": ds,
            "a": a,
        }

    def prepare_fused_lora_dys_dyb_inputs(
        self,
        m: int = 4096,
        n: int = 4096,
        r: int = 16,
        alpha: float = 16.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, Any]:
        """Prepare inputs for fused_lora_dys_dyb kernel."""
        dy = torch.randn(m, n, device="cuda", dtype=dtype) / 10
        b = torch.randn(n, r, device="cuda", dtype=dtype) / 10
        s = torch.randn(m, r, device="cuda", dtype=dtype) / 10

        return {
            "dy": dy,
            "b": b,
            "s": s,
            "alpha": alpha,
        }

    def benchmark_kernel_with_config(
        self,
        kernel_func: Any,
        prepare_func: Any,
        config: TritonConfig,
        kernel_name: str,
    ) -> BenchmarkResult:
        """Benchmark a kernel with a specific configuration.

        Args:
            kernel_func: The kernel function to benchmark.
            prepare_func: Function to prepare inputs.
            config: The configuration to test.
            kernel_name: Name of the kernel.

        Returns:
            Benchmark result.
        """
        try:
            # Temporarily override the kernel configuration
            # This is a bit hacky but necessary for testing different configs
            original_configs = getattr(kernel_func, 'configs', None)
            
            # Create a temporary config list
            temp_configs = [config.to_triton_config()]
            
            # Temporarily replace the configs
            if hasattr(kernel_func, 'configs'):
                kernel_func.configs = temp_configs
            
            # Run benchmark
            time_result = benchmark(
                kernel_func,
                prepare_func=prepare_func,
                use_cuda_graph=True,
                use_cuda_event=True,
            )
            
            # Restore original configs
            if original_configs is not None:
                kernel_func.configs = original_configs
            
            return BenchmarkResult(
                kernel_name=kernel_name,
                config=config,
                time_ms=time_result * 1000,  # Convert to milliseconds
            )
            
        except Exception as e:
            logger.warning(f"Failed to benchmark {kernel_name} with config {config}: {e}")
            return BenchmarkResult(
                kernel_name=kernel_name,
                config=config,
                time_ms=float('inf'),
            )

    def tune_kernel(
        self,
        kernel_name: str,
        kernel_func: Any,
        prepare_func: Any,
        base_config: TritonConfig | None = None,
    ) -> BenchmarkResult:
        """Tune a single kernel.

        Args:
            kernel_name: Name of the kernel.
            kernel_func: The kernel function to tune.
            prepare_func: Function to prepare inputs.
            base_config: Base configuration to start from.

        Returns:
            Best benchmark result.
        """
        logger.info(f"Tuning {kernel_name}...")
        
        candidates = self.generate_config_candidates(kernel_name, base_config)
        logger.info(f"Generated {len(candidates)} configuration candidates")
        
        best_result = None
        best_time = float('inf')
        
        for i, config in enumerate(candidates):
            logger.info(f"Testing config {i+1}/{len(candidates)}: {config}")
            
            result = self.benchmark_kernel_with_config(
                kernel_func, prepare_func, config, kernel_name
            )
            
            if result.time_ms < best_time:
                best_time = result.time_ms
                best_result = result
                logger.info(f"New best time: {best_time:.3f}ms")
        
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
    ) -> dict[str, BenchmarkResult]:
        """Tune all kernels.

        Args:
            m: Batch size dimension.
            n: Output dimension.
            k: Input dimension.
            r: LoRA rank.
            alpha: LoRA alpha.
            dtype: Data type.

        Returns:
            Dictionary of best results for each kernel.
        """
        results = {}
        
        # Tune fused_lora_xw_sb
        logger.info("=" * 80)
        logger.info("TUNING FUSED_LORA_XW_SB")
        logger.info("=" * 80)
        
        prepare_func = lambda: self.prepare_fused_lora_xw_sb_inputs(
            m=m, n=n, k=k, r=r, alpha=alpha, dtype=dtype, with_bias=False
        )
        
        results["fused_lora_xw_sb"] = self.tune_kernel(
            "fused_lora_xw_sb",
            fused_lora_xw_sb,
            prepare_func,
        )
        
        # Tune fused_lora_xw_sb_tma (if available)
        try:
            logger.info("=" * 80)
            logger.info("TUNING FUSED_LORA_XW_SB_TMA")
            logger.info("=" * 80)
            
            results["fused_lora_xw_sb_tma"] = self.tune_kernel(
                "fused_lora_xw_sb_tma",
                fused_lora_xw_sb_tma,
                prepare_func,
            )
        except Exception as e:
            logger.warning(f"Could not tune fused_lora_xw_sb_tma: {e}")
        
        # Tune fused_lora_dyw_dsa
        logger.info("=" * 80)
        logger.info("TUNING FUSED_LORA_DYW_DSA")
        logger.info("=" * 80)
        
        prepare_func = lambda: self.prepare_fused_lora_dyw_dsa_inputs(
            m=m, n=n, k=k, r=r, alpha=alpha, dtype=dtype
        )
        
        results["fused_lora_dyw_dsa"] = self.tune_kernel(
            "fused_lora_dyw_dsa",
            fused_lora_dyw_dsa,
            prepare_func,
        )
        
        # Tune fused_lora_dyw_dsa_tma (if available)
        try:
            logger.info("=" * 80)
            logger.info("TUNING FUSED_LORA_DYW_DSA_TMA")
            logger.info("=" * 80)
            
            results["fused_lora_dyw_dsa_tma"] = self.tune_kernel(
                "fused_lora_dyw_dsa_tma",
                fused_lora_dyw_dsa_tma,
                prepare_func,
            )
        except Exception as e:
            logger.warning(f"Could not tune fused_lora_dyw_dsa_tma: {e}")
        
        # Tune fused_lora_dys_dyb
        logger.info("=" * 80)
        logger.info("TUNING FUSED_LORA_DYS_DYB")
        logger.info("=" * 80)
        
        prepare_func = lambda: self.prepare_fused_lora_dys_dyb_inputs(
            m=m, n=n, r=r, alpha=alpha, dtype=dtype
        )
        
        results["fused_lora_dys_dyb"] = self.tune_kernel(
            "fused_lora_dys_dyb",
            fused_lora_dys_dyb,
            prepare_func,
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
                "throughput_gflops": result.throughput_gflops,
            }
        
        # Add metadata
        metadata = {
            "gpu_name": get_gpu_name(),
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
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


@click.command()
@click.option(
    "--m", type=int, default=4096, help="Batch size dimension."
)
@click.option(
    "--n", type=int, default=4096, help="Output dimension."
)
@click.option(
    "--k", type=int, default=4096, help="Input dimension."
)
@click.option(
    "--r", type=int, default=16, help="LoRA rank."
)
@click.option(
    "--alpha", type=float, default=16.0, help="LoRA alpha."
)
@click.option(
    "--dtype", type=str, default="bfloat16", help="Data type."
)
@click.option(
    "--warmup", type=int, default=100, help="Number of warmup iterations."
)
@click.option(
    "--number", type=int, default=50, help="Number of benchmark iterations."
)
@click.option(
    "--output", type=str, default="./tuning_results.json", help="Output file path."
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
) -> None:
    """Tune Triton kernel configurations."""
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    logger.info(f"Starting kernel tuning on {get_gpu_name()}")
    logger.info(f"Dimensions: M={m}, N={n}, K={k}, R={r}")
    logger.info(f"Data type: {dtype}")
    logger.info(f"Warmup: {warmup}, Number: {number}")
    
    tuner = KernelTuner(warmup=warmup, number=number)
    
    results = tuner.tune_all_kernels(
        m=m, n=n, k=k, r=r, alpha=alpha, dtype=torch_dtype
    )
    
    tuner.save_results(results, output)
    
    # Print summary
    logger.info("=" * 80)
    logger.info("TUNING SUMMARY")
    logger.info("=" * 80)
    
    for kernel_name, result in results.items():
        logger.info(f"{kernel_name}: {result.time_ms:.3f}ms")
        logger.info(f"  Config: {result.config}")


if __name__ == "__main__":
    main()
