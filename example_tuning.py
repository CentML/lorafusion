#!/usr/bin/env python3
"""Example script showing how to use the kernel tuning system."""

import torch
from lorafusion.ops.triton_ops.config import get_gpu_name, get_hardware_config
from tune_kernels import KernelTuner


def example_tuning():
    """Example of how to tune kernels."""
    print(f"Running on GPU: {get_gpu_name()}")
    
    # Create a tuner with custom parameters
    tuner = KernelTuner(warmup=50, number=20)
    
    # Example: Tune a specific kernel with custom dimensions
    print("Tuning fused_lora_xw_sb with custom dimensions...")
    
    # Prepare inputs for a specific kernel
    prepare_func = lambda: tuner.prepare_fused_lora_xw_sb_inputs(
        m=2048,  # Smaller for faster tuning
        n=2048,
        k=2048,
        r=16,
        alpha=16.0,
        dtype=torch.bfloat16,
        with_bias=False
    )
    
    # Import the kernel function
    from lorafusion.ops.triton_ops.fused_lora_xw_sb import fused_lora_xw_sb
    
    # Tune the kernel
    result = tuner.tune_kernel(
        "fused_lora_xw_sb",
        fused_lora_xw_sb,
        prepare_func,
    )
    
    print(f"Best configuration: {result.config}")
    print(f"Best time: {result.time_ms:.3f}ms")
    
    # Save results
    tuner.save_results(
        {"fused_lora_xw_sb": result},
        "./example_tuning_results.json"
    )
    
    print("Tuning completed! Results saved to example_tuning_results.json")


if __name__ == "__main__":
    example_tuning()
