#!/usr/bin/env python3
"""Compare old hardcoded configs with new configurable system."""

import torch
from lorafusion.ops.triton_ops.config import get_gpu_name, get_hardware_config, get_kernel_config


def compare_configurations():
    """Compare old hardcoded configurations with new configurable system."""
    print("=" * 80)
    print("KERNEL CONFIGURATION COMPARISON")
    print("=" * 80)
    
    gpu_name = get_gpu_name()
    print(f"Current GPU: {gpu_name}")
    
    # Get hardware config
    config = get_hardware_config()
    print(f"Selected hardware config: {type(config).__name__}")
    
    print("\n" + "=" * 80)
    print("KERNEL CONFIGURATIONS")
    print("=" * 80)
    
    kernels = [
        "fused_lora_xw_sb",
        "fused_lora_xw_sb_tma", 
        "fused_lora_dyw_dsa",
        "fused_lora_dyw_dsa_tma",
        "fused_lora_dys_dyb"
    ]
    
    for kernel_name in kernels:
        print(f"\n{kernel_name}:")
        print("-" * 40)
        
        try:
            kernel_config = get_kernel_config(kernel_name)
            
            print(f"  Block sizes: {kernel_config.block_size_m}x{kernel_config.block_size_n}x{kernel_config.block_size_k}")
            print(f"  Group size M: {kernel_config.group_size_m}")
            print(f"  Num stages: {kernel_config.num_stages}")
            print(f"  Num warps: {kernel_config.num_warps}")
            
            # Show TMA-specific parameters if they exist
            if kernel_config.epilogue_subtile is not None:
                print(f"  Epilogue subtile: {kernel_config.epilogue_subtile}")
            if kernel_config.loop_unroll_factor is not None:
                print(f"  Loop unroll factor: {kernel_config.loop_unroll_factor}")
            if kernel_config.flatten is not None:
                print(f"  Flatten: {kernel_config.flatten}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 80)
    print("CONFIGURATION BENEFITS")
    print("=" * 80)
    
    print("✅ Hardware-specific optimizations")
    print("✅ Environment variable overrides")
    print("✅ Easy tuning and benchmarking")
    print("✅ Consistent configuration management")
    print("✅ Support for TMA-specific parameters")
    print("✅ Fallback configurations for unknown hardware")
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("1. Use default configuration:")
    print("   from lorafusion.ops.triton_ops.fused_lora_xw_sb import fused_lora_xw_sb")
    print("   result = fused_lora_xw_sb(x, w, s, b, alpha, bias)")
    
    print("\n2. Override via environment variables:")
    print("   export LORAFUSION_CONFIG_FUSED_LORA_XW_SB_BLOCK_SIZE_M=256")
    print("   export LORAFUSION_CONFIG_FUSED_LORA_XW_SB_BLOCK_SIZE_N=512")
    
    print("\n3. Tune for your hardware:")
    print("   python tune_kernels.py --m 4096 --n 4096 --k 4096")
    
    print("\n4. Get configuration programmatically:")
    print("   from lorafusion.ops.triton_ops.config import get_kernel_config")
    print("   config = get_kernel_config('fused_lora_xw_sb')")
    print("   print(f'Block sizes: {config.block_size_m}x{config.block_size_n}x{config.block_size_k}')")


if __name__ == "__main__":
    compare_configurations()
