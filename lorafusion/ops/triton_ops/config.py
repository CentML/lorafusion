"""Hardware-specific configurations for Triton kernels."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import triton


@dataclass
class TritonConfig:
    """Configuration for a Triton kernel."""

    block_size_m: int
    block_size_n: int
    block_size_k: int
    group_size_m: int
    num_stages: int
    num_warps: int
    # TMA-specific parameters
    epilogue_subtile: bool | None = None
    loop_unroll_factor: int | None = None
    flatten: bool | None = None

    def to_triton_config(self) -> triton.Config:
        """Convert to triton.Config object."""
        config_dict = {
            "BLOCK_SIZE_M": self.block_size_m,
            "BLOCK_SIZE_N": self.block_size_n,
            "BLOCK_SIZE_K": self.block_size_k,
            "GROUP_SIZE_M": self.group_size_m,
        }
        
        # Add TMA-specific parameters if they exist
        if self.epilogue_subtile is not None:
            config_dict["EPILOGUE_SUBTILE"] = self.epilogue_subtile
        if self.loop_unroll_factor is not None:
            config_dict["LOOP_UNROLL_FACTOR"] = self.loop_unroll_factor
        if self.flatten is not None:
            config_dict["FLATTEN"] = self.flatten
            
        return triton.Config(
            config_dict,
            num_stages=self.num_stages,
            num_warps=self.num_warps,
        )


@dataclass
class HardwareConfig:
    """Hardware-specific configuration for all kernels."""

    # Fused LoRA XW + SB configurations
    fused_lora_xw_sb: TritonConfig
    fused_lora_xw_sb_tma: TritonConfig
    
    # Fused LoRA DYW + DSA configurations
    fused_lora_dyw_dsa: TritonConfig
    fused_lora_dyw_dsa_tma: TritonConfig
    
    # Fused LoRA DYS + DYB configurations
    fused_lora_dys_dyb: TritonConfig


def get_gpu_name() -> str:
    """Get the GPU name for configuration lookup."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        # Normalize GPU names for consistent lookup
        gpu_name = gpu_name.lower()
        if "h100" in gpu_name:
            return "h100"
        elif "a100" in gpu_name:
            return "a100"
        elif "v100" in gpu_name:
            return "v100"
        elif "rtx" in gpu_name and "4090" in gpu_name:
            return "rtx4090"
        elif "rtx" in gpu_name and "4080" in gpu_name:
            return "rtx4080"
        elif "rtx" in gpu_name and "3090" in gpu_name:
            return "rtx3090"
        else:
            return "default"
    return "cpu"


# Hardware-specific configurations
# These are optimized configurations for different GPU architectures
HARDWARE_CONFIGS: dict[str, HardwareConfig] = {
    "h100": HardwareConfig(
        fused_lora_xw_sb=TritonConfig(
            block_size_m=128,
            block_size_n=256,
            block_size_k=64,
            group_size_m=8,
            num_stages=4,
            num_warps=8,
        ),
        fused_lora_xw_sb_tma=TritonConfig(
            block_size_m=128,
            block_size_n=256,
            block_size_k=64,
            group_size_m=8,
            num_stages=3,
            num_warps=8,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dyw_dsa=TritonConfig(
            block_size_m=128,
            block_size_n=256,
            block_size_k=64,
            group_size_m=8,
            num_stages=4,
            num_warps=8,
        ),
        fused_lora_dyw_dsa_tma=TritonConfig(
            block_size_m=128,
            block_size_n=256,
            block_size_k=64,
            group_size_m=8,
            num_stages=3,
            num_warps=8,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dys_dyb=TritonConfig(
            block_size_m=128,
            block_size_n=256,
            block_size_k=64,
            group_size_m=8,
            num_stages=4,
            num_warps=8,
        ),
    ),
    "a100": HardwareConfig(
        fused_lora_xw_sb=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=3,
            num_warps=4,
        ),
        fused_lora_xw_sb_tma=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dyw_dsa=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=3,
            num_warps=4,
        ),
        fused_lora_dyw_dsa_tma=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dys_dyb=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=3,
            num_warps=4,
        ),
    ),
    "rtx4090": HardwareConfig(
        fused_lora_xw_sb=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
        ),
        fused_lora_xw_sb_tma=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dyw_dsa=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
        ),
        fused_lora_dyw_dsa_tma=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dys_dyb=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
        ),
    ),
    "default": HardwareConfig(
        fused_lora_xw_sb=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
        ),
        fused_lora_xw_sb_tma=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dyw_dsa=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
        ),
        fused_lora_dyw_dsa_tma=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=False,
            loop_unroll_factor=None,
            flatten=False,
        ),
        fused_lora_dys_dyb=TritonConfig(
            block_size_m=64,
            block_size_n=128,
            block_size_k=32,
            group_size_m=8,
            num_stages=2,
            num_warps=4,
        ),
    ),
}


def get_hardware_config() -> HardwareConfig:
    """Get the hardware configuration for the current GPU."""
    gpu_name = get_gpu_name()
    if gpu_name not in HARDWARE_CONFIGS:
        gpu_name = "default"
    return HARDWARE_CONFIGS[gpu_name]


def get_kernel_config(kernel_name: str) -> TritonConfig:
    """Get the configuration for a specific kernel."""
    config = get_hardware_config()
    return getattr(config, kernel_name)


def get_kernel_configs(kernel_name: str) -> list[triton.Config]:
    """Get the list of configurations for a specific kernel (for autotune)."""
    # For now, return a single configuration
    # In the future, this could return multiple configurations for autotune
    config = get_kernel_config(kernel_name)
    return [config.to_triton_config()]


def override_config_from_env() -> None:
    """Override configurations from environment variables."""
    # Allow environment variable overrides
    # Format: LORAFUSION_CONFIG_<KERNEL>_<PARAM>=<VALUE>
    # Example: LORAFUSION_CONFIG_FUSED_LORA_XW_SB_BLOCK_SIZE_M=256
    
    gpu_name = get_gpu_name()
    if gpu_name not in HARDWARE_CONFIGS:
        return
    
    config = HARDWARE_CONFIGS[gpu_name]
    
    # Override configurations based on environment variables
    for kernel_name in ["fused_lora_xw_sb", "fused_lora_xw_sb_tma", 
                       "fused_lora_dyw_dsa", "fused_lora_dyw_dsa_tma", 
                       "fused_lora_dys_dyb"]:
        kernel_config = getattr(config, kernel_name)
        
        # Override block sizes
        for param in ["block_size_m", "block_size_n", "block_size_k"]:
            env_var = f"LORAFUSION_CONFIG_{kernel_name.upper()}_{param.upper()}"
            if env_var in os.environ:
                value = int(os.environ[env_var])
                setattr(kernel_config, param, value)
        
        # Override other parameters
        for param in ["group_size_m", "num_stages", "num_warps"]:
            env_var = f"LORAFUSION_CONFIG_{kernel_name.upper()}_{param.upper()}"
            if env_var in os.environ:
                value = int(os.environ[env_var])
                setattr(kernel_config, param, value)


# Initialize configurations on import
override_config_from_env()
