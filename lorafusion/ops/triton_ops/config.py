"""Hardware-specific configurations for Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass

import triton

from lorafusion.utils.common import get_device_short_name

KERNEL_SPILL_VERBOSE = False


@dataclass(frozen=True)
class LoRATritonConfig:
    """Configuration for a Triton kernel."""

    block_size_m: int
    block_size_n: int | None
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
            "BLOCK_SIZE_K": self.block_size_k,
            "GROUP_SIZE_M": self.group_size_m,
        }

        # Only add BLOCK_SIZE_N if it's not None
        if self.block_size_n is not None:
            config_dict["BLOCK_SIZE_N"] = self.block_size_n

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
    fused_lora_xw_sb: LoRATritonConfig
    fused_lora_xw_sb_tma: LoRATritonConfig | None

    # Fused LoRA DYW + DSA configurations
    fused_lora_dyw_dsa: LoRATritonConfig
    fused_lora_dyw_dsa_tma: LoRATritonConfig | None

    # Fused LoRA DYS + DYB configurations
    fused_lora_dys_dyb: LoRATritonConfig

    # Multi-LoRA configurations
    fused_multi_lora_block_size_m: int
    fused_multi_lora_xw_sb: LoRATritonConfig
    fused_multi_lora_dyw_dsa: LoRATritonConfig
    fused_multi_lora_dys_dyb: LoRATritonConfig


# Hardware-specific configurations
# These are optimized configurations for different GPU architectures
H100_CONFIG = HardwareConfig(
    fused_lora_xw_sb=LoRATritonConfig(128, 256, 64, 8, 4, 8),
    fused_lora_xw_sb_tma=LoRATritonConfig(128, 256, 64, 8, 3, 8),
    fused_lora_dyw_dsa=LoRATritonConfig(128, 256, 64, 8, 3, 8),
    fused_lora_dyw_dsa_tma=LoRATritonConfig(128, 256, 64, 8, 2, 8),
    fused_lora_dys_dyb=LoRATritonConfig(128, None, 128, 8, 5, 8),
    fused_multi_lora_block_size_m=128,
    fused_multi_lora_xw_sb=LoRATritonConfig(128, 256, 64, 8, 4, 8),
    fused_multi_lora_dyw_dsa=LoRATritonConfig(128, 256, 64, 8, 4, 8),
    fused_multi_lora_dys_dyb=LoRATritonConfig(128, None, 128, 8, 5, 8),
)

A100_80GB_PCIE_CONFIG = HardwareConfig(
    fused_lora_xw_sb=LoRATritonConfig(128, 128, 32, 8, 4, 4),
    fused_lora_xw_sb_tma=None,
    fused_lora_dyw_dsa=LoRATritonConfig(128, 128, 64, 8, 4, 8),
    fused_lora_dyw_dsa_tma=None,
    fused_lora_dys_dyb=LoRATritonConfig(128, None, 128, 8, 5, 8),
    fused_multi_lora_block_size_m=128,
    fused_multi_lora_xw_sb=LoRATritonConfig(128, 128, 64, 8, 4, 8),
    fused_multi_lora_dyw_dsa=LoRATritonConfig(128, 128, 32, 8, 4, 4),
    fused_multi_lora_dys_dyb=LoRATritonConfig(128, None, 128, 8, 5, 8),
)

A100_SXM4_80GB_CONFIG = HardwareConfig(
    fused_lora_xw_sb=LoRATritonConfig(128, 128, 32, 8, 4, 4),
    fused_lora_xw_sb_tma=None,
    fused_lora_dyw_dsa=LoRATritonConfig(128, 256, 64, 8, 3, 8),
    fused_lora_dyw_dsa_tma=None,
    fused_lora_dys_dyb=LoRATritonConfig(128, None, 128, 8, 3, 8),
    fused_multi_lora_block_size_m=128,
    fused_multi_lora_xw_sb=LoRATritonConfig(128, 256, 64, 8, 3, 8),
    fused_multi_lora_dyw_dsa=LoRATritonConfig(128, 128, 32, 8, 4, 4),
    fused_multi_lora_dys_dyb=LoRATritonConfig(128, None, 128, 8, 4, 8),
)

RTX3090_CONFIG = HardwareConfig(
    fused_lora_xw_sb=LoRATritonConfig(64, 128, 32, 8, 4, 4),
    fused_lora_xw_sb_tma=None,
    fused_lora_dyw_dsa=LoRATritonConfig(64, 128, 32, 8, 4, 4),
    fused_lora_dyw_dsa_tma=None,
    fused_lora_dys_dyb=LoRATritonConfig(128, None, 128, 8, 4, 8),
    fused_multi_lora_block_size_m=64,
    fused_multi_lora_xw_sb=LoRATritonConfig(64, 128, 32, 8, 4, 4),
    fused_multi_lora_dyw_dsa=LoRATritonConfig(64, 128, 32, 8, 4, 4),
    fused_multi_lora_dys_dyb=LoRATritonConfig(64, None, 128, 8, 3, 8),
)

HARDWARE_CONFIGS: dict[str, HardwareConfig] = {
    "h100-80gb-hbm3": H100_CONFIG,
    "a100-80gb-pcie": A100_80GB_PCIE_CONFIG,
    "geforce-rtx-3090": RTX3090_CONFIG,
}


def get_hardware_config() -> HardwareConfig:
    """Get the hardware configuration for the current GPU."""
    gpu_name = get_device_short_name()
    if gpu_name not in HARDWARE_CONFIGS:
        raise ValueError(
            f"GPU {gpu_name} not supported. Supported GPUs: {HARDWARE_CONFIGS.keys()}. "
            f"Please run tools/tune_kernels.py to tune the kernel configurations for "
            f"your GPU, and then update the lorafusion/ops/triton_ops/config.py file "
            f"with the tuned configurations."
        )
    return HARDWARE_CONFIGS[gpu_name]


def get_lora_kernel_config(kernel_name: str) -> LoRATritonConfig | int | None:
    """Get the configuration for a specific kernel."""
    config = get_hardware_config()
    return getattr(config, kernel_name)
