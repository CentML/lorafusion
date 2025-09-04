# ruff: noqa: UP006, UP007, UP035, E501, SLF001, ERA001
"""Patch the transformers model."""

# This is based on transformers==4.51.1

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from loguru import logger
from torch import nn
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    FlashAttentionKwargs,
    eager_attention_forward,
)
from transformers.utils import logging as hf_logging

from lorafusion.utils.module import (
    COMMON_ATTENTION_CLASS_TYPES,
    COMMON_ATTENTION_CLASSES,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.processing_utils import Unpack

hf_logger = hf_logging.get_logger(__name__)

apply_rotary_pos_emb = liger_rotary_pos_emb


@torch.no_grad()
def align_linear_layer_attrs(original_linear: nn.Linear, new_linear: nn.Linear) -> None:
    """Align the attributes of the linear layer."""
    new_linear.weight.requires_grad = original_linear.weight.requires_grad
    new_linear.weight.data = new_linear.weight.data.to(original_linear.weight.device)
    new_linear.weight.data = new_linear.weight.data.to(original_linear.weight.dtype)
    if original_linear.bias is not None and new_linear.bias is not None:
        new_linear.bias.requires_grad = original_linear.bias.requires_grad
        new_linear.bias.data = new_linear.bias.data.to(original_linear.bias.device)
        new_linear.bias.data = new_linear.bias.data.to(original_linear.bias.dtype)


class QKVMergedCommonAttention(nn.Module):
    """QKV merged attention for common models."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        """Initialize the QKVMergedCommonAttention module."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Get attention_bias from config if available, otherwise default to True for models like Qwen
        if hasattr(config, "attention_bias"):
            attention_bias = config.attention_bias
        elif config.model_type in ("qwen2",):
            attention_bias = True
        else:
            msg = (
                f"Attention bias not found in config for model type {config.model_type}"
            )
            raise ValueError(msg)

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (config.num_attention_heads + 2 * config.num_key_value_heads)
            * self.head_dim,
            bias=attention_bias,
        )
        # Note: we don't reinitialize the o_proj, because we will do it in the
        # `from_original` method.
        # self.o_proj = nn.Linear(
        #     config.num_attention_heads * self.head_dim,
        #     config.hidden_size,
        #     bias=config.attention_bias,
        # )

        self.is_patched: bool | None = None
        self.original_class: type | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv_states = self.qkv_proj(hidden_states)

        q_size = self.config.num_attention_heads * self.head_dim
        kv_size = self.config.num_key_value_heads * self.head_dim

        query_states = qkv_states[..., :q_size].view(hidden_shape).transpose(1, 2)
        key_states = (
            qkv_states[..., q_size : q_size + kv_size]
            .view(hidden_shape)
            .transpose(1, 2)
        )
        value_states = (
            qkv_states[..., q_size + kv_size :].view(hidden_shape).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                hf_logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_original(
        cls, original_module: COMMON_ATTENTION_CLASS_TYPES
    ) -> QKVMergedCommonAttention:
        """Create a QKVMergedCommonAttention from a LlamaAttention module."""
        config = original_module.config
        layer_idx = original_module.layer_idx
        new_module = cls(config, layer_idx)

        # Update the dtype, device, and requires_grad of the new module

        # Copy the parameters into the new module
        with torch.no_grad():
            align_linear_layer_attrs(original_module.q_proj, new_module.qkv_proj)
            q_size = config.num_attention_heads * original_module.head_dim
            kv_size = config.num_key_value_heads * original_module.head_dim
            new_module.qkv_proj.weight[:q_size, :] = original_module.q_proj.weight
            new_module.qkv_proj.weight[q_size : q_size + kv_size, :] = (
                original_module.k_proj.weight
            )
            new_module.qkv_proj.weight[q_size + kv_size :, :] = (
                original_module.v_proj.weight
            )

            # Copy the bias if it exists
            if new_module.qkv_proj.bias is not None:
                if original_module.q_proj.bias is not None:
                    new_module.qkv_proj.bias[:q_size] = original_module.q_proj.bias
                if original_module.k_proj.bias is not None:
                    new_module.qkv_proj.bias[q_size : q_size + kv_size] = (
                        original_module.k_proj.bias
                    )
                if original_module.v_proj.bias is not None:
                    new_module.qkv_proj.bias[q_size + kv_size :] = (
                        original_module.v_proj.bias
                    )

            new_module.o_proj = original_module.o_proj

        new_module.is_patched = True
        new_module.original_class = original_module.__class__

        return new_module

    def to_original(self) -> COMMON_ATTENTION_CLASS_TYPES:
        """Convert the QKVMergedCommonAttention back to a LlamaAttention module."""
        original_module = self.original_class(self.config, self.layer_idx)

        with torch.no_grad():
            q_size = self.config.num_attention_heads * self.head_dim
            kv_size = self.config.num_key_value_heads * self.head_dim
            original_module.q_proj.weight[:q_size, :] = self.qkv_proj.weight[:q_size, :]
            original_module.k_proj.weight[q_size : q_size + kv_size, :] = (
                self.qkv_proj.weight[q_size : q_size + kv_size, :]
            )
            original_module.v_proj.weight[q_size + kv_size :, :] = self.qkv_proj.weight[
                q_size + kv_size :, :
            ]

            # Copy the bias back if it exists
            if self.qkv_proj.bias is not None:
                if (
                    hasattr(original_module.q_proj, "bias")
                    and original_module.q_proj.bias is not None
                ):
                    original_module.q_proj.bias[:] = self.qkv_proj.bias[:q_size]
                if (
                    hasattr(original_module.k_proj, "bias")
                    and original_module.k_proj.bias is not None
                ):
                    original_module.k_proj.bias[:] = self.qkv_proj.bias[
                        q_size : q_size + kv_size
                    ]
                if (
                    hasattr(original_module.v_proj, "bias")
                    and original_module.v_proj.bias is not None
                ):
                    original_module.v_proj.bias[:] = self.qkv_proj.bias[
                        q_size + kv_size :
                    ]

            original_module.o_proj.weight = self.o_proj.weight

        return original_module


def patch_module(
    model: nn.Module,
    from_type: type,
    transform: Callable,
    layer_indices: list[int] | None = None,
) -> None:
    """Patch the LlamaAttention module."""
    for name, module in model.named_modules():
        named_children = list(module.named_children())
        for submodule_name, submodule in named_children:
            if isinstance(submodule, from_type):
                # Skip if the layer index is not in the list of layer indices to patch
                full_name = f"{name}.{submodule_name}."
                if layer_indices is not None and all(
                    f".{layer_idx}." not in full_name for layer_idx in layer_indices
                ):
                    continue

                # Patch the submodule
                transformed = transform(submodule)
                setattr(module, submodule_name, transformed)
                logger.info(
                    f"Patched {name}.{submodule_name} from {from_type} to {type(transformed)}"
                )
    return model


def merge_qkv_proj(
    model: nn.Module, layer_indices: list[int] | None = None
) -> nn.Module:
    """Patch the attention modules to merge QKV projections."""
    return patch_module(
        model,
        COMMON_ATTENTION_CLASSES,
        QKVMergedCommonAttention.from_original,
        layer_indices=layer_indices,
    )
