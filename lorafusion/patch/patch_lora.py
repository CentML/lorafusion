"""Patch the LoRA model."""

import torch
from peft.tuners.lora import Linear as LoRALinear

from lorafusion.ops.lora_torch_ref import torch_ref_linear_lora, torch_ref_linear_multi_lora
from lorafusion.ops.lora_v1 import fused_linear_lora
from lorafusion.ops.multi_lora import fused_linear_multi_lora, get_multi_lora_manager


def lora_forward(  # noqa: PLR0912, PLR0915
    model: LoRALinear, x: torch.Tensor, *args, use_fused: bool = True, **kwargs
) -> torch.Tensor:
    """LoRA forward.

    This function handles both single-LoRA and multi-LoRA cases based on the
    MultiLoRAManager configuration. Uses either fused kernels or standard PyTorch
    operations based on the apply_fused_lora flag.

    Args:
        model: The LoRALinear model.
        x: The input tensor.
        *args: Additional arguments.
        use_fused: Whether to use fused kernels or standard PyTorch operations.
        **kwargs: Additional keyword arguments.

    Returns:
        The output tensor after applying LoRA.
    """
    self = model
    if not isinstance(self, LoRALinear):
        msg = f"model must be a LoRALinear, but got {type(self)}"
        raise TypeError(msg)

    # Get the multi-LoRA manager
    multi_lora_manager = get_multi_lora_manager()

    # Get result dtype to ensure consistent output type
    torch_result_dtype = x.dtype

    # Get bias if available
    linear_bias = self.base_layer.bias if hasattr(self.base_layer, "bias") else None

    if multi_lora_manager is None:
        # Use single-LoRA path if multi-LoRA is not enabled
        if len(self.lora_A) > 1:
            msg = "Multi-LoRA is not enabled, but the model has multiple LoRA adapters."
            raise ValueError(msg)

        active_adapter = next(iter(self.lora_A.keys()))
        lora_A = self.lora_A[active_adapter]  # noqa: N806
        lora_B = self.lora_B[active_adapter]  # noqa: N806
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x = self._cast_input_dtype(x, lora_A.weight.dtype)

        if use_fused:
            result = fused_linear_lora(
                x=x,
                linear_w=self.base_layer.weight,
                lora_a=lora_A.weight,
                lora_b=lora_B.weight,
                alpha=scaling,
                dropout_p=dropout.p if self.training else 0.0,
                linear_bias=linear_bias,
            )
        else:
            result = torch_ref_linear_lora(
                x=x,
                linear_w=self.base_layer.weight,
                lora_a=lora_A.weight,
                lora_b=lora_B.weight,
                alpha=scaling,
                dropout_p=dropout.p if self.training else 0.0,
                linear_bias=linear_bias,
            )
    else:
        # Register this layer with the manager if not already registered
        multi_lora_manager.register_lora_linear(self)
        is_in_forward = multi_lora_manager.is_in_forward

        # Get the appropriate batch info based on whether we're in forward or backward
        if is_in_forward:
            # Real forward pass
            multi_lora_batch_info = multi_lora_manager.get_newest_batch_info()
        else:
            # Backward pass or recomputation
            multi_lora_batch_info = multi_lora_manager.get_oldest_batch_info()

        num_active_adapters = multi_lora_batch_info.num_active_adapters
        if num_active_adapters == 1:
            # Use single-LoRA path
            active_adapter_idx = multi_lora_batch_info.lora_idx_list[0]
            adapter_key = f"lora_{active_adapter_idx}"
            lora_A = self.lora_A[adapter_key]  # noqa: N806
            lora_B = self.lora_B[adapter_key]  # noqa: N806
            dropout = self.lora_dropout[adapter_key]
            scaling = self.scaling[adapter_key]
            x = self._cast_input_dtype(x, lora_A.weight.dtype)

            if use_fused:
                result = fused_linear_lora(
                    x=x,
                    linear_w=self.base_layer.weight,
                    lora_a=lora_A.weight,
                    lora_b=lora_B.weight,
                    alpha=scaling,
                    dropout_p=dropout.p if self.training else 0.0,
                    linear_bias=linear_bias,
                )
            else:
                result = torch_ref_linear_lora(
                    x=x,
                    linear_w=self.base_layer.weight,
                    lora_a=lora_A.weight,
                    lora_b=lora_B.weight,
                    alpha=scaling,
                    dropout_p=dropout.p if self.training else 0.0,
                    linear_bias=linear_bias,
                )
        else:
            # Construct the lora_a_list and lora_b_list
            lora_a_list, lora_b_list = [], []
            for adapter_idx in multi_lora_batch_info.lora_idx_list:
                adapter_key = f"lora_{adapter_idx}"
                lora_a_list.append(self.lora_A[adapter_key].weight)
                lora_b_list.append(self.lora_B[adapter_key].weight)

            if use_fused:
                # Use multi-LoRA path with fused kernels
                result = fused_linear_multi_lora(
                    padded_x=x,
                    linear_w=self.base_layer.weight,
                    lora_a_list=lora_a_list,
                    lora_b_list=lora_b_list,
                    seq_len_list=multi_lora_batch_info.seq_len_list,
                    padded_seq_len_list=multi_lora_batch_info.padded_seq_len_list,
                    block_to_lookup_table=multi_lora_batch_info.block_to_lookup_table,
                    block_to_dropout_p=multi_lora_batch_info.block_to_dropout_p,
                    block_to_alpha=multi_lora_batch_info.block_to_alpha,
                    enable_dropout=self.training,
                    same_dropout_p_value=multi_lora_batch_info.same_dropout_p_value,
                    max_r=multi_lora_batch_info.max_r,
                    linear_bias=linear_bias,
                )
            else:
                # Use multi-LoRA path with standard PyTorch operations
                result = torch_ref_linear_multi_lora(
                    padded_x=x,
                    linear_w=self.base_layer.weight,
                    lora_a_list=lora_a_list,
                    lora_b_list=lora_b_list,
                    seq_len_list=multi_lora_batch_info.seq_len_list,
                    padded_seq_len_list=multi_lora_batch_info.padded_seq_len_list,
                    alpha_list=multi_lora_batch_info.alpha_list,
                    dropout_p_list=[
                        p if self.training else 0.0
                        for p in multi_lora_batch_info.dropout_p_list
                    ],
                    linear_bias=linear_bias,
                )

    return result.to(torch_result_dtype)


def apply_lora(*, use_fused: bool = True) -> None:
    """Apply LoRA.

    Args:
        use_fused: Whether to use fused kernels or standard PyTorch operations.
    """
    # To avoid the "missing 1 required positional argument: 'x'" error,
    # we need to be careful with how we use partial.
    # The original forward function signature from peft is:
    # def forward(self, x: torch.Tensor, *args, **kwargs)

    # Create a wrapper function that preserves the original signature
    def forward_wrapper(
        self: LoRALinear, x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return lora_forward(self, x, *args, use_fused=use_fused, **kwargs)

    # Replace the forward method with our wrapper
    LoRALinear.forward = forward_wrapper
