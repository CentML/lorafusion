"""Pipeline parallel model implementation."""

import itertools
from collections.abc import Callable
from typing import ClassVar

import torch
from peft import PeftModelForCausalLM
from torch import nn
from transformers import AutoConfig

from lorafusion.utils.module import COMMON_MODEL_CLASS_TYPES, COMMON_MODEL_CLASSES


class ShouldNotBeCalled(nn.Module):
    """A dummy module that raises an error when called.

    Used to replace components that should not be executed in the current
    pipeline stage.
    """

    def __init__(self, component_name: str | None = None) -> None:
        """Initialize the ShouldNotBeCalled module.

        Args:
            component_name: Name of the component that should not be called.
        """
        super().__init__()
        self.component_name = component_name

    def forward(self, *args, **kwargs) -> None:
        """Raises an error if this module is called."""
        component = f": {self.component_name}" if self.component_name else ""
        msg = (
            f"This component should not be called{component}. "
            f"Args: {args}, Kwargs: {kwargs}"
        )
        raise ValueError(msg)


def _process_common_model_for_pp(
    model: COMMON_MODEL_CLASS_TYPES, layer_idx_start: int, layer_idx_end: int
) -> nn.Module:
    """Process a LlamaForCausalLM model for pipeline parallelism.

    Args:
        model: The model to process.
        layer_idx_start: Starting layer index for this pipeline stage.
        layer_idx_end: Ending layer index for this pipeline stage.

    Returns:
        The processed model for the current pipeline stage.

    Raises:
        TypeError: If the model is not a LlamaForCausalLM or Qwen2ForCausalLM.
    """
    if not isinstance(model, COMMON_MODEL_CLASSES):
        msg = (
            f"Only {COMMON_MODEL_CLASSES} is supported for "
            f"_process_common_model_for_pp, "
            f"but got {type(model)}. Either choose a different function or "
            f"change the model to {COMMON_MODEL_CLASSES}."
        )
        raise TypeError(msg)

    is_first_stage = layer_idx_start == 0
    is_last_stage = layer_idx_end == model.config.num_hidden_layers

    # Check if the model has already been processed
    decoder = model.get_decoder()
    if getattr(decoder, "is_pp_processed", False):
        return model

    # Update the pre-layer
    if not is_first_stage:
        decoder.set_input_embeddings(ShouldNotBeCalled("input_embeddings"))

    # Update the main layers
    decoder.layers = decoder.layers[layer_idx_start:layer_idx_end]

    # Update the post-layer
    if not is_last_stage:
        decoder.norm = nn.Identity()
        model.lm_head = ShouldNotBeCalled("lm_head")

    # Mark the model as processed
    decoder.is_pp_processed = True
    model.is_pp_processed = True

    # Return the model if it is the last stage since we want to calculate
    # the loss
    if is_last_stage:
        return model

    return decoder


def even_layer_partition(
    num_layers: int, num_stages: int, extra_layer_limit: int = 0
) -> list[int]:
    """Evenly partition the layers into the specified number of stages.

    Args:
        num_layers: Total number of layers in the model.
        num_stages: Number of pipeline parallel stages.
        extra_layer_limit: Maximum number of extra layers to try.

    Returns:
        List of layer counts for each stage.

    Raises:
        ValueError: If no valid partitioning is found within the extra layer limit.
    """
    for extra_layers in range(extra_layer_limit + 1):
        total_layers = num_layers + extra_layers
        if total_layers % num_stages == 0:
            layer_partitions = [total_layers // num_stages] * num_stages
            layer_partitions[-1] = num_layers - sum(layer_partitions[:-1])
            return layer_partitions

    msg = (
        f"No valid layer partition found for {num_layers} layers and {num_stages} "
        f"stages with extra layer limit {extra_layer_limit}."
    )
    raise ValueError(msg)


def get_pipeline_stage_layer_indices(
    num_layers: int,
    num_stages: int,
    stage_idx: int,
    partition: list[int] | None = None,
    extra_layer_limit: int = 0,
) -> tuple[int, int]:
    """Get the layer indices for the current pipeline stage.

    Args:
        num_layers: Total number of layers in the model.
        num_stages: Number of pipeline parallel stages.
        stage_idx: Index of the current pipeline stage.
        partition: Optional list specifying how many layers to assign to each stage.
            If None, layers will be evenly distributed.
        extra_layer_limit: Maximum number of extra layers to try when creating
            an even partition.

    Returns:
        A tuple of (start_index, end_index) for the layers in this stage.
    """
    partition = partition or even_layer_partition(
        num_layers, num_stages, extra_layer_limit
    )
    layer_starts = [0] + list(itertools.accumulate(partition))[:-1]
    layer_ends = list(itertools.accumulate(partition))
    return layer_starts[stage_idx], layer_ends[stage_idx]


class PipeModel(nn.Module):
    """A module that implements pipeline parallel model execution."""

    processor_mapping: ClassVar[dict[str, Callable]] = {
        "llama": _process_common_model_for_pp,
        "qwen2": _process_common_model_for_pp,
    }

    def __init__(
        self,
        model: nn.Module,
        stage_idx: int,
        num_stages: int,
        *,
        is_lora: bool = False,
    ) -> None:
        """Initialize the pipeline parallel model.

        Args:
            model: The processed model for the current stage.
            stage_idx: The index of the current pipeline stage.
            num_stages: The total number of pipeline stages.
            is_lora: Whether the model is a PEFT model.
        """
        super().__init__()
        self.model = model
        # self.config is a TransformerConfig in Megatron Core
        # this will be set through `decorate_model_for_megatron_core`
        self.config = None
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.input_tensor = None
        self.is_lora = is_lora

    @property
    def hf_config(self) -> AutoConfig:
        """Get the Hugging Face configuration."""
        if self.is_lora:
            return self.model.base_model.model.config
        return self.model.config

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set the input tensor for the model.

        Args:
            input_tensor: The input tensor to set, or a list of input tensors.

        Raises:
            ValueError: If the input_tensor is neither None nor a torch.Tensor.
        """
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]
        if input_tensor is not None and not isinstance(input_tensor, torch.Tensor):
            msg = (
                f"Extracted `input_tensor` must be a torch.Tensor or None. "
                f"Stage: {self.stage_idx}, Num stages: {self.num_stages}. "
                f"Input tensor: {input_tensor}"
            )
            raise ValueError(msg)
        self.input_tensor = input_tensor

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the pipeline parallel model."""
        kwargs = self.check_and_update_inputs(*args, **kwargs)
        return self.model(**kwargs)

    def check_and_update_inputs(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Check and update the inputs if necessary.

        Returns:
            Updated kwargs dictionary with proper inputs for the model.

        Raises:
            ValueError: If inputs don't meet requirements for the current stage.
        """
        if len(args) > 1:
            msg = (
                f"Up to one positional argument is allowed, but got {len(args)} "
                f"positional arguments. "
                f"Stage: {self.stage_idx}, Num stages: {self.num_stages}. "
                f"Args: {args}, Kwargs: {kwargs}"
            )
            raise ValueError(msg)

        # For the first stage, either have args or have "input_ids"
        if self.is_first_stage:
            if len(args) == 0 and "input_ids" not in kwargs:
                msg = (
                    f"`input_ids` is required for the first stage. "
                    f"Stage: {self.stage_idx}, Num stages: {self.num_stages}. "
                    f"Args: {args}, Kwargs: {kwargs}"
                )
                raise ValueError(msg)
            input_ids = args[0] if len(args) > 0 else kwargs["input_ids"]
            if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long:
                msg = (
                    f"`input_ids` must be a torch.Tensor with dtype torch.long. "
                    f"Stage: {self.stage_idx}, Num stages: {self.num_stages}. "
                    f"Args: {args}, Kwargs: {kwargs}"
                )
                raise ValueError(msg)
            kwargs["input_ids"] = input_ids

        # For the other stage, disable args and only allow kwargs
        # and we should have input_tensor set
        if not self.is_first_stage:
            if self.input_tensor is None:
                msg = (
                    f"`input_tensor` is required for non-first stages. "
                    f"Stage: {self.stage_idx}, Num stages: {self.num_stages}. "
                )
                raise ValueError(msg)
            if "inputs_embeds" in kwargs:
                msg = (
                    f"`inputs_embeds` is not allowed for non-first stages. "
                    f"It should be set via `set_input_tensor`. "
                    f"Stage: {self.stage_idx}, Num stages: {self.num_stages}. "
                )
                raise ValueError(msg)
            kwargs["inputs_embeds"] = self.input_tensor

        # For the last stage, we should have labels if in training mode
        if self.is_last_stage and self.training and "labels" not in kwargs:
            msg = (
                f"`labels` is required for training in the last stage. "
                f"Stage: {self.stage_idx}, Num stages: {self.num_stages}. "
                f"Kwargs: {kwargs}"
            )
            raise ValueError(msg)

        return kwargs

    @property
    def is_first_stage(self) -> bool:
        """Check if the current stage is the first stage."""
        return self.stage_idx == 0

    @property
    def is_last_stage(self) -> bool:
        """Check if the current stage is the last stage."""
        return self.stage_idx == self.num_stages - 1

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        config: AutoConfig,
        stage_idx: int,
        num_stages: int,
        *,
        partition: list[int] | None = None,
    ) -> "PipeModel":
        """Create a pipeline parallel model from a pretrained model.

        Args:
            model: The model to process. Can be a regular model or a PEFT model.
            config: The model configuration.
            stage_idx: The index of the current pipeline stage.
            num_stages: The total number of pipeline stages.
            partition: Optional list specifying how many layers to assign to each stage.
                If None, layers will be evenly distributed.

        Returns:
            A PipeModel instance configured for the specified pipeline stage.

        Raises:
            ValueError: If the model type is not supported.
        """
        # Extract base model if using PEFT
        is_lora = isinstance(model, PeftModelForCausalLM)
        model_to_process = model.base_model.model if is_lora else model

        # Validate model type
        if config.model_type not in cls.processor_mapping:
            msg = (
                f"Model type {config.model_type} is not supported. "
                f"Supported model types: {list(cls.processor_mapping.keys())}"
            )
            raise ValueError(msg)

        # Calculate layer partitions if not provided
        layer_idx_start, layer_idx_end = get_pipeline_stage_layer_indices(
            config.num_hidden_layers,
            num_stages,
            stage_idx,
            partition,
            extra_layer_limit=0,
        )

        # Process the model for the current pipeline stage
        processed_model = cls.processor_mapping[config.model_type](
            model_to_process,
            layer_idx_start=layer_idx_start,
            layer_idx_end=layer_idx_end,
        )

        # Return appropriate model wrapper
        if is_lora:
            model.base_model.model = processed_model
            return cls(model, stage_idx, num_stages, is_lora=True)

        return cls(processed_model, stage_idx, num_stages, is_lora=False)
