"""Simulator for LoRA models.

Note:
1. LoRA modules are always of torch.bfloat16 dtype.
2. The base model can be one of the following:
    - torch.bfloat16 (default)
    - int8
    - int4
"""

from __future__ import annotations

import csv
import math
from copy import copy
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from lorafusion.simulator import config_dir
from lorafusion.simulator.object import ConfigurableObject
from lorafusion.utils.benchmark import benchmark, set_warmup_and_number
from lorafusion.utils.common import (
    get_dtype_element_size,
    is_array_consistent,
    str_to_torch_dtype,
)
from lorafusion.utils.module import (
    apply_liger_kernel,
    count_trainable_and_all_parameters,
)

if TYPE_CHECKING:
    from collections.abc import Callable

apply_liger_kernel()

LORA_DTYPE = torch.bfloat16


def get_decoder_layers_from_hf_causal_lm(
    model: AutoModelForCausalLM | PeftModelForCausalLM, *, is_lora: bool = False
) -> nn.ModuleList:
    """Get the decoder layers from a Hugging Face causal language model.

    Args:
        model: The Hugging Face causal language model.
        is_lora: Whether the model is a LoRA model.

    Returns:
        The decoder layers.
    """
    if is_lora:
        if not isinstance(model, PeftModelForCausalLM):
            msg = f"The model is not a LoRA model. Model type: {type(model)}"
            raise ValueError(msg)
        return model.base_model.model.model.layers

    if "ForCausalLM" not in model.__class__.__name__:
        msg = f"The model is not a causal language model. Model type: {type(model)}"
        raise ValueError(msg)

    return model.model.layers


def set_decoder_layers_to_hf_causal_lm(
    model: AutoModelForCausalLM | PeftModelForCausalLM,
    decoder_layers: nn.ModuleList,
    *,
    is_lora: bool = False,
) -> AutoModelForCausalLM | PeftModelForCausalLM:
    """Set the decoder layers to a Hugging Face causal language model.

    Args:
        model: The Hugging Face causal language model.
        decoder_layers: The decoder layers to set.
        is_lora: Whether the model is a LoRA model.

    Returns:
        The Hugging Face causal language model with the decoder layers set.
    """
    if is_lora:
        if not isinstance(model, PeftModelForCausalLM):
            msg = f"The model is not a LoRA model. Model type: {type(model)}"
            raise ValueError(msg)
        model.base_model.model.model.layers = decoder_layers
    else:
        if "ForCausalLM" not in model.__class__.__name__:
            msg = f"The model is not a causal language model. Model type: {type(model)}"
            raise ValueError(msg)
        model.model.layers = decoder_layers
    return model


def copy_causal_lm_with_k_layers(
    model: AutoModelForCausalLM,
    *,
    num_layers: int = -1,
    is_lora: bool = False,
) -> AutoModelForCausalLM:
    """Copy a causal language model with k layers.

    Args:
        model: The causal language model to copy.
        num_layers: The number of layers to copy.
        is_lora: Whether the model is a LoRA model.

    Returns:
        The copied causal language model.
    """
    output_model = copy(model)

    if num_layers == -1:
        return output_model

    output_model_decoder_layers = get_decoder_layers_from_hf_causal_lm(
        output_model, is_lora=is_lora
    )
    curr_num_layers = len(output_model_decoder_layers)
    if curr_num_layers < num_layers:
        msg = (
            f"The number of layers to copy ({num_layers}) is greater than the "
            f"number of layers in the model ({curr_num_layers})."
        )
        raise ValueError(msg)
    set_decoder_layers_to_hf_causal_lm(
        output_model, output_model_decoder_layers[:num_layers], is_lora=is_lora
    )
    return output_model


def count_saved_activations_with_peak_intermediate(
    model: nn.Module,
    prepare_func: Callable,
    batch_size: int,
    seq_len: int,
    *,
    disable_lm_head: bool = False,
) -> tuple[int, int, int]:
    """Count the number of saved activations in the model.

    Args:
        model: The model.
        prepare_func: The prepare function.
        batch_size: The batch size.
        seq_len: The sequence length.
        disable_lm_head: Whether to disable the LM head. This is used to count the
            intermediate activations for pipeline parallelism.

    Returns:
        A tuple containing:
        - Number of saved activations
        - Peak memory usage during computation
        - Absolute peak memory usage
    """
    torch.cuda.reset_peak_memory_stats()
    inputs = prepare_func(
        batch_size=batch_size,
        seq_len=seq_len,
        ignore_labels=disable_lm_head,
    )
    start_memory_usage = torch.cuda.memory_allocated()
    loss = model(**inputs)[0]
    fwd_end_memory_usage = torch.cuda.memory_allocated()
    loss.sum().backward()
    peak_memory_usage = torch.cuda.max_memory_allocated()

    # Calculate the number of saved activations
    saved_activations_decoder_layer = fwd_end_memory_usage - start_memory_usage
    return (
        saved_activations_decoder_layer,
        peak_memory_usage - start_memory_usage,
        peak_memory_usage,
    )


def prepare_func(
    batch_size: int,
    seq_len: int,
    *,
    ignore_labels: bool = False,
) -> dict[str, torch.Tensor]:
    """Prepare the input for the forward pass.

    Args:
        batch_size: The batch size.
        seq_len: The sequence length.
        ignore_labels: Whether to ignore the labels. This is used to count the
            intermediate activations for pipeline parallelism.

    Returns:
        Dictionary containing all inputs needed for model forward pass.
    """
    # Prepare input_ids and labels
    input_ids = torch.randint(
        0,
        10,
        (1, batch_size * seq_len),
        device="cuda",
        dtype=torch.long,
    )
    labels = torch.randint(
        0,
        10,
        (1, batch_size * seq_len),
        device="cuda",
        dtype=torch.long,
    )

    # Create position_ids, e.g. Tensor[0, 1, 2, ..., N_1, 0, 1, 2, ..., N_2, ...]
    position_ids = torch.arange(
        seq_len,
        device="cuda",
        dtype=torch.int32,
    )
    position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
    position_ids = position_ids.flatten().unsqueeze(0)

    # Calculate cumulative sequence lengths for queries and keys
    flattened_position_ids = position_ids.flatten()
    indices_q = torch.arange(
        flattened_position_ids.shape[0],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seq_lens_q = torch.cat(
        (
            indices_q[flattened_position_ids == 0],
            torch.tensor(
                flattened_position_ids.size(),
                device=flattened_position_ids.device,
                dtype=torch.int32,
            ),
        )
    )
    max_length_q = flattened_position_ids.max().item() + 1

    outputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "cu_seq_lens_q": cu_seq_lens_q,
        "cu_seq_lens_k": cu_seq_lens_q,
        "max_length_q": max_length_q,
        "max_length_k": max_length_q,
    }
    if not ignore_labels:
        outputs["labels"] = labels
    return outputs


def bwd_func(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
) -> torch.Tensor:
    """Perform forward and backward pass.

    Args:
        model: The model to run.
        input_ids: Input token IDs.
        labels: Target labels.
        position_ids: Position IDs.
        cu_seq_lens_q: Cumulative sequence lengths for queries.
        cu_seq_lens_k: Cumulative sequence lengths for keys.
        max_length_q: Maximum query length.
        max_length_k: Maximum key length.

    Returns:
        The loss tensor.
    """
    loss = model(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_length_q=max_length_q,
        max_length_k=max_length_k,
    ).loss
    loss.backward()
    return loss


@dataclass
class LoRALLMProfileResult(ConfigurableObject):
    """Profiled result."""

    # Identifier
    name: str | None = field(
        default=None,
        metadata={
            "description": "Name of the module, e.g. meta-llama/Llama-3.1-8B-Instruct."
        },
    )
    base_model_dtype: str | None = field(
        default="bfloat16", metadata={"description": "Base model dtype"}
    )
    # Constants
    num_layers: int | None = field(
        default=None, metadata={"description": "Number of layers"}
    )
    hidden_size: int | None = field(
        default=None, metadata={"description": "Hidden size"}
    )
    # Frozen parameters
    decoder_layer_param_count: int | None = field(
        default=None, metadata={"description": "Number of decoder layer parameters"}
    )
    pre_layer_param_count: int | None = field(
        default=None, metadata={"description": "Number of pre-layer parameters"}
    )
    post_layer_param_count: int | None = field(
        default=None, metadata={"description": "Number of post-layer parameters"}
    )
    # Trainable parameters
    decoder_layer_lora_param_count_per_rank: int | None = field(
        default=None,
        metadata={"description": "Number of decoder layer LoRA parameters per rank"},
    )
    # Saved activation
    decoder_layer_saved_activations_no_ckpt_count: float | None = field(
        default=None,
        metadata={
            "description": (
                "Number of decoder layer saved activations (no ckpt). "
                "It can be float because it is calculated by division."
            )
        },
    )
    decoder_layer_saved_activations_with_ckpt_count: float | None = field(
        default=None,
        metadata={
            "description": (
                "Number of decoder layer saved activations (with ckpt). "
                "It can be float because it is calculated by division."
            )
        },
    )
    # Extra Peak Memory
    peak_intermediate_activation_count: float | None = field(
        default=None,
        metadata={
            "description": (
                "Number of model peak intermediate activations. "
                "It can be float because it is calculated by division."
            )
        },
    )
    peak_intermediate_activation_count_without_lm_head: float | None = field(
        default=None,
        metadata={
            "description": (
                "Number of model peak intermediate activations (without LM head). "
            )
        },
    )
    # Execution time
    decoder_layer_fwd_time_by_tokens: dict[int, float] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Forward time of the decoder layer. "
                "Key is the number of tokens and value is the time in seconds."
            )
        },
    )
    decoder_layer_fwd_bwd_time_by_tokens: dict[int, float] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Forward and backward time of the decoder layer. "
                "Key is the number of tokens and value is the time in seconds."
            )
        },
    )
    pre_post_layer_fwd_bwd_time_by_tokens: dict[int, float] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Forward and backward time of the pre- and post-layers. "
                "Key is the number of tokens and value is the time in seconds."
            )
        },
    )
    # Communication Count
    pre_layer_tp_comm_count: int | None = field(
        default=None, metadata={"description": "Number of pre-layer TP communication"}
    )
    post_layer_tp_comm_count: int | None = field(
        default=None, metadata={"description": "Number of post-layer TP communication"}
    )
    # For Verification
    _peak_memory_usage_for_verification: list[int] | None = field(
        default=None,
        metadata={
            "description": (
                "Peak memory usage for verification. "
                "It is a list of integers. The first element is the peak memory usage "
                "of the first num_layers choices and the last element is the peak "
                "memory usage of the last num_layers choices."
            )
        },
    )
    _num_layers_for_verification: list[int] | None = field(
        default=None,
        metadata={
            "description": (
                "Number of layers for verification. "
                "It is a list of integers. The first element is the number of layers "
                "for the first num_layers choices and the last element is the number "
                "of layers for the last num_layers choices."
            )
        },
    )
    _batch_size_for_verification: int | None = field(
        default=None, metadata={"description": "Batch size for verification"}
    )

    @property
    def short_name(self) -> str:
        """Short name of the module.

        Example:
            >>> name = "meta-llama/Llama-3.1-8B-Instruct"
            >>> short_name = "llama-3.1-8b-instruct"

        Returns:
            The short name of the module.
        """
        if not self.name:
            return ""
        return self.name.lower().rpartition("/")[-1]

    @property
    def default_yaml_path(self) -> Path:
        """Default YAML path for the model configuration."""
        return config_dir / f"{self.short_name}.yaml"

    def save(self, yaml_path: str | Path | None = None) -> None:
        """Save the result to a YAML file.

        Args:
            yaml_path: Optional custom path to save the YAML file.
                       If None, uses the default_yaml_path.
        """
        yaml_path = Path(yaml_path) if yaml_path else self.default_yaml_path
        self.to_yaml(yaml_path)
        logger.info(f"Profile results saved to {yaml_path}.")

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: str,
        base_model_dtype: str = "bfloat16",
        *,
        profile_if_not_exists: bool = True,
        override_if_exists: bool = False,
        yaml_path: str | Path | None = None,
    ) -> LoRALLMProfileResult:
        """Load the result from a YAML file or create a new profile if needed.

        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model.
            base_model_dtype: The dtype of the base model.
            profile_if_not_exists: Whether to profile the model if the YAML file
                doesn't exist.
            override_if_exists: Whether to override the YAML file if it exists.
            yaml_path: Optional custom path to the YAML file.

        Returns:
            The loaded or newly created profile result.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist and profile_if_not_exists
                is False.
        """
        # Extract short name from model path
        short_name = pretrained_model_name_or_path.lower().rpartition("/")[-1]

        # Determine the YAML path
        if yaml_path is None:
            yaml_path = config_dir / f"{short_name}.yaml"
        else:
            yaml_path = Path(yaml_path)

        if (not yaml_path.exists() and profile_if_not_exists) or override_if_exists:
            logger.info(f"Launching profiler for {pretrained_model_name_or_path}...")
            profiler = LoRALLMProfiler(pretrained_model_name_or_path, base_model_dtype)
            result = profiler.profile()
            result.save(yaml_path)
            return result

        if yaml_path.exists():
            logger.info(f"Loading profile from {yaml_path} ...")
            return cls.from_yaml(yaml_path)

        msg = f"Profile result not found: {yaml_path}."
        raise FileNotFoundError(msg)


class LoRALLMProfiler:
    """Profiler for LoRA LLM."""

    BASE_SEQ_LEN = 512
    NUM_DECODER_LAYERS_LOWER_FOR_MEM_PROFILE = 1
    NUM_DECODER_LAYERS_UPPER_FOR_MEM_PROFILE = 4
    LAYER_RANGES_FOR_MEM_PROFILE = (
        NUM_DECODER_LAYERS_LOWER_FOR_MEM_PROFILE,
        NUM_DECODER_LAYERS_UPPER_FOR_MEM_PROFILE + 1,
    )
    BATCH_SIZE_LOWER_FOR_MEM_PROFILE = 1
    BATCH_SIZE_UPPER_FOR_MEM_PROFILE = 2
    BATCH_SIZES_FOR_TIME_PROFILE = (1, 2, 4, 8, 16)
    REPEAT_COUNT_FOR_MEM_PROFILE = 2
    DUMMY_LORA_CONFIG = LoraConfig(
        r=16,
        lora_alpha=16.0,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        base_model_dtype: str = "bfloat16",
    ) -> None:
        """Initialize the profiler.

        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model.
            base_model_dtype: The dtype of the base model.
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_model_dtype = base_model_dtype

        # Load config
        ori_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        self.ori_config = ori_config

        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            num_hidden_layers=self.NUM_DECODER_LAYERS_UPPER_FOR_MEM_PROFILE,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=str_to_torch_dtype(self.base_model_dtype),
        )
        self.model.cuda()
        self.pre_layer = self.model.get_input_embeddings()
        self.post_layer = self.model.get_output_embeddings()

        # Get decoder layers
        self.decoder_layers = get_decoder_layers_from_hf_causal_lm(
            self.model, is_lora=False
        )
        if len(self.decoder_layers) != self.NUM_DECODER_LAYERS_UPPER_FOR_MEM_PROFILE:
            msg = (
                f"Expected {self.NUM_DECODER_LAYERS_UPPER_FOR_MEM_PROFILE} decoder "
                f"layers, but found {len(self.decoder_layers)}"
            )
            raise ValueError(msg)

        # Create the LoRA model
        self.lora_model = get_peft_model(self.model, self.DUMMY_LORA_CONFIG)
        self.lora_decoder_layers = get_decoder_layers_from_hf_causal_lm(
            self.lora_model, is_lora=True
        )
        self.lora_model.to(str_to_torch_dtype(self.base_model_dtype))

    def _process_frozen_param_counts(self) -> dict[str, int]:
        """Count frozen parameters in different parts of the model.

        Returns:
            Dictionary with parameter counts for decoder, pre, and post layers.
        """
        decoder_layer_param_count = count_trainable_and_all_parameters(
            self.decoder_layers[0]
        )[1]
        pre_layer_param_count = count_trainable_and_all_parameters(self.pre_layer)[1]
        post_layer_param_count = count_trainable_and_all_parameters(self.post_layer)[1]

        return {
            "decoder_layer_param_count": decoder_layer_param_count,
            "pre_layer_param_count": pre_layer_param_count,
            "post_layer_param_count": post_layer_param_count,
        }

    def _process_trainable_param_counts(self) -> dict[str, int]:
        """Count trainable parameters in the LoRA model.

        Returns:
            Dictionary with trainable parameter counts.
        """
        decoder_layer_lora_param_count_per_rank = count_trainable_and_all_parameters(
            self.lora_decoder_layers[0]
        )[0]

        return {
            "decoder_layer_lora_param_count_per_rank": (
                decoder_layer_lora_param_count_per_rank
            ),
        }

    def _process_saved_activations_and_peak_intermediate(self) -> dict[str, float]:
        """Process saved activations and peak intermediate activations.

        Returns:
            Dictionary mapping component names to their memory usage per token.
        """
        # Collect activation measurements
        saved_activations, peak_intermediate_activations, peak_memory_usage = (
            self._collect_activation_measurements()
        )

        # Validate measurements for consistency
        saved_activations_per_token, peak_intermediate_activations_per_token = (
            self._validate_and_calculate_per_token_metrics(
                saved_activations, peak_intermediate_activations
            )
        )

        # Consider the case without lm_head
        saved_activations_no_lm_head, peak_intermediate_activations_no_lm_head, _ = (
            self._collect_activation_measurements(disable_lm_head=True)
        )
        _, peak_intermediate_activations_no_lm_head_per_token = (
            self._validate_and_calculate_per_token_metrics(
                saved_activations_no_lm_head, peak_intermediate_activations_no_lm_head
            )
        )

        return {
            "decoder_layer_saved_activations_no_ckpt_count": (
                saved_activations_per_token
            ),
            "decoder_layer_saved_activations_with_ckpt_count": (
                self.model.config.hidden_size
            ),
            "peak_intermediate_activation_count": (
                peak_intermediate_activations_per_token
            ),
            "peak_intermediate_activation_count_without_lm_head": (
                peak_intermediate_activations_no_lm_head_per_token
            ),
            "_peak_memory_usage_for_verification": peak_memory_usage,
            "_num_layers_for_verification": list(
                range(*self.LAYER_RANGES_FOR_MEM_PROFILE)
            ),
            "_batch_size_for_verification": self.BATCH_SIZE_LOWER_FOR_MEM_PROFILE,
        }

    def _collect_activation_measurements(
        self,
        *,
        disable_lm_head: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        """Collect activation measurements across multiple runs.

        Returns:
            Tuple containing:
            - List of saved activations
            - List of peak intermediate activations
            - List of peak memory usage values
        """
        lora_model = copy(self.lora_model)
        saved_activations = []
        peak_intermediate_activations = []
        peak_memory_usage = []

        for i in range(1 + self.REPEAT_COUNT_FOR_MEM_PROFILE):
            raw_activations_list = []
            raw_peak_memory_increase_list = []
            # We only need to keep the latest peak memory usage
            peak_memory_usage = []

            for n_layers in range(*self.LAYER_RANGES_FOR_MEM_PROFILE):
                # Configure model with specific number of layers
                set_decoder_layers_to_hf_causal_lm(
                    lora_model,
                    self.lora_decoder_layers[:n_layers],
                    is_lora=True,
                )

                # Measure activations
                _activations, _peak_memory_increase, _peak = (
                    count_saved_activations_with_peak_intermediate(
                        lora_model,
                        prepare_func,
                        batch_size=self.BATCH_SIZE_LOWER_FOR_MEM_PROFILE,
                        seq_len=self.BASE_SEQ_LEN,
                        disable_lm_head=disable_lm_head,
                    )
                )
                logger.debug(
                    f"{disable_lm_head=}, "
                    f"{n_layers=}, {_activations=}, {_peak_memory_increase=}"
                )

                raw_activations_list.append(_activations)
                raw_peak_memory_increase_list.append(_peak_memory_increase)
                peak_memory_usage.append(_peak)

            # Calculate the difference between consecutive measurements
            current_saved_activations = [
                y - x for x, y in pairwise(raw_activations_list)
            ]
            mean_saved_activations = np.median(current_saved_activations).item()

            # Calculate the peak intermediate activations
            current_peak_intermediate_activations = [
                peak_memory - mean_saved_activations * num_layers
                for num_layers, peak_memory in zip(
                    range(*self.LAYER_RANGES_FOR_MEM_PROFILE),
                    raw_peak_memory_increase_list,
                    strict=False,
                )
            ]

            # Skip first run (warmup) when collecting measurements
            if i > 0:
                saved_activations.extend(current_saved_activations)
                peak_intermediate_activations.extend(
                    current_peak_intermediate_activations
                )

        logger.debug(f"{saved_activations=}, {peak_intermediate_activations=}")
        return saved_activations, peak_intermediate_activations, peak_memory_usage

    def _validate_and_calculate_per_token_metrics(
        self, saved_activations: list[int], peak_intermediates: list[int]
    ) -> tuple[float, float]:
        """Validate measurements and calculate per-token metrics.

        Args:
            saved_activations: List of saved activations.
            peak_intermediates: List of peak intermediate activations.

        Returns:
            Tuple containing:
            - Saved activations per token
            - Peak intermediate activations per token
        """
        saved_activations_np = np.array(saved_activations)
        peak_intermediate_activations_np = np.array(peak_intermediates)

        # Validate consistency of measurements
        if not is_array_consistent(saved_activations_np):
            msg = f"The saved activations are not consistent. {saved_activations_np=}"
            raise ValueError(msg)
        if not is_array_consistent(peak_intermediate_activations_np):
            msg = (
                f"The peak intermediate activations are not consistent. "
                f"{peak_intermediate_activations_np=}"
            )
            raise ValueError(msg)

        # Calculate per-token metrics
        token_count = self.BATCH_SIZE_LOWER_FOR_MEM_PROFILE * self.BASE_SEQ_LEN
        dtype_size = get_dtype_element_size(LORA_DTYPE)

        saved_activations_per_token = np.median(saved_activations_np).item() / (
            token_count * dtype_size
        )
        peak_intermediate_activations_per_token = np.median(
            peak_intermediate_activations_np
        ).item() / (token_count * dtype_size)

        return saved_activations_per_token, peak_intermediate_activations_per_token

    def _process_execution_time(self) -> dict[str, dict[int, float]]:
        """Measure execution time for different batch sizes.

        Returns:
            Dictionary containing timing measurements for different components.
        """
        batch_sizes = self.BATCH_SIZES_FOR_TIME_PROFILE
        batch_size_to_fwd_times = {}
        batch_size_to_fwd_bwd_times = {}
        set_warmup_and_number(30, 30)

        for batch_size in batch_sizes:
            fwd_times = []
            fwd_bwd_times = []

            for num_layers in range(*self.LAYER_RANGES_FOR_MEM_PROFILE):
                # Create a copy of the model with the specified number of layers
                lora_model = copy(self.lora_model)
                set_decoder_layers_to_hf_causal_lm(
                    lora_model,
                    self.lora_decoder_layers[:num_layers],
                    is_lora=True,
                )

                # Benchmark forward pass
                fwd_time = benchmark(
                    func=lora_model,
                    prepare_func=partial(
                        prepare_func,
                        batch_size=batch_size,
                        seq_len=self.BASE_SEQ_LEN,
                    ),
                    use_cuda_graph=True,
                    use_cuda_event=True,
                )

                # Benchmark forward+backward pass
                fwd_bwd_time = benchmark(
                    func=partial(bwd_func, model=lora_model),
                    prepare_func=partial(
                        prepare_func,
                        batch_size=batch_size,
                        seq_len=self.BASE_SEQ_LEN,
                    ),
                    use_cuda_graph=True,
                    use_cuda_event=True,
                )

                fwd_times.append(fwd_time)
                fwd_bwd_times.append(fwd_bwd_time)

            batch_size_to_fwd_times[batch_size] = fwd_times
            batch_size_to_fwd_bwd_times[batch_size] = fwd_bwd_times
            logger.debug(f"{batch_size=}, {fwd_times=}, {fwd_bwd_times=}")

        # Process the results
        batch_size_to_decoder_layer_fwd = {}
        batch_size_to_decoder_layer_fwd_bwd = {}
        batch_size_to_pre_post_layer_fwd_bwd = {}

        for batch_size in batch_sizes:
            # Calculate time per layer by taking differences
            fwd_times_with_diff_num_layers = batch_size_to_fwd_times[batch_size]
            fwd_bwd_times_with_diff_num_layers = batch_size_to_fwd_bwd_times[batch_size]

            fwd_times_decoder_layer = [
                y - x for x, y in pairwise(fwd_times_with_diff_num_layers)
            ]
            fwd_bwd_times_decoder_layer = [
                y - x for x, y in pairwise(fwd_bwd_times_with_diff_num_layers)
            ]

            # Validate consistency
            if not is_array_consistent(fwd_times_decoder_layer):
                msg = (
                    f"The forward times of the decoder layer are not consistent. "
                    f"{fwd_times_decoder_layer=}"
                )
                raise ValueError(msg)

            if not is_array_consistent(fwd_bwd_times_decoder_layer):
                msg = (
                    f"The forward and backward times of the decoder layer are not "
                    f"consistent. {fwd_bwd_times_decoder_layer=}"
                )
                raise ValueError(msg)

            # Calculate averages
            average_fwd_time_decoder_layer = np.median(fwd_times_decoder_layer).item()
            average_fwd_bwd_time_decoder_layer = np.median(
                fwd_bwd_times_decoder_layer
            ).item()

            batch_size_to_decoder_layer_fwd[batch_size] = average_fwd_time_decoder_layer
            batch_size_to_decoder_layer_fwd_bwd[batch_size] = (
                average_fwd_bwd_time_decoder_layer
            )

            # Calculate pre and post layer times
            fwd_bwd_times_pre_post_layer = [
                t - num_layers * average_fwd_bwd_time_decoder_layer
                for t, num_layers in zip(
                    fwd_bwd_times_with_diff_num_layers,
                    range(*self.LAYER_RANGES_FOR_MEM_PROFILE),
                    strict=False,
                )
            ]

            if not is_array_consistent(fwd_bwd_times_pre_post_layer):
                msg = (
                    f"The forward and backward times of the pre- and post-layers are "
                    f"not consistent. {fwd_bwd_times_pre_post_layer=}"
                )
                raise ValueError(msg)

            average_fwd_bwd_time_pre_post_layer = np.median(
                fwd_bwd_times_pre_post_layer
            ).item()

            batch_size_to_pre_post_layer_fwd_bwd[batch_size] = (
                average_fwd_bwd_time_pre_post_layer
            )

            logger.debug(
                f"{batch_size=}, "
                f"{average_fwd_time_decoder_layer=}, "
                f"{average_fwd_bwd_time_decoder_layer=}, "
                f"{average_fwd_bwd_time_pre_post_layer=}"
            )

        # Update to per-token time
        def map_to_per_token_time(
            batch_size_to_times: dict[int, list[float] | float],
        ) -> dict[int, list[float]]:
            return {
                batch_size * self.BASE_SEQ_LEN: times
                for batch_size, times in batch_size_to_times.items()
            }

        decoder_layer_fwd_time_by_tokens = map_to_per_token_time(
            batch_size_to_decoder_layer_fwd
        )
        decoder_layer_fwd_bwd_time_by_tokens = map_to_per_token_time(
            batch_size_to_decoder_layer_fwd_bwd
        )
        pre_post_layer_fwd_bwd_time_by_tokens = map_to_per_token_time(
            batch_size_to_pre_post_layer_fwd_bwd
        )

        return {
            "decoder_layer_fwd_time_by_tokens": decoder_layer_fwd_time_by_tokens,
            "decoder_layer_fwd_bwd_time_by_tokens": (
                decoder_layer_fwd_bwd_time_by_tokens
            ),
            "pre_post_layer_fwd_bwd_time_by_tokens": (
                pre_post_layer_fwd_bwd_time_by_tokens
            ),
        }

    def _process_tp_comm_count(self) -> dict[str, int]:
        """Calculate tensor parallelism communication counts.

        Returns:
            Dictionary containing tensor parallelism communication counts.
        """
        return {
            "pre_layer_tp_comm_count": self.model.config.hidden_size,
            "post_layer_tp_comm_count": self.model.config.hidden_size,
        }

    def profile(self) -> LoRALLMProfileResult:
        """Profile the model."""
        frozen_param_counts = self._process_frozen_param_counts()
        trainable_param_counts = self._process_trainable_param_counts()
        saved_activations_and_peak_intermediate = (
            self._process_saved_activations_and_peak_intermediate()
        )
        execution_time = self._process_execution_time()
        tp_comm_count = self._process_tp_comm_count()

        return LoRALLMProfileResult(
            name=self.pretrained_model_name_or_path,
            base_model_dtype=self.base_model_dtype,
            num_layers=self.ori_config.num_hidden_layers,
            hidden_size=self.ori_config.hidden_size,
            **frozen_param_counts,
            **trainable_param_counts,
            **saved_activations_and_peak_intermediate,
            **execution_time,
            **tp_comm_count,
        )


def interpolate_execution_time(
    profiled_times: list[float],
    profiled_token_counts: list[int],
    target_token_count: int,
) -> float:
    """Interpolate the execution time of the model.

    - If the target token is smaller than the smallest profiled token count,
      it raises an error.
    - If the target token is larger than the largest profiled token count,
      it interpolates the execution time using the last two points.
    - Otherwise, it interpolates the execution time using the two points that
      the target token count is between.

    Args:
        profiled_times: The execution times of the model.
        profiled_token_counts: The token counts of the model.
        target_token_count: The target token count.

    Returns:
        The interpolated execution time.

    Example:
        >>> profiled_times = [1.0, 2.0, 3.0]
        >>> profiled_token_counts = [100, 200, 300]
        >>> target_token_count = 150
        >>> interpolate_execution_time(
            profiled_times,
            profiled_token_counts,
            target_token_count,
        )
        >>> 1.5
    """
    if len(profiled_times) != len(profiled_token_counts):
        msg = (
            f"Length mismatch: profiled_times ({len(profiled_times)}) and "
            f"profiled_token_counts ({len(profiled_token_counts)})"
        )
        raise ValueError(msg)

    if not profiled_times:
        msg = "Empty profiled_times and profiled_token_counts"
        raise ValueError(msg)

    # Sort by token counts to ensure correct interpolation
    sorted_data = sorted(zip(profiled_token_counts, profiled_times, strict=True))
    sorted_token_counts, sorted_times = zip(*sorted_data, strict=True)

    # Check if target is smaller than the smallest profiled token count
    if target_token_count < sorted_token_counts[0]:
        msg = (
            f"Target token count ({target_token_count}) is smaller than the "
            f"smallest profiled token count ({sorted_token_counts[0]})"
        )
        raise ValueError(msg)

    # If target is larger than the largest profiled token count, use the last two points
    if target_token_count > sorted_token_counts[-1]:
        x1, y1 = sorted_token_counts[-2], sorted_times[-2]
        x2, y2 = sorted_token_counts[-1], sorted_times[-1]
        return y1 + (y2 - y1) * (target_token_count - x1) / (x2 - x1)

    # Find the two points that the target token count is between
    for i in range(len(sorted_token_counts) - 1):
        if sorted_token_counts[i] <= target_token_count <= sorted_token_counts[i + 1]:
            x1, y1 = sorted_token_counts[i], sorted_times[i]
            x2, y2 = sorted_token_counts[i + 1], sorted_times[i + 1]
            return y1 + (y2 - y1) * (target_token_count - x1) / (x2 - x1)

    # This should never happen due to the checks above
    msg = f"Failed to interpolate for target token count {target_token_count}"
    raise RuntimeError(msg)


def estimate_communication_overhead(
    bytes_: float,
    num_nodes: int,
    num_gpus_per_node: int,
    intra_node_bandwidth: float,
    inter_node_bandwidth: float,
    operator_type: str,
    overlap_factor: float = 0.0,
) -> float:
    """Estimate the communication overhead for NCCL collective operations.

    Args:
        bytes_: Number of bytes to transfer.
        num_nodes: Number of nodes in the cluster (e.g., 1, 2).
        num_gpus_per_node: Number of GPUs per node (e.g., 2, 4, 8).
        intra_node_bandwidth:
            Bandwidth within a node (e.g., NVLink/NVSwitch, in bytes/sec).
        inter_node_bandwidth:
            Bandwidth between nodes (e.g., InfiniBand, in bytes/sec).
        operator_type:
            Type of collective operation ('allreduce', 'reduce-scatter', 'all-gather').
        overlap_factor:
            The overlap factor of the communication, 0.0 means no overlap and 1.0 means
            full overlap.

    Returns:
        float: Estimated communication time in seconds.

    Raises:
        ValueError: If operator_type is not one of 'allreduce', 'reduce-scatter',
        or 'all-gather'.
    """
    # Assign shorthand variables
    n = num_nodes
    g = num_gpus_per_node

    # Calculate intra-node and inter-node time components
    if operator_type == "allreduce":
        # Hierarchical allreduce: intra-node reduce + inter-node allreduce + intra-node
        # broadcast
        intra_time = 2 * ((g - 1) / g) * (bytes_ / intra_node_bandwidth)
        inter_time = (2 * (n - 1) / n) * (bytes_ / inter_node_bandwidth)
    elif operator_type == "reduce-scatter":
        # Hierarchical reduce-scatter: intra-node reduce-scatter + inter-node comm
        intra_time = ((g - 1) / g) * (bytes_ / intra_node_bandwidth)
        inter_time = ((n - 1) / n) * (bytes_ / inter_node_bandwidth)
    elif operator_type == "all-gather":
        # Hierarchical all-gather: intra-node all-gather + inter-node comm
        intra_time = ((g - 1) / g) * (bytes_ / intra_node_bandwidth)
        inter_time = ((n - 1) / n) * (bytes_ / inter_node_bandwidth)
    else:
        msg = f"Invalid operator_type: {operator_type}"
        raise ValueError(msg)

    return max(intra_time, inter_time) + (1 - overlap_factor) * min(
        intra_time, inter_time
    )


def overlap_time(
    time_1: float,
    time_2: float,
    overlap_factor: float = 0.0,
) -> float:
    """Overlap the time of two operations.

    Args:
        time_1: The time of the first operation.
        time_2: The time of the second operation.
        overlap_factor: The overlap factor of the two operations.

    Returns:
        The overlapped time.
    """
    return max(time_1, time_2) + (1 - overlap_factor) * min(time_1, time_2)


@dataclass
class SimulationConfiguration:
    """Simulation configuration."""

    # Model Configuration
    model_name: str
    base_model_dtype: str
    # Hardware and Mesh Configuration
    nnodes: int
    ngpus_per_node: int
    intra_node_bandwidth: float
    inter_node_bandwidth: float
    # Input Configuration
    seq_len: int
    per_device_batch_size: int
    # LoRA Model Configuration
    lora_ranks_per_gpu: int
    # Optimization Configuration
    gradient_accumulation_steps: int
    dp_size: int
    embedding_tp_size: int
    fsdp_size: int
    pp_size: int
    pp_layer_partitions: list[int]
    num_gradient_checkpointing_layers: int
    # Override Configuration
    override_num_hidden_layers: int | None = None
    # Hyper-parameters
    compute_comm_overlap_factor: float = 0.8
    intra_inter_comm_overlap_factor: float = 0.7


@dataclass
class LoRALLMPredictor:
    """LoRA model performance predictor."""

    def __init__(
        self,
        model_name: str,
        base_model_dtype: str,
        *,
        override_if_exists: bool = False,
    ) -> None:
        """Initialize the model performance predictor.

        Args:
            model_name: The name of the model.
            base_model_dtype: The dtype of the base model.
            override_if_exists: Whether to override the existing profile result.
        """
        self.model_name = model_name
        self.base_model_dtype = base_model_dtype
        self.profile_result = LoRALLMProfileResult.load(
            model_name,
            base_model_dtype,
            profile_if_not_exists=True,
            override_if_exists=override_if_exists,
        )

    def estimate(
        self,
        simulation_config: SimulationConfiguration,
    ) -> tuple[float, float]:
        """Estimate the time and peak memory usage of the model."""
        # Fast path if there is no pipeline parallelism
        if simulation_config.pp_size == 1:
            peak_memory_usage = self.estimate_memory(simulation_config)
            execution_time_per_microbatch = self.estimate_execution_time(
                simulation_config
            )
            return (
                execution_time_per_microbatch
                * simulation_config.gradient_accumulation_steps,
                peak_memory_usage,
            )

        # Slow path if there is pipeline parallelism
        stage_memory_usage = []
        stage_execution_time = []
        for stage_idx in range(simulation_config.pp_size):
            stage_memory_usage.append(
                self.estimate_memory(simulation_config, stage_idx)
            )
            stage_execution_time.append(
                self.estimate_execution_time(simulation_config, stage_idx)
            )
        # Summarize the execution time (sum(t) + max(t) * (ga - 1))
        # TODO(zhanda): Improve the estimation of the execution time
        # with no bubble
        if simulation_config.gradient_accumulation_steps < simulation_config.pp_size:
            total_execution_time = sum(stage_execution_time) + max(
                stage_execution_time
            ) * (simulation_config.gradient_accumulation_steps - 1)
        else:
            total_execution_time = simulation_config.gradient_accumulation_steps * max(
                stage_execution_time
            )
        # Summarize the memory usage
        peak_memory_usage = max(stage_memory_usage)
        return total_execution_time, peak_memory_usage

    def _get_num_layers(
        self, simulation_config: SimulationConfiguration, stage_idx: int | None = None
    ) -> int:
        """Get the number of layers."""
        if stage_idx is not None:
            if (
                simulation_config.pp_size <= 1
                or not simulation_config.pp_layer_partitions
            ):
                msg = (
                    f"{stage_idx=} is provided but the pipeline parallelism size "
                    f"is {simulation_config.pp_size} and the layer partitions are "
                    f"{simulation_config.pp_layer_partitions}."
                )
                raise ValueError(msg)
            return simulation_config.pp_layer_partitions[stage_idx]

        if simulation_config.override_num_hidden_layers is not None:
            return simulation_config.override_num_hidden_layers

        return self.profile_result.num_layers

    def _has_pre_post_layer(
        self, simulation_config: SimulationConfiguration, stage_idx: int | None = None
    ) -> bool:
        """Whether the pre/post layer is enabled."""
        has_pre_layer = stage_idx is None or stage_idx == 0
        has_post_layer = stage_idx is None or stage_idx == simulation_config.pp_size - 1
        return has_pre_layer, has_post_layer

    def estimate_execution_time(
        self, simulation_config: SimulationConfiguration, stage_idx: int | None = None
    ) -> float:
        """Estimate the execution time of the model.

        Args:
            simulation_config: The simulation configuration.
            stage_idx: The index of the stage to estimate the execution time.

        Returns:
            The estimated execution time (fwd+bwd) for each microbatch.
        """
        num_layers = self._get_num_layers(simulation_config, stage_idx)
        has_pre_layer, has_post_layer = self._has_pre_post_layer(
            simulation_config, stage_idx
        )
        overlap_factor = simulation_config.compute_comm_overlap_factor
        num_ckpt = simulation_config.num_gradient_checkpointing_layers

        # 1. Get decoder layer time (forward + backward)
        # Each layer can be configured as using recomputation or not. And each
        # decoder layer's computation time can be overlapped with FSDP communication
        # with overlap factor.
        decoder_layer_fwd_execution_time, decoder_layer_bwd_execution_time = (
            self._estimate_decoder_layer_execution_time(simulation_config)
        )
        decoder_layer_fwd_fsdp_time, decoder_layer_bwd_fsdp_time = (
            self._estimate_decoder_layer_fsdp_time(simulation_config)
        )
        decoder_layer_fwd_time = overlap_time(
            decoder_layer_fwd_execution_time,
            decoder_layer_fwd_fsdp_time,
            overlap_factor,
        )
        decoder_layer_bwd_time_with_no_ckpt = overlap_time(
            decoder_layer_bwd_execution_time,
            decoder_layer_bwd_fsdp_time,
            overlap_factor,
        )
        decoder_layer_bwd_time_with_ckpt = overlap_time(
            decoder_layer_fwd_execution_time + decoder_layer_bwd_execution_time,
            decoder_layer_bwd_fsdp_time,
            overlap_factor,
        )
        decoder_layers_time = (
            decoder_layer_fwd_time * num_layers
            + decoder_layer_bwd_time_with_ckpt * num_ckpt
            + decoder_layer_bwd_time_with_no_ckpt * (num_layers - num_ckpt)
        )

        # 2. Get pre/post layer time (including TP communication)
        # execution time is only counted for the post layer (as we think it is
        # more likely to be the bottleneck)
        pre_post_layer_execution_time = (
            self._estimate_pre_post_layer_execution_time(simulation_config)
            * has_post_layer
        )
        # TP communication time should be counted for both pre and post layers
        pre_layer_tp_time, post_layer_tp_time = self._estimate_pre_post_layer_tp_time(
            simulation_config
        )
        pre_post_layer_tp_time = (
            pre_layer_tp_time * has_pre_layer + post_layer_tp_time * has_post_layer * 2
        )
        pre_post_layer_time = pre_post_layer_execution_time + pre_post_layer_tp_time

        # Total time
        total_time = decoder_layers_time + pre_post_layer_time

        logger.debug(
            f"Predicted total time: {total_time:.4f} seconds\n"
            f"  - Decoder layers time: {decoder_layers_time:.4f} seconds\n"
            f"    - FSDP (fwd) time: {decoder_layer_fwd_fsdp_time:.4f} seconds\n"
            f"    - FSDP (bwd) time: {decoder_layer_bwd_fsdp_time:.4f} seconds\n"
            f"    - Exec (fwd) time: {decoder_layer_fwd_execution_time:.4f} seconds\n"
            f"    - Exec (bwd) time: {decoder_layer_bwd_execution_time:.4f} seconds\n"
            f"  - Pre/post layer time: {pre_post_layer_time:.4f} seconds\n"
            f"    - Exec time: {pre_post_layer_execution_time:.4f} seconds\n"
            f"    - TP time: {pre_post_layer_tp_time:.4f} seconds\n"
        )

        return total_time

    def _estimate_decoder_layer_execution_time(
        self, simulation_config: SimulationConfiguration
    ) -> tuple[float, float]:
        """Estimate the execution time of the decoder layer.

        Args:
            simulation_config: The simulation configuration.

        Returns:
            The estimated execution time.
        """
        # Get the execution time per token from the profile result
        fwd_ref_times_dict = self.profile_result.decoder_layer_fwd_time_by_tokens
        fwd_bwd_ref_times_dict = (
            self.profile_result.decoder_layer_fwd_bwd_time_by_tokens
        )

        # Convert token dictionary keys from strings to integers
        fwd_ref_token_counts = [int(k) for k in fwd_ref_times_dict]
        fwd_ref_times = [float(v) for v in fwd_ref_times_dict.values()]
        fwd_bwd_ref_token_counts = [int(k) for k in fwd_bwd_ref_times_dict]
        fwd_bwd_ref_times = [float(v) for v in fwd_bwd_ref_times_dict.values()]

        # Calculate total tokens per GPU for interpolation
        tokens_per_gpu = (
            simulation_config.seq_len * simulation_config.per_device_batch_size
        )

        # Interpolate the execution time based on the number of tokens
        fwd_times = interpolate_execution_time(
            fwd_ref_times, fwd_ref_token_counts, tokens_per_gpu
        )
        fwd_bwd_times = interpolate_execution_time(
            fwd_bwd_ref_times, fwd_bwd_ref_token_counts, tokens_per_gpu
        )
        bwd_times = fwd_bwd_times - fwd_times

        return fwd_times, bwd_times

    def _estimate_decoder_layer_fsdp_time(
        self, simulation_config: SimulationConfiguration
    ) -> tuple[float, float]:
        """Estimate the execution time of the decoder layer with FSDP.

        Args:
            simulation_config: The simulation configuration.

        Returns:
            The estimated forward and backward execution time.
        """
        overlap_factor = simulation_config.intra_inter_comm_overlap_factor

        # Skip if FSDP is not used
        if simulation_config.fsdp_size <= 1:
            return 0.0, 0.0

        # For fwd: we need to all-gather the frozen parameters
        # For bwd: we need to (
        #   all-gather the frozen parameters
        #   we directly ignore the reduce-scatter of the gradients as they are quite
        #   small and the overhead is negligible, and it also shows in the normal
        #   ddp.
        # Therefore, the comm time for fwd and bwd are the same.
        # TODO(zhanda): Consider the communication of the gradients.
        base_param_count = self.profile_result.decoder_layer_param_count
        param_count = base_param_count

        # Get dtype size
        dtype_size = get_dtype_element_size(str_to_torch_dtype(self.base_model_dtype))

        # Calculate bytes to communicate
        bytes_to_communicate = param_count * dtype_size

        # All-gather time
        all_gather_time = estimate_communication_overhead(
            bytes_=bytes_to_communicate,
            num_nodes=simulation_config.nnodes,
            num_gpus_per_node=simulation_config.ngpus_per_node,
            intra_node_bandwidth=simulation_config.intra_node_bandwidth,
            inter_node_bandwidth=simulation_config.inter_node_bandwidth,
            operator_type="all-gather",
            overlap_factor=overlap_factor,  # Partial overlap with computation
        )

        fwd_time = all_gather_time
        bwd_time = all_gather_time

        return fwd_time, bwd_time

    def _estimate_pre_post_layer_execution_time(
        self, simulation_config: SimulationConfiguration
    ) -> float:
        """Estimate the execution time of the pre- and post-layers.

        Args:
            simulation_config: The simulation configuration.

        Returns:
            The estimated execution time.
        """
        # Get the execution time per token from the profile result
        ref_times_dict = self.profile_result.pre_post_layer_fwd_bwd_time_by_tokens

        # Convert token dictionary keys from strings to integers
        ref_token_counts = [int(k) for k in ref_times_dict]
        ref_times = [float(v) for v in ref_times_dict.values()]

        # Calculate total tokens per GPU for interpolation
        tokens_per_gpu = (
            simulation_config.seq_len * simulation_config.per_device_batch_size
        )

        # Interpolate the execution time based on the number of tokens
        return interpolate_execution_time(ref_times, ref_token_counts, tokens_per_gpu)

    def _estimate_pre_post_layer_tp_time(
        self, simulation_config: SimulationConfiguration
    ) -> tuple[float, float]:
        """Estimate the tensor parallelism time of the pre- and post-layers.

        Args:
            simulation_config: The simulation configuration.

        Returns:
            The estimated tensor parallelism time.
        """
        overlap_factor = simulation_config.intra_inter_comm_overlap_factor
        tp_size = simulation_config.embedding_tp_size

        # Skip if TP is not used
        if simulation_config.embedding_tp_size <= 1:
            return 0.0, 0.0

        # Get pre/post layer TP communication counts
        pre_layer_tp_count = self.profile_result.pre_layer_tp_comm_count
        post_layer_tp_count = self.profile_result.post_layer_tp_comm_count

        # Get dtype size
        dtype_size = get_dtype_element_size(str_to_torch_dtype(LORA_DTYPE))

        # Tokens
        tokens_per_gpu = (
            simulation_config.seq_len * simulation_config.per_device_batch_size
        )

        # 1. Please see the _check_simulation_config() for the reason why we set
        # num_nodes=1 and num_gpus_per_node=tp_size.
        # 2. Both comm only happens in the forward pass.
        # Calculate bytes to communicate for pre layer (all-reduce)
        pre_layer_bytes = tokens_per_gpu * pre_layer_tp_count * dtype_size
        pre_layer_time = estimate_communication_overhead(
            bytes_=pre_layer_bytes,
            num_nodes=1,
            num_gpus_per_node=tp_size,
            intra_node_bandwidth=simulation_config.intra_node_bandwidth,
            inter_node_bandwidth=simulation_config.inter_node_bandwidth,
            operator_type="all-reduce",
            overlap_factor=overlap_factor,
        )

        # Calculate bytes to communicate for post layer (all-reduce)
        post_layer_bytes = tokens_per_gpu * post_layer_tp_count * dtype_size
        post_layer_time = estimate_communication_overhead(
            bytes_=post_layer_bytes,
            num_nodes=1,
            num_gpus_per_node=tp_size,
            intra_node_bandwidth=simulation_config.intra_node_bandwidth,
            inter_node_bandwidth=simulation_config.inter_node_bandwidth,
            operator_type="all-reduce",
            overlap_factor=overlap_factor,
        )

        return pre_layer_time, post_layer_time

    def estimate_memory(
        self, simulation_config: SimulationConfiguration, stage_idx: int | None = None
    ) -> float:
        """Estimate the memory of the model.

        Args:
            simulation_config: The simulation configuration.
            stage_idx: The index of the stage to estimate the memory.

        Returns:
            The estimated memory in bytes.
        """
        # 1. Model states memory (including LoRA parameters)
        model_states_mem = self._estimate_model_states_memory(
            simulation_config, stage_idx
        )

        # 2. Saved activations memory
        activations_mem = self._estimate_saved_activations_memory(
            simulation_config, stage_idx
        )

        # 3. Peak intermediate memory
        peak_mem = self._estimate_peak_intermediate_memory(simulation_config, stage_idx)

        # Total memory
        total_mem = model_states_mem + activations_mem + peak_mem

        logger.debug(
            f"Predicted memory usage:\n"
            f" - Stage idx: {stage_idx}\n"
            f" - Model states: {model_states_mem / 1024**3:.2f} GB\n"
            f" - Activations: {activations_mem / 1024**3:.2f} GB\n"
            f" - Peak: {peak_mem / 1024**3:.2f} GB\n"
            f" - Total: {total_mem / 1024**3:.2f} GB\n"
        )
        return total_mem

    def _estimate_model_states_memory(
        self, simulation_config: SimulationConfiguration, stage_idx: int | None = None
    ) -> float:
        """Estimate the memory used by model states.

        Args:
            simulation_config: The simulation configuration.
            stage_idx: The index of the stage to estimate the memory.

        Returns:
            The estimated memory in bytes.
        """
        # Get the number of layers
        num_layers = self._get_num_layers(simulation_config, stage_idx)
        has_pre_layer, has_post_layer = self._has_pre_post_layer(
            simulation_config, stage_idx
        )

        # 1. Base model parameter counts
        # Get dtype size
        base_dtype_size = get_dtype_element_size(
            str_to_torch_dtype(self.base_model_dtype)
        )
        decoder_layer_param_bytes = (
            self.profile_result.decoder_layer_param_count * base_dtype_size
        )
        pre_layer_param_bytes = (
            self.profile_result.pre_layer_param_count * base_dtype_size
        )
        post_layer_param_bytes = (
            self.profile_result.post_layer_param_count * base_dtype_size
        )

        # 2. LoRA parameters (Always LORA_DTYPE, which is default to bfloat16)
        lora_dtype_size = get_dtype_element_size(LORA_DTYPE)
        lora_param_bytes = (
            self.profile_result.decoder_layer_lora_param_count_per_rank
            * simulation_config.lora_ranks_per_gpu
            * lora_dtype_size
        )
        lora_optim_states_bytes = (
            self.profile_result.decoder_layer_lora_param_count_per_rank
            * simulation_config.lora_ranks_per_gpu
            * get_dtype_element_size(torch.float32)
        )

        # 3. Count weights
        fsdp_size = simulation_config.fsdp_size
        tp_size = simulation_config.embedding_tp_size
        return (
            # When FSDP is used, we need 1 complete layer + 1 in communication + rest
            # sharded. For those two layers, weights and grads are not sharded while
            # optim states are sharded.
            # (1) Two full layers
            (
                # Weights
                (decoder_layer_param_bytes + lora_param_bytes)
                # Gradients
                + lora_param_bytes
                # Optim states
                + lora_optim_states_bytes / fsdp_size
            )
            * 2
            # (2) Rest of the layers (sharded for three types of states)
            + (
                (decoder_layer_param_bytes + lora_param_bytes)
                + lora_param_bytes
                + lora_optim_states_bytes
            )
            * (num_layers - 2)
            / fsdp_size
            # (3) Pre/post layers (only affected by TP)
            + (
                # Only weights as they are not trainable
                pre_layer_param_bytes * has_pre_layer
                + post_layer_param_bytes * has_post_layer
            )
            / tp_size
        )

    def _estimate_saved_activations_memory(
        self, simulation_config: SimulationConfiguration, stage_idx: int | None = None
    ) -> float:
        """Estimate the memory used by saved activations.

        Args:
            simulation_config: The simulation configuration.
            stage_idx: The index of the stage to estimate the memory.

        Returns:
            The estimated memory in bytes.
        """
        # Get the number of layers
        num_layers = self._get_num_layers(simulation_config, stage_idx)
        num_ckpt = simulation_config.num_gradient_checkpointing_layers
        num_no_ckpt = num_layers - num_ckpt

        # Get dtype size
        dtype_size = get_dtype_element_size(
            LORA_DTYPE
        )  # Activations are always in LORA_DTYPE

        # Calculate tokens per GPU
        tokens_per_gpu = (
            simulation_config.seq_len * simulation_config.per_device_batch_size
        )

        # Memory for layers without checkpointing
        no_ckpt_activations = (
            self.profile_result.decoder_layer_saved_activations_no_ckpt_count
            * tokens_per_gpu
            * dtype_size
        )

        # Memory for layers with checkpointing
        ckpt_activations = (
            self.profile_result.decoder_layer_saved_activations_with_ckpt_count
            * tokens_per_gpu
            * dtype_size
        )

        # Calculate the saved activations memory
        saved_activations_per_microbatch = (
            num_no_ckpt * no_ckpt_activations + num_ckpt * ckpt_activations
        )

        # In pipeline parallel, we also need to account for the previous microbatches
        # that are still saved in the GPU memory. We assume that the gradient
        # accumulation steps is larger than the number of pipeline stages.
        if simulation_config.pp_size > 1:
            saved_microbatches = simulation_config.pp_size - stage_idx
        else:
            saved_microbatches = 1

        # Total activations memory
        return saved_activations_per_microbatch * saved_microbatches

    def _estimate_peak_intermediate_memory(
        self, simulation_config: SimulationConfiguration, stage_idx: int | None = None
    ) -> float:
        """Estimate the peak intermediate memory.

        Args:
            simulation_config: The simulation configuration.
            stage_idx: The index of the stage to estimate the memory.

        Returns:
            The estimated memory in bytes.
        """
        # Choose the correct peak intermediate memory to use
        _, has_post_layer = self._has_pre_post_layer(simulation_config, stage_idx)
        peak_intermediate_activation_count = (
            self.profile_result.peak_intermediate_activation_count
            if has_post_layer
            else self.profile_result.peak_intermediate_activation_count_without_lm_head
        )

        # Get dtype size
        dtype_size = get_dtype_element_size(
            LORA_DTYPE
        )  # Intermediates are always in LORA_DTYPE

        # Calculate tokens per GPU
        tokens_per_gpu = (
            simulation_config.seq_len * simulation_config.per_device_batch_size
        )

        # Peak intermediate memory
        return peak_intermediate_activation_count * tokens_per_gpu * dtype_size

    def _check_simulation_config(
        self, simulation_config: SimulationConfiguration
    ) -> None:
        """Check the simulation configuration."""
        fsdp_size = simulation_config.fsdp_size
        tp_size = simulation_config.embedding_tp_size
        nnodes = simulation_config.nnodes
        ngpus_per_node = simulation_config.ngpus_per_node

        if fsdp_size > 1 and fsdp_size != nnodes * ngpus_per_node:
            msg = (
                f"Currently, FSDP size ({fsdp_size}) must be equal to the product of "
                f"the number of nodes ({nnodes}) and the number of GPUs per node "
                f"({ngpus_per_node})."
            )
            raise ValueError(msg)

        if tp_size > 1 and tp_size > ngpus_per_node:
            msg = (
                f"Currently, TP size ({tp_size}) must be less than or equal to the "
                f"number of GPUs per node ({ngpus_per_node})."
            )
            raise ValueError(msg)


def optimization_configuration_generator(  # noqa: C901
    num_gpus: int,
    global_batch_size: int,
    num_layers: int,
    num_ckpt_interval: int,
    extra_layer_limit: int = 0,
    upper_bound: int = 32,
) -> list[tuple[int, int, int, int, int, int]]:
    """Generate all possible parallelism configurations.

    Args:
        num_gpus: Total number of GPUs available
        global_batch_size: Desired global batch size (must be power of 2)
        num_layers: Number of layers in the model
        num_ckpt_interval: Number of layers between checkpoints
        extra_layer_limit: Maximum number of extra layers
        upper_bound: Maximum value for any configuration parameter

    Returns:
        List of tuples containing valid configurations in the order:
        (per_device_batch_size, gradient_accumulation_steps, dp_size, fsdp_size,
        pp_size, num_ckpt)

    Raises:
        ValueError: If global_batch_size is not a power of 2
    """
    if not math.log2(global_batch_size).is_integer():
        msg = f"Global batch size ({global_batch_size}) must be a power of 2."
        raise ValueError(msg)

    # Precompute all power of 2 values up to upper bound
    power_of_2_values = [2**i for i in range(math.ceil(math.log2(upper_bound)) + 1)]
    configs: list[tuple[int, int, int, int, int, int]] = []

    # Iterate through valid combinations using early termination
    for per_device_batch_size in power_of_2_values:
        remaining_batch = global_batch_size // per_device_batch_size
        if remaining_batch < 1:
            continue

        for gradient_accumulation_steps in power_of_2_values:
            if gradient_accumulation_steps > remaining_batch:
                continue
            remaining_batch_after_ga = remaining_batch // gradient_accumulation_steps

            for dp_size in power_of_2_values:
                if dp_size > remaining_batch_after_ga:
                    continue
                remaining_batch_after_dp = remaining_batch_after_ga // dp_size

                for fsdp_size in power_of_2_values:
                    if fsdp_size > remaining_batch_after_dp:
                        continue
                    if remaining_batch_after_dp // fsdp_size != 1:
                        continue

                    pp_size = num_gpus // (dp_size * fsdp_size)
                    if (
                        pp_size in power_of_2_values
                        and pp_size > 0
                        and gradient_accumulation_steps >= pp_size
                    ):
                        layer_partitions = even_layer_partition(
                            num_layers, pp_size, extra_layer_limit=extra_layer_limit
                        )
                        configs.extend(
                            (
                                per_device_batch_size,
                                gradient_accumulation_steps,
                                dp_size,
                                fsdp_size,
                                pp_size,
                                num_ckpt,
                            )
                            for num_ckpt in range(
                                0, max(layer_partitions) + 1, num_ckpt_interval
                            )
                        )

    # Debug: print the number of configurations
    for i, config in enumerate(configs):
        logger.info(
            f"Config {i} ({config}): "
            f"PerDeviceBatchSize: {config[0]}, "
            f"GradientAccumulationSteps: {config[1]}, "
            f"DPSize: {config[2]}, "
            f"FSDPSize: {config[3]}, "
            f"PPsize: {config[4]}, "
            f"NumCKPT: {config[5]}, "
        )

    return configs


def even_layer_partition(
    num_layers: int, num_stages: int, extra_layer_limit: int = 0
) -> list[int]:
    """Evenly partition the layers into the number of stages.

    Try from 0 to extra_layer_limit. Put all extra layers into the last stage.
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


model_configs = {
    "8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "hidden_size": 4096,
    },
    "32b": {
        "model_name": "Qwen/QwQ-32B",
        "num_layers": 64,
        "hidden_size": 5120,
    },
    "70b": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "num_layers": 80,
        "hidden_size": 8192,
    },
    "8x7b-moe": {
        "model_name": "mistralai/Mixtral-8x7B-v0.1",
        "num_layers": 32,
        "hidden_size": 4096,
    },
}

if __name__ == "__main__":
    override_num_hidden_layers = None

    model_to_use = "70b"
    available_gpu_memory = 80 * 0.925
    seq_len = 2048
    global_batch_size = 8

    model_name = model_configs[model_to_use]["model_name"]
    num_layers = model_configs[model_to_use]["num_layers"]
    if override_num_hidden_layers is not None:
        num_layers = override_num_hidden_layers
    base_model_dtype = "bfloat16"
    result = LoRALLMProfileResult.load(
        model_name,
        base_model_dtype,
        profile_if_not_exists=True,
        override_if_exists=False,
    )

    # Range of choices
    power_of_2_choices = [1, 2, 4, 8, 16]

    # Define the configuration choices
    # Parallelism Configurations are generated on the fly
    ngpus_choices = power_of_2_choices

    config_key_header = [
        "NNodes",
        "NGpusPerNode",
        "PerDeviceBatchSize",
        "GradientAccumulationSteps",
        "DPSize",
        "FSDPSize",
        "PPSize",
        "NumCKPT",
    ]
    result_value_header = ["Memory", "Time", "FitMemory", "ThroughputPerGPU"]
    header = [*config_key_header, *result_value_header]
    results: dict[
        tuple[int, int, int, int, int, int, int, int], tuple[float, float, bool, float]
    ] = {}

    for ngpus in ngpus_choices:
        # Generate the parallelism configuration
        optimization_configurations = optimization_configuration_generator(
            ngpus,
            global_batch_size,
            num_layers,
            num_ckpt_interval=4,
            extra_layer_limit=0,
            upper_bound=power_of_2_choices[-1],
        )
        for optimization_config in optimization_configurations:
            (
                per_device_batch_size,
                gradient_accumulation_steps,
                dp_size,
                fsdp_size,
                pp_size,
                num_ckpt,
            ) = optimization_config

            # Determine the number of nodes and GPUs per node
            default_num_gpus_per_node = 8
            if ngpus >= default_num_gpus_per_node:
                nnodes = ngpus // default_num_gpus_per_node
                ngpus_per_node = default_num_gpus_per_node
            else:
                nnodes = 1
                ngpus_per_node = ngpus

            # Check the memory profile
            simulation_config = SimulationConfiguration(
                model_name=model_name,
                base_model_dtype=base_model_dtype,
                override_num_hidden_layers=num_layers,
                nnodes=nnodes,
                ngpus_per_node=ngpus_per_node,
                intra_node_bandwidth=180 * 1024**3,
                inter_node_bandwidth=45 * 1024**3,
                seq_len=seq_len,
                per_device_batch_size=per_device_batch_size,
                lora_ranks_per_gpu=16,
                gradient_accumulation_steps=gradient_accumulation_steps,
                dp_size=dp_size,
                embedding_tp_size=1,
                fsdp_size=fsdp_size,
                pp_size=pp_size,
                pp_layer_partitions=even_layer_partition(
                    num_layers, pp_size, extra_layer_limit=0
                ),
                num_gradient_checkpointing_layers=num_ckpt,
            )
            predictor = LoRALLMPredictor(
                model_name=model_name,
                base_model_dtype=base_model_dtype,
            )
            # Estimate memory and execution time
            execution_time, memory = predictor.estimate(simulation_config)
            memory = memory / 1024**3
            # Process to get the throughput per gpu
            fit_memory = memory <= available_gpu_memory
            throughput_per_gpu = global_batch_size * seq_len / execution_time / ngpus
            # Save the results to the results dictionary
            result_key = (
                nnodes,
                ngpus_per_node,
                per_device_batch_size,
                gradient_accumulation_steps,
                dp_size,
                fsdp_size,
                pp_size,
                num_ckpt,
            )
            results[result_key] = (
                memory,
                execution_time,
                fit_memory,
                throughput_per_gpu,
            )
            logger.info(
                f"Config: NNodes={nnodes}, NGpusPerNode={ngpus_per_node}, "
                f"PerDeviceBatchSize={per_device_batch_size}, "
                f"GradientAccumulationSteps={gradient_accumulation_steps}, "
                f"DPSize={dp_size}, FSDP={fsdp_size}, PPSize={pp_size}, "
                f"NumCKPT={num_ckpt}, "
            )
            logger.info(f"Memory: {memory:.2f} GB")
            logger.info(f"Execution time: {execution_time:.4f} s")
            logger.info(f"Throughput per GPU: {throughput_per_gpu:.2f}")

    # Filter the results with fit_memory is True and then sort by throughput_per_gpu
    sorted_results = {k: v for k, v in results.items() if v[2]}
    sorted_results = sorted(sorted_results.items(), key=lambda x: x[1][3], reverse=True)
    results = dict(sorted_results)

    # Save results to a CSV file
    result_path = Path("results") / f"{model_name}_{base_model_dtype}.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result_key, (
            memory,
            execution_time,
            fit_memory,
            throughput_per_gpu,
        ) in results.items():
            writer.writerow(
                [
                    *result_key,
                    f"{memory:.2f}",
                    f"{execution_time:.4f}",
                    fit_memory,
                    f"{throughput_per_gpu:.2f}",
                ]
            )
        logger.info(f"Results saved to {result_path}")
