# ruff: noqa: S301
"""Utility functions for PEFT training."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import peft
import torch
from loguru import logger
from megatron.core import parallel_state
from megatron.core.timers import Timers
from packaging import version
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch import nn
from torch.optim import Optimizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    TrainerCallback,
)
from trl import SFTConfig
from trl.trainer.utils import peft_module_casting_to_bf16

from lorafusion.ops.multi_lora import (
    MultiLoRABatchInfo,
    get_multi_lora_manager,
    init_multi_lora_manager,
    prepare_multi_lora_batch_info,
)
from lorafusion.ops.triton_ops.config import get_lora_kernel_config
from lorafusion.patch.patch_lora import apply_lora
from lorafusion.patch.patch_transformers import merge_qkv_proj
from lorafusion.pipeline_parallel.pipe_module import get_pipeline_stage_layer_indices
from lorafusion.utils.benchmark import create_profiler_context
from lorafusion.utils.common import (
    list_of_floats,
    list_of_ints,
    log_memory_usage,
    logging_with_rank,
    str_to_torch_dtype,
)
from lorafusion.utils.gradient_checkpointing import (
    apply_gradient_checkpointing,
    positional_args_call_fn,
)
from lorafusion.utils.hf import create_packed_dummy_inputs
from lorafusion.utils.module import (
    SUPPORTED_LAYER_TYPES,
    SUPPORTED_MODEL_TYPES,
    CompileModelLevel,
    LigerKernelLevel,
    apply_liger_kernel,
    get_submodules_by_type,
    print_trainable_parameters,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lorafusion.solver.proposed_solver import (
        AdapterGroupStepInfo,
        MicroBatchInfo,
    )

_GLOBAL_TIMERS = None


def init_timers() -> Timers:
    """Initialize the timers."""
    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = Timers(log_level=2, log_option="minmax")
    return _GLOBAL_TIMERS


def get_timers() -> Timers:
    """Get the timers."""
    return _GLOBAL_TIMERS


@dataclass
class ModelArguments:
    """Arguments pertaining to finetuning model/config/tokenizer."""

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "huggingface.co/models"
        }
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute and parameter dtype for the model."},
    )
    num_layers_for_debugging: int | None = field(
        default=None,
        metadata={"help": "Number of layers to use for debugging."},
    )
    disable_optimizer_for_debugging: bool = field(
        default=False,
        metadata={"help": "Disable optimizer for debugging."},
    )
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_r: int = field(default=16)
    lora_target_modules: str = field(
        default="all-linear",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
            "or 'all-linear' to apply to all linear layers"
        },
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: str = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_8bit_quantization: bool = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: bool = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    liger_kernel_level: LigerKernelLevel = field(
        default=LigerKernelLevel.DISABLE,
        metadata={"help": "Enables Liger Kernel for training."},
    )
    torch_compile_level: CompileModelLevel = field(
        default=CompileModelLevel.DISABLE,
        metadata={"help": "Enables Torch Compile for training."},
    )
    use_torch_optimizer: bool = field(
        default=True,
        metadata={
            "help": "Use torch optimizer. Torch Fused AdamW is fast and it is "
            "compatible with FSDP2, while Megatron/Apex Optimizer is not."
        },
    )
    merge_qkv_proj: bool = field(
        default=False,
        metadata={"help": "Merge the qkv_proj into the attention module."},
    )
    apply_fused_lora: bool = field(
        default=False,
        metadata={
            "help": "Apply fused LoRA kernels. If False, standard PyTorch operations "
            "are used."
        },
    )
    use_multi_lora: bool = field(
        default=False,
        metadata={"help": "Use multi-LoRA."},
    )
    num_multi_loras: int = field(
        default=1,
        metadata={"help": "Number of multi-LoRA."},
    )
    multi_lora_alpha: str = field(
        default="32.0",
        metadata={"help": "Multi-LoRA alpha."},
    )
    multi_lora_dropout_p: str = field(
        default="0.1",
        metadata={"help": "Multi-LoRA dropout probability."},
    )
    multi_lora_r: str = field(
        default="16",
        metadata={"help": "Multi-LoRA rank."},
    )

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self.dtype = str_to_torch_dtype(self.dtype)
        if self.apply_fused_lora and not self.merge_qkv_proj:
            logger.warning(
                "apply_fused_lora is True, but merge_qkv_proj is False. "
                "Setting merge_qkv_proj to True."
            )
            self.merge_qkv_proj = True

        if isinstance(self.multi_lora_alpha, str):
            self.multi_lora_alpha = list_of_floats(self.multi_lora_alpha)
        if isinstance(self.multi_lora_dropout_p, str):
            self.multi_lora_dropout_p = list_of_floats(self.multi_lora_dropout_p)
        if isinstance(self.multi_lora_r, str):
            self.multi_lora_r = list_of_ints(self.multi_lora_r)


@dataclass
class TrainingArguments(SFTConfig):
    """Arguments pertaining to the training.

    Args:
        per_device_train_batch_size (from transformers.TrainingArguments):
            The batch size per device accelerator core/CPU for training.
        gradient_accumulation_steps (from transformers.TrainingArguments):
            Number of updates steps to accumulate the gradients for, before
            performing a backward/update pass.
        gradient_checkpointing (from transformers.TrainingArguments):
            If True, use gradient checkpointing to save memory at the expense
            of slower backward pass.
        pipeline_parallel_size:
            The number of pipeline parallel GPUs.
        use_reentrant:
            Whether to use reentrant gradient checkpointing.
        gradient_checkpointing_layers:
            The number of layers to enable gradient checkpointing for.
        profile:
            Whether to profile the training.
    """

    global_batch_size: int | None = field(
        default=None,
        metadata={"help": "The global batch size."},
    )
    use_fsdp: bool = field(
        default=False,
        metadata={"help": "Use FSDP for training."},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "The number of pipeline parallel GPUs."},
    )
    use_reentrant: bool = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    gradient_checkpointing_layers: int | None = field(
        default=None,
        metadata={"help": "Enables gradient checkpointing for the specified layers."},
    )
    profile: bool | None = field(
        default=False,
        metadata={"help": "Enables profiling the training."},
    )
    use_timers: bool = field(
        default=False,
        metadata={"help": "Enables timers for the training."},
    )
    multi_lora_max_microbatch_tokens: int | None = field(
        default=None,
        metadata={"help": "Multi-LoRA max microbatch tokens."},
    )
    multi_lora_global_batch_sizes: str = field(
        default="16",
        metadata={
            "help": "Global batch size for each adapter for gradient synchronization."
        },
    )
    benchmark_baseline_mlora_schedule: bool = field(
        default=False,
        metadata={"help": "Benchmark the baseline MLORA schedule."},
    )

    def validate_global_batch_size(self, dp_size: int) -> None:
        """Validate the global batch size.

        Args:
            dp_size: The data parallel size.
        """
        if (
            self.global_batch_size
            != self.per_device_train_batch_size
            * dp_size
            * self.gradient_accumulation_steps
        ):
            msg = (
                f"global_batch_size ({self.global_batch_size}) must be equal to "
                f"per_device_train_batch_size ({self.per_device_train_batch_size}) * "
                f"dp_size ({dp_size}) * gradient_accumulation_steps "
                f"({self.gradient_accumulation_steps})"
            )
            raise ValueError(msg)

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        super().__post_init__()
        if isinstance(self.multi_lora_global_batch_sizes, str):
            self.multi_lora_global_batch_sizes = list_of_ints(
                self.multi_lora_global_batch_sizes
            )


@dataclass
class MockDataArguments:
    """Arguments pertaining to the mock data."""

    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the mock data."},
    )
    dataset_name: str = field(
        default="",
        metadata={"help": "Name of the dataset to use for the mock data."},
    )
    num_samples: int = field(
        default=1000,
        metadata={"help": "Number of samples to use for the mock data."},
    )
    seed_idx: int = field(
        default=0,
        metadata={"help": "Seed index to use for the mock data."},
    )
    permutation_idx: int = field(
        default=0,
        metadata={"help": "Permutation index to use for the mock data."},
    )
    multi_lora_dataset_schedule_path: str = field(
        default="",
        metadata={"help": "Path to the multi-LoRA dataset schedule."},
    )
    use_dummy_fixed_length_dataset: bool = field(
        default=False,
        metadata={"help": "Use a dummy fixed-length dataset."},
    )
    dummy_fixed_length_dataset_length: int = field(
        default=1024,
        metadata={"help": "Length of the dummy fixed-length dataset."},
    )


def _prepare_quantization_config(
    args: ModelArguments,
) -> tuple[BitsAndBytesConfig | None, str | torch.dtype]:
    """Prepare quantization configuration based on model arguments."""
    config = None
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    if args.use_8bit_quantization:
        config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if config is not None and hasattr(config, "bnb_4bit_quant_storage"):
        quant_storage_dtype = config.bnb_4bit_quant_storage
        if quant_storage_dtype and quant_storage_dtype.is_floating_point:
            return config, quant_storage_dtype
    return config, args.dtype


def _get_extra_model_kwargs(args: ModelArguments) -> dict[str, int]:
    """Get extra model kwargs from ModelArguments."""
    if args.num_layers_for_debugging is not None and args.num_layers_for_debugging > 0:
        return {"num_hidden_layers": args.num_layers_for_debugging}
    return {}


def _configure_lora(args: ModelArguments) -> LoraConfig | None:
    """Configure LoRA based on model arguments."""
    if args.lora_r == 0:
        logger.info("LoRA is disabled (rank=0), skipping PEFT configuration")
        return None

    logger.info(f"Configuring LoRA with rank={args.lora_r}")
    target_modules = (
        args.lora_target_modules.split(",")
        if args.lora_target_modules != "all-linear"
        else args.lora_target_modules
    )

    config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    logger.info(
        f"LoRA configuration: alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
        f"target_modules={args.lora_target_modules}"
    )

    return config


def _configure_multi_lora(args: ModelArguments) -> list[LoraConfig]:
    """Configure multi-LoRA based on model arguments."""
    if not args.use_multi_lora:
        msg = "Multi-LoRA is disabled (use_multi_lora=False)."
        raise ValueError(msg)

    len_alpha = len(args.multi_lora_alpha)
    len_dropout_p = len(args.multi_lora_dropout_p)
    len_r = len(args.multi_lora_r)

    if not (args.num_multi_loras == len_alpha == len_dropout_p == len_r):
        msg = (
            "The length of multi_lora_alpha, multi_lora_dropout_p, multi_lora_r "
            "must be the same."
            f"{args.num_multi_loras=}, {len_alpha=}, {len_dropout_p=}, {len_r=}"
        )
        raise ValueError(msg)

    target_modules = (
        args.lora_target_modules.split(",")
        if args.lora_target_modules != "all-linear"
        else args.lora_target_modules
    )

    configs = [
        LoraConfig(
            lora_alpha=args.multi_lora_alpha[i],
            lora_dropout=args.multi_lora_dropout_p[i],
            r=args.multi_lora_r[i],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        for i in range(args.num_multi_loras)
    ]

    logger.info(
        f"Multi-LoRA configuration: "
        f"alpha={args.multi_lora_alpha}, "
        f"dropout_p={args.multi_lora_dropout_p}, "
        f"r={args.multi_lora_r}, "
        f"target_modules={args.lora_target_modules}"
    )

    return configs


def _create_peft_model(
    model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default"
) -> PreTrainedModel:
    """Create a PEFT model with the given configuration.

    Args:
        model: The base model.
        peft_config: The PEFT configuration.
        adapter_name: The name of the adapter.

    Returns:
        The configured PEFT model.
    """
    # Check for sharded QLora
    is_sharded_qlora = getattr(model, "is_loaded_in_4bit", False) and any(
        param.__class__.__name__ == "Params4bit"
        and param.data.device.type in {"cpu", "meta"}
        for _, param in model.named_parameters()
    )

    # Create PEFT model with appropriate settings
    if (
        version.parse(peft.__version__) >= version.parse("0.12")
        and getattr(model, "is_loaded_in_4bit", False)
        and is_sharded_qlora
    ):
        model = get_peft_model(
            model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=False
        )
    else:
        model = get_peft_model(model, peft_config, adapter_name=adapter_name)

    return model


def _enable_input_require_grads(
    model: PreTrainedModel, *, use_reentrant: bool = False
) -> PreTrainedModel:
    """Enable gradient checkpointing for the model.

    Args:
        model: The model to enable gradient checkpointing for.
        use_reentrant: Whether to use reentrant gradient checkpointing.

    Returns:
        The model with gradient checkpointing enabled.
    """
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(
                module: nn.Module, input_: torch.Tensor, output: torch.Tensor
            ) -> None:
                """Make the inputs require grad.

                Args:
                    module: The module to make the inputs require grad for.
                    input_: The input to the module.
                    output: The output of the module.
                """
                del module, input_  # Unused parameters
                output.requires_grad_(mode=True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def create_model(
    model_args: ModelArguments,
    training_args: TrainingArguments | None = None,
    **extra_model_kwargs,
) -> tuple[AutoModelForCausalLM, PeftConfig | None, AutoTokenizer]:
    """Create and prepare model for training.

    Args:
        model_args: Model arguments.
        training_args: Training arguments.
        extra_model_kwargs: Extra model kwargs.

    Returns:
        A tuple of the model, PEFT configuration (or None if LoRA rank = 0),
        and tokenizer.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Pack the liger kernel if specified
    if model_args.liger_kernel_level is not None:
        apply_liger_kernel(model_args.liger_kernel_level)

    # Apply fused LoRA
    if model_args.apply_fused_lora or model_args.use_multi_lora:
        apply_lora(use_fused=model_args.apply_fused_lora)

    # Prepare quantization config
    bnb_config, torch_dtype = _prepare_quantization_config(model_args)
    extra_model_kwargs.update(_get_extra_model_kwargs(model_args))

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation=(
            "flash_attention_2" if model_args.use_flash_attn else "eager"
        ),
        torch_dtype=torch_dtype,
        **extra_model_kwargs,
    )

    # Merge qkv_proj if specified
    if model_args.apply_fused_lora or model_args.merge_qkv_proj:
        if (
            parallel_state.is_initialized()
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
        ):
            num_pipeline_stages = (
                parallel_state.get_pipeline_model_parallel_world_size()
            )
            stage_idx = parallel_state.get_pipeline_model_parallel_rank()
            layer_idx_start, layer_idx_end = get_pipeline_stage_layer_indices(
                num_layers=model.config.num_hidden_layers,
                num_stages=num_pipeline_stages,
                stage_idx=stage_idx,
            )
            layer_indices = list(range(layer_idx_start, layer_idx_end))
        else:
            layer_indices = None

        model = merge_qkv_proj(model, layer_indices=layer_indices)

    # Configure LoRA
    if not model_args.use_multi_lora:
        peft_config = _configure_lora(model_args)
        # Create the LoRA model
        lora_model = _create_peft_model(model, peft_config)
    elif training_args is None:
        msg = "Multi-LoRA is enabled, but training_args is not provided."
        raise ValueError(msg)
    else:
        # Create multi-LoRA model
        peft_configs = _configure_multi_lora(model_args)
        lora_model = _create_peft_model(model, peft_configs[0], adapter_name="lora_0")
        for i in range(1, model_args.num_multi_loras):
            lora_model.add_adapter(
                peft_config=peft_configs[i],
                adapter_name=f"lora_{i}",
            )
        lora_model.base_model.set_adapter(
            [f"lora_{i}" for i in range(model_args.num_multi_loras)]
        )

        # Init the multi-LoRA manager
        init_multi_lora_manager(
            peft_configs,
            num_pipeline_stages=training_args.pipeline_parallel_size,
            multi_lora_max_microbatch_tokens=training_args.multi_lora_max_microbatch_tokens,
            multi_lora_global_batch_sizes=training_args.multi_lora_global_batch_sizes,
        )

    return lora_model, tokenizer


def _prepare_model_with_gradient_checkpointing(  # noqa: C901
    model: PreTrainedModel, training_args: SFTConfig
) -> PreTrainedModel:
    """Prepare model with gradient checkpointing if needed.

    Args:
        model: The model to prepare.
        training_args: Training arguments.

    Returns:
        The prepared model.
    """
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": training_args.use_reentrant
        }

    if (
        training_args.gradient_checkpointing
        and training_args.gradient_checkpointing_layers is not None
        and training_args.gradient_checkpointing_layers == 0
    ):
        msg = (
            "Gradient checkpointing is enabled, but gradient_checkpointing_layers is "
            "set to 0. This means that gradient checkpointing is disabled."
        )
        logger.warning(msg)
        training_args.gradient_checkpointing_layers = None

    # Detect if model is quantized
    is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    )

    is_sharded_qlora = False
    if getattr(model, "is_loaded_in_4bit", False):
        # Check if model is sharded (FSDP/DS-Zero3)
        for _, param in model.named_parameters():
            if param.__class__.__name__ == "Params4bit":
                is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                break

    # Prepare model for kbit training if needed
    if is_qlora and not is_sharded_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
            or {},
        )
    elif training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
        )
        _enable_input_require_grads(model, use_reentrant=training_args.use_reentrant)

    if (
        training_args.gradient_checkpointing
        and training_args.gradient_checkpointing_layers is not None
        and training_args.gradient_checkpointing_layers > 0
    ):
        # TODO(zhanda): Temporarily because we don't want to change the source
        # code of the transformers library, so we patch the forward method of the
        # decoder layer to make sure that the gradient checkpointing is configured
        # correctly for different layers.
        # Specifically, (1) before we do anything, the path called in `LlamaModdel`
        # is self._gradient_checkpointing_func, which is checkpoint function in
        # torch. We first collect that gradient checkpointing function, and then
        # patch the _gradient_checkpointing_func to be a simple positional_args_call.
        # (2) we patch the first $k$ layers to be the gradient checkpointing function.
        # and patch the others to be a simple call.
        gradient_checkpointing_func = None
        if hasattr(model, "_gradient_checkpointing_func"):
            gradient_checkpointing_func = (
                model._gradient_checkpointing_func  # noqa: SLF001
            )
            model._gradient_checkpointing_func = positional_args_call_fn  # noqa: SLF001
        for module in model.modules():
            if hasattr(module, "_gradient_checkpointing_func"):
                gradient_checkpointing_func = (
                    module._gradient_checkpointing_func  # noqa: SLF001
                )
                module._gradient_checkpointing_func = (  # noqa: SLF001
                    positional_args_call_fn
                )
        if gradient_checkpointing_func is None:
            msg = "Gradient checkpointing function not found"
            logger.error(msg)
            raise ValueError(msg)

        # Extract the layers to be patched
        layers = get_submodules_by_type(model, SUPPORTED_LAYER_TYPES)
        # Patch the first $k$ layers to be the gradient checkpointing function.
        layers_to_be_patched = list(layers.values())[
            : training_args.gradient_checkpointing_layers
        ]
        apply_gradient_checkpointing(
            model, layers_to_be_patched, gradient_checkpointing_func
        )

    return model


def _prepare_model_for_bf16_casting(
    model: PreTrainedModel, training_args: SFTConfig
) -> PreTrainedModel:
    """Prepare model for bf16 casting if needed.

    Args:
        model: The model to prepare for bf16 casting.
        training_args: Training arguments.

    Returns:
        The prepared model.
    """
    is_sharded_qlora = False
    if getattr(model, "is_loaded_in_4bit", False):
        # Check if model is sharded (FSDP/DS-Zero3)
        for _, param in model.named_parameters():
            if param.__class__.__name__ == "Params4bit":
                is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                break
    if (
        training_args.bf16
        and getattr(model, "is_loaded_in_4bit", False)
        and not is_sharded_qlora
    ):
        peft_module_casting_to_bf16(model)


def prepare_model_for_training(
    model: PreTrainedModel,
    model_args: ModelArguments,
    training_args: SFTConfig,
) -> PreTrainedModel:
    """Prepare a PEFT model for training.

    Args:
        model: The model to prepare for PEFT.
        peft_config: The PEFT configuration.
        model_args: The model arguments.
        training_args: The training arguments.

    Returns:
        The prepared model.
    """
    # Prepare the gradient checkpointing
    _prepare_model_with_gradient_checkpointing(model, training_args)

    # Handle bf16 casting for 4-bit models
    _prepare_model_for_bf16_casting(model, training_args)

    # Main logic (change dtyoe and freeze base model parameters)
    model.train()
    model.config.use_cache = False
    model = model.to(model_args.dtype)
    submodules = get_submodules_by_type(model, SUPPORTED_MODEL_TYPES)
    if len(submodules) != 1:
        msg = f"Expected exactly one model core, but got {len(submodules)}"
        raise ValueError(msg)

    return model


def print_model_param_info(
    model: nn.Module,
    layer_pattern: str | None = "layers.*",
    num_layers_to_show: int = 1,
) -> None:
    """Print detailed parameter information for a model's layers.

    This function prints parameter information for specified layers of a model,
    including shape, gradient status, and data type. It can filter layers using
    a pattern and limit the number of layers shown.

    Args:
        model: The model to analyze.
        layer_pattern: Regex pattern to match layer names. Defaults to "layers.*".
            Use None to show all parameters.
        num_layers_to_show: Number of matching layers to display. Defaults to 1.
    """
    if layer_pattern is not None:
        layer_patterns_to_show = [
            layer_pattern.replace("*", str(i)) for i in range(num_layers_to_show)
        ]

        # Filter parameters based on layer pattern
        params_to_show = {
            name: param
            for name, param in model.named_parameters()
            if not re.search(layer_pattern, name)
            or any(re.search(pattern, name) for pattern in layer_patterns_to_show)
        }
    else:
        params_to_show = dict(model.named_parameters())

    # Print parameter information
    for name, param in params_to_show.items():
        logging_with_rank(
            f"[Parameter: {name}]"
            f"{tuple(param.shape)}, "
            f"stride: {tuple(param.stride())}, "
            f"{param.dtype}, "
            f"{param.device}, "
            f"requires_grad: {param.requires_grad}, "
            f"sum: {param.sum().item():.4f}"
        )

    # Print summary statistics
    print_trainable_parameters(model)
    log_memory_usage("after parameter info logging")


class ProfCallback(TrainerCallback):
    """A callback that profiles the model."""

    def __init__(self, prof: torch.profiler.profile) -> None:
        """Initialize the ProfCallback with a torch.profiler.profile."""
        super().__init__()
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs) -> None:  # noqa: ANN001
        """Step end callback."""
        self.prof.step()


class EmptyOptimizer(Optimizer):
    """A no-op optimizer used when there are no trainable parameters.

    This optimizer is used when LoRA is disabled (rank=0) and there are no
    trainable parameters in the model. It implements the Optimizer interface
    but doesn't perform any actual optimization.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the EmptyOptimizer with a dummy parameter."""
        super().__init__([torch.nn.Parameter()], {})
        self.param_groups = []  # Remove all parameter groups

    def step(self, closure: None = ...) -> None:  # type: ignore[override]
        """No-op step method."""

    def zero_grad(self) -> None:
        """No-op zero_grad method."""


def create_optimizer(
    model: nn.Module,
    peft_config: PeftConfig | None,
    *,
    disable_optimizer: bool = False,
) -> torch.optim.Optimizer:
    """Create an optimizer based on whether we're using LoRA.

    Args:
        model: The model to optimize.
        peft_config: PEFT configuration if using LoRA.
        disable_optimizer: Whether to disable the optimizer.

    Returns:
        The optimizer.
    """
    if peft_config is None or disable_optimizer:
        return EmptyOptimizer()

    # When using LoRA, create a real optimizer for trainable parameters
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], capturable=True
    )


class MockDataset:
    """A mock dataset that returns a fixed set of length of sequences."""

    def __init__(self, sample_lengths: list[int]) -> None:
        """Initialize the MockDataset with a fixed set of length of sequences.

        Args:
            sample_lengths: The length of the sequences to sample.
        """
        self.sample_lengths = sample_lengths
        self.stats: dict[str, float] = {
            "median": np.median(self.sample_lengths),
            "mean": np.mean(self.sample_lengths),
            "std": np.std(self.sample_lengths),
            "min": np.min(self.sample_lengths),
            "max": np.max(self.sample_lengths),
        }

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.sample_lengths)

    def __getitem__(self, idx: int | slice) -> list[int] | int:
        """Get the sample at the specified index or slice.

        Args:
            idx: The index or slice to get.

        Returns:
            The sample(s) at the specified index or slice.
        """
        return self.sample_lengths[idx]

    def shuffle(self, seed: int | None = None) -> None:
        """Shuffle the dataset."""
        rng = np.random.default_rng(seed)
        self.sample_lengths = rng.permutation(self.sample_lengths).tolist()

    @classmethod
    def from_dataset_args(cls, dataset_args: MockDataArguments) -> MockDataset:
        """Initialize the MockDataset with a fixed set of length of sequences.

        Args:
            dataset_args: The dataset arguments.
        """
        with Path(dataset_args.dataset_path).open("r") as f:
            dataset_distributions = json.load(f)

        seed_to_data_dict = dataset_distributions[dataset_args.dataset_name]
        seed = seed_to_data_dict["seeds"][dataset_args.seed_idx]
        data_dict = seed_to_data_dict[f"seed_{seed}"]
        sample_lengths = data_dict[f"permutation_{dataset_args.permutation_idx + 1}"]

        logger.success(
            f"Loaded {len(sample_lengths)} samples from {dataset_args.dataset_path}. "
            f"Dataset name: {dataset_args.dataset_name}, "
            f"Seed Index: {dataset_args.seed_idx}, "
            f"Seed: {seed}, "
            f"Permutation Index: {dataset_args.permutation_idx + 1}"
        )
        return cls(sample_lengths)

    @classmethod
    def from_a_fixed_length(cls, length: int, num_samples: int) -> MockDataset:
        """Initialize the MockDataset with a fixed set of length of sequences.

        Args:
            length: The length of the sequences to sample.
            num_samples: The number of samples to sample.
        """
        return cls([length] * num_samples)


class DataProvider:
    """A data provider that returns a fixed set of length of sequences."""

    def __init__(
        self,
        dataset: MockDataset,
        *,
        batch_size: int,
        hidden_size: int,
        parallel_state: Any,  # noqa: ANN401
        return_inputs_embeds: bool = False,
        dp_rank: int = 0,
        dp_size: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialize the DataProvider with a MockDataset.

        Args:
            dataset: The dataset to provide data from.
            batch_size: The batch size per data parallel rank.
            hidden_size: The hidden size of the model.
            parallel_state: The parallel state object.
            return_inputs_embeds: Whether to return input embeddings.
            dp_rank: The data parallel rank of the current process.
            dp_size: Total number of data parallel processes.
            verbose: Whether to print verbose logging.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.parallel_state = parallel_state
        self.return_inputs_embeds = return_inputs_embeds
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.verbose = verbose

        # Initialize iteration state
        # idx is the index of the current idx in the dataset (considering all ranks)
        # e.g. idx is the same across all dp ranks
        self.idx = 0
        self.processed_tokens = 0

    def num_global_batches(self, global_batch_size: int) -> int:
        """Get the number of global batches."""
        return len(self.dataset) // global_batch_size

    def peek_batch(self) -> list[int]:
        """Preview the next batch of sequence lengths without advancing the index.

        Returns:
            List of sequence lengths for the next batch specific to this dp_rank.

        Raises:
            StopIteration: If there are no more batches left.
        """
        if self.idx + self.batch_size * self.dp_size > len(self.dataset):
            msg = (
                f"No more batches left. "
                f"{self.idx=}, {self.batch_size=}, {self.dp_size=}, "
                f"{len(self.dataset)=}"
            )
            raise StopIteration(msg)

        # Calculate start and end indices for this dp_rank
        start_idx = self.idx + self.dp_rank * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))

        return self.dataset[start_idx:end_idx]

    def next_batch(self) -> tuple[torch.Tensor, dict[str, torch.Tensor], None]:
        """Get the next batch of length of sequences and advance the index.

        Returns:
            A tuple of the input tensor and the shared inputs.

        Raises:
            StopIteration: If there are no more batches left.
        """
        seq_len_list = self.peek_batch()
        self.processed_tokens += sum(seq_len_list)

        if self.verbose:
            logging_with_rank(
                f"|- Batch {self.idx + self.dp_rank * self.batch_size}, "
                f"batch_size: {self.batch_size}, "
                f"seq_len_list: {seq_len_list}, "
                f"sum(seq_len_list): {sum(seq_len_list)}."
            )

        # Advance the global index after all ranks have processed their batches
        # Again, self.idx is the same across all dp ranks
        self.idx += self.batch_size * self.dp_size

        all_inputs = create_packed_dummy_inputs(
            hidden_size=self.hidden_size,
            seq_len_list=seq_len_list,
            return_inputs_embeds=self.return_inputs_embeds,
            return_labels=True,
        )

        # Move inputs to GPU
        for input_name, input_value in all_inputs.items():
            if isinstance(input_value, torch.Tensor):
                all_inputs[input_name] = input_value.to("cuda")

        # Extract inputs
        # Keys that are shared across all pipeline stages
        shared_keys = [
            "position_ids",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
        ]
        if self.parallel_state is None or self.parallel_state.is_pipeline_last_stage():
            shared_keys.append("labels")
        if not self.return_inputs_embeds:
            input_ = all_inputs["input_ids"]
        else:
            input_ = all_inputs["inputs_embeds"]
        shared_inputs = {k: v for k, v in all_inputs.items() if k in shared_keys}
        return input_, shared_inputs, None

    def reset(self) -> None:
        """Reset the dataset provider."""
        self.idx = 0


class MockMultiLoRADataset:
    """A mock dataset that returns the inputs for multi-LoRA."""

    def __init__(self, micro_batch_infos: list[MicroBatchInfo]) -> None:
        """Initialize the MockMultiLoRADataset with a list of MicroBatchInfo.

        Args:
            micro_batch_infos: The list of MicroBatchInfo.
        """
        self.micro_batch_infos = micro_batch_infos

    def __len__(self) -> int:
        """Get the number of micro-batches in the dataset."""
        return len(self.micro_batch_infos)

    def __getitem__(self, idx: int) -> MicroBatchInfo:
        """Get the sample at the specified index."""
        if not isinstance(idx, int):
            msg = (
                f"idx must be an integer, but got {type(idx)}. "
                "If it is a slice, I don't think MultiLoRA Dataset supports that."
            )
            raise TypeError(msg)
        return self.micro_batch_infos[idx]

    @classmethod
    def from_dataset_args(cls, dataset_args: MockDataArguments) -> MockMultiLoRADataset:
        """Initialize the MockMultiLoRADataset with a MockDataArguments."""
        multi_lora_dataset_schedule_path = Path(
            dataset_args.multi_lora_dataset_schedule_path
        )
        if not multi_lora_dataset_schedule_path.exists():
            msg = (
                f"Multi-LoRA dataset schedule path {multi_lora_dataset_schedule_path} "
                "does not exist."
            )
            raise FileNotFoundError(msg)

        # We use pickle to load for the simplicity, it should be loaded as a json file
        # as we have also provided a json file for easy inspection
        with multi_lora_dataset_schedule_path.open("rb") as f:
            adapter_group_step_infos: list[AdapterGroupStepInfo] = pickle.load(f)

        micro_batch_infos: list[MicroBatchInfo] = []
        for adapter_group_step_info in adapter_group_step_infos:
            micro_batch_infos.extend(adapter_group_step_info.micro_batch_infos)

        return cls(micro_batch_infos)


class MockMultiLoRADataProvider:
    """A data provider that returns the inputs for multi-LoRA."""

    def __init__(
        self,
        dataset: MockMultiLoRADataset,
        *,
        batch_size: int,
        hidden_size: int,
        parallel_state: Any,  # noqa: ANN401
        return_inputs_embeds: bool = False,
        dp_rank: int = 0,
        dp_size: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialize the MockMultiLoRADataProvider with a MockMultiLoRADataset."""
        if batch_size != 1:
            msg = (
                f"batch_size must be 1 for multi-LoRA, but got {batch_size}. "
                "Because the data has already been packed into micro-batches."
            )
            raise ValueError(msg)

        self.dataset = dataset
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.parallel_state = parallel_state
        self.return_inputs_embeds = return_inputs_embeds
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.verbose = verbose

        # Initialize iteration state
        self.idx = 0
        self.processed_tokens = 0

    def num_global_batches(self, global_batch_size: int) -> int:
        """Get the number of micro-batches in the dataset.

        Note that the global batch size does not matter for multi-LoRA balanced
        """
        del global_batch_size
        return len(self.dataset) // self.dp_size

    def peek_batch(self) -> list[int]:
        """Preview the next batch of sequence lengths without advancing the index.

        Returns:
            List of sequence lengths for the next batch specific to this dp_rank.
        """
        return [self.dataset[self.idx + self.dp_rank].padded_total_tokens]

    def next_batch(  # noqa: C901
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], MultiLoRABatchInfo]:
        """Get the next batch of sequence lengths and advance the index.

        For multi-LoRA, we output:
        - input_ids and the shared inputs (similar to DataProvider.next_batch)
        - MultiLoRABatchInfo containing information about adapters and their
            configurations

        Returns:
            A tuple of the input tensor, shared inputs, and MultiLoRABatchInfo.

        Raises:
            StopIteration: If there are no more batches left.
        """
        # Get the current micro batch
        micro_batch_info = self.dataset[self.idx + self.dp_rank]
        self.processed_tokens += micro_batch_info.total_tokens

        if self.verbose:
            logging_with_rank(
                f"|- Multi-LoRA Batch {self.idx + self.dp_rank}, "
                f"padded_total_tokens: {micro_batch_info.padded_total_tokens}, "
                f"total_tokens: {micro_batch_info.total_tokens}"
            )

        # Advance the index for the next batch
        self.idx += 1 * self.dp_size

        if micro_batch_info.is_empty_marker:
            return None, None, None

        # Prepare seq_len_list for create_packed_dummy_inputs
        # We need to group samples by adapter and add padding if necessary
        seq_len_list = []

        # Get the multi-LoRA manager for adapter configurations
        multi_lora_manager = get_multi_lora_manager()
        if multi_lora_manager is None:
            msg = "Multi-LoRA manager is not initialized."
            raise RuntimeError(msg)

        # Process each adapter in the micro batch
        sorted_adapter_indices = sorted(micro_batch_info.adapter_group_info)
        for adapter_idx in sorted_adapter_indices:
            # Get all samples for this adapter
            adapter_samples = micro_batch_info.adapter_token_lengths_pairs[adapter_idx]

            # Sum actual tokens for this adapter
            adapter_tokens = micro_batch_info.adapter_num_tokens_pairs[adapter_idx]

            # Get padded tokens for this adapter
            padded_tokens = micro_batch_info.padded_adapter_num_tokens_pairs[
                adapter_idx
            ]

            # Calculate padding needed
            padding_size = padded_tokens - adapter_tokens

            # Add all the sample lengths for this adapter
            seq_len_list.extend(adapter_samples)

            # Add padding sample if needed
            if padding_size > 0:
                seq_len_list.append(padding_size)

        # Create model inputs
        all_inputs = create_packed_dummy_inputs(
            hidden_size=self.hidden_size,
            seq_len_list=seq_len_list,
            return_inputs_embeds=self.return_inputs_embeds,
            return_labels=True,
        )

        # Move inputs to GPU
        for input_name, input_value in all_inputs.items():
            if isinstance(input_value, torch.Tensor):
                all_inputs[input_name] = input_value.to("cuda")

        # Extract inputs - shared keys similar to DataProvider.next_batch
        shared_keys = [
            "position_ids",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
        ]
        if self.parallel_state is None or self.parallel_state.is_pipeline_last_stage():
            shared_keys.append("labels")

        if not self.return_inputs_embeds:
            input_ = all_inputs["input_ids"]
        else:
            input_ = all_inputs["inputs_embeds"]

        shared_inputs = {k: v for k, v in all_inputs.items() if k in shared_keys}

        # Prepare MultiLoRABatchInfo
        # Get unpadded sequence lengths for each adapter
        unpadded_seq_len_list = []
        lora_idx_list = []
        lora_rank_list = []
        dropout_p_list = []
        alpha_list = []

        # For each adapter, get its configuration details
        for adapter_idx in sorted_adapter_indices:
            # Get the actual tokens for this adapter (unpadded)
            adapter_tokens = micro_batch_info.adapter_num_tokens_pairs[adapter_idx]
            unpadded_seq_len_list.append(adapter_tokens)

            # Get the LoRA config for this adapter
            lora_config = multi_lora_manager.lora_configs[adapter_idx]
            lora_idx_list.append(adapter_idx)
            lora_rank_list.append(lora_config.r)
            dropout_p_list.append(lora_config.lora_dropout)
            alpha_list.append(lora_config.lora_alpha)

        # Create MultiLoRABatchInfo
        multi_lora_batch_info = prepare_multi_lora_batch_info(
            seq_len_list=unpadded_seq_len_list,
            lora_idx_list=lora_idx_list,
            lora_rank_list=lora_rank_list,
            dropout_p_list=dropout_p_list,
            alpha_list=alpha_list,
            block_size_m=get_lora_kernel_config("fused_multi_lora_block_size_m"),
            micro_batch_info=micro_batch_info,
        )

        # Add the batch info to the MultiLoRAManager's deque
        multi_lora_manager.add_batch_info(multi_lora_batch_info)

        return input_, shared_inputs, multi_lora_batch_info

    def reset(self) -> None:
        """Reset the dataset provider."""
        self.idx = 0

        multi_lora_manager = get_multi_lora_manager()
        if multi_lora_manager is not None:
            multi_lora_manager.clear_batch_info()


def _eager_train_step(
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor | float | int],
    optimizer: torch.optim.Optimizer,
) -> None:
    """Eagerly compute the forward and backward pass.

    Args:
        model: The model to train.
        inputs: The inputs to the model.
        optimizer: The optimizer to use.
    """
    with torch.profiler.record_function("Forward"):
        hidden_states = model(**inputs)[0]
        loss = (hidden_states**2).mean()

    with torch.profiler.record_function("Backward"):
        loss.backward()

    with torch.profiler.record_function("Optimizer step"):
        optimizer.step()
        optimizer.zero_grad()


class DummyDatasetProvider:
    """A dummy dataset provider that returns a fixed set of length of sequences."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the DummyDatasetProvider with a fixed set of length of sequences.

        Args:
            num_samples: Number of samples to generate.
            seed: Seed for reproducibility.
        """
        self.seed = seed
        self.sampled = []
        self.idx = 0

    def sample_uniform_(
        self, num_samples: int, low: int, high: int, seed: int | None = None
    ) -> np.ndarray:
        """Generate samples from a uniform integer distribution.

        Args:
            num_samples: Number of samples to generate.
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).
            seed: Seed for reproducibility.

        Returns:
            np.ndarray: Array of uniformly distributed integers.
        """
        effective_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(effective_seed)
        self.sampled = rng.integers(low, high + 1, size=num_samples)
        return self.sampled

    def sample_normal_(
        self,
        num_samples: int,
        mean: float,
        std: float,
        low: int | None = None,
        high: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate samples from a normal distribution rounded to integers.

        Args:
            num_samples: Number of samples to generate.
            mean: Mean of the distribution.
            std: Standard deviation of the distribution.
            low: Minimum value (inclusive).
            high: Maximum value (inclusive).
            seed: Seed for reproducibility.

        Returns:
            np.ndarray: Array of integers from the normal distribution.
        """
        effective_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(effective_seed)
        samples = rng.normal(mean, std, size=num_samples)
        samples = np.round(samples).astype(int)

        if low is not None:
            samples = np.maximum(samples, low)
        if high is not None:
            samples = np.minimum(samples, high)

        self.sampled = samples

        return samples

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.sampled)

    def next(self, batch_size: int) -> np.ndarray:
        """Get the next batch of samples."""
        if self.idx >= len(self.sampled):
            self.idx = 0
        samples = self.sampled[self.idx : min(self.idx + batch_size, len(self.sampled))]
        self.idx += batch_size
        return samples

    def reset(self) -> None:
        """Reset the dataset provider."""
        self.idx = 0


def prepare_train_step_and_input_provider(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    hidden_size: int,
) -> tuple[
    Callable[[dict[str, torch.Tensor | float | int]], None],
    Callable[[], dict[str, torch.Tensor | float | int]],
]:
    """Prepare a train step for the model.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        hidden_size: The hidden size of the model.
        training_args: Training arguments.

    Returns:
        A tuple of the train step and the input provider.

    Example:
        >>> train_step, input_provider = prepare_train_step_and_input_provider(
        >>>     model, optimizer, hidden_size, use_cuda_graph=use_cuda_graph
        >>> )
        >>> inputs = input_provider()
        >>> train_step(inputs)
    """
    dataset_provider = DummyDatasetProvider()
    dataset_provider.sample_normal_(num_samples=1024, mean=1024, std=32)

    def train_step(inputs: dict[str, torch.Tensor | float | int]) -> None:
        _eager_train_step(model, inputs, optimizer)

    def input_provider() -> dict[str, torch.Tensor | float | int]:
        return create_packed_dummy_inputs(hidden_size, training_args)

    return train_step, input_provider


if __name__ == "__main__":
    test_single_model = False
    if test_single_model:
        model_args = ModelArguments(
            model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        )
        training_args = TrainingArguments(
            gradient_checkpointing=True,
            gradient_checkpointing_layers=32,
        )
        model, tokenizer = create_model(model_args)
        model = prepare_model_for_training(model, model_args, training_args)
        print_model_param_info(model)

        for _ in range(3):
            inputs = create_packed_dummy_inputs(
                hidden_size=model.config.hidden_size,
                seq_len_list=[1024, 1024],
                return_input_ids=True,
                return_labels=True,
            )
            outputs = model(**inputs)
            outputs[0].backward()

        log_memory_usage("after `outputs = model(**inputs)`")

    test_multi_lora_model_loading = True
    if test_multi_lora_model_loading:
        num_multi_loras = 4
        model_args = ModelArguments(
            model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            apply_fused_lora=True,
            use_multi_lora=True,
            num_multi_loras=num_multi_loras,
            multi_lora_alpha=[32.0 for _ in range(num_multi_loras)],
            multi_lora_dropout_p=[0.1 for _ in range(num_multi_loras)],
            multi_lora_r=[16 for _ in range(num_multi_loras)],
        )
        training_args = TrainingArguments(
            gradient_checkpointing=True,
            gradient_checkpointing_layers=32,
            multi_lora_max_microbatch_tokens=4096,
        )
        model, tokenizer = create_model(model_args, training_args=training_args)
        model = prepare_model_for_training(model, model_args, training_args)
        model = model.to("cuda")
        print_model_param_info(model)
        log_memory_usage("after loading model")

    test_multi_lora_dataset_loading = True
    if test_multi_lora_dataset_loading:
        dataset_args = MockDataArguments(
            multi_lora_dataset_schedule_path="benchmarks_paper/datasets/schedules/schedule.pkl",
        )
        multi_lora_dataset = MockMultiLoRADataset.from_dataset_args(dataset_args)
        data_provider = MockMultiLoRADataProvider(
            dataset=multi_lora_dataset,
            batch_size=1,
            hidden_size=model.config.hidden_size,
            parallel_state=None,
        )

        steps = 50
        cuda_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        cuda_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        use_multi_lora_or_not = [None for _ in range(steps)]
        effective_tokens = [None for _ in range(steps)]

        with create_profiler_context(profile=False, skip_first=0) as prof:
            for step in range(steps):
                batch = data_provider.next_batch()

                input_, shared_inputs, multi_lora_batch_info = batch
                multi_lora_manager = get_multi_lora_manager()
                use_multi_lora_or_not[step] = (
                    multi_lora_manager.get_oldest_batch_info().num_active_adapters > 1
                )
                effective_tokens[step] = sum(multi_lora_batch_info.seq_len_list)

                cuda_start_events[step].record()

                output = model(input_, **shared_inputs)
                output[0].backward()
                multi_lora_manager.pop_oldest_batch_info()

                cuda_end_events[step].record()

                if prof:
                    prof.step()

        torch.cuda.synchronize()

        # Summarize the results for multi_lora and without multi_lora
        multi_lora_times = []
        multi_lora_tokens = []
        no_multi_lora_times = []
        no_multi_lora_tokens = []
        for step in range(steps):
            if use_multi_lora_or_not[step]:
                multi_lora_times.append(
                    cuda_start_events[step].elapsed_time(cuda_end_events[step])
                )
                multi_lora_tokens.append(effective_tokens[step])
            else:
                no_multi_lora_times.append(
                    cuda_start_events[step].elapsed_time(cuda_end_events[step])
                )
                no_multi_lora_tokens.append(effective_tokens[step])
        multi_lora_times = multi_lora_times[3:]
        no_multi_lora_times = no_multi_lora_times[3:]
        multi_lora_tokens = multi_lora_tokens[3:]
        no_multi_lora_tokens = no_multi_lora_tokens[3:]
        logger.info(
            f"[Multi-LoRA] "
            f"Mean time: {np.mean(multi_lora_times):.2f}, "
            f"Mean effective tokens: {np.mean(multi_lora_tokens):.2f}, "
            f"Mean effective throughput: "
            f"{np.mean(multi_lora_tokens) / np.mean(multi_lora_times):.2f}"
        )
        logger.info(
            f"[No Multi-LoRA] "
            f"Mean time: {np.mean(no_multi_lora_times):.2f}, "
            f"Mean effective tokens: {np.mean(no_multi_lora_tokens):.2f}, "
            f"Mean effective throughput: "
            f"{np.mean(no_multi_lora_tokens) / np.mean(no_multi_lora_times):.2f}"
        )
