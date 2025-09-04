"""Utilities for modules manipulation."""

from enum import StrEnum

import torch
from loguru import logger
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import (
    LlamaForCausalLM,
    LlamaModel,
    MixtralModel,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer

from lorafusion.utils.common import logging_with_rank

COMMON_MODEL_CLASSES = (LlamaForCausalLM, Qwen2ForCausalLM)
COMMON_ATTENTION_CLASSES = (LlamaAttention, Qwen2Attention)
COMMON_MODEL_CLASS_TYPES = LlamaForCausalLM | Qwen2ForCausalLM
COMMON_ATTENTION_CLASS_TYPES = LlamaAttention | Qwen2Attention

SUPPORTED_MODEL_TYPES = (LlamaModel, MixtralModel, Qwen2Model)
SUPPORTED_LAYER_TYPES = (LlamaDecoderLayer, MixtralDecoderLayer, Qwen2DecoderLayer)
SUPPORTED_LORA_TYPES = (LoraLayer,)


class LigerKernelLevel(StrEnum):
    """The level of Liger kernel to apply."""

    ALL = "all"
    NO_RMS_NORM = "no_rms_norm"
    DISABLE = "disable"


def apply_liger_kernel(level: LigerKernelLevel = LigerKernelLevel.ALL) -> None:
    """Apply Liger kernel to the model.

    Args:
        level: The level of Liger kernel to apply.
    """
    from liger_kernel.transformers import (
        apply_liger_kernel_to_llama,
        apply_liger_kernel_to_mixtral,
        apply_liger_kernel_to_qwen2,
    )

    default_kwargs = {"cross_entropy": False, "fused_linear_cross_entropy": False}

    def _apply_liger_kernel(*, rms_norm: bool = True) -> None:
        apply_liger_kernel_to_llama(rms_norm=rms_norm, **default_kwargs)
        apply_liger_kernel_to_qwen2(rms_norm=rms_norm, **default_kwargs)
        apply_liger_kernel_to_mixtral(rms_norm=rms_norm, **default_kwargs)

    apply_dict = {
        LigerKernelLevel.DISABLE: lambda: None,
        LigerKernelLevel.ALL: lambda: _apply_liger_kernel(),
        LigerKernelLevel.NO_RMS_NORM: lambda: _apply_liger_kernel(rms_norm=False),
    }

    if level not in apply_dict:
        msg = f"Invalid Liger kernel level: {level}"
        raise ValueError(msg)

    apply_dict[level]()


class CompileModelLevel(StrEnum):
    """The level of compilation to apply."""

    MODEL = "model"
    LAYER = "layer"
    LORA = "lora"
    DISABLE = "disable"


def compile_model(model: nn.Module, level: CompileModelLevel | None) -> nn.Module:
    """Compile the model.

    Args:
        model: The model to compile.
        level: The level of compilation to apply.

    Returns:
        The compiled model.
    """
    if level is None or level == CompileModelLevel.DISABLE:
        return model

    def compile_submodules(
        module: nn.Module, target_types: type | tuple[type, ...]
    ) -> None:
        """Compile all the submodules of the given module.

        Args:
            module: The module to compile.
            target_types: The types of the submodules to compile.
        """
        target_types = (
            target_types if isinstance(target_types, tuple) else (target_types,)
        )
        for name, submodule in module.named_children():
            if isinstance(submodule, target_types):
                compiled_submodule = torch.compile(submodule, mode="reduce-overhead")
                setattr(module, name, compiled_submodule)
            else:
                compile_submodules(submodule, target_types)

    if level == CompileModelLevel.MODEL:
        model = torch.compile(model)
    if level == CompileModelLevel.LAYER:
        compile_submodules(model, SUPPORTED_LAYER_TYPES)
    if level == CompileModelLevel.LORA:
        compile_submodules(model, SUPPORTED_LORA_TYPES)

    logger.info(f"Compiled model with level: {level}")

    return model


def get_nb_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Get the number of trainable parameters and all parameters in the model.

    From PeftModel.get_nb_trainable_parameters.

    Args:
        model: The nn.Module.

    Returns:
        A tuple of the number of trainable parameters and the total number
        of parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model: nn.Module) -> None:
    """Print the trainable parameters of the model.

    From PeftModel.print_trainable_parameters.

    Args:
        model: The model to print the trainable parameters of.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging_with_rank(
        f"trainable params: {trainable_params:,d} "
        f"|| all params: {all_param:,d} "
        f"|| trainable%: {100 * trainable_params / all_param:.4f}"
    )


def get_submodules_by_type(
    module: nn.Module, target_types: tuple[type, ...] | tuple[str, ...] | type | str
) -> dict[str, nn.Module]:
    """Get all submodules of a module that are of a certain type.

    Args:
        module: The module to get the submodules from.
        target_types: The types of the submodules to get.

    Returns:
        A dictionary of submodules that are of the target type.
    """
    if not isinstance(target_types, tuple | list):
        target_types = (target_types,)
    if isinstance(target_types[0], type):
        return {
            name: submodule
            for name, submodule in module.named_modules()
            if isinstance(submodule, target_types)
        }
    if isinstance(target_types[0], str):
        return {
            name: submodule
            for name, submodule in module.named_modules()
            if submodule.__class__.__name__ in target_types
        }
    msg = f"Invalid target types: {target_types}"
    raise ValueError(msg)


def count_trainable_and_all_parameters(model: nn.Module) -> tuple[int, int]:
    """Count the number of trainable and all parameters in the model.

    Args:
        model: The model to count the parameters of.

    Returns:
        A tuple of the number of trainable parameters and the total number
        of parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def find_module_list(model: nn.Module) -> nn.ModuleList:
    """Find the module list in the model.

    Args:
        model: The model to find the module list in.
    """
    return [module for module in model.modules() if isinstance(module, nn.ModuleList)]
