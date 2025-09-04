"""Gradient checkpointing utilities."""

from collections.abc import Callable
from functools import partial, wraps
from types import MethodType
from typing import Any

import torch
import torch.utils.checkpoint
from torch import nn


def positional_args_call_fn(fn: Callable[[Any], Any], *args) -> Any:  # noqa: ANN401
    """Call a function with args."""
    return fn(*args)


def wrap_forward_with_gradient_checkpointing(
    module: nn.Module,
    gradient_checkpointing_func: Callable[[Any], Any] | None = None,
) -> None:
    """Wrap the forward method of a module with gradient checkpointing."""
    orig_forward_method = module.forward
    orig_forward_func = type(module).forward

    if gradient_checkpointing_func is None:
        gradient_checkpointing_func = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )

    @wraps(orig_forward_func)
    def wrapped_checkpoint_forward(
        self: nn.Module,
        *args,
        **kwargs,
    ) -> Any:  # noqa: ANN401
        if hasattr(self, "training") and self.training:
            return gradient_checkpointing_func(
                partial(orig_forward_method, **kwargs), *args
            )
        return orig_forward_method(*args, **kwargs)

    def reset_wrapped_forward_method() -> None:
        module.forward = orig_forward_method

    module.forward = MethodType(wrapped_checkpoint_forward, module)
    module._orig_forward_without_ckpt = orig_forward_method  # noqa: SLF001
    module._reset_wrapped_forward_method = reset_wrapped_forward_method  # noqa: SLF001


def apply_gradient_checkpointing(
    module: nn.Module,
    sub_modules: list[nn.Module],
    gradient_checkpointing_func: Callable[[Any], Any] | None = None,
) -> None:
    """Apply gradient checkpointing to a module."""
    replace_list = []
    for parent in module.modules():
        for name, child in parent._modules.items():  # noqa: SLF001
            if child in sub_modules or (
                hasattr(child, "name") and child.name in sub_modules
            ):
                replace_list.append((parent, name, child))

    for _, _, child in replace_list:
        wrap_forward_with_gradient_checkpointing(child, gradient_checkpointing_func)
