"""Click utilities."""

import click
import torch

from lorafusion.utils.common import str_to_torch_dtype


def str_to_torch_dtype_callback(
    ctx: click.Context, param: click.Parameter, value: str
) -> torch.dtype:
    """Callback to convert a string to a torch.dtype."""
    return str_to_torch_dtype(value)
