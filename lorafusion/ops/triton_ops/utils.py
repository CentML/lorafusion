"""Triton utils."""

import torch
import triton.language as tl


def torch_dtype_to_triton_dtype(dtype: torch.dtype) -> tl.dtype:
    """Convert a torch dtype to a triton dtype."""
    mapping = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    if dtype not in mapping:
        msg = (
            f"Unsupported dtype: {dtype}. Supported dtypes: {', '.join(mapping.keys())}"
        )
        raise ValueError(msg)
    return mapping[dtype]
