"""Common utilities."""

import argparse
import gc
import os
import random
import sys
from collections.abc import Callable
from numbers import Number
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger


def set_seed(seed: int) -> None:
    """Set the seed for the random number generator."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


def str_to_torch_dtype(dtype_str: str | torch.dtype) -> torch.dtype:
    """Convert a string to a torch.dtype.

    Args:
        dtype_str: the string to convert

    Returns:
        The converted torch.dtype.
    """
    if isinstance(dtype_str, torch.dtype):
        return dtype_str

    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "half": torch.half,
        "float": torch.float,
        "double": torch.double,
        "int": torch.int,
        "long": torch.long,
        "short": torch.short,
    }

    if dtype_str not in mapping:
        msg = (
            f"Unsupported dtype: {dtype_str}. "
            f"Supported dtypes: {', '.join(mapping.keys())}"
        )
        raise ValueError(msg)

    return mapping[dtype_str]


def torch_dtype_to_str(dtype: torch.dtype | str) -> str:
    """Convert a torch.dtype to a string.

    Args:
        dtype: the torch.dtype to convert

    Returns:
        The converted string.
    """
    mapping = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.half: "float16",
        torch.float: "float32",
        torch.double: "float64",
        torch.int: "int32",
        torch.long: "int64",
        torch.short: "int16",
    }

    if isinstance(dtype, str):
        if dtype not in list(mapping.values()):
            msg = f"dtype {dtype} is a string in torch_dtype_to_str"
            raise ValueError(msg)
        return dtype

    if dtype not in mapping:
        msg = (
            f"Unsupported dtype: {dtype}. Supported dtypes: {', '.join(mapping.keys())}"
        )
        raise ValueError(msg)

    return mapping[dtype]


def get_dtype_element_size(dtype: torch.dtype | str) -> int:
    """Get the element size of a dtype.

    Args:
        dtype: the dtype to get the element size of

    Returns:
        The element size of the dtype.
    """
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype(dtype)
    if not isinstance(dtype, torch.dtype):
        msg = f"Invalid dtype: {dtype}. Must be a torch.dtype or a string."
        raise TypeError(msg)
    return torch.finfo(dtype).bits // 8


def is_array_consistent(
    array: list[Number] | np.ndarray,
    threshold: float = 0.15,
) -> bool:
    r"""Validate the consistency of an array.

    Validate that the standard deviation is less than a threshold times the mean:

    .. math::

        \\text{std} < \\text{threshold} \\times \\text{mean}

    Args:
        array: the array to validate
        threshold: the threshold for the validation

    Returns:
        True if the array is consistent, False otherwise.
    """
    if isinstance(array, list):
        array = np.array(array)
    if not isinstance(array, np.ndarray):
        msg = f"Invalid array: {array}. Must be a list or a numpy array."
        raise TypeError(msg)
    return np.std(array) < threshold * np.mean(array)


def empty_cuda_cache() -> None:
    """Empty the CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def log_memory_usage(
    message: str, print_fn: Callable[[str], None] = logger.info
) -> None:
    """Log the memory usage of the current process.

    Args:
        message: The message to log.
        print_fn: The function to print the message.
    """
    print_fn(
        f"[{message}] "
        f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB. "
        f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )


def update_loguru_level(level: str) -> None:
    """Update the loguru level."""
    if level.upper() not in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]:
        msg = (
            f"Invalid loguru level: {level}. "
            f"Must be one of: INFO, DEBUG, WARNING, ERROR, CRITICAL."
        )
        raise ValueError(msg)
    logger.remove()
    logger.add(sys.stderr, level=level)


def logging_rank0(msg: str, level: str = "info") -> None:
    """Log message only on rank 0.

    Args:
        msg: Message to log
        level: Log level (info, warning, error)

    Raises:
        ValueError: If log level is invalid
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        logging_with_rank(msg, level, depth=2)


def logging_with_rank(msg: str, level: str = "info", depth: int = 1) -> None:
    """Log message with rank information.

    Args:
        msg: Message to log
        level: Log level (info, warning, error)
        depth: Call stack depth to use for identifying the caller

    Raises:
        ValueError: If log level is invalid
    """
    rank_msg = "" if not dist.is_initialized() else f"[Rank {dist.get_rank()}] "

    if level == "info":
        logger.opt(depth=depth).info(f"{rank_msg}{msg}")
    elif level == "warning":
        logger.opt(depth=depth).warning(f"{rank_msg}{msg}")
    elif level == "error":
        logger.opt(depth=depth).error(f"{rank_msg}{msg}")
    elif level == "debug":
        logger.opt(depth=depth).debug(f"{rank_msg}{msg}")
    elif level == "success":
        logger.opt(depth=depth).success(f"{rank_msg}{msg}")
    else:
        msg = f"Invalid log level: {level}"
        raise ValueError(msg)


def logging_if(msg: str, *, condition: bool, level: str = "info") -> None:
    """Log message if condition is met.

    Args:
        msg: Message to log
        condition: Condition to check
        level: Log level (info, warning, error)
    """
    if condition:
        logging_with_rank(msg, level, depth=2)


def maybe_setup_distributed() -> None:
    """Initialize distributed training environment if not already initialized."""
    if dist.is_initialized():
        return

    # Try to initialize distributed training environment
    invalid_value = -1
    rank = int(os.environ.get("RANK", invalid_value))
    world_size = int(os.environ.get("WORLD_SIZE", invalid_value))
    if invalid_value not in (rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(torch.device("cuda", dist.get_rank()))
        logging_rank0("Initialized distributed training environment.")


def get_device_short_name() -> str:
    """Get the short name of the current device."""
    name = torch.cuda.get_device_name()
    # Remove "NVIDIA" from the name
    name = name.replace("NVIDIA ", "").strip()
    return name.replace(" ", "-").lower()


def json_utils_default(obj: Any) -> str:  # noqa: ANN401
    """Serialize an object to a JSON string.

    Args:
        obj: The object to serialize

    Returns:
        The serialized JSON string.
    """
    if isinstance(obj, set):
        return list(obj)
    return obj


def stringify_keys(obj: Any) -> Any:  # noqa: ANN401
    """Serialize the keys of a dictionary to strings.

    Args:
        obj: The object to serialize
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            new_k = ", ".join(map(str, k)) if isinstance(k, tuple | list) else k
            new[new_k] = stringify_keys(v)
        return new
    if isinstance(obj, list | tuple):
        return [stringify_keys(i) for i in obj]
    return obj


def list_of_floats(arg: str) -> list[float]:
    """Parse a string of floats separated by spaces.

    It can be used as an argument parser type.

    Args:
        arg: The string to parse

    Returns:
        The list of floats.
    """
    try:
        return [float(x) for x in arg.replace(",", " ").split()]
    except ValueError as e:
        msg = (
            f"Invalid list of floats: {arg}. "
            "Must be a string of floats separated by spaces."
        )
        raise argparse.ArgumentTypeError(msg) from e


def list_of_ints(arg: str) -> list[int]:
    """Parse a string of ints separated by spaces.

    It can be used as an argument parser type.

    """
    try:
        return [int(x) for x in arg.replace(",", " ").split()]
    except ValueError as e:
        msg = (
            f"Invalid list of ints: {arg}. "
            "Must be a string of ints separated by spaces."
        )
        raise argparse.ArgumentTypeError(msg) from e
