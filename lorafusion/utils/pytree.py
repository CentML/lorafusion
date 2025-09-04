"""Pytree utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from torch.utils._pytree import (
    PyTree,
    tree_flatten,
    tree_unflatten,
)

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
R = TypeVar("R")


def tree_zip_map(fn: Callable[[T, T], R], pytree_1: PyTree, pytree_2: PyTree) -> PyTree:
    """Maps a function which takes two arguments over two pytrees.

    Args:
        fn: Function which takes two arguments and returns a value
        pytree_1: First pytree to map over
        pytree_2: Second pytree to map over

    Returns:
        A new pytree with the mapped values

    Raises:
        AssertionError: If the tree structures don't match
    """
    flat_args_1, treedef_1 = tree_flatten(pytree_1)
    flat_args_2, treedef_2 = tree_flatten(pytree_2)
    if treedef_1 != treedef_2:
        msg = f"Expected TreeDefs to be equal, got {treedef_1} and {treedef_2}"
        raise AssertionError(msg)
    flat_result = [
        fn(arg_1, arg_2) for arg_1, arg_2 in zip(flat_args_1, flat_args_2, strict=True)
    ]
    return tree_unflatten(flat_result, treedef_1)
