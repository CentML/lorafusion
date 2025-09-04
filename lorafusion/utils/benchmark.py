"""CUDA event-based benchmarking utilities.

- This module provides utilities for benchmarking CUDA operations using PyTorch's
  CUDA events.
- It supports measuring execution time and memory usage of functions running on CUDA
  devices.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import tabulate
import torch
from loguru import logger
from mypy_extensions import DefaultNamedArg, NamedArg
from tqdm import tqdm

from lorafusion.utils.common import empty_cuda_cache
from lorafusion.utils.pytree import tree_zip_map

__all__ = [
    "benchmark",
    "benchmark_func_cuda_event",
    "benchmark_func_cudagraph_cuda_event",
    "benchmark_func_cudagraph_walltime",
    "benchmark_func_walltime",
    "format_memory",
    "format_time",
    "set_warmup_and_number",
]

# Constants for memory conversion
MB = 1024 * 1024  # 1 MB in bytes
WARMUP = -1
NUMBER = -1

BenchmarkFunc = Callable[
    [
        Callable[..., Any],  # The 'func' argument (can refine as needed)
        NamedArg(int, "warmup"),
        NamedArg(int, "number"),
        DefaultNamedArg(Callable[[], dict[str, Any] | None] | None, "prepare_func"),
        DefaultNamedArg(Callable[[], None] | None, "sync_func"),
        DefaultNamedArg(int, "num_memory_records"),
        DefaultNamedArg(bool, "enable_tqdm"),
        DefaultNamedArg(bool, "profile"),
        DefaultNamedArg(str, "output_dir"),
        DefaultNamedArg(str | None, "worker_name"),
    ],
    tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
]


def set_warmup_and_number(warmup: int, number: int) -> None:
    """Set the warmup and number for the benchmark.

    Args:
        warmup: The number of warmup runs.
        number: The number of benchmark runs.
    """
    global WARMUP, NUMBER
    WARMUP = warmup
    NUMBER = number


def benchmark(
    func: Callable[..., Any],
    *,
    prepare_func: Callable[[], dict[str, Any] | None] | None = None,
    sync_func: Callable[[], None] | None = None,
    use_cuda_graph: bool = False,
    use_cuda_event: bool = False,
    warmup: int | None = None,
    number: int | None = None,
    enable_tqdm: bool = False,
    profile: bool = False,
    output_dir: str = "./profiling-results",
    worker_name: str | None = None,
    msg: str | None = None,
    printer: Callable[[str], None] = logger.info,
    cuda_graph_strict_mode: bool = False,
) -> float:
    """Benchmark a function.

    Args:
        func: The function to benchmark.
        prepare_func: The prepare function to use.
        sync_func: The sync function to use.
        use_cuda_graph: Whether to use CUDA graph.
        use_cuda_event: Whether to use CUDA event.
        warmup: The number of warmup runs.
        number: The number of benchmark runs.
        enable_tqdm: Whether to show progress bars for warmup and benchmark runs.
        profile: Whether to profile the function.
        output_dir: The directory to save the profiling results.
        worker_name: The name of the worker.
        msg: The message to print.
        printer: The printer to use.
        cuda_graph_strict_mode: Whether to use strict mode for CUDA graph.
    """
    warmup = warmup or WARMUP
    number = number or NUMBER
    if warmup < 0 or number < 0:
        msg = (
            f"Either call `set_warmup_and_number` or provide `warmup` and `number`"
            f"with non-negative values. Got `warmup={warmup}` and `number={number}`."
        )
        raise ValueError(msg)

    benchmark_fn_dict: dict[
        tuple[bool, bool],
        BenchmarkFunc,
    ] = {
        # Tuple of (use_cuda_graph, use_cuda_event)
        (False, False): benchmark_func_walltime,
        (True, False): benchmark_func_cudagraph_walltime,
        (False, True): benchmark_func_cuda_event,
        (True, True): benchmark_func_cudagraph_cuda_event,
    }
    times, (peak_allocated_memories, peak_reserved_memories) = benchmark_fn_dict[
        (use_cuda_graph, use_cuda_event)
    ](
        func,
        warmup=warmup,
        number=number,
        prepare_func=prepare_func,
        sync_func=sync_func,
        enable_tqdm=enable_tqdm,
        profile=profile,
        output_dir=output_dir,
        worker_name=worker_name,
        cuda_graph_strict_mode=cuda_graph_strict_mode,
    )

    torch.cuda.synchronize()
    empty_cuda_cache()

    median_time = np.median(times)

    if msg is not None:
        msg = f"[{msg}] {format_time(median_time)}"
        printer(msg)

    return np.median(times)


def create_profiler_context(
    *,
    profile: bool,
    skip_first: int = 10,
    wait: int = 0,
    warmup: int = 5,
    active: int = 3,
    repeat: int = 1,
    output_dir: str = "./profiling-results",
    worker_name: str | None = None,
    lightweight_mode: bool = False,
) -> torch.profiler.profile | nullcontext:
    if not profile:
        return nullcontext()

    timestr = time.strftime("%m%d-%H%M%S", time.localtime())
    worker_name = f"{worker_name}-{timestr}" if worker_name else timestr

    if lightweight_mode:
        activities = [torch.profiler.ProfilerActivity.CUDA]
        record_shapes = False
    else:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        record_shapes = True

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            skip_first=skip_first,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            output_dir, worker_name
        ),
        record_shapes=record_shapes,
    )


def _check_prepare_func(
    prepare_func: Callable[[], dict[str, Any] | None] | None,
) -> None:
    """Check the prepare_func.

    Args:
        prepare_func: The prepare function to check.
    """
    # Check prepare_func
    if prepare_func is not None:
        if not callable(prepare_func):
            msg = "prepare_func must be a callable"
            raise ValueError(msg)
        checking_inputs = prepare_func()
        if checking_inputs is not None and not isinstance(checking_inputs, dict):
            msg = (
                "prepare_func must return a dict of kwargs or None"
                f"Got {type(checking_inputs)}"
            )
            raise ValueError(msg)


def benchmark_func_cuda_event(
    func: Callable[..., Any],
    *,
    warmup: int,
    number: int,
    prepare_func: Callable[[], dict[str, Any] | None] | None = None,
    sync_func: Callable[[], None] | None = None,
    num_memory_records: int = 0,
    enable_tqdm: bool = False,
    profile: bool = False,
    output_dir: str = "./profiling-results",
    worker_name: str | None = None,
    cuda_graph_strict_mode: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Benchmark a CUDA function using PyTorch CUDA events.

    This function measures the execution time and memory usage of a CUDA function
    using PyTorch's CUDA events. It supports warmup runs, multiple iterations,
    and memory tracking.

    Args:
        func: The CUDA function to benchmark.
        warmup: Number of warmup runs before actual benchmarking.
        number: Number of benchmark iterations.
        prepare_func: Optional function to prepare inputs for each iteration.
            If provided, should return a dict of kwargs or None.
        sync_func: Optional function to synchronize CUDA operations.
            Defaults to a no-op function.
        num_memory_records: Number of memory usage records to collect.
            If 0, no memory tracking is performed.
        enable_tqdm: Whether to show progress bars for warmup and benchmark runs.
        profile: Whether to profile the function.
        output_dir: The directory to save the profiling results.
        worker_name: The name of the worker.
        cuda_graph_strict_mode: Whether to use strict mode for CUDA graph.

    Returns:
        A tuple containing:
            - Array of execution times in seconds
            - Tuple of (peak_allocated_memories, peak_reserved_memories) in MB

    Example:
        >>> def cuda_function(x):
        ...     return torch.matmul(x, x)
        >>> times, (allocated, reserved) = benchmark_func_cuda_event(
        ...     cuda_function,
        ...     warmup=5,
        ...     number=10,
        ...     prepare_func=lambda: ((torch.randn(1000, 1000).cuda(),), {}),
        ...     enable_tqdm=True
        ... )
    """
    # It is just to unify the interface of the benchmark functions
    del cuda_graph_strict_mode

    _check_prepare_func(prepare_func)

    sync_func = sync_func or (lambda: None)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]

    raw_peak_allocated_memories: list[int] = []
    raw_peak_reserved_memories: list[int] = []
    torch.cuda.reset_peak_memory_stats()

    profiler_context = create_profiler_context(
        profile=profile,
        skip_first=number // 2,
        wait=0,
        warmup=number // 3,
        active=3,
        repeat=1,
        output_dir=output_dir,
        worker_name=worker_name,
    )

    def _run_func(step: int = -1) -> None:
        """Run a single iteration of the benchmark.

        Args:
            step: The current step number. -1 for warmup runs.
        """
        start_event = start_events[step]
        end_event = end_events[step]

        # Generate inputs if prepare_func is provided
        kwargs = prepare_func() if prepare_func is not None else None

        # Run the function and do synchronization
        start_event.record()
        if kwargs is not None:
            func(**kwargs)
        else:
            func()
        end_event.record()

        if len(raw_peak_allocated_memories) < num_memory_records:
            raw_peak_allocated_memories.append(torch.cuda.max_memory_allocated())
            raw_peak_reserved_memories.append(torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in tqdm(range(warmup), disable=not enable_tqdm):
        _run_func()

    # Benchmark
    with profiler_context as prof:
        for step in tqdm(range(number), disable=not enable_tqdm):
            _run_func(step)
            if prof is not None:
                prof.step()

    torch.cuda.synchronize()
    costs = [
        start_events[i].elapsed_time(end_events[i]) / 1000.0 for i in range(number)
    ]

    peak_allocated_memories = np.array(raw_peak_allocated_memories) / MB
    peak_allocated_memories = np.sort(np.unique(peak_allocated_memories))
    peak_reserved_memories = np.array(raw_peak_reserved_memories) / MB
    peak_reserved_memories = np.sort(np.unique(peak_reserved_memories))

    return np.array(costs), (peak_allocated_memories, peak_reserved_memories)


def benchmark_func_walltime(
    func: Callable[..., Any],
    *,
    warmup: int,
    number: int,
    prepare_func: Callable[[], dict[str, Any] | None] | None = None,
    sync_func: Callable[[], None] | None = None,
    num_memory_records: int = 0,
    enable_tqdm: bool = False,
    profile: bool = False,
    output_dir: str = "./profiling-results",
    worker_name: str | None = None,
    cuda_graph_strict_mode: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Benchmark a CUDA function using walltime.

    This function measures the execution time of a CUDA function using walltime.
    It supports warmup runs, multiple iterations, and memory tracking.

    Args:
        func: The CUDA function to benchmark.
        warmup: Number of warmup runs before actual benchmarking.
        number: Number of benchmark iterations.
        prepare_func: Optional function to prepare inputs for each iteration.
            If provided, should return a dict of kwargs or None.
        sync_func: Optional function to synchronize CUDA operations.
            Defaults to a no-op function.
        num_memory_records: Number of memory usage records to collect.
            If 0, no memory tracking is performed.
        enable_tqdm: Whether to show progress bars for warmup and benchmark runs.
        profile: Whether to profile the function.
        output_dir: The directory to save the profiling results.
        worker_name: The name of the worker.
        cuda_graph_strict_mode: Whether to use strict mode for CUDA graph.

    Returns:
        A tuple containing:
            - Array of execution times in seconds
            - Tuple of (peak_allocated_memories, peak_reserved_memories) in MB

    Example:
        >>> def cuda_function(x):
        ...     return torch.matmul(x, x)
        >>> times, (allocated, reserved) = benchmark_func_walltime(
        ...     cuda_function,
        ...     warmup=5,
        ...     number=10,
        ... )
    """
    # It is just to unify the interface of the benchmark functions
    del cuda_graph_strict_mode

    _check_prepare_func(prepare_func)

    # Use torch.cuda.synchronize as the default sync function
    # as we are using time.perf_counter to measure the time
    sync_func = sync_func or torch.cuda.synchronize

    raw_peak_allocated_memories: list[int] = []
    raw_peak_reserved_memories: list[int] = []
    torch.cuda.reset_peak_memory_stats()

    profiler_context = create_profiler_context(
        profile=profile,
        skip_first=number // 2,
        wait=0,
        warmup=number // 3,
        active=3,
        repeat=1,
        output_dir=output_dir,
        worker_name=worker_name,
    )

    def _run_func() -> float:
        """Run a single iteration of the benchmark.

        Args:
            step: The current step number. -1 for warmup runs.
        """
        # Generate inputs if prepare_func is provided
        kwargs = prepare_func() if prepare_func is not None else None

        if sync_func is not None:
            sync_func()
        tic = perf_counter()

        # Run the function and do synchronization
        if kwargs is not None:
            func(**kwargs)
        else:
            func()

        if sync_func is not None:
            sync_func()
        toc = perf_counter()

        if len(raw_peak_allocated_memories) < num_memory_records:
            raw_peak_allocated_memories.append(torch.cuda.max_memory_allocated())
            raw_peak_reserved_memories.append(torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()

        return toc - tic

    # Warmup
    for _ in tqdm(range(warmup), disable=not enable_tqdm):
        _run_func()

    # Benchmark
    costs = []
    with profiler_context as prof:
        for _ in tqdm(range(number), disable=not enable_tqdm):
            costs.append(_run_func())
            if prof is not None:
                prof.step()

    peak_allocated_memories = np.array(raw_peak_allocated_memories) / MB
    peak_allocated_memories = np.sort(np.unique(peak_allocated_memories))
    peak_reserved_memories = np.array(raw_peak_reserved_memories) / MB
    peak_reserved_memories = np.sort(np.unique(peak_reserved_memories))

    return np.array(costs), (peak_allocated_memories, peak_reserved_memories)


def prepare_cuda_graph(
    func: Callable[..., Any],
    *,
    prepare_func: Callable[[], dict[str, Any] | None] | None = None,
    cuda_graph_warmup: int = 5,
    cuda_graph_strict_mode: bool = False,
) -> tuple[
    torch.cuda.CUDAGraph,
    dict[str, Any] | None,
    Callable[..., None],
    Callable[[], dict[str, Any] | None],
]:
    """Prepare a CUDA graph for benchmarking.

    This function performs warmup runs and creates a CUDA graph for the given function.
    The CUDA graph can then be used for efficient benchmarking of repeated operations.

    Args:
        func: The CUDA function to create a graph for.
        cuda_graph_warmup: Number of warmup runs before creating the graph.
        prepare_func: Optional function to prepare inputs for each iteration.
            If provided, should return a dict of kwargs or None.
        cuda_graph_strict_mode: Whether to use strict mode for CUDA graph.

    Returns:
        A tuple containing:
            - The prepared CUDA graph
            - The static inputs used for the graph (if prepare_func was provided)
            - The CUDA graph function that replays the graph
            - The CUDA graph prepare function that copies new inputs to static tensors

    Example:
        >>> def cuda_function(x):
        ...     return torch.matmul(x, x)
        >>> def prepare_inputs():
        ...     return {"x": torch.randn(1000, 1000, device="cuda")}
        >>> graph, static_inputs, graph_func, graph_prepare_func = prepare_cuda_graph(
        ...     cuda_function,
        ...     prepare_func=prepare_inputs,
        ... )
    """
    _check_prepare_func(prepare_func)

    # Perform warmup runs
    for _ in range(cuda_graph_warmup):
        kwargs = prepare_func() if prepare_func is not None else None
        if kwargs is not None:
            func(**kwargs)
        else:
            func()

    # Get static inputs for the graph
    static_inputs = prepare_func() if prepare_func is not None else None

    # Provide the CUDA graph prepare function
    def _copy_leaf_for_cuda_graph_inputs(
        from_tensor: Any,  # noqa: ANN401
        to_tensor: Any,  # noqa: ANN401
    ) -> None:
        """Copy the leaf tensor from the static inputs to the new inputs.

        Args:
            from_tensor: The tensor to copy from.
            to_tensor: The tensor to copy to.
        """
        if isinstance(from_tensor, torch.Tensor) and isinstance(
            to_tensor, torch.Tensor
        ):
            to_tensor.data.copy_(from_tensor)
        elif cuda_graph_strict_mode and from_tensor != to_tensor:
            msg = (
                "Inputs of CUDA graph with the non-tensor type should remain unchanged."
                f" Got type: {type(from_tensor)} and {type(to_tensor)}. "
                "Please make sure the inputs are not changed."
            )
            logger.warning(msg)
            raise ValueError(msg)

    def cuda_graph_prepare_func() -> dict[str, Any] | None:
        if prepare_func is None or static_inputs is None:
            return None
        # Becuase static_inputs is not None, new_inputs is not None
        new_inputs = prepare_func()
        tree_zip_map(_copy_leaf_for_cuda_graph_inputs, static_inputs, new_inputs)
        return new_inputs

    # Create and capture the CUDA graph
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        if static_inputs is not None:
            func(**static_inputs)
        else:
            func()

    def cuda_graph_func(**kwargs) -> None:
        cuda_graph.replay()

    # Run once to validate the correctness of the graph
    cuda_graph_prepare_func()
    cuda_graph_func()

    return cuda_graph, static_inputs, cuda_graph_func, cuda_graph_prepare_func


def benchmark_func_cudagraph_cuda_event(
    func: Callable[..., Any],
    *,
    warmup: int,
    number: int,
    prepare_func: Callable[[], dict[str, Any] | None] | None = None,
    sync_func: Callable[[], None] | None = None,
    num_memory_records: int = 0,
    enable_tqdm: bool = False,
    profile: bool = False,
    output_dir: str = "./profiling-results",
    worker_name: str | None = None,
    cuda_graph_strict_mode: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Benchmark a function using CUDA graph and CUDA events.

    This function first creates a CUDA graph from the given function and inputs,
    then measures its execution time using PyTorch's CUDA events.

    Args:
        func: The function to benchmark.
        warmup: Number of warmup runs before benchmarking.
        number: Number of benchmark iterations.
        prepare_func: Optional function to prepare inputs for each iteration.
            If provided, should return a dict of kwargs or None.
        sync_func: Optional function to synchronize CUDA operations.
            Defaults to a no-op function.
        num_memory_records: Number of memory usage records to collect.
            If 0, no memory tracking is performed.
        enable_tqdm: Whether to show progress bars for benchmark runs.
        profile: Whether to profile the function.
        output_dir: The directory to save the profiling results.
        worker_name: The name of the worker.
        cuda_graph_strict_mode: Whether to use strict mode for CUDA graph.

    Returns:
        A tuple containing:
            - Array of execution times in seconds
            - Tuple of (peak_allocated_memories, peak_reserved_memories) in MB

    Example:
        >>> def cuda_function(x):
        ...     return torch.matmul(x, x)
        >>> def prepare_inputs():
        ...     return {"x": torch.randn(1000, 1000, device="cuda")}
        >>> times, (allocated, reserved) = benchmark_func_cudagraph_cuda_event(
        ...     cuda_function,
        ...     warmup=5,
        ...     number=10,
        ...     prepare_func=prepare_inputs,
        ...     enable_tqdm=True
        ... )
    """
    # First prepare the CUDA graph
    cuda_graph, static_inputs, graph_func, graph_prepare_func = prepare_cuda_graph(
        func,
        prepare_func=prepare_func,
        cuda_graph_strict_mode=cuda_graph_strict_mode,
    )

    # Use the original benchmark function with the graph functions
    return benchmark_func_cuda_event(
        graph_func,
        warmup=warmup,
        number=number,
        prepare_func=graph_prepare_func,
        sync_func=sync_func,
        num_memory_records=num_memory_records,
        enable_tqdm=enable_tqdm,
        profile=profile,
        output_dir=output_dir,
        worker_name=worker_name,
    )


def benchmark_func_cudagraph_walltime(
    func: Callable[..., Any],
    *,
    warmup: int,
    number: int,
    prepare_func: Callable[[], dict[str, Any] | None] | None = None,
    sync_func: Callable[[], None] | None = None,
    num_memory_records: int = 0,
    enable_tqdm: bool = False,
    profile: bool = False,
    output_dir: str = "./profiling-results",
    worker_name: str | None = None,
    cuda_graph_strict_mode: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Benchmark a function using CUDA graph and walltime.

    This function first creates a CUDA graph from the given function and inputs,
    then measures its execution time using walltime measurements.

    Args:
        func: The function to benchmark.
        warmup: Number of warmup runs before benchmarking.
        number: Number of benchmark iterations.
        prepare_func: Optional function to prepare inputs for each iteration.
            If provided, should return a dict of kwargs or None.
        sync_func: Optional function to synchronize CUDA operations.
            Defaults to a no-op function.
        num_memory_records: Number of memory usage records to collect.
            If 0, no memory tracking is performed.
        enable_tqdm: Whether to show progress bars for benchmark runs.
        profile: Whether to profile the function.
        output_dir: The directory to save the profiling results.
        worker_name: The name of the worker.
        cuda_graph_strict_mode: Whether to use strict mode for CUDA graph.

    Returns:
        A tuple containing:
            - Array of execution times in seconds
            - Tuple of (peak_allocated_memories, peak_reserved_memories) in MB

    Example:
        >>> def cuda_function(x):
        ...     return torch.matmul(x, x)
        >>> def prepare_inputs():
        ...     return {"x": torch.randn(1000, 1000, device="cuda")}
        >>> times, (allocated, reserved) = benchmark_func_cudagraph_walltime(
        ...     cuda_function,
        ...     warmup=5,
        ...     number=10,
        ...     prepare_func=prepare_inputs,
        ...     enable_tqdm=True
        ... )
    """
    # First prepare the CUDA graph
    cuda_graph, static_inputs, graph_func, graph_prepare_func = prepare_cuda_graph(
        func,
        prepare_func=prepare_func,
        cuda_graph_strict_mode=cuda_graph_strict_mode,
    )

    # Use the original benchmark function with the graph functions
    return benchmark_func_walltime(
        graph_func,
        warmup=warmup,
        number=number,
        prepare_func=graph_prepare_func,
        sync_func=sync_func,
        num_memory_records=num_memory_records,
        enable_tqdm=enable_tqdm,
        profile=profile,
        output_dir=output_dir,
        worker_name=worker_name,
    )


def _format(
    t: float, interval_ratio: float, prefixes: list[str], base_to_min_ratio: float
) -> str:
    """Format time in seconds to a string with appropriate units.

    Args:
        t: Time in seconds.
        interval_ratio: The ratio of the interval to the next unit.
        prefixes: The prefixes to use for the units.
        base_to_min_ratio: The ratio of the base unit to the next unit.

    Returns:
        Formatted time string.
    """
    t = t * base_to_min_ratio
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if t < interval_ratio:
            break
        prefix = new_prefix
        t /= interval_ratio
    return f"{t:.4f} {prefix}"


def format_time(t: float) -> str:
    """Format time in seconds to a string with appropriate units.

    Args:
        t: Time in seconds.

    Returns:
        Formatted time string.
    """
    return _format(t, 1e3, ["us", "ms", "s"], base_to_min_ratio=1e6)


def format_memory(m: float) -> str:
    """Format memory in bytes to a string with appropriate units.

    Args:
        m: Memory in bytes.

    Returns:
        Formatted memory string.
    """
    return _format(m, 1024, ["B", "KB", "MB", "GB"], base_to_min_ratio=1)


def tabulate_2d_benchmark_results(
    times: list[list[float]],
    impl_names: list[str],
    first_col_choices: list[Any],
    first_col_name: str | None = None,
    *,
    show_speedup: bool = True,
    tablefmt: str = "grid",
    path_to_save: str | None = None,
) -> str:
    """Tabulate 2D benchmark results.

    Args:
        times: List of lists of times. Should be a 2D array, of shape (n, m),
            where n is the number of choices (e.g. different sizes of the input)
            and m is the number of implementations.
        impl_names: List of implementation names.
        first_col_choices: List of choices for the first column.
        first_col_name: Name of the first column.
        show_speedup: Whether to show speedup.
        tablefmt: Table format.
        path_to_save: Path to save the tabulated results.

    Returns:
        Tabulated benchmark results.

    Example:
        >>> times = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        >>> impl_names = ["hidet", "triton", "torch"]
        >>> first_col_choices = [1, 2]
        >>> first_col_name = "Batch Size"
        >>> print(
        >>> tabulate_2d_benchmark_results(
        >>>     times,
        >>>     impl_names=impl_names,
        >>>     first_col_choices=first_col_choices,
        >>>     first_col_name=first_col_name,
        >>>     show_speedup=True,
        >>>     tablefmt="grid",
        >>> )
    """
    num_choices = len(times)
    num_impls = len(times[0])
    if len(impl_names) != num_impls:
        msg = (
            f"The number of implementation names ({len(impl_names)}) must match the"
            f"number of implementations ({num_impls})."
        )
        raise ValueError(msg)
    if len(first_col_choices) != num_choices:
        msg = (
            f"The number of choices ({len(first_col_choices)}) must match the"
            f"number of choices ({num_choices})."
        )
        raise ValueError(msg)

    first_col_name = first_col_name or ""
    headers = [first_col_name, *impl_names]
    table = []
    table_to_save = []
    for col_val, curr_times in zip(first_col_choices, times, strict=True):
        base_time = curr_times[0]
        curr_row = [f"{col_val}", f"{format_time(base_time)}"]
        curr_row_to_save = [f"{col_val}", f"{base_time:.6f}"]
        for impl_time in curr_times[1:]:
            if show_speedup:
                speedup = base_time / impl_time
                curr_row.append(f"{format_time(impl_time)}, {speedup:.2f}x")
                curr_row_to_save.append(f"{impl_time:.6f}")
            else:
                curr_row.append(f"{format_time(impl_time)}")
                curr_row_to_save.append(f"{impl_time:.6f}")
        table.append(curr_row)
        table_to_save.append(curr_row_to_save)

    if path_to_save is not None:
        path_to_save = Path(path_to_save)
        if path_to_save.suffix != ".csv":
            msg = f"The path to save must end with .csv. Got {path_to_save}."
            raise ValueError(msg)
        if not path_to_save.exists():
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            with path_to_save.open("w") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(table_to_save)
        else:
            with path_to_save.open("a") as f:
                writer = csv.writer(f)
                writer.writerows(table_to_save)

    return tabulate.tabulate(table, headers=headers, tablefmt=tablefmt)
