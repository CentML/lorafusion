"""Simulate the imbalance of multi LoRA."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import patches
from proposed_solver import solve_schedule_milp

from lorafusion.utils.common import update_loguru_level
from lorafusion.utils.pytree import tree_flatten

update_loguru_level("INFO")


def generate_data(
    num_adapters: int,
    batches_per_adapter: list[int],
    samples_per_batch: list[int],
    seed: int,
    dist: str = "lognormal",
    dist_params: dict[str, float] | None = None,
) -> list[list[list[int]]]:
    """Generate token-lengths for each sample in each batch of each adapter.

    Args:
      num_adapters: int, number of adapters.
      batches_per_adapter: list of int, number of batches for each adapter.
      samples_per_batch: list of int, number of samples each batch for each adapter.
      seed: int, RNG seed.
      dist: str or callable, distribution name ('lognormal') or custom.
      dist_params: dict, parameters for the distribution.
        For 'lognormal', expects keys 'mean' and 'sigma'.
        If dist is callable, pass rng and size to dist_params['func'].

    Returns:
      data: nested list -
        data[i][j][k] = token-count for sample k of batch j of adapter i.
    """
    rng = np.random.default_rng(seed)
    data = []
    if dist_params is None:
        dist_params = {}
    for i in range(num_adapters):
        adapter_batches = []
        for _ in range(batches_per_adapter[i]):
            size = samples_per_batch[i]
            if dist == "lognormal":
                if "mean" not in dist_params or "sigma" not in dist_params:
                    msg = "mean and sigma must be provided for lognormal distribution"
                    raise ValueError(msg)
                mean = dist_params["mean"]
                sigma = dist_params["sigma"]
                lengths = rng.lognormal(mean=mean, sigma=sigma, size=size)
            elif dist == "constant":
                lengths = np.ones(size) * dist_params["length"]
            else:
                lengths = dist_params["func"](size, rng)
            lengths = np.round(lengths).astype(int)
            adapter_batches.append(list(lengths))
        data.append(adapter_batches)
    return data


def _safe_get(arr: np.ndarray, index: tuple[int, int]) -> int:
    try:
        return arr[index]
    except IndexError:
        return 0


def evaluate_schedule_pipeline_parallelism(  # noqa: C901, PLR0912, PLR0915
    data: list[list[list[int]]],
    schedule: list[tuple[list[tuple[int, int, int]], list[int]]],
    num_stages: int,
    *,
    padding: bool,
    fwd_times: list[int],
    bwd_times: list[int],
) -> tuple[
    list[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate the pipeline parallelism schedule.

    Simulates the execution of microbatches through a pipeline with forward and backward
    passes, considering memory and dependency constraints.

    Args:
      data: nested list as returned by generate_data.
      schedule: list of micro-batches; each micro-batch is a list of (i,j,k) tuples.
      num_stages: int, number of pipeline stages.
      padding: bool, whether to pad sample to the longest sample in the microbatch.
      fwd_times: list of int, time multiplier for forward pass at each stage.
      bwd_times: list of int, time multiplier for backward pass at each stage.

    Returns:
      loads: list of int, loads for each microbatch.
      fwd_start: (num_stages, num_microbatches) array, start time of forward
      fwd_end: (num_stages, num_microbatches) array, end time of forward
      bwd_start: (num_stages, num_microbatches) array, start time of backward
      bwd_end: (num_stages, num_microbatches) array, end time of backward
      total_time: float, total execution time.
      bubble_ratio: (num_stages) array, bubble ratio for each stage.
      adjusted_bubble_ratio: (num_stages) array, adjusted bubble ratio for each stage.
    """
    # Construct the loads: which is a list of microbatches, where a number represents
    # the load (number of tokens) of a microbatch.
    loads = []
    two = 2
    if isinstance(schedule[0], tuple) and len(schedule[0]) == two:
        schedule = [s[0] for s in schedule]
    for i, microbatch in enumerate(schedule):
        if padding:
            max_length = max(data[i][j][k] for (i, j, k) in microbatch)
            curr_data = [max_length] * len(microbatch)
        else:
            curr_data = [data[i][j][k] for (i, j, k) in microbatch]
        loads.append(sum(curr_data))
        logger.debug(f"Micro Batch {i:2d}: {curr_data}.  \t Load: {sum(curr_data):3d}.")

    num_microbatches = len(loads)
    fwd_start = np.zeros((num_stages, num_microbatches))
    fwd_end = np.zeros((num_stages, num_microbatches))
    bwd_start = np.zeros((num_stages, num_microbatches))
    bwd_end = np.zeros((num_stages, num_microbatches))

    # For microbatch t,
    # (1) Memory Constraint: its forward pass on stage s depends on the completion of
    # t-(S-s) microbatch's backward pass on the same stage.
    # (2) Dependency Constraint (fwd): its forward pass on stage s depends on the
    # completion of t microbatch's forward pass on the previous stage s-1.
    # (3) Dependency Constraint (bwd): its backward pass on stage s depends on the
    # completion of t microbatch's backward pass on the previous stage s+1.
    # (4) For the last stage, the backward pass depends on the completion of t
    # microbatch's forward pass.

    # Simulate pipeline execution using an event-driven approach inspired by
    # Megatron-LM's forward_backward_pipelining_without_interleaving

    # Define all possible events in the simulation
    events = []

    # > Event: (stage, microbatch, phase, start_time, end_time)
    # phase: 0 for forward, 1 for backward

    def add_event_and_update(
        s: int, t: int, phase: int, start_time: float, end_time: float
    ) -> None:
        """Add an event and update the forward and backward start and end times."""
        events.append((s, t, phase, start_time, end_time))
        if phase == 0:
            if fwd_start[s, t] != 0 or fwd_end[s, t] != 0:
                msg = (
                    f"Forward pass of stage {s} microbatch {t} already scheduled. "
                    f"fwd_start: {fwd_start[s, t]:.2f}, fwd_end: {fwd_end[s, t]:.2f}"
                )
                raise ValueError(msg)
            fwd_start[s, t] = start_time
            fwd_end[s, t] = end_time
        else:
            if bwd_start[s, t] != 0 or bwd_end[s, t] != 0:
                msg = (
                    f"Backward pass of stage {s} microbatch {t} already scheduled. "
                    f"bwd_start: {bwd_start[s, t]:.2f}, bwd_end: {bwd_end[s, t]:.2f}"
                )
                raise ValueError(msg)
            bwd_start[s, t] = start_time
            bwd_end[s, t] = end_time

    # Add the first forward pass at stage 0 for microbatch 0
    add_event_and_update(0, 0, 0, 0.0, fwd_times[0] * loads[0])

    # Process events in chronological order until all microbatches are processed
    while events:
        # Get the earliest event
        s, t, phase, start_time, end_time = events.pop(0)

        # Record the timing
        if phase == 0:  # Forward pass
            logger.debug(
                f"[Stage {s} - Microbatch {t} - FWD]\t"
                f"Start: {start_time:.2f}\tEnd: {end_time:.2f}"
            )

            # 1. Forward pass of next microbatch at this stage
            # (Only during the warmup phase)
            # and should make sure the forward pass of the previous stage has completed
            if (
                t < num_stages - s - 1
                and (s == 0 or fwd_end[s - 1, t] > 0)
                and fwd_end[s, t + 1] == 0
            ):
                next_start_time = max(end_time, _safe_get(fwd_end, (s - 1, t)))
                next_end_time = next_start_time + fwd_times[s] * loads[t + 1]
                add_event_and_update(s, t + 1, 0, next_start_time, next_end_time)
                logger.debug(
                    f"  >>> Scheduling forward pass of stage {s} for microbatch "
                    f"{t + 1} based on event stage {s} microbatch {t}'s forward pass "
                    "completion"
                )

            # 2. Forward pass of next stage
            # It can be scheduled only if t-(S-(s+1))'s backward pass has completed
            if (
                s < num_stages - 1
                and (
                    # For example, S=2, t=1, s=0, then t-(S-s-1)=0, which means the
                    # backward pass should be checked
                    t - (num_stages - s - 1) < 0
                    or bwd_end[s, t - (num_stages - s - 1)] > 0
                )
                and fwd_end[s + 1, t] == 0
            ):
                next_start_time = max(
                    end_time,
                    _safe_get(bwd_end, (s, t - (num_stages - s - 1))),
                    _safe_get(fwd_end, (s + 1, t - 1)),
                )
                next_end_time = next_start_time + fwd_times[s + 1] * loads[t]
                add_event_and_update(s + 1, t, 0, next_start_time, next_end_time)
                logger.debug(
                    f"  >>> Scheduling forward pass of stage {s + 1} for microbatch "
                    f"{t} based on event stage {s} microbatch {t}'s forward pass "
                    "completion"
                )

            # 3. Backward pass of this stage
            # Launch t-(S-s-1)'s backward pass if the previous backward pass has
            # completed
            if s == num_stages - 1 or (
                t - (num_stages - s - 1) >= 0
                and bwd_end[s, t - (num_stages - s - 1)] > 0
            ):
                next_start_time = max(
                    end_time, _safe_get(bwd_end, (s, t - (num_stages - s - 1)))
                )
                next_end_time = (
                    next_start_time + bwd_times[s] * loads[t - (num_stages - s - 1)]
                )
                add_event_and_update(
                    s, t - (num_stages - s - 1), 1, next_start_time, next_end_time
                )
                logger.debug(
                    f"  >>> Scheduling backward pass of stage {s} for microbatch "
                    f"{t - (num_stages - s - 1)} based on event stage {s} microbatch "
                    f"{t}'s forward pass completion"
                )

        else:  # Backward pass
            bwd_start[s, t] = start_time
            bwd_end[s, t] = end_time
            logger.debug(
                f"[Stage {s} - Microbatch {t} - BWD]\t"
                f"Start: {start_time:.2f}\tEnd: {end_time:.2f}"
            )

            # 1. Backward pass of s-1 stage of microbatch t
            # it can only be scheduled if t+(S-s-1+1)'s forward pass has completed
            # and t+(S-s-2)'s backward pass has completed
            if (
                s > 0
                and (
                    t + (num_stages - s) >= num_microbatches
                    or fwd_end[s - 1, t + (num_stages - s)] > 0
                )
                and (t < 1 or bwd_end[s - 1, t - 1] > 0)
            ):
                next_start_time = max(
                    end_time, _safe_get(fwd_end, (s - 1, t + (num_stages - s)))
                )
                next_end_time = next_start_time + bwd_times[s - 1] * loads[t]
                add_event_and_update(s - 1, t, 1, next_start_time, next_end_time)
                logger.debug(
                    f"  >>> Scheduling backward pass of stage {s - 1} for microbatch "
                    f"{t} based on event stage {s} microbatch {t}'s backward pass "
                    "completion"
                )

            # 2. Forward pass of next (S-s)-th microbatch that was waiting on memory
            # constraint. This can only be scheduled if the forward pass of the next
            # microbatch of previous stage has completed
            if t + (num_stages - s) < num_microbatches and (
                s == 0 or fwd_end[s - 1, t + (num_stages - s)] > 0
            ):
                next_start_time = max(
                    end_time, _safe_get(fwd_end, (s - 1, t + (num_stages - s)))
                )
                next_end_time = (
                    next_start_time + fwd_times[s] * loads[t + (num_stages - s)]
                )
                add_event_and_update(
                    s, t + (num_stages - s), 0, next_start_time, next_end_time
                )
                logger.debug(
                    f"  >>> Scheduling forward pass of stage {s} for microbatch "
                    f"{t + (num_stages - s)} based on event stage {s} microbatch "
                    f"{t}'s backward pass completion"
                )

            # 3. Backward pass of next microbatch at this stage
            # (Only during the cooldown phase)
            # and should make sure the backward pass of the previous stage has completed
            if (
                t < num_microbatches - 1
                and t >= num_microbatches - (num_stages - s)
                and bwd_end[s + 1, t] > 0
                and bwd_end[s, t] == 0
            ):
                next_start_time = max(end_time, _safe_get(bwd_end, (s + 1, t)))
                next_end_time = next_start_time + bwd_times[s] * loads[t]
                add_event_and_update(s, t, 1, next_start_time, next_end_time)
                logger.debug(
                    f"  >>> Scheduling backward pass of stage {s} for microbatch {t} "
                    f"based on event stage {s + 1} microbatch {t}'s backward pass "
                    f"completion"
                )

    # Calculate total time as the completion time of the last microbatch's backward pass
    # in the first stage
    total_time = bwd_end[0, num_microbatches - 1]

    # Calculate bubble ratio for each stage
    bubble_ratio = np.zeros(num_stages)
    for s in range(num_stages):
        execution_time = np.sum(fwd_end[s, :] - fwd_start[s, :]) + np.sum(
            bwd_end[s, :] - bwd_start[s, :]
        )
        bubble_ratio[s] = (total_time - execution_time) / total_time
    logger.info(
        "Bubble ratio: "
        f"{
            ', '.join(
                f'Stage {s}: {bubble_ratio[s] * 100:.2f}%' for s in range(num_stages)
            )
        }"
    )

    # Adjusted bubble ratio for each stage: the time range is from the end of
    # the first backward to the end of the last forward
    adjusted_bubble_ratio = np.zeros(num_stages)
    for s in range(num_stages):
        logger.warning(
            f"Adjusted bubble ratio for stage {s}: "
            f"total_time: bwd of 0 -> fwd of {num_microbatches - 1}, "
            f"fwd: [{num_stages - s}:], "
            f"bwd: [:{num_microbatches - (num_stages - s)}]"
        )
        adjusted_total_time = fwd_end[s, -1] - bwd_start[s, 0]
        adjusted_execution_time = np.sum(
            fwd_end[s, num_stages - s :] - fwd_start[s, num_stages - s :]
        ) + np.sum(
            bwd_end[s, : num_microbatches - (num_stages - s)]
            - bwd_start[s, : num_microbatches - (num_stages - s)]
        )
        adjusted_bubble_ratio[s] = (
            adjusted_total_time - adjusted_execution_time
        ) / adjusted_total_time
    logger.info(
        "Adjusted bubble ratio: "
        f"{
            ', '.join(
                f'Stage {s}: {adjusted_bubble_ratio[s] * 100:.2f}%'
                for s in range(num_stages)
            )
        }"
    )

    return (
        loads,
        fwd_start,
        fwd_end,
        bwd_start,
        bwd_end,
        total_time,
        bubble_ratio,
        adjusted_bubble_ratio,
    )


def find_overlaps(
    fwd_start: np.ndarray,
    fwd_end: np.ndarray,
    bwd_start: np.ndarray,
    bwd_end: np.ndarray,
) -> list[list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Find overlaps in each stage."""
    num_stages, num_microbatches = fwd_start.shape
    overlaps_per_stage = []

    for stage in range(num_stages):
        # Gather intervals for this stage
        fwd_intervals = np.stack((fwd_start[stage], fwd_end[stage]), axis=1)
        bwd_intervals = np.stack((bwd_start[stage], bwd_end[stage]), axis=1)
        all_intervals = np.concatenate((fwd_intervals, bwd_intervals), axis=0)

        # Sort by start time
        sorted_intervals = all_intervals[np.argsort(all_intervals[:, 0])]

        # Check for overlaps
        overlaps = []
        for i in range(1, len(sorted_intervals)):
            prev = sorted_intervals[i - 1]
            curr = sorted_intervals[i]
            if curr[0] < prev[1]:  # Overlap detected
                overlaps.append((tuple(prev), tuple(curr)))

        overlaps_per_stage.append(overlaps)

    return overlaps_per_stage


def check_forward_backward_constraints(
    fwd_start: np.ndarray,
    fwd_end: np.ndarray,
    bwd_start: np.ndarray,
    bwd_end: np.ndarray,
) -> list[int]:
    """Check the forward backward constraints."""
    num_stages, num_microbatches = fwd_start.shape
    errors = []

    for stage in range(num_stages - 1):
        # Forward constraint: stage i+1 must start after stage i ends
        fwd_end_i = fwd_end[stage]
        fwd_start_next = fwd_start[stage + 1]

        for mb in range(num_microbatches):
            if fwd_start_next[mb] < fwd_end_i[mb]:
                logger.warning(
                    f"⛔ Forward timing violation at stage {stage + 1}, "
                    f"microbatch {mb}: starts at {fwd_start_next[mb]} < "
                    f"previous end {fwd_end_i[mb]}"
                )
                errors.append(mb)

        # Backward constraint: stage i must start after stage i+1 ends
        bwd_start_i = bwd_start[stage]
        bwd_end_next = bwd_end[stage + 1]

        for mb in range(num_microbatches):
            if bwd_start_i[mb] < bwd_end_next[mb]:
                logger.warning(
                    f"⛔ Backward timing violation at stage {stage}, microbatch {mb}: "
                    f"starts at {bwd_start_i[mb]} < next end {bwd_end_next[mb]}"
                )
                errors.append(mb)

    # Last stage self check: backward must start after forward ends
    last = num_stages - 1
    for mb in range(num_microbatches):
        if bwd_start[last][mb] < fwd_end[last][mb]:
            logger.warning(
                f"⛔ Last stage {last} violation at microbatch {mb}: backward starts "
                f"at {bwd_start[last][mb]} < forward end {fwd_end[last][mb]}"
            )
            errors.append(mb)

    return sorted(set(errors))


def plot_loads_pipeline(
    loads: list[int],
    fwd_start: np.ndarray,
    fwd_end: np.ndarray,
    bwd_start: np.ndarray,
    bwd_end: np.ndarray,
    num_stages: int,
    *,
    output_filename: str = "pipeline_execution_schedule.png",
) -> None:
    """Plot the pipeline parallelism execution schedule with fwd and bwd passes.

    Args:
        loads: List of loads (token counts) for each microbatch
        fwd_start: 2D array of forward pass start times (stages x microbatches)
        fwd_end: 2D array of forward pass end times (stages x microbatches)
        bwd_start: 2D array of backward pass start times (stages x microbatches)
        bwd_end: 2D array of backward pass end times (stages x microbatches)
        num_stages: Number of pipeline stages
        output_filename: str, name of the output figure file.
    """
    num_microbatches = len(loads)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(50, 5))

    # Set colors for pipeline stages and passes
    fwd_colors = plt.cm.Blues(np.linspace(0.5, 0.9, num_stages))
    bwd_colors = plt.cm.Oranges(np.linspace(0.5, 0.9, num_stages))

    # For visualization clarity
    row_height = 0.8  # Height of each row (pipeline stage)
    gap = 0.0  # Gap between rows

    # Find the maximum completion time to set the x-axis limit
    max_time = max(
        np.max(bwd_end) if np.any(bwd_end) else 0,
        np.max(fwd_end) if np.any(fwd_end) else 0,
    )

    # Loop through all stages and microbatches to plot forward and backward passes
    for s in range(num_stages):
        stage_y_pos = (num_stages - 1 - s) * (row_height + gap)

        # Plot forward passes
        for t in range(num_microbatches):
            if fwd_start[s, t] > 0 or fwd_end[s, t] > 0:
                # Create forward pass rectangle
                fwd_rect = patches.Rectangle(
                    (fwd_start[s, t], stage_y_pos),
                    fwd_end[s, t] - fwd_start[s, t],
                    row_height,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=fwd_colors[s],
                    alpha=0.8,
                )
                ax.add_patch(fwd_rect)

                # Add a text label showing microbatch ID and stage
                ax.text(
                    (fwd_start[s, t] + fwd_end[s, t]) / 2,
                    stage_y_pos + row_height / 2,
                    f"MB{t}\nFWD",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                )

        # Plot backward passes
        for t in range(num_microbatches):
            if bwd_start[s, t] > 0 or bwd_end[s, t] > 0:
                # Create backward pass rectangle
                bwd_rect = patches.Rectangle(
                    (bwd_start[s, t], stage_y_pos),
                    bwd_end[s, t] - bwd_start[s, t],
                    row_height,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=bwd_colors[s],
                    alpha=0.8,
                    hatch="/",
                )
                ax.add_patch(bwd_rect)

                # Add a text label showing microbatch ID and stage
                ax.text(
                    (bwd_start[s, t] + bwd_end[s, t]) / 2,
                    stage_y_pos + row_height / 2,
                    f"MB{t}\nBWD",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                )

    # Set the plot limits and labels
    ax.set_xlim(-max_time * 0.01, max_time * 1.05)
    ax.set_ylim(-gap, num_stages * (row_height + gap))

    # Set y-tick labels for pipeline stages
    y_ticks = [
        (num_stages - 1 - i) * (row_height + gap) + row_height / 2
        for i in range(num_stages)
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"Stage {i}" for i in range(num_stages)])

    # Add time markers (vertical lines) at regular intervals
    interval = max(1, int(max_time / 10))
    for t in range(0, int(max_time) + interval, interval):
        ax.axvline(x=t, color="gray", linestyle="--", alpha=0.3)
        ax.text(t, -gap / 2, str(t), ha="center", va="top", fontsize=8)

    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_title("Pipeline Parallelism Execution Schedule")

    # Add a legend
    fwd_patch = patches.Patch(color=fwd_colors[0], label="Forward Pass")
    bwd_patch = patches.Patch(color=bwd_colors[0], hatch="/", label="Backward Pass")
    ax.legend(handles=[fwd_patch, bwd_patch], loc="upper right")

    # Add information about the execution
    fig.text(
        0.02,
        0.02,
        f"Total execution time: {max_time:.1f}\n"
        f"Number of microbatches: {num_microbatches}\n"
        f"Number of stages: {num_stages}",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.show()


def evaluate_schedule_pipeline_parallelism_and_plot(
    data: list[list[list[int]]],
    schedule: list[list[tuple[int, int, int]]],
    num_stages: int,
    fwd_times: list[int],
    bwd_times: list[int],
    *,
    output_filename: str = "pipeline_execution_schedule.png",
) -> None:
    """Evaluate the schedule and plot the pipeline parallelism execution schedule.

    Args:
        data: nested list as returned by generate_data.
        schedule: list of micro-batches; each micro-batch is a list of (i,j,k) tuples.
        num_stages: int, number of pipeline stages.
        fwd_times: list of int, time multiplier for forward pass at each stage.
        bwd_times: list of int, time multiplier for backward pass at each stage.
        output_filename: str, name of the output figure file.
    """
    (
        loads,
        fwd_start,
        fwd_end,
        bwd_start,
        bwd_end,
        total_time,
        bubble_ratio,
        adjusted_bubble_ratio,
    ) = evaluate_schedule_pipeline_parallelism(
        data,
        schedule,
        num_stages=num_stages,
        padding=False,
        fwd_times=fwd_times,
        bwd_times=bwd_times,
    )

    # Check whether the constraints are violated
    # 1. Find overlaps
    overlaps = find_overlaps(fwd_start, fwd_end, bwd_start, bwd_end)
    if any(overlaps):
        logger.warning("Illegal overlaps found in the schedule.")
        logger.warning(overlaps)
    else:
        logger.success("No illegal overlaps found in the schedule.")
    # 2. Check the forward backward constraints
    errors = check_forward_backward_constraints(fwd_start, fwd_end, bwd_start, bwd_end)
    if errors:
        logger.warning(f"Errors found at microbatches: {errors}")
    else:
        logger.success("No errors found for forward backward constraints.")

    # Plot the schedule
    plot_loads_pipeline(
        loads,
        fwd_start,
        fwd_end,
        bwd_start,
        bwd_end,
        num_stages=num_stages,
        output_filename=output_filename,
    )

    return (
        loads,
        fwd_start,
        fwd_end,
        bwd_start,
        bwd_end,
        total_time,
        bubble_ratio,
        adjusted_bubble_ratio,
    )


def generate_naive_schedule(
    data: list[list[list[int]]],
    microbatch_size: int,
    *,
    interleave_adapters: bool = False,
) -> list[list[tuple[int, int, int]]]:
    """Generate a naive schedule by microbatch_size."""
    schedule = []
    data = np.array(data)
    if interleave_adapters:
        # Interleave batches
        for j in range(data.shape[1]):
            # Interleave adapters
            for i in range(data.shape[0]):
                # Interleave samples
                for microbatch_idx in range(0, data.shape[2], microbatch_size):
                    start_idx = microbatch_idx * microbatch_size
                    end_idx = start_idx + microbatch_size
                    schedule.append([(i, j, k) for k in range(start_idx, end_idx)])
    else:
        # Interleave adapters
        for i in range(data.shape[0]):
            # Interleave batches
            for j in range(data.shape[1]):
                # Batch samples
                for microbatch_idx in range(data.shape[2] // microbatch_size):
                    start_idx = microbatch_idx * microbatch_size
                    end_idx = start_idx + microbatch_size
                    schedule.append([(i, j, k) for k in range(start_idx, end_idx)])
    return schedule


def generate_naive_sort_schedule(
    data: list[list[list[int]]],
    microbatch_size: int,
    *,
    interleave_adapters: bool = False,
) -> list[list[tuple[int, int, int]]]:
    """Generate a schedule by microbatch_size with samples sorted by length.

    This function creates microbatches where samples are grouped by similar lengths,
    which can help reduce padding and improve efficiency.

    Args:
        data: Input data as a nested list of token lengths.
        microbatch_size: Size of each microbatch.
        interleave_adapters:
            Whether to interleave adapters or process them sequentially.

    Returns:
        A list of microbatches, where each microbatch contains (adapter_idx, batch_idx,
          sample_idx) tuples.
    """
    schedule = []
    data_np = np.array(data)

    if interleave_adapters:
        # Interleave batches
        for j in range(data_np.shape[1]):
            # Interleave adapters
            for i in range(data_np.shape[0]):
                # Sort samples by length
                sample_lengths = data_np[i, j, :]
                sorted_indices = np.argsort(sample_lengths)

                # Create microbatches from sorted samples
                for start_idx in range(0, len(sorted_indices), microbatch_size):
                    end_idx = min(start_idx + microbatch_size, len(sorted_indices))
                    # Use the sorted indices to select samples
                    microbatch = [
                        (i, j, sorted_indices[k]) for k in range(start_idx, end_idx)
                    ]
                    schedule.append(microbatch)
    else:
        # Process adapters sequentially
        for i in range(data_np.shape[0]):
            # Process batches sequentially
            for j in range(data_np.shape[1]):
                # Sort samples by length
                sample_lengths = data_np[i, j, :]
                sorted_indices = np.argsort(sample_lengths)

                # Create microbatches from sorted samples
                for start_idx in range(0, len(sorted_indices), microbatch_size):
                    end_idx = min(start_idx + microbatch_size, len(sorted_indices))
                    # Use the sorted indices to select samples
                    microbatch = [
                        (i, j, sorted_indices[k]) for k in range(start_idx, end_idx)
                    ]
                    schedule.append(microbatch)

    return schedule


def generate_balanced_schedule_candidates(
    data: list[list[list[int]]],
    microbatch_token_size_range: tuple[int, int],
    num_stages: int,
) -> list[list[tuple[int, int, int]]]:
    """Generate a proposed schedule using the proposed solver."""
    total_tokens = sum(tree_flatten(data)[0])
    min_num_microbatches = total_tokens // microbatch_token_size_range[1]
    max_num_microbatches = total_tokens // microbatch_token_size_range[0]
    num_microbatches_choices = sorted(
        set(range(min_num_microbatches, max_num_microbatches + 1))
    )
    # For debugging
    num_microbatches_choices = [32]

    schedules = []

    for num_microbatches in num_microbatches_choices:
        schedule = solve_schedule_milp(
            data,
            num_microbatches,
            num_stages=num_stages,
            time_limit=30,
        )
        schedules.append(schedule)

    return schedules


if __name__ == "__main__":
    dist_map = {
        "lognormal": {"mean": 7, "sigma": 0.5},
        "constant": {"length": 7},
    }

    # Parameters
    seed = 42
    microbatch_size = 2
    distribution = "lognormal"

    # Size: Number of adapters
    num_adapters = 2
    batches_per_adapter = [4, 4]
    samples_per_batch = [4, 4]
    # Size: Number of pipline stage
    num_stages = 3
    fwd_times = [1, 1, 1]
    bwd_times = [1, 1, 1]

    # Generate data
    data = generate_data(
        num_adapters=num_adapters,
        batches_per_adapter=batches_per_adapter,
        samples_per_batch=samples_per_batch,
        seed=seed,
        dist=distribution,
        dist_params=dist_map[distribution],
    )

    for i in range(num_adapters):
        logger.info(f"Adapter {i}: {data[i]}")

    DO_NAIVE = True
    DO_BALANCED = True

    if DO_NAIVE:
        # naive schedule
        logger.info("Using `Naive schedule`")
        schedule = generate_naive_schedule(data, microbatch_size=microbatch_size)
        loads, _, _, _, _, total_time, bubble_ratio, adjusted_bubble_ratio = (
            evaluate_schedule_pipeline_parallelism_and_plot(
                data,
                schedule,
                num_stages,
                fwd_times,
                bwd_times,
                output_filename="naive_schedule.png",
            )
        )
        logger.info(f"Naive schedule: {schedule}.")
        logger.info(f"GA Steps: {len(schedule)}. Loads: {loads}")
        logger.info(f"Total time (naive): {total_time}")
        logger.info("=" * 100)

        # naive sort schedule
        logger.info("Using `Naive sort schedule`")
        schedule = generate_naive_sort_schedule(data, microbatch_size=microbatch_size)
        loads, _, _, _, _, total_time, bubble_ratio, adjusted_bubble_ratio = (
            evaluate_schedule_pipeline_parallelism_and_plot(
                data,
                schedule,
                num_stages,
                fwd_times,
                bwd_times,
                output_filename="naive_sort_schedule.png",
            )
        )
        logger.info(f"Naive sort schedule: {schedule}.")
        logger.info(f"GA Steps: {len(schedule)}. Loads: {loads}")
        logger.info(f"Total time (naive sort): {total_time}")
        logger.info("=" * 100)

    # generate balanced schedule candidates
    if DO_BALANCED:
        results = generate_balanced_schedule_candidates(
            data,
            microbatch_token_size_range=(1024, 4096),
            num_stages=num_stages,
        )
        for i, result in enumerate(results):
            schedule, fwd_start, fwd_end, bwd_start, bwd_end = result
            num_stages, num_microbatches = fwd_start.shape
            logger.info(f"Using `Balanced schedule {i}`")
            loads, _, _, _, _, total_time, bubble_ratio, adjusted_bubble_ratio = (
                evaluate_schedule_pipeline_parallelism_and_plot(
                    data,
                    schedule,
                    num_stages,
                    fwd_times,
                    bwd_times,
                    output_filename=f"balanced_schedule_{i}.png",
                )
            )
            overlaps = find_overlaps(fwd_start, fwd_end, bwd_start, bwd_end)
            if any(overlaps):
                logger.warning("Illegal overlaps found in the schedule.")
                logger.warning(overlaps)
            else:
                logger.success("No illegal overlaps found in the schedule.")
            errors = check_forward_backward_constraints(
                fwd_start, fwd_end, bwd_start, bwd_end
            )
            if errors:
                logger.warning(f"Errors found at microbatches: {errors}")
            else:
                logger.success("No errors found for forward backward constraints.")
            plot_loads_pipeline(
                loads,
                fwd_start,
                fwd_end,
                bwd_start,
                bwd_end,
                num_stages=num_stages,
                output_filename=f"balanced_schedule_{i}.png",
            )
            logger.info(f"Balanced schedule {i}: {schedule}.")
            logger.info(f"GA Steps: {len(schedule)}. Loads: {loads}")
            logger.info(f"Total time (balanced {i}): {total_time}")
            logger.info("=" * 100)
