"""Simulate the imbalance of multi LoRA."""

import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
from loguru import logger
from main_random_data import evaluate_schedule_pipeline_parallelism_and_plot

from lorafusion.solver.proposed_solver import (
    group_adapters,
    packing_data_to_microbatches_micro_milp,
    packing_data_to_microbatches_micro_milp_with_multiprocessing,
)
from lorafusion.train.training_utils import MockDataArguments, MockDataset
from lorafusion.utils.common import (
    json_utils_default,
    list_of_floats,
    list_of_ints,
    set_seed,
    stringify_keys,
    update_loguru_level,
)

update_loguru_level("INFO")

# Notation:
# - adapter_idx: i
# - batch_idx: j
# - sample_idx: k
# Schedule is a list of (list[tuple[adapter_idx, batch_idx, sample_idx]], list[int])
# where the last element is a list of ints indicating the optimizer step for adapters
# should be performed after this batch is finished


def generate_mlora_schedule(
    aggregated_dataset: list[list[list[int]]],
    microbatch_size: int,
    max_num_batches_to_schedule: int = 10000,
    **kwargs,
) -> list[tuple[list[tuple[int, int, int]], list[int]]]:
    """Perform the mLoRA schedule.

    mLoRA schedule:
    1. For each adapter, do all the batches to finish a global batch.
    2. Switch to the next adapter and repeat.
    """
    num_adapters = len(aggregated_dataset)
    schedule = []
    num_batches_in_each_adapter = [len(dataset) for dataset in aggregated_dataset]
    max_num_batches = max(num_batches_in_each_adapter)
    for j in range(max_num_batches):
        for i in range(num_adapters):
            # If the current adapter has no more batches, skip
            if j >= num_batches_in_each_adapter[i]:
                continue

            # Get the current batch
            batch = aggregated_dataset[i][j]
            num_samples_in_batch = len(batch)

            # Interleave the samples
            for microbatch_idx in range(
                (num_samples_in_batch + microbatch_size - 1) // microbatch_size
            ):
                start_idx = microbatch_idx * microbatch_size
                end_idx = min(start_idx + microbatch_size, num_samples_in_batch)
                sample_indices = [(i, j, k) for k in range(start_idx, end_idx)]
                schedule.append((sample_indices, []))

                if len(schedule) >= max_num_batches_to_schedule:
                    # Return the schedule if we have scheduled enough batches
                    return schedule

            # For the last batch, we need to perform the optimizer step
            schedule[-1][1].append(i)

    return schedule


def convert_schedule_as_dataset_format(
    schedule: list[tuple[list[tuple[int, int, int]], list[int]]],
    aggregated_dataset: list[list[list[int]]],
) -> dict:
    """Convert the schedule to the dataset format."""
    dataset_name = "heterogeneous"
    seed_0 = 0

    micro_batchs = []
    for microbatch, _ in schedule:
        micro_batchs.extend(aggregated_dataset[i][j][k] for i, j, k in microbatch)

    np_micro_batchs = np.array(micro_batchs)
    num_samples = np_micro_batchs.size
    median = np.median(np_micro_batchs).item()
    mean = np.mean(np_micro_batchs).item()
    std = np.std(np_micro_batchs).item()
    min_val = np.min(np_micro_batchs).item()
    max_val = np.max(np_micro_batchs).item()

    return {
        dataset_name: {
            "seeds": [seed_0],
            "num_samples": num_samples,
            "median": median,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            f"seed_{seed_0}": {
                "num_permutations": 1,
                "permutation_1": micro_batchs,
            },
        },
    }


dataset_list = [
    # dataset name, seed_idx, permutation_idx
    ("xsum", 0, 0),
    ("xsum", 1, 0),
    ("xsum", 2, 0),
    ("xsum", 3, 0),
    ("cnn_dailymail", 0, 0),
    ("cnn_dailymail", 1, 0),
    ("cnn_dailymail", 2, 0),
    ("cnn_dailymail", 3, 0),
    ("wikisum", 0, 0),
    ("wikisum", 1, 0),
    ("wikisum", 2, 0),
    ("wikisum", 3, 0),
    ("mix", 0, 0),
    ("mix", 1, 0),
    ("mix", 2, 0),
    ("mix", 3, 0),
]


@click.command()
@click.option(
    "--dataset_path",
    default="datasets/dataset_distributions.json",
    help="Path to the dataset distributions file",
)
@click.option(
    "--capacity",
    default=4096,
    type=int,
    help="Capacity of the model",
)
@click.option("--num_adapters", default=4, type=int, help="Number of adapters")
@click.option(
    "--num_pipeline_stages", default=4, type=int, help="Number of pipeline stages"
)
@click.option(
    "--adapter_to_dataset_idx",
    default="0, 4, 8, 12",
    callback=lambda _, __, value: list_of_ints(value),
    help="Dataset indices for each adapter (comma-separated)",
)
@click.option(
    "--adapter_to_global_batch_size",
    default="8, 8, 8, 8",
    callback=lambda _, __, value: list_of_ints(value),
    help="Global batch sizes for each adapter (comma-separated)",
)
@click.option(
    "--fwd_times",
    default="1, 1, 1, 1.1",
    callback=lambda _, __, value: list_of_floats(value),
    help="Forward times for each pipeline stage (comma-separated)",
)
@click.option(
    "--bwd_times",
    default="1, 1, 1, 1.1",
    callback=lambda _, __, value: list_of_floats(value),
    help="Backward times for each pipeline stage (comma-separated)",
)
@click.option(
    "--min_num_batches",
    default=1000,
    type=int,
    help="Number of batches to use",
)
@click.option(
    "--output_name",
    default="schedule",
    type=str,
    help="Output name",
)
@click.option(
    "--perform_mlora_schedule_only",
    default=False,
    type=bool,
    is_flag=True,
    help="Whether to perform mLoRA schedule only",
)
@click.option(
    "--mlora_microbatch_size",
    default=1,
    type=int,
    help="Microbatch size for mLoRA schedule",
)
def main(  # noqa: PLR0915
    dataset_path: str,
    capacity: int,
    num_adapters: int,
    num_pipeline_stages: int,
    adapter_to_dataset_idx: list[int],
    adapter_to_global_batch_size: list[int],
    fwd_times: list[float],
    bwd_times: list[float],
    min_num_batches: int,
    output_name: str,
    *,
    perform_mlora_schedule_only: bool = False,
    mlora_microbatch_size: int | None = None,
) -> None:
    """Run the imbalance simulator with real datasets."""
    # Perform checks
    path = Path(dataset_path)
    if not path.exists():
        msg = f"Dataset path {dataset_path} does not exist"
        raise FileNotFoundError(msg)
    if (
        len(adapter_to_dataset_idx) != num_adapters
        or len(adapter_to_global_batch_size) != num_adapters
    ):
        msg = (
            f"adapter_to_dataset_idx and adapter_to_global_batch_size must have "
            f"length {num_adapters}"
        )
        raise ValueError(msg)

    if len(fwd_times) != num_pipeline_stages or len(bwd_times) != num_pipeline_stages:
        msg = f"fwd_times and bwd_times must have length {num_pipeline_stages}"
        raise ValueError(msg)

    # Utils
    set_seed(42)

    # Create the datasets
    mock_datasets = []
    aggregated_dataset = []
    for adapter_idx in range(num_adapters):
        # Load the dataset
        dataset_name, seed_idx, permutation_idx = dataset_list[
            adapter_to_dataset_idx[adapter_idx]
        ]
        mock_data_args = MockDataArguments(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            seed_idx=seed_idx,
            permutation_idx=permutation_idx,
        )
        mock_dataset = MockDataset.from_dataset_args(mock_data_args)
        mock_datasets.append(mock_dataset)

        # aggregate the dataset
        global_batch_size = adapter_to_global_batch_size[adapter_idx]
        curr_dataset = []
        for batch_idx in range(len(mock_dataset) // global_batch_size):
            batch = mock_dataset[
                batch_idx * global_batch_size : (batch_idx + 1) * global_batch_size
            ]
            curr_dataset.append(batch)
        aggregated_dataset.append(curr_dataset)

        # Update the minimum number of batches
        min_num_batches = min(min_num_batches, len(curr_dataset))

    # Do the mLoRA schedule
    if perform_mlora_schedule_only:
        if mlora_microbatch_size is None:
            msg = (
                "mlora_microbatch_size must be provided if "
                "perform_mlora_schedule_only is True"
            )
            raise ValueError(msg)
        schedule = generate_mlora_schedule(
            aggregated_dataset,
            microbatch_size=mlora_microbatch_size,
        )
        show_plot = False
        if show_plot:
            loads, _, _, _, _, total_time, bubble_ratio, adjusted_bubble_ratio = (
                evaluate_schedule_pipeline_parallelism_and_plot(
                    aggregated_dataset,
                    schedule,
                    num_pipeline_stages,
                    fwd_times,
                    bwd_times,
                    output_filename="mlora_schedule.png",
                )
            )
        # Save the schedule
        dataset_dir_path = Path(dataset_path).parent
        schedule_dir_path = dataset_dir_path / "schedules"
        schedule_dir_path.mkdir(parents=True, exist_ok=True)
        schedule_file_path = (
            schedule_dir_path / f"baseline_schedule_mlora_{output_name}.json"
        )
        with schedule_file_path.open("w") as f:
            json.dump(schedule, f, indent=2, default=json_utils_default)

        # Save the dataset format
        dataset_format = convert_schedule_as_dataset_format(
            schedule, aggregated_dataset
        )
        dataset_format_file_path = (
            schedule_dir_path / f"baseline_dataset_mlora_{output_name}.json"
        )
        with dataset_format_file_path.open("w") as f:
            json.dump(dataset_format, f, indent=2, default=json_utils_default)

        return

    # Solve the MILP
    # For the testing purposes, we cut the batches of each dataset to the same length
    # although it may not be necessary
    use_mp = True
    schedule_fn = (
        packing_data_to_microbatches_micro_milp_with_multiprocessing
        if use_mp
        else packing_data_to_microbatches_micro_milp
    )

    aggregated_dataset = [data[:min_num_batches] for data in aggregated_dataset]

    # print(type(aggregated_dataset))
    # print(len(aggregated_dataset))
    # print(len(aggregated_dataset[0]))
    # print(len(aggregated_dataset[1]))
    # print(len(aggregated_dataset[2]))
    # print(len(aggregated_dataset[3]))
    # print(len(aggregated_dataset[0][0]))
    # exit(0)

    groups = group_adapters(
        aggregated_dataset,
        num_stages=num_pipeline_stages,
        capacity=capacity,
    )
    start_time = time.time()
    schedule = schedule_fn(
        data=aggregated_dataset,
        groups=groups,
        num_pipeline_stages=num_pipeline_stages,
        num_global_batches_per_adapter=min_num_batches,
        capacity=capacity,
        adapter_padding_multiple=128,
        verbose=False,
        time_limit=3.0,
    )
    end_time = time.time()
    total_micro_batch_size = sum(len(s.micro_batch_infos) for s in schedule)
    total_tokens = sum(m.total_tokens for s in schedule for m in s.micro_batch_infos)
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Total number of micro-batches: {total_micro_batch_size}")
    logger.info(
        f"Avg tokens per micro-batch: {total_tokens / total_micro_batch_size:.2f}"
    )

    # Save the schedule as a json file for easy inspection
    dataset_dir_path = Path(dataset_path).parent
    schedule_dir_path = dataset_dir_path / "schedules"
    schedule_dir_path.mkdir(parents=True, exist_ok=True)
    schedule_file_path = schedule_dir_path / f"{output_name}.json"
    with schedule_file_path.open("w") as f:
        json.dump(
            [stringify_keys(asdict(step)) for step in schedule],
            f,
            indent=2,
            default=json_utils_default,
        )

    # Save the schedule as a pickle file for loading
    schedule_pickle_file_path = schedule_dir_path / f"{output_name}.pkl"
    with schedule_pickle_file_path.open("wb") as f:
        pickle.dump(schedule, f)


if __name__ == "__main__":
    main()
