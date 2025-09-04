# ruff: noqa: INP001, ERA001
"""Generate the dataset length distribution."""

# This function collects the token length of the samples of different datasets.
# And then sample the sample lengths from the dataset length distribution.
# Output:
# {
#     "dataset_name": {
#         "seeds": list[int],
#         "seed_42": {
#             "num_permutations": int,
#             "num_samples": int,
#             "median": float,
#             "mean": float,
#             "std": float,
#             "min": float,
#             "max": float,
#             "permutation_1": list[int],
#             "permutation_2": list[int],
#             ...
#         },
#         "seed_16": {
#             "num_permutations": int,
#             "num_samples": int,
#             "median": float,
#             "mean": float,
#             "std": float,
#             "min": float,
#             "max": float,
#             "permutation_1": list[int],
#             "permutation_2": list[int],
#             ...
#         },
#     },
# }

import os

os.environ["HF_HUB_OFFLINE"] = "0"

import json
import random
from pathlib import Path
from typing import Any

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer

from lorafusion.utils.common import set_seed

# Fix font type for PDF and PS files, which is required by the ACM/IEEE templates.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

SEQ_LEN_UPPER_BOUND = 4096
NUM_PERMUTATIONS = 4
DEFAULT_TOKENIZER_NAME = "meta-llama/Meta-Llama-3-70B"

DATASET_INFO = {
    "xsum": {
        "dataset_name": "EdinburghNLP/xsum",
        "dataset_path": None,
        "dataset_split": "train",
        "dataset_keys": ["document", "summary"],
    },
    "cnn_dailymail": {
        "dataset_name": "abisee/cnn_dailymail",
        "dataset_path": "3.0.0",
        "dataset_split": "train",
        "dataset_keys": ["article", "highlights"],
    },
    "wikisum": {
        "dataset_name": "d0rj/wikisum",
        "dataset_path": None,
        "dataset_split": "train",
        "dataset_keys": ["article", "summary"],
    },
    "arxiv": {
        "dataset_name": "ccdv/arxiv-summarization",
        "dataset_path": "section",
        "dataset_split": "validation",
        "dataset_keys": ["article", "abstract"],
    },
}


def sample_dataset_lengths(
    dataset_key: str,
    seeds: list[int],
    num_samples: int,
    num_permutations: int = NUM_PERMUTATIONS,
    upper_bound: int = SEQ_LEN_UPPER_BOUND,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
) -> dict[str, Any]:
    """Sample the sample lengths from the dataset length distribution."""
    if dataset_key not in DATASET_INFO:
        msg = f"Dataset {dataset_key} not supported"
        raise ValueError(msg)

    dataset_info = DATASET_INFO[dataset_key]
    name, path, split, keys = (
        dataset_info["dataset_name"],
        dataset_info["dataset_path"],
        dataset_info["dataset_split"],
        dataset_info["dataset_keys"],
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    result = {"seeds": seeds}

    for seed in seeds:
        set_seed(seed)

        raw_dataset = load_dataset(name, path, split=split)
        dataset = raw_dataset.shuffle(seed=seed)
        dataset = dataset.select(range(num_samples))

        tokens = [
            min(sum(len(tokenizer.encode(sample[key])) for key in keys), upper_bound)
            for sample in dataset
        ]

        mean, std, median, min_len, max_len = (
            float(np.mean(tokens)),
            float(np.std(tokens)),
            float(np.median(tokens)),
            float(np.min(tokens)),
            float(np.max(tokens)),
        )

        seed_result = {
            "num_permutations": num_permutations,
            "num_samples": num_samples,
            "median": median,
            "mean": mean,
            "std": std,
            "min": min_len,
            "max": max_len,
        }

        tokens_copy = tokens.copy()
        for i in range(num_permutations):
            if i > 0:
                random.shuffle(tokens_copy)
            seed_result[f"permutation_{i + 1}"] = tokens_copy.copy()

        result[f"seed_{seed}"] = seed_result

        logger.info(
            f"Dataset {dataset_key}, seed {seed}: "
            f"{mean=}, {std=}, {median=}, {min_len=}, {max_len=}"
        )

    return result


def plot_dataset_distribution(  # noqa: PLR0915
    dataset_distributions: dict[str, Any],
    output_dir: str,
    *,
    plot_mix: bool,
) -> None:
    """Plot the dataset length distribution in a single figure for all datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a single figure for all datasets
    plt.figure(figsize=(6, 2.35))

    # Use different colors for different datasets
    # colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    # colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    # colors = ["#98d2c0", "#4f959d", "#205781"]
    colors = ["#1f77b4", "#ff9494", "#2ca02c"]

    # First collect all tokens for the mix
    all_datasets_tokens = []

    for idx, (dataset_name, data) in enumerate(dataset_distributions.items()):
        if dataset_name == "mix":
            continue  # Skip mix here, we'll plot it last

        # Update the dataset name
        if dataset_name == "xsum":
            dataset_name_in_plot = "XSum"
        elif dataset_name == "cnn_dailymail":
            dataset_name_in_plot = "CNN/DailyMail"
        elif dataset_name == "wikisum":
            dataset_name_in_plot = "WikiSum"
        else:
            dataset_name_in_plot = dataset_name

        # Combine data from all seeds
        all_tokens = []
        means = []

        for seed in data["seeds"]:
            seed_data = data[f"seed_{seed}"]
            tokens = seed_data["permutation_1"]  # Using first permutation
            all_tokens.extend(tokens)
            means.append(seed_data["mean"])

        # Store tokens for the mix
        all_datasets_tokens.append(all_tokens)

        # Calculate average mean across all seeds
        avg_mean = np.mean(means)

        # Plot KDE for this dataset
        color = colors[idx % len(colors)]
        sns.kdeplot(
            all_tokens,
            color=color,
            label=f"{dataset_name_in_plot}",
            fill=True,
            alpha=0.3,
        )

        # Add vertical line for mean
        plt.axvline(
            avg_mean,
            color=color,
            linestyle="--",
            alpha=0.8,
            label=f"{dataset_name_in_plot} mean",
        )

    # Plot the mix if it exists
    if plot_mix and "mix" in dataset_distributions:
        mix_data = dataset_distributions["mix"]
        mix_tokens = []
        mix_means = []

        for seed in mix_data["seeds"]:
            seed_data = mix_data[f"seed_{seed}"]
            tokens = seed_data["permutation_1"]
            mix_tokens.extend(tokens)
            mix_means.append(seed_data["mean"])

        mix_mean = np.mean(mix_means)
        mix_color = colors[-1]  # Use last color for mix

        sns.kdeplot(mix_tokens, color=mix_color, label="mix", fill=True, alpha=0.3)

        plt.axvline(
            mix_mean,
            color=mix_color,
            linestyle="--",
            alpha=0.8,
            label="mix mean",
        )

    # Set x-axis to logarithmic scale if the data spans multiple orders of magnitude
    magnitude_ratio = 10
    if any(
        max(data[f"seed_{seed}"]["permutation_1"])
        / min(data[f"seed_{seed}"]["permutation_1"])
        > magnitude_ratio
        for dataset_name, data in dataset_distributions.items()
        for seed in data["seeds"]
    ):
        # plt.xscale("log")

        # Define nicer x-axis ticks
        xticks = [512, 1024, 2048, 4096]
        xtick_labels = ["512", "1K", "2K", "4K"]
        plt.xticks(xticks, labels=xtick_labels)

    # Format y-axis to be more readable
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # Increase font size of tick labels
    plt.tick_params(axis="both", which="major", labelsize=12)

    plt.xlabel("# Tokens", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(visible=True, alpha=0.3)
    plt.legend(fontsize=11)

    # Save the combined plot
    plt.tight_layout(pad=0.1)
    png_output_path = output_dir / "combined_distribution.png"
    plt.savefig(png_output_path, dpi=300)
    # Also save the plot as a PDF
    pdf_output_path = output_dir / "combined_distribution.pdf"
    plt.savefig(pdf_output_path, dpi=300)

    # Also save to results directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "combined_distribution.png", dpi=300)
    plt.savefig(results_dir / "combined_distribution.pdf", dpi=300)

    plt.close()

    logger.info(f"Combined plot saved to {png_output_path}")
    logger.info(f"Combined plot saved to {pdf_output_path}")
    logger.info(f"Combined plot saved to {results_dir / 'combined_distribution.png'}")
    logger.info(f"Combined plot saved to {results_dir / 'combined_distribution.pdf'}")


def create_dataset_mix(
    dataset_results: dict[str, Any],
    num_samples_per_dataset: int | None = None,
    *,
    magnitude_ratio: float = 10,
) -> dict[str, Any]:
    """Create a mix of all datasets with equal representation."""
    if not dataset_results:
        return {}

    # Get seeds from the first dataset
    first_dataset = next(iter(dataset_results.values()))
    seeds = first_dataset["seeds"]

    # Get all dataset keys except 'mix' if it exists
    dataset_keys = [k for k in dataset_results if k != "mix"]

    mix_result = {"seeds": seeds}

    for seed in seeds:
        # Collect tokens from each dataset for this seed
        all_datasets_tokens = []

        for dataset_key in dataset_keys:
            dataset_seed_data = dataset_results[dataset_key][f"seed_{seed}"]
            tokens = dataset_seed_data["permutation_1"]

            # If num_samples_per_dataset is specified, sample that many tokens
            if num_samples_per_dataset:
                # If there are fewer tokens than requested, use all available
                if len(tokens) <= num_samples_per_dataset:
                    sampled_tokens = tokens
                else:
                    # Randomly sample without replacement
                    indices = np.random.choice(  # noqa: NPY002
                        len(tokens), size=num_samples_per_dataset, replace=False
                    )
                    sampled_tokens = [tokens[i] for i in indices]
                all_datasets_tokens.append(sampled_tokens)
            else:
                all_datasets_tokens.append(tokens)

        # Flatten the list of tokens
        mixed_tokens = [
            token for dataset_tokens in all_datasets_tokens for token in dataset_tokens
        ]
        # Shuffle with seed
        set_seed(seed)
        random.shuffle(mixed_tokens)
        mixed_tokens = mixed_tokens[:num_samples_per_dataset]

        # Calculate statistics
        mean = float(np.mean(mixed_tokens))
        std = float(np.std(mixed_tokens))
        median = float(np.median(mixed_tokens))
        min_len = float(np.min(mixed_tokens))
        max_len = float(np.max(mixed_tokens))

        # Create seed result
        seed_result = {
            "num_permutations": 1,  # Only storing one permutation
            "num_samples": len(mixed_tokens),
            "median": median,
            "mean": mean,
            "std": std,
            "min": min_len,
            "max": max_len,
            "permutation_1": mixed_tokens,
        }

        mix_result[f"seed_{seed}"] = seed_result

        logger.info(
            f"Mix dataset, seed {seed}: "
            f"{mean=}, {std=}, {median=}, {min_len=}, {max_len=}"
        )

    return mix_result


@click.command()
@click.option(
    "--seeds",
    type=click.INT,
    multiple=True,
    default=[42, 16, 1234, 12345, 52, 1, 2, 3],
    help="The seeds for the random number generator.",
)
@click.option(
    "--datasets",
    type=str,
    default="xsum,cnn_dailymail,wikisum",
    help="The datasets to use. Use comma to separate multiple datasets.",
)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="The number of samples to generate.",
)
@click.option(
    "--num-permutations",
    type=int,
    default=NUM_PERMUTATIONS,
    help="Number of permutations to generate.",
)
@click.option(
    "--plot-distribution",
    is_flag=True,
    default=True,
    help="Whether to plot the dataset length distribution.",
)
@click.option(
    "--output-dir",
    type=str,
    default="./datasets",
    help="The output directory.",
)
@click.option(
    "--tokenizer-name",
    type=str,
    default=DEFAULT_TOKENIZER_NAME,
    help="The tokenizer to use for tokenization.",
)
@click.option(
    "--load-from-json",
    type=str,
    default=None,
    help="The path to the JSON file to load the dataset distributions from.",
)
@click.option(
    "--include-mix",
    is_flag=True,
    default=True,
    help="Whether to include a mix of all datasets.",
)
@click.option(
    "--samples-per-dataset-in-mix",
    type=int,
    default=None,
    help="Number of samples to include from each dataset in the mix. Default is 1000.",
)
@click.option(
    "--plot-mix",
    is_flag=True,
    default=False,
    help="Whether to plot the mix of all datasets.",
)
def main(
    seeds: tuple[int],
    datasets: str,
    num_samples: int,
    num_permutations: int,
    output_dir: str,
    tokenizer_name: str,
    load_from_json: str | None,
    samples_per_dataset_in_mix: int | None,
    *,
    include_mix: bool,
    plot_distribution: bool,
    plot_mix: bool,
) -> None:
    """Generate the dataset length distribution."""
    if samples_per_dataset_in_mix is None:
        samples_per_dataset_in_mix = num_samples

    seeds_list = list(seeds)
    dataset_list = datasets.split(",")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if load_from_json:
        load_from_json_path = Path(load_from_json)
        if not load_from_json_path.exists():
            msg = f"File {load_from_json} not found"
            raise FileNotFoundError(msg)
        with load_from_json_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

    else:
        result = {}

        for dataset_key in dataset_list:
            logger.info(f"Processing dataset: {dataset_key}")
            try:
                dataset_result = sample_dataset_lengths(
                    dataset_key=dataset_key,
                    seeds=seeds_list,
                    num_samples=num_samples,
                    num_permutations=num_permutations,
                    upper_bound=SEQ_LEN_UPPER_BOUND,
                    tokenizer_name=tokenizer_name,
                )
                result[dataset_key] = dataset_result

                # Print statistics
                for seed in seeds_list:
                    seed_data = dataset_result[f"seed_{seed}"]
                    logger.info(
                        f"Dataset {dataset_key}, Seed {seed}: "
                        f"mean={seed_data['mean']:.2f}, std={seed_data['std']:.2f}, "
                        f"median={seed_data['median']:.2f}, "
                        f"min={seed_data['min']:.2f}, "
                        f"max={seed_data['max']:.2f}"
                    )

            except KeyError as e:
                logger.error(f"Error processing dataset {dataset_key}: {e}")
            logger.info("=" * 100)

        # Create mix if requested
        if include_mix and len(result) > 0:
            logger.info("Creating dataset mix")
            mix_result = create_dataset_mix(result, samples_per_dataset_in_mix)
            result["mix"] = mix_result
            logger.info("=" * 100)

        # Save results to JSON
        json_path = output_path / "dataset_distributions.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=False)

        logger.success(f"Results saved to {json_path}")

    if plot_distribution:
        plot_dataset_distribution(result, str(output_path), plot_mix=plot_mix)


if __name__ == "__main__":
    main()
