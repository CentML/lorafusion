"""Kernel Performance."""

"""
In this figure, we want to show the performance of the kernel (tokens/s).

Specifically, we will load two csv files:
1. lora-kernels-h100-ref.csv: ../kernel-results/lora-kernels-h100-ref.csv
2. lora-kernels_multi_lora-h100-ref.csv: ../kernel-results/lora-kernels_multi_lora-h100-ref.csv

We will merge them together.

1. We will have three blocks. Similar to three blocks in Figure 9 of https://arxiv.org/pdf/2504.12984.
2. We use number of tokens as the x-axis.
3. The y-axis is the throughput (tokens/s).
4. I think line chart is better here.
"""

import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from lorafusion.utils.multi_line_plot import MultiLinePlot, MultiLinePlotArguments

# Constants for the figure
FIGURE_SIZE = (6.3, 2.1)
DPI = 300

# Input data paths relative to this script
SINGLE_LORA_CSV = "../kernel-results/lora-kernels.csv"
MULTI_LORA_CSV = "../kernel-results/lora-kernels_multi_lora.csv"

# Hidden dimensions to plot (as per the three blocks in Figure 9)
HIDDEN_DIMS = [4096, 5120, 8192]

# Maximum sequence length to include in the plot
MAX_SEQ_LENGTH = 8192

# Specific x-axis ticks to show
X_TICKS = [2048, 4096, 6144, 8192]

# Palette to match other evaluation figures
# PALETTE = ["#a0a0a0", "#55bdac", "#f6a800", "#d4619d"]
PALETTE = ["#a0a0a0", "#55bdac", "#e88d89", "#d4619d"]

# Model types to compare (removed "Torch Linear" as requested)
MODEL_TYPES = {
    "fwd_bwd": ["Torch LoRA", "FlashLoRA", "MultiLoRA"],
    "fwd": ["Torch LoRA", "FlashLoRA", "MultiLoRA"],
}

# Labels for plot legend
LEGEND_LABELS = {
    "Torch LoRA": "Torch LoRA",
    "FlashLoRA": "FusedLoRA",
    "MultiLoRA": "FusedMultiLoRA",
}


def extract_tokens_from_shape(shape_str: str) -> int | None:
    """Extract the number of tokens from a shape string like [2048x4096x4096x16]."""
    match = re.match(r"\[(\d+)x\d+x\d+x\d+\]", shape_str)
    if match:
        return int(match.group(1))

    # For MultiLoRA format like: MultiLoRA-4x[2048x4096x4096x16]
    match = re.match(r"MultiLoRA-\d+x\[(\d+)x\d+x\d+x\d+\]", shape_str)
    if match:
        return int(match.group(1))

    return None


def read_lora_kernels_csv(
    file_path: str,
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """Read LoRA kernels CSV and extract execution times for both FWD and FWD+BWD."""
    data: dict[int, dict[str, dict[str, dict[str, float]]]] = {}

    try:
        with open(file_path) as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read header row

            for row in reader:
                if not row:  # Skip empty rows
                    continue

                shape = row[0]
                tokens = extract_tokens_from_shape(shape)

                if tokens is None:
                    continue

                # Skip data points with sequence length larger than MAX_SEQ_LENGTH
                if tokens > MAX_SEQ_LENGTH:
                    continue

                # Extract hidden dimension from shape (e.g., 4096 from [2048x4096x4096x16])
                hidden_dim_match = re.search(r"x(\d+)x\1x", shape)
                if not hidden_dim_match:
                    continue

                hidden_dim = int(hidden_dim_match.group(1))

                # Determine if this is FWD only or FWD+BWD
                op_type = "fwd" if "FWD" in shape and "BWD" not in shape else "fwd_bwd"

                # Initialize data structure if needed
                if hidden_dim not in data:
                    data[hidden_dim] = {}

                if str(tokens) not in data[hidden_dim]:
                    data[hidden_dim][str(tokens)] = {"fwd": {}, "fwd_bwd": {}}

                # Store values for the selected model types (skip "Torch Linear")
                for i, header in enumerate(headers[1:], 1):
                    if i < len(row) and row[i] and header != "Torch Linear":
                        try:
                            data[hidden_dim][str(tokens)][op_type][header] = float(
                                row[i]
                            )
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not convert value '{row[i]}' to float for {header}"
                            )
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return data


def read_multi_lora_csv(
    file_path: str,
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """Read MultiLoRA kernels CSV and extract execution times for both FWD and FWD+BWD."""
    data: dict[int, dict[str, dict[str, dict[str, float]]]] = {}

    try:
        with open(file_path) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row

            for row in reader:
                if not row:  # Skip empty rows
                    continue

                shape = row[0]
                tokens = extract_tokens_from_shape(shape)

                if tokens is None:
                    continue

                # Skip data points with sequence length larger than MAX_SEQ_LENGTH
                if tokens > MAX_SEQ_LENGTH:
                    continue

                # Extract hidden dimension from shape
                hidden_dim_match = re.search(r"x(\d+)x\1x", shape)
                if not hidden_dim_match:
                    continue

                hidden_dim = int(hidden_dim_match.group(1))

                # Determine if this is FWD only or FWD+BWD
                op_type = "fwd" if "FWD" in shape and "BWD" not in shape else "fwd_bwd"

                # Initialize data structure if needed
                if hidden_dim not in data:
                    data[hidden_dim] = {}

                if str(tokens) not in data[hidden_dim]:
                    data[hidden_dim][str(tokens)] = {"fwd": {}, "fwd_bwd": {}}

                # Store MultiLoRA value
                if len(row) > 1 and row[1]:
                    try:
                        data[hidden_dim][str(tokens)][op_type]["MultiLoRA"] = float(
                            row[1]
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not convert value '{row[1]}' to float for MultiLoRA"
                        )
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return data


def merge_data(
    single_lora_data: dict[int, dict[str, dict[str, dict[str, float]]]],
    multi_lora_data: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """Merge the single and multi LoRA data."""
    merged_data: dict[int, dict[str, dict[str, dict[str, float]]]] = {}

    # First copy single LoRA data
    for hidden_dim, tokens_dict in single_lora_data.items():
        if hidden_dim not in merged_data:
            merged_data[hidden_dim] = {}

        for tokens, op_types_dict in tokens_dict.items():
            if tokens not in merged_data[hidden_dim]:
                merged_data[hidden_dim][tokens] = {"fwd": {}, "fwd_bwd": {}}

            for op_type, models_dict in op_types_dict.items():
                for model, value in models_dict.items():
                    merged_data[hidden_dim][tokens][op_type][model] = value

    # Then add multi LoRA data
    for hidden_dim, tokens_dict in multi_lora_data.items():
        if hidden_dim not in merged_data:
            merged_data[hidden_dim] = {}

        for tokens, op_types_dict in tokens_dict.items():
            if tokens not in merged_data[hidden_dim]:
                merged_data[hidden_dim][tokens] = {"fwd": {}, "fwd_bwd": {}}

            for op_type, models_dict in op_types_dict.items():
                for model, value in models_dict.items():
                    merged_data[hidden_dim][tokens][op_type][model] = value

    return merged_data


def calculate_throughput(
    time_data: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """Convert execution times to throughput (tokens/second)."""
    throughput_data: dict[int, dict[str, dict[str, dict[str, float]]]] = {}

    for hidden_dim, tokens_dict in time_data.items():
        if hidden_dim not in throughput_data:
            throughput_data[hidden_dim] = {}

        for tokens, op_types_dict in tokens_dict.items():
            if tokens not in throughput_data[hidden_dim]:
                throughput_data[hidden_dim][tokens] = {"fwd": {}, "fwd_bwd": {}}

            for op_type, models_dict in op_types_dict.items():
                for model, time_value in models_dict.items():
                    # Calculate throughput as tokens/second
                    if time_value > 0:
                        throughput = int(tokens) / time_value
                        throughput_data[hidden_dim][tokens][op_type][model] = throughput

    return throughput_data


def normalize_throughput(
    throughput_data: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """Normalize throughput data relative to Torch LoRA's first data point."""
    normalized_data: dict[int, dict[str, dict[str, dict[str, float]]]] = {}

    for hidden_dim, tokens_dict in throughput_data.items():
        if hidden_dim not in normalized_data:
            normalized_data[hidden_dim] = {}

        # Find the baselines for normalization for each operation type
        baselines = {}

        for op_type in ["fwd", "fwd_bwd"]:
            baseline = None

            # Find the first available Torch LoRA data point
            sorted_tokens = sorted([int(t) for t in tokens_dict.keys()])
            for token in sorted_tokens:
                token_str = str(token)
                if token_str in tokens_dict and op_type in tokens_dict[token_str]:
                    if "Torch LoRA" in tokens_dict[token_str][op_type]:
                        baseline = tokens_dict[token_str][op_type]["Torch LoRA"]
                        break

            baselines[op_type] = baseline

        # Normalize the data
        for tokens, op_types_dict in tokens_dict.items():
            if tokens not in normalized_data[hidden_dim]:
                normalized_data[hidden_dim][tokens] = {"fwd": {}, "fwd_bwd": {}}

            for op_type, models_dict in op_types_dict.items():
                baseline = baselines.get(op_type)
                if baseline is None or baseline == 0:
                    # Skip normalization if no valid baseline
                    for model, value in models_dict.items():
                        normalized_data[hidden_dim][tokens][op_type][model] = value
                else:
                    # Normalize against baseline
                    for model, value in models_dict.items():
                        normalized_value = value / baseline
                        normalized_data[hidden_dim][tokens][op_type][
                            model
                        ] = normalized_value

    return normalized_data


def find_local_maxima(data: list[float | None], x_values: list[int]) -> list[int]:
    """Find indices of local maxima in the data series.

    A local maximum is a data point that has a higher value than both its left and right neighbors.
    For data with alternating None values, we compare each point with the nearest non-None neighbors.

    Args:
        data: List of data points, may contain None values
        x_values: List of corresponding x values

    Returns:
        List of indices where local maxima occur in the data series
    """
    maxima_indices = []

    # Need at least 3 points to find a local maximum (including None values)
    if len(data) < 3:
        return maxima_indices

    # First, filter out None values and keep track of original indices
    valid_data = []
    valid_indices = []

    for i, value in enumerate(data):
        if value is not None:
            valid_data.append(value)
            valid_indices.append(i)

    # Need at least 3 valid points to find local maxima
    if len(valid_data) < 3:
        return maxima_indices

    # Check each point in the valid data (except first and last) to see if it's a local maximum
    for i in range(1, len(valid_data) - 1):
        if valid_data[i] > valid_data[i - 1] and valid_data[i] > valid_data[i + 1]:
            # This is a local maximum among the valid points
            maxima_indices.append(valid_indices[i])

    # Check the first point
    if len(valid_data) >= 2 and valid_data[0] > valid_data[1]:
        maxima_indices.append(valid_indices[0])

    # Check the last point
    if len(valid_data) >= 2 and valid_data[-1] > valid_data[-2]:
        maxima_indices.append(valid_indices[-1])

    # Return the indices of the local maxima in the original data array
    return maxima_indices


def prepare_plot_data(
    throughput_data: dict[int, dict[str, dict[str, dict[str, float]]]],
    hidden_dims: list[int],
    model_types: dict[str, list[str]],
) -> dict[str, dict[int, dict[str, list]]]:
    """Prepare data for plotting, organizing by operation type and hidden dimension."""
    # Initialize the data structure for the plot
    subplot_data: dict[str, dict[int, dict[str, list]]] = {
        "fwd_bwd": {},
        "fwd": {},
    }

    # Loop through each hidden dimension we want to plot
    for hidden_dim in hidden_dims:
        if hidden_dim not in throughput_data:
            logger.warning(f"No data for hidden dimension {hidden_dim}")
            continue

        # Loop through each operation type (FWD+BWD or FWD only)
        for op_type in ["fwd_bwd", "fwd"]:
            if op_type not in model_types:
                logger.warning(f"No model types for operation {op_type}")
                continue

            # Get all token values and sort them
            all_tokens = sorted([int(t) for t in throughput_data[hidden_dim].keys()])

            if not all_tokens:
                logger.warning(f"No token values for hidden dimension {hidden_dim}")
                continue

            # Initialize data structure for this hidden dimension
            subplot_data[op_type][hidden_dim] = {
                "x_values": all_tokens,
                "data": [],
                "series": [],
            }

            # For each model type, collect the throughput values
            for model in model_types[op_type]:
                model_data = []
                has_data = False

                for tokens in all_tokens:
                    if (
                        str(tokens) in throughput_data[hidden_dim]
                        and op_type in throughput_data[hidden_dim][str(tokens)]
                        and model in throughput_data[hidden_dim][str(tokens)][op_type]
                    ):
                        model_data.append(
                            throughput_data[hidden_dim][str(tokens)][op_type][model]
                        )
                        has_data = True
                    else:
                        model_data.append(None)  # Missing data point

                if has_data:
                    # Add the regular series
                    subplot_data[op_type][hidden_dim]["data"].append(model_data)
                    subplot_data[op_type][hidden_dim]["series"].append(model)

            # Add local maxima points for each model that should have them
            for model_idx, model in enumerate(
                subplot_data[op_type][hidden_dim]["series"]
            ):
                # Skip if model=="MultiLoRA" and hidden_dim == 8192
                if model == "MultiLoRA" and hidden_dim == 8192:
                    continue

                # Add local maxima for both FlashLoRA and MultiLoRA for all hidden dimensions
                if model in ["FlashLoRA", "MultiLoRA"]:
                    model_data = subplot_data[op_type][hidden_dim]["data"][model_idx]

                    logger.info(f"Model data: {model_data} for {model}")

                    # Find local maxima in this model's data
                    local_maxima_indices = find_local_maxima(model_data, all_tokens)

                    if local_maxima_indices:
                        # Create a new series just for the maxima points
                        maxima_data = [None] * len(all_tokens)

                        # Mark only the local maxima points
                        for idx in local_maxima_indices:
                            maxima_data[idx] = model_data[idx]

                        # Add this as a separate series
                        subplot_data[op_type][hidden_dim]["data"].append(maxima_data)
                        subplot_data[op_type][hidden_dim]["series"].append(
                            f"{model}-maxima"
                        )

    return subplot_data


def create_figure(
    plot_data: dict[str, dict[int, dict[str, list]]],
    hidden_dims: list[int],
) -> None:
    """Create the figure with subplots for each hidden dimension and operation type."""
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Create separate figures for FWD+BWD and FWD
    for op_type in ["fwd_bwd", "fwd"]:
        # Skip if no data for this operation type
        if not plot_data.get(op_type):
            logger.warning(f"No data for operation type {op_type}")
            continue

        # Create a new figure
        fig, axes = plt.subplots(1, len(hidden_dims), figsize=FIGURE_SIZE, dpi=DPI)

        # If there's only one subplot, make sure axes is a list
        if len(hidden_dims) == 1:
            axes = [axes]

        last_plot_instance = None

        # Create each subplot
        for i, hidden_dim in enumerate(hidden_dims):
            if hidden_dim not in plot_data[op_type]:
                logger.warning(
                    f"No data for hidden dimension {hidden_dim} in {op_type}"
                )
                continue

            # Prepare data and series
            series_data = plot_data[op_type][hidden_dim]["data"]
            series_names = plot_data[op_type][hidden_dim]["series"]
            x_values = plot_data[op_type][hidden_dim]["x_values"]

            if not series_names:
                logger.warning(
                    f"No series for hidden dimension {hidden_dim} in {op_type}"
                )
                continue

            # Separate the regular series from the maxima series
            regular_series_data = []
            regular_series_names = []
            maxima_series_data = []
            maxima_series_names = []

            for idx, series_name in enumerate(series_names):
                if "-maxima" in series_name:
                    maxima_series_data.append(series_data[idx])
                    maxima_series_names.append(series_name)
                else:
                    regular_series_data.append(series_data[idx])
                    # Use the legend labels for display
                    if series_name in LEGEND_LABELS:
                        regular_series_names.append(LEGEND_LABELS[series_name])
                    else:
                        regular_series_names.append(series_name)

            # Create a MultiLinePlot for the regular series
            args = MultiLinePlotArguments(
                series=regular_series_names,
                x_values=x_values,
                data=regular_series_data,
                x_axis_name="# Tokens",
                x_axis_extras={"labelpad": 1.0},
                x_axis_fontsize=9,
                x_ticks_fontsize=9,
                x_ticks_extras={"pad": 1.50},
                y_axis_name=(
                    "Normalized Throughput" if i == 0 else None
                ),  # Y-axis title only for the leftmost subplot
                y_axis_fontsize=10,
                y_ticks_fontsize=9,
                palette=PALETTE,
                line_width=1.5,
                marker_size=4.0,
                interpolation="linear",
                show_grid=True,
                line_styles=["-"] * len(regular_series_names),
                markers=["o"] * len(regular_series_names),
                show_legend=False,  # We'll use a shared legend
                x_tick_values=[2048, 4096, 6144, 8192],
                x_limit=(1600, 8800),
            )

            # Create and render the regular series plot
            plot_instance = MultiLinePlot(args)
            plot_instance.render(axes[i])

            # For the last subplot, keep the plot instance for shared legend
            if i == len(hidden_dims) - 1:
                last_plot_instance = plot_instance

            # Now add the maxima points (if any) as separate series with special styling
            if maxima_series_data:
                for idx, (series_name, data) in enumerate(
                    zip(maxima_series_names, maxima_series_data, strict=False)
                ):
                    # Extract the original model name from the maxima series name
                    original_model = series_name.replace("-maxima", "")

                    # Find the index of this model in the original series list
                    for original_idx, original_series_name in enumerate(series_names):
                        if (
                            original_series_name == original_model
                            and "-maxima" not in original_series_name
                        ):
                            # Use the same color as the original series
                            color = PALETTE[original_idx % len(PALETTE)]
                            break
                    else:
                        # Fallback if not found
                        color = PALETTE[idx % len(PALETTE)]

                    # Get valid indices and values for the maxima points
                    valid_indices = [j for j, y in enumerate(data) if y is not None]
                    valid_x = [x_values[j] for j in valid_indices]
                    valid_y = [data[j] for j in valid_indices]

                    if valid_x:
                        # Plot the maxima points connected by a dashed line
                        axes[i].plot(
                            valid_x,
                            valid_y,
                            linestyle="--",  # Dashed line
                            linewidth=1.0,
                            marker="o",  # Diamond marker
                            markersize=1.5,
                            color=color,
                            markeredgecolor="black",
                            markeredgewidth=1.0,
                            zorder=5,  # Ensure it's on top
                        )

            # Add title at the bottom
            axes[i].text(
                0.5,
                -0.325,
                f"N=K={hidden_dim}",
                fontsize=11,
                transform=axes[i].transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Adjust the layout to make room for the bottom titles
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Add space at the bottom for titles

        # Add a shared legend at the top center if we have a plot instance
        if last_plot_instance:
            # Create dummy plot_instance with only regular series for the legend
            legend_args = MultiLinePlotArguments(
                series=regular_series_names,
                x_values=x_values,
                data=regular_series_data,
                palette=PALETTE,
                line_styles=["-"] * len(regular_series_names),
                markers=["o"] * len(regular_series_names),
                line_width=1.25,
                marker_size=3.0,
                x_ticks_fontsize=10,
            )
            legend_plot = MultiLinePlot(legend_args)
            # Render shared legend at the top center
            legend_plot.render_shared_legend(
                fig,
                bbox_to_anchor=(0.5, 1.1),  # Position at the top center
                ncol=len(regular_series_names),  # All series in one row
                fontsize=10,
                frameon=False,
            )

        # Save figure
        op_type_str = "fwd_bwd" if op_type == "fwd_bwd" else "fwd"
        output_pdf_path = (
            results_dir / f"evaluation-fig-5-kernel-perf-{op_type_str}.pdf"
        )
        output_png_path = (
            results_dir / f"evaluation-fig-5-kernel-perf-{op_type_str}.png"
        )

        plt.savefig(output_pdf_path, bbox_inches="tight")
        plt.savefig(output_png_path, bbox_inches="tight")

        logger.info(f"Saved figure to {output_pdf_path}")
        logger.info(f"Saved figure to {output_png_path}")


def print_milestone_improvements(
    throughput_data: dict[int, dict[str, dict[str, dict[str, float]]]],
    hidden_dims: list[int],
    model_types: dict[str, list[str]],
) -> None:
    """Print throughput improvements at milestone sequence lengths for each model compared to baseline."""
    # Define milestone token counts to report
    milestones = [4096, 8192]

    logger.info("\n=== Throughput Improvement Report ===")

    # Track improvement factors across all configurations for summary statistics
    improvement_stats = {
        "fwd_bwd": {model: [] for model in ["FlashLoRA", "MultiLoRA"]},
        "fwd": {model: [] for model in ["FlashLoRA", "MultiLoRA"]},
    }

    for op_type in ["fwd_bwd", "fwd"]:
        logger.info(f"\n## {op_type.upper()} Operation")

        for hidden_dim in hidden_dims:
            if hidden_dim not in throughput_data:
                continue

            logger.info(f"\nHidden Dimension: {hidden_dim}")
            logger.info(
                f"{'Tokens':<10} {'Model':<20} {'Throughput':<15} {'Improvement':<15}"
            )
            logger.info(f"{'-'*60}")

            for tokens in milestones:
                token_str = str(tokens)

                if (
                    token_str not in throughput_data[hidden_dim]
                    or op_type not in throughput_data[hidden_dim][token_str]
                ):
                    continue

                models_dict = throughput_data[hidden_dim][token_str][op_type]

                # Get baseline (Torch LoRA)
                baseline = models_dict.get("Torch LoRA")
                if not baseline:
                    continue

                logger.info(
                    f"{tokens:<10} {'Torch LoRA':<20} {baseline:.2f} tok/s {'1.00x':<15}"
                )

                # Print improvements for other models
                for model in model_types[op_type]:
                    if model != "Torch LoRA" and model in models_dict:
                        throughput = models_dict[model]
                        improvement = throughput / baseline
                        logger.info(
                            f"{'':<10} {model:<20} {throughput:.2f} tok/s {f'{improvement:.2f}x':<15}"
                        )

                        # Store improvement factor for stats
                        if model in improvement_stats[op_type]:
                            improvement_stats[op_type][model].append(improvement)

                logger.info(f"{'-'*60}")

    # Calculate and print summary statistics
    logger.info("\n=== Summary Statistics ===")

    for op_type in ["fwd_bwd", "fwd"]:
        logger.info(f"\n## {op_type.upper()} Operation")
        logger.info(
            f"{'Model':<20} {'Avg Speedup':<15} {'Max Speedup':<15} {'Sample Size':<15}"
        )
        logger.info(f"{'-'*65}")

        for model in sorted(improvement_stats[op_type].keys()):
            improvements = improvement_stats[op_type][model]
            if improvements:
                avg_speedup = sum(improvements) / len(improvements)
                max_speedup = max(improvements)
                logger.info(
                    f"{model:<20} {avg_speedup:.2f}x{'':<10} {max_speedup:.2f}x{'':<10} {len(improvements)}"
                )
            else:
                logger.info(f"{model:<20} No data{'':<10} No data{'':<10} 0")

    logger.info("\n=== End of Report ===\n")


def main():
    """Main function to generate the kernel performance plot."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths to the input CSV files
    single_lora_path = os.path.join(script_dir, SINGLE_LORA_CSV)
    multi_lora_path = os.path.join(script_dir, MULTI_LORA_CSV)

    # Read the data
    logger.info("Reading single LoRA data...")
    single_lora_data = read_lora_kernels_csv(single_lora_path)

    logger.info("Reading multi LoRA data...")
    multi_lora_data = read_multi_lora_csv(multi_lora_path)

    # Merge the data
    logger.info("Merging data...")
    merged_data = merge_data(single_lora_data, multi_lora_data)

    # Calculate throughput from execution times
    logger.info("Calculating throughput...")
    throughput_data = calculate_throughput(merged_data)

    # Print milestone improvements before normalization
    logger.info("Printing milestone improvements...")
    print_milestone_improvements(throughput_data, HIDDEN_DIMS, MODEL_TYPES)

    # Normalize throughput against Torch LoRA's first data point
    logger.info("Normalizing throughput...")
    normalized_data = normalize_throughput(throughput_data)

    # Prepare data for plotting
    logger.info("Preparing plot data...")
    plot_data = prepare_plot_data(normalized_data, HIDDEN_DIMS, MODEL_TYPES)

    # Create the figure
    logger.info("Creating figures...")
    create_figure(plot_data, HIDDEN_DIMS)

    logger.success("Done!")


if __name__ == "__main__":
    main()
