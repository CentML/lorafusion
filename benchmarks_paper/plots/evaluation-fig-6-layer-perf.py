"""Layer Performance Visualization."""

"""
This figure visualizes the layer benchmark results from 'layer_benchmark_results.csv'.

We show performance comparison between:
1. Vanilla
2. FusedLoRA 
3. FusedMultiLoRA

For the following models:
1. Llama-3.1-8B-Instruct
2. Qwen2.5-32B-Instruct
3. Llama-3.1-70B-Instruct

We use batch size as the x-axis and throughput (tokens/second) as the y-axis.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from loguru import logger

from lorafusion.utils.multi_line_plot import MultiLinePlot, MultiLinePlotArguments

# Constants for the figure
FIGURE_SIZE = (6.3, 2.1)
DPI = 300

# Input data path relative to this script
LAYER_BENCHMARK_CSV = "../layer-results/layer_benchmark_results.csv"

# Models to include in the plot
MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-32B-Instruct", "Llama-3.1-70B-Instruct"]
SIMPLIFIED_MODELS = ["Llama-3.1-8B", "Qwen2.5-32B", "Llama-3.1-70B"]

# Palette to match other evaluation figures
PALETTE = ["#a0a0a0", "#55bdac", "#e88d89"]

# Labels for plot legend
LEGEND_LABELS = {
    "Vanilla": "Torch LoRA",
    "FusedLoRA": "FusedLoRA",
    "FusedMultiLoRA": "FusedMultiLoRA",
}

# Modes to compare
MODES = ["Vanilla", "FusedLoRA", "FusedMultiLoRA"]

# Specific x-axis ticks to show
X_TICKS = [4, 8, 12, 16, 20]


def read_layer_benchmark_csv(file_path: str) -> dict[str, dict[str, dict[int, float]]]:
    """Read the layer benchmark results CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dict with structure: {model_name: {mode: {batch_size: throughput}}}
    """
    data: dict[str, dict[str, dict[int, float]]] = {}

    try:
        with open(file_path) as f:
            reader = csv.DictReader(f)

            for row in reader:
                if not row:  # Skip empty rows
                    continue

                mode = row["Mode"]
                model = row["Model"]
                batch_size = int(row["BatchSize"])
                throughput = float(row["ThroughputTokensPerSec"])

                # Initialize nested dictionaries if needed
                if model not in data:
                    data[model] = {}
                if mode not in data[model]:
                    data[model][mode] = {}

                # Store the throughput data
                data[model][mode][batch_size] = throughput

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return data


def normalize_throughput(
    throughput_data: dict[str, dict[str, dict[int, float]]],
) -> dict[str, dict[str, dict[int, float]]]:
    """Normalize throughput data relative to Vanilla mode's first data point.

    Args:
        throughput_data: Dict with structure {model_name: {mode: {batch_size: throughput}}}

    Returns:
        Dict with normalized throughput values
    """
    normalized_data: dict[str, dict[str, dict[int, float]]] = {}

    for model, modes in throughput_data.items():
        if model not in normalized_data:
            normalized_data[model] = {}

        # Find the baseline (first batch size in Vanilla mode)
        baseline = None
        if "Vanilla" in modes:
            batch_sizes = sorted(modes["Vanilla"].keys())
            if batch_sizes:
                baseline = modes["Vanilla"][batch_sizes[0]]

        # If no baseline found, use raw values
        if baseline is None or baseline == 0:
            normalized_data[model] = modes
            continue

        # Normalize all values against the baseline
        for mode, batch_sizes in modes.items():
            if mode not in normalized_data[model]:
                normalized_data[model][mode] = {}

            for batch_size, throughput in batch_sizes.items():
                normalized_value = throughput / baseline
                normalized_data[model][mode][batch_size] = normalized_value

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
    throughput_data: dict[str, dict[str, dict[int, float]]],
    models: list[str],
    modes: list[str],
) -> dict[str, dict[str, list]]:
    """Prepare data for plotting, organizing by model and mode.

    Args:
        throughput_data: Dict with throughput data
        models: List of model names to include
        modes: List of modes to include

    Returns:
        Dict with plot data organized for MultiLinePlot
    """
    plot_data: dict[str, dict[str, list]] = {}

    for model in models:
        if model not in throughput_data:
            logger.warning(f"No data for model {model}")
            continue

        # Get all batch sizes and sort them
        all_batch_sizes = set()
        for mode in modes:
            if mode in throughput_data[model]:
                all_batch_sizes.update(throughput_data[model][mode].keys())

        sorted_batch_sizes = sorted(all_batch_sizes)

        if not sorted_batch_sizes:
            logger.warning(f"No batch sizes for model {model}")
            continue

        # Initialize data structure for this model
        plot_data[model] = {"x_values": sorted_batch_sizes, "data": [], "series": []}

        # For each mode, collect the throughput values
        for mode in modes:
            if mode not in throughput_data[model]:
                logger.warning(f"No data for mode {mode} in model {model}")
                continue

            mode_data = []
            has_data = False

            for batch_size in sorted_batch_sizes:
                if batch_size in throughput_data[model][mode]:
                    mode_data.append(throughput_data[model][mode][batch_size])
                    has_data = True
                else:
                    mode_data.append(None)  # Missing data point

            if has_data:
                plot_data[model]["data"].append(mode_data)
                plot_data[model]["series"].append(mode)

        # Add local maxima points for FusedLoRA and FusedMultiLoRA
        for mode_idx, mode in enumerate(plot_data[model]["series"]):
            if mode == "FusedMultiLoRA" and model == "Llama-3.1-70B-Instruct":
                continue

            # Only add milestones for FusedLoRA and FusedMultiLoRA
            if mode in ["FusedLoRA", "FusedMultiLoRA"]:
                mode_data = plot_data[model]["data"][mode_idx]

                # Find local maxima in this mode's data
                local_maxima_indices = find_local_maxima(mode_data, sorted_batch_sizes)

                if local_maxima_indices:
                    # Create a new series just for the maxima points
                    maxima_data = [None] * len(sorted_batch_sizes)

                    # Mark only the local maxima points
                    for idx in local_maxima_indices:
                        maxima_data[idx] = mode_data[idx]

                    # Add this as a separate series
                    plot_data[model]["data"].append(maxima_data)
                    plot_data[model]["series"].append(f"{mode}-maxima")

    return plot_data


def create_figure(
    plot_data: dict[str, dict[str, list]], models: list[str], normalize: bool = False
) -> None:
    """Create the figure with subplots for each model.

    Args:
        plot_data: Dict with plot data
        models: List of model names
        normalize: Whether the data is normalized
    """
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Create a new figure
    fig, axes = plt.subplots(1, len(models), figsize=FIGURE_SIZE, dpi=DPI)

    # If there's only one subplot, make sure axes is a list
    if len(models) == 1:
        axes = [axes]

    last_plot_instance = None

    # Create each subplot
    for i, model in enumerate(models):
        if model not in plot_data:
            logger.warning(f"No plot data for model {model}")
            continue

        # Prepare data and series
        series_data = plot_data[model]["data"]
        series_names = plot_data[model]["series"]
        x_values = plot_data[model]["x_values"]

        if not series_names:
            logger.warning(f"No series for model {model}")
            continue

        # Separate regular series from maxima series
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
            x_axis_name="Batch Size",
            x_axis_extras={"labelpad": 1.0},
            x_axis_fontsize=9,
            x_ticks_fontsize=9,
            x_ticks_extras={"pad": 1.50},
            y_axis_name="Normalized Throughput" if i == 0 else None,
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
            x_tick_values=X_TICKS,
            x_limit=(3, 21),
            y_limit_padding=0.15,
        )

        # Create and render the plot
        plot_instance = MultiLinePlot(args)
        plot_instance.render(axes[i])

        # For the last subplot, keep the plot instance for shared legend
        if i == len(models) - 1:
            last_plot_instance = plot_instance

        # Now add the maxima points (if any) as separate series with special styling
        if maxima_series_data:
            for idx, (series_name, data) in enumerate(
                zip(maxima_series_names, maxima_series_data, strict=False)
            ):
                # Extract the original model name from the maxima series name
                original_mode = series_name.replace("-maxima", "")

                # Find the index of this model in the original series list
                for original_idx, original_series_name in enumerate(series_names):
                    if (
                        original_series_name == original_mode
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

        # Simplify model name for the title
        model_name = SIMPLIFIED_MODELS[i]

        # Add title at the bottom
        axes[i].text(
            0.5,
            -0.325,
            model_name,
            fontsize=11,
            transform=axes[i].transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Adjust the layout to make room for the bottom titles
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.2, top=0.85
    )  # Add space at the bottom for titles and top for main title

    # Add a shared legend at the top center if we have a plot instance
    if last_plot_instance:
        # Create dummy plot_instance for the legend
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
            bbox_to_anchor=(0.5, 1.03),  # Position at the top center
            ncol=len(regular_series_names),  # All series in one row
            fontsize=10,
            frameon=False,
        )

    # Save figure with correct suffixes for both results directories
    suffix = "-normalized" if normalize else ""

    # Save to the root results directory (as created by the script)
    root_results_dir = Path("results")
    root_results_dir.mkdir(exist_ok=True, parents=True)

    output_pdf_path = root_results_dir / f"evaluation-fig-6-layer-perf{suffix}.pdf"
    output_png_path = root_results_dir / f"evaluation-fig-6-layer-perf{suffix}.png"

    plt.savefig(output_pdf_path, bbox_inches="tight")
    plt.savefig(output_png_path, bbox_inches="tight")

    logger.info(f"Saved figure to {output_pdf_path}")
    logger.info(f"Saved figure to {output_png_path}")

    # Also save to the benchmarks_paper/plots/results directory for consistency
    plots_results_dir = Path("benchmarks_paper/plots/results")
    plots_results_dir.mkdir(exist_ok=True, parents=True)

    output_pdf_path = plots_results_dir / f"evaluation-fig-6-layer-perf{suffix}.pdf"
    output_png_path = plots_results_dir / f"evaluation-fig-6-layer-perf{suffix}.png"

    plt.savefig(output_pdf_path, bbox_inches="tight")
    plt.savefig(output_png_path, bbox_inches="tight")

    logger.info(f"Saved figure to {output_pdf_path}")
    logger.info(f"Saved figure to {output_png_path}")


def calculate_performance_improvements(
    normalized_data: dict[str, dict[str, dict[int, float]]],
    models: list[str],
    modes: list[str],
) -> None:
    """Calculate and log average and peak performance improvements for milestone points.
    
    Args:
        normalized_data: Dict with normalized throughput data
        models: List of model names to analyze
        modes: List of modes to compare
    """
    for model in models:
        if model not in normalized_data:
            logger.warning(f"No data for model {model} when calculating improvements")
            continue
            
        model_data = normalized_data[model]
        
        # Get vanilla data as baseline
        if "Vanilla" not in model_data:
            logger.warning(f"No Vanilla baseline for model {model}")
            continue
            
        vanilla_data = model_data["Vanilla"]
        
        # Calculate improvements for each optimized mode
        for mode in modes:
            if mode == "Vanilla" or mode not in model_data:
                continue
                
            mode_data = model_data[mode]
            
            # Find common batch sizes
            common_batch_sizes = set(vanilla_data.keys()) & set(mode_data.keys())
            if not common_batch_sizes:
                logger.warning(f"No common batch sizes for {model} between Vanilla and {mode}")
                continue
                
            # Calculate improvements
            improvements = []
            for batch_size in common_batch_sizes:
                vanilla_value = vanilla_data[batch_size]
                mode_value = mode_data[batch_size]
                
                if vanilla_value > 0:  # Avoid division by zero
                    improvement = mode_value / vanilla_value
                    improvements.append((batch_size, improvement))
            
            if not improvements:
                logger.warning(f"No valid improvements for {model} in {mode}")
                continue
                
            # Find peak improvement
            peak_batch_size, peak_improvement = max(improvements, key=lambda x: x[1])
            
            # Calculate average improvement
            avg_improvement = sum(imp for _, imp in improvements) / len(improvements)
            
            # Find local maxima (milestone points)
            improvements.sort(key=lambda x: x[0])  # Sort by batch size
            batch_sizes = [bs for bs, _ in improvements]
            improvement_values = [imp for _, imp in improvements]
            
            milestone_indices = []
            for i in range(1, len(improvements) - 1):
                if (improvement_values[i] > improvement_values[i-1] and 
                    improvement_values[i] > improvement_values[i+1]):
                    milestone_indices.append(i)
                    
            # Add first point if it's a local maximum
            if len(improvements) >= 2 and improvement_values[0] > improvement_values[1]:
                milestone_indices.insert(0, 0)
                
            # Add last point if it's a local maximum
            if len(improvements) >= 2 and improvement_values[-1] > improvement_values[-2]:
                milestone_indices.append(len(improvements) - 1)
                
            # Log milestone improvements
            if milestone_indices:
                milestone_improvements = [improvements[i] for i in milestone_indices]
                
                logger.info(f"{model} - {mode} Performance Improvements:")
                logger.info(f"  Average: {avg_improvement:.2f}x")
                logger.info(f"  Peak: {peak_improvement:.2f}x at batch size {peak_batch_size}")
                
                logger.info(f"  Milestone points:")
                for batch_size, improvement in milestone_improvements:
                    logger.info(f"    Batch size {batch_size}: {improvement:.2f}x")
            else:
                logger.info(f"{model} - {mode} Performance Improvements:")
                logger.info(f"  Average: {avg_improvement:.2f}x")
                logger.info(f"  Peak: {peak_improvement:.2f}x at batch size {peak_batch_size}")
                logger.info(f"  No milestone points found")


def main():
    """Main function to generate the layer performance plot."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute path to the input CSV file
    layer_benchmark_path = os.path.join(script_dir, LAYER_BENCHMARK_CSV)

    # Read the data
    logger.info("Reading layer benchmark data...")
    layer_data = read_layer_benchmark_csv(layer_benchmark_path)

    # Prepare raw data for plotting
    logger.info("Preparing raw plot data...")
    raw_plot_data = prepare_plot_data(layer_data, MODELS, MODES)

    # Create the figure with raw data
    logger.info("Creating figure with raw throughput...")
    create_figure(raw_plot_data, MODELS, normalize=False)

    # Normalize throughput against Vanilla's first data point
    logger.info("Normalizing throughput...")
    normalized_data = normalize_throughput(layer_data)

    # Calculate and log performance improvements
    logger.info("Calculating performance improvements...")
    calculate_performance_improvements(normalized_data, MODELS, MODES)

    # Prepare normalized data for plotting
    logger.info("Preparing normalized plot data...")
    normalized_plot_data = prepare_plot_data(normalized_data, MODELS, MODES)

    # Create the figure with normalized data
    logger.info("Creating figure with normalized throughput...")
    create_figure(normalized_plot_data, MODELS, normalize=True)

    logger.success("Done!")


if __name__ == "__main__":
    main()
