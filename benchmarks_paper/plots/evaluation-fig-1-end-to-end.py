# noqa: INP001
"""Draws a plot of the end-to-end evaluation results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from lorafusion.utils.grouped_bar_chart import (
    GroupedBarChart,
    GroupedBarChartArguments,
)

# Plot details:
# - Each row contains 3 / 4 columns (each representing a different model).
# - x_labels are datasets:
#   - XSUM
#   - CNN Daily
#   - WikiSum
#   - Mixed Heterogeneous
# - methods are: FSDP, Megatron-LM PP, mLoRA, LoRAFusion (ours).

# V1: I don't know what this is when I add the comments.
# palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# V2: The Gradient Green Blue.
# PALETTE = ["#f6f8d5", "#98d2c0", "#4f959d", "#205781"]

# V3: Grey and Green-Blue.
# PALETTE = [DEFAULT_PALETTE[0], DEFAULT_PALETTE[1], DEFAULT_PALETTE[2], "#31a871"]

# V4: Grey.
# PALETTE = [DEFAULT_PALETTE[0], DEFAULT_PALETTE[1], DEFAULT_PALETTE[2], "#6cccbc"]
# PALETTE = [DEFAULT_PALETTE[2], DEFAULT_PALETTE[1], DEFAULT_PALETTE[0], "#6cccbc"]
# PALETTE = ["#b0b0b0", "#d0d0d0", "#f0f0f0", "#6cccbc"]
PALETTE = ["#a0a0a0", "#c0c0c0", "#e0e0e0", "#55bdac"]


METHODS = ["Single-GPU / Megatron-LM-FSDP", "Megatron-LM-PP", "mLoRA", "LoRAFusion"]

# Global control flag for using demo data
USE_DEMO_DATA = False

# Demo data variables (original hardcoded data)
llama_8b_data_demo = [
    [12353.35, 14874.99],  # XSUM
    [12231.68, 14595.14],  # CNN Daily
    [9754.33, 13920.32],  # WikiSum
    [10743.86, 14403.5],  # Mixed
    [11270.81, 14157.83],  # Heterogeneous
]

qwen_32b_data_demo = [
    [3021.52, 2882.46, 4232.88, 4968.81],  # XSUM
    [3640.74, 3246.74, 4347.63, 4972.94],  # CNN Daily
    [3941.92, 3389.00, 4083.53, 4946.70],  # WikiSum
    [3543.74, 3122.09, 4099.92, 5120.85],  # Mixed
    [3536.98, 3160.07, 4020.31, 4955.23],  # Heterogeneous
]

llama_70b_data_demo = [
    [2696.72, 2336.50, 3901.68, 4892.90],  # XSUM
    [3475.51, 2567.77, 3989.42, 5182.99],  # CNN Daily
    [3124.19, 2985.42, 3868.17, 5073.56],  # WikiSum
    [3142.48, 2491.07, 3746.07, 5115.41],  # Mixed
    [3109.73, 2595.19, 3517.25, 5136.16],  # Heterogeneous
]


def load_data_from_json(json_path: str) -> list[list[list[list[float]]]]:
    """Load data from parsed results JSON file."""
    with Path(json_path).open() as f:
        data = json.load(f)

    # Convert to the expected format
    llama_8b_data = data["8B"]
    qwen_32b_data = data["32B"]
    llama_70b_data = data["70B"]

    return [[llama_8b_data, qwen_32b_data, llama_70b_data]]


def draw_figure(data_grid: list[list[list[list[float]]]]) -> None:
    """Draw a grouped bar chart from the evaluation results.

    Args:
        data_grid: 4D list containing the data for each model and bit width.
            The structure is:
            - First level: rows of subplots
            - Second level: columns of subplots
            - Third level: bit widths (4/8/16)
            - Fourth level: methods (PEFT, Megatron-LM, mLoRA, LoRAFusion)
    """
    # Create figure with subplots
    n_rows = len(data_grid)
    n_cols = len(data_grid[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.8), dpi=300)

    # Flatten axes if needed
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Model names for each column
    model_names = ["Llama-3.1-8B", "Qwen-2.5-32B", "Llama-3.1-70B"]
    dataset_names = ["XSUM", "CNNDM", "WikiSum", "Mixed", "Het"]
    y_max_tick_values = [15000, 5000, 5000]

    # Draw each subplot
    for i, row in enumerate(data_grid):
        for j, model_data in enumerate(row):
            ax = axes[i][j]

            # Methods in this figure
            num_methods = len(model_data[0])
            curr_methods = METHODS[: num_methods - 1]
            curr_methods.append(METHODS[-1])
            palette = PALETTE[: num_methods - 1]
            palette.append(PALETTE[-1])
            y_max_tick_value = y_max_tick_values[j]
            y_axis_name = "Throughput (K tokens/s)" if j == 0 else None

            def _bar_label_processor(x_label: str, method: str, value: float) -> str:
                x_label_index = dataset_names.index(x_label)
                baseline_value = model_data[x_label_index][0]  # noqa: B023
                speedup = value / baseline_value
                return f"{speedup:.2f}" + r"$\times$"

            # Create chart arguments
            chart_args = GroupedBarChartArguments(
                methods=curr_methods,
                x_labels=dataset_names,
                data=model_data,
                overwrite_num_methods=len(METHODS),
                show_bar_label=True,
                # bar_linewidth=0.75,
                bar_linewidth=0.0,
                bar_label_fontsize=9.5,
                bar_label_rotation=90,
                inter_group_margin=0.31,
                bar_label_processor=_bar_label_processor,
                bar_label_extras={"fontsize": 9},
                y_max_tick_value=y_max_tick_value,
                num_y_ticks=6,
                x_axis_name=model_names[j],
                x_ticks_fontsize=11,
                # Make axis slightly lower
                y_axis_name=y_axis_name,
                show_grid=True,
                show_legend=False,  # Disable individual legends
                palette=palette,
                y_ticks_formatter=lambda x: f"{int(x / 1000)}",
            )

            # Create and render the chart
            chart = GroupedBarChart(chart_args)
            chart.render(ax)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i == 0 and j == 1:
                chart.render_shared_legend(
                    fig,
                    bbox_to_anchor=(0.5, 1.1),
                    fontsize=12,
                    frameon=False,
                )

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_file = Path("results/evaluation-fig-1-end-to-end.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    logger.success(f"Figure saved to {output_file}")

    # Also save as PNG for easier viewing
    png_file = output_file.with_suffix(".png")
    plt.savefig(png_file, bbox_inches="tight", dpi=300)
    logger.success(f"Figure also saved to {png_file}")


if __name__ == "__main__":
    # Load data from JSON file or use demo data
    json_path = "parsed_results.json"

    if USE_DEMO_DATA:
        logger.info("Using demo data")
        data_grid = [[llama_8b_data_demo, qwen_32b_data_demo, llama_70b_data_demo]]
    elif Path(json_path).exists():
        logger.info(f"Loading data from {json_path}")
        data_grid = load_data_from_json(json_path)
    else:
        logger.error(f"JSON file {json_path} not found and USE_DEMO_DATA is False")
        logger.error("Please either:")
        logger.error("1. Set USE_DEMO_DATA = True to use demo data, or")
        logger.error("2. Run parse_main_results.py to generate parsed_results.json")
        raise FileNotFoundError(f"Data file {json_path} not found")

    draw_figure(data_grid)
