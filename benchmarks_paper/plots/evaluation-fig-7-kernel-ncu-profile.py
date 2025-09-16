"""Layer Performance Visualization."""

"""
This figure visualizes the Total DRAM Access Volume (Read + Write).

It should looks like figure 2 in the evaluation section of the paper.

We have 3 groups of bars:
1. 8192x4096x4096
2. 8192x5120x5120
3. 8192x8192x8192

Each group has 3 bars:
1. Torch LoRA
2. FusedLoRA
3. FusedMultiLoRA

We normalize the label values with the Torch LoRA baseline, but show actual raw values on the y-axis.

Data (Total DRAM access in GiB):
    ```
    Torch LoRA	    2.008	2.771	5.289
    Fused LoRA	    1.258	1.836	4.044
    Fused MultiLoRA	1.262	1.838	4.054
    ```
"""

import csv
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from loguru import logger

from lorafusion.utils.grouped_bar_chart import (
    GroupedBarChart,
    GroupedBarChartArguments,
)

# Define methods and matrix sizes
METHODS = ["Torch LoRA", "Fused LoRA", "Fused MultiLoRA"]
MATRIX_SIZES = ["8192x4096x4096", "8192x5120x5120", "8192x8192x8192"]

# Global control flag for using demo data
USE_DEMO_DATA = False

# Demo data variables (original hardcoded data)
data_demo = [
    [2.008, 1.258, 1.262],  # 8192x4096x4096
    [2.771, 1.836, 1.838],  # 8192x5120x5120
    [5.289, 4.044, 4.054],  # 8192x8192x8192
]

# Custom color palette
PALETTE = ["#a0a0a0", "#55bdac", "#e88d89"]


def load_data_from_ncu_reports() -> list[list[float]]:
    """Load data from NCU reports summary.csv files.

    Returns:
        List of lists containing Total DRAM access data for each matrix size.
        Each inner list contains [Torch LoRA, Fused LoRA, Fused MultiLoRA] values.
    """
    # Map CSV file names to method indices
    file_to_method = {
        "raw_lora_ncu.csv": 0,  # Torch LoRA
        "fused_linear_lora_ncu.csv": 1,  # Fused LoRA
        "fused_linear_multi_lora_ncu.csv": 2,  # Fused MultiLoRA
    }

    # Directory names corresponding to matrix sizes
    dir_names = ["dim_4096x4096", "dim_5120x5120", "dim_8192x8192"]

    data = []

    for dir_name in dir_names:
        summary_path = Path("ncu_reports") / dir_name / "summary.csv"

        if not summary_path.exists():
            logger.error(f"Summary file not found: {summary_path}")
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

        # Initialize row data with zeros
        row_data = [0.0, 0.0, 0.0]

        # Read the summary CSV file
        with summary_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row["file"]
                if file_name in file_to_method:
                    method_idx = file_to_method[file_name]
                    total_gib = float(row["Total [GiB]"])
                    row_data[method_idx] = total_gib
                    logger.debug(
                        f"Loaded {file_name}: {total_gib} GiB for method {method_idx}"
                    )

        data.append(row_data)
        logger.info(f"Loaded data for {dir_name}: {row_data}")

    return data


def draw_figure(data: list[list[float]]) -> None:
    """Draw a bar chart showing the DRAM access volume for different LoRA implementations."""
    # Create figure with one subplot
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)

    # Calculate normalized data for labels only
    normalized_data = []
    for size_data in data:
        baseline = size_data[0]
        normalized_data.append([val / baseline for val in size_data])

    def _bar_label_processor(
        x_label: str, method: str, value: float, norm_value: float
    ) -> str:
        # Show normalized values as multiplication factors (e.g., 0.63x)
        return f"{norm_value:.2f}" + r"$\times$"

    # Create chart arguments
    chart_args = GroupedBarChartArguments(
        methods=METHODS,
        x_labels=MATRIX_SIZES,
        data=data,  # Use raw data for the bars
        palette=PALETTE,
        bar_linewidth=0.0,
        show_bar_label=True,
        bar_label_fontsize=9,
        bar_label_processor=lambda x_label, method, value: _bar_label_processor(
            x_label,
            method,
            value,
            normalized_data[MATRIX_SIZES.index(x_label)][METHODS.index(method)],
        ),
        bar_label_extras={"fontsize": 9},
        inter_group_margin=0.3,
        intra_group_margin=0.02,
        y_max_tick_value=6.0,  # Use calculated max value
        num_y_ticks=5,
        x_axis_name=r"M$\times$K$\times$N",
        x_axis_fontsize=12,
        x_ticks_fontsize=10,
        y_axis_name="DRAM Read/Write (GB)",
        y_axis_fontsize=11,
        y_ticks_fontsize=10,
        show_grid=True,
        show_legend=False,  # Explicitly disable the internal legend
    )

    # Create the chart
    chart = GroupedBarChart(chart_args)

    # Render the chart
    chart.render(ax)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Create custom legend patches
    legend_patches = []
    for i, method in enumerate(METHODS):
        color = PALETTE[i]
        patch = mpatches.Patch(color=color, label=method)
        legend_patches.append(patch)

    # Add legend with custom patches
    plt.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(METHODS),
        frameon=False,
        fontsize=10,
    )

    # Adjust layout
    plt.tight_layout()

    # Save the figure to both locations
    plot_dirs = [Path("ncu_reports"), Path("results")]

    for plot_dir in plot_dirs:
        # Create directory if it doesn't exist
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Save PDF
        output_file = plot_dir / "evaluation-fig-7-kernel-ncu-profile.pdf"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        logger.success(f"Figure saved to {output_file}")

        # Save PNG
        png_file = plot_dir / "evaluation-fig-7-kernel-ncu-profile.png"
        plt.savefig(png_file, bbox_inches="tight", dpi=300)
        logger.success(f"Figure also saved to {png_file}")


if __name__ == "__main__":
    # Load data from NCU reports or use demo data
    if USE_DEMO_DATA:
        logger.info("Using demo data")
        data = data_demo
    else:
        logger.info("Loading data from NCU reports")
        try:
            data = load_data_from_ncu_reports()
        except FileNotFoundError as e:
            logger.error(f"Failed to load data from NCU reports: {e}")
            logger.error("Please either:")
            logger.error("1. Set USE_DEMO_DATA = True to use demo data, or")
            logger.error("2. Ensure NCU reports are available in ncu_reports/")
            logger.error(
                "Note: It is also possible that NVIDIA GPU Performance Counters "
                "permissions are not set correctly. Please follow the error messages "
                "to set the permissions correctly."
            )
            raise

    draw_figure(data)
