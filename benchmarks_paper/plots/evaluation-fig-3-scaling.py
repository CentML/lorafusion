"""Scaling of the different methods."""

from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from lorafusion.utils.grouped_bar_chart import (
    GroupedBarChart,
    GroupedBarChartArguments,
)

"""
In this figure, we want to show the scalability experiments.

We will only have a single subfigure (meaning a single plot).

We will have three x_labels (groups of bars):
- 4 GPUs
- 8 GPUs
- 16 GPUs

For each x_label, we will have eight bars:
- Megatron-LM-FSDP (DP Scaling)
- Megatron-LM-FSDP (Job Scaling)
- Megatron-LM-PP (DP Scaling)
- Megatron-LM-PP (Job Scaling)
- mLoRA (DP Scaling)
- mLoRA (Job Scaling)
- LoRAFusion (DP Scaling)
- LoRAFusion (Job Scaling)

1. We use the same color as in other evaluation figures.
2. We set DP scaling with hatch "//" and Job scaling without hatch.
"""

# Using the same palette as in other evaluation figures
PALETTE = ["#a0a0a0", "#c0c0c0", "#e0e0e0", "#55bdac"]

# Define methods for the experiment
METHODS = [
    "Megatron-LM-FSDP (DP Scaling)",
    "Megatron-LM-FSDP (Job Scaling)",
    "Megatron-LM-PP     (DP Scaling)",
    "Megatron-LM-PP     (Job Scaling)",
    "mLoRA        (DP Scaling)",
    "mLoRA        (Job Scaling)",
    "LoRAFusion (DP Scaling)",
    "LoRAFusion (Job Scaling)",
]

# Dataset: Llama 70B Mixed
# Format: [x_labels][methods]
scaling_data = [
    # 4 GPUs
    [3142.48, 3142.48, 2491.07, 2491.07, 3746.07, 3746.07, 5115.41, 5115.41],
    # 8 GPUs
    [5718.00, 6284.96, 4295.79, 5884.54, 6230.87, 7492.14, 9936.74, 10230.82],
    # 16 GPUs
    [10381.60, 12569.92, 8160.81, 11769.08, 13511.95, 14984.28, 18906.69, 20461.64],
]

# Create hatches for DP Scaling vs Job Scaling
HATCHES = ["////", "", "////", "", "////", "", "////", ""]


def draw_figure() -> None:
    """Draw a grouped bar chart showing the scaling experiments."""
    # Create figure with a single subplot
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=300)

    # X-labels for the subplot
    x_labels = ["4 GPUs", "8 GPUs", "16 GPUs"]

    # Set a reasonable y-max value
    y_max_tick_value = 24000

    # Create custom palette that repeats the colors for DP and Job scaling variants
    custom_palette = [
        PALETTE[0],
        PALETTE[0],  # Megatron-LM-FSDP
        PALETTE[1],
        PALETTE[1],  # Megatron-LM-PP
        PALETTE[2],
        PALETTE[2],  # mLoRA
        PALETTE[3],
        PALETTE[3],  # LoRAFusion (Our method)
    ]

    def _bar_label_processor(x_label: str, method: str, value: float) -> str:
        x_label_index = x_labels.index(x_label)
        # Get the corresponding method in 4 GPUs
        baseline_value = scaling_data[0][0]
        speedup = value / baseline_value
        return f"{speedup:.2f}" + r"$\times$"

    # Create chart arguments
    chart_args = GroupedBarChartArguments(
        methods=METHODS,
        x_labels=x_labels,
        data=scaling_data,
        show_bar_label=True,
        figure_margin=0.025,
        inter_group_margin=0.3,
        bar_linewidth=0.0,
        bar_label_fontsize=9.5,
        bar_label_rotation=90,
        bar_label_processor=_bar_label_processor,
        bar_label_extras={"fontsize": 9},
        y_max_tick_value=y_max_tick_value,
        num_y_ticks=5,
        y_limit_ratio=0.1,
        x_axis_name=None,
        x_ticks_fontsize=11,
        y_axis_name="Throughput (K tokens/s)",
        y_axis_fontsize=13,
        show_grid=True,
        show_legend=False,
        palette=custom_palette,
        y_ticks_formatter=lambda x: f"{int(x) // 1000}",
        hatches=HATCHES,
    )

    # Create and render the chart
    chart = GroupedBarChart(chart_args)

    # Override the rendering to add gaps between groups
    def custom_render(ax):
        # Get the original render method
        original_render = chart.render

        # Define our custom render that will adjust bar positions
        def modified_render(ax):
            # Call the original render method but with a small modification:
            # We store the original _add_all_bar_labels method
            original_add_all_bar_labels = chart._add_all_bar_labels

            # Replace it temporarily with a no-op to prevent label rendering
            chart._add_all_bar_labels = lambda ax: None

            # Call the original render which will now skip label rendering
            original_render(ax)

            # Restore the original method
            chart._add_all_bar_labels = original_add_all_bar_labels

            # Now adjust the bars
            containers = chart.containers

            for i, container in enumerate(containers):
                # Add extra spacing after every 2 bars (FSDP, PP, mLoRA, LoRAFusion groups)
                group_idx = i // 2

                # Loop through all bars and adjust their positions
                for bar in container:
                    # Get the current position and width
                    x, width = bar.get_x(), bar.get_width()

                    # Apply different offsets based on group position
                    if group_idx == 0:  # After FSDP group
                        x -= 0.03
                    elif group_idx == 1:  # After PP group
                        x -= 0.01  # Larger gap after PP
                    elif group_idx == 2:  # After mLoRA group
                        x += 0.01
                    elif group_idx == 3:  # After LoRAFusion group
                        x += 0.03

                    # Set the new position
                    bar.set_x(x)

            # Now that all bars are positioned correctly, add the labels
            chart._add_all_bar_labels(ax)

        # Call our modified render function
        modified_render(ax)

        return ax

    # Use our custom render method
    custom_render(ax)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Render legend with better formatting
    chart.render_shared_legend(
        fig,
        bbox_to_anchor=(0.465, 1.0),
        fontsize=9,
        frameon=True,
        handlelength=1.5,
        ncol=2,
        facecolor="white",  # Keep white background
        edgecolor="white",  # Remove the edge/frame
        framealpha=0.8,
    )

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_file = Path("results/evaluation-fig-3-scaling.pdf")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    logger.success(f"Figure saved to {output_file}")

    # Also save as PNG for easier viewing
    png_file = output_file.with_suffix(".png")
    plt.savefig(png_file, bbox_inches="tight", dpi=300)
    logger.success(f"Figure also saved to {png_file}")


if __name__ == "__main__":
    draw_figure()
