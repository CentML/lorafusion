"""Utilities for drawing stacked bar charts for the benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

# Fix font type for PDF and PS files, which is required by the ACM/IEEE templates.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer

# Since we use [0, 1, 2, ...] as the x-axis positions, the margin between
# labels should be 1.0.
X_LABEL_MARGIN = 1.0

# Z-order constants for controlling element layering
ZORDER_GRID = 0
ZORDER_BARS = 2
ZORDER_AXIS = 3
ZORDER_LABELS = 4

# Default color palette matching the provided image
DEFAULT_PALETTE = [
    "#F0F0F0",  # very light gray
    "#C0C0C0",  # light gray
    "#909090",  # medium gray
    "#606060",  # dark gray
    "#000000",  # black
    "#990000",  # deep dark red
    "#CC0000",  # red
]


@dataclass
class StackedBarChartArguments:
    """Arguments for the stacked bar chart."""

    categories: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of categories or components that make up each stack. "
            "e.g. ['Processing', 'I/O', 'Compute']"
        },
    )
    x_labels: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of x-axis labels for the stacked bars. "
            "e.g. ['Model A', 'Model B', 'Model C']."
        },
    )
    data: list[list[float | int | str | None]] = field(
        default_factory=list,
        metadata={
            "help": "2D list of data to be plotted. The outer list represents x labels,"
            " and the inner list represents values for each category in the stack. "
            "The 2D list should have the size of (num_x_labels, num_categories). "
            "Data can be missing, represented as None or a string."
        },
    )
    orientation: Literal["vertical", "horizontal"] = field(
        default="vertical",
        metadata={
            "help": "Orientation of the bars. 'vertical' (default) for up-down bars, "
            "'horizontal' for left-right bars."
        },
    )
    palette: list[str] = field(
        default_factory=lambda: DEFAULT_PALETTE.copy(),
        metadata={"help": "List of colors for the stacked segments."},
    )
    hatches: list[str] | None = field(
        default=None,
        metadata={"help": "List of hatches for the stacked segments. Default is None."},
    )
    figure_margin: float = field(
        default=0.1,
        metadata={"help": "Margin of the figure. Default is 0.1 X_LABEL_MARGIN."},
    )
    figure_margin_left: float | None = field(
        default=None,
        metadata={
            "help": "Left/bottom margin of the figure. If None, uses figure_margin."
        },
    )
    figure_margin_right: float | None = field(
        default=None,
        metadata={
            "help": "Right/top margin of the figure. If None, uses figure_margin."
        },
    )
    bar_edgecolor: str = field(
        default="black",
        metadata={"help": "Edge color for the bars. Default is 'black'."},
    )
    bar_linewidth: float = field(
        default=0.5,
        metadata={"help": "Line width for the bar edges. Default is 0.5."},
    )
    bar_width: float = field(
        default=0.5,
        metadata={"help": "Width of each bar. Default is 0.5."},
    )
    bar_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra arguments for the bars. Default is an empty dictionary."
        },
    )
    show_bar_label: bool = field(
        default=False,
        metadata={"help": "Whether to show the bar labels. Default is False."},
    )
    bar_label_fontsize: float | None = field(
        default=8,
        metadata={"help": ("Font size for bar labels.")},
    )
    bar_label_rotation: int = field(
        default=0,
        metadata={"help": "Rotation for bar label text. Default is 0 (horizontal)."},
    )
    bar_label_type: Literal["center", "edge"] = field(
        default="center",
        metadata={"help": "Type of bar label position. Default is 'center'."},
    )
    bar_label_padding: float = field(
        default=0.0,
        metadata={"help": "Padding between bar and label in points. Default is 0.0."},
    )
    bar_label_formatter: Callable[[float], str] = field(
        default=lambda x: f"{x:.2f}",
        metadata={"help": ("Function to format the bar label values.")},
    )
    bar_label_processor: Callable[[str, str, Any], str] = field(
        default=lambda x_label, category, value: value,
        metadata={
            "help": (
                "Function to process the bar label. Default is a lambda function "
                "that returns the value as is."
            )
        },
    )
    bar_label_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Extra arguments for the bar labels."},
    )
    x_axis_name: str | None = field(
        default=None, metadata={"help": "Title of the x-axis. Default is None."}
    )
    x_axis_fontsize: float = field(
        default=14, metadata={"help": "Font size for the x-axis. Default is 14."}
    )
    x_axis_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra arguments for the x-axis. Default is an empty dictionary."
        },
    )
    x_ticks_fontsize: float = field(
        default=12, metadata={"help": "Font size for the x-axis labels. Default is 12."}
    )
    x_ticks_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra arguments for the x-axis labels. Default is an empty dictionary."
            )
        },
    )
    y_axis_name: str | None = field(
        default=None, metadata={"help": "Title of the y-axis. Default is None."}
    )
    y_axis_fontsize: float = field(
        default=14, metadata={"help": "Font size for the y-axis. Default is 14."}
    )
    y_axis_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": ("Extra arguments for the y-axis. Default is an empty dictionary.")
        },
    )
    num_y_ticks: int = field(
        default=6, metadata={"help": "Number of y-axis ticks. Default is 6."}
    )
    y_ticks_formatter: Callable[[float], str] = field(
        default=lambda x: f"{x:.2f}",
        metadata={
            "help": "Formatter for the y-axis ticks. Default is a lambda function that "
            "returns the value as is."
        },
    )
    y_ticks_fontsize: float = field(
        default=12, metadata={"help": "Font size for the y-axis labels. Default is 12."}
    )
    y_ticks_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra arguments for the y-axis labels. Default is an empty dictionary."
            )
        },
    )
    y_max_tick_value: float | None = field(
        default=None,
        metadata={
            "help": (
                "Upper limit of the y-axis. If None, the upper limit will be set to"
                " the maximum value in the data. Note: the difference between "
                "y_max_tick_value and y_limit is that y_max_tick_value is the maximum "
                "value that will be displayed on the y-axis, while y_limit is the "
                "upper bound of the whole y-axis."
            )
        },
    )
    y_limit: float | None = field(
        default=None,
        metadata={
            "help": (
                "Upper limit of the y-axis. If None, the upper limit will be set to "
                "y_limit_ratio times the maximum value in the data."
            )
        },
    )
    y_limit_ratio: float | None = field(
        default=None,
        metadata={
            "help": (
                "Ratio of the upper limit of the y-axis to the maximum value in the "
                "data. Default is "
                "y_max_tick_value / (num_y_ticks - 1) * (num_y_ticks - 0.01)."
            )
        },
    )
    show_legend: bool = field(
        default=True,
        metadata={"help": "Whether to show the legend. Default is True."},
    )
    legend_fontsize: float = field(
        default=12,
        metadata={"help": "Font size for the legend. Default is 12."},
    )
    legend_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra arguments for the legend. Default is an empty dictionary."
        },
    )
    show_grid: bool = field(
        default=True,
        metadata={"help": "Whether to show the grid. Default is True."},
    )
    grid_extras: dict[str, Any] = field(
        default_factory=lambda: {
            "axis": "y",
            "linestyle": "--",
            "alpha": 0.3,
            "color": "black",
            "linewidth": 0.5,
            "zorder": ZORDER_GRID,  # This ensures grid is behind the bars
        },
        metadata={
            "help": "Extra arguments for the grid. Default is an empty dictionary."
        },
    )

    def _validate_data_exists(self) -> None:
        """Check if data exists."""
        if self.data is None:
            msg = "No data to render"
            raise ValueError(msg)

    def _validate_list_data(self, len_categories: int, len_x_labels: int) -> None:
        """Validate data when it's a list."""
        if len(self.data) != len_x_labels:
            msg = (
                f"The number of x_labels must match the number of rows in the data. "
                f"{len_x_labels} != {len(self.data)}"
            )
            raise ValueError(msg)

        for row in self.data:
            if not isinstance(row, list):
                msg = f"Each row in the data must be a list. {type(row)} is not a list."
                raise TypeError(msg)

            if len(row) != len_categories:
                msg = (
                    f"The number of categories must match the number of columns in the "
                    f"data. {len_categories} != {len(row)}"
                )
                raise ValueError(msg)

    def _validate_numpy_data(self, len_categories: int, len_x_labels: int) -> None:
        """Validate data when it's a numpy array."""
        if self.data.shape[0] != len_x_labels:
            msg = (
                f"The number of x_labels must match the number of rows in the data. "
                f"{len_x_labels} != {self.data.shape[0]}"
            )
            raise ValueError(msg)

        if self.data.shape[1] != len_categories:
            msg = (
                f"The number of categories must match the number of columns in the "
                f"data. {len_categories} != {self.data.shape[1]}"
            )
            raise ValueError(msg)

    def _validate_has_numeric_data(self) -> None:
        """Check if data contains valid numeric values."""
        flat_data = [
            x
            for row in self.data
            for x in row
            if x is not None and isinstance(x, int | float)
        ]

        if not flat_data:
            msg = "No valid numeric data to set y-axis limit"
            raise ValueError(msg)

    def validate(self) -> None:
        """Validate the arguments."""
        len_categories = len(self.categories)
        len_x_labels = len(self.x_labels)

        # Check if data exists
        self._validate_data_exists()

        # Validate data structure based on type
        if isinstance(self.data, list):
            self._validate_list_data(len_categories, len_x_labels)
        elif isinstance(self.data, np.ndarray):
            self._validate_numpy_data(len_categories, len_x_labels)
        else:
            msg = (
                f"The data must be a list or a numpy array. "
                f"{type(self.data)} is not a list or a numpy array."
            )
            raise TypeError(msg)

        # Check if data contains valid numeric values
        self._validate_has_numeric_data()


class StackedBarChart:
    """Class for rendering stacked bar charts."""

    def __init__(self, args: StackedBarChartArguments) -> None:
        """Initialize the stacked bar chart.

        Args:
            args: Arguments for the stacked bar chart.
        """
        self.args = args

    def _setup_axes_styling(self, axes: Axes) -> None:
        """Set up axis labels, ticks, and other styling.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args
        is_horizontal = args.orientation == "horizontal"

        # Apply styling
        self._setup_axis_labels(axes, is_horizontal=is_horizontal)
        self._setup_ticks_and_limits(axes, is_horizontal=is_horizontal)
        self._setup_grid(axes, is_horizontal=is_horizontal)
        self._setup_legend(axes)

    def _setup_axis_labels(self, axes: Axes, *, is_horizontal: bool) -> None:
        """Set up axis labels based on orientation.

        Args:
            axes: Matplotlib axes to render on
            is_horizontal: Whether the chart is horizontal
        """
        args = self.args

        # Prepare styling kwargs
        x_axis_kwargs = {"fontsize": args.x_axis_fontsize}
        y_axis_kwargs = {"fontsize": args.y_axis_fontsize}

        # Add custom parameters
        x_axis_kwargs.update(args.x_axis_extras)
        y_axis_kwargs.update(args.y_axis_extras)

        if is_horizontal:
            # Swap x and y for horizontal orientation
            axes.set_ylabel(args.x_axis_name, **y_axis_kwargs)
            axes.set_xlabel(args.y_axis_name, **x_axis_kwargs)
        else:
            axes.set_xlabel(args.x_axis_name, **x_axis_kwargs)
            axes.set_ylabel(args.y_axis_name, **y_axis_kwargs)

        # Set the zorder for all spine elements (the box around the plot)
        for spine in axes.spines.values():
            spine.set_zorder(ZORDER_AXIS)

    def _calculate_max_tick_value(self) -> float:
        """Calculate the maximum tick value based on data.

        Returns:
            float: Maximum tick value
        """
        args = self.args

        if args.y_max_tick_value is not None:
            return args.y_max_tick_value

        # For stacked bars, we need the sum of each stack
        stack_sums = []
        for row in args.data:
            valid_values = [
                x for x in row if x is not None and isinstance(x, int | float)
            ]
            if valid_values:
                stack_sums.append(sum(valid_values))

        return max(stack_sums) if stack_sums else 1.0

    def _setup_ticks_and_limits(self, axes: Axes, *, is_horizontal: bool) -> None:
        """Set up ticks and axis limits based on orientation.

        Args:
            axes: Matplotlib axes to render on
            is_horizontal: Whether the chart is horizontal
        """
        args = self.args
        max_tick_value = self._calculate_max_tick_value()

        # Prepare styling kwargs
        x_ticks_kwargs = {"fontsize": args.x_ticks_fontsize, "zorder": ZORDER_AXIS}
        y_ticks_kwargs = {"fontsize": args.y_ticks_fontsize, "zorder": ZORDER_AXIS}

        # Add custom parameters
        x_ticks_kwargs.update(args.x_ticks_extras)
        y_ticks_kwargs.update(args.y_ticks_extras)

        # Calculate positions and margins
        positions = list(range(len(args.x_labels)))

        # Determine left/bottom and right/top margins
        left_margin = (
            args.figure_margin_left
            if args.figure_margin_left is not None
            else args.figure_margin
        )
        right_margin = (
            args.figure_margin_right
            if args.figure_margin_right is not None
            else args.figure_margin
        )

        min_pos = min(positions) - X_LABEL_MARGIN / 2
        max_pos = max(positions) + X_LABEL_MARGIN / 2

        # Calculate value axis limit
        if args.y_limit is not None:
            value_limit = args.y_limit
        elif args.y_limit_ratio is not None:
            value_limit = max_tick_value * (args.y_limit_ratio + 1)
        else:
            value_limit = (
                max_tick_value / (args.num_y_ticks - 1) * (args.num_y_ticks - 0.01)
            )

        # Generate value ticks
        value_ticks = np.linspace(0, max_tick_value, args.num_y_ticks)

        if is_horizontal:
            # Set up horizontal orientation
            axes.set_yticks(positions)
            axes.set_yticklabels(args.x_labels, **y_ticks_kwargs)
            axes.set_ylim(min_pos - left_margin, max_pos + right_margin)

            axes.set_xticks(value_ticks)
            axes.set_xticklabels(
                [args.y_ticks_formatter(x) for x in value_ticks], **x_ticks_kwargs
            )
            axes.set_xlim(0, value_limit)
        else:
            # Set up vertical orientation
            axes.set_xticks(positions)
            axes.set_xticklabels(args.x_labels, **x_ticks_kwargs)
            axes.set_xlim(min_pos - left_margin, max_pos + right_margin)

            axes.set_yticks(value_ticks)
            axes.set_yticklabels(
                [args.y_ticks_formatter(y) for y in value_ticks], **y_ticks_kwargs
            )
            axes.set_ylim(0, value_limit)

        # Set zorder for tick lines to ensure they appear above the grid
        for tick in axes.xaxis.get_major_ticks():
            tick.tick1line.set_zorder(ZORDER_AXIS)
            tick.tick2line.set_zorder(ZORDER_AXIS)

        for tick in axes.yaxis.get_major_ticks():
            tick.tick1line.set_zorder(ZORDER_AXIS)
            tick.tick2line.set_zorder(ZORDER_AXIS)

    def _setup_grid(self, axes: Axes, *, is_horizontal: bool) -> None:
        """Set up grid based on orientation.

        Args:
            axes: Matplotlib axes to render on
            is_horizontal: Whether the chart is horizontal
        """
        args = self.args

        if args.show_grid:
            grid_extras = args.grid_extras.copy()

            # Adjust grid axis based on orientation
            if is_horizontal and grid_extras.get("axis") == "y":
                grid_extras["axis"] = "x"
            elif not is_horizontal and grid_extras.get("axis") == "x":
                grid_extras["axis"] = "y"

            axes.grid(**grid_extras)

    def _setup_legend(self, axes: Axes) -> None:
        """Set up legend if enabled.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        if args.show_legend:
            legend_kwargs = {"fontsize": args.legend_fontsize}
            legend_kwargs.update(args.legend_extras)
            axes.legend(**legend_kwargs)

    def _render_bars(self, axes: Axes) -> None:
        """Render the stacked bars on the axes.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args
        is_horizontal = args.orientation == "horizontal"

        # Extract data and dimensions
        categories = args.categories
        x_labels = args.x_labels
        data = args.data
        bar_width = args.bar_width

        # Use default palette if none provided
        palette = args.palette if args.palette else DEFAULT_PALETTE
        hatches = args.hatches if args.hatches is not None else [None] * len(categories)

        # For each position, we'll stack multiple bar segments
        positions = np.arange(len(x_labels))
        left_or_bottom = np.zeros(len(x_labels))

        for i, category in enumerate(categories):
            # Extract values for this category across all positions
            values = []
            for j, _ in enumerate(x_labels):
                val = data[j][i]
                if val is not None and isinstance(val, int | float):
                    values.append(val)
                else:
                    values.append(0)  # Skip non-numeric values

            # Set up bar styling with edges and color
            bar_kwargs = {
                "edgecolor": args.bar_edgecolor,
                "linewidth": args.bar_linewidth,
                "color": palette[i % len(palette)],  # Apply color from palette
                "zorder": ZORDER_BARS,
            }
            bar_kwargs.update(args.bar_extras)

            # Plot this segment of the stack based on orientation
            if is_horizontal:
                container = axes.barh(
                    y=positions,
                    width=values,
                    height=bar_width,
                    left=left_or_bottom,
                    label=category,
                    hatch=hatches[i],
                    **bar_kwargs,
                )
            else:
                container = axes.bar(
                    x=positions,
                    height=values,
                    width=bar_width,
                    bottom=left_or_bottom,
                    label=category,
                    hatch=hatches[i],
                    **bar_kwargs,
                )

            # Add labels if enabled
            if args.show_bar_label:
                self._add_bar_labels(axes, container, values, category, x_labels)

            # Update the positions for the next stack segment
            left_or_bottom += np.array(values)

    def _add_bar_labels(
        self,
        axes: Axes,
        container: BarContainer,
        values: list,
        category: str,
        x_labels: list,
    ) -> None:
        """Add labels to the bars.

        Args:
            axes: Matplotlib axes to render on
            container: The container of bars to label
            values: The numeric values for the current category
            category: The current category name
            x_labels: List of axis labels
        """
        args = self.args
        is_horizontal = args.orientation == "horizontal"

        # Process the labels
        labels = []
        for i, value in enumerate(values):
            label = args.bar_label_processor(x_labels[i], category, value)

            # Format the label if it's a number
            if isinstance(label, int | float):
                label = args.bar_label_formatter(label)

            labels.append(label)

        label_kwargs = {
            "fontsize": args.bar_label_fontsize,
            "rotation": args.bar_label_rotation,
            "label_type": args.bar_label_type,
            "padding": args.bar_label_padding,
            "zorder": ZORDER_LABELS,
        }
        label_kwargs.update(args.bar_label_extras)

        # Apply the labels with appropriate orientation
        if is_horizontal:
            axes.bar_label(labels=labels, container=container, **label_kwargs)
        else:
            axes.bar_label(labels=labels, container=container, **label_kwargs)

    def render(self, axes: Axes) -> None:
        """Render the stacked bar chart on the given matplotlib axes.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        # Validate arguments
        args.validate()

        # Render the bars
        self._render_bars(axes)

        # Setup axes styling
        self._setup_axes_styling(axes)

        # Add figure margins for better spacing
        plt.tight_layout()

    @classmethod
    def create_benchmark_chart(
        cls,
        args: StackedBarChartArguments,
        *,
        title: str | None = None,
        title_fontsize: int = 16,
        title_extras: dict[str, Any] | None = None,
        figsize: tuple = (10, 4),
        dpi: int = 300,
        output_path: str | None = None,
    ) -> tuple:
        """Create a standard benchmark comparison chart.

        Args:
            args: StackedBarChartArguments object containing chart configuration
            title: Optional title for the chart
            title_fontsize: Font size for the title
            title_extras: Additional keyword arguments for the title
            figsize: Figure size (width, height) in inches
            dpi: Resolution in dots per inch
            output_path: Optional path to save the chart

        Returns:
            tuple: (figure, axes, chart) objects for further customization

        Example:
            >>> args = StackedBarChartArguments(
            ...     categories=["Component1", "Component2", "Component3"],
            ...     x_labels=["Model A", "Model B", "Model C"],
            ...     data=[[0.3, 0.4, 0.3], [0.5, 0.2, 0.3], [0.2, 0.5, 0.3]],
            ...     show_bar_label=True,
            ... )
            >>> fig, ax, chart = StackedBarChart.create_benchmark_chart(
            ...     args,
            ...     title="Component Performance",
            ...     title_fontsize=16,
            ...     title_extras={"pad": 20},
            ...     figsize=(12, 5),
            ...     output_path="benchmark.png"
            ... )
        """
        # Create chart instance
        chart = cls(args)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Add title if provided
        if title:
            title_kwargs = {"fontsize": title_fontsize}
            if title_extras is not None:
                title_kwargs.update(title_extras)
            ax.set_title(title, **title_kwargs)

        # Render the chart
        chart.render(ax)

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")

        return fig, ax, chart


if __name__ == "__main__":
    from loguru import logger

    # ========================================================
    # Example 1: Basic vertical stacked bar chart
    logger.info("Example 1: Basic vertical stacked bar chart")

    categories = ["Process Time", "I/O Wait", "Compute"]
    x_labels = ["Model A", "Model B", "Model C", "Model D", "Model E"]

    # Example stack values: rows = x_labels, columns = stack categories
    data = np.array(
        [
            [0.30, 0.25, 0.45],  # Model A
            [0.40, 0.20, 0.40],  # Model B
            [0.25, 0.30, 0.45],  # Model C
            [0.35, 0.35, 0.30],  # Model D
            [0.20, 0.20, 0.60],  # Model E
        ]
    )

    args = StackedBarChartArguments(
        categories=categories,
        x_labels=x_labels,
        data=data,
        orientation="vertical",  # Default, can be omitted
        show_bar_label=True,
        bar_label_type="center",
        y_max_tick_value=1.00,
        x_axis_name="Model",
        y_axis_name="Time (normalized)",
        show_grid=True,
        show_legend=True,
        legend_extras={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.25),
            "ncol": len(categories),
        },
    )
    chart = StackedBarChart(args)

    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
    chart.render(ax1)
    plt.savefig("example1_vertical_stacked_bar_chart.png", bbox_inches="tight")

    # ========================================================
    # Example 2: Horizontal stacked bar chart
    logger.info("Example 2: Horizontal stacked bar chart")

    horizontal_args = StackedBarChartArguments(
        categories=categories,
        x_labels=x_labels,
        data=data,
        orientation="horizontal",  # Set horizontal orientation
        show_bar_label=True,
        bar_label_formatter=lambda x: f"{x * 100:.0f}%",
        y_max_tick_value=1.00,
        x_axis_name="Model",  # Will be on the y-axis for horizontal orientation
        y_axis_name="Time Distribution",  # Will be on the x-axis
        show_grid=True,
        show_legend=True,
        legend_extras={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.25),
            "ncol": len(categories),
            "frameon": False,
        },
    )

    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
    chart2 = StackedBarChart(horizontal_args)
    chart2.render(ax2)
    plt.savefig("example2_horizontal_stacked_bar_chart.png", bbox_inches="tight")

    # ========================================================
    # Example 3: Using the benchmark helper method with horizontal orientation
    logger.info(
        "Example 3: Using the benchmark helper method with horizontal orientation"
    )

    benchmark_args = StackedBarChartArguments(
        categories=categories,
        x_labels=x_labels,
        data=data,
        orientation="horizontal",
        show_bar_label=True,
        bar_label_formatter=lambda x: f"{x * 100:.0f}%",
        y_max_tick_value=1.00,
        y_limit_ratio=0.05,
        y_axis_name="Time",
        x_axis_name="Model",
        show_grid=True,
        show_legend=True,
        legend_extras={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.15),
            "ncol": len(categories),
            "frameon": False,
        },
    )

    fig3, ax3, chart3 = StackedBarChart.create_benchmark_chart(
        benchmark_args,
        figsize=(8, 5),
        dpi=300,
        output_path="example3_horizontal_benchmark_chart.png",
    )
