"""Utilities for drawing grouped bar charts for the benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

# Fix font type for PDF and PS files, which is required by the ACM/IEEE templates.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure

# Since we use [0, 1, 2, ...] as the x-axis positions, the margin between
# labels should be 1.0.
X_LABEL_MARGIN = 1.0

# Z-order constants for controlling element layering
ZORDER_GRID = 0
ZORDER_BARS = 3
ZORDER_AXIS = 5
ZORDER_LABELS = 7

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
class GroupedBarChartArguments:
    """Arguments for the grouped bar chart."""

    methods: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of methods or models whose performance is compared in "
            "adjacent bars. e.g. ['Megatron', 'DeepSpeed', 'Mist']"
        },
    )
    x_labels: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of x-axis labels for the grouped bars. e.g. ['A100', 'H100', "
            "'H200']. These are also referred to as groups."
        },
    )
    data: list[list[float | int | str | None]] = field(
        default_factory=list,
        metadata={
            "help": "2D list of data to be plotted. The outer list represents groups, "
            "and the inner list represents values for each method/model. The 2D list "
            "should have the size of (num_x_labels, num_methods). Data can be missing, "
            "represented as None or a string."
        },
    )
    palette: list[str] = field(
        default_factory=lambda: DEFAULT_PALETTE.copy(),
        metadata={"help": "List of colors for the bars."},
    )
    hatches: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of hatches for the bars. If not provided, no hatches will be "
            "used."
        },
    )
    figure_margin: float = field(
        default=0.1,
        metadata={"help": "Margin of the figure. Default is 0.1 X_LABEL_MARGIN."},
    )
    overwrite_num_methods: int | None = field(
        default=None,
        metadata={
            "help": "Overwrite the number of methods. Default is None, which means "
            "the number of methods is determined by the data. It is mainly used to "
            "control the bar width as it is calculated by the number of methods."
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
    bar_label_padding: float = field(
        default=1.0,
        metadata={"help": "Padding between bar and label in points. Default is 3.0."},
    )
    bar_label_formatter: Callable[[float], str] = field(
        default=lambda x: f"{x:.2f}",
        metadata={"help": ("Function to format the bar label values.")},
    )
    bar_label_processor: Callable[[str, str, Any], str] = field(
        default=lambda x_label, method, value: value,
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
    intra_group_margin: float = field(
        default=0.00,
        metadata={
            "help": "Margin between groups of bars. Default is 0.02 X_LABEL_MARGIN."
        },
    )
    inter_group_margin: float = field(
        default=0.2,
        metadata={
            "help": "Margin between groups of bars. Default is 0.2 X_LABEL_MARGIN."
        },
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
            "zorder": -1,  # Use a negative value to ensure grid is behind everything
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

    def _validate_list_data(self, len_methods: int, len_x_labels: int) -> None:
        """Validate data when it's a list."""
        if len(self.data) != len_x_labels:
            msg = (
                f"The number of groups must match the number of rows in the data. "
                f"{len_x_labels} != {len(self.data)}"
            )
            raise ValueError(msg)

        for row in self.data:
            if not isinstance(row, list):
                msg = f"Each row in the data must be a list. {type(row)} is not a list."
                raise TypeError(msg)

            if len(row) != len_methods:
                msg = (
                    f"The number of methods must match the number of columns in the "
                    f"data. {len_methods} != {len(row)}"
                )
                raise ValueError(msg)

    def _validate_numpy_data(self, len_methods: int, len_x_labels: int) -> None:
        """Validate data when it's a numpy array."""
        if self.data.shape[0] != len_x_labels:
            msg = (
                f"The number of groups must match the number of rows in the data. "
                f"{len_x_labels} != {self.data.shape[0]}"
            )
            raise ValueError(msg)

        if self.data.shape[1] != len_methods:
            msg = (
                f"The number of methods must match the number of columns in the data. "
                f"{len_methods} != {self.data.shape[1]}"
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
        len_methods = len(self.methods)
        len_x_labels = len(self.x_labels)

        # Check if data exists
        self._validate_data_exists()

        # Validate data structure based on type
        if isinstance(self.data, list):
            self._validate_list_data(len_methods, len_x_labels)
        elif isinstance(self.data, np.ndarray):
            self._validate_numpy_data(len_methods, len_x_labels)
        else:
            msg = (
                f"The data must be a list or a numpy array. "
                f"{type(self.data)} is not a list or a numpy array."
            )
            raise TypeError(msg)

        # Check if data contains valid numeric values
        self._validate_has_numeric_data()


class GroupedBarChart:
    """Class for rendering grouped bar charts."""

    def __init__(self, args: GroupedBarChartArguments) -> None:
        """Initialize the grouped bar chart.

        Args:
            args: Arguments for the grouped bar chart.
        """
        self.args = args
        self.handles = None  # Store handles for shared legend

    def _setup_bar_dimensions(self, num_methods: int) -> float:
        """Calculate bar width and positions based on chart parameters.

        Args:
            num_methods: Number of methods/series in the chart

        Returns:
            float: The calculated bar width
        """
        args = self.args

        # Calculate bar width by subtracting the margins from the total width
        # and dividing by the number of methods.
        return (
            (X_LABEL_MARGIN - args.inter_group_margin)
            - (num_methods - 1) * args.intra_group_margin
        ) / num_methods

    def _setup_axes_styling(self, axes: Axes) -> None:
        """Set up axis labels, ticks, and other styling.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        # Prepare styling kwargs
        x_axis_kwargs = {"fontsize": args.x_axis_fontsize}
        y_axis_kwargs = {"fontsize": args.y_axis_fontsize}
        x_ticks_kwargs = {"fontsize": args.x_ticks_fontsize, "zorder": ZORDER_AXIS}
        y_ticks_kwargs = {"fontsize": args.y_ticks_fontsize, "zorder": ZORDER_AXIS}

        # Add custom parameters
        x_axis_kwargs.update(args.x_axis_extras)
        y_axis_kwargs.update(args.y_axis_extras)
        x_ticks_kwargs.update(args.x_ticks_extras)
        y_ticks_kwargs.update(args.y_ticks_extras)

        # Calculate the max value for y-axis
        if args.y_max_tick_value is not None:
            y_max_tick_value = args.y_max_tick_value
        else:
            y_max_tick_value = max(
                x
                for row in args.data
                for x in row
                if x is not None and isinstance(x, int | float)
            )

        # Apply axis labels
        axes.set_xlabel(args.x_axis_name, **x_axis_kwargs)
        axes.set_ylabel(args.y_axis_name, **y_axis_kwargs)

        # Apply x-tick labels
        x_positions = list(range(len(args.x_labels)))
        axes.set_xticks(x_positions)
        axes.set_xticklabels(args.x_labels, **x_ticks_kwargs)

        # Set x-axis limits with margins
        margin = args.figure_margin
        x_min = min(x_positions) - X_LABEL_MARGIN / 2
        x_max = max(x_positions) + X_LABEL_MARGIN / 2
        axes.set_xlim(x_min - margin, x_max + margin)

        # Generate and set y-ticks
        y_ticks = np.linspace(0, y_max_tick_value, args.num_y_ticks)
        axes.set_yticks(y_ticks)
        axes.set_yticklabels(
            [args.y_ticks_formatter(y) for y in y_ticks], **y_ticks_kwargs
        )

        # Set y-axis limit
        if args.y_limit is None and args.y_limit_ratio is None:
            y_limit = (
                y_max_tick_value / (args.num_y_ticks - 1) * (args.num_y_ticks - 0.01)
            )
        elif args.y_limit is not None:
            y_limit = args.y_limit
        else:
            y_limit = y_max_tick_value * (1 + args.y_limit_ratio)
        axes.set_ylim(0, y_limit)

        # Set the zorder for all spine elements (the box around the plot)
        for spine in axes.spines.values():
            spine.set_zorder(ZORDER_AXIS)

        # Set zorder for tick lines to ensure they appear above the grid
        for tick in axes.xaxis.get_major_ticks():
            tick.tick1line.set_zorder(ZORDER_AXIS)
            tick.tick2line.set_zorder(ZORDER_AXIS)

        for tick in axes.yaxis.get_major_ticks():
            tick.tick1line.set_zorder(ZORDER_AXIS)
            tick.tick2line.set_zorder(ZORDER_AXIS)

        # Add grid if enabled
        if args.show_grid:
            axes.grid(**args.grid_extras)

        # Add legend if enabled
        if args.show_legend:
            legend_kwargs = {"fontsize": args.legend_fontsize}
            legend_kwargs.update(args.legend_extras)
            axes.legend(**legend_kwargs)

    def _render_bars(self, axes: Axes, bar_width: float) -> None:
        """Render the bars on the axes.

        Args:
            axes: Matplotlib axes to render on
            bar_width: Width of each bar
        """
        args = self.args

        # Extract data and dimensions
        methods = args.methods
        x_labels = args.x_labels
        data = args.data
        num_methods = len(methods)

        # Use default palette if none provided
        palette = args.palette if args.palette else DEFAULT_PALETTE

        # Get hatches list (if provided)
        hatches = args.hatches

        # Store handles and labels for shared legend
        self.handles = []

        # Store containers and labels for later use
        self.containers = []
        self.all_y_labels = []

        # Plot each method's data as a series of bars
        for j, method in enumerate(methods):
            x_values, y_values, y_labels = self._prepare_series_data(
                j, method, x_labels, data, bar_width, num_methods
            )

            # Set up bar styling with edges and color
            bar_kwargs = {
                "edgecolor": args.bar_edgecolor,
                "linewidth": args.bar_linewidth,
                "color": palette[j % len(palette)],  # Apply color from palette
                "zorder": ZORDER_BARS,  # Ensure bars are on top of grid
                "clip_on": False,  # Prevent bars from being clipped
            }

            # Apply hatch if available
            if hatches and len(hatches) > 0:
                bar_kwargs["hatch"] = hatches[j % len(hatches)]

            bar_kwargs.update(args.bar_extras)

            # Plot the bars
            container = axes.bar(
                x=x_values,
                height=y_values,
                width=bar_width,
                align="edge",
                label=method,
                **bar_kwargs,
            )

            # Store handle for shared legend
            self.handles.append(container)

            # Store container and labels for later
            self.containers.append(container)
            self.all_y_labels.append(y_labels)

    def _add_all_bar_labels(self, axes: Axes) -> None:
        """Add labels to all bars after any potential bar adjustments.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        if not args.show_bar_label:
            return

        for container, y_labels in zip(
            self.containers, self.all_y_labels, strict=False
        ):
            self._add_bar_labels(axes, container, y_labels)

    def _prepare_series_data(
        self,
        series_index: int,
        method: str,
        x_labels: list,
        data: list,
        bar_width: float,
        num_methods: int,
    ) -> tuple:
        """Prepare the data for a single series (method).

        Args:
            series_index: Index of the current series
            method: Name of the current method/series
            x_labels: List of x-axis labels
            data: The data to be plotted
            bar_width: Width of each bar
            num_methods: Total number of methods/series

        Returns:
            tuple: (x_values, y_values, y_labels) for the series
        """
        args = self.args

        x_values = []
        y_values = []
        y_labels = []

        for i, x_label in enumerate(x_labels):
            value = data[i][series_index]
            if value is None:
                continue

            # Calculate the x-position for this bar
            x = (
                i
                - (
                    num_methods * bar_width
                    + (num_methods - 1) * args.intra_group_margin
                )
                / 2
            )
            x += series_index * (bar_width + args.intra_group_margin)

            # Process the label
            y_label = args.bar_label_processor(x_label, method, value)

            # Format the label if it's a number
            if isinstance(y_label, int | float):
                y_label = args.bar_label_formatter(y_label)

            # Add to collection
            x_values.append(x)
            y_values.append(value)
            y_labels.append(y_label)

        return x_values, y_values, y_labels

    def _add_bar_labels(
        self, axes: Axes, container: BarContainer, y_labels: list
    ) -> None:
        """Add labels to the bars.

        Args:
            axes: Matplotlib axes to render on
            container: The container of bars to label
            y_labels: The labels to add
        """
        args = self.args

        label_kwargs = {
            "fontsize": args.bar_label_fontsize,
            "rotation": args.bar_label_rotation,
            "padding": args.bar_label_padding,
            "zorder": ZORDER_LABELS,
        }
        label_kwargs.update(args.bar_label_extras)

        # Apply the labels
        axes.bar_label(labels=y_labels, container=container, **label_kwargs)

    def render(self, axes: Axes) -> None:
        """Render the grouped bar chart on the given matplotlib axes.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        # Clear the axes to start fresh
        axes.clear()

        # Validate arguments
        args.validate()

        # Calculate bar dimensions
        bar_width = self._setup_bar_dimensions(
            len(args.methods)
            if args.overwrite_num_methods is None
            else args.overwrite_num_methods
        )

        # Setup axes styling (including grid) first
        self._setup_axes_styling(axes)

        # Render the bars after the grid
        self._render_bars(axes, bar_width)

        # Add labels to all bars after any custom positioning
        self._add_all_bar_labels(axes)

        # Add figure margins for better spacing
        plt.tight_layout()

    @classmethod
    def create_benchmark_chart(
        cls,
        args: GroupedBarChartArguments,
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
            args: GroupedBarChartArguments object containing chart configuration
            title: Optional title for the chart
            title_fontsize: Font size for the title
            title_extras: Additional keyword arguments for the title
            figsize: Figure size (width, height) in inches
            dpi: Resolution in dots per inch
            output_path: Optional path to save the chart

        Returns:
            tuple: (figure, axes, chart) objects for further customization

        Example:
            >>> args = GroupedBarChartArguments(
            ...     methods=["Method1", "Method2"],
            ...     x_labels=["A", "B", "C"],
            ...     data=[[1.0, 1.2], [1.1, 0.9], [0.9, 1.1]],
            ...     show_bar_label=True,
            ... )
            >>> fig, ax, chart = GroupedBarChart.create_benchmark_chart(
            ...     args,
            ...     title="My Benchmark",
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

    def render_shared_legend(
        self,
        fig: Figure,
        handles: list[Any] | None = None,
        labels: list[str] | None = None,
        *,
        fontsize: float | None = None,
        loc: str = "upper center",
        bbox_to_anchor: tuple[float, float] = (0.5, 1.1),
        ncol: int | None = None,
        frameon: bool = False,
        **kwargs,
    ) -> None:
        """Render a shared legend for multiple grouped bar charts.

        Args:
            fig: The figure to add the legend to
            handles: List of legend handles (bar containers)
            labels: List of legend labels (defaults to self._labels if None)
            fontsize: Font size for the legend (defaults to self.args.legend_fontsize)
            loc: Location of the legend
            bbox_to_anchor: Position of the legend relative to the figure
            ncol: Number of columns in the legend
            frameon: Whether to draw a frame around the legend
            **kwargs: Additional keyword arguments for the legend
        """
        # Use instance values as defaults
        handles = handles if handles is not None else self.handles
        labels = labels if labels is not None else self.args.methods

        # Use arguments from the instance if not provided
        fontsize = self.args.legend_fontsize if fontsize is None else fontsize
        ncol = len(labels) if ncol is None else ncol

        legend_kwargs = {
            "handles": handles,
            "labels": labels,
            "fontsize": fontsize,
            "loc": loc,
            "bbox_to_anchor": bbox_to_anchor,
            "ncol": ncol,
            "frameon": frameon,
        }
        legend_kwargs.update(kwargs)

        fig.legend(**legend_kwargs)


if __name__ == "__main__":
    from loguru import logger

    # ========================================================
    # Example 1: Standard approach with arguments
    logger.info("Example 1: Standard approach with arguments")
    methods = ["MInference", "DuoAttention", "QServe", "vLLM", "LServe (Ours)"]
    x_labels = ["64K", "96K", "128K", "160K", "192K", "224K", "256K", "320K", "Geomean"]

    # Example bar values: rows = x_labels, columns = methods
    data = np.array(
        [
            [0.48, 0.92, 1.05, 1.00, 1.00],
            [0.35, 0.81, 1.05, 1.00, 1.00],
            [0.28, 0.78, 1.02, 1.00, 1.00],
            [0.25, 0.77, 1.00, 1.00, 1.00],
            [0.20, 0.75, 0.98, 1.00, 1.00],
            [0.18, 0.73, 0.97, 1.00, 1.00],
            [0.15, 0.70, 0.95, 1.00, 1.00],
            [0.12, 0.68, 0.93, 1.00, 1.00],
            [0.32, 0.75, 0.90, 1.00, 1.00],  # Geomean
        ]
    )

    args = GroupedBarChartArguments(
        methods=methods,
        x_labels=x_labels,
        data=data,
        show_bar_label=True,
        y_max_tick_value=1.00,
        x_axis_name="Sequence Length",
        y_axis_name="Normalized Speedup",
        show_grid=True,
        show_legend=True,
        legend_extras={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.25),
            "ncol": len(methods),
        },
    )
    chart = GroupedBarChart(args)

    fig1, ax1 = plt.subplots(figsize=(15, 3), dpi=300)
    chart.render(ax1)
    plt.savefig("example1_grouped_bar_chart.png", bbox_inches="tight")

    # ========================================================
    # Example 2: Using the updated benchmark helper method with GroupedBarChartArguments
    logger.info(
        "Example 2: Using the benchmark helper method with GroupedBarChartArguments"
    )
    benchmark_args = GroupedBarChartArguments(
        methods=methods,
        x_labels=x_labels,
        data=data,
        show_bar_label=True,
        y_max_tick_value=1.00,
        x_axis_name="Sequence Length",
        y_axis_name="Normalized Speedup",
        show_grid=True,
        show_legend=True,
        legend_extras={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.25),
            "ncol": len(methods),
            "frameon": False,
        },
    )

    fig2, ax2, chart2 = GroupedBarChart.create_benchmark_chart(
        benchmark_args,
        title=None,
        figsize=(15, 3),
        dpi=300,
        output_path="example2_benchmark_chart.png",
    )

    # ========================================================
    # Example 3: Using hatches for the bars
    logger.info("Example 3: Using hatches for the bars")
    hatches_example_args = GroupedBarChartArguments(
        methods=methods,
        x_labels=x_labels,
        data=data,
        show_bar_label=True,
        y_max_tick_value=1.00,
        x_axis_name="Sequence Length",
        y_axis_name="Normalized Speedup",
        show_grid=True,
        show_legend=True,
        hatches=["", "//", "\\\\", "xx", "++"],  # Different hatches for each method
        legend_extras={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.25),
            "ncol": len(methods),
            "frameon": False,
        },
    )

    fig3, ax3, chart3 = GroupedBarChart.create_benchmark_chart(
        hatches_example_args,
        title=None,
        figsize=(15, 3),
        dpi=300,
        output_path="example3_hatches_chart.png",
    )
