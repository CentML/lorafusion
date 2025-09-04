"""Utilities for drawing multi-line plots for the benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

# Fix font type for PDF and PS files, which is required by the ACM/IEEE templates.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

# Z-order constants for controlling element layering
ZORDER_GRID = 0
ZORDER_AXIS = 1
ZORDER_LINES = 3
ZORDER_MARKERS = 4

# Default color palette matching the provided image
DEFAULT_PALETTE = [
    "#000000",  # black
    "#CC0000",  # red
    "#555555",  # dark gray
    "#AAAAAA",  # gray
    "#EFEFEF",  # light gray/white
]

# Default marker styles
DEFAULT_MARKERS = ["o", "s", "^", "D", "x", "+", "*"]

# Default line styles
DEFAULT_LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


@dataclass
class MultiLinePlotArguments:
    """Arguments for the multi-line plot."""

    series: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of series names whose performance is compared as lines. "
            "Like methods in GroupedBarChart. e.g. ['Megatron', 'DeepSpeed', 'Mist']"
        },
    )
    x_values: list[Any] = field(
        default_factory=list,
        metadata={
            "help": "List of x-axis values for the plot. e.g. [1, 2, 4, 8, 16, 32]."
        },
    )
    data: list[list[float | int | str | None]] = field(
        default_factory=list,
        metadata={
            "help": "2D list of data to be plotted. The outer list represents series, "
            "and the inner list represents y-values for each x-value. The 2D list "
            "should have the size of (num_series, num_x_values). Data can be missing, "
            "represented as None."
        },
    )
    palette: list[str] = field(
        default_factory=lambda: DEFAULT_PALETTE.copy(),
        metadata={"help": "List of colors for the lines."},
    )
    markers: list[str] = field(
        default_factory=lambda: DEFAULT_MARKERS.copy(),
        metadata={"help": "List of marker styles for the lines."},
    )
    line_styles: list[Any] = field(
        default_factory=lambda: DEFAULT_LINE_STYLES.copy(),
        metadata={"help": "List of line styles for the lines."},
    )
    line_width: float = field(
        default=2.0,
        metadata={"help": "Width for the lines. Default is 2.0."},
    )
    marker_size: float = field(
        default=8.0,
        metadata={"help": "Size of markers. Default is 8.0."},
    )
    line_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra arguments for the lines. Default is an empty dictionary."
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
    x_limit: tuple[float | None, float | None] = field(
        default=(None, None),
        metadata={
            "help": "Lower and upper limits of the x-axis. Default is (None, None)."
        },
    )
    x_tick_values: list[Any] = field(
        default_factory=list,
        metadata={"help": "List of x-axis tick values. Default is an empty list."},
    )
    x_scale: Literal["linear", "log", "symlog", "logit"] = field(
        default="linear",
        metadata={"help": "Scale for the x-axis. Default is 'linear'."},
    )
    x_scale_extras: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra arguments for the x-axis scale. "
            "Default is an empty dictionary."
        },
    )
    x_ticks_fontsize: float = field(
        default=12, metadata={"help": "Font size for the x-axis labels. Default is 12."}
    )
    x_ticks_formatter: Callable[[Any], str] | None = field(
        default=None,
        metadata={
            "help": "Formatter for the x-axis ticks. Default is None (auto-formatting)."
        },
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
    y_scale: Literal["linear", "log", "symlog", "logit"] = field(
        default="linear",
        metadata={"help": "Scale for the y-axis. Default is 'linear'."},
    )
    num_y_ticks: int | None = field(
        default=6, metadata={"help": "Number of y-axis ticks. Default is 6."}
    )
    y_ticks_formatter: Callable[[float], str] | None = field(
        default=None,
        metadata={
            "help": "Formatter for the y-axis ticks. Default is None (auto-formatting)."
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
                "Upper limit of the y-axis. If None, the upper limit will be determined"
                " automatically."
            )
        },
    )
    y_min_tick_value: float | None = field(
        default=None,
        metadata={
            "help": (
                "Lower limit of the y-axis. If None, the lower limit will be determined"
                " automatically."
            )
        },
    )
    y_limit: tuple[float | None, float | None] = field(
        default=(None, None),
        metadata={
            "help": (
                "Lower and upper limits of the y-axis. If None, the limits will be "
                "determined automatically."
            )
        },
    )
    y_limit_padding: float = field(
        default=0.05,
        metadata={
            "help": (
                "Padding for y-axis limits as a fraction of the data range. Default is "
                "0.05 (5%)."
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
            "linestyle": "--",
            "alpha": 0.3,
            "color": "black",
            "linewidth": 0.5,
            "zorder": ZORDER_GRID,  # This ensures grid is behind the lines
        },
        metadata={
            "help": "Extra arguments for the grid. Default is an empty dictionary."
        },
    )
    interpolation: Literal["linear", "step", "nearest", "cubic"] = field(
        default="linear",
        metadata={"help": "Type of interpolation to use. Default is 'linear'."},
    )

    def _validate_data_exists(self) -> None:
        """Check if data exists."""
        if self.data is None:
            msg = "No data to render"
            raise ValueError(msg)

    def _validate_list_data(self, num_series: int, num_x_values: int) -> None:
        """Validate data when it's a list."""
        if len(self.data) != num_series:
            msg = (
                f"The number of series must match the number of rows in the data. "
                f"{num_series} != {len(self.data)}"
            )
            raise ValueError(msg)

        for row in self.data:
            if not isinstance(row, list) and not isinstance(row, np.ndarray):
                msg = (
                    f"Each row in the data must be a list or numpy array. "
                    f"{type(row)} is not a list or array."
                )
                raise TypeError(msg)

            if len(row) != num_x_values:
                msg = (
                    "The number of x values must match the number of columns in the "
                    f"data. {num_x_values} != {len(row)}"
                )
                raise ValueError(msg)

    def _validate_numpy_data(self, num_series: int, num_x_values: int) -> None:
        """Validate data when it's a numpy array."""
        if self.data.shape[0] != num_series:
            msg = (
                f"The number of series must match the number of rows in the data. "
                f"{num_series} != {self.data.shape[0]}"
            )
            raise ValueError(msg)

        if self.data.shape[1] != num_x_values:
            msg = (
                f"The number of x values must match the number of columns in the data. "
                f"{num_x_values} != {self.data.shape[1]}"
            )
            raise ValueError(msg)

    def _validate_has_numeric_data(self) -> None:
        """Check if data contains valid numeric values."""
        flat_data = [
            y
            for row in self.data
            for y in row
            if y is not None and isinstance(y, int | float)
        ]

        if not flat_data:
            msg = "No valid numeric data to plot"
            raise ValueError(msg)

    def validate(self) -> None:
        """Validate the arguments."""
        num_series = len(self.series)
        num_x_values = len(self.x_values)

        # Check if data exists
        self._validate_data_exists()

        # Validate data structure based on type
        if isinstance(self.data, list):
            self._validate_list_data(num_series, num_x_values)
        elif isinstance(self.data, np.ndarray):
            self._validate_numpy_data(num_series, num_x_values)
        else:
            msg = (
                f"The data must be a list or a numpy array. "
                f"{type(self.data)} is not a list or a numpy array."
            )
            raise TypeError(msg)

        # Check if data contains valid numeric values
        self._validate_has_numeric_data()


class MultiLinePlot:
    """Class for rendering multi-line plots."""

    def __init__(self, args: MultiLinePlotArguments) -> None:
        """Initialize the multi-line plot.

        Args:
            args: Arguments for the multi-line plot.
        """
        self.args = args

    def _setup_axes_styling(self, axes: Axes) -> None:
        """Set up axis labels, ticks, and other styling.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        # Prepare styling kwargs
        x_axis_kwargs = {"fontsize": args.x_axis_fontsize}
        y_axis_kwargs = {"fontsize": args.y_axis_fontsize}
        x_ticks_kwargs = {"labelsize": args.x_ticks_fontsize}
        y_ticks_kwargs = {"labelsize": args.y_ticks_fontsize}

        # Add custom parameters
        x_axis_kwargs.update(args.x_axis_extras)
        y_axis_kwargs.update(args.y_axis_extras)
        x_ticks_kwargs.update(args.x_ticks_extras)
        y_ticks_kwargs.update(args.y_ticks_extras)

        # Apply axis labels
        axes.set_xlabel(args.x_axis_name, **x_axis_kwargs)
        axes.set_ylabel(args.y_axis_name, **y_axis_kwargs)

        # Set axis scales
        axes.set_xscale(args.x_scale, **args.x_scale_extras)
        axes.set_yscale(args.y_scale)

        # Set x-axis tick values if provided
        if args.x_tick_values:
            axes.set_xticks(args.x_tick_values)

        if args.x_limit:
            axes.set_xlim(*args.x_limit)

        # Apply x-axis formatting if provided
        if args.x_ticks_formatter:
            axes.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: args.x_ticks_formatter(x))
            )
        axes.tick_params(axis="x", **x_ticks_kwargs)

        # Apply y-axis formatting if provided
        if args.y_ticks_formatter:
            axes.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: args.y_ticks_formatter(y))
            )
        axes.tick_params(axis="y", **y_ticks_kwargs)

        # Set y-axis limits if specified
        y_min, y_max = args.y_limit

        # If limits not provided, calculate from data
        if y_min is None or y_max is None:
            data_values = [
                y
                for row in args.data
                for y in row
                if y is not None and isinstance(y, int | float)
            ]

            data_min = min(data_values) if data_values else 0
            data_max = max(data_values) if data_values else 1
            data_range = data_max - data_min

            padding = args.y_limit_padding * data_range

            y_min = data_min - padding if y_min is None else y_min
            y_max = data_max + padding if y_max is None else y_max

        # Override with explicit min/max tick values if provided
        if args.y_min_tick_value is not None:
            y_min = args.y_min_tick_value
        if args.y_max_tick_value is not None:
            y_max = args.y_max_tick_value

        axes.set_ylim(y_min, y_max)

        # Add grid if enabled
        if args.show_grid:
            axes.grid(**args.grid_extras)

        # Add legend if enabled
        if args.show_legend:
            legend_kwargs = {"fontsize": args.legend_fontsize}
            legend_kwargs.update(args.legend_extras)
            axes.legend(**legend_kwargs)

    def _render_lines(self, axes: Axes) -> None:
        """Render the lines on the axes.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        # Extract data and dimensions
        series_names = args.series
        x_values = args.x_values
        data = args.data

        # Use default styles if none provided
        palette = args.palette
        markers = args.markers
        line_styles = args.line_styles

        # Plot each series as a line
        for i, series_name in enumerate(series_names):
            color = palette[i % len(palette)]
            marker = markers[i % len(markers)]
            line_style = line_styles[i % len(line_styles)]

            # Filter out None values for this series
            valid_indices = [
                j
                for j, y in enumerate(data[i])
                if y is not None and isinstance(y, int | float)
            ]
            valid_x = [x_values[j] for j in valid_indices]
            valid_y = [data[i][j] for j in valid_indices]

            if not valid_x:
                continue  # Skip series with no valid data

            # Set up line styling
            line_kwargs = {
                "color": color,
                "marker": marker,
                "linestyle": line_style,
                "linewidth": args.line_width,
                "markersize": args.marker_size,
                "label": series_name,
                "zorder": ZORDER_LINES,
            }
            line_kwargs.update(args.line_extras)

            # Plot the line
            axes.plot(valid_x, valid_y, **line_kwargs)

    def render(self, axes: Axes) -> None:
        """Render the multi-line plot on the given matplotlib axes.

        Args:
            axes: Matplotlib axes to render on
        """
        args = self.args

        # Validate arguments
        args.validate()

        # Render the lines
        self._render_lines(axes)

        # Setup axes styling
        self._setup_axes_styling(axes)

        # Add figure margins for better spacing
        plt.tight_layout()

    @classmethod
    def create_benchmark_plot(
        cls,
        args: MultiLinePlotArguments,
        *,
        title: str | None = None,
        title_fontsize: int = 16,
        title_extras: dict[str, Any] | None = None,
        figsize: tuple = (10, 6),
        dpi: int = 300,
        output_path: str | None = None,
    ) -> tuple:
        """Create a standard benchmark comparison plot.

        Args:
            args: MultiLinePlotArguments object containing plot configuration
            title: Optional title for the plot
            title_fontsize: Font size for the title
            title_extras: Additional keyword arguments for the title
            figsize: Figure size (width, height) in inches
            dpi: Resolution in dots per inch
            output_path: Optional path to save the plot

        Returns:
            tuple: (figure, axes, plot) objects for further customization

        Example:
            >>> args = MultiLinePlotArguments(
            ...     series=["Algorithm1", "Algorithm2"],
            ...     x_values=[1, 2, 4, 8, 16, 32],
            ...     data=[[1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
            ...           [1.1, 1.6, 2.2, 3.5, 6.0, 9.0]],
            ...     x_axis_name="Batch Size",
            ...     y_axis_name="Throughput (samples/sec)",
            ... )
            >>> fig, ax, plot = MultiLinePlot.create_benchmark_plot(
            ...     args,
            ...     title="Throughput Comparison",
            ...     figsize=(12, 8),
            ...     output_path="benchmark_plot.png"
            ... )
        """
        # Create plot instance
        plot = cls(args)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Add title if provided
        if title:
            title_kwargs = {"fontsize": title_fontsize}
            if title_extras is not None:
                title_kwargs.update(title_extras)
            ax.set_title(title, **title_kwargs)

        # Render the plot
        plot.render(ax)

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")

        return fig, ax, plot

    def render_shared_legend(
        self,
        fig: plt.Figure,
        bbox_to_anchor: tuple[float, float] = (0.5, 1.05),
        fontsize: int = 12,
        *,
        frameon: bool = False,
        ncol: int | None = None,
        **kwargs,
    ) -> None:
        """Render a shared legend for multiple subplots.

        This method creates a legend that can be shared across multiple subplots,
        typically placed at the top of the figure.

        Args:
            fig: Matplotlib figure to render the legend on
            bbox_to_anchor: Position of the legend relative to the figure
            fontsize: Font size for the legend text
            frameon: Whether to draw a frame around the legend
            ncol: Number of columns for the legend items, defaults to len(series) if
                None
            **kwargs: Additional keyword arguments to pass to plt.figlegend
        """
        args = self.args

        # Create legend handles and labels
        handles = []
        labels = []

        # Only include non-hidden series in the legend
        for i, series_name in enumerate(args.series):
            if series_name != "_hidden_":
                # Get palette, marker, and line style for this series
                color = args.palette[i % len(args.palette)]
                marker = args.markers[i % len(args.markers)]
                line_style = args.line_styles[i % len(args.line_styles)]

                # Create a line for the legend
                handle = plt.Line2D(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    linewidth=args.line_width,
                    markersize=args.marker_size,
                )

                handles.append(handle)
                labels.append(series_name)

        # Set number of columns if not specified
        if ncol is None:
            ncol = len(handles)

        # Create the legend
        return fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            fontsize=fontsize,
            frameon=frameon,
            ncol=ncol,
            **kwargs,
        )


if __name__ == "__main__":
    from loguru import logger

    # ========================================================
    # Example 1: Basic line plot with multiple series
    logger.info("Example 1: Basic line plot with multiple series")
    series = ["Algorithm A", "Algorithm B", "Algorithm C", "Our Method"]
    x_values = [1, 2, 4, 8, 16, 32, 64]

    # Example data: rows = series, columns = x values
    data = np.array(
        [
            [1.0, 1.8, 3.2, 5.5, 9.8, 17.5, 30.2],  # Algorithm A
            [1.2, 2.0, 3.5, 6.2, 11.0, 19.8, 35.1],  # Algorithm B
            [0.9, 1.5, 2.8, 5.0, 9.2, 16.8, 28.5],  # Algorithm C
            [1.3, 2.3, 4.0, 7.0, 12.5, 22.0, 38.0],  # Our Method
        ]
    )

    args = MultiLinePlotArguments(
        series=series,
        x_values=x_values,
        data=data,
        x_axis_name="Batch Size",
        y_axis_name="Throughput (samples/sec)",
        show_grid=True,
        show_legend=True,
        x_scale="log",  # Use log scale for x-axis
        legend_extras={
            "loc": "upper left",
            "frameon": True,
        },
    )

    plot = MultiLinePlot(args)

    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    plot.render(ax1)
    plt.savefig("example1_multi_line_plot.png", bbox_inches="tight")

    # ========================================================
    # Example 2: Using the helper method with MultiLinePlotArguments
    logger.info("Example 2: Using the helper method with MultiLinePlotArguments")

    # Example with different interpolation
    benchmark_args = MultiLinePlotArguments(
        series=series,
        x_values=x_values,
        data=data,
        x_axis_name="Batch Size",
        y_axis_name="Throughput (samples/sec)",
        show_grid=True,
        x_scale="linear",  # Use linear scale
        y_limit=(0, None),  # Start y-axis at 0
        line_width=2.5,  # Thicker lines
        marker_size=10,  # Larger markers
        show_legend=True,
        legend_extras={
            "loc": "upper left",
            "frameon": True,
            "shadow": True,
        },
    )

    fig2, ax2, plot2 = MultiLinePlot.create_benchmark_plot(
        benchmark_args,
        title="Throughput vs Batch Size",
        title_fontsize=18,
        figsize=(12, 8),
        dpi=300,
        output_path="example2_benchmark_plot.png",
    )
