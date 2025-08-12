"""
Game Performance Log Analysis Tool.

This script parses and analyzes game performance logs containing '[PerfCounters]'
lines, generating insightful visualizations and statistical summaries to help
identify performance bottlenecks.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MatplotlibFigure

import matplotlib.pyplot as plt

# --- Configuration ---

PIO_RENDERER = "browser"
PLOT_STYLE_SEABORN = "seaborn-v0_8-darkgrid"

# --- Constants ---

# Log Parsing
PERF_COUNTER_PREFIX = "[PerfCounters]"

# DataFrame Column Names
COL_FRAME = "Frame"
COL_GAMELOOP = "GameLoop"
COL_SERVICES_SUM = "ServicesSum"
COL_UNACCOUNTED = "UnaccountedTime"
COL_TOP1_SERVICE = "Top1_Service"
COL_TOP1_TIME = "Top1_Time"
COL_TOP2_SERVICE = "Top2_Service"
COL_TOP2_TIME = "Top2_Time"

# Columns that are calculated metadata, not raw service timings.
METADATA_COLS = [
    COL_GAMELOOP,
    COL_SERVICES_SUM,
    COL_UNACCOUNTED,
    COL_TOP1_SERVICE,
    COL_TOP1_TIME,
    COL_TOP2_SERVICE,
    COL_TOP2_TIME,
]

# Defines the rich tooltip shown when hovering over data points in the interactive chart.
HOVER_TEMPLATE = """
<b>Frame %{x}</b><br>
------------------<br>
<b>GameLoop: %{customdata[0]:.2f}ms</b><br>
Services Sum: %{customdata[1]:.2f}ms<br>
Unaccounted: %{customdata[2]:.2f}ms<br>
<br>
<b>Top Contributor:</b> %{customdata[3]} (%{customdata[4]:.2f}ms)<br>
<b>2nd Contributor:</b> %{customdata[5]} (%{customdata[6]:.2f}ms)
<extra></extra>
""".strip()


def parse_log_line(line: str) -> Optional[Dict[str, float]]:
    """Parses a single PerfCounters log line into a dictionary of service timings.

    Args:
        line: A string representing one line from the log file.

    Returns:
        A dictionary mapping service names to their execution time in milliseconds,
        or None if the line is not a valid performance counter line.
    """
    if not line.startswith(PERF_COUNTER_PREFIX):
        return None

    clean_line = line.removeprefix(PERF_COUNTER_PREFIX).strip(" |")
    parts = [p.strip() for p in clean_line.split("|")]
    timings = {}

    for part in parts:
        if not part:
            continue
        try:
            # Split from the right to handle service names with spaces
            name, value_str = part.rsplit(" ", 1)
            value = float(value_str.removesuffix("ms"))
            timings[name] = value
        except (ValueError, IndexError):
            print(f"Warning: Could not parse entry: '{part}'", file=sys.stderr)

    return timings


def _find_top_two_contributors(row: pd.Series) -> pd.Series:
    """Finds the top two service-time contributors for a single frame (row).

    This helper function is intended for use with `pandas.DataFrame.apply()`.
    It takes a Series of service timings for one frame and identifies the two
    services with the highest values.

    Args:
        row: A pandas Series for a single frame, where the index contains
             service names and the values are their execution times.

    Returns:
        A new pandas Series containing the name and time for the top two
        contributors. Uses 'N/A' and 0.0 as placeholders if fewer than
        two services exist.
    """
    top_services = row.nlargest(2)
    top1_service, top1_time = "N/A", 0.0
    top2_service, top2_time = "N/A", 0.0

    if not top_services.empty:
        top1_service, top1_time = top_services.index[0], top_services.iloc[0]
        if len(top_services) > 1:
            top2_service, top2_time = top_services.index[1], top_services.iloc[1]

    return pd.Series(
        {
            COL_TOP1_SERVICE: top1_service,
            COL_TOP1_TIME: top1_time,
            COL_TOP2_SERVICE: top2_service,
            COL_TOP2_TIME: top2_time,
        }
    )


def analyze_and_enrich_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Reads a log, parses PerfCounter lines, and enriches data with statistics.

    This function performs the core data loading and transformation:
    1. Reads all PerfCounter lines from the log file, using the original line
       number as the frame identifier.
    2. Converts the data into a pandas DataFrame.
    3. Calculates aggregate stats like the sum of all services and unaccounted time.
    4. Identifies the top two performance contributors for each frame.

    Args:
        file_path: Path object pointing to the log file.

    Returns:
        A pandas DataFrame containing the analyzed performance data, or None on
        failure (e.g., no valid data found).
    """
    print(f"Reading log file: {file_path}...")

    all_data = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            if parsed_data := parse_log_line(line):
                parsed_data[COL_FRAME] = line_num
                all_data.append(parsed_data)

    if not all_data:
        print("Error: No valid performance counter lines found.", file=sys.stderr)
        return None

    print(f"Found {len(all_data)} valid log entries. Analyzing...")
    df = pd.DataFrame(all_data).set_index(COL_FRAME)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    if COL_GAMELOOP not in df.columns:
        print(
            f"Error: Required counter '{COL_GAMELOOP}' not found. Cannot proceed.",
            file=sys.stderr,
        )
        return None

    service_cols = df.columns.drop(COL_GAMELOOP, errors="ignore")

    tqdm.pandas(desc="Calculating frame statistics")
    df[COL_SERVICES_SUM] = df[service_cols].sum(axis=1)
    df[COL_UNACCOUNTED] = df[COL_GAMELOOP] - df[COL_SERVICES_SUM]

    tqdm.pandas(desc="Finding top contributors")
    top_contributors = df[service_cols].progress_apply(
        _find_top_two_contributors, axis=1
    )
    df = df.join(top_contributors)

    return df


def generate_statistical_summary(
    df: pd.DataFrame, method: str, percentile_level: int
) -> Tuple[pd.DataFrame, float]:
    """Prints a statistical summary of GameLoop times and identifies outlier frames.

    Args:
        df: The main performance DataFrame.
        method: The outlier detection method ("percentile" or "iqr").
        percentile_level: The percentile to use if method is "percentile".

    Returns:
        A tuple containing:
        - A DataFrame of the outlier frames.
        - The calculated outlier threshold time in milliseconds.
    """
    gameloop_times = df[COL_GAMELOOP]

    print("\n--- GameLoop Performance Summary ---")
    print(f"        Mean: {gameloop_times.mean():>7.2f}ms")
    print(f"      Median: {gameloop_times.median():>7.2f}ms (50th percentile)")
    print(f"     Std Dev: {gameloop_times.std():>7.2f}ms")
    print("-" * 34)
    print(f"  90th Pctle: {gameloop_times.quantile(0.90):>7.2f}ms")
    print(f"  95th Pctle: {gameloop_times.quantile(0.95):>7.2f}ms")
    print(f"  99th Pctle: {gameloop_times.quantile(0.99):>7.2f}ms")
    print(f"   Max Frame: {gameloop_times.max():>7.2f}ms at Frame {gameloop_times.idxmax()}")
    print("-" * 34)

    if method == "percentile":
        outlier_threshold = gameloop_times.quantile(percentile_level / 100.0)
        print(f"Using Percentile method for outlier detection ({percentile_level}th percentile).")
    elif method == "iqr":
        q1 = gameloop_times.quantile(0.25)
        q3 = gameloop_times.quantile(0.75)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        print("Using IQR method for outlier detection.")
        print(f"  (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Threshold=Q3+1.5*IQR)")
    else:
        print("Warning: Invalid outlier method. No outliers will be shown.", file=sys.stderr)
        return pd.DataFrame(), float("inf")

    outliers = df[gameloop_times > outlier_threshold]
    print(f"Threshold set at {outlier_threshold:.2f}ms.")
    print(f"Detected {len(outliers)} outlier frames, marked in red on the interactive plot.")
    print("-" * 34)

    return outliers, outlier_threshold


def downsample_for_plot(
    df: pd.DataFrame,
    outliers: pd.DataFrame,
    max_points: int,
    preserve_percentile: float = 0.95,
) -> pd.DataFrame:
    """Intelligently downsamples a DataFrame for plotting, preserving key data points.

    This method uses stratified sampling to create a visually representative
    subset of the data, ensuring that the most critical frames (outliers and
    high-percentile spikes) are always included.

    The strata are:
    1. All identified outlier frames.
    2. All frames with a GameLoop time above a given percentile.
    3. A random sample of the remaining "normal" frames.

    Args:
        df: The full, original DataFrame.
        outliers: The DataFrame containing all identified outlier frames.
        max_points: The target maximum number of points for the plot. Note that
                    the final count may exceed this if the number of outliers
                    and high-percentile frames is large.
        preserve_percentile: The percentile (0.0 to 1.0) used to define
                             "high-value" frames that should be preserved.

    Returns:
        A new, downsampled DataFrame.
    """
    if len(df) <= max_points:
        return df

    print(f"\nSmart-downsampling data to ~{max_points} points for timeline...")

    # Identify all frames that must be preserved.
    outlier_indices = outliers.index
    high_threshold = df[COL_GAMELOOP].quantile(preserve_percentile)
    is_high_value = df[COL_GAMELOOP] >= high_threshold
    is_not_outlier = ~df.index.isin(outlier_indices)
    high_frames_to_add = df[is_high_value & is_not_outlier]

    print(f"  - Preserving all {len(outliers)} outlier frames.")
    print(f"  - Preserving {len(high_frames_to_add)} additional high-value frames (> {high_threshold:.2f}ms).")

    preserved_frames = pd.concat([outliers, high_frames_to_add])
    num_preserved = len(preserved_frames)
    normal_sample = pd.DataFrame()

    if num_preserved >= max_points:
        print(f"  - Warning: The {num_preserved} preserved frames exceed the target of {max_points}.")
        print("  - Only these high-priority frames will be plotted.")
        plot_df = preserved_frames
    else:
        sample_size = max_points - num_preserved
        normal_frames = df.drop(index=preserved_frames.index)
        actual_sample_size = min(sample_size, len(normal_frames))

        if actual_sample_size > 0:
            normal_sample = normal_frames.sample(n=actual_sample_size, random_state=42)
            print(f"  - Randomly sampling {len(normal_sample)} normal-performance frames.")

        plot_df = pd.concat([preserved_frames, normal_sample])

    plot_df = plot_df.sort_index()
    print(f"  - Final plot contains {len(plot_df)} points.")

    return plot_df


def get_top_services_by_average(df: pd.DataFrame, top_n: int) -> pd.Series:
    """Calculates and returns the top N slowest services by average time.

    This function computes the mean execution time for each service across all
    frames, excluding metadata columns, to provide a high-level summary of
    the most expensive services.

    Args:
        df: The main performance DataFrame containing per-frame timings.
        top_n: The number of top services to return.

    Returns:
        A pandas Series with the top N services as the index and their
        average times as values, sorted in descending order.
    """
    service_cols = df.columns.drop(METADATA_COLS, errors="ignore")
    return df[service_cols].mean().nlargest(top_n)


def create_interactive_timeline(
    df: pd.DataFrame,
    top_service_names: List[str],
    outliers_index: pd.Index,
    outlier_threshold: float,
) -> go.Figure:
    """Creates and returns an interactive Plotly chart showing performance over time.

    This high-performance version uses separate traces for lines and outlier
    markers to ensure fast rendering, while carefully assigning customdata to
    each trace to guarantee correct hover information.

    Args:
        df: The main performance DataFrame (can be downsampled).
        top_service_names: A list of the top slowest services to include.
        outliers_index: A pandas Index containing the frame numbers of the outliers.
        outlier_threshold: The time threshold for marking outliers.

    Returns:
        A Plotly Figure object for display or saving.
    """
    print("Generating interactive performance timeline...")
    pio.renderers.default = PIO_RENDERER
    fig = go.Figure()

    all_line_traces = [COL_GAMELOOP, COL_SERVICES_SUM, COL_UNACCOUNTED] + top_service_names
    for col in all_line_traces:
        is_legend_only = col in top_service_names
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            visible='legendonly' if is_legend_only else True,
            customdata=df[METADATA_COLS],
            hovertemplate=HOVER_TEMPLATE,
        ))


    if not outliers_index.empty:
        outlier_data = df.loc[outliers_index]
        fig.add_trace(go.Scatter(
            x=outlier_data.index,
            y=outlier_data[COL_GAMELOOP],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8, symbol='x', line=dict(width=2)),
            customdata=outlier_data[METADATA_COLS],
            hovertemplate=HOVER_TEMPLATE,
        ))


    fig.add_hline(
        y=outlier_threshold,
        line_dash="dot",
        annotation_text=f"Outlier Threshold ({outlier_threshold:.2f}ms)",
        annotation_position="bottom right",
        line_color="orange",
    )

    fig.update_layout(
        title="Interactive GameLoop Performance Analysis",
        xaxis_title="Frame Number (Log Line)",
        yaxis_title="Time (ms)",
        legend_title_text="Metrics (click to toggle)",
        template="plotly_white",
        title_font_size=20,
    )

    return fig


def create_summary_bar_chart(average_times: pd.Series) -> "MatplotlibFigure":
    """Creates and returns a static bar chart of the top N slowest services.

    Args:
        average_times: A Series of service names and their average execution times.

    Returns:
        A Matplotlib Figure object for display or saving.
    """
    top_n = len(average_times)
    print(f"Generating Top-{top_n} slowest services bar chart...")

    plt.style.use(PLOT_STYLE_SEABORN)
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")

    sns.barplot(
        x=average_times.values,
        y=average_times.index,
        hue=average_times.index,
        legend=False,
        ax=ax,
        palette="viridis_r",
        orient="h",
    )

    ax.set_title(
        f"Top {top_n} Slowest Services (by Average Time)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Average Time (ms)", fontsize=12)
    ax.set_ylabel("Service Name", fontsize=12)
    ax.bar_label(ax.containers[0], fmt="%.2fms", padding=3, fontsize=10)

    return fig


def positive_int(value: str) -> int:
    """Argparse type validator for positive integers."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer")
    return ivalue


def main() -> None:
    """Parses command-line arguments and orchestrates the analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze game performance logs to identify spikes and slow services.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("logfile", type=Path, help="Path to the log file to be analyzed.")
    parser.add_argument(
        "--top",
        type=positive_int,
        default=10,
        help="Number of top slowest services to display in graphs. (default: %(default)s)",
    )
    parser.add_argument(
        "--outlier-method",
        choices=["percentile", "iqr"],
        default="percentile",
        help="Method to detect outliers:\n"
        " - percentile: Flags frames above a certain percentile (see --percentile).\n"
        " - iqr: Uses the statistical Interquartile Range method (Q3 + 1.5*IQR).\n"
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=98,
        choices=range(1, 100),
        metavar="[1-99]",
        help="Percentile for outlier detection when --outlier-method=percentile.\n"
        "e.g., 95 means frames in the slowest 5%% are outliers. (default: %(default)s)",
    )
    parser.add_argument(
        "--downsample",
        type=positive_int,
        metavar="N",
        help="Smartly reduce data points to N for large logs to improve plotting performance.\n"
        "Preserves all outliers and high-percentile values while sampling normal frames."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Directory to save plots to. If provided, plots are not shown interactively."
    )

    args = parser.parse_args()

    if not args.logfile.is_file():
        print(f"Error: The file '{args.logfile}' was not found.", file=sys.stderr)
        sys.exit(1)

    try:
        df = analyze_and_enrich_data(args.logfile)
        if df is None:
            sys.exit(1)

        outliers, threshold = generate_statistical_summary(
            df, args.outlier_method, args.percentile
        )

        # Use only outliers relevant to the main DataFrame for downsampling
        plot_df = df
        if args.downsample and len(df) > args.downsample:
            plot_outliers = outliers[outliers.index.isin(df.index)]
            plot_df = downsample_for_plot(df, plot_outliers, args.downsample)

        average_times = get_top_services_by_average(df, args.top)
        top_service_names = average_times.index.tolist()

        outlier_indices_in_plot = outliers.index.intersection(plot_df.index)
        interactive_fig = create_interactive_timeline(
            plot_df, top_service_names, outlier_indices_in_plot, threshold
        )
        summary_fig = create_summary_bar_chart(average_times)

        if args.export_dir:
            export_dir = args.export_dir
            export_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nExporting plots to '{export_dir.resolve()}'...")

            html_path = export_dir / f"{args.logfile.stem}_timeline.html"
            interactive_fig.write_html(html_path)
            print(f"  - Saved interactive plot to: {html_path}")

            summary_path = export_dir / f"{args.logfile.stem}_summary_barchart.png"
            summary_fig.savefig(summary_path, dpi=150)
            plt.close(summary_fig)
            print(f"  - Saved summary chart to:    {summary_path}")

            print("\nAnalysis complete. Plots have been saved.")

        else:
            print("\nAnalysis complete. Displaying interactive plots.")
            print("Close plot windows/tabs to exit the script.")
            interactive_fig.show()
            plt.show()

    except Exception:
        print("\nAn unexpected error occurred. See details below:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()