# Game Performance Log Analyzer

A Python script for parsing and analyzing performance data from game engine logs. It produces a statistical summary, an interactive performance timeline, and a bar chart of the most expensive services.

## Expected Log Format

The script requires log lines to adhere to a specific format to be parsed correctly.

-   **Prefix**: Each relevant line must begin with `[PerfCounters]`.
-   **Structure**: Data is a series of `| ServiceName TimeValue` pairs. Service names can contain spaces.
-   **Values**: The time value must end with `ms` (e.g., `16.67ms`).
-   **Required Counter**: A counter named `GameLoop` must be present, as it is used as the total frame time for analysis.

**Example Log Lines:**
```text
[PerfCounters] GameLoop 16.78ms | RenderService 5.43ms | PhysicsService 2.11ms | AudioUpdate 0.50ms |
[PerfCounters] GameLoop 25.12ms | RenderService 12.80ms | A Service With Spaces 4.50ms | AudioUpdate 0.75ms |
```

## Functionality

-   **Statistical Summary**: Prints a summary of `GameLoop` timings (mean, median, std dev, 90/95/99th percentiles) to the console.
-   **Outlier Detection**: Identifies outlier frames using one of two configurable methods:
    -   `percentile`: Flags frames with a `GameLoop` time above a given percentile.
    -   `iqr`: Uses the Interquartile Range method (`Q3 + 1.5*IQR`) to find statistical outliers.
-   **Plot Generation**:
    -   **Interactive Timeline**: An HTML file (using Plotly) that plots frame timings over the capture duration. Outliers are marked, and a detailed tooltip provides per-frame information. The visibility of each service can be toggled in the legend.
    -   **Static Bar Chart**: A PNG file (using Matplotlib/Seaborn) showing the top N services ranked by their average execution time.
-   **Downsampling**: For very large log files, an optional downsampling mode can be used to reduce the number of points plotted. This mode preserves all identified outliers and high-percentile frames while taking a random sample of the remaining frames to ensure the plot remains representative and responsive.
-   **Export Mode**: Plots can be viewed interactively (default) or saved directly to a specified directory for automation or reporting.

## Requirements

-   Python 3.9+
-   Dependencies are listed in `requirements.txt`.

## Installation

1.  Clone this repository or download the files.
2.  Create and activate a virtual environment.
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate it
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```
3.  Install the required Python packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script from your terminal, providing the path to your log file.

```bash
python perf_analyzer.py "/path/to/logfile" [OPTIONS]
```

### Command-Line Arguments

| Argument             | Description                                                                                                                               | Default      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------| ------------ |
| `logfile` (required) | Path to the log file to be analyzed.                                                                                                      | -            |
| `--top N`            | The number of top slowest services to display in graphs. Must be a positive integer.                                                      | `10`         |
| `--outlier-method`   | The method for detecting outliers. Choices: `percentile` or `iqr` (Q3 + 1.5*IQR).                                                         | `percentile` |
| `--percentile P`     | The percentile (1-99) for outlier detection when using the `percentile` method.                                                           | `98`         |
| `--downsample N`     | Reduce data points to `N` for large logs. Preserves outliers and high-percentile spikes while sampling normal frames.                     | (none)       |
| `--export-dir DIR`   | Directory to save plots to. If provided, plots are saved as files (e.g., `logfile_timeline.html`) instead of being shown interactively.   | (none)       |

### Examples

**1. Basic Analysis**
Analyze a log file and display the plots interactively.
```bash
python perf_analyzer.py "/path/to/logfile"
```

**2. Adjusting Outlier Detection**
Use the 99th percentile for outlier detection and show the top 15 services in the summary chart.
```bash
python perf_analyzer.py "/path/to/logfile" --top 15 --percentile 99
```

**3. Processing a Large Log and Exporting Results**
Analyze a large log, downsample the data to 5000 points for plotting, and save the output artifacts to a directory named `analysis_results`.
```bash
python perf_analyzer.py "./perf.log" --downsample 5000 --export-dir ./analysis_results
```
This command will create `./analysis_results/perf_timeline.html` and `./analysis_results/perf_summary_barchart.png` without opening any plot windows.

## Understanding the Output

The script produces three main artifacts:

1.  **Console Output**:
    -   **Performance Summary**: A table showing key statistics for `GameLoop` time. The percentiles are useful for understanding worst-case performance.
    -   **Outlier Info**: Reports the calculated time threshold for outliers and the number of frames that exceeded it.

2.  **Interactive Timeline (`*_timeline.html`)**:
    -   A Plotly graph where the x-axis is the frame number (log line number) and the y-axis is time in milliseconds.
    -   The main traces (`GameLoop`, `ServicesSum`, `UnaccountedTime`) are visible by default. Other services can be toggled on/off via the legend.
    -   Red 'x' markers indicate frames identified as outliers.
    -   An orange dotted line shows the calculated outlier threshold.
    -   Hovering over any point provides a detailed breakdown of the top contributors for that specific frame.

3.  **Summary Bar Chart (`*_summary_barchart.png`)**:
    -   A horizontal bar chart ranking the top services by their average execution time.
    -   This chart helps identify services that are consistently expensive, as opposed to services that may cause infrequent, large spikes.