"""
Configured pipeline for processing Agilent LP600 growth data.

Edit the values in the parameter block below to point at your workbook, choose
blanking parameters, fit bounds, and output locations. Then run the script with
``python measure_growth_rates_script.py``.

This version bundles the full pipeline implementation so the script can be
copied next to new datasets without dragging the entire repository around.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Excel workbook exported from the Agilent LP600 (relative or absolute path).
# This can be replaced with an absolute path when the script lives elsewhere.
WORKBOOK_PATH = "LP600_example.xlsx"

# How many of the lowest OD readings per well to average for blanking.
BLANK_POINTS = 10

# Where to write/read the table of per-well slopes and OD bounds.
GROWTH_RATES_CSV = "growth_rates.csv"

# Directory for all generated PDF plots.
PLOTS_DIR = "plots"

# Default lower bound (OD) allowed for the log2 fit unless overridden per well.
DEFAULT_OD_MIN = 0.01

# Default upper bound (OD) that terminates the fit window unless overridden.
DEFAULT_OD_MAX = 0.1

# Minimum consecutive time points required above OD_min before fitting begins.
WINDOW_SIZE = 3

# Matplotlib colormap to use for growth-rate heatmaps.
HEATMAP_CMAP = "viridis"

# Optional fixed colorbar limits; leave None to auto-scale per plate.
HEATMAP_VMIN = None
HEATMAP_VMAX = None

# Annotate each heatmap cell with the numeric slope.
HEATMAP_ANNOTATE = True

# ---------------------------------------------------------------------------
# Imports and setup
# ---------------------------------------------------------------------------

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Ensure Matplotlib and fontconfig use writable cache directories even if HOME is read-only.
# This avoids issues when the script runs in sandboxed environments (e.g., HPC nodes).
_MPL_CACHE = Path(".matplotlib_cache")
_XDG_CACHE = Path(".cache")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE.resolve()))
(_XDG_CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
_MPL_CACHE.mkdir(parents=True, exist_ok=True)

from growthreader.growth_curves_module import (
    blank_plate_data,
    compute_time_in_hours,
    fit_log_od_growth_rates,
    load_raw_data,
    plot_growth_rate_heatmaps,
    plot_plate_growth_curves,
    plot_plate_growth_curves_linear,
)
from growthreader.pipeline_utils import (
    canonical_well,
    ensure_plate_rows,
    finalize_ranges_dataframe,
    load_or_initialize_ranges,
    plate_token_to_int,
    render_plate_plots,
    resolve_plot_limits,
    update_global_ranges,
)

# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """User-tunable knobs for the growth-rate pipeline."""

    workbook_path: Path
    blank_points: int = 10
    growth_rates_csv: Path = Path("growth_rates.csv")
    plots_dir: Path = Path("plots")
    default_od_min: float = 0.01
    default_od_max: float = 0.1
    window_size: int = 3
    heatmap_cmap: str = "viridis"
    heatmap_vmin: float | None = None
    heatmap_vmax: float | None = None
    heatmap_annotate: bool = True

    def __post_init__(self) -> None:
        # Normalize paths so users can pass strings/Paths interchangeably.
        self.workbook_path = Path(self.workbook_path)
        self.growth_rates_csv = Path(self.growth_rates_csv)
        self.plots_dir = Path(self.plots_dir)


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """Execute the configured pipeline and return the dataframe written to CSV."""
    # Load all plate worksheets and either read the previous ranges CSV or create defaults.
    plates = load_raw_data(config.workbook_path)
    ranges_df = load_or_initialize_ranges(
        config.growth_rates_csv,
        plates,
        config.default_od_min,
        config.default_od_max,
    )
    per_plate_slopes: Dict[str, Dict[str, float]] = {}
    plate_plot_jobs: list[dict[str, object]] = []
    global_pos_min: float | None = None
    global_pos_max: float | None = None
    global_linear_min: float | None = None
    global_linear_max: float | None = None

    config.plots_dir.mkdir(parents=True, exist_ok=True)

    for plate_name, df_plate in plates.items():
        print(f"Processing {plate_name} ...")
        # Blank the raw readings and convert the time column to hours.
        blanked_plate = blank_plate_data(df_plate, n_points_blank=config.blank_points)
        time_hours = compute_time_in_hours(blanked_plate["Time"])

        wells = [col for col in blanked_plate.columns if col != blanked_plate.columns[0]]
        plate_id = plate_token_to_int(plate_name)

        # Ensure the ranges table has rows for every well on this plate.
        ranges_df = ensure_plate_rows(
            ranges_df,
            plate_id,
            wells,
            config.default_od_min,
            config.default_od_max,
        )
        plate_mask = ranges_df["plate"] == plate_id

        # Build the per-well OD fit bounds.
        per_well_ranges = {
            row["well"]: (float(row["OD_min"]), float(row["OD_max"]))
            for _, row in ranges_df.loc[plate_mask].iterrows()
        }

        # Fit log-OD slopes for every well.
        slopes, fit_windows, window_indices = fit_log_od_growth_rates(
            blanked_plate,
            time_hours,
            OD_min=config.default_od_min,
            OD_max=config.default_od_max,
            window=config.window_size,
            per_well_ranges=per_well_ranges,
        )
        per_plate_slopes[plate_name] = slopes

        # Record slopes and fit windows in the ranges dataframe.
        for well, slope in slopes.items():
            canonical = canonical_well(well)
            mask = (ranges_df["plate"] == plate_id) & (
                ranges_df["well"] == canonical
            )
            ranges_df.loc[mask, "slope"] = slope
            start_time, end_time = fit_windows.get(well, (float("nan"), float("nan")))
            ranges_df.loc[mask, "fit_start_time"] = start_time
            ranges_df.loc[mask, "fit_end_time"] = end_time

        ranges_df = ranges_df.dropna(subset=["well"]).drop_duplicates(
            subset=["plate", "well"], keep="first"
        )

        safe_plate_name = (
            plate_name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        plate_values = np.asarray(blanked_plate[wells], dtype=float)
        (
            global_pos_min,
            global_pos_max,
            global_linear_min,
            global_linear_max,
        ) = update_global_ranges(
            plate_values,
            global_pos_min,
            global_pos_max,
            global_linear_min,
            global_linear_max,
        )

        plate_plot_jobs.append(
            {
                "blanked_plate": blanked_plate,
                "time_hours": time_hours,
                "plate_id": plate_id,
                "safe_plate_name": safe_plate_name,
                "per_well_ranges": per_well_ranges,
            }
        )

    log_y_limits, linear_y_limits = resolve_plot_limits(
        global_pos_min, global_pos_max, global_linear_min, global_linear_max
    )

    render_plate_plots(
        config.plots_dir,
        plate_plot_jobs,
        config.default_od_min,
        config.default_od_max,
        config.window_size,
        log_y_limits,
        linear_y_limits,
    )

    # Write the ranges table and any heatmaps to disk.
    output_df = finalize_ranges_dataframe(ranges_df)
    output_df.to_csv(config.growth_rates_csv, index=False)
    print(f"Wrote growth rates (with OD ranges) to {config.growth_rates_csv}")

    if per_plate_slopes:
        plot_growth_rate_heatmaps(
            per_plate_slopes,
            output_dir=config.plots_dir,
            cmap=config.heatmap_cmap,
            vmin=config.heatmap_vmin,
            vmax=config.heatmap_vmax,
            annotate=config.heatmap_annotate,
        )
        print(f"Wrote plots to {config.plots_dir}")

    print("Done.")
    return output_df

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Derive default workbook path relative to the script so it keeps working when copied elsewhere.
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKBOOK_PATH = Path(WORKBOOK_PATH)
if not _WORKBOOK_PATH.is_absolute():
    _WORKBOOK_PATH = _SCRIPT_DIR / _WORKBOOK_PATH

CONFIG = PipelineConfig(
    workbook_path=_WORKBOOK_PATH,
    blank_points=BLANK_POINTS,
    growth_rates_csv=Path(GROWTH_RATES_CSV),
    plots_dir=Path(PLOTS_DIR),
    default_od_min=DEFAULT_OD_MIN,
    default_od_max=DEFAULT_OD_MAX,
    window_size=WINDOW_SIZE,
    heatmap_cmap=HEATMAP_CMAP,
    heatmap_vmin=HEATMAP_VMIN,
    heatmap_vmax=HEATMAP_VMAX,
    heatmap_annotate=HEATMAP_ANNOTATE,
)


def main() -> None:
    """Execute the configured growth-rate pipeline."""
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()
