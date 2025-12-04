"""
Integration script that generates deterministic LP600 growth curves, runs the
growthreader pipeline, and verifies that the inferred slopes match the
simulated ground truth.
"""

from __future__ import annotations

import math
import os
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

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
    display_well,
    ensure_plate_rows,
    finalize_ranges_dataframe,
    load_or_initialize_ranges,
    plate_token_to_int,
    render_plate_plots,
    resolve_plot_limits,
    update_global_ranges,
)

#############################################################
################### Simulation parameters ###################
#############################################################

# Where to write the Excel workbook.
OUTPUT_PATH = Path(__file__).resolve().parent / "simulated_growth_curves.xlsx"

# Total experiment duration (hours) and sampling interval (minutes).
TOTAL_HOURS = 24
INTERVAL_MINUTES = 10

# Initial OD (baseline) and amplitude for the exponential phase.
BASELINE_OD = 0.7
AMPLITUDE = 0.001

# Once the OD reaches this threshold it saturates at SATURATION_OD.
SATURATION_OD = 1.0

# Slopes increase row-major by this step, starting at SLOPE_OFFSET.
SLOPE_STEP = 0.01



#############################################################
################### Analysis Pipeline parameters ############
#############################################################

WORKBOOK_PATH = OUTPUT_PATH
BLANK_POINTS = 1 # set to 1
GROWTH_RATES_CSV = Path(__file__).resolve().parent / "growth_rates_test.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "test_plots"
DEFAULT_OD_MIN = 0.01
DEFAULT_OD_MAX = 0.9
WINDOW_SIZE = 3
HEATMAP_CMAP = "viridis"
HEATMAP_VMIN = None
HEATMAP_VMAX = None
HEATMAP_ANNOTATE = True
RELATIVE_TOLERANCE = 0.1
ZERO_ABS_TOLERANCE = 0.02

# Ensure Matplotlib and fontconfig use writable cache dirs even if HOME is read-only.
_MPL_CACHE = Path(".matplotlib_cache")
_XDG_CACHE = Path(".cache")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE.resolve()))
(_XDG_CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
_MPL_CACHE.mkdir(parents=True, exist_ok=True)


#############################################################
################### Simulation helpers ###################
#############################################################

def _well_labels() -> list[str]:
    rows = [chr(ord("A") + i) for i in range(8)]
    cols = [str(j) for j in range(1, 13)]
    return [f"{row}{col}" for row, col in product(rows, cols)]


def _display_well_labels() -> list[str]:
    return [display_well(label) for label in _well_labels()]


def _hours_series() -> tuple[np.ndarray, pd.Series]:
    minutes = np.arange(0, TOTAL_HOURS * 60 + INTERVAL_MINUTES, INTERVAL_MINUTES)
    hours = minutes / 60.0
    time_strings = pd.to_timedelta(minutes, unit="m").astype(str)
    return hours, time_strings


def _grow_trace(hours: np.ndarray, slope: float) -> np.ndarray:
    values = BASELINE_OD + AMPLITUDE * np.exp(slope * hours)
    values = np.where(values >= SATURATION_OD + BASELINE_OD, SATURATION_OD + BASELINE_OD, values)
    return values


def build_plate_dataframe() -> pd.DataFrame:
    hours, time_strings = _hours_series()
    data = {"Time": time_strings}
    for idx, well in enumerate(_well_labels()):
        slope = idx * SLOPE_STEP
        data[well] = _grow_trace(hours, slope)
    return pd.DataFrame(data)


def write_workbook(path: Path) -> Path:
    df = build_plate_dataframe()
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(
            writer,
            sheet_name="Plate 1 - Raw Data",
            index=False,
            startrow=1,
        )
    return path

#############################################################
################### Pipeline helpers ###################
#############################################################

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
        self.workbook_path = Path(self.workbook_path)
        self.growth_rates_csv = Path(self.growth_rates_csv)
        self.plots_dir = Path(self.plots_dir)


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """Execute the configured pipeline and return the dataframe written to CSV."""
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
        blanked_plate = blank_plate_data(df_plate, n_points_blank=config.blank_points)
        time_hours = compute_time_in_hours(blanked_plate["Time"])

        wells = [col for col in blanked_plate.columns if col != blanked_plate.columns[0]]
        plate_id = plate_token_to_int(plate_name)

        ranges_df = ensure_plate_rows(
            ranges_df,
            plate_id,
            wells,
            config.default_od_min,
            config.default_od_max,
        )
        plate_mask = ranges_df["plate"] == plate_id

        per_well_ranges = {
            row["well"]: (float(row["OD_min"]), float(row["OD_max"]))
            for _, row in ranges_df.loc[plate_mask].iterrows()
        }

        slopes, fit_windows, _ = fit_log_od_growth_rates(
            blanked_plate,
            time_hours,
            OD_min=config.default_od_min,
            OD_max=config.default_od_max,
            window=config.window_size,
            per_well_ranges=per_well_ranges,
        )
        per_plate_slopes[plate_name] = slopes

        for well, slope in slopes.items():
            canonical = canonical_well(well)
            mask = (ranges_df["plate"] == plate_id) & (ranges_df["well"] == canonical)
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

    return output_df

#############################################################
################### Comparison helpers ###################
#############################################################

def expected_slopes_by_well() -> Dict[str, float]:
    """Compute the analytic log2 slope for each simulated well."""
    conversion = 1.0 / math.log(2.0)
    labels = _well_labels()
    return {
        display_well(labels[idx]): (idx * SLOPE_STEP) * conversion
        for idx in range(len(labels))
    }


def compare_growth_rates(
    measured_df: pd.DataFrame,
    expected: Mapping[str, float],
    rel_tol: float = RELATIVE_TOLERANCE,
    zero_tol: float = ZERO_ABS_TOLERANCE,
) -> bool:
    """
    Print relative differences between measured and expected slopes.

    Returns True when every well falls within the tolerance.
    """
    measured_map = {}
    for _, row in measured_df.iterrows():
        well = row["well"]
        if not isinstance(well, str):
            continue
        measured_map[well] = float(row["slope"]) if pd.notna(row["slope"]) else float("nan")

    print("Well\t\tExpected\t\tMeasured\t\tRelDiff\t\tStatus")
    all_passed = True
    any_measured = False
    for well in _display_well_labels():
        expected_slope = expected[well]
        measured = measured_map.get(well, float("nan"))

        if math.isnan(measured):
            rel_diff = float("nan")
            status = "NA"
            passed = None
        elif abs(expected_slope) < 1e-12:
            rel_diff = abs(measured - expected_slope)
            passed = rel_diff <= zero_tol
            status = "PASS" if passed else "FAIL"
        else:
            rel_diff = abs(measured - expected_slope) / abs(expected_slope)
            passed = rel_diff <= rel_tol
            status = "PASS" if passed else "FAIL"

        print(f"{well}\t\t{expected_slope:.4f}\t\t{measured:.4f}\t\t{rel_diff:.4f}\t\t{status}")
        if passed is not None:
            any_measured = True
            all_passed = all_passed and passed
    return all_passed and any_measured

#############################################################
################### Main entry point ###################
#############################################################

def main() -> None:
    """Generate data, run the pipeline, and validate inferred growth rates."""
    print("Generating synthetic workbook ...")
    write_workbook(WORKBOOK_PATH)
    config = PipelineConfig(
        workbook_path=WORKBOOK_PATH,
        blank_points=BLANK_POINTS,
        growth_rates_csv=GROWTH_RATES_CSV,
        plots_dir=PLOTS_DIR,
        default_od_min=DEFAULT_OD_MIN,
        default_od_max=DEFAULT_OD_MAX,
        window_size=WINDOW_SIZE,
        heatmap_cmap=HEATMAP_CMAP,
        heatmap_vmin=HEATMAP_VMIN,
        heatmap_vmax=HEATMAP_VMAX,
        heatmap_annotate=HEATMAP_ANNOTATE,
    )
    print("Running growthreader pipeline ...")
    measured_df = run_pipeline(config)
    expected = expected_slopes_by_well()
    print("Comparing measured slopes to simulated ground truth ...")
    passed = compare_growth_rates(measured_df, expected)
    if passed:
        print("Simulation check: PASS")
    else:
        print("Simulation check: FAIL")


if __name__ == "__main__":
    main()
