"""High-level pipeline utilities for LP600 growth-rate analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .growth_curves_module import (
    blank_plate_data,
    compute_time_in_hours,
    fit_log_od_growth_rates,
    load_raw_data,
    plot_growth_rate_heatmaps,
    plot_plate_growth_curves,
    plot_plate_growth_curves_linear,
)
from .pipeline_utils import (
    canonical_well,
    ensure_plate_rows,
    finalize_ranges_dataframe,
    load_or_initialize_ranges,
    plate_token_to_int,
)


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
    """
    Execute the configured pipeline and return the dataframe that was written to CSV.
    """
    # Load plate sheets and a baseline ranges table (from disk if it already exists).
    plates = load_raw_data(config.workbook_path)
    ranges_df = load_or_initialize_ranges(
        config.growth_rates_csv,
        plates,
        config.default_od_min,
        config.default_od_max,
    )
    per_plate_slopes: Dict[str, Dict[str, float]] = {}
    # We postpone plotting until we learn the global y-limits, so stash per-plate data.
    plate_plot_jobs = []
    # Running min/max trackers for log and linear scales (across every plate/well).
    global_pos_min: float | None = None
    global_pos_max: float | None = None
    global_linear_min: float | None = None
    global_linear_max: float | None = None

    config.plots_dir.mkdir(parents=True, exist_ok=True)

    for plate_name, df_plate in plates.items():
        print(f"Processing {plate_name} ...")
        # Blanking removes per-well offsets and attaches metadata we need for fitting.
        blanked_plate = blank_plate_data(df_plate, n_points_blank=config.blank_points)
        time_hours = compute_time_in_hours(blanked_plate["Time"])

        wells = [col for col in blanked_plate.columns if col != blanked_plate.columns[0]]
        blank_indices = blanked_plate.attrs.get("blank_indices", {})
        plate_id = plate_token_to_int(plate_name)

        ranges_df = ensure_plate_rows(
            ranges_df,
            plate_id,
            wells,
            config.default_od_min,
            config.default_od_max,
        )
        plate_mask = ranges_df["plate"] == plate_id

        # Build the user-editable OD ranges for this plate.
        per_well_ranges = {
            row["well"]: (float(row["OD_min"]), float(row["OD_max"]))
            for _, row in ranges_df.loc[plate_mask].iterrows()
        }

        slopes, fit_windows, fit_window_indices = fit_log_od_growth_rates(
            blanked_plate,
            time_hours,
            OD_min=config.default_od_min,
            OD_max=config.default_od_max,
            window=config.window_size,
            per_well_ranges=per_well_ranges,
        )
        per_plate_slopes[plate_name] = slopes

        # Persist slope + fit window metadata back into the ranges table.
        for well, slope in slopes.items():
            canonical = canonical_well(well)
            mask = (ranges_df["plate"] == plate_id) & (
                ranges_df["well"] == canonical
            )
            ranges_df.loc[mask, "slope"] = slope
            start_time, end_time = fit_windows.get(well, (float("nan"), float("nan")))
            ranges_df.loc[mask, "fit_start_time"] = start_time
            ranges_df.loc[mask, "fit_end_time"] = end_time

        ranges_df = ranges_df.dropna(subset=["slope", "well"]).drop_duplicates(
            subset=["plate", "well"], keep="first"
        )

        safe_plate_name = (
            plate_name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        plate_values = np.asarray(blanked_plate[wells], dtype=float)
        positive_vals = plate_values[plate_values > 0]
        if positive_vals.size:
            min_positive = float(np.min(positive_vals))
            max_positive = float(np.max(positive_vals))
            global_pos_min = (
                min_positive
                if global_pos_min is None
                else min(global_pos_min, min_positive)
            )
            global_pos_max = (
                max_positive
                if global_pos_max is None
                else max(global_pos_max, max_positive)
            )
        finite_vals = plate_values[np.isfinite(plate_values)]
        if finite_vals.size:
            min_linear = float(np.min(finite_vals))
            max_linear = float(np.max(finite_vals))
            global_linear_min = (
                min_linear
                if global_linear_min is None
                else min(global_linear_min, min_linear)
            )
            global_linear_max = (
                max_linear
                if global_linear_max is None
                else max(global_linear_max, max_linear)
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

    # Convert the raw min/max values into padded y-limits for plotting.
    if global_pos_min is not None and global_pos_max is not None:
        lower = max(global_pos_min * 0.8, 1e-4)
        upper = global_pos_max * 1.2
        if lower >= upper:
            upper = lower * 1.1
        log_y_limits = (lower, upper)
    else:
        log_y_limits = None

    if global_linear_min is not None and global_linear_max is not None:
        lower_linear = (
            global_linear_min * 1.2 if global_linear_min < 0 else global_linear_min * 0.8
        )
        upper_linear = (
            global_linear_max * 0.8
            if global_linear_max < 0
            else global_linear_max * 1.2
        )
        if lower_linear >= upper_linear:
            upper_linear = lower_linear + abs(lower_linear) * 0.1 + 1e-6
        linear_y_limits = (lower_linear, upper_linear)
    else:
        linear_y_limits = None

    # Finally render plots for each plate now that global limits are known.
    for job in plate_plot_jobs:
        blanked_plate = job["blanked_plate"]
        time_hours = job["time_hours"]
        plate_id = job["plate_id"]
        per_well_ranges = job["per_well_ranges"]
        safe_plate_name = job["safe_plate_name"]
        curve_path = config.plots_dir / f"{safe_plate_name}_growth_curves.pdf"
        plot_plate_growth_curves(
            blanked_plate,
            time_hours,
            output_path=curve_path,
            plate_title=f"Plate {plate_id}",
            OD_min=config.default_od_min,
            OD_max=config.default_od_max,
            window=config.window_size,
            per_well_ranges=per_well_ranges,
            y_limits=log_y_limits,
        )
        linear_curve_path = (
            config.plots_dir / f"{safe_plate_name}_growth_curves_linear.pdf"
        )
        plot_plate_growth_curves_linear(
            blanked_plate,
            time_hours,
            output_path=linear_curve_path,
            plate_title=f"Plate {plate_id}",
            y_limits=linear_y_limits,
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

    print("Done.")
    return output_df


__all__ = ["PipelineConfig", "run_pipeline"]
