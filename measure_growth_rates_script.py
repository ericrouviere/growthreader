#!/usr/bin/env python3
"""
Configured pipeline for processing BioTek LP600 growth data.

Edit the constants in the "Configuration" section below to point at your
workbook, choose blanking parameters, fit bounds, and output locations. Then
run the script with ``python measure_growth_rates_script.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from growth_curves_module import (
    blank_plate_data,
    compute_time_in_hours,
    fit_log_od_growth_rates,
    load_raw_data,
    plot_growth_rate_heatmaps,
    plot_plate_growth_curves,
    plot_plate_growth_curves_linear,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKBOOK_PATH = Path("LP600_example.xlsx")
BLANK_POINTS = 10

GROWTH_RATES_CSV = Path("growth_rates.csv")
PLOTS_DIR = Path("plots")

DEFAULT_OD_MIN = 0.01
DEFAULT_OD_MAX = 0.1
WINDOW_SIZE = 3

# Heatmap appearance
HEATMAP_CMAP = "viridis"
HEATMAP_VMIN = None  # set to a float if you prefer fixed colour limits
HEATMAP_VMAX = None
HEATMAP_ANNOTATE = True

def _extract_plate_id(plate_label: str) -> str:
    """Return the trailing integer token from a plate label (e.g. 'Plate 2 - Raw Data')."""
    plate_label = str(plate_label)
    for token in plate_label.replace("-", " ").split():
        if token.isdigit():
            return token
    return plate_label.strip()


def _split_well(well: str) -> tuple[str, int]:
    well = str(well).strip().upper()
    if not well:
        raise ValueError("Empty well label")
    row = well[0]
    col_part = well[1:]
    if not col_part.isdigit():
        raise ValueError(f"Invalid well label: {well}")
    return row, int(col_part)


def _canonical_well(well: str) -> str:
    row, col = _split_well(well)
    return f"{row}{col}"


def _display_well(well: str) -> str:
    row, col = _split_well(well)
    return f"{row}{col:02d}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    plates = load_raw_data(WORKBOOK_PATH)

    per_plate_slopes: Dict[str, Dict[str, float]] = {}

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if GROWTH_RATES_CSV.exists():
        ranges_df = pd.read_csv(GROWTH_RATES_CSV)
        required_cols = {"plate", "well", "slope", "OD_min", "OD_max"}
        if not required_cols.issubset(ranges_df.columns):
            raise ValueError(
                f"{GROWTH_RATES_CSV} must contain columns: {', '.join(sorted(required_cols))}"
            )
        ranges_df["plate"] = ranges_df["plate"].apply(lambda v: int(_extract_plate_id(v)))
        ranges_df["well"] = ranges_df["well"].apply(_canonical_well)
        if "fit_start_time" not in ranges_df.columns:
            ranges_df["fit_start_time"] = float("nan")
        if "fit_end_time" not in ranges_df.columns:
            ranges_df["fit_end_time"] = float("nan")
    else:
        rows = []
        for plate_name, df_plate in plates.items():
            wells = [col for col in df_plate.columns if col != df_plate.columns[0]]
            plate_id = int(_extract_plate_id(plate_name))
            for well in wells:
                rows.append(
                    {
                        "plate": plate_id,
                        "well": _canonical_well(well),
                        "slope": float("nan"),
                        "OD_min": DEFAULT_OD_MIN,
                        "OD_max": DEFAULT_OD_MAX,
                        "fit_start_time": float("nan"),
                        "fit_end_time": float("nan"),
                    }
                )
        ranges_df = pd.DataFrame(rows)

    for plate_name, df_plate in plates.items():
        print(f"Processing {plate_name} ...")

        blanked_plate = blank_plate_data(df_plate, n_points_blank=BLANK_POINTS)
        time_hours = compute_time_in_hours(blanked_plate["Time"])

        wells = [col for col in blanked_plate.columns if col != blanked_plate.columns[0]]
        blank_indices = blanked_plate.attrs.get("blank_indices", {})

        plate_id = int(_extract_plate_id(plate_name))

        plate_mask = ranges_df["plate"] == plate_id
        plate_rows = ranges_df.loc[plate_mask]
        existing_wells = set(plate_rows["well"])
        missing_wells = [w for w in wells if w not in existing_wells]
        if missing_wells:
            new_rows = pd.DataFrame(
                {
                    "plate": [plate_id] * len(missing_wells),
                    "well": [_canonical_well(w) for w in missing_wells],
                    "slope": [float("nan")] * len(missing_wells),
                    "OD_min": [DEFAULT_OD_MIN] * len(missing_wells),
                    "OD_max": [DEFAULT_OD_MAX] * len(missing_wells),
                    "fit_start_time": [float("nan")] * len(missing_wells),
                    "fit_end_time": [float("nan")] * len(missing_wells),
                }
            )
            ranges_df = pd.concat([ranges_df, new_rows], ignore_index=True)
            plate_rows = pd.concat([plate_rows, new_rows], ignore_index=True)
            plate_mask = ranges_df["plate"] == plate_id

        per_well_ranges = {
            row["well"]: (float(row["OD_min"]), float(row["OD_max"]))
            for _, row in ranges_df.loc[plate_mask].iterrows()
        }

        slopes, fit_windows, fit_window_indices = fit_log_od_growth_rates(
            blanked_plate,
            time_hours,
            OD_min=DEFAULT_OD_MIN,
            OD_max=DEFAULT_OD_MAX,
            window=WINDOW_SIZE,
            per_well_ranges=per_well_ranges,
        )
        per_plate_slopes[plate_name] = slopes

        for well, slope in slopes.items():
            canonical_well = _canonical_well(well)
            mask = (ranges_df["plate"] == plate_id) & (ranges_df["well"] == canonical_well)
            ranges_df.loc[mask, "slope"] = slope
            start_time, end_time = fit_windows.get(well, (float("nan"), float("nan")))
            ranges_df.loc[mask, "fit_start_time"] = start_time
            ranges_df.loc[mask, "fit_end_time"] = end_time

            if plate_id == 4 and _display_well(canonical_well) == "H01":
                blank_idxs = blank_indices.get(well, [])
                blank_times = [
                    float(time_hours[i])
                    for i in blank_idxs
                    if isinstance(i, int) and 0 <= i < len(time_hours)
                ]
                start_idx, end_idx = fit_window_indices.get(well, (None, None))
                if (
                    start_idx is not None
                    and end_idx is not None
                    and 0 <= start_idx < len(time_hours)
                    and start_idx < end_idx
                ):
                    fitted_times = [float(t) for t in time_hours[start_idx:end_idx]]
                    fitted_ods = (
                        blanked_plate[well].iloc[start_idx:end_idx].astype(float).tolist()
                    )
                else:
                    fitted_times = []
                    fitted_ods = []
                threshold_time = (
                    float(time_hours[start_idx])
                    if start_idx is not None
                    and 0 <= start_idx < len(time_hours)
                    else None
                )
                print("[DEBUG] Plate 4 - H01 blank times:", blank_times)
                print("[DEBUG] Plate 4 - H01 OD_min threshold time:", threshold_time)
                print("[DEBUG] Plate 4 - H01 fitted time points:", fitted_times)
                print("[DEBUG] Plate 4 - H01 fitted blanked ODs:", fitted_ods)

        ranges_df = ranges_df.dropna(subset=["slope", "well"]).drop_duplicates(
            subset=["plate", "well"], keep="first"
        )

        safe_plate_name = (
            plate_name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        curve_path = PLOTS_DIR / f"{safe_plate_name}_growth_curves.pdf"
        plot_plate_growth_curves(
            blanked_plate,
            time_hours,
            output_path=curve_path,
            plate_title=f"Plate {plate_id}",
            OD_min=DEFAULT_OD_MIN,
            OD_max=DEFAULT_OD_MAX,
            window=WINDOW_SIZE,
            per_well_ranges=per_well_ranges,
        )
        linear_curve_path = PLOTS_DIR / f"{safe_plate_name}_growth_curves_linear.pdf"
        plot_plate_growth_curves_linear(
            blanked_plate,
            time_hours,
            output_path=linear_curve_path,
            plate_title=f"Plate {plate_id}",
        )

    split_values = ranges_df["well"].apply(_split_well)
    ranges_df["well_row"] = split_values.apply(lambda x: x[0])
    ranges_df["well_col"] = split_values.apply(lambda x: x[1])

    ranges_df.sort_values(["plate", "well_row", "well_col"], inplace=True)

    output_df = ranges_df.drop(columns=["well_row", "well_col"]).copy()
    output_df["well"] = output_df["well"].apply(_display_well)
    output_df.to_csv(GROWTH_RATES_CSV, index=False)
    print(f"Wrote growth rates (with OD ranges) to {GROWTH_RATES_CSV}")

    if per_plate_slopes:
        plot_growth_rate_heatmaps(
            per_plate_slopes,
            output_dir=PLOTS_DIR,
            cmap=HEATMAP_CMAP,
            vmin=HEATMAP_VMIN,
            vmax=HEATMAP_VMAX,
            annotate=HEATMAP_ANNOTATE,
        )
        print(f"Wrote plots to {PLOTS_DIR}")

    print("Done.")


if __name__ == "__main__":
    run_pipeline()
