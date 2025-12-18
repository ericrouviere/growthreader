"""Utilities shared by the growth-rate pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .growth_curves_module import (
    plot_plate_growth_curves,
    plot_plate_growth_curves_linear,
)


def extract_plate_id(plate_label: str) -> str:
    """Return the trailing integer token from a plate label."""
    plate_label = str(plate_label)
    for token in plate_label.replace("-", " ").split():
        if token.isdigit():
            return token
    return plate_label.strip()


def plate_token_to_int(plate_label: str) -> int:
    token = extract_plate_id(plate_label)
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            f"Plate label '{plate_label}' does not contain an integer identifier."
        ) from exc


def split_well(well: str) -> tuple[str, int]:
    well = str(well).strip().upper()
    if not well:
        raise ValueError("Empty well label")
    row = well[0]
    col_part = well[1:]
    if not col_part.isdigit():
        raise ValueError(f"Invalid well label: {well}")
    return row, int(col_part)


def canonical_well(well: str) -> str:
    row, col = split_well(well)
    return f"{row}{col}"


def display_well(well: str) -> str:
    row, col = split_well(well)
    return f"{row}{col:02d}"


def required_columns() -> set[str]:
    return {"plate", "well", "slope", "OD_min", "OD_max"}


def load_or_initialize_ranges(
    csv_path: Path,
    plates: Mapping[str, pd.DataFrame],
    default_od_min: float,
    default_od_max: float,
) -> pd.DataFrame:
    if csv_path.exists():
        ranges_df = pd.read_csv(csv_path)
        if not required_columns().issubset(ranges_df.columns):
            required = ", ".join(sorted(required_columns()))
            raise ValueError(f"{csv_path} must contain columns: {required}")
        ranges_df["plate"] = ranges_df["plate"].apply(lambda v: plate_token_to_int(v))
        ranges_df["well"] = ranges_df["well"].apply(canonical_well)
        if "fit_start_time" not in ranges_df.columns:
            ranges_df["fit_start_time"] = float("nan")
        if "fit_end_time" not in ranges_df.columns:
            ranges_df["fit_end_time"] = float("nan")
        return ranges_df

    rows = []
    for plate_name, df_plate in plates.items():
        wells = [col for col in df_plate.columns if col != df_plate.columns[0]]
        plate_id = plate_token_to_int(plate_name)
        for well in wells:
            rows.append(
                {
                    "plate": plate_id,
                    "well": canonical_well(well),
                    "slope": float("nan"),
                    "OD_min": default_od_min,
                    "OD_max": default_od_max,
                    "fit_start_time": float("nan"),
                    "fit_end_time": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def ensure_plate_rows(
    ranges_df: pd.DataFrame,
    plate_id: int,
    wells: Iterable[str],
    default_od_min: float,
    default_od_max: float,
) -> pd.DataFrame:
    existing_wells = set(
        ranges_df.loc[ranges_df["plate"] == plate_id, "well"].tolist()
    )
    missing_wells = [w for w in wells if canonical_well(w) not in existing_wells]
    if not missing_wells:
        return ranges_df

    new_rows = pd.DataFrame(
        {
            "plate": [plate_id] * len(missing_wells),
            "well": [canonical_well(w) for w in missing_wells],
            "slope": [float("nan")] * len(missing_wells),
            "OD_min": [default_od_min] * len(missing_wells),
            "OD_max": [default_od_max] * len(missing_wells),
            "fit_start_time": [float("nan")] * len(missing_wells),
            "fit_end_time": [float("nan")] * len(missing_wells),
        }
    )
    return pd.concat([ranges_df, new_rows], ignore_index=True)


def finalize_ranges_dataframe(ranges_df: pd.DataFrame) -> pd.DataFrame:
    split_values = ranges_df["well"].apply(split_well)
    ranges_df["well_row"] = split_values.apply(lambda x: x[0])
    ranges_df["well_col"] = split_values.apply(lambda x: x[1])
    ranges_df.sort_values(["plate", "well_row", "well_col"], inplace=True)

    output_df = ranges_df.drop(columns=["well_row", "well_col"]).copy()
    output_df["well"] = output_df["well"].apply(display_well)
    return output_df


def update_global_ranges(
    plate_values: np.ndarray,
    pos_min: float | None,
    pos_max: float | None,
    linear_min: float | None,
    linear_max: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return updated global min/max statistics for log and linear plots."""
    positive_vals = plate_values[plate_values > 0]
    if positive_vals.size:
        min_positive = float(np.min(positive_vals))
        max_positive = float(np.max(positive_vals))
        pos_min = min_positive if pos_min is None else min(pos_min, min_positive)
        pos_max = max_positive if pos_max is None else max(pos_max, max_positive)

    finite_vals = plate_values[np.isfinite(plate_values)]
    if finite_vals.size:
        min_linear = float(np.min(finite_vals))
        max_linear = float(np.max(finite_vals))
        linear_min = min_linear if linear_min is None else min(linear_min, min_linear)
        linear_max = max_linear if linear_max is None else max(linear_max, max_linear)

    return pos_min, pos_max, linear_min, linear_max


def resolve_plot_limits(
    pos_min: float | None,
    pos_max: float | None,
    lin_min: float | None,
    lin_max: float | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Convert raw min/max trackers into padded y-axis limits."""
    if pos_min is not None and pos_max is not None:
        lower = max(pos_min * 0.8, 1e-4)
        upper = pos_max * 1.2
        if lower >= upper:
            upper = lower * 1.1
        log_limits: tuple[float, float] | None = (lower, upper)
    else:
        log_limits = None

    if lin_min is not None and lin_max is not None:
        lower_linear = lin_min * 1.2 if lin_min < 0 else lin_min * 0.8
        upper_linear = lin_max * 0.8 if lin_max < 0 else lin_max * 1.2
        if lower_linear >= upper_linear:
            upper_linear = lower_linear + abs(lower_linear) * 0.1 + 1e-6
        linear_limits: tuple[float, float] | None = (lower_linear, upper_linear)
    else:
        linear_limits = None

    return log_limits, linear_limits


def render_plate_plots(
    plots_dir: Path,
    plate_jobs: Sequence[dict[str, object]],
    OD_min: float,
    OD_max: float,
    log_limits: tuple[float, float] | None,
    linear_limits: tuple[float, float] | None,
    log_ylim_range: float = 1e-4,
) -> None:
    """Generate log/linear PDFs for every plate."""
    for job in plate_jobs:
        blanked_plate = job["blanked_plate"]
        time_hours = job["time_hours"]
        plate_id = job["plate_id"]
        per_well_ranges = job["per_well_ranges"]
        safe_plate_name = job["safe_plate_name"]

        curve_path = plots_dir / f"{safe_plate_name}_growth_curves.pdf"
        plot_plate_growth_curves(
            blanked_plate,
            time_hours,
            output_path=curve_path,
            plate_title=f"Plate {plate_id}",
            OD_min=OD_min,
            OD_max=OD_max,
            per_well_ranges=per_well_ranges,
            y_limits=log_limits,
            log_ylim_range=log_ylim_range,
        )

        linear_curve_path = plots_dir / f"{safe_plate_name}_growth_curves_linear.pdf"
        plot_plate_growth_curves_linear(
            blanked_plate,
            time_hours,
            output_path=linear_curve_path,
            plate_title=f"Plate {plate_id}",
            y_limits=linear_limits,
        )


__all__ = [
    "canonical_well",
    "display_well",
    "ensure_plate_rows",
    "finalize_ranges_dataframe",
    "load_or_initialize_ranges",
    "plate_token_to_int",
    "split_well",
    "update_global_ranges",
    "resolve_plot_limits",
    "render_plate_plots",
]
