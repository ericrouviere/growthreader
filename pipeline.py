"""High-level pipeline utilities for LP600 growth-rate analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

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


def _extract_plate_id(plate_label: str) -> str:
    """Return the trailing integer token from a plate label."""
    plate_label = str(plate_label)
    for token in plate_label.replace("-", " ").split():
        if token.isdigit():
            return token
    return plate_label.strip()


def _plate_token_to_int(plate_label: str) -> int:
    token = _extract_plate_id(plate_label)
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            f"Plate label '{plate_label}' does not contain an integer identifier."
        ) from exc


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


def _required_columns() -> set[str]:
    return {"plate", "well", "slope", "OD_min", "OD_max"}


def _load_or_initialize_ranges(
    config: PipelineConfig, plates: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    csv_path = config.growth_rates_csv
    if csv_path.exists():
        ranges_df = pd.read_csv(csv_path)
        if not _required_columns().issubset(ranges_df.columns):
            raise ValueError(
                f"{csv_path} must contain columns: {', '.join(sorted(_required_columns()))}"
            )
        ranges_df["plate"] = ranges_df["plate"].apply(
            lambda v: _plate_token_to_int(v)
        )
        ranges_df["well"] = ranges_df["well"].apply(_canonical_well)
        if "fit_start_time" not in ranges_df.columns:
            ranges_df["fit_start_time"] = float("nan")
        if "fit_end_time" not in ranges_df.columns:
            ranges_df["fit_end_time"] = float("nan")
        return ranges_df

    rows = []
    for plate_name, df_plate in plates.items():
        wells = [col for col in df_plate.columns if col != df_plate.columns[0]]
        plate_id = _plate_token_to_int(plate_name)
        for well in wells:
            rows.append(
                {
                    "plate": plate_id,
                    "well": _canonical_well(well),
                    "slope": float("nan"),
                    "OD_min": config.default_od_min,
                    "OD_max": config.default_od_max,
                    "fit_start_time": float("nan"),
                    "fit_end_time": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _ensure_plate_rows(
    ranges_df: pd.DataFrame,
    plate_id: int,
    wells: Iterable[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    existing_wells = set(
        ranges_df.loc[ranges_df["plate"] == plate_id, "well"].tolist()
    )
    missing_wells = [w for w in wells if _canonical_well(w) not in existing_wells]
    if not missing_wells:
        return ranges_df

    new_rows = pd.DataFrame(
        {
            "plate": [plate_id] * len(missing_wells),
            "well": [_canonical_well(w) for w in missing_wells],
            "slope": [float("nan")] * len(missing_wells),
            "OD_min": [config.default_od_min] * len(missing_wells),
            "OD_max": [config.default_od_max] * len(missing_wells),
            "fit_start_time": [float("nan")] * len(missing_wells),
            "fit_end_time": [float("nan")] * len(missing_wells),
        }
    )
    return pd.concat([ranges_df, new_rows], ignore_index=True)


def _finalize_ranges_dataframe(ranges_df: pd.DataFrame) -> pd.DataFrame:
    split_values = ranges_df["well"].apply(_split_well)
    ranges_df["well_row"] = split_values.apply(lambda x: x[0])
    ranges_df["well_col"] = split_values.apply(lambda x: x[1])
    ranges_df.sort_values(["plate", "well_row", "well_col"], inplace=True)

    output_df = ranges_df.drop(columns=["well_row", "well_col"]).copy()
    output_df["well"] = output_df["well"].apply(_display_well)
    return output_df


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """
    Execute the configured pipeline and return the dataframe that was written to CSV.
    """
    plates = load_raw_data(config.workbook_path)
    ranges_df = _load_or_initialize_ranges(config, plates)
    per_plate_slopes: Dict[str, Dict[str, float]] = {}

    config.plots_dir.mkdir(parents=True, exist_ok=True)

    for plate_name, df_plate in plates.items():
        print(f"Processing {plate_name} ...")
        blanked_plate = blank_plate_data(df_plate, n_points_blank=config.blank_points)
        time_hours = compute_time_in_hours(blanked_plate["Time"])

        wells = [col for col in blanked_plate.columns if col != blanked_plate.columns[0]]
        blank_indices = blanked_plate.attrs.get("blank_indices", {})
        plate_id = _plate_token_to_int(plate_name)

        ranges_df = _ensure_plate_rows(ranges_df, plate_id, wells, config)
        plate_mask = ranges_df["plate"] == plate_id

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

        for well, slope in slopes.items():
            canonical_well = _canonical_well(well)
            mask = (ranges_df["plate"] == plate_id) & (
                ranges_df["well"] == canonical_well
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
        )
        linear_curve_path = config.plots_dir / f"{safe_plate_name}_growth_curves_linear.pdf"
        plot_plate_growth_curves_linear(
            blanked_plate,
            time_hours,
            output_path=linear_curve_path,
            plate_title=f"Plate {plate_id}",
        )

    output_df = _finalize_ranges_dataframe(ranges_df)
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
