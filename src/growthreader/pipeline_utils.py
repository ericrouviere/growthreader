"""Utilities shared by the growth-rate pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


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


__all__ = [
    "canonical_well",
    "display_well",
    "ensure_plate_rows",
    "finalize_ranges_dataframe",
    "load_or_initialize_ranges",
    "plate_token_to_int",
    "split_well",
]
