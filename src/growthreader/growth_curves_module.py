#!/usr/bin/env python3
"""Minimal helpers for loading and processing plate-reader data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RAW_SUFFIX = " - Raw Data"
DEFAULT_OD_MIN = 0.01
DEFAULT_OD_MAX = 0.1


@dataclass(frozen=True)
class SynergyH1MeasurementBlock:
    """Represents a single measurement block (e.g., OD or fluorescence)."""

    plate_name: str
    channel_label: str
    dataframe: pd.DataFrame
    measurement_kind: str

    def matches_keywords(self, keywords: Sequence[str]) -> bool:
        """Return True if ``channel_label`` contains any keyword."""
        label = self.channel_label.lower()
        return any(keyword.lower() in label for keyword in keywords)

    def safe_label(self) -> str:
        """Return a filesystem-friendly identifier for naming outputs."""
        safe_plate = _sanitize_label(self.plate_name)
        safe_channel = _sanitize_label(self.channel_label)
        return f"{safe_plate}_{safe_channel}"


def _sanitize_label(label: str) -> str:
    label = str(label)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "measurement"


def _is_well_label(value: object) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip().upper()
    if len(value) < 2:
        return False
    row_char = value[0]
    col_part = value[1:]
    return row_char in "ABCDEFGH" and col_part.isdigit()


def _extract_plate_name(sheet: pd.DataFrame) -> str | None:
    first_col = sheet.iloc[:, 0]
    mask = first_col.astype(str).str.contains("Plate Number", case=False, na=False)
    indices = sheet.index[mask].tolist()
    if not indices:
        return None
    value = sheet.iloc[indices[0], 1]
    return str(value).strip() if pd.notna(value) else None


def _infer_synergyh1_measurement_kind(channel_label: str) -> str:
    label = str(channel_label).lower()
    if "600" in label or "od" in label:
        return "od"
    return "fluorescence"


def _extract_synergyh1_block(
    sheet: pd.DataFrame,
    header_row_idx: int,
    next_header_idx: int | None,
) -> pd.DataFrame | None:
    """Slice a measurement block using the header row definition."""
    header_row = sheet.iloc[header_row_idx]
    valid_columns = header_row[header_row.notna()]
    if "Time" not in valid_columns.values:
        return None
    start_idx = header_row_idx + 1
    end_idx = next_header_idx if next_header_idx is not None else len(sheet)
    block = sheet.iloc[start_idx:end_idx, valid_columns.index].copy()
    block.columns = valid_columns.tolist()
    if block.empty:
        return None

    if "Time" not in block.columns:
        return None

    block = block.dropna(subset=["Time"], how="all")
    well_columns = [col for col in block.columns if _is_well_label(col)]
    if not well_columns:
        return None

    output = block[["Time"] + well_columns].copy()
    output.reset_index(drop=True, inplace=True)
    return output


def load_raw_data(workbook_path: Path) -> Dict[str, pd.DataFrame]:
    """Return every sheet whose name ends with ``RAW_SUFFIX``."""
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"No such workbook: {workbook_path}")

    excel = pd.ExcelFile(workbook_path)
    raw_sheets = [name for name in excel.sheet_names if name.endswith(RAW_SUFFIX)]
    if not raw_sheets:
        raise ValueError(
            f"No sheets ending with '{RAW_SUFFIX}' found in {workbook_path.name}"
        )

    data: Dict[str, pd.DataFrame] = {}
    for sheet in raw_sheets:
        frame = pd.read_excel(excel, sheet_name=sheet, header=1)
        frame = frame.dropna(axis=1, how="all").copy()
        data[sheet] = frame
    return data


def load_synergyh1_measurements(
    workbook_path: Path,
) -> list[SynergyH1MeasurementBlock]:
    """Load all measurement blocks (OD + fluorescence) from a SynergyH1 export."""
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"No such workbook: {workbook_path}")

    sheet = pd.read_excel(workbook_path, sheet_name=0, header=None)
    plate_name = _extract_plate_name(sheet) or "Plate 1"
    first_col = sheet.iloc[:, 0]
    second_col = sheet.iloc[:, 1].astype(str).str.strip()
    header_mask = (
        first_col.isna()
        & (second_col.str.lower() == "time")
        & sheet.iloc[:, 2].notna()
    )
    header_indices = sheet.index[header_mask].tolist()
    if not header_indices:
        raise ValueError(
            f"No measurement blocks detected in {workbook_path.name}. "
            "Expected rows with (NaN, 'Time', <channel label>)."
        )

    measurements: list[SynergyH1MeasurementBlock] = []
    for idx, header_row_idx in enumerate(header_indices):
        next_idx = header_indices[idx + 1] if idx + 1 < len(header_indices) else None
        block = _extract_synergyh1_block(sheet, header_row_idx, next_idx)
        if block is None:
            continue

        channel_label = str(sheet.iloc[header_row_idx, 2]).strip()
        measurement_kind = _infer_synergyh1_measurement_kind(channel_label)

        measurements.append(
            SynergyH1MeasurementBlock(
                plate_name=plate_name,
                channel_label=channel_label,
                dataframe=block,
                measurement_kind=measurement_kind,
            )
        )

    if not measurements:
        raise ValueError(
            f"Found measurement headers in {workbook_path.name} but could not "
            "extract any data tables."
        )
    return measurements


def compute_time_in_hours(time_series: Sequence[pd.Timestamp]) -> np.ndarray:
    """Convert Excel timestamps to a monotonic array measured in hours."""
    deltas = []
    for value in time_series:
        if isinstance(value, pd.Timedelta):
            delta = value
        elif isinstance(value, np.timedelta64):
            delta = pd.to_timedelta(value)
        elif isinstance(value, (pd.Timestamp, datetime)):
            delta = pd.to_timedelta(value.time().isoformat())
        elif isinstance(value, time):
            delta = pd.to_timedelta(value.isoformat())
        else:
            delta = pd.to_timedelta(str(value))
        deltas.append(delta)

    td = pd.to_timedelta(deltas)
    hours = np.asarray(td.total_seconds() / 3600, dtype=float)
    for idx in range(1, hours.size):
        if hours[idx] < hours[idx - 1]:
            hours[idx] += 24
    return hours


def blank_plate_data(frame: pd.DataFrame, n_points_blank: int) -> pd.DataFrame:
    """Subtract the mean of the first ``n_points_blank`` rows from each well."""
    if n_points_blank < 1:
        raise ValueError("n_points_blank must be at least 1")
    if len(frame) < n_points_blank:
        raise ValueError(
            f"Cannot blank using {n_points_blank} points; frame only has {len(frame)} rows"
        )
    if frame.shape[1] < 2:
        raise ValueError("Expected at least one data column in addition to time")

    time_col = frame.columns[0]
    value_cols = list(frame.columns[1:])

    numeric_values = frame[value_cols].apply(pd.to_numeric, errors="coerce")
    empty_cols = [col for col in value_cols if numeric_values[col].dropna().empty]
    if empty_cols:
        numeric_values = numeric_values.drop(columns=empty_cols)
        value_cols = [col for col in value_cols if col not in empty_cols]
        if not value_cols:
            raise ValueError("All well columns are empty; nothing to blank.")
    # Old approach (mean of first N timepoints):
    # blanks = numeric_values.iloc[:n_points_blank].mean(axis=0)

    blanks: Dict[str, float] = {}
    guard_indices: Dict[str, int] = {}
    blank_indices: Dict[str, list[int]] = {}
    for col in value_cols:
        series = numeric_values[col]
        values = series.dropna().nsmallest(n_points_blank)
        if values.empty:
            raise ValueError(
                f"Not enough non-NaN values to compute blank for column '{col}'"
            )
        blanks[col] = float(values.median())
        guard_indices[col] = int(values.index.max())
        blank_indices[col] = [int(idx) for idx in values.index.tolist()]

    blanked = frame[[time_col] + value_cols].copy()
    blanked[value_cols] = numeric_values.sub(pd.Series(blanks), axis=1)
    blanked[time_col] = frame[time_col]
    blanked.attrs["blank_guard_indices"] = guard_indices
    blanked.attrs["blank_indices"] = blank_indices
    return blanked


def _resolve_linear_limits(values: np.ndarray) -> tuple[float, float] | None:
    """Return padded (min, max) bounds for linear axes or None if empty."""
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return None
    vmin = float(finite_vals.min())
    vmax = float(finite_vals.max())
    if vmin == vmax:
        delta = abs(vmin) * 0.1 if vmin != 0 else 1e-3
        vmin -= delta
        vmax += delta
    lower = vmin * 1.2 if vmin < 0 else vmin * 0.8
    upper = vmax * 0.8 if vmax < 0 else vmax * 1.2
    if lower >= upper:
        upper = lower + abs(lower) * 0.1 + 1e-6
    return lower, upper


def _normalize_log_limits(
    y_limits: tuple[float, float] | None,
    log_ylim_range: float,
) -> tuple[float, float] | None:
    """Clamp log-scale limits to positive values using a fallback range."""
    if y_limits is None:
        return None
    lower, upper = y_limits
    if upper <= 0:
        return (max(upper, 1e-12), max(upper * (1 + log_ylim_range), 1e-12))
    if lower <= 0:
        safe_lower = max(upper * log_ylim_range, 1e-12)
        return (safe_lower, upper)
    return y_limits


def _fit_log_od_peak(
    blanked_readings: Sequence[float],
    time_hours: Sequence[float],
    OD_min: float,
    OD_max: float,
    min_start_idx: int | None = None,
) -> tuple[float, float, np.ndarray | None, int | None, int | None]:
    """Perform log2 regression using the peak-descend window selection."""
    if OD_min <= 0 or OD_max <= 0:
        raise ValueError("OD bounds must be positive.")

    time_axis = np.asarray(time_hours, dtype=float)
    blanked_values = np.asarray(blanked_readings, dtype=float)
    if time_axis.shape != blanked_values.shape:
        raise ValueError(
            "Time axis and readings must have the same length "
            f"(got {time_axis.shape} vs {blanked_values.shape})."
        )

    finite_mask = np.isfinite(blanked_values)
    if not np.any(finite_mask):
        return float("nan"), float("nan"), None, None, None
    readings = blanked_values.astype(float, copy=True)
    readings[~finite_mask] = np.nan

    try:
        peak_idx = int(np.nanargmax(readings))
    except ValueError:
        return float("nan"), float("nan"), None, None, None

    search_floor = 0 if min_start_idx is None else int(max(0, min_start_idx))
    max_index = readings.shape[0] - 1
    if max_index < 0 or search_floor > max_index:
        return float("nan"), float("nan"), None, None, None
    peak_idx = min(max(peak_idx, search_floor), max_index)

    def _find_index_below(threshold: float, start_idx: int) -> int | None:
        idx = start_idx
        while idx >= search_floor:
            value = readings[idx]
            if np.isfinite(value) and value < threshold:
                return idx
            idx -= 1
        return None

    upper_idx = _find_index_below(OD_max, peak_idx)
    if upper_idx is None:
        return float("nan"), float("nan"), None, None, None

    lower_idx = None
    idx = upper_idx
    while idx >= search_floor:
        value = readings[idx]
        if np.isfinite(value) and value < OD_min:
            lower_idx = idx + 1
            break
        idx -= 1
    if lower_idx is None:
        lower_idx = search_floor

    if lower_idx >= upper_idx:
        return float("nan"), float("nan"), None, lower_idx, upper_idx

    start_idx = lower_idx
    end_idx = upper_idx + 1

    slice_values = readings[start_idx:end_idx]
    slice_times = time_axis[start_idx:end_idx]
    positive_mask = np.isfinite(slice_values) & (slice_values > 0)
    if np.count_nonzero(positive_mask) < 2:
        return float("nan"), float("nan"), None, start_idx, end_idx

    x_fit = slice_times[positive_mask]
    y_fit = np.log2(slice_values[positive_mask])
    if not np.all(np.isfinite(y_fit)):
        return float("nan"), float("nan"), None, start_idx, end_idx

    slope, intercept, *_ = stats.linregress(x_fit, y_fit)
    return float(slope), float(intercept), x_fit, start_idx, end_idx


def fit_log_od_growth_rate(
    readings: Sequence[float],
    time_hours: Sequence[float],
    OD_min: float = DEFAULT_OD_MIN,
    OD_max: float = DEFAULT_OD_MAX,
) -> float:
    """Fit a log2-growth slope using the peak-descend window heuristic."""
    slope, _intercept, _, _, _ = _fit_log_od_peak(
        blanked_readings=readings,
        time_hours=time_hours,
        OD_min=OD_min,
        OD_max=OD_max,
    )
    return float(slope)


def fit_log_od_growth_rates(
    df_plate: pd.DataFrame,
    time_hours: Sequence[float],
    wells: Iterable[str] | None = None,
    OD_min: float = DEFAULT_OD_MIN,
    OD_max: float = DEFAULT_OD_MAX,
    per_well_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[
    Dict[str, float],
    Dict[str, tuple[float | None, float | None]],
    Dict[str, tuple[int | None, int | None]],
]:
    """
    Run ``fit_log_od_growth_rate`` for every well column in ``df_plate``.

    ``wells`` lets you choose a subset of columns; by default, every column
    except the first (time) column is used.
    """
    time_axis = np.asarray(time_hours, dtype=float)

    if wells is None:
        wells = [col for col in df_plate.columns if col != df_plate.columns[0]]
    wells = list(wells)

    guard_indices = df_plate.attrs.get("blank_guard_indices", {})

    results: Dict[str, float] = {}
    windows: Dict[str, tuple[float | None, float | None]] = {}
    window_indices: Dict[str, tuple[int | None, int | None]] = {}
    for well in wells:
        blanked_readings = df_plate[well]
        if per_well_ranges and well in per_well_ranges:
            od_min, od_max = per_well_ranges[well]
        else:
            od_min, od_max = OD_min, OD_max
        min_start_idx = guard_indices.get(well)
        slope, _intercept, _, start_idx, end_idx = _fit_log_od_peak(
            blanked_readings=blanked_readings,
            time_hours=time_axis,
            OD_min=od_min,
            OD_max=od_max,
            min_start_idx=None if min_start_idx is None else min_start_idx + 1,
        )
        results[well] = float(slope)
        if (
            start_idx is not None
            and end_idx is not None
            and start_idx < end_idx
            and start_idx < len(time_axis)
        ):
            start_time = float(time_axis[start_idx])
            end_time = float(time_axis[min(end_idx - 1, len(time_axis) - 1)])
        else:
            start_time = None
            end_time = None
        windows[well] = (start_time, end_time)
        window_indices[well] = (start_idx, end_idx)
    return results, windows, window_indices


def _well_to_indices(well: str) -> tuple[int, int]:
    """Convert a well label like 'B7' into zero-based (row, column) indices."""
    well = well.strip().upper()
    if len(well) < 2:
        raise ValueError(f"Invalid well label: '{well}'")
    row_char = well[0]
    col_part = well[1:]
    if not row_char.isalpha() or not col_part.isdigit():
        raise ValueError(f"Invalid well label: '{well}'")

    row_idx = ord(row_char) - ord("A")
    if row_idx < 0 or row_idx >= 8:
        raise ValueError(f"Row '{row_char}' is outside the 96-well range (A–H).")

    col_idx = int(col_part) - 1
    if col_idx < 0 or col_idx >= 12:
        raise ValueError(f"Column '{col_part}' is outside the 96-well range (1–12).")

    return row_idx, col_idx


def plot_plate_growth_curves(
    df_plate: pd.DataFrame,
    time_hours: Sequence[float],
    output_path: Path = Path("plots/plate_growth_curves.pdf"),
    wells: Iterable[str] | None = None,
    OD_min: float = 0.01,
    OD_max: float = 0.1,
    per_well_ranges: Mapping[str, tuple[float, float]] | None = None,
    plate_title: str | None = None,
    figsize: tuple[float, float] = (20.0, 12.0),
    y_limits: tuple[float, float] | None = None,
    log_ylim_range: float = 1e-4,
) -> None:
    """
    Plot all well traces with their fitted log-OD lines on an 8×12 grid.
    """
    time_axis = np.asarray(time_hours, dtype=float)

    if wells is None:
        wells = [col for col in df_plate.columns if col != df_plate.columns[0]]
    wells = list(wells)

    if len(wells) > 96:
        raise ValueError(
            f"Expected at most 96 wells for an 8×12 grid, got {len(wells)}."
        )

    guard_indices = df_plate.attrs.get("blank_guard_indices", {})

    resolved_y_limits: tuple[float, float] | None = y_limits
    if resolved_y_limits is None:
        plate_values = np.asarray(df_plate[wells], dtype=float)
        resolved_y_limits = _resolve_linear_limits(plate_values)
    resolved_y_limits = _normalize_log_limits(resolved_y_limits, log_ylim_range)

    fig, axes = plt.subplots(8, 12, figsize=figsize, sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for idx, well in enumerate(wells):
        ax = axes_flat[idx]
        readings = np.asarray(df_plate[well], dtype=float)

        positive_mask = readings > 0
        positive_values = readings[positive_mask]

        if np.any(positive_mask):
            ax.plot(
                time_axis[positive_mask],
                positive_values,
                "o",
                color="black",
                markersize=0.5,
                zorder=2,
            )
        else:
            ax.plot([], [], "o", color="black", markersize=2, zorder=2)

        if per_well_ranges and well in per_well_ranges:
            od_min, od_max = per_well_ranges[well]
        else:
            od_min, od_max = OD_min, OD_max

        guard_idx = guard_indices.get(well)
        slope, intercept, x_fit, start_idx, end_idx = _fit_log_od_peak(
            blanked_readings=readings,
            time_hours=time_axis,
            OD_min=od_min,
            OD_max=od_max,
            min_start_idx=None if guard_idx is None else guard_idx + 1,
        )

        if x_fit is not None and np.isfinite(slope) and np.isfinite(intercept):
            y_line = np.exp2(slope * x_fit + intercept)
            ax.plot(
                x_fit,
                y_line,
                color="red",
                linewidth=3,
                zorder=1,
            )

        ax.set_yscale("log")
        ax.set_xlim(time_axis[0], time_axis[-1])

        if resolved_y_limits is not None:
            ax.set_ylim(*resolved_y_limits)
        else:
            positive_vals = readings[positive_mask]
            if positive_vals.size:
                ymin = positive_vals.min()
                ymax = positive_vals.max()
                lower = max(ymin * 0.8, 1e-4)
                upper = ymax * 1.2
                if lower >= upper:
                    upper = lower * 1.1
                local_limits = _normalize_log_limits((lower, upper), log_ylim_range)
                ax.set_ylim(*local_limits)
            else:
                ax.set_ylim(1e-4, 1)

        if np.isfinite(slope):
            ax.set_title(f"{well} (slope={slope:.3f})", fontsize=6)
        else:
            ax.set_title(f"{well} (slope=nan)", fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(wells), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    if plate_title:
        fig.suptitle(plate_title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_plate_growth_curves_linear(
    df_plate: pd.DataFrame,
    time_hours: Sequence[float],
    output_path: Path = Path("plots/plate_growth_curves_linear.pdf"),
    wells: Iterable[str] | None = None,
    plate_title: str | None = None,
    figsize: tuple[float, float] = (20.0, 12.0),
    y_limits: tuple[float, float] | None = None,
) -> None:
    """
    Plot all well traces on an 8×12 grid using a linear y-axis (no fit lines).
    """
    time_axis = np.asarray(time_hours, dtype=float)

    if wells is None:
        wells = [col for col in df_plate.columns if col != df_plate.columns[0]]
    wells = list(wells)

    if len(wells) > 96:
        raise ValueError(
            f"Expected at most 96 wells for an 8×12 grid, got {len(wells)}."
        )

    if y_limits is None:
        plate_values = np.asarray(df_plate[wells], dtype=float)
        y_limits = _resolve_linear_limits(plate_values)

    fig, axes = plt.subplots(8, 12, figsize=figsize, sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for idx, well in enumerate(wells):
        ax = axes_flat[idx]
        readings = np.asarray(df_plate[well], dtype=float)

        ax.plot(time_axis, readings, "o", color="black", markersize=0.5)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        else:
            ax.set_ylim(-0.1, 0.1)
        ax.set_title(well, fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(wells), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    if plate_title:
        fig.suptitle(plate_title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_growth_rate_heatmap(
    slopes: Mapping[str, float],
    output_path: Path,
    plate_title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (6.0, 4.5),
    annotate: bool = True,
) -> None:
    """
    Render an 8×12 heatmap of growth rates and save it as ``output_path``.
    """
    data = np.full((8, 12), np.nan, dtype=float)
    for well, slope in slopes.items():
        try:
            r, c = _well_to_indices(well)
        except ValueError:
            continue
        data[r, c] = slope

    masked = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels([str(i) for i in range(1, 13)], fontsize=8)
    ax.set_yticks(range(8))
    ax.set_yticklabels([chr(ord("A") + i) for i in range(8)], fontsize=8)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if plate_title:
        ax.set_title(plate_title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Growth rate (slope)", rotation=90, fontsize=9)

    if annotate:
        for r in range(8):
            for c in range(12):
                value = data[r, c]
                if np.isfinite(value):
                    ax.text(
                        c,
                        r,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if im.norm(value) > 0.5 else "black",
                    )

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_growth_rate_heatmaps(
    all_slopes: Mapping[str, Mapping[str, float]],
    output_dir: Path = Path("plots"),
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
) -> None:
    """
    Generate a heatmap PDF for each plate in ``all_slopes``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for plate_name, slopes in all_slopes.items():
        safe_name = (
            plate_name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        output_path = output_dir / f"{safe_name}_growth_rates.pdf"
        plot_growth_rate_heatmap(
            slopes=slopes,
            output_path=output_path,
            plate_title=plate_name,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
        )


__all__ = [
    "RAW_SUFFIX",
    "load_raw_data",
    "compute_time_in_hours",
    "blank_plate_data",
    "fit_log_od_growth_rate",
    "fit_log_od_growth_rates",
    "plot_plate_growth_curves",
    "plot_plate_growth_curves_linear",
    "plot_growth_rate_heatmap",
    "plot_growth_rate_heatmaps",
]
