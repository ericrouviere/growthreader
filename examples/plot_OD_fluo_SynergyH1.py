"""
Utility script for plotting all OD and fluorescence channels from a SynergyH1 run.

Point ``WORKBOOK_PATH`` at a SynergyH1 export (same layout as the sample workbook),
adjust ``BLANK_POINTS``/``PLOTS_DIR`` if needed, and run:

    python plot_synergyh1_od_fluo_script.py

The script writes one linear 8x12 PDF per channel (OD + each fluorescence read).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# SynergyH1 workbook path (relative to this script or absolute).
WORKBOOK_PATH = "SynergyH1_example.xlsx"

# Number of lowest readings per well to average when blanking.
BLANK_POINTS = 5

# Minimum fraction of the log-scale upper y-limit to use when data dips below zero.
LOG_YLIM_RANGE = 1e-4

# Default OD window used for fitting log-growth slopes.
DEFAULT_OD_MIN = 0.01
DEFAULT_OD_MAX = 0.1

# Output directory for all PDFs.
PLOTS_DIR = "plots_SynergyH1_example"

# Destination for the exported growth-rate CSV (set to None to disable).
GROWTH_RATES_CSV = "growth_rates_plotter.csv"

# ---------------------------------------------------------------------------
# Imports and setup
# ---------------------------------------------------------------------------

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure Matplotlib/fontconfig caches live in writable directories.
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
    load_synergyh1_measurements,
    plot_plate_growth_curves,
    plot_plate_growth_curves_linear,
    SynergyH1MeasurementBlock,
)
from growthreader.pipeline_utils import display_well

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


@dataclass
class PlotConfig:
    workbook_path: Path
    blank_points: int = 5
    plots_dir: Path = Path("plots")
    log_ylim_range: float = LOG_YLIM_RANGE
    default_od_min: float = DEFAULT_OD_MIN
    default_od_max: float = DEFAULT_OD_MAX
    growth_rates_csv: Path | None = (
        Path(GROWTH_RATES_CSV) if GROWTH_RATES_CSV is not None else None
    )

    def __post_init__(self) -> None:
        self.workbook_path = Path(self.workbook_path)
        self.plots_dir = Path(self.plots_dir)
        if self.growth_rates_csv is not None:
            self.growth_rates_csv = Path(self.growth_rates_csv)


def _all_well_labels() -> list[str]:
    rows = [chr(ord("A") + idx) for idx in range(8)]
    cols = [str(idx) for idx in range(1, 13)]
    return [f"{row}{col}" for row in rows for col in cols]


def render_block(
    block: SynergyH1MeasurementBlock,
    config: PlotConfig,
    growth_records: list[dict[str, object]],
) -> list[Path]:
    """Blank a measurement block, export plots, and capture growth rates when possible."""
    blanked = blank_plate_data(block.dataframe, n_points_blank=config.blank_points)
    time_hours = compute_time_in_hours(blanked["Time"])
    safe_name = block.safe_label()
    outputs: list[Path] = []
    wells = _all_well_labels()
    missing = [well for well in wells if well not in blanked.columns]
    for well in missing:
        blanked[well] = np.nan
    blanked = blanked[["Time"] + wells]
    plate_title = f"{block.plate_name} - {block.channel_label}"

    linear_path = config.plots_dir / f"{safe_name}_linear.pdf"
    plot_plate_growth_curves_linear(
        blanked,
        time_hours,
        output_path=linear_path,
        plate_title=plate_title,
        wells=wells,
    )
    outputs.append(linear_path)

    log_path = config.plots_dir / f"{safe_name}_log.pdf"
    plot_plate_growth_curves(
        blanked,
        time_hours,
        output_path=log_path,
        plate_title=plate_title,
        wells=wells,
        log_ylim_range=config.log_ylim_range,
    )
    outputs.append(log_path)

    if block.measurement_kind == "od":
        slopes, fit_windows, _ = fit_log_od_growth_rates(
            blanked,
            time_hours,
            OD_min=config.default_od_min,
            OD_max=config.default_od_max,
        )
        for well, slope in slopes.items():
            start_time, end_time = fit_windows.get(well, (float("nan"), float("nan")))
            growth_records.append(
                {
                    "plate": block.plate_name,
                    "channel": block.channel_label,
                    "well": display_well(well),
                    "slope": slope,
                    "OD_min": config.default_od_min,
                    "OD_max": config.default_od_max,
                    "fit_start_time": start_time,
                    "fit_end_time": end_time,
                }
            )

    return outputs


def main(config: PlotConfig) -> None:
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    blocks = load_synergyh1_measurements(config.workbook_path)
    if not blocks:
        raise RuntimeError("No measurement blocks were detected in the workbook.")

    growth_records: list[dict[str, object]] = []

    for block in blocks:
        print(f"Plotting {block.channel_label} ({block.measurement_kind}) ...")
        output_paths = render_block(block, config, growth_records)
        for path in output_paths:
            print(f"  wrote {path}")

    if config.growth_rates_csv is not None and growth_records:
        df = pd.DataFrame(growth_records)
        df.sort_values(["plate", "channel", "well"], inplace=True)
        config.growth_rates_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.growth_rates_csv, index=False)
        print(f"Wrote growth rates to {config.growth_rates_csv}")

    print("Done.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKBOOK_PATH = Path(WORKBOOK_PATH)
if not _WORKBOOK_PATH.is_absolute():
    _WORKBOOK_PATH = _SCRIPT_DIR / _WORKBOOK_PATH

CONFIG = PlotConfig(
    workbook_path=_WORKBOOK_PATH,
    blank_points=BLANK_POINTS,
    plots_dir=Path(PLOTS_DIR),
    log_ylim_range=LOG_YLIM_RANGE,
    default_od_min=DEFAULT_OD_MIN,
    default_od_max=DEFAULT_OD_MAX,
    growth_rates_csv=GROWTH_RATES_CSV,
)

if __name__ == "__main__":
    main(CONFIG)
