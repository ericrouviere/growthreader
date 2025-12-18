"""
Utility script for plotting all OD and fluorescence channels from a BioTek run.

Point ``WORKBOOK_PATH`` at a BioTek export (same layout as the sample workbook),
adjust ``BLANK_POINTS``/``PLOTS_DIR`` if needed, and run:

    python plot_biotek_od_fluo_script.py

The script writes one linear 8x12 PDF per channel (OD + each fluorescence read).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# BioTek workbook path (relative to this script or absolute).
WORKBOOK_PATH = "Biotek_fluorescence_example.xlsx"

# Number of lowest readings per well to average when blanking.
BLANK_POINTS = 5

# Output directory for all PDFs.
PLOTS_DIR = "plots"

# ---------------------------------------------------------------------------
# Imports and setup
# ---------------------------------------------------------------------------

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure Matplotlib/fontconfig caches live in writable directories.
_MPL_CACHE = Path(".matplotlib_cache")
_XDG_CACHE = Path(".cache")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE.resolve()))
(_XDG_CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
_MPL_CACHE.mkdir(parents=True, exist_ok=True)

from growthreader.growth_curves_module import (
    BiotekMeasurementBlock,
    blank_plate_data,
    compute_time_in_hours,
    load_biotek_measurements,
    plot_plate_growth_curves_linear,
)

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


@dataclass
class PlotConfig:
    workbook_path: Path
    blank_points: int = 5
    plots_dir: Path = Path("plots")

    def __post_init__(self) -> None:
        self.workbook_path = Path(self.workbook_path)
        self.plots_dir = Path(self.plots_dir)


def _all_well_labels() -> list[str]:
    rows = [chr(ord("A") + idx) for idx in range(8)]
    cols = [str(idx) for idx in range(1, 13)]
    return [f"{row}{col}" for row in rows for col in cols]


def render_block(block: BiotekMeasurementBlock, config: PlotConfig) -> Path:
    """Blank a single measurement block and save a linear 8x12 PDF."""
    blanked = blank_plate_data(block.dataframe, n_points_blank=config.blank_points)
    time_hours = compute_time_in_hours(blanked["Time"])
    safe_name = block.safe_label()
    output_path = config.plots_dir / f"{safe_name}_linear.pdf"
    wells = _all_well_labels()
    missing = [well for well in wells if well not in blanked.columns]
    for well in missing:
        blanked[well] = np.nan
    blanked = blanked[["Time"] + wells]
    plot_plate_growth_curves_linear(
        blanked,
        time_hours,
        output_path=output_path,
        plate_title=f"{block.plate_name} - {block.channel_label}",
        wells=wells,
    )
    return output_path


def main(config: PlotConfig) -> None:
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    blocks = load_biotek_measurements(config.workbook_path)
    if not blocks:
        raise RuntimeError("No measurement blocks were detected in the workbook.")

    for block in blocks:
        print(f"Plotting {block.channel_label} ({block.measurement_kind}) ...")
        output_path = render_block(block, config)
        print(f"  wrote {output_path}")

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
)

if __name__ == "__main__":
    main(CONFIG)
