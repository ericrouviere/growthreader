"""
Configured pipeline for processing Agilent LP600 growth data.

Edit the values in the parameter block below to point at your workbook, choose
blanking parameters, fit bounds, and output locations. Then run the script with
``python measure_growth_rates_script.py``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Excel workbook exported from the Agilent LP600 (relative or absolute path).
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
from pathlib import Path

# Ensure Matplotlib and fontconfig use writable cache directories even if HOME is read-only.
_MPL_CACHE = Path(".matplotlib_cache")
_XDG_CACHE = Path(".cache")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE.resolve()))
(_XDG_CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
_MPL_CACHE.mkdir(parents=True, exist_ok=True)

try:
    from growthreader import PipelineConfig, run_pipeline
except ModuleNotFoundError:  # pragma: no cover - convenience for in-repo runs
    import sys

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
        from growthreader import PipelineConfig, run_pipeline
    else:
        raise

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = PipelineConfig(
    workbook_path=Path(WORKBOOK_PATH),
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
