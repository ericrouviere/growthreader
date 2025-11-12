"""
Configured pipeline for processing Agilent LP600 growth data.

Edit the values in ``CONFIG`` below to point at your workbook, choose blanking
parameters, fit bounds, and output locations. Then run the script with
``python measure_growth_rates_script.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure Matplotlib and fontconfig use writable cache directories even if HOME is read-only.
_MPL_CACHE = Path(".matplotlib_cache")
_XDG_CACHE = Path(".cache")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE.resolve()))
(_XDG_CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
_MPL_CACHE.mkdir(parents=True, exist_ok=True)

from pipeline import PipelineConfig, run_pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = PipelineConfig(
    workbook_path=Path("LP600_example.xlsx"),
    blank_points=10,
    growth_rates_csv=Path("growth_rates.csv"),
    plots_dir=Path("plots"),
    default_od_min=0.01,
    default_od_max=0.1,
    window_size=3,
    heatmap_cmap="viridis",
    heatmap_vmin=None,
    heatmap_vmax=None,
    heatmap_annotate=True,
)


def main() -> None:
    """Execute the configured growth-rate pipeline."""
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()
