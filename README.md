# Growthreader

Utilities for processing Agilent LP600 and SynergyH1 plate-reader exports.
The toolkit blanks raw workbooks, converts timestamps to hours, fits
per‑well log₂ growth rates, and emits CSV + PDF reports so you can audit OD and
fluorescence behaviour quickly.

## Highlights
- Parse every `*- Raw Data` worksheet from LP600 Excel files.
- Load SynergyH1 exports (one plate per file) and pull every OD or
  fluorescence channel automatically.
- Blank wells, fit log₂(OD) models between configurable `OD_min`/`OD_max`
  bounds, and record the exact window chosen for each well.
- Produce log + linear growth-curve grids and growth-rate heatmaps.
- Reuse the helpers directly from notebooks (all heavy lifting lives under
  `src/growthreader/`).

## Quick start
1. Install once:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
2. Copy `examples/measure_growth_rates_LP600.py` next to your data (or run it
   in place) and edit the parameter block:
   - `WORKBOOK_PATH`: LP600 or SynergyH1 workbook path.
   - `MACHINE`: `"lp600"` or `"synergyh1"` to pick the parser.
   - `BLANK_POINTS`: number of low OD readings averaged per well.
   - `DEFAULT_OD_MIN` / `DEFAULT_OD_MAX`: bounds for the fitter. It walks
     backward from each well’s global peak: the first point below
     `DEFAULT_OD_MAX` ends the fit, and the first point above
     `DEFAULT_OD_MIN` on that descent starts it.
   - `GROWTH_RATES_CSV` / `PLOTS_DIR`: output locations.
   - `SYNERGYH1_OD_KEYWORDS`: strings that mark a channel as OD when parsing
     SynergyH1 exports (others are plotted linearly without fitting).
3. Run the pipeline:
   ```bash
   python measure_growth_rates_LP600.py
   ```
4. Inspect `growth_rates.csv`, tweak per-well `OD_min`/`OD_max` if needed, and
   re-run to regenerate the plots in `plots/`.

For SynergyH1 fluorescence-only runs, use
`examples/plot_OD_fluo_SynergyH1.py`; it blanks each channel, writes linear and
log PDFs, and exports a `growth_rates_plotter.csv` when an OD channel is
present.

## Regression + notebook usage
- `python tests/simulated_growth_pipeline_test.py` synthesizes a deterministic
  workbook, runs the same pipeline as the LP600 script, and compares the fitted
  slopes against ground truth. Use it whenever you touch core logic.
- Direct imports:
  ```python
  from growthreader.growth_curves_module import load_raw_data, compute_time_in_hours, fit_log_od_growth_rates
  ```
  make it easy to embed the workflow inside larger analyses.

## Troubleshooting
- **No `*- Raw Data` sheets** – tweak `RAW_SUFFIX` inside
  `src/growthreader/growth_curves_module.py`.
- **NaN growth rates** – usually the OD never exceeded `OD_min`, never dipped
  below `OD_max`, or stayed non-positive in the extracted window; tighten the
  bounds in the CSV and rerun.
- **Heatmaps look clipped** – fix the colour range with
  `HEATMAP_VMIN`/`HEATMAP_VMAX`.
- **Fontconfig cache warnings** – the scripts create `.cache/` +
  `.matplotlib_cache/` locally, but on shared systems set `MPLCONFIGDIR` and
  `XDG_CACHE_HOME` to writable folders.

Open issues or PRs if you add new instruments—the SynergyH1 loader is a
template for future readers.
