# Growthreader

Utilities for processing Agilent LP600 and SynergyH1 plate-reader exports.
The toolkit blanks raw workbooks, converts timestamps to hours, fits
per‑well growth rates (μ, hr⁻¹), and emits CSV + PDF plots so you can audit OD and
fluorescence behavior quickly.

## Quick start

1. Clone or download this repository to a local folder. Make sure the folder is
   named `growthreader`.

2. Copy a script from `examples/` to the directory where your data lives:
   - **LP600 or SynergyH1 (growth rates):** `examples/measure_growth_rates.py`
   - **SynergyH1 (OD + fluorescence plots):** `examples/plot_OD_fluo_SynergyH1.py`

3. In the copied script, edit the parameter block at the top:
   - `GROWTHREADER_REPO`: absolute path to your local `growthreader` folder.
   - `WORKBOOK_PATH`: name of your `.xlsx` data file.

4. Run the script:
   ```bash
   python measure_growth_rates.py   # or plot_OD_fluo_SynergyH1.py
   ```
5. Inspect the output CSV, tweak per-well `OD_min`/`OD_max` if needed, and
   re-run to regenerate the plots.

## Scripts

### `measure_growth_rates.py`

The main growth-rate pipeline. Point it at an LP600 or SynergyH1 workbook and
it will:

- **Blank** each well by averaging its lowest `BLANK_POINTS` OD readings and
  subtracting that baseline from the full time series.
- **Fit exponential growth rates** for every well. The fitter identifies the
  largest OD peak in each well, then walks backward to find a window where OD
  is between `OD_min` and `OD_max`. A linear regression on ln(OD) vs time
  within that window gives the growth rate μ (hr⁻¹).
- **Write a CSV** (`<prefix>_<workbook name>_growth_rates.csv`) with one row per
  well containing the fitted slope, the OD bounds used, and the start/end times
  of the fit window. You can manually adjust `OD_min`/`OD_max` for individual
  wells in this file and re-run the script to update only those wells.
- **Write PDFs** to a `plots_<workbook name>/` directory: one log-scale
  growth-curve grid, one linear grid, and one heatmap of growth rates across
  the plate.

Works with both LP600 (multi-plate workbooks) and SynergyH1 (single-plate)
exports — the format is detected automatically.

### `plot_OD_fluo_SynergyH1.py`

A plotting script for SynergyH1 runs that include fluorescence channels in
addition to OD. For each measurement channel in the workbook it:

- **Blanks** the data using the same low-point averaging as above.
- **Writes a linear and a log PDF** showing the full 8×12 plate grid for that
  channel.
- **Fits growth rates on OD channels** and exports them to a CSV, identical in
  format to the one produced by `measure_growth_rates.py`. Fluorescence
  channels are plotted but not fitted.

Use this script when you want side-by-side plots of OD and reporter
fluorescence from the same run.

## Regression + notebook usage
- `python tests/simulated_growth_pipeline_test.py` synthesizes a deterministic
  workbook, runs the full pipeline, and compares the fitted slopes against
  ground truth. Use it whenever you touch core logic.
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
