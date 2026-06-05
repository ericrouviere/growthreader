# Growthreader

Utilities for processing Agilent LP600 and SynergyH1 plate-reader exports.
The toolkit blanks raw workbooks, converts timestamps to hours, fits
per‑well growth rates (μ, hr⁻¹), and emits CSV + PDF plots so you can audit OD and
fluorescence behavior quickly.

## Highlights
- Parse LP600 Excel files (every `*- Raw Data` worksheet) or SynergyH1 exports
  (one plate per file), pulling every OD or fluorescence channel automatically.
- Blank wells, fit ln(OD) models between configurable `OD_min`/`OD_max`
  bounds, and record the exact window chosen for each well. The reported slope
  is the exponential growth rate μ (hr⁻¹), i.e. the slope of ln(OD) vs time.
- Produce log + linear growth-curve grids and growth-rate heatmaps.
- Reuse the helpers directly from notebooks (all heavy lifting lives under
  `src/growthreader/`).

## Quick start

1. Clone or download this repository to a local folder. Make sure the folder is
   named `growthreader`.

2. Copy the example script that matches your instrument from `examples/` to the
   directory where your data lives:
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
