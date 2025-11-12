# Growthreader

Python utilities for processing Agilent LP600 plate reader exports. The project
loads the “Raw Data” worksheets from an LP600 workbook, blanks the plate,
computes per-well log₂ growth rates, and produces CSV and PDF reports that make
it easy to audit growth behaviour across the plate.

## Features
- Parse every `*- Raw Data` worksheet from an Agilent LP600 Excel workbook.
- Convert instrument timestamps to monotonic hours, blank wells, and fit
  log₂(OD) models to configurable OD windows.
- Export growth rates plus the OD bounds that were used for each well so the
  fit window can be reviewed and tuned.
- Generate plate-wide growth-curve grids (log and linear) as well as heatmaps
  of the fitted slopes.
- Reuse the `growth_curves_module.py` helpers directly from notebooks or other
  analysis scripts.

## Repository layout
- `measure_growth_rates_script.py` – turnkey pipeline; edit the constants in the
  “Configuration” section, then run the script to process a workbook.
- `growth_curves_module.py` – reusable helpers for loading, blanking, fitting,
  and plotting growth data.
- `LP600_example.xlsx` – sample workbook you can use to test the pipeline.
- `growth_rates.csv` – CSV output containing one row per plate+well with the
  fitted slope, OD bounds, and fitted time window (regenerated when you re-run
  the pipeline).
- `plots/` – PDF plots written by the script (log-scale curves, linear curves,
  and heatmaps).

## Requirements
- Python 3.9 or newer (tested with Python 3.11).
- Python packages: `pandas`, `numpy`, `scipy`, and `matplotlib`.

Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pandas numpy scipy matplotlib
```

## Quick start
1. Copy your LP600 workbook (`.xlsx`) into the repository (or update
   `WORKBOOK_PATH` to point elsewhere).
2. Adjust the configuration block near the top of
   `measure_growth_rates_script.py`. Important knobs include:
   - `BLANK_POINTS`: how many of the lowest OD readings to average per well.
   - `DEFAULT_OD_MIN` / `DEFAULT_OD_MAX`: OD bounds for the log₂ fit.
   - `WINDOW_SIZE`: number of consecutive points that must exceed `OD_min`
     before the fit can start.
   - `GROWTH_RATES_CSV` / `PLOTS_DIR`: output locations.
3. Run the pipeline:
   ```bash
   python measure_growth_rates_script.py
   ```
4. Inspect `growth_rates.csv` to review the slopes and (optionally) edit the
   `OD_min`/`OD_max` columns per well. Re-running the script respects those
   overrides, making it easy to tighten/loosen fit windows on a well-by-well
   basis.
5. Open the PDFs inside `plots/` to visually audit the raw curves and the
   heatmaps of per-plate growth rates.

## Using the module directly
You can import any helper from `growth_curves_module.py` for custom analyses:

```python
from growth_curves_module import load_raw_data, fit_log_od_growth_rates

plates = load_raw_data("LP600_example.xlsx")
plate_df = plates["Plate 1 - Raw Data"]
time_hours = compute_time_in_hours(plate_df["Time"])
slopes, fit_windows, _ = fit_log_od_growth_rates(plate_df, time_hours)
```

This makes it straightforward to embed the logic inside notebooks or larger
automation pipelines.

## Troubleshooting
- **No `*- Raw Data` sheets found** – confirm the sheet names in your workbook
  match the LP600 default naming. You can adjust `RAW_SUFFIX` in
  `growth_curves_module.py` if your export uses a different suffix.
- **NaN growth rates** – usually indicates the OD never exceeded `OD_min`,
  there were not enough consecutive points (see `WINDOW_SIZE`), or all values in
  the fit window were non-positive. Inspect the per-well curve PDFs to pick a
  better OD window.
- **Heatmaps look clipped** – set `HEATMAP_VMIN`/`HEATMAP_VMAX` in the script to
  enforce fixed colour limits across plates.

Feel free to open issues or submit PRs if you extend the workflow for other
plate formats or instruments.
