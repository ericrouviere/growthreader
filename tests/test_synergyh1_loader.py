from pathlib import Path

import pandas as pd

from growthreader import load_synergyh1_measurements


def test_load_synergyh1_measurements_extracts_all_channels() -> None:
    workbook = Path("examples/SynergyH1_example.xlsx")
    blocks = load_synergyh1_measurements(workbook)
    assert len(blocks) == 3

    kinds = [block.measurement_kind for block in blocks]
    assert kinds.count("od") == 1
    assert kinds.count("fluorescence") == 2

    time_columns = [block.dataframe["Time"] for block in blocks]
    assert all(isinstance(series, pd.Series) for series in time_columns)
    # Ensure all well labels survived the import and at least one column has values.
    for block in blocks:
        well_cols = [col for col in block.dataframe.columns if col != "Time"]
        assert len(well_cols) == 96
        assert block.dataframe[well_cols].notna().any().any()
