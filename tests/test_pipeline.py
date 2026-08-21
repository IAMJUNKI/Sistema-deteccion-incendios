import json

import numpy as np
import pandas as pd
import xarray as xr

from src.pipeline import construir_datacubo_completo


def test_construir_datacubo_marca_como_ausente_el_periodo_sin_egif(tmp_path) -> None:
    spatial = xr.Dataset(
        {"is_galicia": (("y", "x"), np.array([[1]], dtype=np.uint8))},
        coords={"y": [0.0], "x": [0.0]},
    )
    topography = xr.Dataset(
        {"elevation_mean": (("y", "x"), np.array([[100.0]], dtype=np.float32))},
        coords={"y": [0.0], "x": [0.0]},
    )
    landcover = xr.Dataset(
        {"scrub": (("y", "x"), np.array([[0.5]], dtype=np.float32))},
        coords={"y": [0.0], "x": [0.0]},
    )
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    temporal = xr.Dataset({"month": ("time", [1, 1])}, coords={"time": time})
    meteorology = xr.Dataset(
        {"temperature_max": (("time", "y", "x"), np.array([[[10.0]], [[11.0]]]))},
        coords={"time": time, "y": [0.0], "x": [0.0]},
    )

    paths = {
        "spatial": tmp_path / "spatial.nc",
        "topography": tmp_path / "topography.nc",
        "landcover": tmp_path / "landcover.nc",
        "temporal": tmp_path / "time.nc",
        "meteorology": tmp_path / "meteorology.nc",
        "target": tmp_path / "target.parquet",
        "metadata": tmp_path / "metadata.json",
        "output": tmp_path / "complete.nc",
    }
    spatial.to_netcdf(paths["spatial"])
    topography.to_netcdf(paths["topography"])
    landcover.to_netcdf(paths["landcover"])
    temporal.to_netcdf(paths["temporal"])
    meteorology.to_netcdf(paths["meteorology"])
    pd.DataFrame(
        {"cell_id": [0], "fecha": [pd.Timestamp("2020-01-01")], "target_ignicion": [1]}
    ).to_parquet(paths["target"], index=False)
    paths["metadata"].write_text(
        json.dumps({"available_event_period": {"end": "2020-01-01"}}),
        encoding="utf-8",
    )

    construir_datacubo_completo(
        paths["spatial"],
        paths["topography"],
        paths["landcover"],
        paths["temporal"],
        paths["meteorology"],
        paths["target"],
        paths["metadata"],
        paths["output"],
    )

    with xr.open_dataset(paths["output"]) as result:
        assert result["target_ignicion"].sel(time="2020-01-01").item() == 1
        assert np.isnan(result["target_ignicion"].sel(time="2020-01-02").item())
        assert result["egif_observed"].values.tolist() == [1, 0]

