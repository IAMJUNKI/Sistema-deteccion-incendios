import numpy as np
import pandas as pd
import xarray as xr

from src.pipeline import _calcular_vecindad_igniciones, construir_datacubo_completo


def test_vecindad_igniciones_marca_25x25_y_diez_dias_anteriores() -> None:
    target = np.zeros((12, 30, 30), dtype=np.float32)
    target[10, 15, 15] = 1
    active = np.ones((30, 30), dtype=bool)

    near = _calcular_vecindad_igniciones(target, active)

    assert near[0, 3, 3] == 1
    assert near[10, 27, 27] == 1
    assert near[11].sum() == 0
    assert near[0, 2, 2] == 0


def test_vecindad_igniciones_se_recorta_en_bordes_y_fuera_de_galicia() -> None:
    target = np.zeros((3, 4, 4), dtype=np.float32)
    target[2, 0, 0] = 1
    active = np.ones((4, 4), dtype=bool)
    active[0, 1] = False

    near = _calcular_vecindad_igniciones(target, active)

    assert near[1, 3, 3] == 1
    assert near[1, 0, 1] == 0
    assert near[2, 0, 0] == 1


def test_construir_datacubo_asigna_ceros_fuera_de_las_igniciones(tmp_path) -> None:
    spatial = xr.Dataset(
        {"is_galicia": (("y", "x"), np.array([[1]], dtype=np.uint8))},
        coords={"y": [0.0], "x": [0.0]},
    )
    topography = xr.Dataset(
        {
            "elevation_mean": (("y", "x"), np.array([[100.0]], dtype=np.float32)),
            "aspect_no_data_fraction": (("y", "x"), np.array([[0.0]], dtype=np.float32)),
        },
        coords={"y": [0.0], "x": [0.0]},
    )
    landcover = xr.Dataset(
        {"scrub": (("y", "x"), np.array([[0.5]], dtype=np.float32))},
        coords={"y": [0.0], "x": [0.0]},
    )
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    temporal = xr.Dataset({"month": ("time", [1, 1])}, coords={"time": time})
    meteorology = xr.Dataset(
        {
            "temperature_max": (("time", "y", "x"), np.array([[[10.0]], [[11.0]]])),
            "precipitation_sum_1d": (("time", "y", "x"), np.array([[[1.0]], [[2.0]]])),
        },
        coords={"time": time, "y": [0.0], "x": [0.0]},
    )

    paths = {
        "spatial": tmp_path / "spatial.nc",
        "topography": tmp_path / "topography.nc",
        "landcover": tmp_path / "landcover.nc",
        "temporal": tmp_path / "time.nc",
        "meteorology": tmp_path / "meteorology.nc",
        "target": tmp_path / "target.parquet",
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

    construir_datacubo_completo(
        paths["spatial"],
        paths["topography"],
        paths["landcover"],
        paths["temporal"],
        paths["meteorology"],
        paths["target"],
        paths["output"],
    )

    with xr.open_dataset(paths["output"]) as result:
        assert result["target_ignicion"].sel(time="2020-01-01").item() == 1
        assert result["target_ignicion"].sel(time="2020-01-02").item() == 0
        assert result["is_near_ignition_25x25_10d"].sel(time="2020-01-01").item() == 1
        assert "egif_observed" not in result
        assert "aspect_no_data_fraction" not in result
        assert "precipitation_sum_1d" not in result
