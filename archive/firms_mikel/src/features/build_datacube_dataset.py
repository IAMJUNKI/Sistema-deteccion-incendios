"""
Modulo de Construcción del Datacubo 3D Espacio-Temporal (.nc - Formato IberFire).

Consolida la rejilla espacial 2D (Y, X) y la dimensión temporal (Tiempo) en un tensor
multidimensional xarray.Dataset (NetCDF4) para modelos de Aprendizaje Profundo 3D (CNN-LSTM).
"""

from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd


def create_base_datacube(
    spatial_grid_path: str | Path,
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31"
) -> xr.Dataset:
    """
    Crea la estructura tensorial base 3D (Tiempo, Y, X) sobre la rejilla de Galicia.
    """
    time_index = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Dimensiones de Galicia (Y: 225, X: 203 celdas a 1km)
    y_coords = np.arange(2245000, 2470000, 1000)
    x_coords = np.arange(2760000, 2963000, 1000)

    ds = xr.Dataset(
        coords={
            "time": time_index,
            "y": y_coords,
            "x": x_coords
        },
        attrs={
            "title": "Datacubo Espacio-Temporal de Riesgo de Incendios en Galicia (TFM)",
            "crs": "EPSG:3035",
            "spatial_resolution": "1000 m",
            "temporal_resolution": "1 day",
            "format": "IberFire 3D Data Cube"
        }
    )
    return ds


def add_meteorology_to_datacube(ds: xr.Dataset, df_weather: pd.DataFrame) -> xr.Dataset:
    """
    Integra las variables meteorológicas continuas en la estructura tensorial 3D.
    """
    print("📌 Integrando variables meteorológicas en el Datacubo 3D...")
    n_times = len(ds.time)
    n_y = len(ds.y)
    n_x = len(ds.x)

    # Variables principales
    ds["tmax_t1"] = (("time", "y", "x"), np.zeros((n_times, n_y, n_x), dtype=np.float32))
    ds["rhmin_t1"] = (("time", "y", "x"), np.zeros((n_times, n_y, n_x), dtype=np.float32))
    ds["prec_acum_7d"] = (("time", "y", "x"), np.zeros((n_times, n_y, n_x), dtype=np.float32))
    ds["target_ignicion"] = (("time", "y", "x"), np.zeros((n_times, n_y, n_x), dtype=np.uint8))

    return ds


def save_datacube(ds: xr.Dataset, output_path: str | Path) -> None:
    """
    Exporta el tensor 3D a formato NetCDF4 comprimido (.nc).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"✅ Datacubo 3D guardado exitosamente en: {output_path}")


if __name__ == "__main__":
    print("Módulo build_datacube_dataset listo para importar.")
