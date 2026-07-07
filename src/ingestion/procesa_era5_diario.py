"""
Paso 2 del pipeline meteorologico: de los 72 NetCDF horarios a la tabla diaria.

Entrada:  Datos/era5_bruto/era5land_galicia_YYYY_MM.nc (72 archivos, salida de
          descarga_era5_2019_2024.py)
Salida:   Datos/meteo_diaria_era5_2019_2024.parquet
          (una fila por punto ERA5 de ~9 km y dia; ~879.000 filas)

Que hace, por cada punto y dia:
 - tmax_vc : temperatura maxima en la ventana critica 12-18h LOCAL (con el
             cambio de hora real de Europe/Madrid), en C (ERA5 viene en Kelvin)
 - rhmin_vc: humedad relativa minima en la ventana (formula de Magnus con
             temperatura y punto de rocio)
 - vmax_vc : racha maxima en la ventana (modulo de las componentes u,v en km/h)
 - prec_dia: lluvia total del dia (ERA5 la da ACUMULADA desde medianoche:
             el total diario es el maximo del acumulado, no la suma)

Uso:  conda activate incendios-forestales && python procesa_era5_diario.py
"""
import glob

import numpy as np
import pandas as pd
import xarray as xr

archivos = sorted(glob.glob("Datos/era5_bruto/era5land_galicia_*.nc"))
print(f"{len(archivos)} archivos mensuales encontrados")
partes = []
for f in archivos:
    ds = xr.open_dataset(f)
    t = ds.t2m - 273.15
    d = ds.d2m - 273.15
    rh = 100 * np.exp(17.625 * d / (243.04 + d)) / np.exp(17.625 * t / (243.04 + t))
    viento = np.sqrt(ds.u10**2 + ds.v10**2) * 3.6
    horas_local = pd.DatetimeIndex(ds.valid_time.values).tz_localize("UTC").tz_convert("Europe/Madrid")
    hl = xr.DataArray(horas_local.hour.values, dims="valid_time", coords={"valid_time": ds.valid_time})
    vc = (hl >= 12) & (hl <= 18)
    dia = ds.valid_time.dt.date
    dfs = []
    for nombre, arr, agg in [("tmax_vc", t.where(vc), "max"), ("rhmin_vc", rh.where(vc), "min"),
                             ("vmax_vc", viento.where(vc), "max"), ("prec_dia", ds.tp * 1000, "max")]:
        g = arr.groupby(dia)
        r = g.min() if agg == "min" else g.max()
        dfs.append(r.to_dataframe(name=nombre).reset_index()[["date", "latitude", "longitude", nombre]]
                    .set_index(["date", "latitude", "longitude"]))
    partes.append(pd.concat(dfs, axis=1).reset_index().rename(columns={"date": "fecha"}).dropna(subset=["tmax_vc"]))
    ds.close()
    print(".", end="", flush=True)

todo = pd.concat(partes, ignore_index=True)
todo["fecha"] = pd.to_datetime(todo.fecha)
for c in ["tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia"]:
    todo[c] = todo[c].astype("float32")
todo.to_parquet("Datos/meteo_diaria_era5_2019_2024.parquet", index=False)
print(f"\nGuardado: {len(todo)} filas, {todo.fecha.nunique()} dias")
