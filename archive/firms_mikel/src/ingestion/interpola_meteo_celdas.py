"""
Paso 3 del pipeline meteorologico: de la rejilla ERA5 de 9 km a las celdas de 1 km.

Entrada:  Datos/meteo_diaria_era5_2019_2024.parquet (salida del paso anterior)
          + el grid de la Fase 1 (galicia_grid_1km_2018.parquet)
Salida:   Datos/meteo_celdas/meteo_celdas_1km_YYYY.parquet (uno por anio;
          ~11,2M filas/anio; una fila por celda y dia)

Metodo (el "C", validado contra 3.566 observaciones de estaciones MeteoGalicia
de julio-2022 con un 14% menos de error que el vecino mas cercano):
 1. Interpolacion bilineal: cada celda mezcla los 4 puntos ERA5 que la rodean.
 2. Correccion de temperatura por altitud: -0,65 C por cada 100 m de diferencia
    entre la altitud real de la celda (del grid) y la altitud media que "ve"
    ERA5 en su pixel.
 3. Relleno costero: las celdas cuya interpolacion cae en mar (ERA5-Land solo
    cubre tierra) copian el punto TERRESTRE con datos mas cercano.

Uso:  python interpola_meteo_celdas.py 2019   (un anio por ejecucion)
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

anio = int(sys.argv[1])
m = pd.read_parquet("Datos/meteo_diaria_era5_2019_2024.parquet")
m = m[m.fecha.dt.year == anio]

grid = gpd.read_parquet("Sistema-deteccion-incendios-main/data/galicia_grid_1km_2018.parquet")
la = xr.DataArray(grid.lat_centroid.values, dims="cell")
lo = xr.DataArray(grid.lon_centroid.values, dims="cell")
cubo = m.set_index(["fecha", "latitude", "longitude"]).to_xarray()
lats, lons = cubo.latitude.values, cubo.longitude.values

# Puntos ERA5 terrestres (con datos) y, para cada celda, su terrestre mas cercano
mask = ~np.isnan(cubo["tmax_vc"].isel(fecha=0).values)
LON, LAT = np.meshgrid(lons, lats)
pts_lat, pts_lon = LAT[mask], LON[mask]
d2 = (grid.lat_centroid.values[:, None] - pts_lat)**2 + (grid.lon_centroid.values[:, None] - pts_lon)**2
idx = d2.argmin(1)
land_lat = xr.DataArray(pts_lat[idx], dims="cell")
land_lon = xr.DataArray(pts_lon[idx], dims="cell")

# Altitud aparente de cada pixel ERA5 = media de las celdas de 1 km asignadas a el
grid["ilat"] = np.abs(grid.lat_centroid.values[:, None] - lats).argmin(1)
grid["ilon"] = np.abs(grid.lon_centroid.values[:, None] - lons).argmin(1)
z_px = grid.groupby(["ilat", "ilon"]).altitud_media.mean()
z = np.full((len(lats), len(lons)), np.nan)
for (i, j), v in z_px.items():
    z[i, j] = v
z_da = xr.DataArray(z, coords={"latitude": lats, "longitude": lons})
z_bil = z_da.interp(latitude=la, longitude=lo, method="linear")
z_land = z_da.sel(latitude=land_lat, longitude=land_lon, method="nearest")
z_cel = z_bil.where(~np.isnan(z_bil), z_land)
z_cel = z_cel.where(~np.isnan(z_cel), xr.DataArray(grid.altitud_media.values, dims="cell"))

salida = {"cell_id": np.repeat(grid.cell_id.values[None, :], cubo.sizes["fecha"], 0).ravel(),
          "fecha": np.repeat(cubo.fecha.values, len(grid))}
for var in ["tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia"]:
    bil = cubo[var].interp(latitude=la, longitude=lo, method="linear")
    landv = cubo[var].sel(latitude=land_lat, longitude=land_lon, method="nearest")
    v = bil.where(~np.isnan(bil), landv)
    if var == "tmax_vc":
        v = v - 0.0065 * (xr.DataArray(grid.altitud_media.values, dims="cell") - z_cel)
    salida[var] = v.values.astype("float32").ravel()

df = pd.DataFrame(salida)
assert df.tmax_vc.notna().all(), "quedan huecos!"
df.to_parquet(f"Datos/meteo_celdas/meteo_celdas_1km_{anio}.parquet", index=False)
print(f"{anio}: {len(df)} filas, sin huecos")
