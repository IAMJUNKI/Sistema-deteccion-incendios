"""
Paso 4 del pipeline: cruce del fuego con la rejilla + ensamblaje del prototipo.

Hace dos cosas:
 A) Cruza los focos limpios de FIRMS con la rejilla de la Fase 1 (spatial join:
    a que celda pertenece cada foco) y agrega por (celda, dia).
    -> Datos/Datos_NASA/target_prototipo_celda_dia.csv
 B) Ensambla el dataset maestro prototipo de julio-2022: todas las celdas x
    todos los dias del mes, con terreno (grid) + meteo (metodo C, archivo de
    meteo_celdas) + target (A).
    -> Datos/dataset_maestro_prototipo_julio2022.parquet

Nota: el prototipo usa la meteo del MISMO dia (valido como demo de estructura).
El dataset de entrenamiento real usara datos hasta T-1 (regla anti-leakage).

Uso:  conda activate incendios-forestales && python construye_prototipo.py
"""
import geopandas as gpd
import pandas as pd

GRID = "Sistema-deteccion-incendios-main/data/galicia_grid_1km_2018.parquet"

# --- A) Cruce espacial FIRMS x rejilla ---
grid = gpd.read_parquet(GRID)
from pathlib import Path
_f = Path("Datos/Datos_NASA/firms_galicia_2019_2024_limpio.csv")
if _f.exists():
    focos = pd.read_csv(_f, parse_dates=["acq_date"])
else:  # version Excel
    focos = pd.read_excel(_f.with_suffix(".xlsx"), parse_dates=["acq_date"])
gf = gpd.GeoDataFrame(focos, geometry=gpd.points_from_xy(focos.longitude, focos.latitude),
                      crs="EPSG:4326").to_crs(grid.crs)   # GPS -> UTM del grid
j = gpd.sjoin(gf, grid[["cell_id", "geometry"]], how="inner", predicate="within")
target = (j.groupby(["cell_id", j.acq_date.dt.date.rename("fecha")])
            .agg(n_focos=("frp", "size"), frp_max=("frp", "max"), frp_sum=("frp", "sum"))
            .reset_index())
target.to_csv("Datos/Datos_NASA/target_prototipo_celda_dia.csv", index=False)
print(f"A) Target: {len(j)} focos asignados -> {len(target)} filas celda-dia")

# --- B) Prototipo julio-2022 (con la meteo definitiva del metodo C) ---
meteo = pd.read_parquet("Datos/meteo_celdas/meteo_celdas_1km_2022.parquet")
meteo["fecha"] = pd.to_datetime(meteo.fecha)
jul = meteo[(meteo.fecha >= "2022-07-01") & (meteo.fecha <= "2022-07-31")].copy()
jul["fecha"] = jul.fecha.dt.date

base = jul.merge(grid[["cell_id", "altitud_media", "pendiente_media", "orientacion_clase",
                       "combustible_clase", "combustible_pct_forestal"]], on="cell_id")
base = base.merge(target, on=["cell_id", "fecha"], how="left")
base["target"] = (base.n_focos.fillna(0) > 0).astype(int)
base.to_parquet("Datos/dataset_maestro_prototipo_julio2022.parquet", index=False)
print(f"B) Prototipo: {len(base)} filas | positivos: {base.target.sum()} ({100*base.target.mean():.3f}%)")
