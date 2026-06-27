---
name: geospatial-processing
description: >
  Instrucciones para construir la infraestructura geoespacial del proyecto: rejilla de celdas 
  1km×1km sobre Galicia, extracción de variables topográficas del DEM de Copernicus, 
  procesamiento de CORINE Land Cover y operaciones espaciales con GeoPandas y Rasterio.
  Activar en la Fase 1 del proyecto o cuando se trabaje con datos espaciales.
---

# Skill: Procesamiento Geoespacial — Fase 1

## Contexto

El sistema predictivo divide Galicia en una **rejilla de celdas de ~1km × 1km**. Cada celda tiene un identificador único (`cell_id`) y almacena variables estáticas (topografía, vegetación) calculadas una sola vez. Esta es la estructura base sobre la que se cruzan todos los datos temporales (meteorología, incendios).

---

## Librería Stack

```python
import geopandas as gpd           # Operaciones espaciales vectoriales
import rasterio                    # Lectura y escritura de rasters
import rioxarray                   # Rasters con xarray (NetCDF, GeoTIFF)
import numpy as np
from shapely.geometry import box
from pathlib import Path
```

## CRS (Sistema de Referencia de Coordenadas)

- **CRS de trabajo**: `EPSG:25829` (ETRS89 / UTM zone 29N) — estándar para Galicia.
- **CRS fuente de los datos externos**: `EPSG:4326` (WGS84).
- **Regla**: Siempre reprojectar a `EPSG:25829` antes de calcular distancias, áreas o pendientes.

```python
gdf = gdf.to_crs("EPSG:25829")
```

---

## 1. Construcción de la Rejilla de Celdas

### Paso 1: Cargar el límite de Galicia

```python
# Descargar desde CNIG/IGN: https://centrodedescargas.cnig.es
gdf_galicia = gpd.read_file("data/raw/igm/galicia_boundary.geojson")
gdf_galicia = gdf_galicia.to_crs("EPSG:25829")
bounds = gdf_galicia.total_bounds  # (minx, miny, maxx, maxy)
```

### Paso 2: Generar rejilla regular

```python
from shapely.geometry import box
import numpy as np

CELL_SIZE_M = 1000  # 1km x 1km

minx, miny, maxx, maxy = bounds
cols = np.arange(minx, maxx, CELL_SIZE_M)
rows = np.arange(miny, maxy, CELL_SIZE_M)

cells = []
for x in cols:
    for y in rows:
        cell = box(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M)
        cells.append(cell)

gdf_grid = gpd.GeoDataFrame(geometry=cells, crs="EPSG:25829")
```

### Paso 3: Recortar al límite de Galicia

```python
# Solo celdas que intersectan con el territorio de Galicia
gdf_grid = gpd.sjoin(gdf_grid, gdf_galicia[["geometry"]], 
                      how="inner", predicate="intersects")
gdf_grid = gdf_grid.reset_index(drop=True)
gdf_grid["cell_id"] = gdf_grid.index.astype(int)

# Añadir centroide en WGS84 para visualización
centroids_wgs84 = gdf_grid.geometry.centroid.to_crs("EPSG:4326")
gdf_grid["lat_centroid"] = centroids_wgs84.y
gdf_grid["lon_centroid"] = centroids_wgs84.x
```

### Alternativa: Rejilla hexagonal con H3

```python
import h3
# Resolución 8 de H3 ≈ 0.7 km² de área por celda (similar al 1km²)
# Ventaja: vecindad más uniforme (6 vecinos equidistantes)
```

---

## 2. Extracción de Variables Topográficas (Copernicus DEM)

### Descarga

Fuente: [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/)
Formato: GeoTIFF, resolución ~30m

### Cálculo de altitud media por celda

```python
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

with rasterio.open("data/raw/dem/dem_galicia_glo30.tif") as src:
    # Reprojectar grid al CRS del raster si es necesario
    gdf_grid_dem = gdf_grid.to_crs(src.crs)
    
    altitudes = []
    for _, row in gdf_grid_dem.iterrows():
        geom = [mapping(row.geometry)]
        try:
            out_image, _ = mask(src, geom, crop=True, nodata=np.nan)
            altitudes.append(float(np.nanmean(out_image)))
        except Exception:
            altitudes.append(np.nan)

gdf_grid["altitud_media"] = altitudes
```

### Cálculo de pendiente y orientación

```python
import rioxarray

dem = rioxarray.open_rasterio("data/raw/dem/dem_galicia_glo30.tif").squeeze()

# Pendiente en grados
dx, dy = np.gradient(dem.values, dem.rio.resolution()[1], dem.rio.resolution()[0])
pendiente = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

# Orientación: 0=N, 90=E, 180=S, 270=O
orientacion_rad = np.arctan2(-dx, dy)
orientacion_deg = np.degrees(orientacion_rad) % 360
```

---

## 3. Procesamiento de CORINE Land Cover

### Descarga

Fuente: [CORINE Land Cover (Copernicus)](https://land.copernicus.eu/en/products/corine-land-cover)

### Agrupación en clases de combustible

Las 44 clases de CORINE se agrupan en macro-clases de riesgo:

```python
CORINE_A_COMBUSTIBLE = {
    # Alto riesgo
    311: "bosque_frondosas",     # Bosques de frondosas
    312: "bosque_coniferas",     # Bosques de coníferas
    313: "bosque_mixto",         # Bosques mixtos
    321: "pastizal",             # Pastizales naturales
    322: "brezal_matorral",      # Brezales y matorrales
    323: "matorral_transicional",# Vegetación esclerófila
    324: "matorral_transicional",# Vegetación de transición
    # Bajo riesgo
    111: "urbano",               # Tejido urbano continuo
    112: "urbano",               # Tejido urbano discontinuo
    211: "agricola",             # Tierras de labor en secano
    # Añadir resto de clases...
}

gdf_corine["combustible_clase"] = gdf_corine["CODE_18"].map(CORINE_A_COMBUSTIBLE)
```

### Extracción por celda

```python
# Spatial join para obtener clase predominante por celda
joined = gpd.sjoin(gdf_grid, gdf_corine[["geometry", "combustible_clase"]], 
                    how="left", predicate="intersects")
clase_predominante = joined.groupby("cell_id")["combustible_clase"].agg(
    lambda x: x.value_counts().index[0] if len(x) > 0 else "sin_datos"
)
gdf_grid["combustible_clase"] = gdf_grid["cell_id"].map(clase_predominante)
```

---

## 4. Guardado del Dataset Estático

```python
# Guardar en formato GeoParquet (geoespacial + eficiente)
gdf_grid.to_parquet("data/processed/grid/galicia_grid_1km_v1.parquet")

# También en GeoJSON para verificación visual en QGIS
gdf_grid.to_file("data/processed/grid/galicia_grid_1km_v1.geojson", driver="GeoJSON")
```

---

## 5. Verificación Visual

Para verificar el resultado visualmente antes de continuar:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf_grid.plot(column="combustible_clase", ax=ax, legend=True, 
               cmap="RdYlGn_r", alpha=0.7)
gdf_galicia.boundary.plot(ax=ax, color="black", linewidth=1)
ax.set_title("Rejilla 1km×1km — Galicia (MVP)")
plt.savefig("docs/verificacion_rejilla.png", dpi=150, bbox_inches="tight")
```

---

## Checklist de Calidad

Antes de dar por terminada la Fase 1, verificar:

- [ ] La rejilla cubre todo el territorio de Galicia sin huecos.
- [ ] El `cell_id` es único y sin duplicados.
- [ ] Todas las celdas tienen valores válidos de altitud, pendiente y orientación (o NaN documentado para mar/fronteras).
- [ ] Las clases CORINE cubren >95% de las celdas terrestres.
- [ ] El CRS de todos los outputs es `EPSG:25829` (o `EPSG:4326` para visualización).
- [ ] El archivo Parquet se carga correctamente: `gpd.read_parquet("...").shape` devuelve ~30.000 filas.

---

## Entregable Final

```
data/processed/grid/
├── galicia_grid_1km_v1.parquet   # Dataset principal (cell_id, coords, topo, combustible)
└── galicia_grid_1km_v1.geojson   # Para verificación en QGIS (puede omitirse en producción)
```

Columnas esperadas: `cell_id`, `geometry`, `lat_centroid`, `lon_centroid`, `altitud_media`, `pendiente_media`, `orientacion_media`, `combustible_clase`.
