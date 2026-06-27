# Módulo: ingestion — Ingesta Histórica y Construcción del Target

**Fase 2 del proyecto.** Descarga y procesamiento del histórico de incendios (NASA FIRMS) y meteorología histórica (ERA5-Land). Construcción de la variable objetivo mediante clustering espacio-temporal y generación de negativos difíciles.

## Responsabilidad

- Descargar focos de calor MODIS/VIIRS de NASA FIRMS para Galicia (2019-2024).
- Aplicar clustering DBSCAN para consolidar focos en eventos únicos de incendio.
- Generar muestras negativas estratificadas (vecinos espaciales, temporales y aleatorios).
- Descargar y preprocesar meteorología ERA5-Land en formato NetCDF.

## Entregable

Dataset integrado con columnas `cell_id`, `fecha`, `target` (0/1) y variables meteorológicas base.

## Fuentes de Datos

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) — MODIS y VIIRS
- [ERA5-Land (Copernicus CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) — vía `cdsapi`
