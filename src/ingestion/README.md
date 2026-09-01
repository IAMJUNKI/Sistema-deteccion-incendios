# Ingesta histórica

Este módulo procesa exclusivamente las fuentes activas: ERA5-Land para
meteorología y EGIF-MITECO para incendios. La entrada pública es
`src.ingestion.pipeline`; normalmente se invoca mediante `src.workflow`.
**Fase 2 del proyecto.** Descarga y procesamiento del histórico de incendios (NASA FIRMS) y meteorología histórica (ERA5-Land). Construcción de la variable objetivo mediante clustering espacio-temporal y generación de negativos difíciles.

## Responsabilidad

- Descargar focos de calor MODIS/VIIRS de NASA FIRMS para Galicia (2019-2024).
- Aplicar clustering DBSCAN para consolidar focos en eventos únicos de incendio.
- Generar muestras negativas estratificadas (vecinos espaciales, temporales y aleatorios).
- Descargar y preprocesar meteorología ERA5-Land en formato NetCDF.
- Descargar y normalizar el forecast horario WRF de MeteoGalicia mediante
  `src/ingestion/meteogalicia_forecast.py`.
- Usar AEMET OpenData como alternativa configurable mediante
  `src/ingestion/aemet_forecast.py` mientras no exista la clave de MeteoGalicia.
- Mantener el estado meteorológico diario reciente mediante
  `src/ingestion/weather_state.py`; este estado es obligatorio para la
  inferencia operativa y no se sustituye silenciosamente por una fecha histórica.
- Recuperar observaciones diarias históricas de AEMET mediante
  `src/ingestion/aemet_observations.py`, interpolarlas a la rejilla y usarlas
  para backfill/reconciliación con `scripts/ingest_aemet_weather_state.py`.
- Acumular observaciones horarias recientes de AEMET con
  `scripts/ingest_aemet_current_observations.py`; la ventana móvil de AEMET
  hace insuficiente una llamada diaria aislada para cerrar D-1.

## Entregable

Dataset integrado con columnas `cell_id`, `fecha`, `target` (0/1) y variables meteorológicas base.

## Fuentes de Datos

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) — MODIS y VIIRS
- [ERA5-Land (Copernicus CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) — vía `cdsapi`
- [MeteoGalicia MeteoSIX v5](https://www.meteogalicia.gal/datosred/infoweb/meteo/proxectos/meteosix/API_MeteoSIX_v5_es.pdf) — forecast horario WRF para producción
- [AEMET OpenData](https://opendata.aemet.es/dist/index.html) — forecast municipal alternativo y climatología diaria observada para bootstrap del estado

## Estado operativo reciente

`scripts/update_weather_state.py` recibe un CSV o Parquet diario con una fila
por celda y fecha, valida rangos físicos y hace un upsert atómico en
`data/processed/state/weather_daily_state.parquet`. A las 05:00 solo se
consideran días completos anteriores a la ejecución. La ruta de producción
falla si no encuentra los 30 días requeridos por celda.

La previsión operativa intenta WRF `1km` y utiliza WRF `04km` como fallback
explícito si la malla preferida falla o no cubre todas las horas. El resultado
se etiqueta `fresh`, `fresh_fallback` o `stale`; la malla efectiva nunca se
oculta.

El cliente MeteoSIX v5 no envía el parámetro opcional `units`: la API aplica sus
unidades por defecto (`degC`, `perc`, `lm2` y `kmh_deg`), que coinciden con el
contrato interno del proyecto. Esto evita que una variante de nomenclatura de
unidades sea rechazada por el validador de la API. Las unidades esperadas y el
hecho de usar los valores por defecto quedan registrados en los metadatos del
forecast bruto.

El proveedor se selecciona con `FORECAST_PROVIDER=meteogalicia|aemet|auto`.
`auto` intenta WRF 1 km, después WRF 04 km, y solo si ambas mallas fallan
intenta AEMET cuando `FORECAST_AUTO_AEMET_FALLBACK=true`. Esa contingencia se
marca como `fresh_aemet_degraded`: su predicción es municipal y no equivale a
una malla WRF de 1 km. Una selección explícita de `meteogalicia` no cambia a
AEMET silenciosamente.

## Bootstrap y actualización del estado

El archivo `data/processed/observations/weather_daily_latest.parquet` se genera
con la climatología diaria validada de AEMET para los últimos 30 días
publicados. AEMET limita cada consulta a 15 días y publica estos datos con un
retraso aproximado de cuatro días; por eso el script hace bloques consecutivos
de como máximo 15 días, devuelve valores de estaciones y asigna cada celda
mediante IDW a los cuatro vecinos más cercanos. Conserva un JSON bruto por
bloque en
`data/raw/aemet/observations/`.

```bash
PYTHONPATH=. python scripts/ingest_aemet_weather_state.py
```

La misma ejecución puede publicar `data/processed/state/weather_daily_state.parquet`
cuando el rango solicitado está completo. La primera ejecución evita esperar
30 ciclos para disponer de un histórico validado, pero no elimina el retraso
de publicación de AEMET. Para D-1 se usa el colector horario:

```bash
PYTHONPATH=. python scripts/ingest_aemet_current_observations.py --loop --interval-hours 6
```

El colector conserva una ventana horaria local, agrega los días que alcanzan
la cobertura mínima y los interpola a la misma rejilla. Si AEMET devuelve menos
fechas u horas de las necesarias, la ingesta no inventa valores ni publica un
estado incompleto.
