# Datacubo de riesgo de incendio en Galicia

TFM para construir un datacubo diario de 1 km × 1 km sobre Galicia. El target
es la ignición oficial de EGIF-MITECO; las fuentes explicativas son Copernicus
DEM GLO-30, CORINE Land Cover 2018 y ERA5-Land.

## Contrato del dataset

- Área: Galicia, rejilla regular de 1 km, `EPSG:3035`.
- Contexto meteorológico: desde 30 días antes del inicio hasta la última fecha
  registrada en el XML EGIF.
- Periodo de modelado: 2019-01-01 hasta la última fecha registrada en EGIF.
- Target: `target_ignicion` de EGIF. Un cero significa ausencia de ignición.
- Temporalidad: las variables meteorológicas y sus acumulados describen el
  mismo día `T`; este producto es un **nowcast/análisis diario**, no una
  previsión a 24 horas.

La arquitectura y el diccionario de datos se documentan en
[docs/architecture/datacube.md](docs/architecture/datacube.md).

## Estructura

```text
src/geospatial/   rejilla, DEM y CORINE
src/ingestion/    ERA5-Land y EGIF
src/features/     calendario y exportación tabular
src/workflow.py   orquestación completa
data/raw/         fuentes originales locales
data/processed/   productos generados (ignorados por Git)
archive/          código FIRMS/Mikel histórico, fuera de la ruta operativa
```

## Ejecución

Activa el entorno `incendios-forestales` y ejecuta, desde la raíz:

```powershell
& C:\Users\alfon\anaconda3\envs\incendios-forestales\python.exe -m src.workflow `
  --egif-xml data/raw/fire_history/Xml_20260821_223019_1.xml
```

El workflow reutiliza las capas estáticas, ERA5 y EGIF ya disponibles de forma
local; no vuelve a descargar fuentes. Genera el NetCDF final y un Parquet
tabular consolidado por año para cada año de datos.

## Validación

```powershell
& C:\Users\alfon\anaconda3\envs\incendios-forestales\python.exe -m pytest tests -q
```

Para validar los productos generados sin descargar ni modificar datos, ejecutar:

- `notebooks/07_validacion_datacubo.ipynb`: contrato, mapas, consistencia
  meteorologica y cobertura del target en el NetCDF.
- `notebooks/08_validacion_dataset_parquet.ipynb`: esquema, particiones,
  target, meteorologia y preparacion de los Parquet para ML.
