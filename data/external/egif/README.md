# Artefactos externos EGIF

Esta carpeta contiene los artefactos locales utilizados para preparar y entrenar el modelo 2D EGIF:

- `galicia_1km.nc`: datacubo geoespacial de Galicia a 1 km.
- `dataset_2019.parquet` … `dataset_2023.parquet`: datasets anuales del modelo 2D.
- `metadata.json`: metadatos del datacubo y del dataset.
- `variables.md`: catálogo de variables disponible.

Los ficheros tabulares y el NetCDF ocupan varios gigabytes y están excluidos de Git mediante `.gitignore`. El repositorio conserva el código, los contratos, la documentación y las instrucciones; los datos deben distribuirse mediante Drive, almacenamiento de objetos o DVC.

## Uso local

Con estos ficheros en esta carpeta, los comandos principales son:

```bash
PYTHONPATH=. python scripts/prepare_operational_grid.py \
  --cube data/external/egif/galicia_1km.nc \
  --output data/processed/grid/galicia_grid_1km_egif.parquet

PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/external/egif \
  --output-dir data/models
```

El script de entrenamiento realiza lectura por lotes con PyArrow y submuestreo determinista de negativos. No se debe cargar el conjunto completo de varios años en memoria con `pandas.read_parquet`.

## Distribución reproducible

Para un servidor nuevo, copiar estos ficheros desde el almacenamiento externo y comprobar sus tamaños o hashes antes de ejecutar la preparación. No se incluyen secretos ni claves de API en esta carpeta.
