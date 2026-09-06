# Estado de la construcción del datacubo y del modelo de 48 variables
 
Este documento deja constancia de la construcción local realizada para adaptar el modelo EGIF de
48 variables a producción y de qué salidas están completas o siguen en curso.
 
## Estado comprobado
 
La comprobación se realizó el 5 de septiembre de 2026 en el entorno local.
 
### Completado
 
- Los datos de `misc/Datos/raw` se han normalizado a `data/raw`.
- La copia se verificó mediante SHA-256 y el origen se conserva.
- Las capas estáticas se han preparado para una rejilla operativa de 29.601 celdas.
- Se ha generado ERA5 diario y su interpolación a la rejilla de 1 km.
- Se ha interpolado el FWI como referencia física, no como predictor del modelo de 48.
- Se ha construido el target diario EGIF.
- Se ha ensamblado el NetCDF canónico, de aproximadamente 4 GB.
- El código de contratos, entrenamiento, inferencia, rollback y shadow está implementado.
 
### Exportación finalizada
 
La exportación del NetCDF a Parquet anual ha terminado correctamente. Existen las ocho
particiones consolidadas, de 2016 a 2023, y ya no quedan archivos temporales `part-*.parquet`.
El directorio contiene `metadata.json` y la exportación no descartó filas por predictores
incompletos (`dropped_rows_incomplete_predictors=0`).
 
Cada año tiene aproximadamente 10,8 millones de filas: una fila por celda activa y día. Los
años bisiestos 2016 y 2020 tienen 10.833.966 filas; los demás tienen 10.804.365. El total es
86.494.122 filas y el metadata publica 50 predictores canónicos.
 
Los bloques intermedios ya fueron consolidados y pueden utilizarse como dataset histórico, sujeto
a la comprobación temporal descrita más abajo.
 
La salida consolidada contiene `metadata.json` y un Parquet anual para cada año:
 
```text
data/processed/tabular/egif/metadata.json
data/processed/tabular/egif/year=2016/dataset_2016.parquet
data/processed/tabular/egif/year=2017/dataset_2017.parquet
data/processed/tabular/egif/year=2018/dataset_2018.parquet
data/processed/tabular/egif/year=2019/dataset_2019.parquet
data/processed/tabular/egif/year=2020/dataset_2020.parquet
data/processed/tabular/egif/year=2021/dataset_2021.parquet
data/processed/tabular/egif/year=2022/dataset_2022.parquet
data/processed/tabular/egif/year=2023/dataset_2023.parquet
```
 
Los archivos `part-*.parquet` fueron bloques temporales de trabajo y ya no quedan en el directorio.
Cada Parquet anual está listo para validación y entrenamiento, siempre respetando la comprobación
temporal descrita en la sección siguiente.
 
Para consultar el estado sin modificar nada:
 
```bash
find data/processed/tabular/egif -maxdepth 1 -type f -print
find data/processed/tabular/egif -mindepth 2 -maxdepth 2 -type f -name 'dataset_*.parquet' -print
find data/processed/tabular/egif -type f -name 'part-*.parquet' | wc -l
```
 
La señal de finalización se ha cumplido: existe `metadata.json`, están las ocho particiones anuales
y no quedan bloques temporales.
 
## 1. Normalización de los datos raw
 
El equipo recibió los datos en una estructura de intercambio. El workflow espera una estructura
estable, por lo que se añadió `scripts/migrate_raw_data.py` con copia, verificación y manifiesto:
 
```text
misc/Datos/raw/meteorology/era5           → data/raw/meteorology/era5
misc/Datos/raw/meteorology/FWI            → data/raw/meteorology/fwi
misc/Datos/raw/fire_history               → data/raw/fire_history
misc/Datos/raw/human_activity             → data/raw/human_activity/galicia-220101-free-shp
misc/Datos/raw/landcover/*.tif            → data/raw/corine/
misc/Datos/raw/galicia_boundary/*.geojson → data/raw/igm/
```
 
El resultado se registra en `data/raw/migration_manifest.json`, que contiene origen, destino,
tamaño y SHA-256. El origen se mantiene hasta validar completamente el nuevo dataset. No se ha
eliminado información de `misc/Datos/raw`.
 
## 2. Preparación de las capas estáticas
 
Las capas estáticas describen topografía, combustible y actividad humana: elevación, pendiente,
orientación, cobertura del suelo, carreteras, edificios y superficie residencial.
 
La reconstrucción desde el DEM de Copernicus no pudo completarse porque la sesión de AWS utilizada
para descargarlo había caducado. Para desbloquear la integración local se añadió
`scripts/import_static_from_datacube.py`.
 
Este script reutiliza únicamente las capas estáticas de `/Users/junki/Downloads/galicia_1km.nc`.
No reutiliza su meteorología, FWI ni target. Produce capas estáticas en `data/processed/static/` y
la rejilla operativa en `data/processed/grid/`.
 
El manifiesto está en `data/processed/static/static_import_manifest.json` y marca la salida como
`shared_static_layer_import`. Es válida para pruebas locales, pero la ejecución científica final
debe repetir la construcción con un DEM validado y credenciales renovadas.
 
## 3. Construcción del datacubo
 
El workflow se lanzó con:
 
```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  -m src.workflow \
  --start-year 2016 \
  --end-year 2023 \
  --no-download-missing-era5
```
 
El parámetro `--no-download-missing-era5` es correcto porque los datos ERA5 necesarios ya estaban
disponibles localmente. Las etapas son:
 
1. validación de capas estáticas y rejilla;
2. preparación del calendario 2016–2023;
3. generación de ERA5 diario;
4. interpolación meteorológica a 1 km;
5. interpolación del FWI;
6. construcción del target EGIF;
7. ensamblado del NetCDF canónico;
8. exportación en bloques y consolidación anual.
 
El cubo conserva 50 predictores canónicos para trazabilidad. Esto no contradice el modelo
operativo de 48: el modelo selecciona un subconjunto versionado del cubo.
 
### Comprobación temporal adicional
 
La exportación completa no implica por sí sola que el Parquet pueda conectarse directamente al
entrenamiento operativo. El `metadata.json` del cubo declara:
 
```text
time_contract: No temporal shift; meteorological accumulations include date T.
```
 
Esto describe un cubo retrospectivo con acumulaciones inclusivas del día de la fila. En cambio,
la inferencia operativa calcula las memorias con días anteriores al objetivo. Esta diferencia ya
no se resuelve dentro del lector de entrenamiento: se ha implementado una exportación separada
en `data/processed/tabular/egif_operational/`, generada por
`scripts/build_operational_benchmark.py`. El proceso conserva el cubo canónico intacto,
recalcula las memorias con `shift(1)` de forma streaming y arrastra hasta 30 días entre años.

La familia 48 y el control alineado de 50 solo podrán entrenarse con esa exportación. Si falta
su `alignment_version` o se declara la semántica inclusiva, el entrenamiento falla antes de leer
las particiones.
 
## 4. Por qué el modelo consume 48 variables
 
El contrato operativo se llama `egif-2d-48-v1` y deriva del contrato canónico de 50 variables
eliminando exactamente:
 
```text
precipitation_sum
consecutive_dry_days
```
 
Las memorias de precipitación de 3, 7, 14 y 30 días se mantienen porque son ventanas temporales
diferenciadas. El FWI se conserva como baseline físico, pero no se entrega al modelo de 48 como
predictor.
 
La lista, el orden y la validación están congelados en `src/features/canonical_contract.py`.
La inferencia rechaza columnas faltantes, extras, identificadores, coordenadas, fechas o FWI.
 
## 5. Corrección temporal
 
En producción, para un horizonte `h`, se utiliza:
 
```text
issue_time
    + forecast del target_date
    + memoria observada/prevista disponible
    → target_ignicion(target_date)
```
 
Esto implica que:
 
- T+1 usa el forecast del día siguiente;
- T+2 y T+3 usan también los días intermedios en sus memorias;
- ninguna observación posterior a `issue_time` entra en una feature;
- la ventana crítica se agrega en `Europe/Madrid` y se normaliza internamente a UTC;
- las memorias de precipitación y temperatura son anteriores al día objetivo.
 
Como no existen vintages históricos de forecasts de MeteoGalicia, el entrenamiento histórico se
marca como `era5_perfect_benchmark`: la meteorología del día objetivo se conoce retrospectivamente.
Es un benchmark científico, no una medición final del error de WRF.
 
## 6. Entrenamiento previsto
 
Se conservarán dos experimentos:
 
```text
comparable: train 2019–2020, calibración 2021, validación 2022, test ciego 2023
expanded:   train 2016–2020, calibración 2021, validación 2022, test ciego 2023
```
 
Cada horizonte tendrá su modelo y calibrador:
 
```text
forecast_risk_egif_48_t1.joblib
forecast_risk_egif_48_t2.joblib
forecast_risk_egif_48_t3.joblib
```
 
La familia de 50 variables permanece separada para rollback/shadow:
 
```text
forecast_risk_egif_t1.joblib
forecast_risk_egif_t2.joblib
forecast_risk_egif_t3.joblib
```
 
La exportación alineada ya se ha generado y validado en
`data/processed/tabular/egif_operational/`. Contiene los ocho años 2016--2023,
86.494.122 filas, 29.601 celdas activas, cero filas descartadas y las nueve memorias
recalculadas con frontera T-1. La metadata conserva el hash del datacubo padre y los footers
Parquet coinciden con los conteos declarados. Como el origen empieza el 1/01/2016, las memorias
de ese primer día no tienen contexto anterior; esas filas se conservan para auditoría y el
entrenamiento las excluye explícitamente como calentamiento, sin imputarlas silenciosamente.

Los artefactos de 48 todavía no se han generado. Esta es una decisión intencionada: no se debe
entrenar hasta revisar la salida completa y no se deben confundir los modelos actuales de 50 con
la nueva familia.
 
La exportación alineada se construye con:

```bash
PYTHONPATH=. python scripts/build_operational_benchmark.py \
  --source-dir data/processed/tabular/egif \
  --output-dir data/processed/tabular/egif_operational \
  --years 2016-2023 \
  --static-layer-quality provisional_import
```

Después se ejecuta `scripts/train_egif_operational.py` con
`--feature-contract-version egif-2d-48-v1` y el experimento correspondiente. Para ejecutar también
el challenger XGBoost se añade `--include-xgboost-challenger`. El control justo de 50 variables
usa `--feature-contract-version egif-2d-v1 --aligned-50-control` y se guarda fuera de los
artefactos publicados.
 
## 7. Inferencia y visualización
 
La inferencia selecciona familias completas en este orden:
 
```text
familia 48 completa → familia 50 completa → familia legacy completa
```
 
Cuando existan los tres artefactos 48 se podrá activar `FORECAST_MODEL_FAMILY=egif_48`.
`SHADOW_50_MODEL=true` permitirá comparar el modelo de 50 sin cambiar el mapa publicado.
El manifest registra el modelo publicado, el shadow, el contrato, el horizonte, la calidad del
forecast y los hashes de los artefactos.
 
## 8. Validación y pendientes
 
La suite completa de la implementación termina con `276 passed`. También se comprueban el
contrato nuevo, la compilación de `src`, `scripts` y `tests`, y `git diff --check`.
 
Queda pendiente:

- entrenar y comparar los experimentos comparable y expanded;
- promover una familia 48 solo tras revisar PR-AUC, Brier, fiabilidad y recall con presupuesto;
- repetir las capas estáticas desde un DEM validado;
- acumular forecasts MeteoGalicia emparejados con observaciones para medir la degradación real;
- evaluar la utilidad operativa con presupuestos de vigilancia del 1 %, 5 % y 10 %.

Hasta entonces, la implementación es reproducible para pruebas, pero ERA5-perfect no debe
presentarse en la memoria como el rendimiento definitivo del forecast meteorológico.
 
## Referencias internas
 
- `scripts/migrate_raw_data.py`: normalización y SHA-256.
- `scripts/import_static_from_datacube.py`: bootstrap local de capas estáticas.
- `src/workflow.py`: construcción del datacubo.
- `src/features/canonical_contract.py`: contratos de 50 y 48 variables.
- `src/models/canonical_training.py`: entrenamiento y calibración.
- `scripts/train_egif_operational.py`: experimentos reproducibles.
- `scripts/run_daily_inference.py`: inferencia y shadow.
- `docs/explanations/adaptacion_modelo_48_produccion.md`: diseño completo.
