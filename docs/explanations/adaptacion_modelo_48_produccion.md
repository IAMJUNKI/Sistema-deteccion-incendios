# Adaptación del modelo EGIF de 48 variables a producción

## Propósito

Este documento fija la frontera entre el experimento científico y el sistema operativo. El
modelo retrospectivo de 48 variables se construyó con ERA5-Land, donde la meteorología del día
del incendio ya es conocida. Esa configuración sirve para estudiar la señal aprendida, pero no
puede publicarse como si fuera una previsión. La producción sustituye la meteorología del día
objetivo por un forecast horario de MeteoGalicia WRF.

La implementación mantiene tres familias separadas:

| Familia | Artefacto | Uso | Calibración |
|---|---|---|---|
| `egif-2d-48-v1` | `forecast_risk_egif_48_t1/t2/t3.joblib` | Nueva candidata operativa | Prior + Platt |
| `egif-2d-v1` | `forecast_risk_egif_t1/t2/t3.joblib` | Rollback/shadow | Calibrador histórico serializado |
| `operational-risk-v1` | `forecast_risk_t1/t2/t3.joblib` | Compatibilidad legacy | Legacy |

El selector `FORECAST_MODEL_FAMILY=auto` exige una familia completa de tres horizontes y elige
48, después 50 y finalmente legacy. Una familia parcial nunca se completa con artefactos de otra.

## Contrato exacto `egif-2d-48-v1`

El datacubo conserva 50 predictores para trazabilidad. El modelo 48 elimina exactamente:

```text
precipitation_sum
consecutive_dry_days
```

La lista y su orden están definidos en
`src/features/canonical_contract.py`, no en el orden accidental de un Parquet. El contrato no
incluye `cell_id`, coordenadas, año, fecha, variables de resultado, `fire_weather_index` ni
variables auxiliares de muestreo. El FWI se almacena en el cubo como baseline, pero no se entrega
al modelo.

La inferencia valida simultáneamente:

1. versión `egif-2d-48-v1`;
2. 48 columnas y el orden exacto;
3. ausencia de las dos columnas excluidas;
4. horizonte declarado en el artefacto;
5. rejilla EGIF esperada y columnas estáticas necesarias.

Si una comprobación falla, la ejecución termina antes de publicar un mapa. No se rellenan
features ausentes para hacer compatible un artefacto de 50 variables.

## Semántica temporal

Para cada celda y horizonte se usa esta relación:

```text
issue_time + forecast del target_date + memoria observada/prevista
                               ↓
                       target_ignicion(target_date)
```

`issue_time` es la hora local de ejecución, normalizada a UTC para comparar timestamps. La fecha
del objetivo es `issue_date + horizonte`. En la agregación horaria se convierte cada instante a
`Europe/Madrid` y la ventana crítica es inclusiva: 12:00, 13:00, …, 18:00.

Para T+1 se combinan las observaciones cerradas antes de `issue_time` con el forecast del día
siguiente. Para T+2 y T+3, las horas previstas de los días intermedios también entran en las
memorias de 3, 7, 14 y 30 días. Las observaciones del día de emisión solo entran si están
cerradas y son anteriores al instante de ejecución; una fila diaria incompleta no se interpreta
como el día completo.

El benchmark histórico aplica la misma geometría temporal de forma retrospectiva: la fila de
features es la del `target_date`, sus acumulaciones usan días anteriores y `issue_date` se
reconstruye como `target_date - horizon`. Esto permite comparar el contrato entre entrenamiento e
inferencia sin afirmar que ERA5 sea una predicción.

## Dataset operativo alineado

El Parquet histórico del datacubo declara `No temporal shift; meteorological
accumulations include date T.`. Por eso no se debe entrenar la familia 48
directamente sobre `data/processed/tabular/egif/`. La exportación alineada se
construye con:

```bash
PYTHONPATH=. python scripts/build_operational_benchmark.py \
  --source-dir data/processed/tabular/egif \
  --output-dir data/processed/tabular/egif_operational \
  --years 2016-2023 \
  --static-layer-quality provisional_import
```

La salida recalcula memorias y rachas con datos hasta
`target_date - 1 day`, conserva el cubo original sin modificar y mantiene el
contexto entre años. El entrenamiento operativo exige el metadata de esta
exportación y falla si se le entrega el cubo inclusivo.

La explicación completa, incluida la comparación y la promoción, está en
`docs/explanations/implementacion_alineacion_operativa.md`.

## Entrenamiento reproducible

El script `scripts/train_egif_operational.py` acepta intervalos y listas de años. Las dos
configuraciones que se deben conservar son:

```text
comparable: train 2019–2020, calibration 2021, validation 2022, blind test 2023
expanded:   train 2016–2020, calibration 2021, validation 2022, blind test 2023
```

Los años se refieren al día objetivo del benchmark. Se conservan todos los positivos y se
submuestrean negativos de forma determinista. La evaluación y la calibración recorren la
población completa, con muestreo acotado únicamente para el ajuste del calibrador cuando la
memoria lo exige.

El script protege los dos experimentos para que no se sobrescriban: al indicar
`--experiment-name comparable` o `--experiment-name expanded`, los artefactos
se escriben en `data/models/experiments/<nombre>/`. El nombre `production`
mantiene los artefactos publicados directamente en `data/models/`.

El entrenamiento de la familia 48 usa LightGBM y guarda tres modelos independientes. Con
`--include-xgboost-challenger` se guardan además artefactos challenger con el sufijo
`_xgboost_challenger`; no se promueven automáticamente. Cada JSON incluye años, hash del dataset,
semilla, contrato, horizonte, fuente de entrenamiento, semántica temporal, método de
calibración y métricas de validación/test.

Ejemplos:

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/processed/tabular/egif_operational \
  --output-dir data/models \
  --feature-contract-version egif-2d-48-v1 \
  --train-years 2019-2020 \
  --calibration-year 2021 \
  --validation-year 2022 \
  --test-years 2023 \
  --experiment-name comparable \
  --negative-ratio 100 \
  --batch-size 100000 \
  --n-estimators 400
```

La ampliada solo cambia `--train-years 2016-2020` y
`--experiment-name expanded`:

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/processed/tabular/egif_operational \
  --output-dir data/models \
  --feature-contract-version egif-2d-48-v1 \
  --train-years 2016-2020 \
  --calibration-year 2021 \
  --validation-year 2022 \
  --test-years 2023 \
  --experiment-name expanded
```

No debe ejecutarse sobre un dataset que no contenga realmente 2016–2018: el error de partición
ausente es preferible a entrenar sin avisar con menos años.

El control alineado de 50 variables se entrena aparte y no se publica:

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/processed/tabular/egif_operational \
  --output-dir data/models \
  --feature-contract-version egif-2d-v1 \
  --aligned-50-control \
  --train-years 2019-2020 \
  --calibration-year 2021 \
  --validation-year 2022 \
  --test-years 2023 \
  --experiment-name aligned50
```

La calibración 48 aplica corrección de prior por el submuestreo y después Platt scaling sobre
2021. La validación 2022 y el test 2023 se transforman con el calibrador ya cerrado. Brier se
calcula sobre la probabilidad calibrada; PR-AUC y métricas de ranking se calculan sobre la
población completa.

## Migración y datacubo

`misc/Datos/raw` era una carpeta de intercambio, no la interfaz del workflow. El script
`scripts/migrate_raw_data.py` conoce sus nombres reales y los normaliza:

```text
misc/Datos/raw/meteorology/era5       → data/raw/meteorology/era5
misc/Datos/raw/meteorology/FWI        → data/raw/meteorology/fwi
misc/Datos/raw/fire_history            → data/raw/fire_history
misc/Datos/raw/human_activity          → data/raw/human_activity/galicia-220101-free-shp
misc/Datos/raw/landcover/*.tif         → data/raw/corine/U2018_CLC2018_V2020_20u1.tif
misc/Datos/raw/galicia_boundary/*.geojson → data/raw/igm/galicia_boundary.geojson
```

La migración ya se aplicó con copia y SHA-256; el manifiesto queda en
`data/raw/migration_manifest.json`. El origen se conserva hasta completar el workflow. Para
eliminarlo más adelante hay que invocar explícitamente `--remove-source` después de validar las
salidas.

El cubo ampliado se construye con:

```bash
PYTHONPATH=. python -m src.workflow \
  --start-year 2016 \
  --end-year 2023 \
  --rebuild-static \
  --no-download-missing-era5
```

El comando necesita el DEM de Copernicus si las capas estáticas aún no existen. El resultado
esperado es una rejilla EGIF de aproximadamente 29.601 celdas, particiones 2016–2023, metadata
de 50 predictores y FWI almacenado como baseline. Que el cubo tenga 50 columnas no contradice el
modelo operativo de 48: el filtrado sucede en el contrato de entrenamiento/inferencia.

## Inferencia y shadow

Con los tres artefactos 48 presentes, `FORECAST_MODEL_FAMILY=auto` selecciona automáticamente la
nueva familia. También se puede fijar explícitamente:

```bash
FORECAST_MODEL_FAMILY=egif_48 \
PYTHONPATH=. python scripts/run_daily_inference.py --no-stale
```

Para comparar sin alterar el mapa publicado:

```bash
SHADOW_50_MODEL=true FORECAST_MODEL_FAMILY=egif_48 \
PYTHONPATH=. python scripts/run_daily_inference.py --no-stale
```

El resultado principal contiene `prob_risk`; si el shadow está disponible añade sus
probabilidades y diferencia de ranking. El manifest conserva qué familia generó el mapa y el
JSON de cada modelo aporta el contrato y hash del artefacto.

La cadena meteorológica sigue siendo:

```text
MeteoGalicia WRF 1 km → WRF 04 km → AEMET degradado habilitado → stale explícito
```

La calidad del forecast (`fresh`, `fresh_fallback`, `fresh_aemet_degraded`, `stale`, etc.) no
modifica el contrato de features y siempre aparece en el mapa, el output y el manifest.

## Qué queda pendiente

- No hay todavía histórico de forecasts MeteoGalicia emparejados con observaciones; ERA5-perfect
  es un benchmark, no la medición final de error operativo.
- La calibración Platt puede trasladarse al cambiar ERA5 por WRF; debe auditarse tras archivar
  suficientes temporadas.
- 2023 es el test ciego y no debe usarse iterativamente para seleccionar el modelo.
- El modelo es tabular; no aprende explícitamente la dependencia entre celdas vecinas.
- La utilidad para movilizar agentes debe evaluarse con presupuestos reales del 1 %, 5 % y 10 %
  y con restricciones de accesibilidad y medios.
