# Implementación de la alineación temporal y modelos EGIF operativos

## Propósito

El datacubo `data/processed/tabular/egif/` es una salida histórica y
retrospectiva. Sus acumulaciones meteorológicas incluyen el día de la fila,
por lo que no debe conectarse directamente al entrenamiento operativo. En
producción, para publicar el riesgo de un día objetivo, solo se dispone de la
observación cerrada hasta la hora de ejecución y del forecast posterior.

La implementación añade una segunda salida:

```text
data/processed/tabular/egif_operational/
```

El datacubo canónico no se modifica. La nueva salida conserva sus filas,
variables directas, etiquetas y FWI, pero recalcula las memorias con una
frontera estricta:

```text
target-day weather + memory through target_date - 1 day
```

Esta separación permite auditar qué se utilizó para entrenar, repetir la
transformación y evitar que una ejecución futura use por accidente las
acumulaciones inclusivas del cubo histórico.

## Estado de esta implementación

La exportación real ya está materializada en
`data/processed/tabular/egif_operational/`: cubre 2016--2023, conserva 86.494.122 filas y
29.601 celdas activas, y declara `dropped_rows=0`. Se han comprobado sus ocho footers Parquet,
el hash del datacubo padre y una muestra de continuidad diciembre--enero. Los modelos 48 y el
control alineado de 50 siguen siendo el siguiente paso; todavía no se han promovido a
`data/models/`.

El 1/01/2016 queda como calentamiento inevitable: al no existir 2015 en el datacubo de entrada,
las memorias históricas de esa fecha son nulas. Esas filas se conservan para auditoría y el
entrenamiento las cuenta como `rows_without_valid_features` y las excluye antes de ajustar el
modelo. Desde el 2/01/2016 y durante los años posteriores sí existe contexto suficiente.

## Qué se ha implementado

### Dataset alineado

`src/features/operational_benchmark.py` procesa las particiones Parquet por
lotes y mantiene una cola de hasta 30 días por celda. Esto permite cruzar los
límites entre años sin cargar los 86 millones de filas en memoria.

Se recalculan:

```text
precipitation_sum_3d
precipitation_sum_7d
precipitation_sum_14d
precipitation_sum_30d
temperature_mean_7d
relative_humidity_mean_7d
relative_humidity_mean_14d
wind_speed_mean_7d
consecutive_dry_days
```

La precipitación, temperatura, humedad y viento directos del día objetivo se
conservan como benchmark ERA5-perfect. En producción serán sustituidos por la
agregación del forecast MeteoGalicia.

El resultado incluye `metadata.json` con el hash del datacubo padre, años,
filas, celdas activas, versión de alineación, contrato, calidad de capas
estáticas y columnas recalculadas.

### Contrato de features

La familia operativa utiliza:

```text
egif-2d-48-v1
```

Son las 50 variables canónicas menos:

```text
precipitation_sum
consecutive_dry_days
```

El FWI no se entrega al modelo. Se conserva como baseline físico. Los
identificadores, coordenadas, fechas, geometrías y etiquetas tampoco pueden
aparecer en la matriz de predictores.

### Entrenamiento

El entrenamiento de 48 variables rechaza datasets sin la alineación temporal
operativa. Se generan tres modelos independientes:

```text
forecast_risk_egif_48_t1.joblib
forecast_risk_egif_48_t2.joblib
forecast_risk_egif_48_t3.joblib
```

Cada artefacto guarda su horizonte, contrato, versión del dataset, hash,
semilla, años, proveedor de entrenamiento, semántica temporal, calibrador y
métricas.

La calibración de 48 variables es independiente por horizonte:

```text
corrección de prior + Platt scaling
```

El modelo actual de 50 variables se mantiene sin modificar para rollback y
shadow. Además se puede entrenar un control de 50 variables alineado,
separado y no publicable automáticamente:

```text
forecast_risk_egif_50_aligned_t1.joblib
forecast_risk_egif_50_aligned_t2.joblib
forecast_risk_egif_50_aligned_t3.joblib
```

Este control permite comparar 48 y 50 variables con la misma semántica
temporal. Comparar directamente el modelo 48 alineado con el modelo 50
histórico no es una comparación completamente justa.

## Ejecución

### 1. Construir el dataset alineado

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/build_operational_benchmark.py \
  --source-dir data/processed/tabular/egif \
  --output-dir data/processed/tabular/egif_operational \
  --years 2016-2023 \
  --batch-size 250000 \
  --static-layer-quality provisional_import
```

La salida previa no se reemplaza por defecto. Para reconstruirla después de
validar el motivo:

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/build_operational_benchmark.py \
  --source-dir data/processed/tabular/egif \
  --output-dir data/processed/tabular/egif_operational \
  --years 2016-2023 \
  --static-layer-quality provisional_import \
  --force
```

### 2. Entrenamiento comparable

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/train_egif_operational.py \
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
  --n-estimators 400 \
  --bootstrap-samples 200
```

Los artefactos quedan en:

```text
data/models/experiments/comparable/
```

### 3. Entrenamiento ampliado

Se cambia únicamente el periodo de entrenamiento y el nombre del experimento:

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/train_egif_operational.py \
  --dataset-dir data/processed/tabular/egif_operational \
  --output-dir data/models \
  --feature-contract-version egif-2d-48-v1 \
  --train-years 2016-2020 \
  --calibration-year 2021 \
  --validation-year 2022 \
  --test-years 2023 \
  --experiment-name expanded \
  --negative-ratio 100 \
  --batch-size 100000 \
  --n-estimators 400 \
  --bootstrap-samples 200
```

### 4. Control alineado de 50 variables

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/train_egif_operational.py \
  --dataset-dir data/processed/tabular/egif_operational \
  --output-dir data/models \
  --feature-contract-version egif-2d-v1 \
  --aligned-50-control \
  --train-years 2019-2020 \
  --calibration-year 2021 \
  --validation-year 2022 \
  --test-years 2023 \
  --experiment-name aligned50 \
  --negative-ratio 100
```

### 5. Evaluación conjunta

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/evaluate_model_families.py \
  --models-root data/models \
  --dataset-dir data/processed/tabular/egif_operational \
  --legacy-dataset-dir data/processed/tabular/egif \
  --test-years 2023 \
  --bootstrap-samples 200
```

El informe se guarda en:

```text
data/models/evaluation/model_family_comparison.json
```

Las familias alineadas se comparan sobre la misma población. El modelo 50
histórico aparece marcado como `legacy_semantics`, porque fue entrenado con
otra semántica. El FWI aparece como baseline y su Brier no se interpreta como
probabilidad calibrada.

### 6. Promoción

La promoción solo acepta los tres artefactos 48, sus JSON coherentes y un
dataset con capas estáticas validadas:

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/promote_model_family.py \
  --source-dir data/models/experiments/expanded \
  --destination-dir data/models \
  --dataset-dir data/processed/tabular/egif_operational \
  --prefix forecast_risk_egif_48 \
  --dry-run
```

Con la calidad estática actual (`provisional_import`) el comando debe bloquear
la promoción real. Solo para pruebas locales se puede usar
`--allow-provisional`. Para producción científica, primero hay que repetir las
capas estáticas usando un DEM validado y volver a generar el dataset alineado.

### 6.1 Cierre de la prueba local

La familia ampliada se promocionó localmente con `--allow-provisional` para
probar el circuito completo. La promoción copió los tres `.joblib`, sus JSON y
`active_model_manifest.json` a `data/models/`; no cambia el estado de producción
científica ni autoriza todavía el despliegue en servidor.

La prueba utilizó el forecast archivado del 30 de agosto de 2026 y simuló que
el último día observado era el 26 de agosto. Antes fue necesario regenerar el
estado de AEMET utilizando la rejilla EGIF canónica, porque el estado antiguo
había sido construido con 30.697 celdas frente a las 29.601 de la rejilla
operativa.

La inferencia ahora detecta también forecasts archivados con `cell_id` de una
rejilla anterior. Si conservan `forecast_lat` y `forecast_lon`, los reasigna a
la rejilla canónica y elimina duplicados por coordenada y hora antes de
agregar. Si no hay coordenadas, falla explícitamente y exige descargar de nuevo
el forecast.

Resultado de la ejecución validada:

```text
forecast_quality=fresh
forecast_grid=1km
coverage=100 %
rows=88.803
cells=29.601
horizons=T+1,T+2,T+3
pipeline_run_mode=local_state_simulation
health_check=status ok
```

El modelo 48 fue el principal y el modelo 50 se ejecutó en shadow. Las
predicciones son válidas para comprobar el funcionamiento técnico; no deben
interpretarse como validación operativa de MeteoGalicia porque el entrenamiento
se realizó con `era5_perfect_benchmark`.

## Interpretación temporal

Los tres modelos existen porque el sistema publica tres decisiones distintas:

```text
T+1: incendio del día siguiente
T+2: incendio dentro de dos días
T+3: incendio dentro de tres días
```

Cada modelo tiene su artefacto y calibrador para poder adaptarse a la
degradación del forecast con la distancia temporal. En el benchmark
ERA5-perfect las métricas pueden ser muy parecidas entre horizontes: el tiempo
futuro se conoce retrospectivamente. La degradación real por horizonte solo se
medirá cuando se acumulen forecasts MeteoGalicia archivados junto con sus
observaciones.

## Verificaciones realizadas

Se han añadido pruebas sintéticas para comprobar:

- contrato exacto de 48 variables;
- rechazo de `fire_weather_index` como predictor;
- rechazo de un dataset sin alineación temporal;
- acumulados sin el día objetivo;
- continuidad de memoria entre diciembre y enero;
- conservación del número de filas;
- metadata y hash del dataset padre;
- generación de tres artefactos independientes;
- calibración Platt por horizonte.

Los datos reales y los modelos serializados no se versionan en Git. Deben
transferirse al servidor mediante el procedimiento de artefactos y `rsync`
documentado en `docs/deployment/`.
