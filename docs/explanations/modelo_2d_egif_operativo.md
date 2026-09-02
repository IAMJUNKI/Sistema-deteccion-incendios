# Modelo 2D EGIF y previsión meteorológica operativa

## Propósito

El producto operativo estima, para cada celda activa de 1 km y para cada fecha
futura, la probabilidad calibrada de que se produzca una ignición:

```text
P(target_ignicion = 1 | meteorología prevista, terreno, combustible, actividad humana, historial)
```

El sistema no predice la atmósfera a partir de la temperatura actual. La
previsión meteorológica la produce un modelo numérico externo —WRF de
MeteoGalicia en el caso principal— y el modelo EGIF aprende la relación entre
esa previsión y el riesgo de ignición. ERA5-Land solo se utiliza para construir
el benchmark histórico y no representa el error que tendría un forecast real.

## Contrato canónico

La fuente normativa es `metadata.json` del datacubo tabular EGIF. El contrato
actual es `egif-2d-v1` y contiene exactamente 50 predictores. Incluye:

- fracciones de cobertura CORINE;
- elevación, dispersión de elevación, pendiente y ocho fracciones de aspecto;
- carreteras, edificios y superficies residenciales;
- temperatura, humedad, viento, VPD y precipitación diaria;
- acumulados y memorias de 3, 7, 14 y 30 días;
- días secos consecutivos.

No son predictores `cell_id`, `fecha`, año, coordenadas auxiliares, resultados
del incendio ni `is_near_ignition_25x25_10d`. La variable
`precipitation_sum_1d` se considera duplicada y se rechaza. Si un metadata
declara 47 columnas, la ejecución se detiene: antes de entrenar hay que
resolver la versión del contrato, no seleccionar silenciosamente una lista.

La rejilla canónica tiene 29.601 celdas activas. La rejilla legacy de 30.697
celdas se conserva para el rollback del modelo antiguo, pero no puede mezclarse
con los artefactos EGIF.

## Conversión meteorológica

Todos los proveedores terminan en el contrato horario interno:

```text
temperature_c
relative_humidity_pct
precipitation_mm
wind_speed_kmh
wind_dir_deg
valid_time (UTC)
issued_at (UTC)
```

La agregación se realiza en `Europe/Madrid`, y la ventana crítica incluye las
horas 12:00, 13:00, ..., 18:00. Se derivan máximas, mínimas, medias,
precipitación, VPD y memorias. El almacenamiento de timestamps permanece en
UTC para que un cambio horario de verano no desplace las fechas; solo la
agrupación diaria usa la zona de Galicia.

La prioridad operativa es:

```text
WRF 1 km → WRF 04 km → AEMET municipal degradado → último forecast válido stale
```

MeteoSIX se consulta por puntos representativos en lotes de 20 y se asigna a la
rejilla final. No se hacen 29.601 peticiones. WRF 04 km conserva el producto de
riesgo a 1 km, pero no recupera el detalle meteorológico perdido; por eso queda
marcado como `fresh_fallback`.

AEMET consulta municipios, no una malla WRF. En producción debe utilizarse un
catálogo denso de municipios mediante `AEMET_MUNICIPALITIES_FILE`, no las cuatro
capitales de la configuración de prueba. `probPrecipitacion` nunca se convierte
en milímetros. Si falta precipitación cuantitativa, el horizonte queda
degradado o no disponible; el proxy cero solo se permite para una prueba
explícita y queda etiquetado.

Cada resultado registra proveedor, modelo, versión, resolución, emisión,
descarga, validez, cobertura, horas ausentes, variables ausentes, calidad y
distancia de asignación. El forecast bruto se archiva antes de la agregación.

## Estado meteorológico y memoria

Para construir las memorias de sequedad, la inferencia solo utiliza observación
cerrada hasta la frontera de emisión. Una ejecución a las 05:00 del día `D` no
usa la fila diaria de `D`, porque aún no representa el día completo. Para T+2 y
T+3 se incorporan los días intermedios previstos y el día objetivo previsto;
las observaciones posteriores a `issue_time` nunca entran.

El estado mínimo contiene 30 días completos por celda. AEMET puede hacer el
bootstrap mediante dos consultas de 15 días, pero su climatología llega con
retraso. El colector horario acumula observaciones actuales y cierra un día
cuando alcanza la cobertura configurada. MeteoGalicia forecast no sustituye
este estado histórico: una predicción no es una observación pasada.

Las columnas legacy (`tmax_vc`, `rhmin_vc`, `vmax_vc`, `prec_dia`) siguen
existiendo como aliases de rollback. Si el estado solo contiene esas columnas,
las columnas EGIF derivadas se marcan como `state_feature_quality=legacy_proxy`.
Esto permite pruebas técnicas, pero no debe confundirse con una historia
horaria completa.

## Targets y entrenamiento

Cada modelo tiene un artefacto independiente:

```text
forecast_risk_egif_t1.joblib
forecast_risk_egif_t2.joblib
forecast_risk_egif_t3.joblib
```

La etiqueta del benchmark histórico para T+h es `target_ignicion` del día
`issue_date + h`. Los modelos no comparten calibrador. La configuración inicial
usa LightGBM, 2019–2021 para entrenamiento, 2022 para calibración/validación y
2023 como test ciego. Solo se submuestrean negativos en train; validación y test
se recorren completos por lotes. Las métricas incluyen PR-AUC, ROC-AUC, Brier,
calibración, recall a FPR fija y recall al presupuesto diario del 1 %.

XGBoost se puede entrenar como challenger con
`--include-xgboost-challenger`. Sus artefactos tienen sufijo
`_xgboost_challenger` y no sustituyen LightGBM automáticamente. Solo se
promueve si mejora consistentemente las métricas operativas sobre la misma
población y con incertidumbre compatible.

Los artefactos incluyen modelo, calibrador, horizonte, contrato, versión del
dataset, contexto meteorológico, semilla, hash y métricas. La inferencia carga
el artefacto; nunca entrena LightGBM durante la publicación diaria.

## Comandos

Preparar la rejilla vectorial canónica a partir del NetCDF EGIF:

```bash
PYTHONPATH=. python scripts/prepare_operational_grid.py \
  --cube data/external/egif/galicia_1km.nc \
  --output data/processed/grid/galicia_grid_1km_egif.parquet
```

Entrenar los tres modelos:

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/external/egif \
  --output-dir data/models
```

Para conservar también el challenger:

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/external/egif \
  --output-dir data/models \
  --include-xgboost-challenger
```

Antes de ejecutar inferencia, configurar `GRID_PATH` a la rejilla EGIF,
`EGIF_DATASET_DIR` al dataset canónico y `FORECAST_MODEL_FAMILY=auto` o
`canonical`. La ejecución falla si no están los tres modelos o si la rejilla no
tiene 29.601 celdas y las variables estáticas canónicas.

```bash
PYTHONPATH=. python scripts/check_meteogalicia_forecast.py
PYTHONPATH=. python scripts/run_daily_inference.py --no-stale
PYTHONPATH=. python scripts/check_operational_run.py
```

Durante la transición se puede activar `SHADOW_LEGACY_MODEL=true`. El resultado
conserva `shadow_prob_risk`, `shadow_percentil_riesgo` y `shadow_rank_delta`
para inspeccionar si el nuevo modelo cambia de forma material las prioridades.

## Salidas de una ejecución

Para `predicciones_operativas.parquet` se guardan también:

```text
predicciones_operativas.features.parquet
predicciones_operativas.json
predicciones_operativas.manifest.json
```

El manifest enlaza el forecast archivado, las features, checksums, modelos,
calidad, edad, cobertura, contrato y estado meteorológico. Los estados
`fresh`, `fresh_fallback`, `fresh_aemet_degraded` y `stale` son visibles. Un
forecast incompleto no se rellena silenciosamente y no se presenta como mapa
normal.

## Interpretación preventiva

La probabilidad calibrada expresa riesgo condicionado al forecast seleccionado;
no integra todavía todos los escenarios posibles de la atmósfera. El percentil
es una herramienta de priorización relativa y el nivel táctico se calcula para
presupuestos de recursos. El sistema puede seleccionar el 1 %, 5 % o 10 % de
celdas, pero la movilización final debe añadir accesibilidad, disponibilidad de
medios, exposición de población y confirmación del centro de mando.
