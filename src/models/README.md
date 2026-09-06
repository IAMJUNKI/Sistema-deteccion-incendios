# Módulo: models — Modelado y Calibración

**Fase 4 del proyecto.** Entrenamiento, evaluación con validación temporal rigurosa y calibración de umbrales de riesgo accionables.

## Responsabilidad

- Entrenar modelos en orden de complejidad: Regresión Logística (baseline) → XGBoost / LightGBM.
- Aplicar validación temporal por bloques anuales completos (prohibido K-Fold aleatorio).
- Evaluar con métricas orientadas a clases desbalanceadas: AUC-ROC, PR-AUC, F1.
- Calibrar umbrales de riesgo con la tasa real de incidencia del conjunto de validación.
- Serializar el modelo y la tabla de umbrales calibrados.
- Mantener un artefacto independiente y calibrado para cada horizonte T+1,
  T+2 y T+3, con versión explícita del esquema de features.
- Calcular explicaciones TreeSHAP desde el artefacto serializado cuando la
  dependencia opcional `shap` esté disponible.

## Contratos de features y familias de modelos

El datacubo conserva el contrato histórico de 50 predictores
(`egif-2d-v1`), pero la familia operativa recomendada es
`egif-2d-48-v1`. Esta familia utiliza exactamente 48 columnas: las 50
canónicas menos `precipitation_sum` y `consecutive_dry_days`. El FWI no es un
predictor del modelo; se conserva como baseline físico independiente.

La producción mantiene tres artefactos independientes y contractualmente
validados:

```text
data/models/forecast_risk_egif_48_t1.joblib
data/models/forecast_risk_egif_48_t2.joblib
data/models/forecast_risk_egif_48_t3.joblib
```

La familia `forecast_risk_egif_t{1,2,3}.joblib` de 50 variables permanece
disponible para rollback o shadow. La inferencia nunca mezcla familias ni
acepta un artefacto cuyo orden de columnas no coincida con su contrato.

El control científico alineado de 50 variables se guarda con el prefijo
`forecast_risk_egif_50_aligned_t{1,2,3}.joblib` dentro del experimento
`aligned50`. No se publica automáticamente y no sustituye a los artefactos
50 históricos de rollback.

Los modelos de 48 variables se entrenan con semántica de benchmark
`era5_perfect_benchmark`: las features meteorológicas son las del día objetivo
y la fecha de emisión se retrocede según el horizonte. Esto permite comparar
la arquitectura científica, pero no equivale todavía a evaluar un forecast
real de MeteoGalicia. La producción sustituye las features futuras por la
secuencia de observaciones disponibles y previsiones meteorológicas.

El entrenamiento solo acepta el dataset alineado
`data/processed/tabular/egif_operational/`, cuya metadata declara
`egif-operational-tminus1-v1`. El datacubo canónico conserva la semántica
retrospectiva inclusiva y se mantiene para auditoría.

## Partición temporal

```
Experimento comparable:  entrenamiento 2019–2020 · calibración 2021 ·
                         validación 2022 · test ciego 2023
Experimento ampliado:    entrenamiento 2016–2020 · calibración 2021 ·
                         validación 2022 · test ciego 2023
```

La partición ampliada solo puede ejecutarse cuando existan los datasets EGIF
de 2016–2018. La evaluación se realiza sobre la población completa; el
submuestreo se limita al ajuste del modelo.

## Entregable

- Modelo serializado (`.joblib`).
- Calibrador independiente por horizonte: corrección de prior seguida de
  Platt para la familia de 48 variables.
- Tabla de niveles/ranking y reporte de métricas en calibración, validación y
  test.
- Metadatos reproducibles: versión del modelo, contrato, dataset, años,
  fuente meteorológica, semilla, hash y métricas.
- XGBoost puede entrenarse como challenger con
  `--include-xgboost-challenger`; no se publica automáticamente como modelo
  operativo.

## Regla crítica

Nunca consultar el conjunto de test hasta que el modelo esté completamente entrenado y los hiperparámetros fijados con el conjunto de validación.
