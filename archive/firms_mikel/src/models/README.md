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

## Partición temporal

```
Entrenamiento:  2019 · 2020 · 2021
Validación:     2022
Test (ciego):   2023 · 2024
```

## Entregable

- Modelo serializado (`.pkl` o `.json` para XGBoost).
- Tabla de umbrales calibrados: Bajo / Moderado / Alto / Extremo.
- Reporte de métricas en train, val y test.
- `data/models/forecast_risk_t1.joblib`, `forecast_risk_t2.joblib` y
  `forecast_risk_t3.joblib`, más sus JSON de métricas.

## Regla crítica

Nunca consultar el conjunto de test hasta que el modelo esté completamente entrenado y los hiperparámetros fijados con el conjunto de validación.
