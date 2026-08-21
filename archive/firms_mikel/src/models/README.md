# Módulo: models — Modelado y Calibración

**Fase 4 del proyecto.** Entrenamiento, evaluación con validación temporal rigurosa y calibración de umbrales de riesgo accionables.

## Responsabilidad

- Entrenar modelos en orden de complejidad: Regresión Logística (baseline) → XGBoost / LightGBM.
- Aplicar validación temporal por bloques anuales completos (prohibido K-Fold aleatorio).
- Evaluar con métricas orientadas a clases desbalanceadas: AUC-ROC, PR-AUC, F1.
- Calibrar umbrales de riesgo con la tasa real de incidencia del conjunto de validación.
- Serializar el modelo y la tabla de umbrales calibrados.

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

## Regla crítica

Nunca consultar el conjunto de test hasta que el modelo esté completamente entrenado y los hiperparámetros fijados con el conjunto de validación.
