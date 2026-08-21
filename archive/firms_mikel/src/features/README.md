# Módulo: features — Ingeniería de Características

**Fase 3 del proyecto.** Enriquecimiento del dataset con variables acumuladas y derivadas, y auditoría estricta contra el data leakage temporal.

## Responsabilidad

- Calcular acumulados meteorológicos en ventanas de 3, 7, 14 y 30 días.
- Extraer variables exclusivamente en la franja de mayor riesgo: **12h-18h** (temperatura máxima, humedad mínima, viento máximo).
- Calcular días consecutivos sin lluvia (estrés hídrico).
- Añadir variables de proximidad humana (distancia a carreteras, núcleos urbanos vía OSM).
- Calcular historial de incendios acumulado en el vecindario de cada celda.
- Garantizar que ninguna feature use información del día T o posterior para predecir T.

## Entregable

**Dataset Maestro** en formato `.parquet` con todas las features finales, listo para ML.

## Reglas críticas anti-leakage

1. Para predecir el riesgo del día `T`, solo usar datos hasta `T-1`.
2. La ventana meteorológica 12h-18h debe calcularse con datos del día `T-1`.
3. Excluir como negativos cualquier celda a ±5 días de un incendio real.
