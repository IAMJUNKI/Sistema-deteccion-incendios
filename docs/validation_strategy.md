# Validación robusta del modelo

La selección provisional usa `temporal_compacto` (44 variables). Antes de usar
2023 se repite la comparación contra el conjunto completo con varios residuos
de `cell_id % 25`; cada residuo conserva todos los positivos y un 4 % distinto
de negativos. Después se analizan métricas por mes en 2022.

El código reutilizable vive en `src/modeling`; el notebook posterior solo
ejecuta los experimentos, visualiza tablas y guarda los resultados. 2023 no
puede aparecer en los parámetros de entrenamiento, validación o selección.
