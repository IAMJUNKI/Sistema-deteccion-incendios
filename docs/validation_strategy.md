# Validación robusta del modelo

La selección provisional debe contrastarse contra el conjunto completo que
publique `metadata.json`. Antes de usar 2023 se repite la comparación con
submuestras deterministas que conservan todos los positivos y eligen negativos
mediante un hash de `(cell_id, fecha)`; así se reparten por espacio y tiempo y
se evita la fuga del antiguo residuo fijo `cell_id % 25`. Después se analizan
métricas por mes en 2022.

El código reutilizable vive en `src/entrenamiento` y `src/modeling`; el notebook posterior solo
ejecuta los experimentos, visualiza tablas y guarda los resultados. 2023 no
puede aparecer en los parámetros de entrenamiento, validación o selección.
