# Archivo histórico

`firms_mikel/` conserva el código, notebooks, resultados y documentación del
experimento anterior basado en los Parquet de Mikel y target NASA FIRMS. No se
importa desde el pipeline activo ni constituye evidencia del modelo EGIF.

Se mantiene por trazabilidad académica: su contrato temporal, fuente de target,
rejilla y columnas no son compatibles con el datacubo actual.

`vecindad/` documenta las variables de contexto espacial: qué se intentó, qué se
descartó y por qué, el resultado medido (+5,67 pp de recall, concluyente en test
pareado) y el motivo por el que **no se ha adoptado**, que es operativo y no
estadístico. A diferencia de `firms_mikel/`, el código no está aquí: sigue en
`src/entrenamiento/vecindad.py`, funcionando y probado, tras una bandera apagada
por defecto. Lo que se archiva es la evidencia y el razonamiento.
