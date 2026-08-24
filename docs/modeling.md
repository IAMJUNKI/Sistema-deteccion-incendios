# Modelado y selección de variables

Esta fase es independiente de la construcción del datacubo. El producto de datos
permanece inmutable: se lee desde `data/processed/tabular/egif` mediante el
metadato publicado por la exportación.

## Protocolo temporal

- Entrenamiento: 2019–2021.
- Validación y selección de variables: 2022.
- Test ciego final: 2023. No se usa para elegir variables ni hiperparámetros.

Los conjuntos `completo`, `sin_topografia_redundante` y `temporal_compacto` son
hipótesis experimentales. Se comparan con PR-AUC como métrica primaria, además
de ROC-AUC y recall dentro del 1 % de celdas con mayor riesgo. Solo el conjunto
que funcione mejor y de forma estable en 2022 podrá convertirse en el contrato
de producción.

La segunda ronda separa cada decisión: calendario redundante, topografía
redundante y las alternativas meteorológicas diaria frente a ventana 12–18 h.

Las columnas de identificación, fecha, año y resultados EGIF están bloqueadas
como predictores en `src/modeling/features.py`.
