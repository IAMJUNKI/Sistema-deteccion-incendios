# 07. Análisis del Dataset Tabular 2D: Ingeniería de Variables y Robustez del Protocolo de Evaluación

---

## 1. ¿Qué se ha hecho?

- **Auditoría de la trazabilidad del target:** verificación del origen real de la variable objetivo utilizada en el benchmark. Se ha comprobado que el fichero `fire_history.xml` (EGIF-MITECO) disponible en el repositorio compartido cubre exclusivamente el periodo 2014-2017 (9.550 registros de Galicia), sin solape con el periodo de entrenamiento y test (2019-2023), y que todos los scripts de experimentación (`scripts/run_*_experiment.py`) consumen el target derivado de NASA FIRMS.
- **Construcción del Dataset 2D Enriquecido** (`construye_dataset_2d_enriquecido.py`): 56.052.722 filas × 41 columnas para el periodo 2019-2023, con arquitectura de target intercambiable (FIRMS / EGIF) parametrizada.
- **Ingeniería de variables adicionales** derivada de la comparación sistemática con el pipeline 3D: `dias_sin_lluvia`, Vapor Pressure Deficit (`vpd_t1`), memorias móviles de humedad y viento a 7 días, acumulados de precipitación a 3 y 14 días, desglose one-hot de las 8 macro-clases de combustible y de las 4 orientaciones, y codificación cíclica de la estacionalidad.
- **Análisis de discriminación temporal frente a espacial** de las variables meteorológicas, controlando el efecto estacional.
- **Evaluación comparativa controlada** (`entrena_2d_comparativa.py`) y marco de evaluación robusta con validación temporal expansiva, ablación por grupos de variables e intervalos de confianza bootstrap (`evalua_2d_robusto.py`).

## 2. ¿Por qué se ha hecho?

- **Garantizar la trazabilidad del *ground truth*:** la memoria del TFM afirma emplear el EGIF verificado en campo. La discrepancia debe resolverse (renombrando la fuente o incorporando el EGIF real) antes de la redacción final.
- **Asegurar la validez estadística de las conclusiones de modelado:** con 102 igniciones en el conjunto de test de 2023, las diferencias de rendimiento que ordenan el benchmark oficial (2 a 5 puntos de Recall) corresponden a 2-5 incendios individuales, magnitud indistinguible del ruido de muestreo. La selección del "modelo oficial de producción" requiere una base empírica más sólida.
- **Agotar el potencial del enfoque tabular guiado por conocimiento del dominio** antes de atribuir superioridad a arquitecturas más complejas, cuantificando el aporte marginal real de cada bloque de variables.

## 3. ¿Cómo se ha hecho?

1. **Auditoría del target:** parseo directo del XML del EGIF con `xml.etree.ElementTree`, filtrado por códigos de provincia de Galicia (15, 27, 32, 36) y tabulación por año de detección. Recuento independiente de positivos en los `.parquet` del dataset tabular mediante lectura selectiva con filtros de PyArrow.
2. **Construcción del dataset:** procesamiento por bloques de 2.000 celdas (la carga íntegra de un año, 11,2 M de filas, excede la memoria disponible), con margen de 35 días del año anterior para que las ventanas móviles de 30 días dispongan de historia completa. Las ventanas se calculan en forma matricial (celdas × días) mediante sumas acumuladas, y todas ellas excluyen el día en curso (`shift(1)` previo) en cumplimiento de la regla anti-data leakage.
3. **Análisis de discriminación:** comparación de medias entre clases bajo tres condiciones de control progresivo — año completo, restricción a la temporada estival, y restricción a las fechas exactas en que se registró alguna ignición (control espacial puro).
4. **Evaluación:** protocolo idéntico al del equipo (LightGBM, Hard Negative Mining 1:50, división temporal estricta, test sobre el año completo sin submuestrear), variando exclusivamente el conjunto de variables. La evaluación robusta añade tres años de test independientes, siete configuraciones de variables e intervalos de confianza al 90 % obtenidos por remuestreo bootstrap de la clase positiva.

## 4. ¿Por qué se han elegido estas tecnologías?

- **Vapor Pressure Deficit (VPD):** integra temperatura y humedad relativa en la magnitud físicamente relevante — la demanda evaporativa de la atmósfera sobre el combustible fino. Una humedad del 30 % a 15 °C y a 35 °C representan estados de desecación radicalmente distintos que la humedad relativa aislada no distingue.
- **Codificación cíclica de la estacionalidad (seno/coseno del día del año):** evita la discontinuidad artificial entre el 31 de diciembre y el 1 de enero, que en una codificación ordinal aparecen como los días más alejados del calendario siendo meteorológicamente contiguos.
- **Desglose *one-hot* del combustible:** el indicador agregado `combustible_pct_forestal` asigna el mismo valor a una celda de matorral y a una de frondosas, pese a que su comportamiento frente al fuego es opuesto (riesgo extremo frente a riesgo bajo).
- **Bootstrap sobre la clase positiva:** en problemas de prevalencia extrema la incertidumbre de las métricas está dominada por el número de positivos; remuestrearlos proporciona el intervalo de confianza pertinente sin asumir normalidad.
- **Procesamiento por bloques con reanudación:** garantiza reproducibilidad en máquinas con memoria limitada y permite reanudar tras una interrupción sin recalcular lo ya construido.

## 5. ¿Qué conseguimos con ello?

- **Integridad metodológica de la memoria:** identificación, antes de la redacción final, de una discrepancia entre la fuente de datos documentada y la efectivamente empleada, con tres vías de resolución evaluadas. La vía mixta recomendada (entrenamiento con FIRMS 2019-2023 y validación externa con EGIF 2014-2017 sobre incendios verificados en campo) transforma la limitación en una fortaleza: validación cruzada entre fuentes independientes.
- **Criterio de selección de modelos defendible:** un protocolo multi-año con intervalos de confianza permite afirmar qué diferencias son concluyentes y cuáles no, evitando conclusiones no sostenibles ante el tribunal.
- **Evidencia de que el protocolo de evaluación condiciona la conclusión:** sobre el test único de 2023 el conjunto enriquecido presenta un rendimiento inferior al de referencia (Recall@FPR≤5 % de 17,7 % frente a 20,6 %); promediando tres años de test independientes la relación se invierte (42,2 % frente a 37,5 %). Dado que la totalidad de los intervalos de confianza del año 2023 se solapan entre sí, ninguna ordenación de modelos basada exclusivamente en ese conjunto resulta concluyente.
- **Cuantificación del aporte real de las variables incorporadas:** en el ejercicio de 2022, que concentra 703 igniciones y constituye por tanto el escenario de mayor potencia estadística de la serie, la incorporación de las memorias móviles de humedad y viento eleva el Recall@FPR≤5 % del 57,0 % (IC 90 %: 53,8-60,5) al 75,8 % (IC 90 %: 73,1-78,7), con intervalos disjuntos.
- **Identificación de la métrica de selección adecuada:** el ROC-AUC permanece prácticamente invariante entre configuraciones (0,847-0,857) mientras el Recall@FPR≤5 % varía entre el 36,4 % y el 42,2 %. Con una prevalencia del orden de 1 entre 75.000, el ROC-AUC está dominado por la clase negativa y se satura, por lo que se recomienda relegarlo a métrica de contexto y ordenar el benchmark por la métrica operativa acompañada de sus intervalos de confianza.
- **Caracterización de la naturaleza de los predictores:** las variables de sequía acumulada discriminan la dimensión temporal (qué días presentan riesgo) pero no la espacial (qué celda concreta), mientras que las térmicas e higrométricas discriminan ambas. Esta distinción orienta la ingeniería de variables futura hacia predictores con variabilidad intra-diaria entre celdas (accesibilidad humana, historial de recurrencia, heterogeneidad del combustible).
- **Infraestructura reutilizable:** dataset con target parametrizable que no requiere reconstrucción si el equipo modifica la fuente del ground truth.

---

**Documentos generados:**
`construye_dataset_2d_enriquecido.py` · `entrena_2d_comparativa.py` · `evalua_2d_robusto.py` · `quantile_mapping_era5.py` · `Datos/resultados_2d/*.csv`
