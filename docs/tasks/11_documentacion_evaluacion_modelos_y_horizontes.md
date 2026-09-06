# 11. Documentación de Evaluación de Modelos, Desacoplo de Horizontes y Comparativa Parsimoniosa (EGIF 48 vs. EGIF 50)

---

## 1. ¿Qué se ha hecho?
* **Creación de la Guía Explicativa de Modelos y Métricas:** Se ha generado el documento [docs/explanations/guia_explicativa_modelos_y_metricas_egif.md](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/docs/explanations/guia_explicativa_modelos_y_metricas_egif.md), un recurso pedagógico y riguroso diseñado para alinear al equipo técnico y preparar la defensa del TFM ante el tribunal.
* **Formalización de la Especialización de Horizontes ($T+1, T+2, T+3$):** Se ha documentado exhaustivamente por qué no se emplea un único predictor genérico, detallando las diferencias en etiquetas temporales, semillas de submuestreo (`seed = 42 + h`), splits de árboles en LightGBM y curvas de calibración sigmoide independientes (Platt scaling).
* **Justificación del Principio de Parsimonia (48 vs. 50 Variables):** Se ha desglosado el motivo físico de suprimir `precipitation_sum` del día objetivo y `consecutive_dry_days`, probando que el nuevo contrato `egif-2d-48-v1` elimina redundancias y problemas de latencia sin menoscabar la capacidad predictiva.
* **Integración en la Memoria del TFM:** Se ha incorporado la subsección `5.15` en el borrador oficial de la memoria [docs/tfm_borrador_memoria.md](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/docs/tfm_borrador_memoria.md), consolidando el marco teórico y la justificación metodológica de la evaluación retrospectiva sobre el test ciego de 2023 (10.804.365 filas y 530 igniciones).

## 2. ¿Por qué se ha hecho?
* **Alineación del Equipo y Claridad Operativa:** Los miembros del equipo necesitaban comprender el significado exacto de los gráficos de evaluación generados (Figuras 1 y 2 del informe), los cuales presentaban conceptos de alta densidad técnica (PR-AUC, Recall con FPR del 5%, Recall en el 1% superior diario y gain interno normalizado).
* **Defensa Científica y Metodológica ante el Tribunal:** Para evitar objeciones habituales de comités evaluadores —tales como asumir que un único modelo puede evaluar horizontes temporales distintos o dudar de si la reducción de variables perjudica la precisión—, es imprescindible justificar el diseño experimental pareado (`EGIF 50 control` vs. `EGIF 48 comparable` vs. `EGIF 48 ampliado` vs. `FWI CEMS`).

## 3. ¿Cómo se ha hecho?
* **Análisis del Flujo de Entrenamiento:** Se revisó la implementación en `src/models/canonical_training.py` para evidenciar cómo cada horizonte entrena con su propia etiqueta ($issue\_date + h$), muestra de negativos determinista y calibrador de probabilidades.
* **Traducción de Métricas a Lenguaje Operativo:** Se reformuló el valor de las métricas abstractas en capacidades de decisión para protección civil: explicar que el *Recall en el top 1%* representa la capacidad de detectar el 10%–11% de los fuegos patrullando únicamente el 1% del territorio más susceptible.
* **Estructuración en Capas:** La guía se organizó con un resumen ejecutivo de 2 minutos para consulta rápida, seguido de diagramas Mermaid explicativos del flujo de datos y análisis pormenorizado de las familias evaluadas.

## 4. ¿Por qué se han elegido estas tecnologías?
* **Diagramas Mermaid en Markdown:** Permiten representar visualmente el flujo de datos temporales (observaciones EMA cerradas en $T+1$ vs. forecasts encadenados en $T+2$ y $T+3$) directamente en el repositorio sin recurrir a imágenes externas difíciles de versionar.
* **Platt Scaling por Horizonte:** Se justificó frente a la calibración isotónica global porque produce transformaciones sigmoides suaves y bien condicionadas para extrapolaciones en colas de baja prevalencia ($\approx 0{,}005\%$).
* **LightGBM Gain Normalizado:** Utilizado en la Figura 2 para visualizar qué variables aportan más reducción de pérdida en los splits, complementando la visión física del proceso de combustión (VPD, humedad de combustible y accesibilidad humana).

## 5. ¿Qué conseguimos con ello?
* **Discurso Coherente y Unificado:** Todo el equipo de trabajo comparte el mismo lenguaje conceptual y entiende los gráficos generados.
* **Validación de la Hipótesis del TFM:** Se demuestra de forma concluyente que `EGIF 48 ampliado` cuadruplica la eficacia del índice estándar europeo FWI (Recall top 1% de 10% vs. 2.8%), sentando una base inatacable para la memoria y la exposición oral del proyecto.
