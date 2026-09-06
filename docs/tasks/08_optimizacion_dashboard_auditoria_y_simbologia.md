# 08. Optimización del Dashboard, Auditoría del Sistema y Rediseño de Simbología

---

## 1. ¿Qué se ha hecho?

- **Auditoría Técnica Integral del Pipeline y Visualización:**
  - Inspección exhaustiva de los artefactos del modelo en producción (`active_model_manifest.json`, modelos `forecast_risk_egif_48_t1/t2/t3.joblib`).
  - Verificación del dataset operativo (`data/processed/predicciones_operativas.parquet`, 88.803 filas, 29.601 celdas activas $\times$ 3 horizontes $T+1, T+2, T+3$).
  - Comprobación de la ingesta meteorológica WRF 1 km nominal de MeteoGalicia (manifiesto con calidad `fresh`).

- **Corrección de Discrepancia Crítica de Features (Bugs de Nombres de Columnas):**
  - Se identificó que `concello_lookup.py` y `shap_simulator.py` leían variables del prototipo legacy (`tmax_vc`, `rhmin_vc`, `vmax_vc`, `prec_acum_30d`, `pendiente_media`), las cuales no existían en el parquet del contrato EGIF 48 (`temperature_max_12_18h`, `relative_humidity_min_12_18h`, `wind_speed_max_12_18h`, `precipitation_sum_30d`, `slope_mean`). Esto provocaba que todas las consultas municipales y diagnósticos mostraran valores de respaldo fijados por defecto (28 °C, 40% HR, 18 km/h).
  - Se implementó en `src/webapp/utils/data_loader.py` (`enrich_dataset_metadata`) la normalización bidireccional automática entre variables canónicas EGIF-48 y variables legacy.
  - Se implementó la síntesis de clase de combustible dominante (`combustible_clase`) y porcentaje de arbolado forestal (`combustible_pct_forestal`) a partir de las fracciones CORINE del modelo (`broadleaf_forest`, `coniferous_forest`, `mixed_forest`, `scrub`, `agriculture`).
  - Se actualizaron `concello_lookup.py` y `shap_simulator.py` para priorizar el acceso a las variables canónicas.
  - Se corrigió el simulador *What-If* en `shap_simulator.py` para sincronizar las 48 variables canónicas (`temperature_max_12_18h`, `relative_humidity_min_12_18h`, `wind_speed_max_12_18h`, `vpd_max_12_18h`, etc.) antes de evaluar el artefacto LightGBM calibrado.

- **Eliminación del «Criterio Térmico» y Rediseño de Simbología Cartográfica:**
  - Se reemplazó la etiqueta engañosa «Criterio Térmico» por «Simbología de Riesgo» en `src/webapp/components/sidebar.py`.
  - Se estructuró en dos modos conceptualmente rigurosos:
    1. **Riesgo Absoluto Calibrado $P(Y=1)$:** Refleja la probabilidad física calibrada del evento de ignición.
    2. **Priorización Relativa por Percentil (%):** Aísla las celdas más calientes del territorio independientemente de la severidad global del día (útil para optimización de patrullaje).
  - Se adaptó `get_color_gradient` en `src/webapp/components/map_view.py` para soportar de forma nativa las nuevas denominaciones manteniendo retrocompatibilidad.
  - **Sincronización Dinámica de la Leyenda del Mapa:** Se reemplazó el bloque HTML estático y hardcodeado por la función reactiva `build_map_legend_html(color_mode)`, haciendo que el título, los umbrales numéricos y los pies explicativos de la leyenda flotante cambien de inmediato según el modo seleccionado por el usuario.
  - **Corrección del Filtro de Despacho:** Se subsanó la colisión en la que la opción *"Mostrar Todas las Celdas con Riesgo > 0.5%"* disparaba por error la rama restrictiva de celdas al Top 0.5%.

- **Reducción de Ruido Cognitivo y Jerarquización UX/UI:**
  - Rediseño del sidebar aplicando el principio de **Divulgación Progresiva (Progressive Disclosure)**: controles esenciales visibles en primer plano (Horizonte $T+1/2/3$, Simbología y Filtro de despacho); selectores secundarios (capa base, datasets históricos `.parquet`, recarga de caché) agrupados en un expander colapsado `st.expander("⚙️ Opciones Avanzadas / Capas")`.
  - Inclusión de contexto de prevalencia basal en la tarjeta KPI principal: explicación concisa de que la probabilidad basal diaria en Galicia es de $\sim 0.02\%$, aclarando de forma inmediata por qué un valor de $P > 5\%$ constituye un riesgo extremo en protección civil.

- **Batería de Pruebas Unitarias y Calidad:**
  - Ampliación de `tests/test_webapp_components.py` con tests específicos para la normalización EGIF-48, cálculo de la regla 30-30-30, síntesis de vegetación, evaluación de gradientes de color, generación de leyendas dinámicas y cálculo de severidad diaria (`test_get_day_severity_info`).
  - 100% de tests unitarios pasando (13/13 tests). Verificación de formato y linteo estricto con Ruff (0 errores).

- **Corrección del Bug Crítico de "Nivel 5 — Extremo Permanente" (Falsas Alarmas Tautológicas):**
  - Se detectó una inconsistencia grave en `header_kpis.py` (L80) y `operational_protocols.py` (L45): la severidad evaluaba `if num_top05 > 100: alert_label = "Nivel 5 — Extremo"`. Dado que la malla de Galicia consta de 29.601 celdas de 1 km², el Top 0.5% contiene siempre $29.601 \times 0.005 = 148$ celdas. Como $148 > 100$ es una tautología matemática, el sistema activaba "Nivel 5 — Extremo" **todos los días del año**, incluso con probabilidades máximas moderadas del 1.88% o insignificantes en invierno (0.01%).
  - Se creó la función unificada y canónica `get_day_severity_info(max_prob, num_critical_cells)` que clasifica la severidad diaria en función del riesgo físico real $P(Y=1)$ y la escala institucional del proyecto (Nivel 1 Bajo < 1.0%, Nivel 2 Moderado 1.0–2.5%, Nivel 3 Alto 2.5–6.0%, Nivel 4 Muy Alto 6.0–12.0%, Nivel 5 Extremo $\ge 12.0\%$).
  - Ahora, una jornada con probabilidad máxima del 1.88% se clasifica fielmente como **Nivel 2 — Moderado** con tarjeta ámbar (`alert-warning`), eliminando las falsas alarmas catastrofistas (efecto "Pedro y el lobo").

- **Dinamización de la Tarjeta 2 de KPIs y Eliminación del Cómputo Fijo de 1.481 km²:**
  - La tarjeta 2 mostraba de forma fija e inmutable "1.481 km² en Alerta Urgente" en rojo crítico (`alert-critical`), cifra que correspondía simplemente al Top 5% matemático ($29.601 \times 0.05 = 1.480,05 \approx 1.481$).
  - Se dinamizó la Tarjeta 2 para adaptarse a la severidad física real: en jornadas de Nivel 1 o 2, muestra la superficie en riesgo activo ($P \ge 1.0\%$) o el contingente de despacho preventivo (Top 0.5%: 148 km²) en estilo ámbar/verde, reservando el estilo crítico y el cómputo de alta severidad ($P \ge 2.5\%$) exclusivamente para días con riesgo físico comprobado.
  - El Resumen Ejecutivo matinal y el informe PLADIGA ahora adaptan tanto sus cifras territoriales como sus recomendaciones tácticas (vigilancia aérea, retenes y permisos de quema) a la severidad meteorológica del día.

- **Resolución del Mapa Monocromático y Sub-graduación de Percentiles:**
  - Se solventó el problema por el cual filtrar al Top 0.5% en modo "Priorización Relativa" volvía todo el mapa morado uniforme (`#800026`).
  - Se refinó la escala de percentiles introduciendo una sub-graduación superior: Top 0.2% Crítico (&ge;99.8%, `#800026`), Top 0.5% Muy Alto (&ge;99.5%, `#BD0026`), Top 2.0% Prioritario (&ge;98.0%, `#E31A1C`), etc., asegurando contraste visual incluso dentro de selecciones estrechas de máxima prioridad.
  - Se incorporó una alerta táctica contextual en el visor que orienta al analista: si activa un filtro percentil estrecho, se le informa de que la variación de probabilidad física continua (ej. de 1.2% a 1.88%) se analiza activando *"Riesgo Absoluto Calibrado P(Y=1)"*.
  - Se renombraron las opciones del selector de despacho en `sidebar.py` (ej. "Top 0.5% (Máxima Prioridad de Despacho)" en lugar de "Top 0.5% Riesgo Extremo") para erradicar la confusión conceptual entre prioridad de despacho relativo y severidad de ignición física.

- **Auditoría y Armonización Semántica Integral de las 5 Pestañas de Apoyo:**
  - *Tab 1 (Consulta por Concello — `concello_lookup.py`):* Se incorporó el valor exacto de probabilidad calibrada $P(Y=1)$ en la insignia municipal (`Nivel 2 — Moderado (1.88%)`), resolviendo la falta de cuantificación numérica para técnicos municipales.
  - *Tab 2 (Situación Territorial y Rankings — `territorial_analytics.py`):* Se renombraron las columnas de los rankings distritales a `Cuadrículas Top 5%` y `Cuadrículas Top 0.5%`, erradicando la etiquetación engañosa de "Extremo" o "Alerta Urgente" que confundía el ordenamiento estadístico relativo con peligro físico letal.
  - *Tab 3 (Diagnóstico y Simulador — `shap_simulator.py`):* Se añadieron los porcentajes cuantitativos de probabilidad estimada (`{prob*100:.2f}%` basal vs `{sim_prob_val*100:.2f}%` simulado) en las tarjetas de comparación tras ajustar sliders de temperatura, humedad o viento, proporcionando feedback biofísico numérico inmediato.
  - *Tab 4 (Medidas y Despacho PLADIGA — `operational_protocols.py`):* Se reemplazó la matriz obsoleta de 4 niveles (Nivel 0–3) por la matriz institucional canónica de 5 niveles del proyecto (Nivel 1 Bajo a Nivel 5 Extremo $\ge 12.0\%$), garantizando total armonización semántica con los KPIs de cabecera y el resumen ejecutivo.
  - *Tab 5 (Auditoría del Sistema — `app.py`):* Se eliminó el banner redundante de proxies legacy del lienzo de mando principal y se confinaron las notas técnicas de trazabilidad exclusivamente a la pestaña de auditoría.
  - *Corrección de Solapamiento en el Encabezado (`styles.py`):* Se solucionó el recorte del título *"SISTEMA DE ALERTA TEMPRANA DE INCENDIOS FORESTALES"* ajustando el padding superior del contenedor principal a `padding-top: 3.25rem !important;` y redefiniendo la jerarquía tipográfica para evitar solapamientos con la barra de navegación nativa de Streamlit.

---

## 2. ¿Por qué se ha hecho?

1. **Rigor Científico y Defensa del TFM:** El término «Criterio Térmico» en el ámbito de incendios forestales sugiere umbrales de temperatura o detección por infrarrojos satelitales (como MODIS/VIIRS FIRMS). Llamar así a la rampa de color generaba confusión metodológica grave ante el tribunal y desvirtuaba el significado de las probabilidades del modelo supervisado.
2. **Integridad de Datos en la Aplicación:** Las celdas municipales mostraban datos estáticos de respaldo debido al desacoplamiento entre el contrato canónico EGIF-48 y los componentes web. Corregir esta normalización era imprescindible para que la herramienta reflejase la meteorología real de MeteoGalicia.
3. **Credibilidad Operativa y Eliminación del Síndrome de "Pedro y el Lobo":** Un sistema de apoyo a la decisión que declara "Nivel 5 — Extremo" un día de primavera con probabilidad máxima del 1.88% pierde de inmediato toda credibilidad ante los mandos de extinción y protección civil. El sistema debe ser matemáticamente intachable.
4. **Ergonomía y Usabilidad en Centros de Mando (EOC):** La interfaz presentaba una sobrecarga de controles técnicos simultáneos. En situaciones de emergencia o durante una presentación académica, los mandos requieren interpretar la severidad en menos de 10 segundos sin distraerse con parámetros internos de archivos `.parquet`.

---

## 3. ¿Cómo se ha hecho?

1. **Normalización en el Pipeline de Visualización:**
   - En `data_loader.py`, `enrich_dataset_metadata(df)` verifica la presencia de nombres canónicos (`temperature_max_12_18h`, etc.) y genera los aliases de respaldo (`tmax_vc`, etc.) y viceversa.
   - Si las fracciones CORINE están presentes, calcula el porcentaje forestal arbóreo sumando frondosas, coníferas y mixto, e infiere la clase dominante mediante una función vectorial.
2. **Severidad Centralizada y Calibrada:**
   - En `header_kpis.py`, `get_day_severity_info(max_prob, num_critical_cells)` evalúa el riesgo físico de forma determinista y coherente con `get_risk_level_info` de la ficha municipal.
   - En `operational_protocols.py`, se conecta la misma severidad y se condiciona el texto de los protocolos PLADIGA (medidas aéreas, brigadas terrestres y suspensión de quemas).
3. **Reorganización del Panel de Control y Cartografía:**
   - En `sidebar.py`, se extrajeron los controles secundarios hacia un contenedor expandible, y se renombraron los filtros de despacho separando la prioridad táctica del peligro físico.
   - En `map_view.py`, se enriqueció `get_color_gradient` con sub-escalones en percentiles altos y se dinamizó la leyenda flotante HTML.
4. **Simulación What-If Coherente:**
   - En `shap_simulator.py`, los desplazamientos de los sliders de temperatura, humedad y viento se propagan a todas las variables afectadas del vector de 48 features, recalculando la Deficiencia de Presión de Vapor (VPD) mediante la formulación biofísica de Tetens.
5. **Armonización de Pestañas y Componentes Secundarios:**
   - En `concello_lookup.py`, se integró la probabilidad $P(Y=1)$ exacta en la insignia visual del concello (`risk_info['level'] (prob_val*100:.2f%)`).
   - En `territorial_analytics.py`, se renombraron las columnas de ranking distrital a `Cuadrículas Top 5%` y `Cuadrículas Top 0.5%`, desvinculando la jerarquía estadística de las etiquetas cualitativas de peligro.
   - En `operational_protocols.py`, se reemplazó la matriz de 4 niveles por la escala canónica institucional de 5 niveles del PLADIGA (Nivel 1 Bajo a Nivel 5 Extremo $\ge 12.0\%$).
   - En `styles.py`, se incrementó el margen superior a `padding-top: 3.25rem !important;` y se adaptó la cabecera para eliminar recortes producidos por la interfaz de Streamlit.

---

## 4. ¿Por qué se han elegido estas tecnologías?

- **Streamlit (`st.sidebar.expander`, `st.sidebar.radio`):** Permite implementar el patrón de diseño de divulgación progresiva sin añadir dependencias externas ni penalizar los tiempos de carga en el navegador.
- **Normalización Vectorizada con Pandas / NumPy:** Garantiza que el enriquecimiento de metadatos sobre las 29.601 celdas se ejecute en milisegundos sin bloquear la reactividad del dashboard.
- **Escalas Calibradas Discretas:** Las 5 categorías de riesgo (Nivel 1 a 5) siguen las pautas de Protección Civil y de la escala europea EFFIS, adaptadas a la distribución de probabilidad condicionada por el desbalance de clases del modelo.
- **Formulación Biofísica de Tetens (VPD):** Asegura que las variaciones simultáneas de temperatura y humedad en el simulador What-If mantengan la coherencia termodinámica del aire.
- **Ruff & Pytest:** Garantizan el estándar de calidad de código y la ausencia de regresiones en los componentes.

---

## 5. ¿Qué conseguimos con ello?

- **Claridad Conceptual Total:** Distinción nítida entre la severidad biofísica real (Probabilidad calibrada) y la priorización táctica de recursos escasos (Percentil relativo).
- **Eliminación Total de Falsas Alarmas:** Fin de la alerta de Nivel 5 fija; los mandos reciben un informe verídico donde 1.88% es Nivel 2 — Moderado y los 1.481 km² fijos son reemplazados por la superficie real en riesgo.
- **Simplificación del Filtro de Celdas y Reducción de Ruido Visual:**
  - Se simplificó la nomenclatura del selector territorial en `sidebar.py` a percentiles territoriales puros y ordenados (`Top 0.5%`, `Top 1.0%`, `Top 2.0%`, `Top 5.0%`, `Top 10.0%`, `Top 20.0%`), eliminando la opción ambigua de "Todas las celdas (>0.5%)" que colisionaba con el Top 0.5%.
  - Se eliminó el banner de advertencia técnica sobre proxies legacy de la pantalla principal en `app.py`, manteniendo el lienzo operacional limpio y trasladando las advertencias técnicas exclusivamente a la pestaña de "Auditoría del Sistema".
- **Consistencia de Datos 100% Verificada:** Eliminación de los fallbacks artificiales; los 313 municipios de Galicia y las 29.601 celdas muestran sus verdaderas lecturas de temperatura, humedad, viento y vegetación.
- **Experiencia de Usuario Profesional para el TFM:** Un centro de mando limpio, moderno, reactivo y libre de ruido visual, perfectamente defendible ante el tribunal académico y listo para su uso operativo en Protección Civil y PLADIGA.
