# 06. Inferencia Operativa y Dashboard Streamlit (Centro de Mando Táctico)

---

## 1. ¿Qué se ha hecho?
- **Rediseño integral de la interfaz de usuario (UI/UX) e iconografía institucional:** Transformación del dashboard en un **Centro de Mando de Alerta Temprana (Emergency Operations Center - EOC)** sobrio e institucional, eliminando emojis informales e integrando la librería oficial **Google Material Symbols and Icons (Outlined)** vía CDN de Google Fonts.
- **Estructura modular de navegación operativa en 6 pestañas:**
  1. `🗺️ Centro de Mando Cartográfico`: Visualizador geoespacial a 1 km² con capas ráster institucionales libres de marcas de agua (Esri Gray Canvas, Esri Satellite, IGN España PNOA Ortofoto), presets de zoom comarcales y polígonos disueltos continuos.
  2. `🏛️ Consulta por Concello`: Ficha semafórica municipal directa para alcaldes y mandos de Protección Civil con diagnóstico de situación en lenguaje natural, factores climáticos locales y recomendaciones preventivas.
  3. `📊 Situación Territorial y Rankings`: Cuadro de mando agregado por provincias y distritos forestales, ranking de municipios en alerta y exportador de datos a CSV.
  4. `🔬 Diagnóstico y Simulador`: Explicabilidad en lenguaje claro de factores agravantes/atenuantes y simulador interactivo *What-If* para evaluar variaciones climáticas en caliente.
  5. `🚨 Medidas y Despacho (PLADIGA)`: Matriz de intervención preventiva basada en el plan autonómico de incendios y generador en 1-clic de Informes Ejecutivos de Situación en Markdown.
  6. `⚙️ Auditoría del Sistema`: Linaje de datos, calidad del pronóstico WRF/AEMET, calibración isotónica y comparativa con el estado del arte (IberFire, AEMET IPIF, EFFIS).
- **Máscara de enfoque territorial (*Spatial Focus Mask*):** Generación automática de una máscara vectorial inversa (*donut mask*) sobre el exterior de Galicia (Portugal, Castilla y León, Asturias y océano) para atenuar suavemente el entorno circundante y focalizar la atención operativa en el territorio gallego.
- **Resumen Ejecutivo Matinal (Lectura en 30 segundos):** Banner superior matinal auto-generado que sintetiza la severidad del día, horas de ventana crítica y distritos de atención preferente.
- **Carga de datos resiliente con recuperación de coordenadas:** Blindaje en `data_loader.py` y `map_view.py` ante discrepancias de esquemas parquets, resolviendo cualquier ausencia de `lat_centroid`/`lon_centroid` mediante cálculo al vuelo de centroides vectoriales o fusión automática con la rejilla maestra.
- **Suite de pruebas unitarias:** Creación y actualización de `tests/test_webapp_components.py` cubriendo la traducción de niveles de riesgo (1 a 5), asignación territorial, robustez de esquemas de datos y generación de informes ejecutivos.

## 2. ¿Por qué se ha hecho?
- Para transformar una interfaz puramente técnica de *Data Science* en una herramienta de soporte a la decisión accesible y directamente utilizable por perfiles de mando no técnicos (alcaldes, directores de extinción, técnicos de Protección Civil).
- Para eliminar la fricción interpretativa asociada a probabilidades numéricas bajas ($P=0.04$), traduciéndolas a una **escala de amenaza intuitiva del 1 al 5 (Bajo, Moderado, Alto, Muy Alto, Extremo)** alineada con los estándares europeos de EFFIS/Copernicus.
- Para garantizar una experiencia de usuario limpia y profesional, libre de marcas de agua o dependencias de claves API en la cartografía base.

## 3. ¿Cómo se ha hecho?
- **Modularización desacoplada:** Separación de componentes en `src/webapp/components/` (`header_kpis.py`, `map_view.py`, `concello_lookup.py`, `territorial_analytics.py`, `shap_simulator.py`, `operational_protocols.py`, `system_status.py`, `sidebar.py`) y utilidades en `src/webapp/utils/` (`geo_helpers.py`, `data_loader.py`), coordinados por el orquestador `app.py`.
- **Máscara espacial con Shapely & GeoPandas:** `load_galicia_focus_layers()` calcula la diferencia geométrica entre un bounding box ibérico amplio y el polígono unificado de Galicia (`outer_box.difference(galicia_geom)`), proyectando una capa de atenuación translúcida `#0b0f19` en Folium.
- **Normalización defensiva de datos:** `enrich_dataset_metadata()` enriquece dinámicamente cualquier dataset cargado asegurando columnas espaciales, reglas 30-30-30 y clasificaciones de acción preventiva.
- **Simulador What-If reactivo:** Ejecuta inferencia local sobre el modelo LightGBM serializado recalculando el vector de 23 variables ante cambios de temperatura, humedad o viento en tiempo real.

## 4. ¿Por qué se han elegido estas tecnologías?
- **Streamlit + Custom CSS (Inter & Material Symbols):** Combina la reactividad en Python con la estética de aplicaciones web institucionales de alto impacto.
- **Folium con Teselas Libres Institucionales (Esri Gray Canvas, IGN PNOA, OpenStreetMap):** Elimina cualquier dependencia de claves API o marcas de agua de proveedores comerciales y ofrece un contraste térmico ideal.
- **Shapely & GeoPandas:** Procesamiento espacial vectorial estándar para disolución geométrica de celdas y cálculo de máscaras territoriales.
- **TreeSHAP:** Algoritmo exacto de interpretabilidad para modelos de gradiente boosting, traducido a narrativas en lenguaje natural para su uso en salas de crisis.

## 5. ¿Qué conseguimos con ello?
- Una plataforma de nivel operativo real que combina rigor biofísico y geoespacial a 1 km² con una usabilidad intuitiva para cualquier usuario institucional.
- Aceleración en la toma de decisiones matinales mediante el resumen ejecutivo de 30 segundos y la búsqueda directa por concello.
- Cumplimiento estricto de las directrices académicas y de gobernanza del TFM, demostrando tanto excelencia técnica en Machine Learning como capacidad de transferencia tecnológica y diseño centrado en el usuario (UX).
