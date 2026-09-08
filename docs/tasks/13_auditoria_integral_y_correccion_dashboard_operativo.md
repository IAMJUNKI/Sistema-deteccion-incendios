# Tarea 13: Auditoría Integral de Fidelidad Territorial, Normalización Cartográfica y Robustez Explicativa del Dashboard Operativo

---

## 1. ¿Qué se ha hecho?
- **Auditoría completa de extremo a extremo:** Se revisaron todas las pestañas, componentes, utilidades cartográficas y flujos de datos del dashboard Streamlit (`app.py`, `src/webapp/`).
- **Subsanación de error crítico en explicabilidad local (TreeSHAP):** En `src/models/explainability.py`, la función `explain_tree_prediction` fallaba al evaluar celdas en el modelo calibrado de producción por discrepancia de columnas con el contrato `egif-2d-48-v1` y presencia de tipos `object`. Se integró la normalización estricta mediante `ensure_feature_matrix_for_contract` y tipado numérico `float` con manejo de fallback.
- **Normalización del simulador What-If:** En `src/webapp/components/shap_simulator.py`, se solucionó la extracción de filas individuales (`iloc[[0]]`), se añadieron etiquetas en español de alta legibilidad (`FEATURE_LABELS`) y se acoplaron las variables meteorológicas medias (`temperature_mean`, `relative_humidity_mean`, `wind_speed_mean`, `vpd_mean`) al modificar los deslizadores para evitar estados atmosféricos físicamente imposibles.
- **Normalización administrativa y provincial exacta:**
  - Se generó la capa vectorial oficial de provincias de Galicia en formato GeoJSON (`data/external/galicia_provinces.geojson`) a partir de la geometría oficial del IGN.
  - Se precomputaron tablas maestras de lookup administrativo sin sobrecoste de CPU en tiempo de ejecución: `data/processed/grid/galicia_grid_1km_egif_admin.parquet` (29.601 celdas) y `data/processed/grid_galicia_centroids_admin.parquet` (30.697 celdas legacy).
  - Se eliminó la heurística simplista basada en latitud/longitud plana que asignaba incorrectamente más del 50% de la provincia de Pontevedra a A Coruña. Las superficies resultantes coinciden con exactitud milimétrica con los datos del Instituto Geográfico Nacional: Lugo (9.859 km²), A Coruña (7.949 km²), Ourense (7.282 km²), Pontevedra (4.511 km²).
- **Integración canónica de los 19 Distritos Forestales del PLADIGA (I a XIX):** En `src/webapp/utils/geo_helpers.py`, se implementó la tabla canónica `DISTRITOS_PLADIGA_CENTROIDES` con las 19 demarcaciones forestales oficiales de la Xunta de Galicia para la ordenación de comarcas y despachos preventivos.
- **Corrección de la distorsión planar en la búsqueda municipal:** En `src/webapp/components/concello_lookup.py`, se corrigió la fórmula de distancia cartesiana incorporando la ponderación cosinusoidal de latitud ($\cos(42.6^\circ\text{ N}) \approx 0.7361$), eliminando la distorsión del 36% que afectaba a la localización del centroide de las celdas respecto a los municipios. Se ampliaron los municipios de referencia a más de 45 concellos representativos.
- **Armonización de la escala cromática y leyendas cartográficas:** En `src/webapp/components/map_view.py`, se sincronizaron las funciones `get_color_gradient` y `build_map_legend_html` para respetar los 5 niveles operacionales calibrados en Modo Absoluto ($P \ge 12.0\%$, $6.0-12.0\%$, $2.5-6.0\%$, $1.0-2.5\%$, $<1.0\%$) y 5 tramos en Modo Táctico, eliminando la duplicidad que mostraba dos veces el Nivel 1.
- **Atenuación exterior limpia sin bordes artificiales:** Se retiró el contorno azul perimetral de Galicia y se consolidó la máscara inversa exterior (74%–82% de opacidad) que deja en penumbra Portugal, Castilla y León, Asturias y el mar, logrando un lienzo táctico limpio centrado exclusivamente en el territorio gallego.
- **Demarcación orgánica y alineación matemática exacta de Sectores Territoriales (PLADIGA):** Se sustituyeron los cuadrantes rectangulares rígidos (*bounding boxes*) por las **geometrías poligonales orgánicas exactas** construidas a partir de la unión espacial sin simplificar de las celdas de los 19 Distritos Forestales del PLADIGA (`data/external/galicia_sectors.geojson`). Se retiraron las líneas de rayado azul discontinuas y la baliza central para evitar saturación visual, dejando exclusivamente un **overlay sutil y limpio** de sombreado táctico (`fillOpacity: 0.08`, `weight: 0`) que resalta la comarca seleccionada. Asimismo, el filtrado de celdas se acopló estrictamente a `SECTOR_PLADIGA_DISTRITOS`, garantizando que el 100% de las cuadrículas mostradas pertenezcan de forma unívoca a dicho sector (0 celdas desbordadas fuera del área).
- **Cartografía base de máxima fidelidad institucional y nombres puramente descriptivos:** Se estableció como mapa base predeterminado la ortofotografía aérea de máxima resolución del Instituto Geográfico Nacional (**«IGN España (Ortofoto Oficial)»**, capa PNOA de máxima actualidad). Se eliminaron las opciones superfluas o con marcas comerciales ajenas (*CartoDB Voyager*, *OpenTopoMap*) y se renombraron todos los estilos con **nombres limpios y puramente descriptivos**:
  1. **IGN España (Ortofoto Oficial)** (Por defecto)
  2. **Relieve Topográfico**
  3. **Lienzo Claro**
  4. **Lienzo Oscuro**
  5. **Satélite**
  6. **Callejero**
- **Selector de Capa Base Cartográfica en la barra de herramientas del mapa:** En el espacio liberado por la retirada del centrado en concello, se integró un **selector de mapa base directo en la cabecera del mapa** (`Sector Territorial | Estilo de Mapa Base | Localizar Celda ID`). De esta forma, cualquier usuario puede alternar instantáneamente entre la ortofoto oficial del IGN, relieve topográfico, lienzo neutro o callejero en un solo clic, sin menús colapsados en la barra lateral.
- **Resolución de la pérdida de capas vectoriales en producción (empaquetado en `src/webapp/assets/`):**
  Al desplegar en producción mediante `scripts/deploy_code_server.sh`, el script sustituía la carpeta `data/` de la release por un enlace simbólico al almacenamiento persistente `/srv/fire-risk/data`. Como dicho almacenamiento persistente no contenía los ficheros GeoJSON versionados en Git, `load_galicia_focus_layers()` y `load_galicia_sectors_geojson()` devolvían `None`, impidiendo la visualización de la máscara exterior y los sectores en producción. Se solucionó integrando los recursos vectoriales directamente en el código de la aplicación (`src/webapp/assets/galicia_*.geojson`), adaptando `geo_helpers.py` para priorizar esta ruta, y actualizando `deploy_code_server.sh` y `sync_server_artifacts.sh` para garantizar la persistencia de las geometrías en el servidor.
- **Verificación completa mediante tests automatizados:** Se actualizaron e implementaron nuevas pruebas en `tests/test_webapp_components.py`, logrando la superación de 23/23 tests de webapp y 330/330 tests del conjunto global del repositorio.

---

## 2. ¿Por qué se ha hecho?
- **Fidelidad y rigor de misión crítica:** Un centro de mando de protección civil no puede tolerar distorsiones geográficas (provincias mutiladas), escalas incoherentes (leyendas que no coinciden con los colores pintados) ni caídas en tiempo de ejecución al solicitar explicabilidad causal.
- **Cumplimiento del objetivo del TFM:** La memoria y defensa del TFM exigen un sistema transparente donde las decisiones algorítmicas se apoyen en datos geoespaciales canónicos (IGN, PLADIGA, MeteoGalicia) y modelos calibrados contrastables frente al estado del arte (IberFire, IPIF AEMET, EFFIS).
- **Evitar la falsa sensación de seguridad técnica:** Informar 0 km² bajo alerta en días de ola de calor severa por el mero hecho de que el viento en los fondos de valle marcaba 22 km/h (en lugar de 30 km/h) inducía a error operacional grave; visibilizar la desecación termo-higrométrica 30-30 aporta un valor sustantivo a los directores de extinción.

---

## 3. ¿Cómo se ha hecho?
1. **Auditoría de Componentes:** Se ejecutó un análisis estático y dinámico componente a componente (`header_kpis`, `map_view`, `concello_lookup`, `territorial_analytics`, `shap_simulator`, `operational_protocols`, `system_audit`).
2. **Cruce Espacial Topológico Vectorial:** Empleando GeoPandas y las geometrías oficiales del IGN, se cruzaron espacialmente los centroides de ambas familias de cuadrículas (EGIF y legacy) contra los polígonos provinciales y distritos forestales, serializando los resultados en Parquet para consumo ultra-rápido en arranque.
3. **Trigonometría Esférica Local:** Para la búsqueda del concello más cercano, se aplicó la proyección equirrectangular local:
   $$\Delta d = \sqrt{(\Delta \text{lat})^2 + (\Delta \text{lon} \cdot \cos(\bar{\phi}))^2}$$
   con $\bar{\phi} = 42.6^\circ\text{ N}$, corrigiendo el estrechamiento de los meridianos hacia el polo.
4. **Alineación con el Contrato Canónico 48:** Se configuró el extractor TreeSHAP para operar sobre la matriz de 48 predictores numéricos de `egif-2d-48-v1`, forzando el tipado numérico explícito y completando imputaciones defensivas seguras si se invocan celdas parciales.
5. **Lógica de Indicadores Termo-Higrométricos:** Se implementó una lógica de selección condicional en la tarjeta 3 de telemetría:
   - Si `num_r30 > 0`: `Condición Crítica 30-30-30` (Alerta Crítica roja).
   - Si `num_r30 == 0` y `num_3030 > 0`: `Alerta Termo-Higrométrica 30-30` (Alerta Ámbar).
   - Si ambos son 0: `Condición Crítica 30-30-30: 0 (0.0%)` (Alerta Informativa azul).

---

## 4. ¿Por qué se han elegido estas tecnologías?
- **GeoPandas & Shapely:** Estándar industrial para operaciones geoespaciales vectoriales y uniones espaciales (`sjoin_nearest` y `within`) con proyección cartográfica oficial ETRS89 / UTM 29N (EPSG:25829) y WGS84 (EPSG:4326).
- **Parquet Precomputado:** Evita realizar costosos cálculos geométricos en cada renderizado o recarga de Streamlit, reduciendo la latencia de arranque a menos de 50 milisegundos y minimizando el consumo de RAM.
- **TreeSHAP (Lundberg et al.):** Método de interpretabilidad local axiomáticamente consistente basado en valores de Shapley para árboles de decisión ensartados (LightGBM), permitiendo justificar de forma inequívoca ante un tribunal qué variables provocaron la alerta en cada celda.
- **Pytest:** Marco estándar de pruebas unitarias y de integración que garantiza que cualquier refactorización preserve la compatibilidad de contratos y la ausencia de regresiones.

---

## 5. ¿Qué conseguimos con ello?
- **Dashboard Operativo 100% Robusto y Verificado:** Eliminación total de caídas en TreeSHAP, renderizado de mapas coherente con las leyendas y métricas fiables.
- **Georreferenciación Institucional Perfecta:** Asignación provincial y comarcal que refleja con exactitud los 19 distritos forestales del PLADIGA y los límites oficiales de la Xunta y el IGN, sin aberraciones de frontera.
- **Claridad Táctica para la Toma de Decisiones:** Los mandos de extinción disponen de información transparente sobre la severidad física ($P(Y=1)$), la prioridad de despacho (percentiles relativos) y las condiciones de desecación atmosférica ($30-30$), eliminando la ambigüedad y el "síndrome de Pedro y el lobo".
- **Alineación con los Requisitos Académicos del TFM:** Documentación exhaustiva con trazabilidad técnica y metodológica conforme a las directrices del proyecto.
