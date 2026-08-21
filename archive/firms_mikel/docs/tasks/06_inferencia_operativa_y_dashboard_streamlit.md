# 06. Pipeline de Inferencia Operativa Diaria y Dashboard Interactivo Streamlit

---

## 1. ¿Qué se ha hecho?
- **Pipeline de Inferencia Operativa Diaria en `scripts/run_daily_inference.py`:**
  - Ingesta matutina de previsiones numéricas **MeteoGalicia (WRF)** a 1-4 km.
  - Corrección de sesgo mediante **Quantile Mapping** respecto a la climatología ERA5-Land.
  - Inferencia estocástica supervisada con el modelo oficial **LightGBM Standard** sobre las 30.697 celdas de Galicia.
- **Refinamiento de Reglas Físicas de Extinción:**
  - Anulación física determinista a $0.00\%$ de probabilidad cuando $P_{\text{dia}} \ge 5.0\text{ mm}$ o cuando $RH_{\min} \ge 70.0\% \land P_{\text{dia}} \ge 3.0\text{ mm}$ (humedad de combustible fino $MC_{ff} > 30\%$).
  - **Revisión y Supresión del Filtro Artificial de Arbolado:** Se eliminó el filtro plano `combustible_pct_forestal < 5%` al constatar empíricamente que en Galicia más del $60\%$ de las igniciones ocurren en **matorral / monte bajo de secano (CORINE 321/322)** con $0\%$ de masa arbolada pero con biomasa vegetal hiper-inflamable (ej. Celda #6818 y Celda #29599).
- **Aplicación Web Interactiva en Streamlit (`app.py`):**
  - **Visor Geoespacial Folium de Alta Definición:** Integración de basemaps CartoDB Positron, DarkMatter, OpenStreetMap y Esri Satellite.
  - **Masa Poligonal Continua Fusionada (Dissolved GIS Geometries):** Uso de `shapely.ops.unary_union` y `GeoPandas.dissolve` para agrupar rectángulos contiguos del mismo color en una única forma poligonal compleja sin rejillas negras divisorias (`interactive=False`).
  - **Capa Doble Interactiva Celda a Celda (Hover & Click):** Superposición de rectángulos de 1 km² con popups interactivos que muestran Celda ID, Probabilidad $P(Y=1)$, Percentil Relativo, variables meteorológicas $T-1$ y diferenciación nítida del combustible (**Matorral / Monte Bajo 🔥 vs. Bosque Arbolado 🌲**).
  - **Selector de Horizontes Temporales:** Diccionario exacto `horizon_date_map` para evaluación de Hoy ($T$), Mañana ($T+1$), Pasado Mañana ($T+2$), 3 Días Vista ($T+3$) y Casos Emblemáticos.
  - **Modos de Coloreado Espacial:** Gradiente Térmico Continuo por Probabilidad $P(Y=1)$ (YlOrRd / Plasma), Gradiente Continuo por Percentil Relativo (%) y Selección Táctica por Niveles Fijos.
  - **Panel de Explicabilidad SHAP:** Descomposición local interactiva de los 5 factores principales de riesgo por celda seleccionada.
- **Verificación Retrospectiva (Backtesting) en `scripts/run_backtest_verification.py`.**

## 2. ¿Por qué se ha hecho?
- **Paso a Producción Operativa (Fase 5):** Un modelo de Machine Learning solo aporta valor de negocio si puede ser ejecutado diariamente a las 07:00 AM para alimentar la toma de decisiones del centro de mando de incendios.
- **Eliminación del Domain Shift:** MeteoGalicia (WRF) y ERA5-Land son modelos numéricos distintos; el *Quantile Mapping* garantiza que el modelo LightGBM reciba datos en producción en la misma escala que aprendió en entrenamiento.
- **Visualización de Envolturas Espaciales en Matorral:** Los incendios en matorral (como la Celda #7255 o #29366) muestran probabilidades moderadas en el punto exacto pero están rodeados por una masa espacial continua de celdas vecinas en riesgo elevado. El gradiente continuo permite visualizar este envoltorio micro-climático.

## 3. ¿Cómo se ha hecho?
1. Se leen o descargan los predictores meteorológicos WRF para la fecha objetivo ($T, T+1, T+2, T+3$).
2. `apply_quantile_mapping()` ajusta las variables $T_{\max}$, $RH_{\min}$, $V_{\max}$ y $P_{\text{dia}}$ a la escala climatológica ERA5-Land.
3. **Inferencia Híbrida (Física Determinista + ML Supervisado):**
   - Si se cumplen las condiciones de extinción hídrica ($P_{\text{dia}} \ge 5.0\text{ mm}$ o $RH_{\min} \ge 70\% \land P_{\text{dia}} \ge 3\text{ mm}$), la probabilidad se fija en $0.0000$.
   - Para las celdas secas, el modelo **LightGBM Standard** calcula la probabilidad posterior $P(Y=1)$ y clasifica el percentil relativo en Galicia.
4. `build_complex_dissolved_shapes()` ejecuta la unión espacial de polígonos contiguos en GeoPandas y Streamlit renderiza la mapa Folium interactiva sin recargas mediante `st_folium(m, returned_objects=[])`.

## 4. ¿Por qué se han elegido estas tecnologías?
- **Paradigma Híbrido frente a Entrenar con 18M de Ceros Invernales:**
  - Evita la contaminación del dataset con 18.37 millones de ceros invernales triviales (que habrían diluido el gradiente a ratio 1:15.000) y protege en producción contra artefactos de extrapolación fuera de rango.
- **Streamlit + Folium + GeoPandas Unary Union:** Renderizado ligero, rápido y estéticamente superior para visualización geoespacial sin necesidad de servidores GIS pesados.
- **Quantile Mapping Empírico:** Estándar en meteorología operativa para corrección no paramétrica de sesgos numéricos.

## 5. ¿Qué conseguimos con ello?
- **Éxito Probado en Verificación Retrospectiva (Backtesting Agosto 2023):**
  - **Percentil Medio Asignado a Fuegos Reales: `84.5%`**.
  - **Gran Incendio 23/08/2023 (Celda 6818):** Asignado en el **Percentil `99.5%`** (Top 0.5% Riesgo Extremo, Probabilidad `21.62%`).
- **Producto Mínimo Viable (MVP) Operativo Completo:** El proyecto pasa de ser una colección de experimentos a ser una **plataforma web funcional y desplegable**.
- **Cierre del Ciclo del TFM (Fases 5 y 6):** Disponibilidad del dashboard funcional para demostraciones en tiempo real ante el tribunal.

---

## 6. Guía de Ejecución Manual de Scripts y Aplicación Web

### ⚙️ 6.1 Activación del Entorno Virtual
```bash
conda activate incendios-forestales
```

### 🚀 6.2 Ejecución del Pipeline de Inferencia Operativa Diaria
Para ejecutar manualmente la descarga/ingesta de datos de previsión y calcular las probabilidades de riesgo para cualquier fecha objetivo:

- **Inferencia para el Día Actual (Por Defecto - Hoy):**
  ```bash
  PYTHONPATH=. python3 scripts/run_daily_inference.py
  ```
- **Inferencia para una Fecha Específica ($T+1$, $T+2$, $T+3$ o Histórica):**
  ```bash
  PYTHONPATH=. python3 scripts/run_daily_inference.py --date 2026-07-26
  ```

*El script entrenará el modelo LightGBM en histórico, aplicará Quantile Mapping, evaluará las reglas de extinción hídrica y exportará el archivo de resultados a `data/processed/inferencia_YYYY-MM-DD.parquet`.*

### 🌐 6.3 Despliegue e Inicio del Dashboard Interactivo Streamlit
Para iniciar el panel web interactivo de control:
```bash
streamlit run app.py
```
*Se abrirá automáticamente el navegador en `http://localhost:8501`, permitiendo explorar el mapa en gradiente térmico continuo, cambiar basemaps y analizar los desgloses de explicabilidad SHAP por celda.*
