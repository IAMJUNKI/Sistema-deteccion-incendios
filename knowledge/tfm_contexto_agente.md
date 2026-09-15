# Contexto del TFM — Documento de Referencia para Agentes IA

> **Propósito:** Este documento existe para que cualquier asistente IA entienda el proyecto en su primer mensaje, sin necesidad de re-explicar el contexto. Leerlo completo antes de ayudar con cualquier tarea.

---

## Qué es este proyecto

**TFM de Máster en Big Data e Inteligencia Artificial.**

Sistema de alerta temprana que predice el **riesgo de inicio de incendio forestal** con 24/48/72 horas de antelación, célula a célula (rejilla de 1km×1km), para la Comunidad Autónoma de **Galicia** como MVP.

No predecimos incendios que ya están ardiendo. Predecimos el **inicio**, porque es lo único con valor operativo real.

---

## Equipo y restricciones del TFM

- **Tamaño del equipo:** 6 personas trabajando en paralelo, todos tocan todas las fases.
- **Entregable académico:** Memoria de ~20 páginas (orientación negocio) + Anexos técnicos sin límite + Repositorio GitHub.
- **Opción TFM:** Propuesta personalizada (Opción 3 de la guía). Ya aprobada por los tutores (Carlos Ortega, Santiago Mota).
- **Idioma:** Español.
- **Estado:** Infraestructura del repositorio creada. Trabajo técnico aún no empezado.

---

## Decisiones arquitectónicas tomadas (no reabrir)

| Decisión | Elección | Razón |
|---|---|---|
| **Región MVP** | Galicia | Mayor densidad de incendios en España, ~30K celdas manejables, MeteoGalicia disponible |
| **Resolución** | 1km × 1km × 1 día | Estándar del campo, coherente con IberFire y sistemas operativos |
| **Fuente meteorológica entrenamiento** | ERA5-Land (Copernicus CDS) | Reanálisis completo, homogéneo, 2019-2024 |
| **Fuente meteorológica producción** | MeteoGalicia (observaciones + previsión) | 170 estaciones en Galicia vs ~35 de AEMET. API abierta, sin clave. |
| **Target (variable objetivo)** | NASA FIRMS VIIRS — inicio de incendio (T₀ del cluster) | Estándar internacional de teledetección |
| **Negativos** | Difíciles estratificados (vecinos espaciales + temporales + aleatorios) | Evita que el modelo aprenda trivialidades estacionales |
| **Validación** | Temporal por años completos (prohibido K-Fold aleatorio) | Simula producción real |
| **Modelo principal** | XGBoost / LightGBM | Alto rendimiento en datos tabulares desbalanceados |
| **Entorno** | Conda, Python 3.11 | Compatibilidad con stack geoespacial (GDAL, rasterio) |
| **Branching** | Git Flow simplificado (main / develop / feature/) | Equipo de 6 personas |
| **Pipeline datos** | Construido desde cero (no usar IberFire como dataset) | Demuestra capacidad de ingeniería, más control metodológico |

---

## El papel de IberFire (importante)

**IberFire** (Ercibengoa et al., arXiv:2505.00837, 2025) es el trabajo más parecido existente:
- Datacube 1km×1día para toda España, 2007-2024, ~260 features
- Código abierto en GitHub, dataset en Zenodo
- **Mismas fuentes que nosotros:** ERA5, FIRMS, CORINE, DEM

**Cómo lo usamos nosotros:**
- ❌ **NO** como dataset de entrenamiento (no lo descargamos y entrenamos encima)
- ✅ **SÍ** como validación: comparar nuestros eventos de ignición detectados con los suyos
- ✅ **SÍ** como benchmark: comparar AUC-ROC de nuestro modelo vs. el suyo
- ✅ **SÍ** como referencia metodológica para decisiones de diseño
- ✅ **SÍ** como cita obligatoria en la memoria del TFM

**Nuestra diferencia respecto a IberFire:**
1. Pipeline operativo end-to-end (ellos solo publican el dataset)
2. MeteoGalicia para producción (ellos solo usan ERA5)
3. Calibración de umbrales con tasas reales de incidencia
4. Dashboard Streamlit funcional
5. Análisis cuantificado de degradación ERA5 → previsión meteorológica

---

## Las 5 Fases del Proyecto

### Fase 1 — Infraestructura Geoespacial
Construir la rejilla 1km×1km sobre Galicia. Calcular variables estáticas por celda.

**Fuentes:** Copernicus DEM GLO-30, CORINE Land Cover, CNIG/IGN límites administrativos.
**Entregable:** `data/processed/grid/galicia_grid_1km_v1.parquet`
**Columnas clave:** `cell_id`, `lat_centroid`, `lon_centroid`, `altitud_media`, `pendiente_media`, `orientacion_media`, `combustible_clase`
**Se ejecuta una sola vez.**

### Fase 2 — Ingesta Histórica y Construcción del Target
Descargar incendios reales (NASA FIRMS VIIRS, 2019-2024, Galicia). Aplicar clustering DBSCAN espaciotemporal (radio 5km, ventana 4 días) para identificar eventos únicos. Generar negativos difíciles. Descargar ERA5-Land horario.

**Entregable:** Dataset con `cell_id`, `fecha`, `target` (0/1) + meteo base.

### Fase 3 — Ingeniería de Características
Calcular acumulados de precipitación (1/3/7/14/30 días), días sin lluvia, ventana crítica meteorológica **12h-18h** (temperatura máxima, humedad mínima, viento máximo), historial de incendios en vecindario, proximidad a carreteras/núcleos.

**Regla crítica anti-leakage:** Para predecir día T, solo usar datos hasta T-1. Nunca datos de T o posteriores.
**Entregable:** `data/processed/features/dataset_maestro.parquet`

### Fase 4 — Modelado y Calibración
Entrenamiento con validación temporal estricta (2019-2021 train / 2022 val / 2023-2024 test). Comparar Regresión Logística → XGBoost → LightGBM. Métricas: AUC-ROC, PR-AUC, F1. Calibración de umbrales de riesgo (Bajo / Moderado / Alto / Extremo) basada en tasas reales de incidencia.

**Entregable:** Modelo serializado + tabla de umbrales calibrados.

### Fase 5 — WebApp e Integración Operativa
Dashboard Streamlit con mapas de riesgo (Folium/PyDeck), filtros por provincia/municipio, panel SHAP de explicabilidad. Pipeline de inferencia diaria con MeteoGalicia (cron job 05:00 AM). Análisis de degradación ERA5 (entrenamiento) → MeteoGalicia previsión (producción).

**Entregable:** WebApp funcional + análisis científico de degradación.

---

## Principio fundamental: el gap entre entrenamiento y producción

El modelo aprende con **ERA5** (reanálisis histórico, casi perfecto, disponible con ~5 días de retraso).
En producción se alimenta con **previsiones de MeteoGalicia** (con error de previsión).

Esta brecha es inevitable en cualquier sistema operativo de predicción meteorológica. La contribución científica del TFM incluye **cuantificar esa degradación** por horizonte temporal (24h vs. 48h vs. 72h).

```
AUC-ROC con ERA5 real  → techo teórico del sistema
AUC-ROC con MeteoGalicia previsión → rendimiento real en producción
Diferencia → contribución científica medida y documentada
```

---

## Fuentes de Datos

| Fuente | URL | Uso | Fase | Autenticación |
|---|---|---|---|---|
| NASA FIRMS VIIRS | https://firms.modaps.eosdis.nasa.gov | Target (Y=1) | 2 | API Key gratuita |
| ERA5-Land (Copernicus CDS) | https://cds.climate.copernicus.eu | Meteorología entrenamiento | 2-3 | UID + API Key |
| MeteoGalicia EMA | https://servizos.meteogalicia.gal | Meteo producción | 5 | **Sin clave** |
| MeteoGalicia Previsión | https://servizos.meteogalicia.gal | Forecast 72h | 5 | **Sin clave** |
| CORINE Land Cover | https://land.copernicus.eu | Tipo vegetación | 1 | Sin clave |
| Copernicus DEM GLO-30 | https://dataspace.copernicus.eu | Topografía | 1 | Cuenta gratuita |
| CNIG/IGN | https://centrodedescargas.cnig.es | Límites administrativos | 1 | Sin clave |
| OpenStreetMap/Geofabrik | https://download.geofabrik.de | Carreteras, núcleos | 3 | Sin clave |
| IberFire (benchmark) | Zenodo (arXiv:2505.00837) | Validación metodológica | 4 | Sin clave |

---

## Stack Tecnológico

```
Python 3.11 | Conda
Geoespacial:  GeoPandas, Rasterio, Rioxarray, Shapely, GDAL, H3
Datos:        Xarray, NetCDF4, cdsapi, Pandas, PyArrow (Parquet)
ML:           Scikit-learn, XGBoost, LightGBM, imbalanced-learn
Explicabilidad: SHAP
Visualización:  Matplotlib, Folium, PyDeck, Plotly
WebApp:       Streamlit, streamlit-folium
Calidad:      Ruff, pytest, pre-commit
```

---

## Estructura del Repositorio

```
Sistema-deteccion-incendios/
├── src/
│   ├── geospatial/   ← Fase 1: rejilla, DEM, CORINE
│   ├── ingestion/    ← Fase 2: FIRMS, ERA5, target, negativos
│   ├── features/     ← Fase 3: feature engineering, anti-leakage
│   ├── models/       ← Fase 4: entrenamiento, calibración
│   └── webapp/       ← Fase 5: Streamlit + inferencia diaria
├── data/
│   ├── raw/          ← datos descargados sin procesar (no en git)
│   ├── processed/    ← datos transformados (no en git)
│   └── models/       ← modelos serializados (no en git)
├── notebooks/        ← exploración y EDA
├── tests/
├── docs/             ← documentación técnica y decisiones
├── knowledge/        ← este documento y otros de alcance
└── .agents/skills/   ← guías internas de trabajo
```

---

## Guías de trabajo disponibles en este proyecto

Las siguientes guías documentan las convenciones y procedimientos del proyecto:

- **`project-conventions`** — Git Flow, commits, CRS, nomenclatura, anti-leakage
- **`geospatial-processing`** — rejilla, DEM, CORINE, GeoPandas, rasterio
- **`data-ingestion`** — NASA FIRMS, ERA5-Land, MeteoGalicia, AEMET, clustering DBSCAN
- **`feature-engineering`** — ventana 12-18h, acumulados, negativos difíciles
- **`model-training`** — validación temporal, XGBoost, calibración, SHAP
- **`webapp-dashboard`** — Streamlit, Folium, inferencia diaria, análisis degradación

---

## Lo que NO hacemos (decisiones cerradas)

- ❌ Predecir propagación de incendios (solo inicio)
- ❌ Computer Vision / CNNs sobre imágenes satelitales (fuera de alcance del MVP)
- ❌ K-Fold Cross-Validation aleatorio (rompe la causalidad temporal)
- ❌ Escala nacional desde el inicio (primero Galicia, luego se escala)
- ❌ Usar IberFire como dataset de entrenamiento principal
- ❌ Datos sintéticos
- ❌ Predecir a más de 72h (fiabilidad meteorológica insuficiente)
