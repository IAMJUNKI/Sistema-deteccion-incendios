# Datacubo de riesgo de incendio en Galicia

TFM para construir un datacubo diario de 1 km × 1 km sobre Galicia. El target
es la ignición oficial de EGIF-MITECO; las fuentes explicativas son Copernicus
DEM GLO-30, CORINE Land Cover 2018, ERA5-Land y el baseline físico FWI de
CEMS/EFFIS.

## Contrato del dataset

- Área: Galicia, rejilla regular de 1 km, `EPSG:3035`.
- Contexto meteorológico: desde 30 días antes del inicio hasta la última fecha
  registrada en el XML EGIF.
- Periodo de modelado: 2016-01-01 hasta la última fecha registrada en EGIF.
- Target: `target_ignicion` de EGIF. Un cero significa ausencia de ignición.
- Temporalidad: las variables meteorológicas y sus acumulados describen el
  mismo día `T`; este producto es un **nowcast/análisis diario**, no una
  previsión a 24 horas.

Consulta [las variables](docs/variables.md) y la guía para
[ejecutar el pipeline](docs/ejecutar_pipeline.md).

Para desplegar el código por releases y transferir datasets/modelos pesados sin
subirlos a GitHub, consulta
[despliegue de código y datos](docs/deployment/despliegue_codigo_y_datos.md).

## Estructura

```text
src/geospatial/   rejilla, DEM y CORINE
src/ingestion/    ERA5-Land, FWI (CEMS) y EGIF
src/features/     calendario y exportación tabular
src/modeling/     carga, selección y evaluación temporal de modelos
src/workflow.py   orquestación completa
data/raw/         fuentes originales locales
data/processed/   productos generados (ignorados por Git)
archive/          código FIRMS/Mikel histórico, fuera de la ruta operativa
# 🔥 Sistema Predictivo de Anticipación de Incendios Forestales

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Conda](https://img.shields.io/badge/Conda-24.x-44A833?style=flat-square&logo=anaconda&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-En%20Desarrollo-orange?style=flat-square)
![TFM](https://img.shields.io/badge/TFM-Máster%20en%20Big%20Data%20e%20IA-blue?style=flat-square)
![MVP](https://img.shields.io/badge/MVP%20Region-Galicia-green?style=flat-square)

> **TFM** — Sistema de alerta temprana que estima el riesgo de incendio forestal por zona geográfica a partir de datos meteorológicos, satelitales, geoespaciales y ambientales. El objetivo es anticipar qué zonas presentan mayor probabilidad de incendio en las próximas **24 / 72 horas**.

---

## 📋 Tabla de Contenidos

- [Objetivo del Proyecto](#objetivo-del-proyecto)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [MVP: Galicia como Región Piloto](#mvp-galicia-como-región-piloto)
- [Fuentes de Datos](#fuentes-de-datos)
- [Stack Tecnológico](#stack-tecnológico)
- [Instalación y Configuración](#instalación-y-configuración)
- [Fases del Proyecto](#fases-del-proyecto)
- [Metodología](#metodología)
- [Documentación operativa detallada](#documentación-operativa-detallada)
- [Despliegue en servidor](#despliegue-en-servidor)
- [Convenciones y Guía de Contribución](#convenciones-y-guía-de-contribución)
- [Roadmap](#roadmap)
- [Equipo](#equipo)

---

## 🎯 Objetivo del Proyecto

Construir un sistema **actualizable diariamente** que, para cada celda de 1 km × 1 km del territorio de Galicia, genere:

- Una **probabilidad estimada de incendio** para las próximas 24/72 horas.
- Un **nivel de riesgo calibrado**: Bajo · Moderado · Alto · Extremo.
- Un **mapa interactivo de riesgo** y un **ranking de zonas más vulnerables**.
- Una **explicación de las variables** que justifican cada predicción (interpretabilidad via SHAP).

La aplicación práctica es clara: apoyar la prevención, priorizar recursos, anticipar zonas críticas y facilitar la toma de decisiones en gestión forestal y protección civil.

---

## 🏗️ Arquitectura del Sistema

```mermaid
flowchart TD
    subgraph DATOS_ESTATICOS["📦 Datos Estáticos (Fase 1 — Una sola vez)"]
        DEM["🏔️ Copernicus DEM\nAltitud · Pendiente · Orientación"]
        CLC["🌿 CORINE Land Cover\nTipo de vegetación y suelo"]
        IGN["📐 CNIG/IGN\nLímites administrativos"]
    end

    subgraph DATOS_HISTORICOS["📅 Datos Históricos (Fases 2-3 — Entrenamiento)"]
        FIRMS["🛰️ NASA FIRMS\nFocos de calor históricos"]
        ERA5["🌡️ ERA5-Land (Copernicus)\nMeteorología horaria histórica"]
        OSM["🗺️ OpenStreetMap\nCarreteras y núcleos urbanos"]
    end

    subgraph PIPELINE_DATOS["⚙️ Pipeline de Datos"]
        REJILLA["🔲 Rejilla 1km×1km\n~30.000 celdas (Galicia)"]
        TARGET["🎯 Target Construction\nClustering espacio-temporal DBSCAN\nPositivos + Negativos difíciles"]
        FEATURES["🔧 Feature Engineering\nAcumulados meteo · Ventana 12-18h\nHistorial incendios · Anti-leakage"]
    end

    subgraph MODELO["🤖 Modelado (Fase 4)"]
        BASELINE["📊 Regresión Logística\nBaseline interpretable"]
        TREE["🌲 XGBoost / LightGBM\nModelo principal"]
        CALIBRACION["🎚️ Calibración Isotónica\nUmbrales de riesgo con datos reales"]
    end

    subgraph PRODUCCION["🚀 Producción (Fase 5)"]
        METEOGALICIA["☁️ MeteoGalicia / AEMET\nProveedor configurable"]
        INFERENCIA["⚡ Pipeline Inferencia Diaria\nWRF 1km → 04km → AEMET"]
        WEBAPP["🖥️ Dashboard Streamlit\nMapas · Filtros · SHAP"]
    end

    DATOS_ESTATICOS --> REJILLA
    DATOS_HISTORICOS --> TARGET
    REJILLA --> TARGET
    TARGET --> FEATURES
    FEATURES --> BASELINE
    FEATURES --> TREE
    TREE --> CALIBRACION
    CALIBRACION --> INFERENCIA
    METEOGALICIA --> INFERENCIA
    INFERENCIA --> WEBAPP
```

---

## 📁 Estructura del Repositorio

```
Sistema-deteccion-incendios/
│
├── 📂 src/                          # Código fuente principal (paquete modular)
│   ├── __init__.py
│   ├── 📂 geospatial/               # Fase 1: Rejilla, DEM, CORINE
│   │   ├── __init__.py
│   │   └── README.md
│   ├── 📂 ingestion/                # Fase 2: NASA FIRMS, ERA5-Land, target
│   │   ├── __init__.py
│   │   └── README.md
│   ├── 📂 features/                 # Fase 3: Feature engineering, anti-leakage
│   │   ├── __init__.py
│   │   └── README.md
│   ├── 📂 models/                   # Fase 4: Entrenamiento y calibración
│   │   ├── __init__.py
│   │   └── README.md
│   └── 📂 webapp/                   # Fase 5: Dashboard Streamlit
│       ├── __init__.py
│       └── README.md
│
├── 📂 data/                         # ⚠️ NO incluir en git (ver .gitignore)
│   ├── raw/                         # Datos descargados sin procesar
│   ├── processed/                   # Datos transformados y limpios
│   └── models/                      # Modelos serializados (.pkl, .json)
│
├── 📂 notebooks/                    # Exploración, EDA y prototipado
├── 📂 tests/                        # Tests unitarios e integración
├── 📂 docs/                         # Documentación adicional y memoria TFM
├── 📂 configs/                      # Configuración de hiperparámetros y pipelines
│
├── 📂 knowledge/                    # Documentación del alcance del proyecto
│   ├── Project Plan - Predicción incendios forestales.md
│   ├── fases_proyecto_plan_ataque.md
│   └── justificacion_mvp_comunidad_autonoma.md
│
├── 📂 .agents/skills/               # Skills de Gemini para el proyecto
│   ├── project-conventions/
│   ├── geospatial-processing/
│   ├── data-ingestion/
│   ├── feature-engineering/
│   ├── model-training/
│   └── webapp-dashboard/
│
├── .gitignore
├── .env.example                     # Template de variables de entorno
├── environment.yml                  # Entorno Conda reproducible
└── README.md
```

> **Nota sobre los datos:** Los archivos de datos (NetCDF, Parquet, Shapefiles, modelos .pkl) **no se suben al repositorio** por su tamaño. Ver la sección [Fuentes de Datos](#fuentes-de-datos) para saber cómo descargarlos.

---

## 🗺️ MVP: Galicia como Región Piloto

El MVP se centra en **Galicia** por tres razones:

| Criterio | Detalle |
|---|---|
| **Alta densidad de incendios** | Galicia concentra el mayor número de incendios forestales de España, garantizando suficientes casos positivos para el entrenamiento |
| **Manejabilidad computacional** | ~30.000 celdas de 1km² frente a ~500.000 celdas en toda España (reducción del 94%) |
| **Ecosistema específico** | Minifundio forestal, clima oceánico y alta proporción de eucalipto/pino: dinámicas de incendio diferenciadas y bien documentadas |

Una vez validado el modelo en Galicia (AUC-ROC > 0.85, PR-AUC satisfactorio), el código está diseñado para **escalar a nivel nacional** simplemente cambiando el archivo de frontera geométrica de entrada.

---

## 📊 Fuentes de Datos

### Datasets Prioritarios

| # | Fuente | Uso | Fase |
|---|---|---|---|
| 1 | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) | Variable objetivo (focos de incendio) | 2 |
| 2 | [ERA5-Land (Copernicus CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Meteorología horaria histórica | 2-3 |
| 3 | [CORINE Land Cover](https://land.copernicus.eu/en/products/corine-land-cover) | Tipo de vegetación y cobertura del suelo | 1 |
| 4 | [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) | Altitud, pendiente y orientación del terreno | 1 |
| 5 | [CNIG/IGN](https://centrodedescargas.cnig.es/CentroDescargas/index.jsp) | Límites administrativos para agregación espacial | 1 |
| 6 | [MeteoGalicia MeteoSIX v5](https://www.meteogalicia.gal/datosred/infoweb/meteo/proxectos/meteosix/API_MeteoSIX_v5_es.pdf) | Forecast horario WRF 1 km con fallback 04 km para la inferencia operativa | 5 |

### Datasets Complementarios

| Fuente | Uso | Recomendación |
|---|---|---|
| [EGIF-MITECO](https://www.miteco.gob.es/es/biodiversidad/temas/incendios-forestales/estadisticas-incendios.html) | Estadística oficial española de incendios | Validación y contexto, no target principal |
| [AEMET OpenData](https://opendata.aemet.es/dist/index.html) | Proveedor municipal alternativo para pruebas mientras no exista la clave MeteoGalicia | `FORECAST_PROVIDER=aemet` o `auto` |
| [OpenStreetMap/Geofabrik](https://download.geofabrik.de/europe/spain.html) | Proximidad a carreteras y núcleos urbanos | Factor humano, prioridad secundaria |

---

## 🛠️ Stack Tecnológico

| Categoría | Tecnologías |
|---|---|
| **Lenguaje** | Python 3.11 |
| **Entorno** | Conda (Miniconda) |
| **Geoespacial** | GeoPandas · Rasterio · Rioxarray · Shapely · GDAL · H3 |
| **Datos temporales** | Xarray · NetCDF4 · cdsapi |
| **ML** | Scikit-learn · XGBoost · LightGBM · imbalanced-learn |
| **Explicabilidad** | SHAP |
| **Visualización** | Matplotlib · Seaborn · Plotly · Folium · PyDeck |
| **WebApp** | Streamlit |
| **Calidad de código** | Ruff · pre-commit · pytest |
| **Notebooks** | JupyterLab |

---

## ⚙️ Instalación y Configuración

### Prerrequisitos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/) instalado.
- Git instalado.

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/Sistema-deteccion-incendios.git
cd Sistema-deteccion-incendios
```

### 2. Crear el entorno Conda

```bash
conda env create -f environment.yml
conda activate incendios-forestales
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` y completar las claves de API necesarias:

- **NASA FIRMS MAP KEY** → [Registrarse aquí](https://firms.modaps.eosdis.nasa.gov/api/area/)
- **Copernicus CDS API KEY** → [Registrarse aquí](https://cds.climate.copernicus.eu/user/register)
- **MeteoGalicia API KEY** → completar `METEOGALICIA_API_KEY` en `.env` *(fuente preferida)*
- **AEMET API KEY** → completar `AEMET_API_KEY` en `.env` *(alternativa para pruebas/contingencia)*

### 4. Instalar el paquete en modo desarrollo

```bash
pip install -e .
```

## Validación

```powershell
& C:\Users\alfon\anaconda3\envs\incendios-forestales\python.exe -m pytest tests -q
```

Para validar los productos generados sin descargar ni modificar datos, ejecutar:

- `notebooks/07_validacion_datacubo.ipynb`: contrato, mapas, consistencia
  meteorologica y cobertura del target en el NetCDF.
- `notebooks/08_validacion_dataset_parquet.ipynb`: esquema, particiones,
  target, meteorologia y preparacion de los Parquet para ML.
- `notebooks/09_exploracion_y_seleccion_features.ipynb`: exploración reproducible,
  señal univariante, redundancia y definición de conjuntos de predictores para modelado.
- `notebooks/10_experimentos_feature_selection.ipynb`: comparación temporal
  controlada de conjuntos de variables con LightGBM, usando 2022 como validación.
- `notebooks/11_validacion_robusta_modelo.ipynb`: réplica con varios subconjuntos
  de negativos y revisión mensual antes de desbloquear el test de 2023.

La fase posterior de selección y evaluación está descrita en
[modelado](docs/modeling.md). No forma parte del pipeline de construcción de datos.
```bash
python -c "import src, folium, streamlit_folium, geopandas; print('✅ Entorno configurado correctamente')"
```

Para iniciar el dashboard, usa el mismo entorno Conda con el que se instalaron
las dependencias:

```bash
conda activate incendios-forestales
python -m streamlit run app.py
```

Usar `streamlit run app.py` desde el entorno base puede producir
`ModuleNotFoundError: No module named 'folium'`. En ese caso, comprueba que
`which python` y `which streamlit` apuntan a
`.../envs/incendios-forestales/`, o ejecuta directamente:

```bash
/opt/anaconda3/envs/incendios-forestales/bin/python -m streamlit run app.py
```

---

## 🏃 Ejecución y Uso (Fase 1)

Una vez completada la instalación, puedes generar y visualizar los grids geoespaciales base del proyecto.

### 1. Preprocesar y Recortar CORINE Land Cover (CLC)
Si has descargado los archivos de CORINE España del CNIG (en formato `.gpkg`), guárdalos en `data/raw/corine/clc_espana_2018.gpkg` (o `clc_espana_2012.gpkg`) y ejecuta el recorte para generar los rasters de Galicia:

```bash
# Recortar y rasterizar versión 2012
python -m src.geospatial.preprocesar_corine --gpkg data/raw/corine/clc_espana_2012.gpkg --output data/raw/corine/clc_galicia_2012.tif

# Recortar y rasterizar versión 2018
python -m src.geospatial.preprocesar_corine --gpkg data/raw/corine/clc_espana_2018.gpkg --output data/raw/corine/clc_galicia.tif
```

### 2. Ejecutar el Pipeline Geoespacial
Genera los archivos Parquet definitivos de la rejilla de 1 km² de Galicia cruzados con Copernicus DEM y CORINE:

```bash
# Generar rejilla 2012 (entrenamiento 2013-2018)
AWS_NO_SIGN_REQUEST=YES AWS_PROFILE="" python -m src.geospatial.pipeline --corine data/raw/corine/clc_galicia_2012.tif --output data/processed/grid/galicia_grid_1km_2012.parquet

# Generar rejilla 2018 (entrenamiento 2019-2024)
AWS_NO_SIGN_REQUEST=YES AWS_PROFILE="" python -m src.geospatial.pipeline --corine data/raw/corine/clc_galicia.tif --output data/processed/grid/galicia_grid_1km_2018.parquet
```

### 3. Visualizar e Inspeccionar los Grids Generados
Puedes inspeccionar rápidamente estadísticas, tipos de datos y mapear los grids espaciales por pantalla:

```bash
# Ver estadísticas de la rejilla 2018 por consola
python -m src.geospatial.visualizar_grid --input data/processed/grid/galicia_grid_1km_2018.parquet

# Abrir el mapa visual interactivo de combustibles de 2012
python -m src.geospatial.visualizar_grid --input data/processed/grid/galicia_grid_1km_2012.parquet --plot
```

---

## 🗺️ Fases del Proyecto

El proyecto se estructura en **5 fases secuenciales**:

### Fase 1 — Infraestructura Geoespacial
> Construir el tablero de juego: rejilla de 1km×1km sobre Galicia, datos topográficos (DEM) y cobertura del suelo (CORINE).

**Entregable:** DataFrame estático con `cell_id`, coordenadas, altitud, pendiente, orientación y tipo de combustible.

### Fase 2 — Ingesta Histórica y Construcción del Target
> Descargar incendios reales (NASA FIRMS), aplicar clustering espacio-temporal (DBSCAN) para identificar eventos únicos, y generar negativos difíciles estratificados.

**Entregable:** Dataset de entrenamiento con positivos (Y=1, inicio de incendio) y negativos difíciles (Y=0).

### Fase 3 — Ingeniería de Características
> Enriquecer el dataset con variables meteorológicas acumuladas (ventana 12h-18h), historial de incendios, variables de proximidad humana, y auditoría estricta anti-data-leakage.

**Entregable:** Dataset Maestro en formato `.parquet` listo para ML.

### Fase 4 — Modelado y Calibración
> Entrenar modelos (Regresión Logística → XGBoost/LightGBM) con validación temporal por años completos. Calibrar umbrales de riesgo con la tasa real de incidencia.

**Entregable:** Modelo serializado + tabla de umbrales calibrados (Bajo / Moderado / Alto / Extremo).

### Fase 5 — WebApp e Integración Operativa
> Dashboard interactivo en Streamlit con mapas de riesgo (Folium/PyDeck), filtros por municipio, panel SHAP y pipeline de inferencia diaria alimentado por un proveedor meteorológico configurable.

**Entregable:** WebApp desplegada + memoria final del TFM.

La ruta operativa requiere además un estado meteorológico reciente de 30 días
por celda (`data/processed/state/weather_daily_state.parquet`). Si falta o no
tiene cobertura, la inferencia falla explícitamente para evitar que una memoria
de sequedad desconocida se convierta en ceros.

---

## 📐 Metodología

### División espaciotemporal

El dataset se construye como una **rejilla de celdas de 1km × 1km**, donde cada fila representa `(celda, día)`. Para cada observación se respeta la causalidad temporal: solo se usan datos disponibles hasta `T-1` para predecir el riesgo del día `T`.

### Estrategia de validación

Se prohíbe el K-Fold aleatorio. La validación es estrictamente temporal:

```
Entrenamiento:   2019 · 2020 · 2021
Validación:      2022
Test (ciego):    2023 · 2024
```

### Métricas principales

- **AUC-ROC** — Capacidad discriminativa general.
- **PR-AUC** — Fundamental para clases desbalanceadas (incendios << no-incendios).
- **F1-Score** — Balance precisión/recall en los umbrales calibrados.

### Limitaciones reconocidas

- Nubosidad puede impedir detección satelital (documentado, no corregible).
- El modelo de producción usa previsiones meteorológicas (WRF de MeteoGalicia o AEMET municipal durante las pruebas) en lugar de datos ERA5 perfectos: se cuantifica la degradación por proveedor y horizonte.
- No se predice la propagación del incendio, solo el **inicio**.

## 📚 Documentación operativa detallada

La explicación granular de contratos de datos, zonas horarias, fórmulas,
anti-leakage, fallback stale, manifiestos, checksums, modelos, pruebas,
operación diaria y roadmap está en
[`docs/explanations/pipeline_operativo_detallado.md`](docs/explanations/pipeline_operativo_detallado.md).

Para la previsión meteorológica y la diferencia entre forecast MeteoGalicia y
reanálisis ERA5-Land, consulta también
[`docs/explanations/prevision_meteorologica_operativa.md`](docs/explanations/prevision_meteorologica_operativa.md).

## 🖥️ Despliegue en servidor

La guía de instalación en Linux, almacenamiento persistente, secretos,
systemd timers, Streamlit, Nginx/TLS, backups, rollback, recuperación y
checklist de aceptación está en
[`docs/deployment/servidor_produccion.md`](docs/deployment/servidor_produccion.md).

Para ejecutar AEMET en un ordenador personal sin servidor ni `cron`, consulta
[`docs/deployment/ejecucion_local_aemet.md`](docs/deployment/ejecucion_local_aemet.md).

Para probar el pipeline con un estado AEMET simulado cuando el colector horario
no estuvo activo, consulta
[`docs/deployment/simulacion_local_estado_aemet.md`](docs/deployment/simulacion_local_estado_aemet.md).

---

## 🌿 Convenciones y Guía de Contribución

### Estrategia de ramas (Git Flow simplificado)

```
main          ← código estable, releases del TFM
  └── develop ← rama de integración activa
        └── feature/fase-X-descripcion-corta   ← trabajo individual
```

**Flujo de trabajo:**

```bash
# 1. Crear rama desde develop
git checkout develop
git pull origin develop
git checkout -b feature/fase2-descarga-firms

# 2. Trabajar y hacer commits descriptivos
git add .
git commit -m "feat(ingestion): descarga histórico NASA FIRMS Galicia 2019-2024"

# 3. Abrir Pull Request hacia develop (no directamente a main)
```

### Convenciones de commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

| Prefijo | Uso |
|---|---|
| `feat(módulo):` | Nueva funcionalidad |
| `fix(módulo):` | Corrección de bug |
| `data(módulo):` | Cambios en scripts de datos |
| `docs:` | Documentación |
| `refactor(módulo):` | Refactoring sin cambio de funcionalidad |
| `test(módulo):` | Tests |
| `chore:` | Mantenimiento (deps, config) |

### Módulos disponibles

`geospatial` · `ingestion` · `features` · `models` · `webapp`

### Gestión de datos

- Los datos **nunca** se suben al repositorio. Ver `.gitignore`.
- Documentar la fuente, versión y proceso de descarga de cada dataset en `docs/`.
- Usar `data/raw/` para datos originales sin procesar y `data/processed/` para datos transformados.

---

## 🗓️ Roadmap

- [x] Configuración inicial del repositorio
- [x] Documentación del alcance (knowledge/)
- [x] **Fase 1** — Rejilla geoespacial + DEM + CORINE
- [ ] **Fase 2** — Ingesta NASA FIRMS + ERA5 + construcción del target
- [ ] **Fase 3** — Feature engineering + dataset maestro
- [ ] **Fase 4** — Entrenamiento XGBoost/LightGBM + calibración
- [ ] **Fase 5** — Dashboard Streamlit + pipeline inferencia diaria
- [ ] Memoria final del TFM
- [ ] Escalado a nivel nacional (post-TFM)

---

## 👥 Equipo

Trabajo Fin de Máster — Máster en Big Data e Inteligencia Artificial.

Diego Junquera
Miquel Jimenez
Alfonso García
Raúl Utrilla
Enrique Bravo
Santiago Mateos
---

*Para dudas sobre el proyecto, abre un issue en el repositorio o consulta la documentación en `knowledge/`.*
