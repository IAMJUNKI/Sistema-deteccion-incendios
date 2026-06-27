---
name: project-conventions
description: >
  Convenciones del proyecto TFM de predicción de incendios forestales: estructura de ramas Git, 
  estilo de código Python, nombrado de archivos y variables, gestión de datos, formato de commits 
  y guía para PRs. Activar cuando se trabaje en cualquier parte del proyecto.
---

# Convenciones del Proyecto — TFM Incendios Forestales

## Contexto del Proyecto

Sistema predictivo de anticipación de incendios forestales con rejilla espacial de 1km×1km sobre Galicia (MVP). Código modular en Python 3.11, entorno Conda, pipeline de datos → ML → Streamlit webapp.

---

## 1. Estrategia de Ramas (Git Flow Simplificado)

```
main          ← código estable, releases del TFM
  └── develop ← rama de integración activa
        └── feature/fase-X-descripcion-corta
```

### Reglas de ramas

- **Nunca** hacer commits directamente a `main`.
- Todo el trabajo va en ramas `feature/` creadas desde `develop`.
- Los PRs se abren **siempre hacia `develop`**, no hacia `main`.
- `main` solo recibe merges desde `develop` cuando hay un hito estable (fin de fase, entregable TFM).

### Nombrado de ramas

```
feature/fase1-rejilla-geopandas
feature/fase2-descarga-firms
feature/fase3-negativos-dificiles
feature/fase4-entrenamiento-xgboost
feature/fase5-dashboard-streamlit
fix/fase2-clustering-dbscan
docs/actualizar-readme
```

---

## 2. Convenciones de Commits (Conventional Commits)

Formato: `tipo(módulo): descripción corta en minúsculas`

| Tipo | Cuándo usarlo |
|---|---|
| `feat(módulo):` | Nueva funcionalidad o script |
| `fix(módulo):` | Corrección de bug |
| `data(módulo):` | Scripts de descarga o procesamiento de datos |
| `refactor(módulo):` | Refactoring sin cambio funcional |
| `docs:` | Cambios en documentación o README |
| `test(módulo):` | Añadir o corregir tests |
| `chore:` | Mantenimiento: deps, config, .gitignore |

### Módulos válidos

`geospatial` · `ingestion` · `features` · `models` · `webapp`

### Ejemplos correctos

```
feat(ingestion): descarga histórico NASA FIRMS Galicia 2019-2024
data(geospatial): generación rejilla 1km x 1km con GeoPandas
fix(features): corregir data leakage en ventana meteorológica 12-18h
refactor(models): separar calibración isotónica en función independiente
docs: actualizar sección de instalación en README
test(features): añadir tests para cálculo de días secos acumulados
```

---

## 3. Estilo de Código Python

### Formateo y linting

- **Formateador**: `ruff format` (compatible con Black).
- **Linter**: `ruff check` (sustituye a flake8 + isort + pyupgrade).
- Configuración en `pyproject.toml` o `ruff.toml` en la raíz.

### Configuración recomendada de Ruff

```toml
[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "N", "W"]
ignore = ["E501"]  # line length gestionado por formatter
```

### Type hints

Usar type hints en todas las funciones públicas:

```python
import geopandas as gpd
from pathlib import Path

def cargar_rejilla(ruta: Path) -> gpd.GeoDataFrame:
    """Carga la rejilla geoespacial del MVP desde un archivo Parquet o GeoJSON."""
    ...
```

### Docstrings

Formato Google style:

```python
def calcular_pendiente(dem_array: np.ndarray, resolucion_m: float = 30.0) -> np.ndarray:
    """Calcula la pendiente media en grados a partir de un DEM.

    Args:
        dem_array: Array numpy 2D con valores de elevación en metros.
        resolucion_m: Resolución espacial del DEM en metros. Por defecto 30m (GLO-30).

    Returns:
        Array numpy 2D con la pendiente en grados para cada pixel.
    """
    ...
```

---

## 4. Nomenclatura de Variables y Archivos

### Variables Python

- `snake_case` para variables y funciones: `cell_id`, `pendiente_media`, `dias_sin_lluvia`.
- `UPPER_SNAKE_CASE` para constantes: `GRID_RESOLUTION_KM = 1`, `CRITICAL_WINDOW_START = 12`.
- Prefijos descriptivos para DataFrames: `df_incendios`, `gdf_galicia`, `df_features`.

### Nombres de archivos

```
# Scripts
descarga_firms.py
construccion_rejilla.py
feature_engineering.py

# Notebooks (prefijo numérico para ordenación)
01_exploracion_firms.ipynb
02_analisis_era5.ipynb
03_feature_importance.ipynb

# Datos procesados (incluir fecha o versión)
galicia_grid_1km_v1.parquet
dataset_maestro_2019_2024.parquet
modelo_xgboost_v1.pkl
umbrales_calibrados_v1.json
```

### Identificadores de celda

El `cell_id` es la referencia central de todo el sistema. Formato recomendado: `{lat_min}_{lon_min}` en grados decimales con 4 decimales, o un índice entero secuencial. Debe ser **consistente** entre todos los módulos.

---

## 5. Gestión de Datos

### Regla de oro

**Los datos nunca se suben al repositorio.** Ver `.gitignore`. Los archivos de datos se descargan localmente siguiendo las instrucciones de cada módulo.

### Estructura de directorios de datos

```
data/
├── raw/           # Datos originales descargados, sin procesar
│   ├── firms/     # Archivos CSV/GeoJSON de NASA FIRMS
│   ├── era5/      # Archivos NetCDF de Copernicus ERA5-Land
│   ├── dem/       # Rasters GeoTIFF del Copernicus DEM
│   ├── corine/    # CORINE Land Cover (vectorial o raster)
│   └── igm/       # Límites administrativos CNIG/IGN
├── processed/     # Datos transformados y limpios
│   ├── grid/      # Rejilla estática Galicia
│   ├── target/    # Dataset con variable objetivo
│   └── features/  # Dataset Maestro final
└── models/        # Modelos serializados y umbrales
```

### Variables de entorno para rutas

Usar siempre `python-dotenv` y las variables definidas en `.env`:

```python
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
DATA_RAW = Path(os.getenv("DATA_RAW_PATH", "data/raw"))
```

---

## 6. Anti-Data-Leakage (Regla Crítica del Proyecto)

Para predecir el riesgo del día `T`:
- Solo usar información disponible hasta `T-1`.
- La ventana meteorológica crítica (12h-18h) se calcula sobre el día `T-1`.
- Los negativos nunca pueden estar a ±5 días de un incendio real en la misma zona.
- La validación es **siempre temporal**: entrenamiento (años pasados) → validación (año siguiente) → test ciego (años más recientes).

---

## 7. Sistema de Coordenadas de Referencia (CRS)

- **CRS de trabajo**: ETRS89 / UTM zone 29N → `EPSG:25829` (estándar para Galicia/España).
- **CRS de almacenamiento/visualización**: WGS84 → `EPSG:4326`.
- Siempre reprojectar al CRS de trabajo antes de hacer operaciones espaciales (distancias, áreas).
- Documentar el CRS en los metadatos de cada archivo geoespacial.

---

## 8. Tests

- Tests en `tests/` con `pytest`.
- Cada módulo de `src/` debe tener su test correspondiente en `tests/test_{módulo}.py`.
- Los tests no deben requerir acceso a internet ni a archivos de datos reales: usar fixtures con datos mínimos sintéticos.
- Ejecutar con: `pytest tests/ -v --cov=src`
