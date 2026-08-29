"""Contrato único de espacio, tiempo y fuentes del datacubo histórico."""

from pathlib import Path

CRS = "EPSG:3035"
GRID_CELL_SIZE_M = 1000

# El periodo de predicción es el que se entrega a ML.  ERA5 empieza antes solo
# para poder calcular las ventanas de precipitación sin valores incompletos.
METEOROLOGY_CONTEXT_START = "2018-12-01"
DATACUBE_START = "2019-01-01"
DATACUBE_END = "2023-11-26"  # Fallback para CLI; workflow lo deriva del XML EGIF.
EGIF_START = "2018-01-01"

# Rutas ancladas al repositorio: funcionan igual desde CLI, PyCharm o notebooks.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
STATIC_DIR = PROCESSED_DIR / "static"
GRID_DIR = PROCESSED_DIR / "grid"
METEOROLOGY_DIR = PROCESSED_DIR / "meteorology"
TARGET_DIR = PROCESSED_DIR / "target"
DATACUBE_DIR = PROCESSED_DIR / "datacube"
HUMAN_ACTIVITY_RAW_DIR = RAW_DIR / "human_activity"

BOUNDARY_PATH = RAW_DIR / "igm" / "galicia_boundary.geojson"
CORINE_PATH = RAW_DIR / "corine" / "U2018_CLC2018_V2020_20u1.tif"
ERA5_RAW_DIR = RAW_DIR / "meteorology" / "era5"

SPATIAL_CUBE_PATH = STATIC_DIR / "spatial_grid_1km.nc"
TOPOGRAPHY_CUBE_PATH = STATIC_DIR / "topography_1km.nc"
LANDCOVER_CUBE_PATH = STATIC_DIR / "landcover_1km.nc"
HUMAN_ACTIVITY_CUBE_PATH = STATIC_DIR / "human_activity_1km.nc"
GRID_PATH = GRID_DIR / "galicia_grid_1km.gpkg"
TIME_CUBE_PATH = PROCESSED_DIR / "calendar_2019_2023.nc"
ERA5_DAILY_PATH = METEOROLOGY_DIR / "era5_daily.nc"
METEOROLOGY_CUBE_PATH = METEOROLOGY_DIR / "era5_grid_1km.nc"
EGIF_EVENTS_PATH = TARGET_DIR / "egif_events.gpkg"
EGIF_TARGET_PATH = TARGET_DIR / "egif_target.parquet"
EGIF_METADATA_PATH = TARGET_DIR / "egif_metadata.json"
DATACUBE_PATH = DATACUBE_DIR / "galicia_1km.nc"
TABULAR_DATASET_DIR = PROCESSED_DIR / "tabular" / "egif"
