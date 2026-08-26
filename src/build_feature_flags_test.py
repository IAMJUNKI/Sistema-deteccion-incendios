"""Construye un cubo real de siete días para validar el perfil de variables.

No sobrescribe las capas ni el datacubo histórico canónico. Reutiliza las
fuentes locales ya descargadas y deja todo en ``data/processed/feature_flags_test``.
"""

from pathlib import Path

import xarray as xr

from src.config import (
    ERA5_RAW_DIR, ERA5_DAILY_PATH, FEATURE_FLAGS_TEST_CUBE_PATH, GRID_PATH,
    HUMAN_ACTIVITY_RAW_DIR, LANDCOVER_CUBE_PATH, SPATIAL_CUBE_PATH, TOPOGRAPHY_CUBE_PATH,
    EGIF_TARGET_PATH,
)
from src.feature_flags import TEST_PROFILE
from src.features.time import crear_datacubo_temporal, guardar_datacubo_temporal
from src.geospatial.human_activity import construir_capa_actividad_humana
from src.ingestion.meteorology import (
    cargar_era5_horario, crear_meteorologia_diaria, interpolar_al_grid, listar_archivos_era5,
)
from src.pipeline import construir_datacubo_completo

START = "2019-01-01"
END = "2019-01-07"


def _filtered_static(source: Path, destination: Path, variables: dict[str, bool]) -> None:
    with xr.open_dataset(source) as dataset:
        selected = [name for name in dataset.data_vars if variables.get(name, True)]
        dataset[selected].to_netcdf(destination)


def main() -> Path:
    root = FEATURE_FLAGS_TEST_CUBE_PATH.parent
    root.mkdir(parents=True, exist_ok=True)
    topography = root / "topography_test.nc"
    landcover = root / "landcover_test.nc"
    human = root / "human_activity_test.nc"
    temporal = root / "time_test.nc"
    daily = root / "era5_daily_test.nc"
    meteorology = root / "era5_grid_test.nc"
    _filtered_static(TOPOGRAPHY_CUBE_PATH, topography, TEST_PROFILE["topography"])
    _filtered_static(LANDCOVER_CUBE_PATH, landcover, TEST_PROFILE["landcover"])
    construir_capa_actividad_humana(
        GRID_PATH, SPATIAL_CUBE_PATH, human, HUMAN_ACTIVITY_RAW_DIR,
        inclusion_flags=TEST_PROFILE["human_activity"],
    )
    guardar_datacubo_temporal(
        crear_datacubo_temporal(START, END, TEST_PROFILE["time"]), temporal
    )
    hourly = cargar_era5_horario(listar_archivos_era5(ERA5_RAW_DIR, "2018-12-01", END))
    try:
        crear_meteorologia_diaria(hourly, TEST_PROFILE["meteorology"]).to_netcdf(daily)
    finally:
        hourly.close()
    if meteorology.exists():
        meteorology.unlink()
    interpolar_al_grid(daily, SPATIAL_CUBE_PATH, GRID_PATH, meteorology, START, END)
    construir_datacubo_completo(
        SPATIAL_CUBE_PATH, topography, landcover, human, temporal, meteorology,
        EGIF_TARGET_PATH, FEATURE_FLAGS_TEST_CUBE_PATH, START, END,
    )
    return FEATURE_FLAGS_TEST_CUBE_PATH


if __name__ == "__main__":
    print(main())
