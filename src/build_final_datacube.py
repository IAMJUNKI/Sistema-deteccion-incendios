"""Promueve el perfil validado al datacubo canónico sin exportar Parquet.

Todas las salidas se construyen primero como temporales y se sustituyen al final;
por tanto, un fallo no deja un cubo final a medio escribir.
"""

import shutil
from pathlib import Path

import xarray as xr

from src.config import (
    DATACUBE_END, DATACUBE_PATH, DATACUBE_START, EGIF_TARGET_PATH, ERA5_DAILY_PATH,
    ERA5_RAW_DIR, GRID_PATH, HUMAN_ACTIVITY_CUBE_PATH, HUMAN_ACTIVITY_RAW_DIR,
    LANDCOVER_CUBE_PATH, METEOROLOGY_CONTEXT_START, METEOROLOGY_CUBE_PATH,
    SPATIAL_CUBE_PATH, TIME_CUBE_PATH, TOPOGRAPHY_CUBE_PATH,
)
from src.feature_flags import TEST_PROFILE
from src.features.time import crear_datacubo_temporal, guardar_datacubo_temporal
from src.geospatial.human_activity import construir_capa_actividad_humana
from src.ingestion.meteorology import (
    cargar_era5_horario, crear_meteorologia_diaria, interpolar_al_grid, listar_archivos_era5,
)
from src.pipeline import construir_datacubo_completo


def _filter_layer(path: Path, flags: dict[str, bool], temporary: Path) -> None:
    with xr.open_dataset(path) as source:
        variables = [name for name in source.data_vars if flags.get(name, True)]
        source[variables].to_netcdf(temporary)


def main() -> Path:
    work = DATACUBE_PATH.parent / "_building_final"
    work.mkdir(parents=True, exist_ok=True)
    topography = work / "topography.nc"
    landcover = work / "landcover.nc"
    human = work / "human_activity.nc"
    temporal = work / "time.nc"
    daily = work / "era5_daily.nc"
    meteorology = work / "era5_grid.nc"
    cube = work / "galicia_1km.nc"

    _filter_layer(TOPOGRAPHY_CUBE_PATH, TEST_PROFILE["topography"], topography)
    _filter_layer(LANDCOVER_CUBE_PATH, TEST_PROFILE["landcover"], landcover)
    construir_capa_actividad_humana(
        GRID_PATH, SPATIAL_CUBE_PATH, human, HUMAN_ACTIVITY_RAW_DIR,
        inclusion_flags=TEST_PROFILE["human_activity"],
    )
    guardar_datacubo_temporal(
        crear_datacubo_temporal(DATACUBE_START, DATACUBE_END, TEST_PROFILE["time"]), temporal
    )
    hourly = cargar_era5_horario(
        listar_archivos_era5(ERA5_RAW_DIR, METEOROLOGY_CONTEXT_START, DATACUBE_END)
    )
    try:
        crear_meteorologia_diaria(hourly, TEST_PROFILE["meteorology"]).to_netcdf(daily)
    finally:
        hourly.close()
    interpolar_al_grid(daily, SPATIAL_CUBE_PATH, GRID_PATH, meteorology, DATACUBE_START, DATACUBE_END)
    construir_datacubo_completo(
        SPATIAL_CUBE_PATH, topography, landcover, human, temporal, meteorology,
        EGIF_TARGET_PATH, cube, DATACUBE_START, DATACUBE_END,
    )

    # Promoción atómica por archivo; los Parquet no participan en este script.
    for temporary, destination in (
        (topography, TOPOGRAPHY_CUBE_PATH), (landcover, LANDCOVER_CUBE_PATH),
        (human, HUMAN_ACTIVITY_CUBE_PATH), (temporal, TIME_CUBE_PATH),
        (daily, ERA5_DAILY_PATH), (meteorology, METEOROLOGY_CUBE_PATH), (cube, DATACUBE_PATH),
    ):
        shutil.copy2(temporary, destination)
    return DATACUBE_PATH


if __name__ == "__main__":
    print(main())
