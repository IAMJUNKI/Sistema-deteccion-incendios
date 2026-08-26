"""Perfiles de variables almacenadas para construir cubos alternativos.

Los diccionarios viven junto a cada categoría (topografía, CORINE, etc.). Este
módulo solo reúne el perfil de prueba para que el workflow no tenga flags
dispersos. No define qué variables utiliza un modelo: esa decisión pertenece a
los experimentos de ML.
"""

from src.features.time import TEST_DATACUBE_VARIABLE_FLAGS as TIME_FLAGS
from src.geospatial.human_activity import TEST_DATACUBE_VARIABLE_FLAGS as HUMAN_FLAGS
from src.geospatial.topography import TEST_DATACUBE_VARIABLE_FLAGS as TOPOGRAPHY_FLAGS
from src.geospatial.vegetation import TEST_DATACUBE_VARIABLE_FLAGS as LANDCOVER_FLAGS
from src.ingestion.meteorology import TEST_DATACUBE_VARIABLE_FLAGS as METEOROLOGY_FLAGS

TEST_PROFILE = {
    "topography": TOPOGRAPHY_FLAGS,
    "landcover": LANDCOVER_FLAGS,
    "human_activity": HUMAN_FLAGS,
    "time": TIME_FLAGS,
    "meteorology": METEOROLOGY_FLAGS,
}
