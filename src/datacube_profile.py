"""Contrato único de variables del datacubo canónico.

Este perfil decide únicamente qué capas se almacenan. La selección final de
predictores para cada modelo se hace después, a partir de ``metadata.json``.
"""

from src.features.time import CANONICAL_DATACUBE_VARIABLE_FLAGS as TIME_FLAGS
from src.geospatial.human_activity import CANONICAL_DATACUBE_VARIABLE_FLAGS as HUMAN_FLAGS
from src.geospatial.topography import CANONICAL_DATACUBE_VARIABLE_FLAGS as TOPOGRAPHY_FLAGS
from src.geospatial.vegetation import CANONICAL_DATACUBE_VARIABLE_FLAGS as LANDCOVER_FLAGS
from src.ingestion.meteorology import CANONICAL_DATACUBE_VARIABLE_FLAGS as METEOROLOGY_FLAGS
from src.ingestion.fwi import CANONICAL_DATACUBE_VARIABLE_FLAGS as FWI_FLAGS

CANONICAL_PROFILE = {
    "topography": TOPOGRAPHY_FLAGS,
    "landcover": LANDCOVER_FLAGS,
    "human_activity": HUMAN_FLAGS,
    "time": TIME_FLAGS,
    "meteorology": METEOROLOGY_FLAGS,
    "fwi": FWI_FLAGS,
}
