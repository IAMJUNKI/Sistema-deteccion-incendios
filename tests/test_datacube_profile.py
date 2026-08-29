"""Contrato de la única vía de construcción del datacubo."""

from src.datacube_profile import CANONICAL_PROFILE
from src.features.tabular import obtener_columnas_predictoras


def test_perfil_canonico_excluye_calendario_y_variables_descartadas() -> None:
    assert not any(CANONICAL_PROFILE["time"].values())
    assert not CANONICAL_PROFILE["topography"]["roughness_mean"]
    assert not CANONICAL_PROFILE["topography"]["roughness_std"]
    assert not CANONICAL_PROFILE["landcover"]["forest_cover_fraction"]
    assert not CANONICAL_PROFILE["human_activity"]["distance_to_road_m"]
    assert not CANONICAL_PROFILE["human_activity"]["distance_to_residential_area_m"]


def test_contrato_publica_cincuenta_predictores() -> None:
    static_variables = {
        name
        for group in ("topography", "landcover", "human_activity")
        for name, include in CANONICAL_PROFILE[group].items()
        if include
    }
    meteorology_variables = {
        name for name, include in CANONICAL_PROFILE["meteorology"].items() if include
    }
    outcomes = {
        "is_galicia",
        "target_ignicion",
        "burned_area_ha",
        "large_fire_500ha",
        "is_near_ignition_25x25_10d",
    }

    predictors = obtener_columnas_predictoras(static_variables | meteorology_variables | outcomes)

    assert len(predictors) == 50
    assert {"x", "y", "fecha", "cell_id", "year"}.isdisjoint(predictors)
