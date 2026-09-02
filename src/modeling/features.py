"""Contratos de conjuntos de predictores para los experimentos de ML."""

from collections.abc import Iterable

FORBIDDEN_COLUMNS = {
    "fecha",
    "cell_id",
    "year",
    "is_galicia",
    "target_ignicion",
    "burned_area_ha",
    "large_fire_500ha",
    "is_near_ignition_25x25_10d",
    "fire_weather_index",
}

# Hipótesis de simplificación a validar en 2022. No alteran el Parquet ni el cubo.
FEATURE_SET_REMOVALS = {
    "completo": set(),
    "sin_topografia_redundante": {"roughness_mean", "roughness_std"},
    "sin_calendario_redundante": {
        "day_of_year",
        "month",
        "iso_week",
        "month_sin",
        "month_cos",
    },
    "temporal_compacto": {
        "roughness_mean",
        "roughness_std",
        "day_of_year",
        "month",
        "iso_week",
        "month_sin",
        "month_cos",
    },
    "temporal_compacto_meteo_diaria": {
        "roughness_mean",
        "roughness_std",
        "day_of_year",
        "month",
        "iso_week",
        "month_sin",
        "month_cos",
        "temperature_max_12_18h",
        "relative_humidity_min_12_18h",
        "wind_speed_max_12_18h",
    },
    "temporal_compacto_meteo_ventana": {
        "roughness_mean",
        "roughness_std",
        "day_of_year",
        "month",
        "iso_week",
        "month_sin",
        "month_cos",
        "temperature_max",
        "relative_humidity_min",
        "wind_speed_max",
    },
}


def resolve_feature_set(predictors: Iterable[str], name: str = "completo") -> list[str]:
    """Devuelve un conjunto de variables válido y libre de fuga de información.

    Los conjuntos reducidos son candidatos experimentales. La decisión se toma
    comparando exclusivamente la validación temporal de 2022.
    """
    if name not in FEATURE_SET_REMOVALS:
        valid = ", ".join(sorted(FEATURE_SET_REMOVALS))
        raise ValueError(f"Conjunto desconocido: {name}. Opciones: {valid}")

    available = set(predictors)
    forbidden = sorted(available & FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Predictores prohibidos detectados: {forbidden}")

    selected = available - FEATURE_SET_REMOVALS[name]
    if not selected:
        raise ValueError("El conjunto de variables no puede estar vacío.")
    return sorted(selected)
