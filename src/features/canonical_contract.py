"""Contrato versionado de entrada del modelo 2D EGIF.

El datacubo contiene columnas de trazabilidad, resultados y variables
predictoras. Este módulo mantiene la lista que puede llegar al modelo y evita
que el pipeline operativo vuelva a depender del contrato histórico de 23
variables.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

# ``egif-2d-v1`` es el contrato histórico de 50 variables. Se mantiene
# congelado porque los artefactos ya entrenados lo necesitan para rollback.
CANONICAL_FEATURE_SCHEMA_VERSION = "egif-2d-v1"
EGIF_48_FEATURE_CONTRACT_VERSION = "egif-2d-48-v1"
TARGET_COLUMN = "target_ignicion"

# Lista publicada por el metadata.json del datacubo EGIF recibido. Se conserva
# aquí como fallback para que la inferencia pueda validar el esquema aunque el
# dataset de entrenamiento no esté montado en el servidor.
DEFAULT_CANONICAL_FEATURES = (
    "agriculture",
    "artificial",
    "aspect_000_045_fraction",
    "aspect_045_090_fraction",
    "aspect_090_135_fraction",
    "aspect_135_180_fraction",
    "aspect_180_225_fraction",
    "aspect_225_270_fraction",
    "aspect_270_315_fraction",
    "aspect_315_360_fraction",
    "broadleaf_forest",
    "building_area_fraction",
    "coniferous_forest",
    "consecutive_dry_days",
    "elevation_mean",
    "elevation_std",
    "mixed_forest",
    "open_spaces",
    "precipitation_sum",
    "precipitation_sum_14d",
    "precipitation_sum_30d",
    "precipitation_sum_3d",
    "precipitation_sum_7d",
    "relative_humidity_mean",
    "relative_humidity_mean_14d",
    "relative_humidity_mean_7d",
    "relative_humidity_min",
    "relative_humidity_min_12_18h",
    "residential_area_fraction",
    "road_length_km",
    "road_length_local_km",
    "road_length_main_km",
    "road_length_other_km",
    "road_length_track_km",
    "scrub",
    "slope_mean",
    "slope_std",
    "temperature_max",
    "temperature_max_12_18h",
    "temperature_mean",
    "temperature_mean_7d",
    "temperature_min",
    "vpd_max_12_18h",
    "vpd_mean",
    "water",
    "wetlands",
    "wind_speed_max",
    "wind_speed_max_12_18h",
    "wind_speed_mean",
    "wind_speed_mean_7d",
)

# Alias corto usado por los módulos de features y por integraciones externas.
# La fuente normativa sigue siendo ``DEFAULT_CANONICAL_FEATURES`` y el
# ``metadata.json`` del datacubo cuando está disponible.
CANONICAL_FEATURES = DEFAULT_CANONICAL_FEATURES

# Contrato operativo que se entrenará con el modelo 2D revisado. Las dos
# variables retiradas son derivadas de la misma señal de precipitación y de la
# racha seca; mantenerlas junto con las memorias de precipitación introducía
# duplicación y hacía que el contrato documentado no coincidiera con el
# artefacto operativo.
EGIF_48_EXCLUDED_FEATURES = frozenset({"precipitation_sum", "consecutive_dry_days"})
EGIF_48_FEATURES = tuple(
    column for column in DEFAULT_CANONICAL_FEATURES if column not in EGIF_48_EXCLUDED_FEATURES
)

CANONICAL_WEATHER_FEATURES = (
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "temperature_max_12_18h",
    "relative_humidity_mean",
    "relative_humidity_min",
    "relative_humidity_min_12_18h",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_speed_max_12_18h",
    "vpd_mean",
    "vpd_max_12_18h",
    "precipitation_sum",
    "precipitation_sum_3d",
    "precipitation_sum_7d",
    "precipitation_sum_14d",
    "precipitation_sum_30d",
    "temperature_mean_7d",
    "relative_humidity_mean_7d",
    "wind_speed_mean_7d",
    "relative_humidity_mean_14d",
    "consecutive_dry_days",
)

FORBIDDEN_CANONICAL_COLUMNS = frozenset(
    {
        "cell_id",
        "fecha",
        "year",
        "x",
        "y",
        "is_galicia",
        "target_ignicion",
        "burned_area_ha",
        "large_fire_500ha",
        "is_near_ignition_25x25_10d",
        "precipitation_sum_1d",
        "aspect_no_data_fraction",
    }
)


def load_feature_columns(dataset_dir: str | Path | None = None) -> list[str]:
    """Carga los predictores publicados o devuelve el contrato incorporado.

    Args:
        dataset_dir: Directorio que contiene ``metadata.json``. Si no existe,
            se usa la lista versionada en el código.

    Returns:
        Predictores ordenados de forma estable.

    Raises:
        ValueError: Si el metadata declara resultados o identificadores como
            predictores, o si contiene duplicados.
    """

    columns = list(DEFAULT_CANONICAL_FEATURES)
    if dataset_dir is not None:
        metadata_path = Path(dataset_dir) / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            declared = metadata.get("predictor_columns")
            if not isinstance(declared, list) or not declared:
                raise ValueError("metadata.json no declara predictor_columns válidos.")
            columns = [str(column) for column in declared]

    if len(columns) != len(set(columns)):
        raise ValueError("El contrato canónico contiene predictores duplicados.")
    if set(columns) != set(DEFAULT_CANONICAL_FEATURES):
        missing = sorted(set(DEFAULT_CANONICAL_FEATURES) - set(columns))
        extra = sorted(set(columns) - set(DEFAULT_CANONICAL_FEATURES))
        raise ValueError(
            "El metadata no coincide con el contrato EGIF canónico de 50 variables: "
            f"faltan={missing}, extras={extra}. Resuelve primero la discrepancia entre "
            "las versiones de 47 y 50 variables."
        )
    forbidden = sorted(set(columns) & FORBIDDEN_CANONICAL_COLUMNS)
    if forbidden:
        raise ValueError(f"El contrato canónico contiene columnas prohibidas: {forbidden}")
    return sorted(columns)


def load_feature_columns_for_contract(
    contract_version: str,
    dataset_dir: str | Path | None = None,
) -> list[str]:
    """Devuelve la lista ordenada de predictores de una versión concreta.

    El datacubo publica 50 columnas porque también conserva ``precipitation_sum``
    y ``consecutive_dry_days`` para auditoría. La familia de 48 deriva su lista
    de ese contrato, pero nunca modifica el metadata del cubo ni acepta una
    lista arbitraria desde el entorno.
    """

    if contract_version == CANONICAL_FEATURE_SCHEMA_VERSION:
        return load_feature_columns(dataset_dir)
    if contract_version != EGIF_48_FEATURE_CONTRACT_VERSION:
        raise ValueError(f"Contrato de features no soportado: {contract_version}")
    canonical = load_feature_columns(dataset_dir)
    result = [column for column in canonical if column not in EGIF_48_EXCLUDED_FEATURES]
    if len(result) != 48:
        raise ValueError(
            "El contrato EGIF de 48 variables requiere exactamente 48 predictores; "
            f"se obtuvieron {len(result)}."
        )
    return result


def validate_feature_contract(
    feature_columns: Iterable[str],
    contract_version: str,
) -> None:
    """Comprueba número, conjunto y orden del contrato serializado."""

    expected = load_feature_columns_for_contract(contract_version)
    actual = list(feature_columns)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"El artefacto no coincide con {contract_version}: "
            f"n={len(actual)} (esperado {len(expected)}), faltan={missing}, extras={extra}."
        )
    forbidden = sorted(set(actual) & {"fire_weather_index", *EGIF_48_EXCLUDED_FEATURES})
    if contract_version == EGIF_48_FEATURE_CONTRACT_VERSION and forbidden:
        raise ValueError(
            f"El contrato {contract_version} contiene predictores prohibidos: {forbidden}"
        )


def validate_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_CANONICAL_FEATURES,
) -> None:
    """Valida la presencia y el carácter numérico de las features canónicas."""

    expected = list(feature_columns)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan features canónicas: {missing}")
    non_numeric = [
        column
        for column in expected
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"Las features canónicas no son numéricas: {non_numeric}")


def ensure_canonical_matrix(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_CANONICAL_FEATURES,
) -> pd.DataFrame:
    """Devuelve una matriz numérica ordenada para el modelo canónico.

    Los modelos de árboles admiten valores ausentes, pero se convierten los
    infinitos a ``NaN`` y se conserva el indicador de ausencia para que el
    comportamiento sea idéntico durante entrenamiento e inferencia.
    """

    columns = list(feature_columns)
    validate_feature_columns(frame, columns)
    matrix = frame[columns].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def ensure_feature_matrix_for_contract(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    contract_version: str,
) -> pd.DataFrame:
    """Construye una matriz numérica únicamente para el contrato declarado."""

    validate_feature_contract(feature_columns, contract_version)
    expected = list(feature_columns)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan features de {contract_version}: {missing}")
    matrix = frame[expected].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)
