"""Local explanations for the serialized operational risk models.

The dashboard must expose model-derived contributions, not hand-written
heuristics that merely resemble SHAP values. SHAP remains an optional runtime
dependency because inference itself does not need it; when it is unavailable,
the caller receives an explicit error and can show a non-misleading warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.operational_features import ensure_feature_matrix


class ExplainabilityUnavailable(RuntimeError):
    """SHAP cannot be used in the current runtime or model artifact."""


def _binary_shap_values(raw_values: object, n_features: int) -> np.ndarray:
    """Normalize SHAP's binary-class return variants to one row."""

    values = raw_values.values if hasattr(raw_values, "values") else raw_values
    if isinstance(values, list):
        if len(values) == 2:
            values = values[1]
        elif values:
            values = values[0]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, 1]
    if array.ndim == 2 and array.shape[1] != n_features and array.shape[0] == n_features:
        array = array.T
    if array.ndim != 2 or array.shape[1] != n_features:
        raise ExplainabilityUnavailable(
            "SHAP devolvió una forma incompatible con el esquema de features."
        )
    return array


def explain_tree_prediction(model: object, features: pd.DataFrame) -> pd.DataFrame:
    """Return signed TreeSHAP contributions for the first supplied row.

    The contribution scale is the model's native output scale (normally
    LightGBM raw score/log-odds), not a percentage of probability. The
    probability itself remains the calibrated value shown by the pipeline.
    """

    if not hasattr(model, "base_model"):
        raise ExplainabilityUnavailable("El artefacto no expone un modelo base explicable.")
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on production env
        raise ExplainabilityUnavailable(
            "La dependencia opcional 'shap' no está instalada."
        ) from exc

    feature_columns = getattr(model, "feature_columns", None)
    schema_version = getattr(
        model,
        "feature_schema_version",
        getattr(model, "metadata", {}).get("feature_schema_version", None) if hasattr(model, "metadata") else None,
    )
    from src.features.canonical_contract import (
        CANONICAL_FEATURE_SCHEMA_VERSION,
        EGIF_48_FEATURE_CONTRACT_VERSION,
        ensure_feature_matrix_for_contract,
    )

    if schema_version in {
        CANONICAL_FEATURE_SCHEMA_VERSION,
        EGIF_48_FEATURE_CONTRACT_VERSION,
    } and feature_columns:
        matrix = ensure_feature_matrix_for_contract(
            features, feature_columns, schema_version
        )
    else:
        matrix = ensure_feature_matrix(features, feature_columns)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if matrix.empty:
        raise ExplainabilityUnavailable("No hay filas para explicar.")
    try:
        explainer = shap.TreeExplainer(model.base_model)
        raw_values = explainer.shap_values(matrix.iloc[[0]])
        values = _binary_shap_values(raw_values, matrix.shape[1])[0]
    except Exception as exc:  # pragma: no cover - backend/version dependent
        raise ExplainabilityUnavailable(f"No se pudo calcular TreeSHAP: {exc}") from exc

    return (
        pd.DataFrame(
            {
                "feature": matrix.columns,
                "value": matrix.iloc[0].to_numpy(dtype=float),
                "contribution": values,
            }
        )
        .assign(abs_contribution=lambda frame: frame["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )
