"""Entrenamiento y carga de modelos de riesgo por horizonte.

Cada horizonte diario tiene su propio modelo y calibrador. El modelo base se
entrena con un muestreo de negativos para hacer viable el problema. La familia
operativa de 48 variables aplica corrección de prior y Platt sobre un año de
calibración separado; la familia histórica de 50 conserva su calibrador
isotónico para rollback. En ambos casos se preserva la prevalencia real antes
de exponer ``prob_risk`` en producción.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.features.canonical_contract import (
    CANONICAL_FEATURE_SCHEMA_VERSION,
    EGIF_48_FEATURE_CONTRACT_VERSION,
    ensure_feature_matrix_for_contract,
    load_feature_columns_for_contract,
    validate_feature_contract,
)
from src.features.operational_benchmark import (
    OPERATIONAL_ALIGNMENT_VERSION,
    OPERATIONAL_TEMPORAL_SEMANTICS,
)
from src.features.operational_features import OPERATIONAL_FEATURES, ensure_feature_matrix

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - el entorno del proyecto incluye LightGBM
    LGBMClassifier = None

LOGGER = logging.getLogger(__name__)
FEATURE_SCHEMA_VERSION = "operational-risk-v1"


class IdentityCalibrator:
    """Calibrador neutro para validaciones sintéticas sin positivos."""

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)


@dataclass
class ForecastRiskModel:
    """Modelo serializable con calibración y contrato de features."""

    base_model: object
    calibrator: object
    feature_columns: list[str]
    horizon_days: int
    metadata: dict[str, object]
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    model_family: str = "legacy"

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        schema_version = getattr(
            self,
            "feature_schema_version",
            self.metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        )
        if schema_version in {
            CANONICAL_FEATURE_SCHEMA_VERSION,
            EGIF_48_FEATURE_CONTRACT_VERSION,
        }:
            matrix = ensure_feature_matrix_for_contract(
                features, self.feature_columns, schema_version
            )
        else:
            matrix = ensure_feature_matrix(features, self.feature_columns)
        raw = self.base_model.predict_proba(matrix)[:, 1]
        calibrated = np.clip(self.calibrator.transform(raw), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])


def _sample_training_rows(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    negative_ratio: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    positives = df[df[target_col] == 1]
    negatives = df[df[target_col] == 0]
    if positives.empty:
        raise ValueError("El conjunto de entrenamiento no contiene positivos.")
    desired = min(len(negatives), len(positives) * negative_ratio)
    sampled_negatives = negatives.sample(n=desired, random_state=random_state)
    return pd.concat([positives, sampled_negatives], ignore_index=True).sample(
        frac=1, random_state=random_state
    )


def _metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    result = {
        "n_samples": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "pr_auc": 0.0,
        "roc_auc": 0.5,
        "brier_score": 0.0,
    }
    if len(y) and len(np.unique(y)) > 1:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["brier_score"] = float(brier_score_loss(y, p))
    return result


def train_horizon_model(
    dataset: pd.DataFrame,
    *,
    horizon_days: int,
    output_dir: str | Path = "data/models",
    train_years: Iterable[int] = (2019, 2020, 2021),
    validation_year: int = 2022,
    test_years: Iterable[int] = (2023, 2024),
    negative_ratio: int = 50,
    random_state: int = 42,
) -> tuple[ForecastRiskModel, dict[str, object]]:
    """Entrena, calibra y serializa un modelo para un horizonte."""

    if LGBMClassifier is None:
        raise RuntimeError("LightGBM no está instalado en el entorno actual.")
    if "fecha" not in dataset.columns or "target" not in dataset.columns:
        raise ValueError("El dataset debe contener 'fecha' y 'target'.")

    data = dataset.copy()
    data["fecha"] = pd.to_datetime(data["fecha"])
    if "horizon_days" in data.columns:
        data = data[data["horizon_days"] == horizon_days]
    if data.empty:
        raise ValueError(f"No hay datos para horizonte T+{horizon_days}.")
    data["year"] = data["fecha"].dt.year

    train_years = tuple(train_years)
    test_years = tuple(test_years)
    train_raw = data[data["year"].isin(train_years)].copy()
    validation = data[data["year"] == validation_year].copy()
    test = data[data["year"].isin(test_years)].copy()
    train = _sample_training_rows(
        train_raw,
        negative_ratio=negative_ratio,
        random_state=random_state,
    )

    matrix_train = ensure_feature_matrix(train)
    matrix_validation = ensure_feature_matrix(validation)
    matrix_test = ensure_feature_matrix(test)
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(matrix_train, train["target"])

    validation_raw = model.predict_proba(matrix_validation)[:, 1]
    if len(validation) and validation["target"].nunique() > 1:
        calibrator: object = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(validation_raw, validation["target"].to_numpy())
    else:
        calibrator = IdentityCalibrator()

    validation_calibrated = np.clip(calibrator.transform(validation_raw), 0.0, 1.0)
    test_raw = model.predict_proba(matrix_test)[:, 1]
    test_calibrated = np.clip(calibrator.transform(test_raw), 0.0, 1.0)
    metrics = {
        "horizon_days": horizon_days,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "weather_source": "era5_perfect_benchmark",
        "provider_training_context": "era5_perfect_benchmark",
        "model_version": f"legacy-lightgbm-t{horizon_days}",
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "train": _metrics(train["target"], model.predict_proba(matrix_train)[:, 1]),
        "validation": _metrics(validation["target"], validation_calibrated),
        "test": _metrics(test["target"], test_calibrated),
        "train_years": list(train_years),
        "validation_year": validation_year,
        "test_years": list(test_years),
        "negative_ratio": negative_ratio,
        "feature_columns": OPERATIONAL_FEATURES,
    }
    artifact = ForecastRiskModel(
        base_model=model,
        calibrator=calibrator,
        feature_columns=list(OPERATIONAL_FEATURES),
        horizon_days=horizon_days,
        metadata=metrics,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_family="legacy-operational",
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / f"forecast_risk_t{horizon_days}.joblib")
    (output / f"forecast_risk_t{horizon_days}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return artifact, metrics


def train_all_horizon_models(
    horizon_datasets: dict[int, pd.DataFrame],
    *,
    output_dir: str | Path = "data/models",
    **kwargs: object,
) -> dict[int, dict[str, object]]:
    """Entrena y guarda T+1, T+2 y T+3."""

    reports: dict[int, dict[str, object]] = {}
    for horizon, dataset in sorted(horizon_datasets.items()):
        _, report = train_horizon_model(
            dataset,
            horizon_days=horizon,
            output_dir=output_dir,
            **kwargs,
        )
        reports[horizon] = report
    return reports


def load_horizon_model(
    horizon_days: int,
    *,
    model_dir: str | Path = "data/models",
    model_family: str | None = None,
) -> ForecastRiskModel:
    """Carga un modelo serializado sin volver a entrenarlo."""

    directory = Path(model_dir)
    requested_family = (model_family or os.getenv("FORECAST_MODEL_FAMILY", "auto")).strip().lower()
    egif48_path = directory / f"forecast_risk_egif_48_t{horizon_days}.joblib"
    canonical_path = directory / f"forecast_risk_egif_t{horizon_days}.joblib"
    legacy_path = directory / f"forecast_risk_t{horizon_days}.joblib"
    aliases = {
        "48": "egif_48",
        "egif48": "egif_48",
        "egif-48": "egif_48",
        "egif-2d-48": "egif_48",
        EGIF_48_FEATURE_CONTRACT_VERSION: "egif_48",
        "50": "canonical",
        "egif50": "canonical",
        "egif_50": "canonical",
        "egif-2d": "canonical",
    }
    requested_family = aliases.get(requested_family, requested_family)
    if requested_family not in {"auto", "egif_48", "canonical", "legacy"}:
        raise ValueError(
            "FORECAST_MODEL_FAMILY debe ser auto, egif_48, canonical o legacy."
        )
    if requested_family == "egif_48":
        candidates = [egif48_path]
    elif requested_family == "canonical":
        candidates = [canonical_path]
    elif requested_family == "legacy":
        candidates = [legacy_path]
    else:
        candidates = [egif48_path, canonical_path, legacy_path]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No existe el modelo operativo para T+{horizon_days}: "
            f"{', '.join(str(candidate) for candidate in candidates)}. "
            "Ejecuta el entrenamiento offline antes de iniciar la inferencia."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, ForecastRiskModel):
        raise TypeError(f"El artefacto {path} no contiene ForecastRiskModel.")
    if artifact.horizon_days != horizon_days:
        raise ValueError(
            f"El artefacto {path} declara T+{artifact.horizon_days}, "
            f"pero se solicitó T+{horizon_days}."
        )
    schema_version = getattr(
        artifact,
        "feature_schema_version",
        artifact.metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
    )
    declared_schema = artifact.metadata.get("feature_schema_version", schema_version)
    if schema_version != declared_schema:
        raise ValueError(f"El artefacto {path} declara dos esquemas de features distintos.")
    if schema_version in {
        CANONICAL_FEATURE_SCHEMA_VERSION,
        EGIF_48_FEATURE_CONTRACT_VERSION,
    }:
        expected = load_feature_columns_for_contract(
            schema_version, os.getenv("EGIF_DATASET_DIR")
        )
        validate_feature_contract(artifact.feature_columns, schema_version)
        if artifact.feature_columns != expected:
            raise ValueError(
                f"El artefacto {path} no coincide con el contrato {schema_version}."
            )
        if schema_version == EGIF_48_FEATURE_CONTRACT_VERSION:
            if artifact.metadata.get("alignment_version") != OPERATIONAL_ALIGNMENT_VERSION:
                raise ValueError(
                    f"El artefacto {path} no declara la alineación temporal "
                    f"{OPERATIONAL_ALIGNMENT_VERSION}."
                )
            if artifact.metadata.get("training_temporal_semantics") != OPERATIONAL_TEMPORAL_SEMANTICS:
                raise ValueError(
                    f"El artefacto {path} no declara la semántica temporal operativa esperada."
                )
    elif schema_version == FEATURE_SCHEMA_VERSION:
        if artifact.feature_columns != list(OPERATIONAL_FEATURES):
            raise ValueError(
                f"El artefacto {path} no coincide con el contrato de features "
                f"{FEATURE_SCHEMA_VERSION}."
            )
    else:
        raise ValueError(f"El artefacto {path} usa un esquema no soportado: {schema_version}.")
    return artifact


def add_risk_outputs(
    features: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Añade probabilidad, ranking y una recomendación preventiva.

    El nivel se basa en percentiles para priorizar recursos escasos. No es una
    orden automática de despliegue: la decisión final debe incorporar medios
    disponibles, accesibilidad, exposición y confirmación del centro operativo.
    """

    output = features.copy()
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    output["prob_risk"] = probabilities
    output["prob_riesgo"] = probabilities
    output["percentil_riesgo"] = pd.Series(probabilities, index=output.index).rank(pct=True)
    output["risk_level"] = pd.cut(
        output["percentil_riesgo"],
        bins=[-np.inf, 0.80, 0.95, 0.99, np.inf],
        labels=["bajo", "moderado", "alto", "extremo"],
        include_lowest=True,
    ).astype(str)
    output["nivel_riesgo"] = output["risk_level"]
    output["operational_priority"] = output["percentil_riesgo"]
    output["recommended_action"] = pd.cut(
        output["percentil_riesgo"],
        bins=[-np.inf, 0.80, 0.95, 0.99, np.inf],
        labels=[
            "vigilancia_rutinaria",
            "vigilancia_reforzada",
            "preposicion_medios",
            "preposicion_prioritaria",
        ],
        include_lowest=True,
    ).astype(str)
    return output
