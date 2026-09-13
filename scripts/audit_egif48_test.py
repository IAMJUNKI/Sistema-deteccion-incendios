#!/usr/bin/env python3
"""Audita la familia EGIF-48 congelada sobre el año reservado.

Este script no reentrena modelos. Carga ``forecast_risk_egif_48_t{1,2,3}.joblib``
y repite, sobre la población completa del año indicado, las comprobaciones que
el pipeline definitivo hace sobre su modelo interno: contrato, cobertura,
métricas de ranking, señal espacial diaria, FWI, fiabilidad, errores, target y
niveles de riesgo.

La familia EGIF-48 se entrena con ``target_day_features=True``. Por tanto, el
resultado es un benchmark retrospectivo con meteorología ERA5 del día objetivo,
no una evaluación de un forecast meteorológico archivado. El informe lo marca
explícitamente para evitar presentar este test como validación operacional pura.

Ejemplo:

    PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
        scripts/audit_egif48_test.py --include-van-wagner
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.entrenamiento import calibracion, metricas
from src.features.canonical_contract import (
    EGIF_48_FEATURE_CONTRACT_VERSION,
    ensure_feature_matrix_for_contract,
    load_feature_columns_for_contract,
    validate_feature_contract,
)
from src.features.operational_benchmark import (
    MEMORY_COLUMNS,
    OPERATIONAL_ALIGNMENT_VERSION,
    OPERATIONAL_TEMPORAL_SEMANTICS,
    validate_operational_dataset,
)
from src.models.canonical_training import _dataset_for_year
from src.models.forecast_risk_model import ForecastRiskModel
from src.operational.artifacts import atomic_write_json, sha256_file

HORIZONS = (1, 2, 3)
FPR_OBJECTIVE = 0.05
TOP_K_DAILY = 0.01
SEASON_MONTHS = (6, 7, 8, 9)
EXPECTED_CELLS = 29_601
CONTEXT_COLUMNS = (
    "temperature_max_12_18h",
    "relative_humidity_min_12_18h",
    "wind_speed_max_12_18h",
    "precipitation_sum",
    "consecutive_dry_days",
    "elevation_mean",
    "scrub",
    "road_length_km",
    "burned_area_ha",
    "large_fire_500ha",
)
DETERMINISTIC_METRICS = (
    "n_rows",
    "n_positives",
    "prevalence",
    "pr_auc",
    "lift_vs_azar",
    "roc_auc",
    "recall_at_fpr5",
    "threshold_at_fpr",
    "fpr_achieved",
    "recall_at_top1%_daily",
    "brier_score",
)


def _parse_years(value: str) -> tuple[int, ...]:
    """Acepta ``2023`` o un intervalo/lista de años."""

    value = value.strip()
    if "-" in value:
        start, end = (int(item.strip()) for item in value.split("-", 1))
        if start > end:
            raise ValueError("El intervalo de años debe ir de menor a mayor.")
        return tuple(range(start, end + 1))
    years = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not years:
        raise ValueError("Debe indicarse al menos un año.")
    return years


def _json_safe(value: Any) -> Any:
    """Convierte escalares de NumPy/Pandas y NaN a valores JSON portables."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):  # noqa: UP038 - compatible con NumPy antiguo
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _normalise_dates(values: pd.Series) -> pd.Series:
    """Normaliza fechas diarias a la fecha civil de Galicia."""

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert("Europe/Madrid").dt.normalize().dt.tz_localize(None)


def _iter_test_frames(
    dataset_dir: Path,
    years: Iterable[int],
    feature_columns: list[str],
    *,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """Recorre las particiones conservando las columnas de auditoría."""

    requested = [
        "cell_id",
        "fecha",
        "target_ignicion",
        *feature_columns,
        "x",
        "y",
        "fire_weather_index",
        *CONTEXT_COLUMNS,
    ]
    for year in years:
        dataset = _dataset_for_year(dataset_dir, int(year))
        available = set(dataset.schema.names)
        columns = list(dict.fromkeys(column for column in requested if column in available))
        missing = sorted(set(feature_columns) - available)
        if missing:
            raise ValueError(f"Faltan features EGIF-48 en {dataset_dir}, año {year}: {missing}")
        for batch in dataset.scanner(columns=columns, batch_size=batch_size).to_batches():
            frame = batch.to_pandas()
            frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="coerce")
            frame["fecha"] = _normalise_dates(frame["fecha"])
            frame["target_ignicion"] = pd.to_numeric(
                frame["target_ignicion"], errors="coerce"
            )
            yield frame


def _load_models(
    model_dir: Path,
    prefix: str,
    feature_columns: list[str],
) -> tuple[dict[int, ForecastRiskModel], list[dict[str, Any]]]:
    """Carga y audita los tres artefactos antes de leer el año reservado."""

    models: dict[int, ForecastRiskModel] = {}
    audits: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        path = model_dir / f"{prefix}_t{horizon}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Falta el artefacto T+{horizon}: {path}")
        model = joblib.load(path)
        if not isinstance(model, ForecastRiskModel):
            raise TypeError(f"El artefacto no contiene ForecastRiskModel: {path}")
        if model.horizon_days != horizon:
            raise ValueError(f"{path} declara T+{model.horizon_days}, no T+{horizon}.")
        validate_feature_contract(model.feature_columns, EGIF_48_FEATURE_CONTRACT_VERSION)
        if list(model.feature_columns) != feature_columns:
            raise ValueError(f"El orden de features de {path} no coincide con el dataset.")

        metadata = dict(model.metadata or {})
        sidecar_path = path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
        checks = {
            "horizon": int(metadata.get("horizon_days", -1)) == horizon,
            "contract": metadata.get("feature_contract_version")
            == EGIF_48_FEATURE_CONTRACT_VERSION,
            "feature_count": len(metadata.get("feature_columns", [])) == 48,
            "train_excludes_test": 2023 not in metadata.get("train_years", []),
            "calibration_excludes_test": metadata.get("calibration_year") != 2023,
            "validation_excludes_test": metadata.get("validation_year") != 2023,
            "test_declares_year": 2023 in metadata.get("test_years", []),
            "operational_dataset_semantics": metadata.get("training_temporal_semantics")
            == OPERATIONAL_TEMPORAL_SEMANTICS,
            "platt_calibration": metadata.get("calibration_method") == "prior_correction+platt",
            "no_fwi_predictor": "fire_weather_index" not in model.feature_columns,
            "sidecar_present": sidecar_path.exists(),
            "sidecar_contract": sidecar.get("feature_contract_version")
            == EGIF_48_FEATURE_CONTRACT_VERSION
            if sidecar
            else False,
        }
        audits.append(
            {
                "horizon_days": horizon,
                "artifact": str(path),
                "artifact_sha256": sha256_file(path),
                "sidecar": str(sidecar_path),
                "model_family": model.model_family,
                "model_version": metadata.get("model_version"),
                "checks": checks,
                "status": "ok" if all(checks.values()) else "warning",
            }
        )
        models[horizon] = model
    return models, audits


def _roc_auc_in_season(y: np.ndarray, score: np.ndarray, dates: np.ndarray) -> float:
    months = pd.to_datetime(dates).month.to_numpy()
    selected = np.isin(months, SEASON_MONTHS)
    if selected.sum() == 0 or np.unique(y[selected]).size < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y[selected], score[selected]))


def _roc_auc_within_day(y: np.ndarray, score: np.ndarray, day_index: np.ndarray) -> float:
    """Promedia ROC-AUC diario sin construir un DataFrame de 11 millones de filas."""

    order = np.argsort(day_index, kind="stable")
    days = day_index[order]
    cuts = np.flatnonzero(np.diff(days)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(days)]))
    from sklearn.metrics import roc_auc_score

    values = []
    for start, end in zip(starts, ends):
        y_day = y[order[start:end]]
        if np.unique(y_day).size == 2:
            values.append(roc_auc_score(y_day, score[order[start:end]]))
    return float(np.mean(values)) if values else float("nan")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_safe(frame.to_dict(orient="records"))


def _metadata_comparison(
    audit: dict[str, Any], model: ForecastRiskModel, actual: dict[str, float]
) -> dict[str, Any]:
    """Comprueba que el ``test`` guardado junto al modelo se reproduce."""

    declared = model.metadata.get("test", {})
    differences = {}
    for key in DETERMINISTIC_METRICS:
        expected = declared.get(key)
        observed = actual.get(key)
        if expected is None or observed is None:
            continue
        differences[key] = {
            "declared": expected,
            "observed": observed,
            "absolute_difference": abs(float(expected) - float(observed)),
        }
    consistent = all(item["absolute_difference"] <= 1e-8 for item in differences.values())
    audit["metadata_test_comparison"] = {
        "status": "ok" if consistent else "warning",
        "metrics": differences,
    }
    return audit


def _risk_report(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    levels, thresholds = calibracion.risk_levels(probability)
    names = {0: "Bajo", 1: "Moderado", 2: "Alto", 3: "Extremo"}
    rows = []
    for level in sorted(np.unique(levels)):
        selected = levels == level
        count = int(selected.sum())
        positives = int(y[selected].sum())
        rows.append(
            {
                "nivel": names[int(level)],
                "celdas_dia": count,
                "porcentaje_territorio": count / len(y) if len(y) else 0.0,
                "igniciones": positives,
                "incidencia": positives / count if count else 0.0,
            }
        )
    result = {"thresholds": thresholds, "levels": rows}
    baseline = next((row["incidencia"] for row in rows if row["nivel"] == "Bajo"), 0.0)
    if baseline:
        for row in rows:
            row["veces_sobre_nivel_bajo"] = row["incidencia"] / baseline
    return result


def _target_report(
    y: np.ndarray,
    raw: np.ndarray,
    probability: np.ndarray,
    days: np.ndarray,
    positive_context: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Repite la sensibilidad del pipeline cambiando solo la definición del positivo."""

    if positive_context.empty:
        return []
    area = positive_context["burned_area_ha"].to_numpy(dtype=float)
    large = positive_context["large_fire_500ha"].to_numpy(dtype=float)
    positive_order = np.flatnonzero(y == 1)
    if len(positive_order) != len(area):
        raise ValueError("El contexto de positivos no está alineado con el vector de etiquetas.")
    definitions = (
        ("todas las igniciones", np.ones(len(area), dtype=bool)),
        ("superficie ≥ 1 ha", area >= 1),
        ("superficie ≥ 10 ha", area >= 10),
        ("superficie ≥ 100 ha", area >= 100),
        ("grandes incendios (≥500 ha)", large == 1),
    )
    rows = []
    for label, positive_mask in definitions:
        count = int(positive_mask.sum())
        if count < 10:
            continue
        target = np.zeros_like(y)
        target[positive_order] = positive_mask.astype(np.int8)
        result = metricas.evaluate(
            target,
            raw,
            probability if label == "todas las igniciones" else None,
            days,
            fpr_max=FPR_OBJECTIVE,
            top_k=TOP_K_DAILY,
            n_boot=bootstrap_samples,
            seed=seed,
        )
        rows.append({"definicion": label, "igniciones": count, **result})
    return rows


def _fwi_cems_report(
    y: np.ndarray,
    fwi: np.ndarray,
    days: np.ndarray,
    *,
    bootstrap_samples: int,
) -> dict[str, Any] | None:
    valid = np.isfinite(fwi) & (fwi != -9999)
    if not valid.any():
        return None
    return {
        "variante": "FWI oficial CEMS",
        "n_rows": int(valid.sum()),
        "metrics": metricas.evaluate(
            y[valid],
            fwi[valid],
            None,
            days[valid],
            fpr_max=FPR_OBJECTIVE,
            top_k=TOP_K_DAILY,
            n_boot=bootstrap_samples,
            seed=142,
        ),
        "caveat": "Baseline externo; fire_weather_index no es predictor EGIF-48.",
    }


def _fwi_van_wagner_report(
    dataset_dir: Path,
    years: tuple[int, ...],
    *,
    batch_size: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Calcula el baseline Van Wagner a 1 km, como en ``etapa_fwi``."""

    from src.baselines.fwi_van_wagner import calcular_fwi

    required = [
        "cell_id",
        "fecha",
        "temperature_max_12_18h",
        "relative_humidity_min_12_18h",
        "wind_speed_max_12_18h",
        "precipitation_sum",
        "target_ignicion",
    ]
    frames = []
    for year in years:
        dataset = _dataset_for_year(dataset_dir, year)
        missing = sorted(set(required) - set(dataset.schema.names))
        if missing:
            raise ValueError(f"Faltan columnas para Van Wagner en {dataset_dir}: {missing}")
        for batch in dataset.scanner(columns=required, batch_size=batch_size).to_batches():
            frame = batch.to_pandas()
            frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="coerce").astype("int64")
            frame["fecha"] = _normalise_dates(frame["fecha"])
            frame["target_ignicion"] = pd.to_numeric(frame["target_ignicion"], errors="coerce")
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    indices = calcular_fwi(
        data,
        col_celda="cell_id",
        col_fecha="fecha",
        col_t="temperature_max_12_18h",
        col_h="relative_humidity_min_12_18h",
        col_w="wind_speed_max_12_18h",
        col_p="precipitation_sum",
        verbose=False,
    )
    score = np.nan_to_num(indices["fwi"].to_numpy(dtype=float), nan=0.0)
    y = data["target_ignicion"].to_numpy(dtype=np.int8)
    days = data["fecha"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    result = {
        "variante": "FWI Van Wagner 1 km",
        "n_rows": int(len(data)),
        "metrics": metricas.evaluate(
            y,
            score,
            None,
            days,
            fpr_max=FPR_OBJECTIVE,
            top_k=TOP_K_DAILY,
            n_boot=bootstrap_samples,
            seed=143,
        ),
        "caveat": (
            "Usa extremos 12-18 h en lugar de mediodía solar; sirve como ordenación comparativa, "
            "no como categoría oficial FWI."
        ),
    }
    del data, indices
    return result


def run_audit(
    *,
    model_dir: Path,
    dataset_dir: Path,
    years: tuple[int, ...],
    prefix: str,
    batch_size: int,
    bootstrap_samples: int,
    include_van_wagner: bool,
) -> dict[str, Any]:
    """Ejecuta la auditoría completa sobre artefactos ya entrenados."""

    dataset_metadata = validate_operational_dataset(
        dataset_dir, require_contract=EGIF_48_FEATURE_CONTRACT_VERSION
    )
    feature_columns = load_feature_columns_for_contract(
        EGIF_48_FEATURE_CONTRACT_VERSION, dataset_dir
    )
    models, artifact_audits = _load_models(model_dir, prefix, feature_columns)

    labels: list[np.ndarray] = []
    days: list[np.ndarray] = []
    cell_ids: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    fwi_values: list[np.ndarray] = []
    predictions = {horizon: {"raw": [], "probability": []} for horizon in HORIZONS}
    positive_context: dict[int, list[pd.DataFrame]] = {horizon: [] for horizon in HORIZONS}
    rows_seen = 0
    rows_dropped = 0
    for frame in _iter_test_frames(
        dataset_dir, years, feature_columns, batch_size=batch_size
    ):
        rows_seen += len(frame)
        matrix = ensure_feature_matrix_for_contract(
            frame, feature_columns, EGIF_48_FEATURE_CONTRACT_VERSION
        )
        target = frame["target_ignicion"]
        valid = target.notna() & target.isin([0, 1])
        valid &= matrix.notna().all(axis=1)
        valid &= np.isfinite(matrix.to_numpy(dtype=float)).all(axis=1)
        rows_dropped += int((~valid).sum())
        if not valid.any():
            continue
        frame = frame.loc[valid].reset_index(drop=True)
        matrix = matrix.loc[valid].reset_index(drop=True)
        y = frame["target_ignicion"].to_numpy(dtype=np.int8)
        date_values = frame["fecha"].to_numpy(dtype="datetime64[D]")
        labels.append(y)
        days.append(date_values.astype(np.int64))
        cell_ids.append(frame["cell_id"].to_numpy(dtype=np.int64))
        if {"x", "y"}.issubset(frame.columns):
            scales.append(
                pd.to_numeric(frame["x"], errors="coerce").to_numpy(dtype=float)
                + pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
            )
        if "fire_weather_index" in frame:
            fwi_values.append(pd.to_numeric(frame["fire_weather_index"], errors="coerce").to_numpy(dtype=float))

        for horizon, model in models.items():
            # Se conserva float64 para que las métricas deterministas coincidan con las
            # guardadas por train_egif_operational.py, que evalúa sin esta reducción.
            raw = model.base_model.predict_proba(matrix)[:, 1]
            probability = model.predict_proba(frame)[:, 1]
            predictions[horizon]["raw"].append(raw)
            predictions[horizon]["probability"].append(probability)
            positive = y == 1
            if positive.any():
                context_columns = [
                    column for column in ("cell_id", "fecha", *CONTEXT_COLUMNS) if column in frame
                ]
                context = frame.loc[positive, context_columns].copy()
                context["raw_score"] = raw[positive]
                context["probability"] = probability[positive]
                positive_context[horizon].append(context.reset_index(drop=True))

    if not labels:
        raise ValueError("No hay filas completas evaluables en el año reservado.")
    y = np.concatenate(labels)
    day_index = np.concatenate(days)
    cell_id = np.concatenate(cell_ids)
    fwi = np.concatenate(fwi_values) if fwi_values else np.full(len(y), np.nan)
    scale = np.concatenate(scales) if scales else np.full(len(y), np.nan)

    frame_days = pd.to_datetime(day_index, unit="D")
    day_counts = pd.Series(day_index).value_counts()
    coverage = {
        "rows_seen": rows_seen,
        "rows_evaluated": int(len(y)),
        "rows_dropped_incomplete_or_invalid": rows_dropped,
        "expected_rows_from_metadata": sum(
            int(dataset_metadata.get("annual_files", {}).get(str(year), {}).get("rows", 0))
            for year in years
        ),
        "unique_days": int(pd.Series(day_index).nunique()),
        "first_date": str(frame_days.min().date()),
        "last_date": str(frame_days.max().date()),
        "cells_per_day_min": int(day_counts.min()),
        "cells_per_day_max": int(day_counts.max()),
        "unique_cells": int(np.unique(cell_id).size),
        "full_population": bool(
            rows_seen == len(y)
            and rows_dropped == 0
            and np.unique(cell_id).size == EXPECTED_CELLS
            and day_counts.min() == EXPECTED_CELLS
            and day_counts.max() == EXPECTED_CELLS
        ),
    }

    metrics_by_horizon: list[dict[str, Any]] = []
    reliability_by_horizon: dict[str, Any] = {}
    risk_by_horizon: dict[str, Any] = {}
    errors_by_horizon: dict[str, Any] = {}
    targets_by_horizon: dict[str, Any] = {}
    importances_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        raw = np.concatenate(predictions[horizon]["raw"])
        probability = np.concatenate(predictions[horizon]["probability"])
        result = metricas.evaluate(
            y,
            raw,
            probability,
            day_index,
            fpr_max=FPR_OBJECTIVE,
            top_k=TOP_K_DAILY,
            n_boot=bootstrap_samples,
            seed=42 + horizon,
        )
        result["roc_auc_temporada"] = _roc_auc_in_season(y, raw, frame_days.to_numpy())
        result["roc_auc_dentro_del_dia"] = _roc_auc_within_day(y, raw, day_index)
        result["horizon_days"] = horizon
        result["model_version"] = models[horizon].metadata.get("model_version")
        metrics_by_horizon.append(result)
        reliability_by_horizon[str(horizon)] = _records(metricas.reliability_table(y, probability, 12))
        risk_by_horizon[str(horizon)] = _risk_report(y, probability)

        context = pd.concat(positive_context[horizon], ignore_index=True)
        threshold = metricas.umbral_at_fpr(raw[y == 0], FPR_OBJECTIVE)
        context["detectada"] = context["raw_score"] >= threshold
        context["mes"] = pd.to_datetime(context["fecha"]).dt.month
        error_rows = []
        available_context = [column for column in CONTEXT_COLUMNS if column in context]
        for column in [*available_context, "mes"]:
            detected_values = pd.to_numeric(
                context.loc[context.detectada, column], errors="coerce"
            )
            missed_values = pd.to_numeric(
                context.loc[~context.detectada, column], errors="coerce"
            )
            error_rows.append(
                {
                    "variable": column,
                    "media_detectadas": detected_values.mean(),
                    "media_perdidas": missed_values.mean(),
                    "diferencia": detected_values.mean() - missed_values.mean(),
                }
            )
        monthly = (
            context.groupby("mes")
            .agg(igniciones=("detectada", "size"), detectadas=("detectada", "sum"))
            .assign(recall=lambda data: data.detectadas / data.igniciones)
            .reset_index()
        )
        errors_by_horizon[str(horizon)] = {
            "threshold_at_fpr": threshold,
            "detectadas": int(context.detectada.sum()),
            "perdidas": int((~context.detectada).sum()),
            "por_variable": _records(pd.DataFrame(error_rows)),
            "por_mes": _records(monthly),
        }
        targets_by_horizon[str(horizon)] = _target_report(
            y,
            raw,
            probability,
            day_index,
            context,
            bootstrap_samples=bootstrap_samples,
            seed=42 + horizon,
        )

        feature_importance = getattr(models[horizon].base_model, "feature_importances_", None)
        if feature_importance is not None:
            importance = pd.DataFrame(
                {"feature": feature_columns, "importance": np.asarray(feature_importance, dtype=float)}
            ).sort_values("importance", ascending=False)
            importances_by_horizon[str(horizon)] = _records(importance.head(15))

        audit = next(item for item in artifact_audits if item["horizon_days"] == horizon)
        _metadata_comparison(audit, models[horizon], result)

    raw_scores = {
        f"T+{horizon}": np.concatenate(predictions[horizon]["raw"]) for horizon in HORIZONS
    }
    paired = _records(
        metricas.comparar_modelos(
            y,
            raw_scores,
            day_index=day_index,
            fpr_max=FPR_OBJECTIVE,
            top_k=TOP_K_DAILY,
            n_boot=bootstrap_samples,
            seed=42,
        )
    )
    baseline_paired: list[dict[str, Any]] = []
    valid_fwi = np.isfinite(fwi) & (fwi != -9999)
    if valid_fwi.any():
        baseline_comparison = metricas.comparar_modelos(
            y[valid_fwi],
            {**{name: score[valid_fwi] for name, score in raw_scores.items()}, "FWI CEMS": fwi[valid_fwi]},
            day_index=day_index[valid_fwi],
            fpr_max=FPR_OBJECTIVE,
            top_k=TOP_K_DAILY,
            n_boot=bootstrap_samples,
            seed=142,
        )
        baseline_comparison = baseline_comparison[
            (baseline_comparison["modelo_a"] == "FWI CEMS")
            | (baseline_comparison["modelo_b"] == "FWI CEMS")
        ]
        baseline_paired = _records(baseline_comparison)

    spatial_slices: list[dict[str, Any]] = []
    if np.isfinite(scale).all():
        cuts = np.quantile(scale, np.linspace(0, 1, 6)[1:-1])
        bands = np.searchsorted(cuts, scale)
        for horizon in HORIZONS:
            raw = raw_scores[f"T+{horizon}"]
            for band in range(5):
                selected = bands == band
                if y[selected].sum() == 0:
                    # metricas.evaluate() calcula la media de capturas diarias y NumPy avisa
                    # cuando una banda no contiene ningún positivo; aquí el cero es el
                    # resultado correcto y explícito.
                    result = metricas.evaluate(
                        y[selected],
                        raw[selected],
                        None,
                        None,
                        fpr_max=FPR_OBJECTIVE,
                        top_k=TOP_K_DAILY,
                        n_boot=bootstrap_samples,
                        seed=100 + horizon + band,
                    )
                    result["recall_at_top1%_daily"] = 0.0
                    result["recall_at_top1%_daily_ci90_low"] = 0.0
                    result["recall_at_top1%_daily_ci90_high"] = 0.0
                else:
                    result = metricas.evaluate(
                        y[selected],
                        raw[selected],
                        None,
                        day_index[selected],
                        fpr_max=FPR_OBJECTIVE,
                        top_k=TOP_K_DAILY,
                        n_boot=bootstrap_samples,
                        seed=100 + horizon + band,
                    )
                result.update(
                    {
                        "horizon_days": horizon,
                        "banda_diagonal": band,
                        "rows": int(selected.sum()),
                        "cells_aprox": int(np.unique(cell_id[selected]).size),
                        "roc_auc_dentro_del_dia": _roc_auc_within_day(
                            y[selected], raw[selected], day_index[selected]
                        ),
                    }
                )
                spatial_slices.append(result)

    report: dict[str, Any] = {
        "evaluation_protocol": "frozen_artifact_pipeline_checks",
        "test_years": list(years),
        "dataset": str(dataset_dir),
        "model_dir": str(model_dir),
        "model_prefix": prefix,
        "feature_contract_version": EGIF_48_FEATURE_CONTRACT_VERSION,
        "dataset_alignment_version": dataset_metadata.get("alignment_version"),
        "dataset_temporal_semantics": dataset_metadata.get("training_temporal_semantics"),
        "coverage": coverage,
        "artifact_checks": artifact_audits,
        "metrics": metrics_by_horizon,
        "paired_comparisons": paired,
        "baseline_paired_comparisons": baseline_paired,
        "fwi": [],
        "spatial_slices": spatial_slices,
        "feature_importance_top15": importances_by_horizon,
        "reliability": reliability_by_horizon,
        "risk_levels": risk_by_horizon,
        "errors": errors_by_horizon,
        "target_sensitivity": targets_by_horizon,
        "limitations": {
            "target_day_weather": True,
            "target_day_weather_detail": (
                "El benchmark usa meteorología ERA5 del día objetivo; las memorias sí se "
                "recalcularon hasta T-1. No equivale a un forecast meteorológico real."
            ),
            "spatial_check": (
                "Las bandas son cortes descriptivos con artefactos congelados. La validación "
                "leave-one-band-out de pipeline_definitivo requeriría reentrenar cinco modelos."
            ),
            "seed_stability": "No se reentrena: no se puede estimar dispersión por semilla con artefactos únicos.",
            "horizon_comparison": (
                "T+1/T+2/T+3 se comparan sobre las mismas fechas objetivo del benchmark; la "
                "comparación pareada no demuestra por sí sola ventaja de lead time operacional."
            ),
        },
    }
    cems = _fwi_cems_report(y, fwi, day_index, bootstrap_samples=bootstrap_samples)
    if cems is not None:
        report["fwi"].append(cems)
    if include_van_wagner:
        report["fwi"].append(
            _fwi_van_wagner_report(
                dataset_dir,
                years,
                batch_size=batch_size,
                bootstrap_samples=bootstrap_samples,
            )
        )

    report["checks"] = [
        {
            "name": "dataset_operational_contract",
            "status": "ok",
            "detail": f"{OPERATIONAL_ALIGNMENT_VERSION}; memorias hasta T-1: {sorted(MEMORY_COLUMNS)}",
        },
        {
            "name": "population_complete",
            "status": "ok" if coverage["full_population"] else "warning",
            "detail": coverage,
        },
        {
            "name": "artifact_contract_and_temporal_split",
            "status": "ok"
            if all(item["status"] == "ok" for item in artifact_audits)
            else "warning",
            "detail": "Contrato EGIF-48, separación temporal y sidecars auditados.",
        },
        {
            "name": "target_day_weather_benchmark",
            "status": "warning",
            "detail": "Resultado válido como benchmark ERA5; no es forecast meteorológico real.",
        },
        {
            "name": "spatial_leave_one_band_out",
            "status": "not_run",
            "detail": "No se reentrenaron modelos fuera de alcance de una auditoría de artefactos.",
        },
        {
            "name": "seed_stability",
            "status": "not_run",
            "detail": "No se reentrenaron réplicas con otras semillas.",
        },
    ]
    return _json_safe(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("data/models"))
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/processed/tabular/egif_operational")
    )
    parser.add_argument("--test-years", default="2023")
    parser.add_argument("--prefix", default="forecast_risk_egif_48")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--include-van-wagner",
        action="store_true",
        help="Calcula también el FWI Van Wagner a 1 km; requiere más memoria y tiempo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/technical/egif48_test_2023_audit.json"),
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("--batch-size y --bootstrap-samples deben ser positivos.")
    years = _parse_years(args.test_years)
    report = run_audit(
        model_dir=args.models_root,
        dataset_dir=args.dataset_dir,
        years=years,
        prefix=args.prefix,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
        include_van_wagner=args.include_van_wagner,
    )
    atomic_write_json(report, args.output)
    print(f"Informe: {args.output}")
    print("\nMétricas sobre población completa:")
    table = pd.DataFrame(report["metrics"])
    columns = [
        "horizon_days",
        "pr_auc",
        "roc_auc",
        "recall_at_fpr5",
        "recall_at_top1%_daily",
        "brier_score",
    ]
    print(table[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
