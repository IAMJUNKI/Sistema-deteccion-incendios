#!/usr/bin/env python3
"""Compara modelos EGIF y FWI con evaluación temporal y población completa.

Las familias 48 y 50 alineada se evalúan sobre el mismo dataset operativo.
El artefacto 50 histórico se evalúa sobre el datacubo canónico con su propia
semántica y se etiqueta como comparación no estrictamente pareada.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.entrenamiento.metricas import evaluate
from src.features.canonical_contract import (
    CANONICAL_FEATURE_SCHEMA_VERSION,
    EGIF_48_FEATURE_CONTRACT_VERSION,
    ensure_feature_matrix_for_contract,
    load_feature_columns_for_contract,
    validate_feature_contract,
)
from src.models.canonical_training import (
    _dataset_for_year,
    _iter_year,
    _load_target_lookup,
)
from src.models.forecast_risk_model import ForecastRiskModel
from src.operational.artifacts import atomic_write_json, sha256_file

HORIZONS = (1, 2, 3)


def _years(value: str) -> tuple[int, ...]:
    value = value.strip()
    if "-" in value:
        start, end = (int(item.strip()) for item in value.split("-", 1))
        if start > end:
            raise ValueError("El intervalo de años debe ir de menor a mayor.")
        return tuple(range(start, end + 1))
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("Debe indicarse al menos un año.")
    return result


def _artifact(path: Path, horizon: int, contract: str) -> ForecastRiskModel:
    if not path.exists():
        raise FileNotFoundError(f"Falta el artefacto T+{horizon}: {path}")
    model = joblib.load(path)
    if not isinstance(model, ForecastRiskModel):
        raise TypeError(f"El artefacto no contiene ForecastRiskModel: {path}")
    if model.horizon_days != horizon:
        raise ValueError(f"{path} declara T+{model.horizon_days}, no T+{horizon}.")
    validate_feature_contract(model.feature_columns, contract)
    return model


def _evaluate_artifact(
    model_dir: Path,
    prefix: str,
    dataset_dir: Path,
    years: tuple[int, ...],
    *,
    contract: str,
    target_day_features: bool,
    batch_size: int,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    columns = load_feature_columns_for_contract(contract, dataset_dir)
    reports = []
    for horizon in HORIZONS:
        model_path = model_dir / f"{prefix}_t{horizon}.joblib"
        model = _artifact(model_path, horizon, contract)
        labels: list[np.ndarray] = []
        raw_scores: list[np.ndarray] = []
        calibrated: list[np.ndarray] = []
        days: list[np.ndarray] = []
        for year in years:
            target_lookup = None
            if not target_day_features:
                target_lookup = _load_target_lookup(dataset_dir, (year, year + 1))
            for frame in _iter_year(
                dataset_dir,
                year,
                columns,
                batch_size=batch_size,
                target_lookup=target_lookup,
                horizon_days=horizon,
                target_day_features=target_day_features,
            ):
                frame = frame.dropna(subset=["target_ignicion"]).reset_index(drop=True)
                if frame.empty:
                    continue
                matrix = ensure_feature_matrix_for_contract(frame, columns, contract)
                raw_scores.append(model.base_model.predict_proba(matrix)[:, 1])
                calibrated.append(model.predict_proba(frame)[:, 1])
                labels.append(frame["target_ignicion"].to_numpy(dtype=np.int8))
                days.append(
                    pd.to_datetime(frame["fecha"], errors="coerce")
                    .to_numpy(dtype="datetime64[D]")
                    .astype(np.int64)
                )
        if not labels:
            raise ValueError(f"No hay filas evaluables para {model_path}.")
        y = np.concatenate(labels)
        raw = np.concatenate(raw_scores)
        probability = np.concatenate(calibrated)
        day = np.concatenate(days)
        metadata = model.metadata
        reports.append(
            {
                "family": str(model.model_family),
                "artifact": str(model_path),
                "artifact_sha256": sha256_file(model_path),
                "horizon_days": horizon,
                "contract": contract,
                "dataset": str(dataset_dir),
                "target_day_features": target_day_features,
                "comparison_scope": "aligned" if target_day_features else "legacy_semantics",
                "metrics": evaluate(
                    y,
                    raw,
                    probability,
                    day,
                    top_k=0.01,
                    n_boot=bootstrap_samples,
                    seed=42 + horizon,
                ),
                "model_version": metadata.get("model_version"),
                "dataset_hash": metadata.get("dataset_hash"),
            }
        )
    return reports


def _evaluate_fwi(
    dataset_dir: Path,
    years: tuple[int, ...],
    *,
    batch_size: int,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    reports = []
    for horizon in HORIZONS:
        labels: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        days: list[np.ndarray] = []
        for year in years:
            dataset = _dataset_for_year(dataset_dir, year)
            for batch in dataset.scanner(
                columns=["fecha", "target_ignicion", "fire_weather_index"],
                batch_size=batch_size,
            ).to_batches():
                frame = batch.to_pandas()
                frame["target_ignicion"] = pd.to_numeric(
                    frame["target_ignicion"], errors="coerce"
                )
                frame["fire_weather_index"] = pd.to_numeric(
                    frame["fire_weather_index"], errors="coerce"
                )
                valid = frame["target_ignicion"].notna() & np.isfinite(
                    frame["fire_weather_index"]
                ) & frame["fire_weather_index"].ne(-9999)
                if not valid.any():
                    continue
                selected = frame.loc[valid]
                labels.append(selected["target_ignicion"].to_numpy(dtype=np.int8))
                scores.append(selected["fire_weather_index"].to_numpy(dtype=float))
                days.append(
                    pd.to_datetime(selected["fecha"], errors="coerce")
                    .to_numpy(dtype="datetime64[D]")
                    .astype(np.int64)
                )
        if not labels:
            raise ValueError("El dataset no contiene fire_weather_index evaluable.")
        y = np.concatenate(labels)
        fwi = np.concatenate(scores)
        reports.append(
            {
                "family": "fwi-cems-baseline",
                "horizon_days": horizon,
                "contract": "baseline-not-predictor",
                "dataset": str(dataset_dir),
                "comparison_scope": "aligned_baseline",
                "metrics": evaluate(
                    y,
                    fwi,
                    None,
                    np.concatenate(days),
                    top_k=0.01,
                    n_boot=bootstrap_samples,
                    seed=142 + horizon,
                ),
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("data/models"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/tabular/egif_operational"),
    )
    parser.add_argument(
        "--legacy-dataset-dir",
        type=Path,
        default=Path("data/processed/tabular/egif"),
    )
    parser.add_argument("--test-years", default="2023")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/evaluation/model_family_comparison.json"),
    )
    parser.add_argument("--skip-fwi", action="store_true")
    args = parser.parse_args()
    years = _years(args.test_years)
    reports = []
    reports.extend(
        _evaluate_artifact(
            args.models_root / "experiments" / "comparable",
            "forecast_risk_egif_48",
            args.dataset_dir,
            years,
            contract=EGIF_48_FEATURE_CONTRACT_VERSION,
            target_day_features=True,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    reports.extend(
        _evaluate_artifact(
            args.models_root / "experiments" / "expanded",
            "forecast_risk_egif_48",
            args.dataset_dir,
            years,
            contract=EGIF_48_FEATURE_CONTRACT_VERSION,
            target_day_features=True,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    reports.extend(
        _evaluate_artifact(
            args.models_root / "experiments" / "aligned50",
            "forecast_risk_egif_50_aligned",
            args.dataset_dir,
            years,
            contract=CANONICAL_FEATURE_SCHEMA_VERSION,
            target_day_features=True,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    reports.extend(
        _evaluate_artifact(
            args.models_root,
            "forecast_risk_egif",
            args.legacy_dataset_dir,
            years,
            contract=CANONICAL_FEATURE_SCHEMA_VERSION,
            target_day_features=False,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    if not args.skip_fwi:
        reports.extend(
            _evaluate_fwi(
                args.dataset_dir,
                years,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
            )
        )
    payload = {
        "evaluation_protocol": "temporal_population_complete",
        "test_years": list(years),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "reports": reports,
    }
    atomic_write_json(payload, args.output)
    print(json.dumps({"output": str(args.output), "reports": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
