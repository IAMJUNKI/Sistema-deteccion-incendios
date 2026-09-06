#!/usr/bin/env python3
"""Health check for the latest operational prediction artifact.

The command is suitable for cron, CI or a container liveness probe. It checks
the manifest, output checksum, three horizons, forecast freshness and hourly
coverage without starting a new inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.features.canonical_contract import (
    CANONICAL_FEATURE_SCHEMA_VERSION,
    EGIF_48_FEATURE_CONTRACT_VERSION,
)
from src.operational.artifacts import sha256_file


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.getenv(
                "PREDICTIONS_OUTPUT_PATH",
                "data/processed/predicciones_operativas.parquet",
            )
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            Path(os.getenv("PREDICTIONS_MANIFEST_PATH"))
            if os.getenv("PREDICTIONS_MANIFEST_PATH")
            else None
        ),
    )
    parser.add_argument("--max-age-hours", type=float, default=30.0)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument(
        "--allow-local-simulation",
        action="store_true",
        help="Permite validar una ejecución marcada como simulación local.",
    )
    parser.add_argument(
        "--fail-on-degraded",
        action="store_true",
        help="Falla también cuando se utilizó WRF 04 km como fallback.",
    )
    return parser.parse_args()


def check_run(
    output_path: Path,
    *,
    manifest_path: Path | None = None,
    max_age_hours: float = 30.0,
    min_coverage: float = 1.0,
    allow_stale: bool = False,
    fail_on_degraded: bool = False,
    allow_local_simulation: bool = False,
) -> dict[str, object]:
    """Validate one published run and raise ``RuntimeError`` on failure."""

    manifest = manifest_path or output_path.with_suffix(".manifest.json")
    if not output_path.exists():
        raise RuntimeError(f"No existe el output operativo: {output_path}")
    if not manifest.exists():
        raise RuntimeError(f"No existe el manifiesto operativo: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No se pudo leer el manifiesto {manifest}") from exc

    expected_sha = payload.get("output_sha256")
    actual_sha = sha256_file(output_path)
    if expected_sha and expected_sha != actual_sha:
        raise RuntimeError("El checksum del output no coincide con el manifiesto.")
    forecast_path = payload.get("forecast_archive_path")
    forecast_sha = payload.get("forecast_archive_sha256")
    if forecast_path and forecast_sha:
        forecast_file = Path(forecast_path)
        if not forecast_file.exists() or sha256_file(forecast_file) != forecast_sha:
            raise RuntimeError("El checksum del forecast archivado no coincide con el manifiesto.")

    models = payload.get("models", {})
    if isinstance(models, dict):
        for horizon, model in models.items():
            if not isinstance(model, dict):
                continue
            model_path = model.get("path")
            model_sha = model.get("sha256")
            if model_path and model_sha:
                model_file = Path(model_path)
                if not model_file.exists() or sha256_file(model_file) != model_sha:
                    raise RuntimeError(
                        f"El checksum del modelo T+{horizon} no coincide con el manifiesto."
                    )

    horizons = set(int(value) for value in payload.get("horizons", []))
    if horizons != {1, 2, 3}:
        raise RuntimeError(f"El manifiesto no contiene T+1/T+2/T+3: {sorted(horizons)}")
    quality = str(payload.get("forecast_quality", "unknown"))
    run_mode = str(payload.get("pipeline_run_mode", "live"))
    if run_mode == "local_state_simulation" and not allow_local_simulation:
        raise RuntimeError(
            "La última inferencia usa estado meteorológico simulado localmente; "
            "no es una ejecución de producción. Usa --allow-local-simulation "
            "solo para una comprobación local."
        )
    if quality == "stale" and not allow_stale:
        raise RuntimeError("La última inferencia usa un forecast stale.")
    if quality in {
        "fresh_fallback",
        "fresh_aemet",
        "fresh_aemet_proxy",
        "fresh_aemet_degraded",
        "incomplete",
        "invalid",
        "unavailable",
    } and fail_on_degraded:
        detail = (
            "WRF 04 km como fallback"
            if quality == "fresh_fallback"
            else "AEMET o forecast no nominal"
        )
        raise RuntimeError(f"La última inferencia usa {detail}.")
    age = float(payload.get("forecast_age_hours", float("inf")))
    if age > max_age_hours:
        raise RuntimeError(f"El forecast tiene {age:.1f} horas; máximo permitido={max_age_hours:.1f}.")
    coverage = payload.get("forecast_coverage", {})
    minimum_ratio = float(coverage.get("minimum_ratio", 0.0))
    if minimum_ratio < min_coverage:
        raise RuntimeError(
            f"Cobertura horaria insuficiente: {minimum_ratio:.1%}; "
            f"mínimo requerido={min_coverage:.1%}."
        )

    feature_contract = str(payload.get("feature_contract_version", "unknown"))
    if feature_contract in {
        CANONICAL_FEATURE_SCHEMA_VERSION,
        EGIF_48_FEATURE_CONTRACT_VERSION,
    }:
        if int(payload.get("n_cells", 29601)) != 29601:
            raise RuntimeError("El manifest canónico no declara las 29.601 celdas EGIF.")
        models = payload.get("models", {})
        if set(str(key) for key in models) != {"1", "2", "3"}:
            raise RuntimeError("El manifest canónico no contiene los tres modelos por horizonte.")
        for horizon, model in models.items():
            if model.get("feature_schema_version") != feature_contract:
                raise RuntimeError(
                    f"El modelo T+{horizon} no usa el contrato {feature_contract}."
                )
            metadata = model.get("metadata", {})
            if isinstance(metadata, dict):
                declared_columns = metadata.get("feature_columns", [])
                expected_count = 48 if feature_contract == EGIF_48_FEATURE_CONTRACT_VERSION else 50
                if len(declared_columns) != expected_count:
                    raise RuntimeError(
                        f"El modelo T+{horizon} declara {len(declared_columns)} features; "
                        f"se esperaban {expected_count}."
                    )

    features_path = payload.get("features_path")
    features_sha = payload.get("features_sha256")
    if features_path and features_sha:
        feature_file = Path(features_path)
        if not feature_file.exists() or sha256_file(feature_file) != features_sha:
            raise RuntimeError("El checksum de las features no coincide con el manifiesto.")

    frame = pd.read_parquet(output_path, columns=["horizon_days", "cell_id", "prob_risk"])
    if set(frame["horizon_days"].astype(int).unique()) != {1, 2, 3}:
        raise RuntimeError("El Parquet publicado no contiene los tres horizontes.")
    if frame["prob_risk"].isna().any() or ((frame["prob_risk"] < 0) | (frame["prob_risk"] > 1)).any():
        raise RuntimeError("El output contiene probabilidades inválidas.")

    return {
        "status": "ok",
        "output": str(output_path),
        "manifest": str(manifest),
        "forecast_quality": quality,
        "forecast_grid": payload.get("forecast_grid", "unknown"),
        "forecast_selection": payload.get("forecast_selection", {}),
        "degraded": quality in {
            "fresh_fallback",
            "fresh_aemet",
            "fresh_aemet_proxy",
            "fresh_aemet_degraded",
            "stale",
        },
        "forecast_age_hours": age,
        "minimum_coverage": minimum_ratio,
        "rows": int(len(frame)),
        "cells": int(frame["cell_id"].nunique()),
    }


def main() -> int:
    args = parse_args()
    try:
        report = check_run(
            args.output,
            manifest_path=args.manifest,
            max_age_hours=args.max_age_hours,
            min_coverage=args.min_coverage,
            allow_stale=args.allow_stale,
            fail_on_degraded=args.fail_on_degraded,
            allow_local_simulation=args.allow_local_simulation,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
