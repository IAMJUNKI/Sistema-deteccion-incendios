#!/usr/bin/env python3
"""Valida y publica atómicamente una familia completa de modelos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from src.features.canonical_contract import (
    EGIF_48_FEATURE_CONTRACT_VERSION,
    load_feature_columns_for_contract,
    validate_feature_contract,
)
from src.features.operational_benchmark import (
    OPERATIONAL_ALIGNMENT_VERSION,
    OPERATIONAL_TEMPORAL_SEMANTICS,
    validate_operational_dataset,
)
from src.models.forecast_risk_model import ForecastRiskModel
from src.operational.artifacts import atomic_write_json, sha256_file

HORIZONS = (1, 2, 3)


def validate_model_family(
    source_dir: str | Path,
    *,
    prefix: str,
    dataset_dir: str | Path,
    allow_provisional: bool = False,
) -> dict[str, object]:
    """Valida contrato, metadata, hashes y coherencia de los tres artefactos."""

    source = Path(source_dir)
    dataset_metadata = validate_operational_dataset(
        dataset_dir,
        require_contract=EGIF_48_FEATURE_CONTRACT_VERSION,
    )
    quality = dataset_metadata.get("static_layer_quality")
    if quality != "validated" and not allow_provisional:
        raise ValueError(
            "La promoción está bloqueada: las capas estáticas no están validadas "
            f"(quality={quality!r}). Usa --allow-provisional solo para una prueba local."
        )
    expected = load_feature_columns_for_contract(
        EGIF_48_FEATURE_CONTRACT_VERSION, dataset_dir
    )
    expected_dataset_version = Path(dataset_dir).name
    artifacts = []
    dataset_hash = None
    for horizon in HORIZONS:
        model_path = source / f"{prefix}_t{horizon}.joblib"
        json_path = source / f"{prefix}_t{horizon}.json"
        if not model_path.exists() or not json_path.exists():
            raise FileNotFoundError(
                f"La familia está incompleta para T+{horizon}: {model_path} / {json_path}"
            )
        model = joblib.load(model_path)
        if not isinstance(model, ForecastRiskModel):
            raise TypeError(f"{model_path} no contiene ForecastRiskModel.")
        if model.horizon_days != horizon:
            raise ValueError(f"{model_path} no declara el horizonte T+{horizon}.")
        if model.feature_schema_version != EGIF_48_FEATURE_CONTRACT_VERSION:
            raise ValueError(f"{model_path} no pertenece al contrato de 48 variables.")
        validate_feature_contract(model.feature_columns, EGIF_48_FEATURE_CONTRACT_VERSION)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("horizon_days") != horizon:
            raise ValueError(f"{json_path} no declara el horizonte T+{horizon}.")
        if metadata.get("feature_contract_version") != EGIF_48_FEATURE_CONTRACT_VERSION:
            raise ValueError(f"{json_path} declara un contrato distinto.")
        if metadata.get("dataset_version") != expected_dataset_version:
            raise ValueError(
                f"{json_path} fue entrenado con dataset_version="
                f"{metadata.get('dataset_version')!r}; se esperaba {expected_dataset_version!r}."
            )
        if metadata.get("alignment_version") != OPERATIONAL_ALIGNMENT_VERSION:
            raise ValueError(f"{json_path} no declara la alineación operativa requerida.")
        if metadata.get("training_temporal_semantics") != OPERATIONAL_TEMPORAL_SEMANTICS:
            raise ValueError(f"{json_path} no declara la semántica temporal requerida.")
        if metadata.get("feature_columns") != expected:
            raise ValueError(f"{json_path} no conserva el orden oficial de las features.")
        if metadata.get("static_layer_quality") != quality:
            raise ValueError(
                f"{json_path} declara una calidad estática distinta a la del dataset."
            )
        if model.metadata.get("dataset_version") != metadata.get("dataset_version"):
            raise ValueError(f"{model_path} y {json_path} no comparten dataset_version.")
        if model.metadata.get("dataset_hash") != metadata.get("dataset_hash"):
            raise ValueError(f"{model_path} y {json_path} no comparten dataset_hash.")
        current_hash = metadata.get("dataset_hash")
        if not current_hash:
            raise ValueError(f"{json_path} no contiene dataset_hash.")
        if dataset_hash is None:
            dataset_hash = current_hash
        elif dataset_hash != current_hash:
            raise ValueError("Los tres modelos se entrenaron con datasets diferentes.")
        artifacts.extend(
            [
                {
                    "horizon_days": horizon,
                    "joblib": str(model_path),
                    "joblib_sha256": sha256_file(model_path),
                    "metadata": str(json_path),
                    "metadata_sha256": sha256_file(json_path),
                    "model_version": metadata.get("model_version"),
                }
            ]
        )
    return {
        "family": str(prefix),
        "source_dir": str(source),
        "dataset_dir": str(dataset_dir),
        "dataset_hash": dataset_hash,
        "feature_contract_version": EGIF_48_FEATURE_CONTRACT_VERSION,
        "alignment_version": OPERATIONAL_ALIGNMENT_VERSION,
        "static_layer_quality": quality,
        "artifacts": artifacts,
    }


def promote_model_family(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    prefix: str = "forecast_risk_egif_48",
    dataset_dir: str | Path = "data/processed/tabular/egif_operational",
    allow_provisional: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Publica los tres modelos validados en el directorio operativo."""

    report = validate_model_family(
        source_dir,
        prefix=prefix,
        dataset_dir=dataset_dir,
        allow_provisional=allow_provisional,
    )
    destination = Path(destination_dir)
    report["destination_dir"] = str(destination)
    report["promoted_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    report["dry_run"] = dry_run
    if dry_run:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    for artifact in report["artifacts"]:
        for key in ("joblib", "metadata"):
            source_path = Path(artifact[key])
            target_path = destination / source_path.name
            temporary = destination / f".{target_path.name}.{uuid4().hex}.tmp"
            try:
                shutil.copy2(source_path, temporary)
                os.replace(temporary, target_path)
            finally:
                temporary.unlink(missing_ok=True)
    manifest_path = destination / "active_model_manifest.json"
    atomic_write_json(report, manifest_path)
    report["manifest"] = str(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/tabular/egif_operational"))
    parser.add_argument("--prefix", default="forecast_risk_egif_48")
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = promote_model_family(
        args.source_dir,
        args.destination_dir,
        prefix=args.prefix,
        dataset_dir=args.dataset_dir,
        allow_provisional=args.allow_provisional,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
