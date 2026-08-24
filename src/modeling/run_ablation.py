"""Ejecuta una réplica de la ablación en un proceso aislado."""

import argparse
import json
from pathlib import Path

from src.modeling.data import (
    TRAIN_YEARS,
    VALIDATION_YEARS,
    load_dataset_contract,
    sample_years_for_training,
)
from src.modeling.experiments import fit_lightgbm_experiment
from src.modeling.features import resolve_feature_set


def main() -> None:
    """Escribe resultados de una réplica y termina para liberar memoria."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--remainder", type=int, required=True)
    parser.add_argument("--modulus", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/modeling"))
    args = parser.parse_args()

    contract = load_dataset_contract()
    train = sample_years_for_training(
        contract, TRAIN_YEARS, contract.predictors, args.modulus, args.remainder
    )
    validation = sample_years_for_training(
        contract, VALIDATION_YEARS, contract.predictors, args.modulus, args.remainder
    )
    results = []
    for name in ("completo", "temporal_compacto"):
        predictors = resolve_feature_set(contract.predictors, name)
        _, metrics = fit_lightgbm_experiment(train, validation, predictors)
        results.append({"remainder": args.remainder, "feature_set": name, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"robust_remainder_{args.remainder}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
