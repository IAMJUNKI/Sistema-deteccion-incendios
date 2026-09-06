#!/usr/bin/env python3
"""Entrena los tres modelos operativos del datacubo EGIF.

Ejemplo:
    PYTHONPATH=. python scripts/train_egif_operational.py \
        --dataset-dir /ruta/al/egif \
        --output-dir data/models

El script lee por lotes, submuestrea solo negativos durante el ajuste y
evalúa sobre población completa. Por defecto genera la familia operativa 48:
``forecast_risk_egif_48_t{1,2,3}.joblib``. La familia histórica de 50 se puede
solicitar explícitamente y se conserva como rollback.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.features.canonical_contract import (
    CANONICAL_FEATURE_SCHEMA_VERSION,
    EGIF_48_FEATURE_CONTRACT_VERSION,
)
from src.models.canonical_training import (
    train_all_aligned50_horizons,
    train_all_canonical_horizons,
    train_all_egif48_horizons,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--negative-ratio", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--feature-contract-version",
        choices=(EGIF_48_FEATURE_CONTRACT_VERSION, CANONICAL_FEATURE_SCHEMA_VERSION),
        default=EGIF_48_FEATURE_CONTRACT_VERSION,
        help="Contrato que se entrenará. Por defecto, la familia operativa de 48 variables.",
    )
    parser.add_argument(
        "--train-years",
        default="2019-2020",
        help="Años de entrenamiento, por ejemplo 2016-2020 o 2016,2017,2018.",
    )
    parser.add_argument("--calibration-year", type=int, default=2021)
    parser.add_argument("--validation-year", type=int, default=2022)
    parser.add_argument("--test-years", default="2023")
    parser.add_argument(
        "--experiment-name",
        default="production",
        help=(
            "Nombre del experimento. 'comparable' y 'expanded' se guardan bajo "
            "output-dir/experiments/<nombre> para no sobrescribirse; production "
            "mantiene los nombres operativos en output-dir."
        ),
    )
    parser.add_argument(
        "--include-xgboost-challenger",
        action="store_true",
        help="Entrena XGBoost como challenger y lo guarda fuera del artefacto operativo.",
    )
    parser.add_argument(
        "--aligned-50-control",
        action="store_true",
        help=(
            "Entrena el control canónico de 50 variables con memoria T-1; "
            "no se publica automáticamente."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _parse_years(value: str) -> tuple[int, ...]:
    """Acepta ``2016-2020`` o una lista separada por comas."""

    value = value.strip()
    if "-" in value:
        start, end = (int(item.strip()) for item in value.split("-", 1))
        if start > end:
            raise ValueError("El intervalo de años debe estar ordenado de menor a mayor.")
        return tuple(range(start, end + 1))
    years = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not years:
        raise ValueError("Debe indicarse al menos un año.")
    return years


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    train_years = _parse_years(args.train_years)
    test_years = _parse_years(args.test_years)
    common = dict(
        negative_ratio=args.negative_ratio,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        bootstrap_samples=args.bootstrap_samples,
        train_years=train_years,
        calibration_year=args.calibration_year,
        validation_year=args.validation_year,
        test_years=test_years,
        experiment_name=args.experiment_name,
    )
    output_dir = args.output_dir
    if args.experiment_name in {"comparable", "expanded"}:
        output_dir = output_dir / "experiments" / args.experiment_name
    if args.aligned_50_control:
        if args.feature_contract_version != CANONICAL_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "--aligned-50-control requiere --feature-contract-version egif-2d-v1."
            )
        experiment_dir = (
            "aligned50" if args.experiment_name == "production" else args.experiment_name
        )
        output_dir = output_dir / "experiments" / experiment_dir
        train_function = train_all_aligned50_horizons
    elif args.feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION:
        train_function = train_all_egif48_horizons
    else:
        train_function = train_all_canonical_horizons
    reports = train_function(
        args.dataset_dir,
        output_dir=output_dir,
        include_challenger=args.include_xgboost_challenger,
        **common,
    )
    print("\nModelos EGIF serializados:")
    for horizon, report in reports.items():
        validation = report["validation"]
        test = report["test"]
        print(
            f"T+{horizon}: validation PR-AUC={validation['pr_auc']:.6f} | "
            f"test PR-AUC={test['pr_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
