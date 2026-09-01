#!/usr/bin/env python3
"""Entrena los tres modelos operativos del datacubo EGIF.

Ejemplo:
    PYTHONPATH=. python scripts/train_egif_operational.py \
        --dataset-dir /ruta/al/egif \
        --output-dir data/models

El script lee por lotes, submuestrea solo negativos durante el ajuste y
evalúa sobre 2022/2023 completos. El resultado son tres artefactos
``forecast_risk_egif_t{1,2,3}.joblib``; la inferencia diaria los carga, nunca
los vuelve a entrenar.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.models.canonical_training import train_all_canonical_horizons


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--negative-ratio", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--include-xgboost-challenger",
        action="store_true",
        help="Entrena XGBoost como challenger y lo guarda fuera del artefacto operativo.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    reports = train_all_canonical_horizons(
        args.dataset_dir,
        output_dir=args.output_dir,
        negative_ratio=args.negative_ratio,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        bootstrap_samples=args.bootstrap_samples,
        include_challenger=args.include_xgboost_challenger,
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
