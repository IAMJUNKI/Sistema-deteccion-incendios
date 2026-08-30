#!/usr/bin/env python3
"""Backtest temporal del benchmark ERA5 para T+1/T+2/T+3.

Este script no llama a la API meteorológica ni reutiliza el artefacto de
inferencia operativa. Usa el benchmark ``era5_perfect`` (meteorología real del
día objetivo) para comprobar por separado la alineación de cada horizonte.
La evaluación del error real de MeteoGalicia empezará cuando exista un archivo
histórico de forecasts emitidos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.features.operational_features import build_historical_horizon_dataset
from src.models.forecast_risk_model import load_horizon_model


DEFAULT_COLUMNS = [
    "cell_id",
    "fecha",
    "tmax_vc",
    "rhmin_vc",
    "vmax_vc",
    "prec_dia",
    "altitud_media",
    "pendiente_media",
    "orientacion_media",
    "combustible_pct_forestal",
    "target",
]


def load_master_dataset(dataset_dir: Path, years: list[int]) -> pd.DataFrame:
    files = [dataset_dir / f"dataset_maestro_{year}.parquet" for year in years]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"No se encontraron datasets: {missing}")
    return pd.concat(
        [pd.read_parquet(path, columns=DEFAULT_COLUMNS) for path in files],
        ignore_index=True,
    )


def run_backtesting_on_real_fires(
    *,
    dataset_dir: Path = Path("misc/Dataset/Mike"),
    model_dir: Path = Path("data/models"),
    test_years: tuple[int, ...] = (2023, 2024),
) -> None:
    print("=" * 70)
    print("BACKTEST TEMPORAL DEL BENCHMARK ERA5 — T+1/T+2/T+3")
    print("=" * 70)

    years = [2019, 2020, 2021, 2022, *test_years]
    master = load_master_dataset(dataset_dir, sorted(set(years)))
    horizon_datasets = build_historical_horizon_dataset(master)

    for horizon, dataset in sorted(horizon_datasets.items()):
        model = load_horizon_model(horizon, model_dir=model_dir)
        test = dataset[pd.to_datetime(dataset["fecha"]).dt.year.isin(test_years)].copy()
        test = test.reset_index(drop=True)
        probabilities = model.predict_proba(test)[:, 1]
        y_true = test["target"].to_numpy()
        report = {
            "n_samples": len(test),
            "prevalence": float(y_true.mean()),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "brier_score": float(brier_score_loss(y_true, probabilities)),
        }
        print(
            f"T+{horizon}: n={report['n_samples']:,} | "
            f"PR-AUC={report['pr_auc']:.6f} | ROC-AUC={report['roc_auc']:.6f} | "
            f"Brier={report['brier_score']:.6f}"
        )

        fires = test[test["target"] == 1]
        if not fires.empty:
            fire_probabilities = probabilities[fires.index.to_numpy()]
            best_position = int(fire_probabilities.argmax())
            fire = fires.iloc[best_position]
            print(
                f"  Fuego de referencia: fecha={fire['fecha'].date()} | "
                f"celda={int(fire['cell_id'])} | "
                f"probabilidad={fire_probabilities[best_position] * 100:.2f}%"
            )


if __name__ == "__main__":
    run_backtesting_on_real_fires()
