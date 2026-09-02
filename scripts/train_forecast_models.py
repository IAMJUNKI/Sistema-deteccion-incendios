#!/usr/bin/env python3
"""Entrenamiento offline de los modelos de riesgo T+1/T+2/T+3.

Ejemplo:
    PYTHONPATH=. python scripts/train_forecast_models.py \
        --dataset-dir misc/Dataset/Mike \
        --output-dir data/models

El script nunca se ejecuta desde la inferencia diaria. El benchmark construido
con ERA5 se etiqueta como ``era5_perfect`` para que no se confunda con una
evaluación histórica de forecasts de MeteoGalicia. Por defecto procesa los
años de forma secuencial y conserva una muestra estratificada por año para
evitar que el histórico completo agote la memoria. ``--max-rows`` se reserva
para smoke tests porque toma las primeras filas de cada fichero.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from src.features.operational_features import (
    build_historical_features,
    build_historical_horizon_dataset,
)
from src.models.canonical_training import train_all_canonical_horizons
from src.models.forecast_risk_model import train_all_horizon_models, train_horizon_model

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


def load_master_dataset(dataset_dir: Path, years: list[int], max_rows: int | None = None) -> pd.DataFrame:
    files = [dataset_dir / f"dataset_maestro_{year}.parquet" for year in years]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"No se encontraron datasets: {missing}")
    frames = []
    for path in files:
        frame = pd.read_parquet(path, columns=DEFAULT_COLUMNS)
        frame = _compact_dtypes(frame)
        if max_rows is not None:
            frame = frame.head(max_rows)
        frames.append(frame)
        print(f"Cargado {path.name}: {len(frame):,} filas", flush=True)
    return pd.concat(frames, ignore_index=True)


def _compact_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce el consumo de memoria sin cambiar el contrato de datos."""

    compact = frame.copy()
    for column in DEFAULT_COLUMNS:
        if column in compact.columns and column not in {"cell_id", "fecha", "target"}:
            compact[column] = pd.to_numeric(compact[column], errors="coerce").astype("float32")
    if "cell_id" in compact.columns:
        compact["cell_id"] = pd.to_numeric(compact["cell_id"], errors="coerce").astype("int32")
    if "target" in compact.columns:
        compact["target"] = pd.to_numeric(compact["target"], errors="coerce").fillna(0).astype("int8")
    compact["fecha"] = pd.to_datetime(compact["fecha"], errors="coerce")
    return compact


def _bounded_stratified_sample(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Conserva positivos y toma negativos aleatorios hasta un límite por año."""

    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    positives = frame[frame["target"] == 1]
    negatives = frame[frame["target"] == 0]
    if len(positives) >= max_rows:
        return positives.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    negative_count = min(len(negatives), max_rows - len(positives))
    sampled_negatives = negatives.sample(n=negative_count, random_state=random_state)
    return pd.concat([positives, sampled_negatives], ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)


def load_sampled_historical_features(
    dataset_dir: Path,
    years: list[int],
    *,
    rows_per_year: int,
) -> pd.DataFrame:
    """Construye features año a año para evitar cargar todos los años en RAM.

    Las ventanas de 3/7/14/30 días se calculan sobre el año completo y un
    solapamiento de 30 días del año anterior. El muestreo se aplica después de
    calcularlas, por lo que no altera las memorias meteorológicas. Las métricas
    finales corresponden a la muestra conservada, no a todo el dataset.
    """

    if rows_per_year <= 0:
        raise ValueError("rows_per_year debe ser un entero positivo.")

    sampled_years: list[pd.DataFrame] = []
    carry: pd.DataFrame | None = None
    for year in years:
        path = dataset_dir / f"dataset_maestro_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el dataset: {path}")

        frame = _compact_dtypes(pd.read_parquet(path, columns=DEFAULT_COLUMNS))
        print(f"Procesando {path.name}: {len(frame):,} filas", flush=True)
        combined = pd.concat([carry, frame], ignore_index=True) if carry is not None else frame
        features = build_historical_features(combined)
        year_features = features[features["fecha"].dt.year == year].copy()
        sampled = _bounded_stratified_sample(
            year_features,
            max_rows=rows_per_year,
            random_state=42 + year,
        )
        sampled_years.append(sampled)
        print(
            f"  Conservadas {len(sampled):,} filas ({int(sampled['target'].sum()):,} positivos)",
            flush=True,
        )

        last_date = frame["fecha"].max()
        carry = frame[frame["fecha"] >= last_date - pd.Timedelta(days=30)].copy()
        del combined, features, year_features, sampled, frame
        gc.collect()

    return pd.concat(sampled_years, ignore_index=True)


def train_sampled_models(
    dataset: pd.DataFrame,
    *,
    output_dir: Path,
    negative_ratio: int,
) -> dict[int, dict[str, object]]:
    """Entrena T+1/T+2/T+3 secuencialmente y libera cada modelo intermedio."""

    reports: dict[int, dict[str, object]] = {}
    for horizon in (1, 2, 3):
        horizon_dataset = dataset.copy()
        horizon_dataset["horizon_days"] = horizon
        _, report = train_horizon_model(
            horizon_dataset,
            horizon_days=horizon,
            output_dir=output_dir,
            negative_ratio=negative_ratio,
        )
        reports[horizon] = report
        del horizon_dataset
        gc.collect()
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("misc/Dataset/Mike"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023, 2024])
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=None,
        help="Negativos por positivo: 100 en EGIF y 50 en el modelo legacy si no se especifica.",
    )
    parser.add_argument(
        "--include-xgboost-challenger",
        action="store_true",
        help="En datasets EGIF, entrena XGBoost fuera del artefacto operativo.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Límite por año para smoke tests")
    parser.add_argument(
        "--rows-per-year",
        type=int,
        default=300_000,
        help="Filas conservadas por año tras calcular features; evita cargar todo el histórico en RAM",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # El datacubo EGIF se reconoce por su metadata y usa el contrato canónico
    # de 50 predictores. Se mantiene el camino antiguo para reproducir el
    # modelo de 23 variables y disponer de rollback.
    if (args.dataset_dir / "metadata.json").exists():
        available_years = [
            year
            for year in args.years
            if any(
                (args.dataset_dir / candidate).exists()
                for candidate in (
                    f"year={year}/dataset_{year}.parquet",
                    f"dataset_{year}.parquet",
                    f"dataset_maestro_{year}.parquet",
                )
            )
        ]
        train_years = tuple(year for year in available_years if year <= 2021)
        validation_year = 2022 if 2022 in available_years else None
        test_years = tuple(year for year in available_years if year == 2023)
        if not train_years or validation_year is None or not test_years:
            raise FileNotFoundError(
                "El datacubo EGIF necesita particiones 2019-2021 para train, "
                "2022 para validación y 2023 para test."
            )
        reports = train_all_canonical_horizons(
            args.dataset_dir,
            output_dir=args.output_dir,
            train_years=train_years,
            validation_year=validation_year,
            test_years=test_years,
            negative_ratio=args.negative_ratio or 100,
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
        return
    if args.max_rows is not None:
        master = load_master_dataset(args.dataset_dir, args.years, args.max_rows)
        horizon_datasets = build_historical_horizon_dataset(master)
        reports = train_all_horizon_models(
            horizon_datasets,
            output_dir=args.output_dir,
            negative_ratio=args.negative_ratio or 50,
        )
    else:
        sampled_features = load_sampled_historical_features(
            args.dataset_dir,
            args.years,
            rows_per_year=args.rows_per_year,
        )
        reports = train_sampled_models(
            sampled_features,
            output_dir=args.output_dir,
            negative_ratio=args.negative_ratio or 50,
        )
        del sampled_features
        gc.collect()
    print("\nModelos serializados:")
    for horizon, report in reports.items():
        validation = report["validation"]
        test = report["test"]
        print(
            f"T+{horizon}: validation PR-AUC={validation['pr_auc']:.6f} | "
            f"test PR-AUC={test['pr_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
