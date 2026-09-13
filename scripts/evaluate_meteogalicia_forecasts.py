#!/usr/bin/env python3
"""Evalúa forecasts MeteoGalicia archivados frente a observaciones posteriores.

El script no recalcula predicciones de riesgo ni modifica el estado meteorológico.
Agrega cada forecast horario con la misma semántica diaria que usa la inferencia,
lo empareja por ``cell_id`` y fecha con observaciones ya cerradas y calcula una
evaluación preliminar por horizonte.

El día de emisión se obtiene de ``downloaded_at`` del forecast archivado. Esto
representa el día en el que el forecast estuvo disponible para el pipeline. Si
ese campo no existe, se usa ``forecast_run_at`` como fallback y se documenta en
el informe.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.operational_features import aggregate_hourly_forecast
from src.operational.artifacts import atomic_write_json

GALICIA_TZ = ZoneInfo("Europe/Madrid")
HORIZONS = (1, 2, 3)
WEATHER_VARIABLES = ("tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia")
OBSERVATION_EXCLUDED_SOURCES = {"forecast_proxy"}
RAIN_THRESHOLD_MM = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=Path("data/raw/meteogalicia"),
        help="Directorio con forecast_*.parquet archivados.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/processed/state/weather_daily_state.parquet"),
        help="Estado diario con observaciones interpoladas a la rejilla.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/evaluation/meteogalicia"),
        help="Directorio de salida de los informes.",
    )
    parser.add_argument(
        "--pattern",
        default="forecast_*_1km.parquet",
        help="Patrón de forecasts a evaluar; por defecto solo WRF 1 km.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=1,
        help="Mínimo de celdas emparejadas para contar un caso.",
    )
    return parser.parse_args()


def _as_local_date(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(GALICIA_TZ).normalize().tz_localize(None)


def _issue_date(forecast: pd.DataFrame) -> tuple[pd.Timestamp, str]:
    for column in ("downloaded_at", "issue_time", "forecast_run_at"):
        if column not in forecast.columns:
            continue
        values = pd.to_datetime(forecast[column], errors="coerce", utc=True).dropna()
        if not values.empty:
            source = "downloaded_at" if column == "downloaded_at" else column
            return _as_local_date(values.min()), source
    raise ValueError("El forecast no contiene downloaded_at, issue_time ni forecast_run_at válidos.")


def _load_observations(path: Path) -> pd.DataFrame:
    required = ["cell_id", "fecha", "source", *WEATHER_VARIABLES]
    state = pd.read_parquet(path, columns=required)
    state["fecha"] = pd.to_datetime(state["fecha"], errors="coerce")
    if getattr(state["fecha"].dt, "tz", None) is not None:
        state["fecha"] = state["fecha"].dt.tz_convert(GALICIA_TZ).dt.tz_localize(None)
    else:
        state["fecha"] = state["fecha"].dt.tz_localize(None)
    state = state.dropna(subset=["cell_id", "fecha"])
    source = state["source"].astype("string")
    state = state[~source.isin(OBSERVATION_EXCLUDED_SOURCES)].copy()
    return state.drop_duplicates(["cell_id", "fecha"], keep="last")


def _metric_row(
    horizon: int,
    variable: str,
    errors: np.ndarray,
    *,
    forecast_count: int,
) -> dict[str, Any]:
    if errors.size == 0:
        return {
            "horizon_days": horizon,
            "variable": variable,
            "n_pairs": 0,
            "forecast_files": forecast_count,
            "mae": None,
            "rmse": None,
            "bias": None,
        }
    return {
        "horizon_days": horizon,
        "variable": variable,
        "n_pairs": int(errors.size),
        "forecast_files": forecast_count,
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(errors.mean()),
    }


def evaluate_forecasts(
    forecast_dir: str | Path,
    state_path: str | Path,
    *,
    pattern: str = "forecast_*_1km.parquet",
    min_cells: int = 1,
) -> dict[str, Any]:
    """Calcula métricas preliminares por variable y horizonte.

    Args:
        forecast_dir: Directorio de Parquet horarios archivados.
        state_path: Estado diario con observaciones posteriores.
        pattern: Patrón de archivos de forecast que se evaluarán.
        min_cells: Celdas mínimas para considerar una comparación válida.

    Returns:
        Informe serializable con casos comparables, métricas y advertencias.
    """
    forecast_root = Path(forecast_dir)
    state_file = Path(state_path)
    if not forecast_root.exists():
        raise FileNotFoundError(f"No existe el directorio de forecasts: {forecast_root}")
    if not state_file.exists():
        raise FileNotFoundError(f"No existe el estado meteorológico: {state_file}")
    if min_cells < 1:
        raise ValueError("min_cells debe ser al menos 1.")

    observations = _load_observations(state_file)
    files = sorted(forecast_root.glob(pattern), key=lambda path: path.stat().st_mtime)
    errors: defaultdict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    rain_hits: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    cases: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    issue_sources: defaultdict[str, int] = defaultdict(int)

    for path in files:
        try:
            forecast = pd.read_parquet(path)
            issue_date, issue_source = _issue_date(forecast)
            issue_sources[issue_source] += 1
            daily = aggregate_hourly_forecast(forecast)
            if daily.empty:
                skipped.append({"file": path.name, "reason": "forecast_daily_empty"})
                continue

            for horizon in HORIZONS:
                target_date = issue_date + pd.Timedelta(days=horizon)
                forecast_day = daily[daily["fecha"] == target_date].copy()
                observed_day = observations[observations["fecha"] == target_date].copy()
                if forecast_day.empty or observed_day.empty:
                    continue

                joined = forecast_day.merge(
                    observed_day,
                    on=["cell_id", "fecha"],
                    suffixes=("_forecast", "_observed"),
                )
                if len(joined) < min_cells:
                    continue

                cases.append(
                    {
                        "forecast_file": path.name,
                        "issue_date": issue_date.date().isoformat(),
                        "issue_date_source": issue_source,
                        "target_date": target_date.date().isoformat(),
                        "horizon_days": horizon,
                        "cells": int(joined["cell_id"].nunique()),
                    }
                )

                for variable in WEATHER_VARIABLES:
                    forecast_values = pd.to_numeric(
                        joined[f"{variable}_forecast"], errors="coerce"
                    ).to_numpy(dtype=float)
                    observed_values = pd.to_numeric(
                        joined[f"{variable}_observed"], errors="coerce"
                    ).to_numpy(dtype=float)
                    valid = np.isfinite(forecast_values) & np.isfinite(observed_values)
                    if valid.any():
                        errors[(horizon, variable)].append(
                            forecast_values[valid] - observed_values[valid]
                        )

                forecast_rain = pd.to_numeric(
                    joined["prec_dia_forecast"], errors="coerce"
                ).to_numpy(dtype=float)
                observed_rain = pd.to_numeric(
                    joined["prec_dia_observed"], errors="coerce"
                ).to_numpy(dtype=float)
                valid_rain = np.isfinite(forecast_rain) & np.isfinite(observed_rain)
                if valid_rain.any():
                    rain_hits[horizon].append(
                        (
                            (forecast_rain[valid_rain] >= RAIN_THRESHOLD_MM)
                            == (observed_rain[valid_rain] >= RAIN_THRESHOLD_MM)
                        )
                    )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({"file": path.name, "reason": str(exc)})

    metrics: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        forecast_count = sum(
            1 for case in cases if case["horizon_days"] == horizon
        )
        for variable in WEATHER_VARIABLES:
            chunks = errors[(horizon, variable)]
            combined = np.concatenate(chunks) if chunks else np.array([], dtype=float)
            metrics.append(
                _metric_row(
                    horizon,
                    variable,
                    combined,
                    forecast_count=forecast_count,
                )
            )

        rain_chunks = rain_hits[horizon]
        rain_combined = np.concatenate(rain_chunks) if rain_chunks else np.array([], dtype=bool)
        metrics.append(
            {
                "horizon_days": horizon,
                "variable": "rain_event_accuracy",
                "threshold_mm": RAIN_THRESHOLD_MM,
                "n_pairs": int(rain_combined.size),
                "forecast_files": forecast_count,
                "accuracy": (
                    float(rain_combined.mean()) if rain_combined.size else None
                ),
            }
        )

    state_dates = observations["fecha"].dropna()
    return {
        "generated_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "evaluation_type": "preliminary_forecast_observation_case_study",
        "forecast_dir": str(forecast_root),
        "state_path": str(state_file),
        "pattern": pattern,
        "forecast_files_found": len(files),
        "comparison_cases": len(cases),
        "observation_rows_used": int(len(observations)),
        "observation_date_range": (
            [state_dates.min().date().isoformat(), state_dates.max().date().isoformat()]
            if not state_dates.empty
            else []
        ),
        "issue_date_sources": dict(issue_sources),
        "excluded_observation_sources": sorted(OBSERVATION_EXCLUDED_SOURCES),
        "weather_variables": list(WEATHER_VARIABLES),
        "metrics": metrics,
        "cases": cases,
        "skipped": skipped,
        "interpretation": (
            "Caso de estudio preliminar; no representa todavía una validación estadística "
            "de toda la temporada ni la precisión del modelo de riesgo."
        ),
    }


def _write_csv(report: dict[str, Any], destination: Path) -> None:
    metrics = pd.DataFrame(report["metrics"])
    temporary = destination.with_name(f".{destination.name}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    report = evaluate_forecasts(
        args.forecast_dir,
        args.state,
        pattern=args.pattern,
        min_cells=args.min_cells,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "meteogalicia_forecast_metrics.json"
    csv_path = args.output_dir / "meteogalicia_forecast_metrics.csv"
    atomic_write_json(report, json_path)
    _write_csv(report, csv_path)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "forecast_files_found": report["forecast_files_found"],
        "comparison_cases": report["comparison_cases"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
