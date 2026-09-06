#!/usr/bin/env python3
"""Genera observaciones diarias en rejilla y actualiza el estado operativo.

El script recupera un rango histórico de climatología diaria AEMET, lo limita a
Galicia, interpola las estaciones a la rejilla de 1 km y publica:

* ``data/processed/observations/weather_daily_latest.parquet``
* ``data/processed/state/weather_daily_state.parquet``

Por defecto descarga los 30 días completos disponibles hasta ``hoy - 4 días``.
Ese desfase evita pedir días que AEMET todavía no ha publicado en la
climatología validada. La fuente es observada/reconstruida desde estaciones
AEMET y no debe confundirse con el forecast municipal utilizado por la
inferencia futura.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.aemet_observations import (
    AemetObservationError,
    build_aemet_client_from_env,
    interpolate_aemet_daily_to_grid,
)
from src.ingestion.weather_state import merge_weather_state, save_weather_state
from src.operational.artifacts import RunLockError, atomic_write_parquet, run_lock

GALICIA_TZ = "Europe/Madrid"
DEFAULT_GRID = Path("data/processed/grid/galicia_grid_1km_egif.parquet")
DEFAULT_OBSERVATIONS = Path("data/processed/observations/weather_daily_latest.parquet")
DEFAULT_STATE = Path("data/processed/state/weather_daily_state.parquet")
DEFAULT_RAW = Path("data/raw/aemet/observations")
DEFAULT_LOCK = Path("data/processed/.aemet_ingest.lock")


def _date(value: str | None, *, default: date) -> date:
    return pd.Timestamp(value).date() if value else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path(os.getenv("GRID_PATH", str(DEFAULT_GRID))))
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(os.getenv("WEATHER_OBSERVATIONS_PATH", str(DEFAULT_OBSERVATIONS))),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(os.getenv("WEATHER_STATE_PATH", str(DEFAULT_STATE))),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.getenv("AEMET_OBSERVATIONS_RAW_DIR", str(DEFAULT_RAW))),
    )
    parser.add_argument("--start-date", default=None, help="Primer día completo YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Último día publicado YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="Días a recuperar si no se da start-date")
    parser.add_argument(
        "--publication-lag-days",
        type=int,
        default=int(os.getenv("AEMET_DAILY_PUBLICATION_LAG_DAYS", "4")),
        help="Desfase de publicación de la climatología diaria (por defecto: 4).",
    )
    parser.add_argument("--retention-days", type=int, default=60)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(
            os.getenv(
                "WEATHER_STATE_INGEST_LOCK_PATH",
                os.getenv("AEMET_INGEST_LOCK_PATH", str(DEFAULT_LOCK)),
            )
        ),
    )
    parser.add_argument("--no-state", action="store_true", help="Solo publica observaciones, no actualiza el estado")
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    if args.days <= 0:
        raise ValueError("--days debe ser positivo.")
    if args.publication_lag_days < 0:
        raise ValueError("--publication-lag-days no puede ser negativo.")
    if not args.grid.exists():
        raise FileNotFoundError(f"No existe la rejilla: {args.grid}")

    available_until = (
        pd.Timestamp.now(tz=GALICIA_TZ) - pd.Timedelta(days=args.publication_lag_days)
    ).date()
    end_date = _date(args.end_date, default=available_until)
    start_date = (
        _date(args.start_date, default=end_date - timedelta(days=args.days - 1))
        if args.start_date
        else end_date - timedelta(days=args.days - 1)
    )
    if start_date > end_date:
        raise ValueError("--start-date no puede ser posterior a --end-date.")

    grid = pd.read_parquet(args.grid, columns=["cell_id", "lat_centroid", "lon_centroid"])
    client = build_aemet_client_from_env()
    station_daily = client.fetch_daily_history(
        start_date,
        end_date,
        raw_dir=args.raw_dir,
    )
    grid_daily = interpolate_aemet_daily_to_grid(
        station_daily,
        grid,
        neighbors=int(os.getenv("AEMET_OBSERVATION_IDW_NEIGHBORS", "4")),
    )
    if args.observations.exists():
        existing_observations = pd.read_parquet(args.observations)
        observations = pd.concat([existing_observations, grid_daily], ignore_index=True, sort=False)
        observations["fecha"] = pd.to_datetime(observations["fecha"], errors="coerce").dt.tz_localize(None)
        observations = (
            observations.dropna(subset=["cell_id", "fecha"])
            .drop_duplicates(["cell_id", "fecha"], keep="last")
            .sort_values(["fecha", "cell_id"])
            .reset_index(drop=True)
        )
    else:
        observations = grid_daily
    atomic_write_parquet(observations, args.observations)
    print(
        f"Observaciones guardadas en {args.observations}: "
        f"{len(observations):,} filas, {observations['cell_id'].nunique():,} celdas, "
        f"{observations['fecha'].min().date()} → {observations['fecha'].max().date()}"
    )

    if args.no_state:
        return
    existing = pd.read_parquet(args.state) if args.state.exists() else None
    state_as_of = pd.Timestamp.now(tz=GALICIA_TZ)
    incoming = grid_daily.copy()
    incoming["state_as_of"] = state_as_of
    state = merge_weather_state(
        existing,
        incoming,
        as_of=state_as_of,
        retention_days=args.retention_days,
    )
    save_weather_state(state, args.state)
    print(
        f"Estado guardado en {args.state}: {len(state):,} filas, "
        f"{state['cell_id'].nunique():,} celdas, "
        f"{state['fecha'].min()} → {state['fecha'].max()}"
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    with run_lock(args.lock):
        _run(args)


if __name__ == "__main__":
    try:
        main()
    except (AemetObservationError, RunLockError) as exc:
        raise SystemExit(f"Error de ingesta AEMET: {exc}") from exc
