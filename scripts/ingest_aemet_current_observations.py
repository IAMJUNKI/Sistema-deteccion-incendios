#!/usr/bin/env python3
"""Acumula observaciones horarias AEMET y cierra días completos en la rejilla.

AEMET conserva en ``observacion/convencional/todas`` únicamente una ventana
móvil de observaciones recientes. Este comando debe ejecutarse varias veces al
día (o en modo ``--loop``) para no perder horas. Cada captura se archiva en
Parquet y solo se publica un día en el estado operativo cuando las estaciones
han alcanzado la cobertura mínima configurada.

Ejemplos locales::

    PYTHONPATH=. python scripts/ingest_aemet_current_observations.py --once
    PYTHONPATH=. python scripts/ingest_aemet_current_observations.py --loop --interval-hours 6

El modo ``--loop`` está pensado para dejar una terminal abierta en el
ordenador del proyecto. ``Ctrl+C`` lo detiene limpiamente.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.aemet_observations import (
    AemetObservationError,
    aggregate_aemet_hourly_to_daily,
    build_aemet_client_from_env,
    interpolate_aemet_daily_to_grid,
)
from src.ingestion.weather_state import merge_weather_state, save_weather_state
from src.operational.artifacts import RunLockError, atomic_write_parquet, run_lock

DEFAULT_GRID = Path("data/processed/grid/galicia_grid_1km_egif.parquet")
DEFAULT_HOURLY = Path("data/processed/observations/aemet_hourly_observations.parquet")
DEFAULT_DAILY = Path("data/processed/observations/weather_daily_latest.parquet")
DEFAULT_STATE = Path("data/processed/state/weather_daily_state.parquet")
DEFAULT_RAW = Path("data/raw/aemet/observations")
DEFAULT_LOCK = Path("data/processed/.aemet_ingest.lock")
LOCAL_TZ = "Europe/Madrid"


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise AemetObservationError(f"No se pudo leer el Parquet existente {path}.") from exc


def _merge_hourly(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    *,
    now: pd.Timestamp,
    retention_hours: int,
) -> pd.DataFrame:
    frames = [frame for frame in (existing, incoming) if frame is not None and not frame.empty]
    if not frames:
        return incoming.copy()
    merged = pd.concat(frames, ignore_index=True)
    merged["valid_time"] = pd.to_datetime(merged["valid_time"], errors="coerce", utc=True)
    cutoff = now.tz_convert("UTC") - pd.Timedelta(hours=retention_hours)
    merged = merged[merged["valid_time"] >= cutoff]
    return (
        merged.dropna(subset=["station_id", "valid_time"])
        .drop_duplicates(["station_id", "valid_time"], keep="last")
        .sort_values(["valid_time", "station_id"])
        .reset_index(drop=True)
    )


def _merge_daily(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, incoming) if frame is not None and not frame.empty]
    if not frames:
        return incoming.copy()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["fecha"] = pd.to_datetime(merged["fecha"], errors="coerce").dt.tz_localize(None)
    return (
        merged.dropna(subset=["cell_id", "fecha"])
        .drop_duplicates(["cell_id", "fecha"], keep="last")
        .sort_values(["fecha", "cell_id"])
        .reset_index(drop=True)
    )


def ingest_once(args: argparse.Namespace) -> dict[str, object]:
    """Ejecuta una captura y devuelve un resumen apto para logs."""

    for path in (args.grid, args.hourly.parent, args.daily.parent, args.state.parent, args.raw_dir):
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    if not args.grid.exists():
        raise FileNotFoundError(f"No existe la rejilla: {args.grid}")

    now = pd.Timestamp.now(tz=LOCAL_TZ)
    client = build_aemet_client_from_env()
    incoming = client.fetch_current_observations(
        raw_dir=args.raw_dir,
        downloaded_at=now,
    )
    existing_hourly = _read_optional(args.hourly)
    hourly = _merge_hourly(
        existing_hourly,
        incoming,
        now=now,
        retention_hours=args.hourly_retention_hours,
    )
    atomic_write_parquet(hourly, args.hourly)

    result: dict[str, object] = {
        "captured_rows": int(len(incoming)),
        "hourly_rows": int(len(hourly)),
        "hourly_path": str(args.hourly),
        "closed_days": 0,
        "state_updated": False,
    }
    try:
        station_daily = aggregate_aemet_hourly_to_daily(
            hourly,
            min_coverage_hours=args.min_coverage_hours,
        )
    except AemetObservationError as exc:
        result["message"] = str(exc)
        return result

    grid = pd.read_parquet(args.grid, columns=["cell_id", "lat_centroid", "lon_centroid"])
    grid_daily = interpolate_aemet_daily_to_grid(
        station_daily,
        grid,
        neighbors=args.idw_neighbors,
        source="aemet_current_observation_idw",
    )
    grid_daily["state_as_of"] = now
    existing_daily = _read_optional(args.daily)
    daily = _merge_daily(existing_daily, grid_daily)
    atomic_write_parquet(daily, args.daily)

    existing_state = _read_optional(args.state)
    state = merge_weather_state(
        existing_state,
        grid_daily,
        as_of=now,
        retention_days=args.retention_days,
    )
    save_weather_state(state, args.state)
    result.update(
        {
            "closed_days": int(grid_daily["fecha"].nunique()),
            "closed_date_min": str(grid_daily["fecha"].min().date()),
            "closed_date_max": str(grid_daily["fecha"].max().date()),
            "state_updated": True,
            "state_path": str(args.state),
            "state_rows": int(len(state)),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path(os.getenv("GRID_PATH", str(DEFAULT_GRID))))
    parser.add_argument(
        "--hourly",
        type=Path,
        default=Path(os.getenv("AEMET_HOURLY_OBSERVATIONS_PATH", str(DEFAULT_HOURLY))),
    )
    parser.add_argument(
        "--daily",
        type=Path,
        default=Path(os.getenv("WEATHER_OBSERVATIONS_PATH", str(DEFAULT_DAILY))),
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
    parser.add_argument(
        "--min-coverage-hours",
        type=int,
        default=int(os.getenv("AEMET_CURRENT_MIN_COVERAGE_HOURS", "20")),
        help="Horas distintas mínimas para cerrar una estación-día (por defecto: 20).",
    )
    parser.add_argument(
        "--hourly-retention-hours",
        type=int,
        default=int(os.getenv("AEMET_HOURLY_RETENTION_HOURS", "72")),
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.getenv("WEATHER_STATE_RETENTION_DAYS", "60")),
    )
    parser.add_argument(
        "--idw-neighbors",
        type=int,
        default=int(os.getenv("AEMET_OBSERVATION_IDW_NEIGHBORS", "4")),
    )
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
    parser.add_argument("--loop", action="store_true", help="Repetir hasta Ctrl+C.")
    parser.add_argument("--interval-hours", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.interval_hours <= 0:
        raise ValueError("--interval-hours debe ser positivo.")
    while True:
        try:
            with run_lock(args.lock):
                summary = ingest_once(args)
            print("Captura AEMET completada: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
        except (AemetObservationError, FileNotFoundError, OSError, RunLockError, ValueError) as exc:
            print(f"Error de captura AEMET: {exc}")
            if not args.loop:
                raise SystemExit(1) from exc
        if not args.loop:
            return
        try:
            time.sleep(args.interval_hours * 3600)
        except KeyboardInterrupt:
            print("Captura AEMET detenida por el usuario.")
            return


if __name__ == "__main__":
    main()
