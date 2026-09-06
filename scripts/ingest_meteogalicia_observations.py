#!/usr/bin/env python3
"""Ingesta de observaciones de la red de estaciones de MeteoGalicia (EMA) y actualización del estado.

MeteoGalicia publica las medidas diarias de sus más de 140 estaciones en Galicia
a través de su servicio Open Data. Este script permite:

1. Rellenar automáticamente el hueco temporal entre la última observación consolidada
   en ``data/processed/state/weather_daily_state.parquet`` y el día D-1 (ayer).
2. Interpolar las observaciones a la rejilla de 1 km (EPSG:25829 / WGS84) mediante IDW.
3. Actualizar atómicamente el estado meteorológico diario requerido para la inferencia de incendios.

Ejemplos de uso::

    # Detección automática del hueco (por defecto hasta D-1):
    PYTHONPATH=. python scripts/ingest_meteogalicia_observations.py --auto-fill-gap

    # Recuperar los últimos 5 días:
    PYTHONPATH=. python scripts/ingest_meteogalicia_observations.py --days 5

    # Rango explícito de fechas:
    PYTHONPATH=. python scripts/ingest_meteogalicia_observations.py --start-date 2026-09-02 --end-date 2026-09-05
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.meteogalicia_observations import (
    MeteoGaliciaObservationClient,
    MeteoGaliciaObservationError,
    interpolate_meteogalicia_daily_to_grid,
    normalise_meteogalicia_daily_payload,
)
from src.ingestion.weather_state import merge_weather_state, save_weather_state
from src.operational.artifacts import RunLockError, atomic_write_parquet, run_lock

LOGGER = logging.getLogger("ingest_meteogalicia_observations")
GALICIA_TZ = "Europe/Madrid"

DEFAULT_GRID = Path("data/processed/grid/galicia_grid_1km_egif.parquet")
DEFAULT_OBSERVATIONS = Path("data/processed/observations/weather_daily_meteogalicia.parquet")
DEFAULT_STATE = Path("data/processed/state/weather_daily_state.parquet")
DEFAULT_RAW = Path("data/raw/meteogalicia/observations")
DEFAULT_LOCK = Path("data/processed/.meteogalicia_ingest.lock")


def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        LOGGER.warning("No se pudo leer el Parquet existente %s: %s", path, exc)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path(os.getenv("GRID_PATH", str(DEFAULT_GRID))),
        help="Ruta a la rejilla de 1 km de Galicia.",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(os.getenv("METEOGALICIA_OBSERVATIONS_PATH", str(DEFAULT_OBSERVATIONS))),
        help="Ruta de guardado para las observaciones interpoladas de MeteoGalicia.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(os.getenv("WEATHER_STATE_PATH", str(DEFAULT_STATE))),
        help="Ruta a weather_daily_state.parquet para actualización atómica.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.getenv("METEOGALICIA_OBSERVATIONS_RAW_DIR", str(DEFAULT_RAW))),
        help="Directorio para archivar el JSON bruto descargado de MeteoGalicia.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Fecha de inicio (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Fecha de fin (YYYY-MM-DD). Por defecto: ayer (D-1).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Número de días a recuperar si no se especifica start-date (por defecto: 5 días).",
    )
    parser.add_argument(
        "--auto-fill-gap",
        action="store_true",
        default=True,
        help="Detecta la última fecha en el estado y descarga desde fecha+1 hasta ayer.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.getenv("WEATHER_STATE_RETENTION_DAYS", "60")),
        help="Días máximos a retener en weather_daily_state.parquet.",
    )
    parser.add_argument(
        "--idw-neighbors",
        type=int,
        default=4,
        help="Número de estaciones vecinas para la interpolación IDW (por defecto: 4).",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.getenv("METEOGALICIA_INGEST_LOCK_PATH", str(DEFAULT_LOCK))),
        help="Ruta al archivo lock para evitar ejecuciones concurrentes.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Solo genera el archivo de observaciones interpoladas, sin actualizar el estado.",
    )
    return parser.parse_args()


def run_ingest(args: argparse.Namespace) -> dict[str, Any]:
    """Ejecuta la ingesta de MeteoGalicia y devuelve un resumen estructurado."""
    if not args.grid.exists():
        raise FileNotFoundError(f"No existe la rejilla de Galicia: {args.grid}")

    now_local = pd.Timestamp.now(tz=GALICIA_TZ)
    yesterday = (now_local - pd.Timedelta(days=1)).date()

    existing_state = _read_optional_parquet(args.state)

    # Determinar rango de fechas
    if args.start_date and args.end_date:
        start_date = pd.Timestamp(args.start_date).date()
        end_date = pd.Timestamp(args.end_date).date()
    elif args.start_date:
        start_date = pd.Timestamp(args.start_date).date()
        end_date = yesterday
    elif args.auto_fill_gap and existing_state is not None and not existing_state.empty:
        last_state_date = pd.to_datetime(existing_state["fecha"]).max().date()
        start_date = last_state_date + timedelta(days=1)
        end_date = pd.Timestamp(args.end_date).date() if args.end_date else yesterday
        if start_date > end_date:
            LOGGER.info(
                "El estado ya está al día hasta %s. No hay fechas que recuperar.",
                last_state_date,
            )
            return {
                "status": "already_up_to_date",
                "last_state_date": str(last_state_date),
                "target_end_date": str(end_date),
                "closed_days": 0,
            }
    else:
        end_date = pd.Timestamp(args.end_date).date() if args.end_date else yesterday
        start_date = end_date - timedelta(days=args.days - 1)

    if end_date > yesterday:
        LOGGER.warning(
            "end_date (%s) es superior a ayer (%s). MeteoGalicia solo consolida observaciones hasta D-1.",
            end_date,
            yesterday,
        )
        end_date = yesterday

    if start_date > end_date:
        raise MeteoGaliciaObservationError(
            f"start_date ({start_date}) no puede ser posterior a end_date ({end_date})."
        )

    print(f"📡 Descargando observaciones MeteoGalicia desde {start_date} hasta {end_date}...")
    client = MeteoGaliciaObservationClient()
    payload = client.fetch_daily_observations(
        start_date=start_date,
        end_date=end_date,
        raw_dir=args.raw_dir,
    )

    print("⚙️ Normalizando medidas de estaciones...")
    station_daily = normalise_meteogalicia_daily_payload(
        payload,
        start_date=start_date,
        end_date=end_date,
    )

    print(
        f"📍 {station_daily['station_id'].nunique()} estaciones procesadas en "
        f"{station_daily['fecha'].nunique()} fechas."
    )

    print(f"🗺️ Interpolando estaciones a la rejilla de 1 km mediante IDW ({args.idw_neighbors} vecinos)...")
    grid = pd.read_parquet(args.grid, columns=["cell_id", "lat_centroid", "lon_centroid"])
    grid_daily = interpolate_meteogalicia_daily_to_grid(
        station_daily,
        grid,
        neighbors=args.idw_neighbors,
        source="meteogalicia_ema_idw",
    )
    grid_daily["state_as_of"] = now_local

    # Guardar observaciones interpoladas
    args.observations.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(grid_daily, args.observations)

    summary: dict[str, Any] = {
        "status": "success",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "stations": int(station_daily["station_id"].nunique()),
        "closed_days": int(grid_daily["fecha"].nunique()),
        "total_cell_records": int(len(grid_daily)),
        "observations_path": str(args.observations),
        "state_updated": False,
    }

    if not args.no_state:
        print("💾 Actualizando estado diario (weather_daily_state.parquet)...")
        updated_state = merge_weather_state(
            existing_state,
            grid_daily,
            as_of=now_local,
            retention_days=args.retention_days,
        )
        save_weather_state(updated_state, args.state)
        summary["state_updated"] = True
        summary["state_path"] = str(args.state)
        summary["state_rows"] = int(len(updated_state))

    return summary


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    try:
        with run_lock(args.lock):
            summary = run_ingest(args)
        print("✅ Ingesta de MeteoGalicia completada:")
        for k, v in summary.items():
            print(f"   • {k}: {v}")
    except (MeteoGaliciaObservationError, FileNotFoundError, RunLockError) as exc:
        print(f"❌ Error en ingesta MeteoGalicia: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
