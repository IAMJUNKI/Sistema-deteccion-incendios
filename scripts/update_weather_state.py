#!/usr/bin/env python3
"""Update the compact recent-weather state used by daily inference.

The input must contain one daily row per cell with ``cell_id``, ``fecha``,
``tmax_vc``, ``rhmin_vc``, ``vmax_vc`` and ``prec_dia``. In production the
input is produced by the observation/analysis ingestion job; ERA5-Land may be
used only for bootstrap, never as a silent daily fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingestion.weather_state import merge_weather_state, save_weather_state


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Observaciones diarias CSV/Parquet")
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/processed/state/weather_daily_state.parquet"),
    )
    parser.add_argument("--as-of", default=None, help="Instante local de actualización")
    parser.add_argument("--source", default="meteogalicia_observation")
    parser.add_argument("--retention-days", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of = args.as_of or pd.Timestamp.now(tz="Europe/Madrid").isoformat()
    incoming = _read_table(args.input)
    existing = pd.read_parquet(args.state) if args.state.exists() else None
    incoming["source"] = args.source
    incoming["state_as_of"] = pd.Timestamp(as_of)
    merged = merge_weather_state(
        existing,
        incoming,
        as_of=as_of,
        retention_days=args.retention_days,
    )
    save_weather_state(merged, args.state)
    print(
        f"Estado guardado en {args.state}: {len(merged):,} filas, "
        f"{merged['cell_id'].nunique():,} celdas, "
        f"{merged['fecha'].min()} → {merged['fecha'].max()}"
    )


if __name__ == "__main__":
    main()
