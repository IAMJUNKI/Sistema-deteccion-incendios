#!/usr/bin/env python3
"""Comprueba MeteoSIX v5 sin exigir el estado histórico de observaciones.

Este smoke test valida exclusivamente la parte futura del pipeline: credencial,
endpoint v5, consulta por puntos y cobertura horaria. Por defecto consulta un
solo lote de hasta 20 puntos, suficiente para comprobar credencial, endpoint,
unidades y parser. ``--full-grid`` ejecuta además la descarga completa que usa
producción; puede requerir muchas peticiones porque MeteoSIX limita cada lote a
20 localizaciones. No ejecuta los modelos de riesgo ni necesita que AEMET haya
cerrado todavía los 30 días de memoria meteorológica.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from scripts.run_daily_inference import (
    _issue_timestamp,
    _provider_grid_candidates,
    _provider_query_resolution_km,
    _target_window,
    fetch_fresh_forecast,
    load_grid,
)
from src.ingestion.meteogalicia_forecast import (
    ForecastConfig,
    ForecastError,
    ForecastPoint,
    MeteoGaliciaClient,
    sample_provider_points,
    validate_hourly_forecast,
)

DEFAULT_GRID = Path("data/processed/grid/galicia_grid_1km_egif.parquet")
DEFAULT_FORECAST_DIR = Path("data/raw/meteogalicia")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path(os.getenv("GRID_PATH", str(DEFAULT_GRID))))
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=Path(os.getenv("METEOGALICIA_FORECAST_DIR", str(DEFAULT_FORECAST_DIR))),
    )
    parser.add_argument("--issue-time", default=None, help="Instante local/ISO para probar")
    parser.add_argument(
        "--max-points",
        type=int,
        default=20,
        help="Puntos consultados por malla en la prueba rápida (por defecto: 20).",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Descarga todos los puntos representativos y valida la malla completa.",
    )
    return parser.parse_args()


def _config_for_grid(grid_name: str) -> ForecastConfig:
    """Build the MeteoSIX configuration used by the quick check."""

    return ForecastConfig(
        api_key=os.getenv("METEOGALICIA_API_KEY", "").strip(),
        base_url=os.getenv("METEOGALICIA_BASE_URL", "https://servizos.meteogalicia.gal/apiv5"),
        model=os.getenv("METEOGALICIA_MODEL", "WRF"),
        grid=grid_name,
        api_version="v5",
        timezone_name="Europe/Madrid",
        auto_adjust_position=os.getenv("METEOGALICIA_AUTO_ADJUST_POSITION", "true").lower()
        in {"1", "true", "yes"},
        timeout_seconds=int(os.getenv("METEOGALICIA_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("METEOGALICIA_MAX_RETRIES", "3")),
    )


def _print_result(
    forecast: pd.DataFrame,
    issued_at: pd.Timestamp,
    quality: str,
    report: dict[str, object],
) -> None:
    """Print a provider result without exposing credentials."""

    valid_time = pd.to_datetime(forecast["valid_time"], utc=True)
    point_column = "point_id" if "point_id" in forecast.columns else "cell_id"
    points = forecast[point_column].nunique()
    hours_per_point = forecast.groupby(point_column)["valid_time"].nunique()
    print("MeteoSIX v5 OK")
    print(f"  calidad={quality}")
    print(f"  malla={report.get('selected_grid', report.get('grid'))}")
    print(f"  modelo={forecast.get('model', pd.Series(['unknown'])).iloc[0]}")
    print(f"  emisión={issued_at.isoformat()}")
    print(f"  válido_utc={valid_time.min()} → {valid_time.max()}")
    print(f"  puntos_con_datos={points}")
    print(f"  horas_por_punto={hours_per_point.min()}–{hours_per_point.max()}")
    print(f"  puntos_consultados={report.get('requested_points', points)}")
    print(f"  forecast_dir={report.get('forecast_dir')}")


def _check_sample(
    grid: pd.DataFrame,
    args: argparse.Namespace,
    issue_time: pd.Timestamp,
) -> None:
    """Check one small request per candidate grid before a full download."""

    if args.max_points < 1:
        raise SystemExit("--max-points debe ser al menos 1.")
    start_local, end_local = _target_window(issue_time)
    query_resolution = _provider_query_resolution_km()
    attempts: list[dict[str, str]] = []
    for grid_name in _provider_grid_candidates():
        points_df = sample_provider_points(
            grid,
            resolution_km=query_resolution,
        ).head(args.max_points)
        points = [
            ForecastPoint(str(row.point_id), float(row.lat_centroid), float(row.lon_centroid))
            for row in points_df.itertuples(index=False)
        ]
        total_batches = (len(points) + 19) // 20
        print(
            f"Consultando WRF {grid_name}: {len(points)} puntos en {total_batches} lote(s) "
            f"({start_local.isoformat()} → {end_local.isoformat()})",
            flush=True,
        )
        try:
            client = MeteoGaliciaClient(_config_for_grid(grid_name))
            raw = client.fetch_points(
                points,
                start_time=start_local,
                end_time=end_local,
                raw_dir=args.forecast_dir / "raw" / "smoke" / grid_name,
                progress_callback=lambda completed, total: print(
                    f"  lote {completed}/{total} completado", flush=True
                ),
            )
            validate_hourly_forecast(
                raw,
                expected_start=start_local,
                expected_end=end_local,
                min_points=1,
            )
            issued_at = pd.to_datetime(raw["forecast_run_at"].iloc[0], utc=True)
            _print_result(
                raw,
                issued_at,
                "fresh",
                {
                    "selected_grid": grid_name,
                    "requested_points": len(points),
                    "forecast_dir": args.forecast_dir,
                },
            )
            return
        except (ForecastError, OSError) as exc:
            attempts.append({"grid": grid_name, "error": str(exc)})
            print(
                f"No se pudo obtener WRF {grid_name}; se probará la siguiente malla: {exc}",
                flush=True,
            )
    raise ForecastError(f"Fallaron todas las mallas MeteoSIX configuradas: {attempts}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not os.getenv("METEOGALICIA_API_KEY", "").strip():
        raise SystemExit("METEOGALICIA_API_KEY no está configurada en .env.")
    if not args.grid.exists():
        raise SystemExit(f"No existe la rejilla operativa: {args.grid}")

    issue_time = _issue_timestamp(args.issue_time, None)
    grid = load_grid(args.grid)
    if args.full_grid:
        forecast, issued_at, quality, report = fetch_fresh_forecast(
            grid,
            issue_time=issue_time,
            forecast_dir=args.forecast_dir,
        )
        report["forecast_dir"] = args.forecast_dir
        _print_result(forecast, issued_at, quality, report)
    else:
        _check_sample(grid, args, issue_time)


if __name__ == "__main__":
    main()
