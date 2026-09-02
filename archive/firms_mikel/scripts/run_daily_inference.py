#!/usr/bin/env python3
"""Inferencia operativa de riesgo T+1/T+2/T+3.

La inferencia consume un forecast horario real de MeteoGalicia, lo asigna a la
rejilla y carga modelos serializados. Si la descarga falla, solo usa el último
forecast archivado que cubra los tres días y marca el resultado como ``stale``.
No existe fallback silencioso a una fecha histórica.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - el entorno de producción incluye GeoPandas
    gpd = None

from src.features.operational_features import build_operational_features
from src.ingestion.aemet_forecast import (
    AEMET_API_VERSION,
    AEMET_BASE_URL,
    AemetClient,
    AemetForecastConfig,
    parse_municipality_env,
)
from src.ingestion.meteogalicia_forecast import (
    ForecastConfig,
    ForecastError,
    ForecastPoint,
    ForecastValidationError,
    MeteoGaliciaClient,
    _as_utc_timestamp,
    assign_forecast_to_grid,
    load_latest_forecast,
    sample_provider_points,
    save_forecast_parquet,
    validate_hourly_forecast,
)
from src.ingestion.weather_state import load_weather_state
from src.models.forecast_risk_model import add_risk_outputs, load_horizon_model
from src.operational.artifacts import (
    atomic_write_json,
    atomic_write_parquet,
    run_lock,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)
GALICIA_TZ = ZoneInfo("Europe/Madrid")
HORIZONS = (1, 2, 3)
DEFAULT_GRID_PATH = Path("data/processed/grid/galicia_grid_1km_2018.parquet")
DEFAULT_FORECAST_DIR = Path("data/raw/meteogalicia")
DEFAULT_OUTPUT_PATH = Path("data/processed/predicciones_operativas.parquet")
DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_STATE_PATH = Path("data/processed/state/weather_daily_state.parquet")
DEFAULT_LOCK_PATH = Path("data/processed/.daily_inference.lock")
DEFAULT_HISTORY_DAYS = 30
DEFAULT_METEOSIX_URL = "https://servizos.meteogalicia.gal/apiv5"
DEFAULT_METEOSIX_GRIDS = ("1km", "04km")
DEFAULT_QUERY_RESOLUTION_KM = 4.0
DEFAULT_FORECAST_PROVIDER = "auto"


def _max_source_distance_km() -> float:
    return float(os.getenv("FORECAST_MAX_SOURCE_DISTANCE_KM", "10"))


def _history_days() -> int:
    return int(os.getenv("INFERENCE_HISTORY_DAYS", str(DEFAULT_HISTORY_DAYS)))


def _provider_grid_candidates() -> tuple[str, ...]:
    """Devuelve las mallas MeteoSIX en orden de preferencia.

    ``METEOGALICIA_GRIDS`` permite configurar una lista explícita, por ejemplo
    ``1km,04km``. Se conserva compatibilidad con la variable antigua
    ``METEOGALICIA_GRID`` para instalaciones existentes.
    """

    configured = os.getenv("METEOGALICIA_GRIDS", "")
    if configured.strip():
        candidates = [value.strip() for value in configured.split(",") if value.strip()]
    else:
        primary = os.getenv("METEOGALICIA_GRID", DEFAULT_METEOSIX_GRIDS[0]).strip()
        fallback = os.getenv("METEOGALICIA_FALLBACK_GRID", DEFAULT_METEOSIX_GRIDS[1]).strip()
        candidates = [primary, fallback]
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if not unique:
        raise ForecastError("METEOGALICIA_GRIDS no contiene ninguna malla válida.")
    return tuple(unique)


def _provider_query_resolution_km() -> float:
    """Controla cuántos puntos representativos se consultan al proveedor."""

    value = float(os.getenv("METEOGALICIA_QUERY_RESOLUTION_KM", str(DEFAULT_QUERY_RESOLUTION_KM)))
    if value <= 0:
        raise ForecastError("METEOGALICIA_QUERY_RESOLUTION_KM debe ser positivo.")
    return value


def _first_column_value(frame: pd.DataFrame, columns: tuple[str, ...], default: str = "unknown") -> str:
    """Obtiene el primer valor disponible de una lista de columnas de metadatos."""

    for column in columns:
        if column in frame.columns and not frame[column].empty:
            return str(frame[column].iloc[0])
    return default


def _issue_timestamp(issue_time: datetime | pd.Timestamp | str | None, target_date: str | None) -> pd.Timestamp:
    if issue_time is None:
        if target_date:
            issue = pd.Timestamp(target_date).tz_localize(GALICIA_TZ) + pd.Timedelta(hours=5)
        else:
            issue = pd.Timestamp.now(tz=GALICIA_TZ)
    else:
        issue = pd.Timestamp(issue_time)
        if issue.tzinfo is None:
            issue = issue.tz_localize(GALICIA_TZ)
    return issue.tz_convert(GALICIA_TZ)


def _local_simulation_settings(issue_time: pd.Timestamp) -> dict[str, object]:
    """Resolve the optional local-only historical-state simulation.

    The simulation does not move the real forecast clock. It only validates
    the observed state as if the last available observation were the day
    before a simulated run. This is useful on a laptop when the AEMET hourly
    collector has not been running continuously, but it is not a production
    readiness mode.
    """

    enabled = os.getenv("LOCAL_SIMULATION_MODE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    environment = os.getenv("PIPELINE_ENVIRONMENT", "local").strip().lower()
    if not enabled:
        return {
            "enabled": False,
            "environment": environment or "local",
            "mode": environment or "local",
        }
    if environment not in {"local", "development", "dev", "test"}:
        raise ForecastError(
            "LOCAL_SIMULATION_MODE solo puede activarse con "
            "PIPELINE_ENVIRONMENT=local|development|test."
        )

    raw_as_of = os.getenv("LOCAL_SIMULATION_AS_OF_DATE", "").strip()
    if not raw_as_of:
        raise ForecastError(
            "LOCAL_SIMULATION_AS_OF_DATE es obligatoria cuando "
            "LOCAL_SIMULATION_MODE=true (formato YYYY-MM-DD)."
        )
    try:
        as_of_timestamp = pd.Timestamp(raw_as_of)
    except (TypeError, ValueError) as exc:
        raise ForecastError(
            "LOCAL_SIMULATION_AS_OF_DATE debe tener formato YYYY-MM-DD."
        ) from exc
    if as_of_timestamp.tzinfo is None:
        as_of_timestamp = as_of_timestamp.tz_localize(GALICIA_TZ)
    else:
        as_of_timestamp = as_of_timestamp.tz_convert(GALICIA_TZ)
    as_of_date = as_of_timestamp.normalize().tz_localize(None)
    real_issue_date = issue_time.tz_convert(GALICIA_TZ).normalize().tz_localize(None)
    if as_of_date >= real_issue_date:
        raise ForecastError(
            "LOCAL_SIMULATION_AS_OF_DATE debe ser anterior a la fecha real de ejecución."
        )
    simulated_issue_time = (
        as_of_date.tz_localize(GALICIA_TZ) + pd.Timedelta(days=1, hours=5)
    )
    return {
        "enabled": True,
        "environment": environment or "local",
        "mode": "local_state_simulation",
        "as_of_date": as_of_date,
        "simulated_issue_time": simulated_issue_time,
        "real_issue_time": issue_time,
    }


def _target_window(issue_time: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    issue_date = issue_time.normalize()
    start_local = issue_date + pd.Timedelta(days=1)
    end_local = issue_date + pd.Timedelta(days=4) - pd.Timedelta(hours=1)
    return start_local, end_local


def load_grid(grid_path: str | Path = DEFAULT_GRID_PATH) -> gpd.GeoDataFrame:
    """Carga la rejilla y ofrece un error accionable para Parquet corrupto."""

    path = Path(grid_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la rejilla operativa: {path}")
    if gpd is None:
        raise ForecastError(
            "GeoPandas no está instalado; activa el entorno incendios-forestales "
            "para cargar la rejilla Parquet."
        )
    try:
        grid = gpd.read_parquet(path)
    except OSError as exc:
        raise ForecastError(
            f"No se pudo leer la rejilla {path}. El Parquet parece incompatible o corrupto; "
            "regenera la rejilla antes de lanzar producción."
        ) from exc
    required = {"cell_id", "lat_centroid", "lon_centroid"}
    missing = required.difference(grid.columns)
    if missing:
        raise ForecastError(f"La rejilla operativa carece de columnas: {sorted(missing)}")
    return grid


def load_history_from_master(
    dataset_dir: str | Path = "misc/Dataset/Mike",
    years: Iterable[int] = range(2019, 2025),
) -> pd.DataFrame:
    """Carga la historia meteorológica disponible para memorias de sequedad."""

    columns = [
        "cell_id",
        "fecha",
        "tmax_vc",
        "rhmin_vc",
        "vmax_vc",
        "prec_dia",
    ]
    frames = []
    base = Path(dataset_dir)
    for year in years:
        path = base / f"dataset_maestro_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError(f"No hay datasets históricos en {base}")
    return pd.concat(frames, ignore_index=True)


def load_operational_history(
    state_path: str | Path,
    *,
    issue_time: pd.Timestamp,
    grid: pd.DataFrame,
    history_days: int = DEFAULT_HISTORY_DAYS,
) -> pd.DataFrame:
    """Load the recent state required for a safe production inference."""

    return load_weather_state(
        state_path,
        issue_time=issue_time,
        required_cells=grid["cell_id"].tolist(),
        history_days=history_days,
    )


def _forecast_coverage_report(
    forecast: pd.DataFrame,
    *,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
) -> dict[str, object]:
    """Summarise per-cell hourly coverage for the run manifest."""

    expected_hours = len(
        pd.date_range(
            start=_as_utc_timestamp(expected_start),
            end=_as_utc_timestamp(expected_end),
            freq="h",
            tz="UTC",
        )
    )
    times = pd.to_datetime(forecast["valid_time"], utc=True)
    per_cell = forecast.assign(_valid_time=times).groupby("cell_id")["_valid_time"].nunique()
    ratios = per_cell / expected_hours
    return {
        "expected_hours_per_cell": expected_hours,
        "cells": int(per_cell.size),
        "minimum_ratio": float(ratios.min()) if len(ratios) else 0.0,
        "mean_ratio": float(ratios.mean()) if len(ratios) else 0.0,
        "complete_cells": int((ratios >= 1.0).sum()),
    }


def _validate_grid_coverage(
    forecast_df: pd.DataFrame,
    *,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    expected_cells: int,
) -> None:
    """Comprueba cobertura horaria completa para cada celda asignada."""

    expected_hours = pd.date_range(
        start=_as_utc_timestamp(expected_start),
        end=_as_utc_timestamp(expected_end),
        freq="h",
        tz="UTC",
    )
    times = pd.to_datetime(forecast_df["valid_time"], utc=True)
    grouped = forecast_df.assign(_valid_time=times).groupby("cell_id")
    if grouped.ngroups != expected_cells:
        raise ForecastValidationError(
            f"El forecast solo cubre {grouped.ngroups} celdas de {expected_cells}."
        )
    for cell_id, group in grouped:
        actual = pd.DatetimeIndex(group["_valid_time"].unique()).sort_values()
        missing = expected_hours.difference(actual)
        if len(missing):
            raise ForecastValidationError(
                f"La celda {cell_id} carece de {len(missing)} horas; primera ausente: {missing[0]}"
            )


def _fetch_fresh_meteogalicia_forecast(
    grid: gpd.GeoDataFrame,
    *,
    issue_time: pd.Timestamp,
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
) -> tuple[pd.DataFrame, pd.Timestamp, str, dict[str, object]]:
    """Descarga WRF 1 km y usa 04 km solo si la malla preferida falla.

    La rejilla de riesgo sigue siendo de 1 km en ambos casos. La variable
    ``forecast_quality`` distingue ``fresh`` de ``fresh_fallback`` para que la
    degradación espacial sea visible en el manifiesto y en el dashboard.
    """

    api_key = os.getenv("METEOGALICIA_API_KEY", "")
    issue_time = _issue_timestamp(issue_time, None)
    start_local, end_local = _target_window(issue_time)
    base_url = os.getenv("METEOGALICIA_BASE_URL", DEFAULT_METEOSIX_URL)
    if "/apiv4" in base_url.rstrip("/").lower() and os.getenv(
        "METEOGALICIA_ALLOW_V4", "false"
    ).lower() not in {"1", "true", "yes"}:
        raise ForecastError(
            "El entorno todavía apunta a MeteoSIX v4. Actualiza "
            "METEOGALICIA_BASE_URL a /apiv5; usa METEOGALICIA_ALLOW_V4=true "
            "solo para una prueba de compatibilidad controlada."
        )
    query_resolution = _provider_query_resolution_km()
    attempts: list[dict[str, object]] = []

    for index, grid_name in enumerate(_provider_grid_candidates()):
        config = ForecastConfig(
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("METEOGALICIA_MODEL", "WRF"),
            grid=grid_name,
            api_version="v5",
            timezone_name="Europe/Madrid",
            auto_adjust_position=os.getenv("METEOGALICIA_AUTO_ADJUST_POSITION", "true").lower()
            in {"1", "true", "yes"},
            timeout_seconds=int(os.getenv("METEOGALICIA_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("METEOGALICIA_MAX_RETRIES", "3")),
        )
        try:
            points_df = sample_provider_points(grid, resolution_km=query_resolution)
            points = [
                ForecastPoint(str(row.point_id), float(row.lat_centroid), float(row.lon_centroid))
                for row in points_df.itertuples(index=False)
            ]
            total_batches = (
                len(points) + 20 - 1
            ) // 20
            LOGGER.info(
                "Consultando WRF %s: %s puntos en %s lote(s) (%s → %s)",
                grid_name,
                len(points),
                total_batches,
                start_local.isoformat(),
                end_local.isoformat(),
            )
            client = MeteoGaliciaClient(config)
            downloaded_at = pd.Timestamp.now(tz=GALICIA_TZ)
            raw = client.fetch_points(
                points,
                start_time=start_local,
                end_time=end_local,
                raw_dir=Path(forecast_dir) / "raw" / grid_name,
                downloaded_at=downloaded_at,
                progress_callback=lambda completed, total, name=grid_name: LOGGER.info(
                    "WRF %s: lote %s/%s completado",
                    name,
                    completed,
                    total,
                ),
            )
            assigned = assign_forecast_to_grid(raw, grid)
            validate_hourly_forecast(
                assigned,
                expected_start=start_local,
                expected_end=end_local,
                min_points=len(grid),
                max_source_distance_km=_max_source_distance_km(),
            )
            _validate_grid_coverage(
                assigned,
                expected_start=start_local,
                expected_end=end_local,
                expected_cells=len(grid),
            )
            issued = _as_utc_timestamp(raw["forecast_run_at"].iloc[0])
            quality = "fresh" if index == 0 else "fresh_fallback"
            assigned["forecast_quality"] = quality
            assigned["forecast_selection"] = "primary" if index == 0 else "fallback"
            assigned["forecast_grid"] = grid_name
            assigned["forecast_api_version"] = config.api_version
            assigned["forecast_query_resolution_km"] = query_resolution
            assigned["forecast_requested_points"] = len(points)
            output_name = (
                f"forecast_{issued.strftime('%Y%m%dT%H%M%SZ')}_{grid_name}.parquet"
            )
            save_forecast_parquet(assigned, Path(forecast_dir) / output_name)
            report = {
                "selected_grid": grid_name,
                "quality": quality,
                "selection": "primary" if index == 0 else "fallback",
                "source": "provider",
                "api_version": config.api_version,
                "query_resolution_km": query_resolution,
                "requested_points": len(points),
                "attempts": attempts,
            }
            return assigned, issued, quality, report
        except (ForecastError, OSError, requests.RequestException) as exc:
            attempt = {"grid": grid_name, "error": str(exc)}
            attempts.append(attempt)
            LOGGER.warning(
                "No se pudo obtener WRF %s; se probará la siguiente malla: %s",
                grid_name,
                exc,
            )

    raise ForecastError(f"Fallaron todas las mallas MeteoSIX configuradas: {attempts}")


def _has_real_credential(value: str | None) -> bool:
    if not value or not value.strip():
        return False
    return not value.strip().lower().startswith(("your_", "tu_", "replace_", "changeme"))


def _forecast_provider() -> str:
    """Selecciona el proveedor explicitado o disponible en ``.env``."""

    configured = os.getenv("FORECAST_PROVIDER", DEFAULT_FORECAST_PROVIDER).strip().lower()
    aliases = {"meteo": "meteogalicia", "meteosix": "meteogalicia", "aemet": "aemet"}
    configured = aliases.get(configured, configured)
    if configured not in {"auto", "meteogalicia", "aemet"}:
        raise ForecastError(
            "FORECAST_PROVIDER debe ser auto, meteogalicia o aemet."
        )
    if configured != "auto":
        return configured
    if _has_real_credential(os.getenv("METEOGALICIA_API_KEY")):
        return "meteogalicia"
    if _has_real_credential(os.getenv("AEMET_API_KEY")):
        return "aemet"
    raise ForecastError(
        "No hay ningún proveedor meteorológico configurado: añade "
        "METEOGALICIA_API_KEY o AEMET_API_KEY, o usa FORECAST_PROVIDER explícitamente."
    )


def _aemet_precipitation_fallback() -> float | None:
    """Lee un proxy de precipitación solo si se habilita explícitamente."""

    raw = os.getenv("AEMET_MISSING_PRECIPITATION_FALLBACK", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ForecastError(
            "AEMET_MISSING_PRECIPITATION_FALLBACK debe ser un número no negativo."
        ) from exc
    if value < 0:
        raise ForecastError(
            "AEMET_MISSING_PRECIPITATION_FALLBACK debe ser no negativo."
        )
    return value


def _fetch_fresh_aemet_forecast(
    grid: gpd.GeoDataFrame,
    *,
    issue_time: pd.Timestamp,
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
) -> tuple[pd.DataFrame, pd.Timestamp, str, dict[str, object]]:
    """Obtiene un forecast AEMET municipal degradado para pruebas y contingencia."""

    api_key = os.getenv("AEMET_API_KEY", "")
    issue_time = _issue_timestamp(issue_time, None)
    start_local, end_local = _target_window(issue_time)
    config = AemetForecastConfig(
        api_key=api_key,
        base_url=os.getenv("AEMET_BASE_URL", AEMET_BASE_URL),
        timezone_name="Europe/Madrid",
        timeout_seconds=int(os.getenv("AEMET_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("AEMET_MAX_RETRIES", "3")),
        use_hourly_overlay=os.getenv("AEMET_USE_HOURLY_OVERLAY", "true").lower()
        in {"1", "true", "yes"},
        missing_precipitation_fallback=_aemet_precipitation_fallback(),
    )
    points = parse_municipality_env(os.getenv("AEMET_MUNICIPALITIES"))
    client = AemetClient(config)
    downloaded_at = pd.Timestamp.now(tz=GALICIA_TZ)
    raw = client.fetch_points(
        points,
        start_time=start_local,
        end_time=end_local,
        raw_dir=Path(forecast_dir) / "raw" / "aemet",
        downloaded_at=downloaded_at,
    )
    assigned = assign_forecast_to_grid(raw, grid)
    validate_hourly_forecast(
        assigned,
        expected_start=start_local,
        expected_end=end_local,
        min_points=len(grid),
        max_source_distance_km=float(os.getenv("AEMET_MAX_SOURCE_DISTANCE_KM", "80")),
    )
    _validate_grid_coverage(
        assigned,
        expected_start=start_local,
        expected_end=end_local,
        expected_cells=len(grid),
    )
    issued = _as_utc_timestamp(raw["forecast_run_at"].min())
    quality = (
        "fresh_aemet_proxy"
        if config.missing_precipitation_fallback is not None
        else "fresh_aemet"
    )
    assigned["forecast_quality"] = quality
    assigned["forecast_selection"] = "aemet"
    assigned["forecast_grid"] = "municipal"
    assigned["forecast_api_version"] = AEMET_API_VERSION
    assigned["forecast_query_resolution_km"] = np.nan
    assigned["forecast_requested_points"] = len(points)
    output_name = f"forecast_{issued.strftime('%Y%m%dT%H%M%SZ')}_aemet.parquet"
    save_forecast_parquet(assigned, Path(forecast_dir) / output_name)
    return assigned, issued, quality, {
        "selected_grid": "municipal",
        "quality": quality,
        "selection": "aemet",
        "source": "provider",
        "provider": "aemet",
        "api_version": AEMET_API_VERSION,
        "requested_points": len(points),
        "municipalities": [point.municipality_id for point in points],
        "note": (
            "Predicción municipal AEMET; no equivale a WRF 1 km."
            if config.missing_precipitation_fallback is None
            else "AEMET con proxy explícito de precipitación; solo para pruebas."
        ),
        "precipitation_mode": (
            "provider_amount"
            if config.missing_precipitation_fallback is None
            else "explicit_fallback"
        ),
    }


def fetch_fresh_forecast(
    grid: gpd.GeoDataFrame,
    *,
    issue_time: pd.Timestamp,
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
) -> tuple[pd.DataFrame, pd.Timestamp, str, dict[str, object]]:
    """Descarga el proveedor configurado en ``FORECAST_PROVIDER``."""

    provider = _forecast_provider()
    if provider == "aemet":
        return _fetch_fresh_aemet_forecast(
            grid,
            issue_time=issue_time,
            forecast_dir=forecast_dir,
        )
    return _fetch_fresh_meteogalicia_forecast(
        grid,
        issue_time=issue_time,
        forecast_dir=forecast_dir,
    )


def _load_forecast_with_fallback(
    grid: gpd.GeoDataFrame,
    *,
    issue_time: pd.Timestamp,
    forecast_df: pd.DataFrame | None,
    forecast_dir: str | Path,
    allow_stale: bool,
) -> tuple[pd.DataFrame, pd.Timestamp, str, dict[str, object]]:
    start_local, end_local = _target_window(issue_time)
    if forecast_df is not None:
        assigned = forecast_df.copy()
        if "cell_id" not in assigned.columns:
            assigned = assign_forecast_to_grid(assigned, grid)
        validate_hourly_forecast(
            assigned,
            expected_start=start_local,
            expected_end=end_local,
            min_points=len(grid),
            max_source_distance_km=_max_source_distance_km(),
        )
        _validate_grid_coverage(
            assigned,
            expected_start=start_local,
            expected_end=end_local,
            expected_cells=len(grid),
        )
        issued = _as_utc_timestamp(assigned["forecast_run_at"].iloc[0]) if "forecast_run_at" in assigned else _as_utc_timestamp(issue_time)
        return assigned, issued, "fresh", {
            "selected_grid": _first_column_value(assigned, ("forecast_grid", "grid")),
            "quality": "fresh",
            "selection": "injected",
            "source": "injected",
        }

    try:
        return fetch_fresh_forecast(grid, issue_time=issue_time, forecast_dir=forecast_dir)
    except (ForecastError, OSError, requests.RequestException) as exc:
        LOGGER.exception("No se pudo descargar un forecast fresco: %s", exc)
        if not allow_stale:
            raise
        stale, issued, _ = load_latest_forecast(
            forecast_dir,
            required_start=start_local,
            required_end=end_local,
        )
        assigned = stale if "cell_id" in stale.columns else assign_forecast_to_grid(stale, grid)
        validate_hourly_forecast(
            assigned,
            expected_start=start_local,
            expected_end=end_local,
            min_points=len(grid),
            max_source_distance_km=_max_source_distance_km(),
        )
        _validate_grid_coverage(
            assigned,
            expected_start=start_local,
            expected_end=end_local,
            expected_cells=len(grid),
        )
        selected_grid = _first_column_value(assigned, ("forecast_grid", "grid"))
        return assigned, issued, "stale", {
            "selected_grid": selected_grid,
            "quality": "stale",
            "selection": "archive",
            "source": "archive",
        }


def run_daily_inference_pipeline(
    df_master: pd.DataFrame | None = None,
    target_date: str | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    issue_time: datetime | pd.Timestamp | str | None = None,
    forecast_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    grid_df: gpd.GeoDataFrame | pd.DataFrame | None = None,
    grid_path: str | Path = DEFAULT_GRID_PATH,
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    weather_state_path: str | Path = DEFAULT_STATE_PATH,
    manifest_path: str | Path | None = None,
    lock_path: str | Path | None = DEFAULT_LOCK_PATH,
    allow_stale: bool = True,
    use_lock: bool = True,
) -> pd.DataFrame:
    """Run the operational inference with an exclusive publication lock."""

    lock_context = run_lock(lock_path) if use_lock and lock_path else nullcontext()
    with lock_context:
        return _run_daily_inference_pipeline(
            df_master=df_master,
            target_date=target_date,
            output_path=output_path,
            issue_time=issue_time,
            forecast_df=forecast_df,
            history_df=history_df,
            grid_df=grid_df,
            grid_path=grid_path,
            forecast_dir=forecast_dir,
            model_dir=model_dir,
            weather_state_path=weather_state_path,
            manifest_path=manifest_path,
            allow_stale=allow_stale,
        )


def _run_daily_inference_pipeline(
    df_master: pd.DataFrame | None = None,
    target_date: str | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    issue_time: datetime | pd.Timestamp | str | None = None,
    forecast_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    grid_df: gpd.GeoDataFrame | pd.DataFrame | None = None,
    grid_path: str | Path = DEFAULT_GRID_PATH,
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    weather_state_path: str | Path = DEFAULT_STATE_PATH,
    manifest_path: str | Path | None = None,
    allow_stale: bool = True,
) -> pd.DataFrame:
    """Internal inference body; callers should use the locked public wrapper."""

    load_dotenv()
    started_at = time.perf_counter()
    issue = _issue_timestamp(issue_time, target_date)
    simulation = _local_simulation_settings(issue)
    state_issue = (
        simulation["simulated_issue_time"]
        if simulation["enabled"]
        else issue
    )
    grid = grid_df if grid_df is not None else load_grid(grid_path)
    if history_df is not None:
        history = history_df
        state_report = {"source": "injected"}
    elif df_master is not None:
        history = df_master
        state_report = {"source": "legacy_master_argument"}
    else:
        history = load_operational_history(
            weather_state_path,
            issue_time=state_issue,
            grid=grid,
            history_days=_history_days(),
        )
        if simulation["enabled"]:
            history = history[
                pd.to_datetime(history["fecha"]) <= simulation["as_of_date"]
            ].copy()
        state_report = {
            "source": "weather_state",
            "path": str(weather_state_path),
            "rows": int(len(history)),
            "latest_date": str(pd.to_datetime(history["fecha"]).max().date()),
            "validation_mode": "simulated_as_of" if simulation["enabled"] else "live",
            "simulation_as_of_date": (
                str(simulation["as_of_date"].date()) if simulation["enabled"] else None
            ),
        }
    LOGGER.info(
        "Estado meteorológico listo: %s filas, última fecha %s",
        len(history),
        pd.to_datetime(history["fecha"]).max().date(),
    )
    forecast, issued_at, quality, forecast_selection = _load_forecast_with_fallback(
        grid,
        issue_time=issue,
        forecast_df=forecast_df,
        forecast_dir=forecast_dir,
        allow_stale=allow_stale,
    )
    LOGGER.info(
        "Forecast listo: %s filas, emisión %s, calidad=%s",
        len(forecast),
        issued_at.isoformat(),
        quality,
    )
    features = build_operational_features(
        forecast,
        history,
        grid,
        issue_time=issue,
        horizons=HORIZONS,
    )
    LOGGER.info(
        "Features operativas listas: %s filas (%s celdas × %s horizontes)",
        len(features),
        features["cell_id"].nunique(),
        features["horizon_days"].nunique(),
    )
    start_local, end_local = _target_window(issue)
    coverage_report = _forecast_coverage_report(
        forecast,
        expected_start=start_local,
        expected_end=end_local,
    )
    features["forecast_run_at"] = issued_at
    features["forecast_quality"] = quality
    features["forecast_selection"] = str(
        forecast_selection.get("selection", forecast_selection.get("source", "provider"))
    )
    features["forecast_provider"] = _first_column_value(
        forecast, ("provider",), default=str(forecast_selection.get("provider", "unknown"))
    )
    features["forecast_grid"] = _first_column_value(
        forecast, ("forecast_grid", "grid")
    )
    features["forecast_api_version"] = _first_column_value(
        forecast, ("forecast_api_version",), default="v5"
    )
    features["forecast_query_resolution_km"] = pd.to_numeric(
        forecast.get("forecast_query_resolution_km", pd.Series([float("nan")])),
        errors="coerce",
    ).iloc[0]
    features["forecast_requested_points"] = pd.to_numeric(
        forecast.get("forecast_requested_points", pd.Series([float("nan")])),
        errors="coerce",
    ).iloc[0]
    if "forecast_run_source" in forecast.columns:
        features["forecast_run_source"] = str(forecast["forecast_run_source"].iloc[0])
    if "downloaded_at" in forecast.columns:
        features["forecast_downloaded_at"] = forecast["downloaded_at"].iloc[0]
    if "source_distance_km" in forecast.columns:
        features["forecast_source_distance_km"] = float(forecast["source_distance_km"].max())
    features["forecast_age_hours"] = max(
        0.0,
        (issue.tz_convert("UTC") - issued_at).total_seconds() / 3600.0,
    )
    features["pipeline_environment"] = simulation["environment"]
    features["pipeline_run_mode"] = simulation["mode"]
    features["weather_state_validation_mode"] = (
        "simulated_as_of" if simulation["enabled"] else "live"
    )
    features["weather_state_simulation_as_of_date"] = (
        str(simulation["as_of_date"].date()) if simulation["enabled"] else None
    )

    results = []
    model_report: dict[str, object] = {}
    for horizon in HORIZONS:
        horizon_features = features[features["horizon_days"] == horizon].copy()
        LOGGER.info(
            "Ejecutando modelo de riesgo T+%s sobre %s celdas",
            horizon,
            len(horizon_features),
        )
        model = load_horizon_model(horizon, model_dir=model_dir)
        probabilities = model.predict_proba(horizon_features)
        scored = add_risk_outputs(horizon_features, probabilities[:, 1])
        results.append(scored)
        model_path = Path(model_dir) / f"forecast_risk_t{horizon}.joblib"
        model_report[str(horizon)] = {
            "path": str(model_path),
            "sha256": sha256_file(model_path) if model_path.exists() else None,
            "metadata": getattr(model, "metadata", {}),
        }
    output = pd.concat(results, ignore_index=True)
    output["forecast_quality_note"] = {
        "fresh": "WRF 1 km preferido disponible.",
        "fresh_fallback": "WRF 1 km no superó la validación; se utilizó WRF 04 km.",
        "fresh_aemet": (
            "Proveedor AEMET municipal; se usa para pruebas o contingencia y no equivale a WRF 1 km."
        ),
        "fresh_aemet_proxy": (
            "AEMET municipal con proxy explícito de precipitación; solo para pruebas técnicas."
        ),
        "stale": "No hubo descarga válida; se reutilizó el último forecast archivado.",
    }.get(quality, "Calidad meteorológica no determinada.")
    generated_at = pd.Timestamp.now(tz="UTC")
    output["generated_at"] = generated_at

    destination = Path(output_path)
    atomic_write_parquet(output, destination)
    metadata = {
        "run_id": generated_at.strftime("%Y%m%dT%H%M%SZ"),
        "issue_time": issue.isoformat(),
        "forecast_run_at": issued_at.isoformat(),
        "forecast_quality": quality,
        "forecast_selection": forecast_selection,
        "forecast_provider": _first_column_value(
            forecast, ("provider",), default=str(forecast_selection.get("provider", "unknown"))
        ),
        "forecast_grid": _first_column_value(forecast, ("forecast_grid", "grid")),
        "forecast_api_version": _first_column_value(
            forecast, ("forecast_api_version",), default="v5"
        ),
        "forecast_query_resolution_km": float(
            features["forecast_query_resolution_km"].iloc[0]
        ),
        "forecast_requested_points": int(features["forecast_requested_points"].iloc[0])
        if pd.notna(features["forecast_requested_points"].iloc[0])
        else None,
        "history_days": _history_days(),
        "pipeline_environment": simulation["environment"],
        "pipeline_run_mode": simulation["mode"],
        "weather_state_validation_mode": (
            "simulated_as_of" if simulation["enabled"] else "live"
        ),
        "weather_state_simulation_as_of_date": (
            str(simulation["as_of_date"].date()) if simulation["enabled"] else None
        ),
        "forecast_run_source": str(
            forecast.get("forecast_run_source", pd.Series(["unknown"])).iloc[0]
        ),
        "forecast_downloaded_at": str(
            forecast.get("downloaded_at", pd.Series([None])).iloc[0]
        ),
        "forecast_age_hours": float(output["forecast_age_hours"].iloc[0]),
        "horizons": list(HORIZONS),
        "n_rows": int(len(output)),
        "n_cells": int(output["cell_id"].nunique()),
        "duration_seconds": round(time.perf_counter() - started_at, 3),
    }
    atomic_write_json(metadata, destination.with_suffix(".json"))
    manifest = {
        **metadata,
        "generated_at": generated_at.isoformat(),
        "forecast_run_source": metadata["forecast_run_source"],
        "forecast_coverage": coverage_report,
        "forecast_rows": int(len(forecast)),
        "state": state_report,
        "models": model_report,
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
    }
    atomic_write_json(
        manifest,
        manifest_path or destination.with_suffix(".manifest.json"),
    )
    LOGGER.info("Inferencia operativa guardada en %s (%s)", destination, quality)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    manifest_env = os.getenv("INFERENCE_MANIFEST_PATH")
    parser.add_argument("--issue-time", default=None, help="Instante local/ISO de emisión")
    parser.add_argument("--date", default=None, help="Compatibilidad: fecha local de emisión")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--forecast-dir", type=Path, default=DEFAULT_FORECAST_DIR)
    parser.add_argument(
        "--forecast-file",
        type=Path,
        default=None,
        help=(
            "Parquet de forecast horario ya descargado; se valida y reutiliza "
            "sin volver a consultar MeteoSIX (útil para pruebas locales)."
        ),
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(os.getenv("WEATHER_STATE_PATH", str(DEFAULT_STATE_PATH))),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(manifest_env) if manifest_env else None,
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.getenv("INFERENCE_LOCK_PATH", str(DEFAULT_LOCK_PATH))),
    )
    parser.add_argument("--no-lock", action="store_true", help="Desactiva el lock solo para pruebas")
    parser.add_argument("--no-stale", action="store_true", help="Falla si no existe forecast fresco")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    args = parse_args()
    forecast_df = None
    if args.forecast_file is not None:
        if not args.forecast_file.exists():
            raise FileNotFoundError(f"No existe el forecast indicado: {args.forecast_file}")
        LOGGER.info("Reutilizando forecast horario local: %s", args.forecast_file)
        forecast_df = pd.read_parquet(args.forecast_file)
    run_daily_inference_pipeline(
        target_date=args.date,
        issue_time=args.issue_time,
        output_path=args.output,
        forecast_df=forecast_df,
        grid_path=args.grid,
        forecast_dir=args.forecast_dir,
        model_dir=args.model_dir,
        weather_state_path=args.state,
        manifest_path=args.manifest,
        lock_path=args.lock,
        allow_stale=not args.no_stale,
        use_lock=not args.no_lock,
    )


if __name__ == "__main__":
    main()
