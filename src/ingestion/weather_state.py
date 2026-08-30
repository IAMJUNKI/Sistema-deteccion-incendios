"""Estado meteorológico reciente necesario para la inferencia operativa.

El histórico ERA5-Land sirve para entrenar y arrancar el sistema, pero no debe
ser leído completo en cada ejecución diaria. Este módulo mantiene un estado
diario compacto por celda y corta la ejecución si no existe cobertura reciente
suficiente. Es preferible fallar explícitamente a convertir una memoria de
sequedad desconocida en ceros.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.features.operational_features import calculate_vpd
from src.operational.artifacts import atomic_write_parquet

GALICIA_TZ = ZoneInfo("Europe/Madrid")
STATE_COLUMNS = [
    "cell_id",
    "fecha",
    "tmax_vc",
    "rhmin_vc",
    "vmax_vc",
    "prec_dia",
    "vpd_vc",
    "source",
    "coverage_hours",
    "state_as_of",
]


class WeatherStateError(RuntimeError):
    """The recent weather state cannot support a safe inference."""


def _issue_date(issue_time: datetime | pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(issue_time)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(GALICIA_TZ)
    return timestamp.tz_convert(GALICIA_TZ).normalize().tz_localize(None)


def normalise_weather_state(frame: pd.DataFrame, *, source: str = "observation") -> pd.DataFrame:
    """Normalize daily observations/analyses to the state-store contract."""

    aliases = {
        "date": "fecha",
        "tmax": "tmax_vc",
        "rh_min": "rhmin_vc",
        "rhmin": "rhmin_vc",
        "wind_speed": "vmax_vc",
        "vmax": "vmax_vc",
        "precip": "prec_dia",
        "precip_mm": "prec_dia",
    }
    state = frame.copy()
    for old, new in aliases.items():
        if old in state.columns and new not in state.columns:
            state = state.rename(columns={old: new})

    required = {"cell_id", "fecha", "tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia"}
    missing = required.difference(state.columns)
    if missing:
        raise WeatherStateError(f"Faltan columnas del estado meteorológico: {sorted(missing)}")

    state["fecha"] = pd.to_datetime(state["fecha"], errors="coerce").dt.tz_localize(None)
    state = state.dropna(subset=["cell_id", "fecha"]).copy()
    if "vpd_vc" not in state.columns:
        state["vpd_vc"] = calculate_vpd(state["tmax_vc"], state["rhmin_vc"])
    if "source" not in state.columns:
        state["source"] = source
    if "coverage_hours" not in state.columns:
        state["coverage_hours"] = 24.0
    if "state_as_of" not in state.columns:
        state["state_as_of"] = pd.Timestamp.now(tz="UTC")

    state = state[STATE_COLUMNS].copy()
    numeric = [
        "tmax_vc",
        "rhmin_vc",
        "vmax_vc",
        "prec_dia",
        "vpd_vc",
        "coverage_hours",
    ]
    state[numeric] = state[numeric].apply(pd.to_numeric, errors="coerce")
    state["state_as_of"] = pd.to_datetime(state["state_as_of"], utc=True, errors="coerce")
    state = state.sort_values(["cell_id", "fecha"]).drop_duplicates(
        ["cell_id", "fecha"], keep="last"
    )
    return state.reset_index(drop=True)


def merge_weather_state(
    existing: pd.DataFrame | None,
    updates: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | str,
    retention_days: int = 60,
) -> pd.DataFrame:
    """Upsert daily rows and retain only the recent operational window."""

    as_of_date = _issue_date(as_of)
    current = normalise_weather_state(existing, source="state") if existing is not None else pd.DataFrame(columns=STATE_COLUMNS)
    incoming = normalise_weather_state(updates)
    merged = pd.concat([current, incoming], ignore_index=True)
    merged = merged.sort_values(["cell_id", "fecha", "state_as_of"])
    merged = merged.drop_duplicates(["cell_id", "fecha"], keep="last")
    first_date = as_of_date - pd.Timedelta(days=retention_days)
    merged = merged[merged["fecha"].between(first_date, as_of_date, inclusive="both")]
    return merged.sort_values(["cell_id", "fecha"]).reset_index(drop=True)


def validate_weather_state(
    state: pd.DataFrame,
    *,
    issue_time: datetime | pd.Timestamp | str,
    required_cells: Iterable[object],
    history_days: int = 30,
    minimum_coverage_ratio: float = 1.0,
    minimum_daily_coverage_hours: float = 18.0,
) -> dict[str, object]:
    """Validate freshness and per-cell coverage before feature generation."""

    normalised = normalise_weather_state(state, source="state")
    required_numeric = ["tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia", "vpd_vc"]
    if normalised["state_as_of"].isna().any():
        raise WeatherStateError("El estado contiene timestamps state_as_of inválidos.")
    missing_values = normalised[required_numeric].isna().sum()
    if missing_values.any():
        raise WeatherStateError(
            "El estado meteorológico contiene valores ausentes: "
            f"{missing_values[missing_values > 0].to_dict()}"
        )
    if (normalised["rhmin_vc"] < 0).any() or (normalised["rhmin_vc"] > 100).any():
        raise WeatherStateError("La humedad relativa del estado está fuera del intervalo [0, 100].")
    if (normalised[["vmax_vc", "prec_dia", "coverage_hours"]] < 0).any().any():
        raise WeatherStateError("El estado contiene viento, precipitación o cobertura negativos.")
    if (normalised["coverage_hours"] > 24).any():
        raise WeatherStateError("La cobertura diaria del estado no puede superar 24 horas.")
    if (normalised["coverage_hours"] < minimum_daily_coverage_hours).any():
        raise WeatherStateError(
            "El estado meteorológico contiene días con cobertura horaria insuficiente: "
            f"se requieren al menos {minimum_daily_coverage_hours:g} horas."
        )
    if (normalised["tmax_vc"] < -50).any() or (normalised["tmax_vc"] > 70).any():
        raise WeatherStateError("La temperatura máxima del estado está fuera de un rango físico válido.")
    cells = set(required_cells)
    issue_date = _issue_date(issue_time)
    cutoff = issue_date - pd.Timedelta(days=1)
    first_date = cutoff - pd.Timedelta(days=history_days - 1)
    required_dates = pd.date_range(first_date, cutoff, freq="D")
    window = normalised[normalised["fecha"].between(first_date, cutoff, inclusive="both")]

    by_cell = window.groupby("cell_id")["fecha"].nunique()
    coverage = by_cell.reindex(cells, fill_value=0)
    ratio = coverage / len(required_dates)
    missing_cells = coverage[ratio < minimum_coverage_ratio].index.tolist()
    latest_date = window["fecha"].max() if not window.empty else None
    if missing_cells:
        raise WeatherStateError(
            "El estado meteorológico no tiene cobertura suficiente: "
            f"{len(missing_cells)} celdas por debajo de {minimum_coverage_ratio:.0%}; "
            f"última fecha disponible={latest_date}. "
            "Actualiza observaciones antes de inferir."
        )

    return {
        "required_cells": len(cells),
        "covered_cells": int((ratio >= minimum_coverage_ratio).sum()),
        "history_days": history_days,
        "latest_date": latest_date,
        "minimum_cell_coverage_ratio": float(ratio.min()) if len(ratio) else 0.0,
        "source_counts": normalised["source"].value_counts(dropna=False).to_dict(),
    }


def load_weather_state(
    path: str | Path,
    *,
    issue_time: datetime | pd.Timestamp | str,
    required_cells: Iterable[object],
    history_days: int = 30,
) -> pd.DataFrame:
    """Load and validate the recent state required by an operational run."""

    state_path = Path(path)
    if not state_path.exists():
        raise WeatherStateError(
            f"No existe el estado meteorológico reciente: {state_path}. "
            "Carga observaciones diarias antes de ejecutar producción."
        )
    try:
        state = pd.read_parquet(state_path)
    except (OSError, ValueError) as exc:
        raise WeatherStateError(f"No se pudo leer el estado meteorológico {state_path}.") from exc
    validate_weather_state(
        state,
        issue_time=issue_time,
        required_cells=required_cells,
        history_days=history_days,
    )
    return normalise_weather_state(state, source="state")


def save_weather_state(state: pd.DataFrame, path: str | Path) -> Path:
    """Persist a validated state snapshot atomically."""

    return atomic_write_parquet(normalise_weather_state(state, source="state"), path)
