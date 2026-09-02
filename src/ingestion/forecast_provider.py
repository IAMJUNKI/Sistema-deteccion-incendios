"""Contrato común para proveedores de previsión meteorológica.

Los clientes concretos (MeteoSIX y AEMET) pueden tener APIs y resoluciones
distintas, pero la inferencia solo debe consumir este resultado normalizado.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import pandas as pd

FORECAST_QUALITY_STATES = frozenset(
    {
        "fresh",
        "fresh_fallback",
        "fresh_aemet_degraded",
        "fresh_aemet",
        "fresh_aemet_proxy",
        "stale",
        "incomplete",
        "invalid",
        "unavailable",
    }
)


@dataclass(frozen=True)
class ForecastResult:
    """Forecast horario normalizado y su procedencia auditable."""

    hourly_forecast: pd.DataFrame
    provider: str
    model: str
    model_version: str
    grid: str
    issued_at: pd.Timestamp
    downloaded_at: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    coverage_percentage: float
    missing_variables: tuple[str, ...] = ()
    missing_hours: tuple[str, ...] = ()
    quality: str = "invalid"
    source_resolution: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quality not in FORECAST_QUALITY_STATES:
            raise ValueError(f"Estado de forecast no soportado: {self.quality}")
        if not 0.0 <= float(self.coverage_percentage) <= 100.0:
            raise ValueError("coverage_percentage debe estar entre 0 y 100.")


class ForecastProvider(Protocol):
    """Interfaz que debe implementar cualquier proveedor operativo."""

    def fetch(
        self,
        issue_time: datetime | pd.Timestamp | str,
        points: Sequence[object],
        horizons: tuple[int, ...] = (1, 2, 3),
    ) -> ForecastResult:
        ...
