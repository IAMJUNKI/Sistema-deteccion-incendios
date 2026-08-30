"""Cliente y adaptador de AEMET OpenData para pruebas operativas.

AEMET OpenData no ofrece la misma malla que MeteoSIX: la predicción se
consulta por municipio y la predicción horaria cubre hasta 48 horas. Este
módulo usa la predicción diaria de AEMET para garantizar el horizonte de
72 horas del pipeline, y superpone la predicción horaria cuando está
disponible. El resultado debe etiquetarse como degradado y no como equivalente
a WRF 1 km.

La API de AEMET utiliza dos peticiones: la primera devuelve una URL temporal en
``datos`` y la segunda descarga el JSON real. El cliente conserva ambos
artefactos cuando se solicita un directorio de archivo.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.operational.artifacts import atomic_write_json

from .meteogalicia_forecast import (
    ForecastError,
    _as_utc_timestamp,
    _redact_sensitive_text,
)

LOGGER = logging.getLogger(__name__)
GALICIA_TZ = ZoneInfo("Europe/Madrid")
AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"
AEMET_API_VERSION = "OpenData-v2.0"

# Códigos INE de las capitales provinciales. Son suficientes para probar el
# circuito extremo a extremo, pero no para representar toda la variabilidad
# local de Galicia. En operación deben sustituirse por una lista más densa.
DEFAULT_MUNICIPALITIES = (
    ("15030", 43.3623, -8.4115),  # A Coruña
    ("27028", 43.0097, -7.5568),  # Lugo
    ("32054", 42.3367, -7.8639),  # Ourense
    ("36038", 42.4310, -8.6440),  # Pontevedra
)


@dataclass(frozen=True)
class AemetPoint:
    """Municipio AEMET utilizado como punto meteorológico representativo."""

    municipality_id: str
    lat: float
    lon: float


@dataclass(frozen=True)
class AemetForecastConfig:
    """Configuración del cliente AEMET OpenData."""

    api_key: str
    base_url: str = AEMET_BASE_URL
    timezone_name: str = "Europe/Madrid"
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    use_hourly_overlay: bool = True
    missing_precipitation_fallback: float | None = None


def _as_local_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(GALICIA_TZ)
    return timestamp.tz_convert(GALICIA_TZ)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number > -9990 else np.nan


def _first_value(record: Mapping[str, Any]) -> float:
    for key in ("value", "valor", "moduleValue", "module_value"):
        if key in record:
            return _to_float(record[key])
    return np.nan


def _records(raw: Any) -> list[Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        for key in ("dato", "data", "values"):
            if isinstance(raw.get(key), list):
                return [item for item in raw[key] if isinstance(item, Mapping)]
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _period_hours(period: Any) -> list[int]:
    """Convierte periodos AEMET como ``09``, ``00-06`` o ``0814`` a horas."""

    text = str(period or "").strip()
    if not text:
        return []
    if len(text) == 4 and text.isdigit():
        parts = [text[:2], text[2:]]
    elif text.isdigit():
        hour = int(text)
        return [hour] if 0 <= hour <= 24 else []
    if "-" in text:
        parts = text.split("-", 1)
    elif len(text) != 4 or not text.isdigit():
        return []
    try:
        start, end = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return []
    if end == 24:
        end = 24
    if not (0 <= start <= 24 and 0 <= end <= 24 and end > start):
        return []
    return list(range(start, end))


def _payload_entry(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, list):
        if not payload or not isinstance(payload[0], Mapping):
            raise ForecastError("La respuesta de datos AEMET no contiene un municipio.")
        return payload[0]
    if isinstance(payload, Mapping):
        return payload
    raise ForecastError("La respuesta de datos AEMET no es un objeto JSON válido.")


def _day_variables(day: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "temperature": _records(day.get("temperatura")),
        "humidity": _records(day.get("humedadRelativa")),
        "precipitation": _records(day.get("precipitacion")),
        "wind": _records(day.get("vientoAndRachaMax")),
    }


def _fill_series(values: list[float], fallback: float = np.nan) -> list[float]:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        series = series.interpolate(limit_direction="both")
    else:
        series[:] = fallback
    return series.tolist()


def _daily_fallback(day: Mapping[str, Any], variable: str) -> float:
    key = {
        "temperature": "temperatura",
        "humidity": "humedadRelativa",
    }.get(variable)
    if key is None or not isinstance(day.get(key), Mapping):
        return np.nan
    block = day[key]
    minimum = _to_float(block.get("minima"))
    maximum = _to_float(block.get("maxima"))
    values = [value for value in (minimum, maximum) if pd.notna(value)]
    return float(np.mean(values)) if values else np.nan


def parse_aemet_payload(
    payload: Any,
    point: AemetPoint,
    *,
    start_time: datetime | pd.Timestamp | str,
    end_time: datetime | pd.Timestamp | str,
    downloaded_at: datetime | pd.Timestamp | str | None = None,
    issued_at: datetime | pd.Timestamp | str | None = None,
    source_resolution: str = "daily_expansion",
    precipitation_fallback: float | None = None,
) -> pd.DataFrame:
    """Normaliza una respuesta diaria u horaria de AEMET.

    La predicción diaria puede contener valores por periodos. Para mantener
    el contrato horario interno, los valores de un periodo se distribuyen en
    sus horas y la precipitación se reparte uniformemente, conservando su suma
    diaria. Esta expansión es una aproximación de contingencia y queda
    registrada en ``source_resolution``.
    """

    entry = _payload_entry(payload)
    prediction = entry.get("prediccion") or {}
    days = prediction.get("dia") if isinstance(prediction, Mapping) else None
    if not isinstance(days, list):
        raise ForecastError("La respuesta AEMET no contiene prediccion.dia.")

    downloaded = _as_utc_timestamp(downloaded_at or pd.Timestamp.now(tz="UTC"))
    provider_issued = issued_at or entry.get("elaborado") or downloaded
    issued = _as_utc_timestamp(provider_issued)
    start = _as_utc_timestamp(start_time)
    end = _as_utc_timestamp(end_time)
    rows: list[dict[str, Any]] = []

    for day in days:
        if not isinstance(day, Mapping) or not day.get("fecha"):
            continue
        day_start = _as_local_timestamp(str(day["fecha"])[:19])
        values = {name: [np.nan] * 24 for name in ("temperature", "humidity", "precipitation", "wind")}
        variables = _day_variables(day)
        for variable, records in variables.items():
            for record in records:
                hours = _period_hours(record.get("periodo") or record.get("hora"))
                if not hours:
                    continue
                if variable == "wind":
                    speed = _to_float(record.get("velocidad"))
                    if pd.isna(speed) and "value" in record and not record.get("direccion"):
                        continue
                    value = speed
                else:
                    value = _first_value(record)
                if pd.isna(value):
                    continue
                divisor = len(hours) if variable == "precipitation" and len(hours) > 1 else 1
                for hour in hours:
                    if hour < 24:
                        values[variable][hour] = value / divisor

        values["temperature"] = _fill_series(
            values["temperature"], _daily_fallback(day, "temperature")
        )
        values["humidity"] = _fill_series(
            values["humidity"], _daily_fallback(day, "humidity")
        )
        values["wind"] = _fill_series(values["wind"])
        values["precipitation"] = _fill_series(
            values["precipitation"], precipitation_fallback
        )

        for hour in range(24):
            valid_local = day_start + pd.Timedelta(hours=hour)
            valid_time = valid_local.tz_convert("UTC")
            if not start <= valid_time <= end:
                continue
            rows.append(
                {
                    "point_id": f"aemet_{point.municipality_id}",
                    "municipality_id": point.municipality_id,
                    "forecast_lat": point.lat,
                    "forecast_lon": point.lon,
                    "valid_time": valid_time,
                    "forecast_run_at": issued,
                    "issued_at": issued,
                    "downloaded_at": downloaded,
                    "forecast_run_source": "provider",
                    "provider": "aemet",
                    "model": "AEMET-municipal",
                    "model_version": str(entry.get("version") or AEMET_API_VERSION),
                    "grid": "municipal",
                    "source_resolution": source_resolution,
                    "temperature_c": values["temperature"][hour],
                    "relative_humidity_pct": values["humidity"][hour],
                    "precipitation_mm": values["precipitation"][hour],
                    "wind_speed_kmh": values["wind"][hour],
                    "wind_dir_deg": np.nan,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ForecastError("La respuesta AEMET no cubre la ventana solicitada.")
    return frame.sort_values(["point_id", "valid_time"]).reset_index(drop=True)


class AemetClient:
    """Cliente AEMET OpenData con la doble petición ``datos`` + URL temporal."""

    def __init__(
        self,
        config: AemetForecastConfig,
        *,
        session: requests.Session | None = None,
        request_get: Callable[..., Any] | None = None,
    ) -> None:
        if not config.api_key:
            raise ForecastError("AEMET_API_KEY no está configurada.")
        self.config = config
        self.session = session or requests.Session()
        self._request_get = request_get or self.session.get

    def _get_json(self, url: str, *, follow_data_url: bool = True) -> Any:
        params = {"api_key": self.config.api_key}
        headers = {"Cache-Control": "no-cache"}
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._request_get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                envelope = response.json()
                if not isinstance(envelope, (Mapping, list)):
                    raise ForecastError("AEMET devolvió una respuesta no estructurada.")
                if follow_data_url and isinstance(envelope, Mapping) and envelope.get("datos"):
                    data_url = str(envelope["datos"])
                    return self._get_json(data_url, follow_data_url=False)
                if isinstance(envelope, Mapping) and envelope.get("estado") not in (None, 200):
                    status = envelope.get("estado")
                    description = envelope.get("descripcion") or envelope.get("mensaje")
                    detail = f" ({description})" if description else ""
                    raise ForecastError(f"AEMET devolvió estado {status}{detail}.")
                return envelope
            except (requests.RequestException, ValueError, ForecastError) as exc:
                safe_error = ForecastError(_redact_sensitive_text(exc))
                last_error = safe_error
                if attempt + 1 < self.config.max_retries:
                    delay = self.config.retry_backoff_seconds * (2**attempt)
                    LOGGER.warning(
                        "Error AEMET, reintento %s/%s en %.1fs: %s",
                        attempt + 1,
                        self.config.max_retries,
                        delay,
                        safe_error,
                    )
                    import time

                    time.sleep(delay)
        raise ForecastError("No se pudo descargar el forecast de AEMET.") from last_error

    def fetch_endpoint(self, endpoint: str) -> Any:
        """Descarga un recurso AEMET relativo a ``AEMET_BASE_URL``.

        El método mantiene la doble petición propia de OpenData y permite
        reutilizar el cliente para observaciones y climatología diaria, no
        solo para predicción municipal.
        """

        path = endpoint.lstrip("/")
        return self._get_json(f"{self.config.base_url.rstrip('/')}/{path}")

    def fetch_points(
        self,
        points: Sequence[AemetPoint],
        *,
        start_time: datetime | pd.Timestamp | str,
        end_time: datetime | pd.Timestamp | str,
        raw_dir: str | Path | None = None,
        downloaded_at: datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Descarga municipios configurados y devuelve el contrato horario común."""

        if not points:
            raise ValueError("Debe indicarse al menos un municipio AEMET.")
        downloaded = _as_utc_timestamp(downloaded_at or pd.Timestamp.now(tz="UTC"))
        frames: list[pd.DataFrame] = []
        raw_path = Path(raw_dir) if raw_dir else None
        if raw_path:
            raw_path.mkdir(parents=True, exist_ok=True)

        for point in points:
            endpoint_base = f"{self.config.base_url.rstrip('/')}/prediccion/especifica/municipio"
            daily_url = f"{endpoint_base}/diaria/{point.municipality_id}"
            daily_payload = self._get_json(daily_url)
            if raw_path:
                atomic_write_json(
                    daily_payload,
                    raw_path / f"aemet_{point.municipality_id}_daily.json",
                )
            daily = parse_aemet_payload(
                daily_payload,
                point,
                start_time=start_time,
                end_time=end_time,
                downloaded_at=downloaded,
                source_resolution="daily_expansion",
                precipitation_fallback=self.config.missing_precipitation_fallback,
            )

            if self.config.use_hourly_overlay:
                hourly_url = f"{endpoint_base}/horaria/{point.municipality_id}"
                try:
                    hourly_payload = self._get_json(hourly_url)
                    if raw_path:
                        atomic_write_json(
                            hourly_payload,
                            raw_path / f"aemet_{point.municipality_id}_hourly.json",
                        )
                    hourly = parse_aemet_payload(
                        hourly_payload,
                        point,
                        start_time=start_time,
                        end_time=end_time,
                        downloaded_at=downloaded,
                        source_resolution="hourly",
                    )
                    daily = _overlay_hourly(daily, hourly)
                except ForecastError as exc:
                    LOGGER.warning(
                        "No se pudo superponer la predicción horaria AEMET para %s: %s",
                        point.municipality_id,
                        exc,
                    )
            frames.append(daily)

        return pd.concat(frames, ignore_index=True).sort_values(
            ["point_id", "valid_time"]
        ).reset_index(drop=True)


def _overlay_hourly(daily: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Sustituye la expansión diaria por valores horarios coincidentes."""

    if hourly.empty:
        return daily
    result = daily.copy().set_index(["point_id", "valid_time"])
    overlay = hourly.set_index(["point_id", "valid_time"])
    common = result.index.intersection(overlay.index)
    columns = [
        "forecast_run_at",
        "issued_at",
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
        "wind_dir_deg",
    ]
    for column in columns:
        if column in overlay.columns:
            result.loc[common, column] = overlay.loc[common, column]
    result.loc[common, "source_resolution"] = "hourly"
    return result.reset_index()


def parse_municipality_env(value: str | None) -> tuple[AemetPoint, ...]:
    """Parsea ``codigo:lat:lon,...`` para mantener la selección auditable."""

    if not value or not value.strip():
        return tuple(AemetPoint(code, lat, lon) for code, lat, lon in DEFAULT_MUNICIPALITIES)
    points: list[AemetPoint] = []
    for item in value.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ForecastError(
                "AEMET_MUNICIPALITIES debe usar codigo:lat:lon,codigo:lat:lon."
            )
        try:
            points.append(AemetPoint(parts[0], float(parts[1]), float(parts[2])))
        except ValueError as exc:
            raise ForecastError(f"Municipio AEMET inválido: {item}") from exc
    if not points:
        raise ForecastError("AEMET_MUNICIPALITIES no contiene puntos válidos.")
    return tuple(points)
