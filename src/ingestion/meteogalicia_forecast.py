"""Ingesta y normalización del forecast numérico de MeteoGalicia v5.

El módulo separa tres responsabilidades que antes estaban mezcladas en la
inferencia diaria:

* acceso al endpoint numérico MeteoSIX v5;
* normalización a un contrato horario independiente del proveedor;
* validación, persistencia y asignación espacial a la rejilla de 1 km.

La API limita las consultas a 20 localizaciones. Por eso el cliente trabaja en
lotes y nunca realiza una llamada por cada celda de la rejilla final.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.operational.artifacts import atomic_write_json, atomic_write_parquet

LOGGER = logging.getLogger(__name__)

_SENSITIVE_QUERY_PARAMETER = re.compile(
    r"(?i)([?&](?:api[_-]?key|subscription-key|token)=)[^&\s)]+"
)


def _redact_sensitive_text(value: Any) -> str:
    """Remove credentials that may be embedded in provider error messages.

    HTTP client exceptions often include the complete request URL. Since both
    provider APIs authenticate through query parameters, logging the original
    exception could leak a credential into a terminal, log collector, or
    monitoring system.
    """

    return _SENSITIVE_QUERY_PARAMETER.sub(r"\1<redacted>", str(value))

GALICIA_TZ = ZoneInfo("Europe/Madrid")
DEFAULT_BASE_URL = "https://servizos.meteogalicia.gal/apiv5"
DEFAULT_MODEL = "WRF"
DEFAULT_GRID = "1km"
MAX_LOCATIONS_PER_REQUEST = 20
REQUIRED_HOURLY_COLUMNS = (
    "valid_time",
    "forecast_lat",
    "forecast_lon",
    "temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
)
API_VARIABLES = ("temperature", "relative_humidity", "precipitation_amount", "wind")
API_UNITS = {
    "temperature": "degC",
    "relative_humidity": "perc",
    "precipitation_amount": "lm2",
    # MeteoSIX v5 names the wind compound unit kmh_deg in its response.
    "wind": "kmh_deg",
}
API_DEFAULT_UNITS = API_UNITS.copy()


class ForecastError(RuntimeError):
    """Error controlado de descarga, parseo o validación del forecast."""


class ForecastValidationError(ForecastError):
    """El forecast no contiene cobertura suficiente para inferir."""


@dataclass(frozen=True)
class ForecastPoint:
    """Punto WGS84 que se consulta al proveedor."""

    point_id: str
    lat: float
    lon: float


@dataclass(frozen=True)
class ForecastConfig:
    """Configuración de acceso al endpoint MeteoSIX."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    grid: str = DEFAULT_GRID
    api_version: str = "v5"
    timezone_name: str = "Europe/Madrid"
    auto_adjust_position: bool = True
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0


def _as_utc_timestamp(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Convierte un instante a ``Timestamp`` UTC con zona horaria."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(GALICIA_TZ)
    return timestamp.tz_convert("UTC")


def _parse_number(value: Any) -> float:
    """Convierte valores JSON, tratando ``-9999`` como dato ausente."""

    if value is None or value == "":
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number) or number <= -9990:
        return np.nan
    return number


def _normalise_variable_name(name: Any) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _extract_time(value: Mapping[str, Any]) -> Any:
    """Extrae el instante de una observación MeteoSIX."""

    for key in ("timeInstant", "time_instant", "time", "valid_time"):
        if key in value:
            raw = value[key]
            if isinstance(raw, Mapping):
                return raw.get("timeInstant") or raw.get("value")
            return raw
    return None


def _extract_values(variable: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    values = variable.get("values")
    if values is None:
        return []
    if isinstance(values, Mapping):
        values = [values]
    return [item for item in values if isinstance(item, Mapping)]


def _extract_point_id(properties: Mapping[str, Any], index: int) -> str:
    value = properties.get("id") or properties.get("point_id")
    return str(value) if value is not None else f"point_{index}"


def _extract_provider_run_at(payload: Mapping[str, Any]) -> Any:
    """Best-effort extraction of a provider model-run timestamp.

    MeteoSIX payloads can expose this value under different metadata keys. If
    it is absent, the caller records the download time and marks the source.
    """

    keys = ("issued_at", "issuedAt", "referenceTime", "runTime", "modelRun")
    for key in keys:
        if key in payload and payload[key]:
            return payload[key]
    for container_name in ("metadata", "properties", "forecast"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            for key in keys:
                if key in container and container[key]:
                    return container[key]
    return None


def _extract_value_model_run(value: Mapping[str, Any]) -> Any:
    """Extrae el ``modelRun`` que MeteoSIX v5 incluye en cada hora."""

    for key in ("modelRun", "model_run", "runTime", "referenceTime"):
        raw = value.get(key)
        if isinstance(raw, Mapping):
            raw = raw.get("timeInstant") or raw.get("value")
        if raw:
            return raw
    return None


def _extract_model_version(payload: Mapping[str, Any]) -> str | None:
    """Extract a provider model version when the response exposes one."""

    keys = ("modelVersion", "model_version", "version")
    containers: list[Mapping[str, Any]] = [payload]
    for container_name in ("metadata", "properties", "forecast"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            containers.append(container)
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value:
                return str(value)
    return None


def parse_meteogalicia_payload(
    payload: Mapping[str, Any],
    *,
    issued_at: datetime | pd.Timestamp | str | None = None,
    provider_run_at: datetime | pd.Timestamp | str | None = None,
    downloaded_at: datetime | pd.Timestamp | str | None = None,
    provider: str = "meteogalicia",
    model: str = DEFAULT_MODEL,
    model_version: str | None = None,
    grid: str = DEFAULT_GRID,
) -> pd.DataFrame:
    """Aplana una respuesta GeoJSON de ``getNumericForecastInfo``.

    Args:
        payload: Respuesta JSON de MeteoSIX.
        issued_at: Legacy alias for the forecast run timestamp.
        provider_run_at: Timestamp supplied by the weather provider, when available.
        downloaded_at: Timestamp at which this system retrieved the payload.
        provider: Identificador de la fuente.
        model: Modelo numérico solicitado.
        grid: Malla numérica solicitada.

    Returns:
        DataFrame horario con una fila por punto e instante.

    Raises:
        ForecastError: Si la API devuelve una excepción global.
    """

    if "exception" in payload and payload["exception"]:
        exception = payload["exception"]
        message = exception.get("message", "Error desconocido") if isinstance(exception, Mapping) else exception
        raise ForecastError(f"MeteoSIX devolvió una excepción: {message}")

    features = payload.get("features")
    if features is None:
        raise ForecastError("La respuesta MeteoSIX no contiene 'features'.")

    downloaded = _as_utc_timestamp(downloaded_at or pd.Timestamp.now(tz="UTC"))
    provider_value = provider_run_at or issued_at or _extract_provider_run_at(payload)
    resolved_model_version = model_version or _extract_model_version(payload) or model
    rows: list[dict[str, Any]] = []
    variable_mapping = {
        "temperature": "temperature_c",
        "relative_humidity": "relative_humidity_pct",
        "precipitation_amount": "precipitation_mm",
        "wind": "wind_speed_kmh",
    }

    for feature_index, feature in enumerate(features):
        if not isinstance(feature, Mapping):
            continue
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        forecast_lon, forecast_lat = float(coordinates[0]), float(coordinates[1])
        properties = feature.get("properties") or {}
        if not isinstance(properties, Mapping):
            properties = {}
        if feature.get("exception"):
            LOGGER.warning(
                "MeteoSIX omitió el punto %s: %s",
                feature.get("exception", {}).get("code"),
                feature.get("exception", {}).get("message"),
            )
            continue

        days = properties.get("days") or []
        for day in days:
            if not isinstance(day, Mapping):
                continue
            variables = day.get("variables") or []
            for variable in variables:
                if not isinstance(variable, Mapping):
                    continue
                variable_name = _normalise_variable_name(variable.get("name"))
                target_column = variable_mapping.get(variable_name)
                if target_column is None:
                    continue
                for value in _extract_values(variable):
                    valid_time = _extract_time(value)
                    if valid_time is None:
                        continue
                    value_provider = _extract_value_model_run(value) or provider_value
                    forecast_run_source = (
                        "provider" if value_provider is not None else "download_time"
                    )
                    forecast_run = _as_utc_timestamp(value_provider or downloaded)
                    variable_model = str(variable.get("model") or model)
                    variable_grid = str(variable.get("grid") or grid)
                    row = {
                        "point_id": _extract_point_id(properties, feature_index),
                        "forecast_lat": forecast_lat,
                        "forecast_lon": forecast_lon,
                        "valid_time": _as_utc_timestamp(valid_time),
                        "forecast_run_at": forecast_run,
                        "issued_at": forecast_run,
                        "downloaded_at": downloaded,
                        "forecast_run_source": forecast_run_source,
                        "provider": provider,
                        "model": variable_model,
                        "model_version": resolved_model_version,
                        "grid": variable_grid,
                        "temperature_c": np.nan,
                        "relative_humidity_pct": np.nan,
                        "precipitation_mm": np.nan,
                        "wind_speed_kmh": np.nan,
                        "wind_dir_deg": np.nan,
                    }
                    if variable_name == "wind":
                        row["wind_speed_kmh"] = _parse_number(
                            value.get("moduleValue")
                            or value.get("module_value")
                            or value.get("value")
                        )
                        row["wind_dir_deg"] = _parse_number(
                            value.get("directionValue")
                            or value.get("direction_value")
                            or value.get("direction")
                        )
                    else:
                        row[target_column] = _parse_number(value.get("value"))
                    rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                *REQUIRED_HOURLY_COLUMNS,
                "point_id",
                "forecast_run_at",
                "issued_at",
                "downloaded_at",
                "forecast_run_source",
                "model_version",
                "grid",
            ]
        )

    df = pd.DataFrame(rows)
    # La API devuelve cada variable como una serie independiente. Se pivotan
    # aquí para obtener una única fila por punto e instante.
    index_columns = [
        "point_id",
        "forecast_lat",
        "forecast_lon",
        "valid_time",
        "forecast_run_at",
        "issued_at",
        "downloaded_at",
        "forecast_run_source",
        "provider",
        "model",
        "model_version",
        "grid",
    ]
    value_columns = [
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
        "wind_dir_deg",
    ]
    df = df.groupby(index_columns, dropna=False, as_index=False)[value_columns].first()
    df["horizon_hours"] = (
        pd.to_datetime(df["valid_time"], utc=True)
        - pd.to_datetime(df["issued_at"], utc=True)
    ).dt.total_seconds() / 3600.0
    return df.sort_values(["point_id", "valid_time"]).reset_index(drop=True)


def validate_hourly_forecast(
    forecast_df: pd.DataFrame,
    *,
    expected_start: datetime | pd.Timestamp | str,
    expected_end: datetime | pd.Timestamp | str,
    min_points: int = 1,
    require_hourly: bool = True,
    max_source_distance_km: float | None = None,
) -> None:
    """Valida columnas, cobertura temporal y variables meteorológicas."""

    missing = [column for column in REQUIRED_HOURLY_COLUMNS if column not in forecast_df.columns]
    if missing:
        raise ForecastValidationError(f"Faltan columnas meteorológicas: {missing}")
    if forecast_df.empty:
        raise ForecastValidationError("El forecast está vacío.")

    times = pd.to_datetime(forecast_df["valid_time"], utc=True, errors="coerce")
    if times.isna().any():
        raise ForecastValidationError("Hay timestamps inválidos en el forecast.")
    start = _as_utc_timestamp(expected_start)
    end = _as_utc_timestamp(expected_end)
    in_window = times.between(start, end, inclusive="both")
    if not in_window.any():
        raise ForecastValidationError("El forecast no cubre la ventana solicitada.")

    point_column = "cell_id" if "cell_id" in forecast_df.columns else (
        "point_id" if "point_id" in forecast_df.columns else "forecast_lat"
    )
    if forecast_df[point_column].nunique(dropna=True) < min_points:
        raise ForecastValidationError("El forecast no contiene suficientes puntos.")

    numeric_columns = [
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
    ]
    invalid = forecast_df[numeric_columns].isna().sum()
    if invalid.any():
        raise ForecastValidationError(f"Variables con valores ausentes: {invalid[invalid > 0].to_dict()}")

    if (forecast_df["relative_humidity_pct"] < 0).any() or (forecast_df["relative_humidity_pct"] > 100).any():
        raise ForecastValidationError("La humedad relativa está fuera del intervalo [0, 100].")
    if (forecast_df["precipitation_mm"] < 0).any() or (forecast_df["wind_speed_kmh"] < 0).any():
        raise ForecastValidationError("La precipitación o el viento contienen valores negativos.")
    if max_source_distance_km is not None and "source_distance_km" in forecast_df.columns:
        source_distance = pd.to_numeric(forecast_df["source_distance_km"], errors="coerce")
        if source_distance.isna().any() or (source_distance > max_source_distance_km).any():
            raise ForecastValidationError(
                "Hay celdas demasiado alejadas del punto meteorológico asignado: "
                f"máximo permitido={max_source_distance_km} km."
            )

    if require_hourly:
        expected_hours = pd.date_range(start=start, end=end, freq="h", tz="UTC")
        actual_hours = pd.DatetimeIndex(times[in_window].unique()).sort_values()
        missing_hours = expected_hours.difference(actual_hours)
        if len(missing_hours):
            raise ForecastValidationError(
                f"Faltan {len(missing_hours)} horas del forecast; primera ausente: {missing_hours[0]}"
            )


def sample_provider_points(
    grid_df: pd.DataFrame,
    *,
    resolution_km: float = 4.0,
) -> pd.DataFrame:
    """Reduce la rejilla de 1 km a puntos de consulta aproximados de 4 km.

    MeteoSIX ajusta el punto a la malla numérica solicitada. Agrupar antes de
    consultar evita lanzar decenas de miles de peticiones redundantes.
    """

    required = {"cell_id", "lat_centroid", "lon_centroid"}
    missing = required.difference(grid_df.columns)
    if missing:
        raise ValueError(f"La rejilla no contiene: {sorted(missing)}")
    points = grid_df[["cell_id", "lat_centroid", "lon_centroid"]].copy()
    lat_step = resolution_km / 111.32
    points["_lat_bin"] = np.floor(points["lat_centroid"] / lat_step).astype(int)
    lon_step = resolution_km / (111.32 * np.cos(np.deg2rad(points["lat_centroid"].mean())))
    points["_lon_bin"] = np.floor(points["lon_centroid"] / lon_step).astype(int)
    sampled = (
        points.sort_values("cell_id")
        .drop_duplicates(["_lat_bin", "_lon_bin"])
        .rename(columns={"cell_id": "representative_cell_id"})
        .reset_index(drop=True)
    )
    sampled["point_id"] = [f"provider_{i:05d}" for i in range(len(sampled))]
    return sampled[["point_id", "representative_cell_id", "lat_centroid", "lon_centroid"]]


def assign_forecast_to_grid(
    forecast_df: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    """Asigna cada predicción horaria al centroide de celda más cercano."""

    required_grid = {"cell_id", "lat_centroid", "lon_centroid"}
    if not required_grid.issubset(grid_df.columns):
        raise ValueError(f"La rejilla debe contener {sorted(required_grid)}")
    if forecast_df.empty:
        raise ForecastValidationError("No se puede asignar un forecast vacío.")

    source = forecast_df.dropna(subset=["forecast_lat", "forecast_lon"]).copy()
    if source.empty:
        raise ForecastValidationError("El forecast no contiene coordenadas válidas.")

    source_points = source[["forecast_lat", "forecast_lon"]].drop_duplicates().reset_index(drop=True)
    grid_coords = grid_df[["lat_centroid", "lon_centroid"]].to_numpy(float)
    source_coords = source_points.to_numpy(float)
    # Distancia equirectangular suficiente para seleccionar el punto más cercano
    # dentro de Galicia. Se calcula una sola vez, no una vez por hora.
    mean_lat = np.deg2rad(grid_coords[:, 0].mean())
    source_xy = source_coords * np.array([1.0, np.cos(mean_lat)])
    grid_xy = grid_coords * np.array([1.0, np.cos(mean_lat)])
    nearest_by_cell = []
    distance_by_cell_km = []
    for start in range(0, len(grid_xy), 5000):
        chunk = grid_xy[start : start + 5000]
        distances = ((chunk[:, None, :] - source_xy[None, :, :]) ** 2).sum(axis=2)
        nearest = distances.argmin(axis=1)
        nearest_by_cell.extend(nearest)
        distance_by_cell_km.extend(np.sqrt(distances[np.arange(len(chunk)), nearest]) * 111.32)

    source_points["_source_key"] = (
        source_points["forecast_lat"].round(7).astype(str)
        + ":"
        + source_points["forecast_lon"].round(7).astype(str)
    )
    grid_mapping = grid_df[["cell_id", "lat_centroid", "lon_centroid"]].copy()
    grid_mapping["_source_key"] = source_points.iloc[nearest_by_cell]["_source_key"].to_numpy()
    grid_mapping["source_distance_km"] = distance_by_cell_km
    source["_source_key"] = (
        source["forecast_lat"].round(7).astype(str)
        + ":"
        + source["forecast_lon"].round(7).astype(str)
    )
    mapped = source.merge(grid_mapping, on="_source_key", how="inner", suffixes=("", "_grid"))
    if mapped.empty:
        raise ForecastValidationError("No se pudo asignar ninguna hora a la rejilla.")
    return mapped.drop(columns=["_source_key"])


def _format_api_time(
    value: datetime | pd.Timestamp | str,
    *,
    timezone_name: str = "Europe/Madrid",
) -> str:
    """Format a query instant in the API display timezone.

    MeteoSIX v5 receives ``startTime`` and ``endTime`` without an offset and
    interprets them using the timezone supplied in ``tz``. Converting to UTC
    and then dropping the offset would shift summer dates by two hours.
    """

    timestamp = pd.Timestamp(value)
    target_timezone = ZoneInfo(timezone_name)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(target_timezone)
    else:
        timestamp = timestamp.tz_convert(target_timezone)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S")


class MeteoGaliciaClient:
    """Cliente con reintentos para ``getNumericForecastInfo``."""

    def __init__(
        self,
        config: ForecastConfig,
        *,
        session: requests.Session | None = None,
        request_get: Callable[..., Any] | None = None,
    ) -> None:
        if not config.api_key:
            raise ForecastError("METEOGALICIA_API_KEY no está configurada.")
        self.config = config
        self.session = session or requests.Session()
        self._request_get = request_get or self.session.get

    def _request_payload(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/getNumericForecastInfo"
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._request_get(url, params=params, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise ForecastError("La respuesta MeteoSIX no es un objeto JSON.")
                return payload
            except (requests.RequestException, ValueError, ForecastError) as exc:
                safe_error = ForecastError(_redact_sensitive_text(exc))
                last_error = safe_error
                if attempt + 1 < self.config.max_retries:
                    delay = self.config.retry_backoff_seconds * (2**attempt)
                    LOGGER.warning(
                        "Error MeteoSIX, reintento %s/%s en %.1fs: %s",
                        attempt + 1,
                        self.config.max_retries,
                        delay,
                        safe_error,
                    )
                    import time

                    time.sleep(delay)
        raise ForecastError("No se pudo descargar el forecast de MeteoGalicia.") from last_error

    def fetch_points(
        self,
        points: Sequence[ForecastPoint],
        *,
        start_time: datetime | pd.Timestamp | str,
        end_time: datetime | pd.Timestamp | str,
        raw_dir: str | Path | None = None,
        issued_at: datetime | pd.Timestamp | str | None = None,
        provider_run_at: datetime | pd.Timestamp | str | None = None,
        downloaded_at: datetime | pd.Timestamp | str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Download and normalize a set of points in batches of twenty.

        Args:
            progress_callback: Optional callback receiving ``(completed, total)``
                after each successful batch. It is intended for CLI progress
                reporting and does not affect the downloaded data.
        """

        if not points:
            raise ValueError("Debe indicarse al menos un punto de forecast.")
        downloaded = _as_utc_timestamp(downloaded_at or pd.Timestamp.now(tz="UTC"))
        variables = API_VARIABLES
        frames: list[pd.DataFrame] = []
        raw_path = Path(raw_dir) if raw_dir else None
        if raw_path:
            raw_path.mkdir(parents=True, exist_ok=True)
        total_batches = (len(points) + MAX_LOCATIONS_PER_REQUEST - 1) // MAX_LOCATIONS_PER_REQUEST

        for batch_index in range(0, len(points), MAX_LOCATIONS_PER_REQUEST):
            batch = points[batch_index : batch_index + MAX_LOCATIONS_PER_REQUEST]
            coords = ";".join(f"{point.lon:.6f},{point.lat:.6f}" for point in batch)
            params = {
                "coords": coords,
                "variables": ",".join(variables),
                "models": ",".join([self.config.model] * len(variables)),
                "grids": ",".join([self.config.grid] * len(variables)),
                "format": "application/json",
                "lang": "es",
                "tz": self.config.timezone_name,
                "autoAdjustPosition": str(self.config.auto_adjust_position).lower(),
                "startTime": _format_api_time(
                    start_time,
                    timezone_name=self.config.timezone_name,
                ),
                "endTime": _format_api_time(
                    end_time,
                    timezone_name=self.config.timezone_name,
                ),
                "API_KEY": self.config.api_key,
            }
            payload = self._request_payload(params)
            payload_run_at = provider_run_at or _extract_provider_run_at(payload)
            payload_model_version = _extract_model_version(payload) or self.config.model
            if raw_path:
                filename = f"forecast_{downloaded.strftime('%Y%m%dT%H%M%SZ')}_batch_{batch_index // MAX_LOCATIONS_PER_REQUEST:04d}.json"
                raw_file = raw_path / filename
                atomic_write_json(payload, raw_file)
                metadata = {
                    "provider_run_at": (
                        _as_utc_timestamp(payload_run_at).isoformat()
                        if payload_run_at is not None
                        else None
                    ),
                    "downloaded_at": downloaded.isoformat(),
                    "forecast_run_source": "provider" if payload_run_at is not None else "download_time",
                    "valid_start": _as_utc_timestamp(start_time).isoformat(),
                    "valid_end": _as_utc_timestamp(end_time).isoformat(),
                    "provider": "meteogalicia",
                    "api_version": self.config.api_version,
                    "model": self.config.model,
                    "model_version": payload_model_version,
                    "grid": self.config.grid,
                    "variables": list(variables),
                    "units": API_DEFAULT_UNITS,
                    "units_requested": None,
                    "units_source": "meteosix_v5_defaults",
                    "auto_adjust_position": self.config.auto_adjust_position,
                    "point_ids": [point.point_id for point in batch],
                    "request_url": f"{self.config.base_url.rstrip('/')}/getNumericForecastInfo",
                }
                atomic_write_json(metadata, raw_file.with_suffix(".metadata.json"))
            frame = parse_meteogalicia_payload(
                payload,
                provider_run_at=payload_run_at,
                downloaded_at=downloaded,
                model=self.config.model,
                model_version=payload_model_version,
                grid=self.config.grid,
            )
            if not frame.empty:
                frames.append(frame)
            if progress_callback is not None:
                progress_callback(
                    batch_index // MAX_LOCATIONS_PER_REQUEST + 1,
                    total_batches,
                )

        if not frames:
            raise ForecastValidationError("MeteoGalicia no devolvió datos para ningún punto.")
        result = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["forecast_lat", "forecast_lon", "valid_time", "temperature_c", "relative_humidity_pct", "precipitation_mm", "wind_speed_kmh"]
        ).reset_index(drop=True)
        if result["forecast_run_at"].nunique(dropna=False) != 1:
            raise ForecastValidationError(
                "Los lotes de MeteoGalicia devuelven emisiones meteorológicas distintas."
            )
        return result


def save_forecast_parquet(forecast_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Guarda un forecast normalizado en Parquet."""

    return atomic_write_parquet(forecast_df, output_path)


def load_latest_forecast(
    directory: str | Path,
    *,
    required_start: datetime | pd.Timestamp | str,
    required_end: datetime | pd.Timestamp | str,
) -> tuple[pd.DataFrame, pd.Timestamp, bool]:
    """Carga el último forecast archivado que cubre la ventana requerida.

    Returns:
        ``(forecast, issued_at, stale)``. El resultado se considera stale porque
        procede de un archivo anterior a la ejecución actual; el llamador decide
        si debe mostrarlo como aviso operativo.
    """

    files = sorted(Path(directory).glob("forecast_*.parquet"), reverse=True)
    if not files:
        raise ForecastError(f"No hay forecasts archivados en {directory}.")
    start = _as_utc_timestamp(required_start)
    end = _as_utc_timestamp(required_end)
    for path in files:
        try:
            frame = pd.read_parquet(path)
            if frame.empty or "forecast_run_at" not in frame.columns:
                continue
            times = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
            if times.min() <= start and times.max() >= end:
                issued = _as_utc_timestamp(frame["forecast_run_at"].iloc[0])
                # Mantener la ruta fuera de las columnas evita contaminar el
                # contrato horario, pero permite que la inferencia enlace el
                # fallback stale con su archivo original en el manifest.
                frame.attrs["archive_path"] = str(path)
                return frame, issued, True
        except (OSError, ValueError, ForecastError) as exc:
            LOGGER.warning("No se pudo leer forecast archivado %s: %s", path, exc)
    raise ForecastError("No existe un forecast archivado que cubra T+1/T+2/T+3.")
