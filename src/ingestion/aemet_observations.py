"""Ingesta de observaciones diarias AEMET y asignación a la rejilla de Galicia.

AEMET OpenData ofrece dos recursos complementarios: observaciones horarias de
las últimas 12 horas y valores climatológicos diarios para un rango de fechas.
Para inicializar el estado operativo usamos el segundo recurso. AEMET limita
cada rango diario a 15 días, así que una ventana de 30 días se divide en dos
peticiones lógicas (la API sigue el patrón ``datos`` + URL temporal). Las
estaciones se interpolan a la rejilla mediante IDW sobre los cuatro vecinos más
cercanos.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.operational_features import calculate_vpd
from src.operational.artifacts import atomic_write_json

from .aemet_forecast import AEMET_API_VERSION, AEMET_BASE_URL, AemetClient, AemetForecastConfig
from .meteogalicia_forecast import ForecastError

GALICIA_TZ = ZoneInfo("Europe/Madrid")
GALICIA_BOUNDS = {
    "min_lat": 41.7,
    "max_lat": 43.9,
    "min_lon": -9.5,
    "max_lon": -6.5,
}
AEMET_DAILY_HISTORY_ENDPOINT = (
    "valores/climatologicos/diarios/datos/fechaini/{start}/fechafin/{end}/todasestaciones/"
)
AEMET_STATIONS_ENDPOINT = "valores/climatologicos/inventarioestaciones/todasestaciones/"
AEMET_CURRENT_OBSERVATIONS_ENDPOINT = "observacion/convencional/todas"
AEMET_MAX_DAILY_RANGE_DAYS = 15
DAILY_WEATHER_COLUMNS = ["cell_id", "fecha", "tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia"]
CANONICAL_DAILY_WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "temperature_max_12_18h",
    "relative_humidity_mean",
    "relative_humidity_min",
    "relative_humidity_min_12_18h",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_speed_max_12_18h",
    "vpd_mean",
    "vpd_max_12_18h",
    "precipitation_sum",
]


class AemetObservationError(ForecastError):
    """La observación AEMET no permite construir un estado seguro."""


@dataclass(frozen=True)
class AemetObservationConfig:
    """Configuración de la ingesta histórica y diaria AEMET."""

    api_key: str
    base_url: str = AEMET_BASE_URL
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    idw_neighbors: int = 4


def _normalise_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _pick(record: dict[str, Any], *aliases: str) -> Any:
    values = {_normalise_key(key): value for key, value in record.items()}
    for alias in aliases:
        if _normalise_key(alias) in values:
            return values[_normalise_key(alias)]
    return None


def _numeric(value: Any, *, trace_as_zero: bool = False) -> float:
    if value is None or isinstance(value, (dict, list, tuple)):
        return np.nan
    text = str(value).strip().replace(",", ".")
    if not text or text.lower() in {"-", "na", "nan", "n/a", "null"}:
        return np.nan
    if trace_as_zero and text.lower() in {"ip", "tr", "traza", "trace"}:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return np.nan
    return number if np.isfinite(number) and number > -9990 else np.nan


def parse_aemet_coordinate(value: Any) -> float:
    """Convierte coordenadas decimales o DMS AEMET como ``433621N``."""

    if value is None:
        return np.nan
    text = str(value).strip().replace(",", ".")
    if not text:
        return np.nan
    try:
        number = float(text)
        if np.isfinite(number):
            return number
    except ValueError:
        pass
    match = re.fullmatch(r"(\d{4,7})([NSEW])", text.upper())
    if not match:
        return np.nan
    body, hemisphere = match.groups()
    degree_length = 2 if hemisphere in {"N", "S"} or len(body) == 6 else 3
    if len(body) < degree_length + 4:
        return np.nan
    degrees = float(body[:degree_length])
    minutes = float(body[degree_length : degree_length + 2])
    seconds = float(body[degree_length + 2 :])
    result = degrees + minutes / 60.0 + seconds / 3600.0
    return -result if hemisphere in {"S", "W"} else result


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("datos", "data", "items", "results", "estaciones"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _station_id(record: dict[str, Any]) -> str:
    value = _pick(record, "indicativo", "idema", "estacion", "station_id", "id")
    return str(value).strip() if value is not None else ""


def _inventory_by_station(payload: Any) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for record in _records(payload):
        station_id = _station_id(record)
        if station_id:
            inventory[station_id] = record
    return inventory


def _date_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(GALICIA_TZ).dt.normalize().dt.tz_localize(None)


def split_aemet_date_range(
    start_date: date | str,
    end_date: date | str,
    *,
    maximum_days: int = AEMET_MAX_DAILY_RANGE_DAYS,
) -> list[tuple[date, date]]:
    """Divide un rango inclusivo en bloques aceptados por AEMET."""

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise AemetObservationError("end_date no puede ser anterior a start_date.")
    if maximum_days <= 0:
        raise AemetObservationError("maximum_days debe ser positivo.")

    chunks: list[tuple[date, date]] = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=maximum_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def normalise_aemet_daily_payload(
    daily_payload: Any,
    *,
    inventory_payload: Any | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Normaliza la climatología diaria AEMET a una tabla por estación."""

    raw_records = _records(daily_payload)
    inventory = _inventory_by_station(inventory_payload) if inventory_payload is not None else {}
    rows: list[dict[str, Any]] = []
    for record in raw_records:
        station_id = _station_id(record)
        station_meta = inventory.get(station_id, {})
        latitude = parse_aemet_coordinate(
            _pick(record, "latitud", "latitude", "lat")
            or _pick(station_meta, "latitud", "latitude", "lat")
        )
        longitude = parse_aemet_coordinate(
            _pick(record, "longitud", "longitude", "lon")
            or _pick(station_meta, "longitud", "longitude", "lon")
        )
        if not station_id or pd.isna(latitude) or pd.isna(longitude):
            continue
        if not (
            GALICIA_BOUNDS["min_lat"] <= latitude <= GALICIA_BOUNDS["max_lat"]
            and GALICIA_BOUNDS["min_lon"] <= longitude <= GALICIA_BOUNDS["max_lon"]
        ):
            continue

        temperature = _numeric(_pick(record, "tmax", "temp_max", "temperatura_maxima"))
        humidity_min = _numeric(
            _pick(record, "hrMin", "hrmin", "humedad_minima", "relative_humidity_min")
        )
        humidity_source = "observed_minimum"
        if pd.isna(humidity_min):
            humidity_min = _numeric(_pick(record, "hrMedia", "hrmedia", "humedad_media"))
            humidity_source = "daily_mean_proxy"
        wind = _numeric(_pick(record, "racha", "vmax", "wind_gust", "velmedia"))
        wind_source = "maximum_gust" if pd.notna(_pick(record, "racha", "vmax")) else "mean_wind"
        precipitation = _numeric(
            _pick(record, "prec", "precipitacion", "precipitation", "prec_mm"),
            trace_as_zero=True,
        )
        rows.append(
            {
                "station_id": station_id,
                "fecha": _pick(record, "fecha", "date", "valid_date"),
                "lat": latitude,
                "lon": longitude,
                "station_name": _pick(record, "nombre", "name")
                or _pick(station_meta, "nombre", "name"),
                "station_province": _pick(record, "provincia", "province")
                or _pick(station_meta, "provincia", "province"),
                "tmax_vc": temperature,
                "rhmin_vc": humidity_min,
                "vmax_vc": wind,
                "prec_dia": precipitation,
                "rhmin_source": humidity_source,
                "wind_source": wind_source,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AemetObservationError("AEMET no devolvió estaciones de Galicia con coordenadas.")
    frame["fecha"] = _date_series(frame["fecha"])
    if start_date is not None:
        frame = frame[frame["fecha"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["fecha"] <= pd.Timestamp(end_date)]
    frame = frame.dropna(subset=["fecha"]).drop_duplicates(["station_id", "fecha"], keep="last")
    if frame.empty:
        raise AemetObservationError("AEMET no devolvió fechas dentro del rango solicitado.")
    return frame.sort_values(["fecha", "station_id"]).reset_index(drop=True)


def normalise_aemet_hourly_payload(
    hourly_payload: Any,
    *,
    inventory_payload: Any | None = None,
    start_time: Any | None = None,
    end_time: Any | None = None,
) -> pd.DataFrame:
    """Normaliza ``observacion/convencional/todas`` a observaciones horarias.

    AEMET entrega en este endpoint una fotografía de las observaciones recibidas
    durante las últimas horas. La tabla se conserva por estación e instante,
    porque no es seguro convertir una única fotografía en un día completo. El
    proceso local acumula varias fotografías y solo agrega un día cuando alcanza
    la cobertura mínima configurada.

    ``fint`` es el instante de observación; los timestamps sin zona se
    interpretan como UTC, que es la convención del producto de observación de
    AEMET. Las coordenadas se toman del propio registro y, como respaldo, del
    inventario climatológico.
    """

    inventory = _inventory_by_station(inventory_payload) if inventory_payload is not None else {}
    rows: list[dict[str, Any]] = []
    for record in _records(hourly_payload):
        station_id = _station_id(record)
        station_meta = inventory.get(station_id, {})
        latitude = parse_aemet_coordinate(
            _pick(record, "lat", "latitud", "latitude")
            or _pick(station_meta, "lat", "latitud", "latitude")
        )
        longitude = parse_aemet_coordinate(
            _pick(record, "lon", "longitud", "longitude")
            or _pick(station_meta, "lon", "longitud", "longitude")
        )
        instant = _pick(record, "fint", "fechaHora", "valid_time", "timestamp", "fecha")
        if not station_id or pd.isna(latitude) or pd.isna(longitude) or instant in (None, ""):
            continue
        if not (
            GALICIA_BOUNDS["min_lat"] <= latitude <= GALICIA_BOUNDS["max_lat"]
            and GALICIA_BOUNDS["min_lon"] <= longitude <= GALICIA_BOUNDS["max_lon"]
        ):
            continue

        valid_time = pd.to_datetime(instant, errors="coerce", utc=True)
        if pd.isna(valid_time):
            continue
        rows.append(
            {
                "station_id": station_id,
                "valid_time": valid_time,
                "lat": latitude,
                "lon": longitude,
                "station_name": _pick(record, "ubi", "nombre", "name")
                or _pick(station_meta, "ubi", "nombre", "name"),
                "temperature_c": _numeric(_pick(record, "ta", "temperature", "temperatura")),
                "relative_humidity_pct": _numeric(
                    _pick(record, "hr", "humedad", "relative_humidity")
                ),
                "wind_speed_kmh": _numeric(
                    _pick(record, "vmax", "vv", "velocidad", "wind_speed")
                ),
                "precipitation_mm": _numeric(
                    _pick(record, "prec", "precipitacion", "precipitation"),
                    trace_as_zero=True,
                ),
                "source": "aemet_current_observation",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AemetObservationError(
            "AEMET no devolvió observaciones horarias válidas de estaciones de Galicia."
        )
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["valid_time"])
    if start_time is not None:
        start = pd.to_datetime(start_time, utc=True)
        frame = frame[frame["valid_time"] >= start]
    if end_time is not None:
        end = pd.to_datetime(end_time, utc=True)
        frame = frame[frame["valid_time"] <= end]
    frame = frame.drop_duplicates(["station_id", "valid_time"], keep="last")
    if frame.empty:
        raise AemetObservationError("Las observaciones AEMET no cubren la ventana solicitada.")
    return frame.sort_values(["valid_time", "station_id"]).reset_index(drop=True)


def aggregate_aemet_hourly_to_daily(
    hourly: pd.DataFrame,
    *,
    min_coverage_hours: int = 20,
) -> pd.DataFrame:
    """Agrega observaciones horarias acumuladas a variables diarias del modelo.

    Solo devuelve pares estación-día con al menos ``min_coverage_hours``
    instantes distintos. No rellena horas ausentes: la incompletitud se propaga
    al estado y permite que la ingesta falle explícitamente.
    """

    required = {
        "station_id",
        "valid_time",
        "lat",
        "lon",
        "temperature_c",
        "relative_humidity_pct",
        "wind_speed_kmh",
        "precipitation_mm",
    }
    missing = required.difference(hourly.columns)
    if missing:
        raise AemetObservationError(
            f"Faltan columnas de observación horaria AEMET: {sorted(missing)}"
        )
    if min_coverage_hours <= 0 or min_coverage_hours > 24:
        raise AemetObservationError("min_coverage_hours debe estar entre 1 y 24.")

    frame = hourly.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["valid_time"]).copy()
    frame["fecha"] = (
        frame["valid_time"].dt.tz_convert(GALICIA_TZ).dt.normalize().dt.tz_localize(None)
    )
    numeric = [
        "temperature_c",
        "relative_humidity_pct",
        "wind_speed_kmh",
        "precipitation_mm",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["vpd_hourly"] = calculate_vpd(
        frame["temperature_c"], frame["relative_humidity_pct"]
    )
    frame["local_hour"] = frame["valid_time"].dt.tz_convert(GALICIA_TZ).dt.hour

    rows: list[dict[str, Any]] = []
    for (station_id, fecha), group in frame.groupby(["station_id", "fecha"], sort=True):
        coverage = int(group["valid_time"].dt.floor("h").nunique())
        if coverage < min_coverage_hours:
            continue
        critical = group[group["local_hour"].between(12, 18)]
        rows.append(
            {
                "station_id": station_id,
                "fecha": fecha,
                "lat": float(group["lat"].iloc[-1]),
                "lon": float(group["lon"].iloc[-1]),
                "station_name": group.get("station_name", pd.Series(dtype=object)).dropna().iloc[-1]
                if "station_name" in group and group["station_name"].notna().any()
                else None,
                "tmax_vc": group["temperature_c"].max(),
                "rhmin_vc": group["relative_humidity_pct"].min(),
                "vmax_vc": group["wind_speed_kmh"].max(),
                "prec_dia": group["precipitation_mm"].sum(min_count=1),
                "temperature_mean": group["temperature_c"].mean(),
                "temperature_min": group["temperature_c"].min(),
                "temperature_max": group["temperature_c"].max(),
                "temperature_max_12_18h": critical["temperature_c"].max(),
                "relative_humidity_mean": group["relative_humidity_pct"].mean(),
                "relative_humidity_min": group["relative_humidity_pct"].min(),
                "relative_humidity_min_12_18h": critical["relative_humidity_pct"].min(),
                "wind_speed_mean": group["wind_speed_kmh"].mean(),
                "wind_speed_max": group["wind_speed_kmh"].max(),
                "wind_speed_max_12_18h": critical["wind_speed_kmh"].max(),
                "vpd_mean": group["vpd_hourly"].mean(),
                "vpd_max_12_18h": critical["vpd_hourly"].max(),
                "precipitation_sum": group["precipitation_mm"].sum(min_count=1),
                "coverage_hours": float(coverage),
                "source": "aemet_current_observation_aggregate",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise AemetObservationError(
            "No hay ningún día-estación con la cobertura horaria mínima "
            f"({min_coverage_hours} h) para cerrar observaciones."
        )
    return result.sort_values(["fecha", "station_id"]).reset_index(drop=True)


def _project_coordinates(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    reference_latitude = float(np.nanmean(latitude))
    return np.column_stack(
        [longitude * 111.32 * np.cos(np.deg2rad(reference_latitude)), latitude * 110.57]
    )


def _idw_values(
    station_xy: np.ndarray,
    station_values: np.ndarray,
    grid_xy: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    valid = np.isfinite(station_values)
    if not valid.any():
        return np.full(len(grid_xy), np.nan), np.full(len(grid_xy), np.nan)
    valid_xy = station_xy[valid]
    valid_values = station_values[valid]
    tree = cKDTree(valid_xy)
    count = min(max(int(neighbors), 1), len(valid_values))
    distances, indices = tree.query(grid_xy, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    zero_distance = distances == 0
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights[zero_distance] = 0.0
    zero_rows = zero_distance.any(axis=1)
    if zero_rows.any():
        weights[zero_rows] = zero_distance[zero_rows].astype(float)
    values = valid_values[indices]
    denominator = weights.sum(axis=1)
    result = np.divide(
        (values * weights).sum(axis=1),
        denominator,
        out=np.full(len(grid_xy), np.nan),
        where=denominator > 0,
    )
    nearest_distance = distances[:, 0]
    return result, nearest_distance


def interpolate_aemet_daily_to_grid(
    station_daily: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    neighbors: int = 4,
    source: str = "aemet_daily_climatology_idw",
) -> pd.DataFrame:
    """Interpola observaciones diarias de estaciones a cada celda de la rejilla."""

    station_daily = station_daily.copy()
    # La climatología diaria de AEMET antigua solo contiene los cuatro
    # agregados de compatibilidad. Los derivados canónicos se rellenan aquí
    # como proxy explícito; cuando la fuente procede de observaciones horarias
    # ya estarán presentes y conservarán su resolución real.
    aliases = {
        "temperature_mean": "tmax_vc",
        "temperature_min": "tmax_vc",
        "temperature_max": "tmax_vc",
        "temperature_max_12_18h": "tmax_vc",
        "relative_humidity_mean": "rhmin_vc",
        "relative_humidity_min": "rhmin_vc",
        "relative_humidity_min_12_18h": "rhmin_vc",
        "wind_speed_mean": "vmax_vc",
        "wind_speed_max": "vmax_vc",
        "wind_speed_max_12_18h": "vmax_vc",
        "precipitation_sum": "prec_dia",
    }
    for canonical, legacy in aliases.items():
        if canonical not in station_daily.columns and legacy in station_daily.columns:
            station_daily[canonical] = station_daily[legacy]
    if "vpd_mean" not in station_daily.columns:
        station_daily["vpd_mean"] = calculate_vpd(
            station_daily["temperature_mean"], station_daily["relative_humidity_mean"]
        )
    if "vpd_max_12_18h" not in station_daily.columns:
        station_daily["vpd_max_12_18h"] = calculate_vpd(
            station_daily["temperature_max_12_18h"],
            station_daily["relative_humidity_min_12_18h"],
        )
    required_station = {"station_id", "fecha", "lat", "lon", *DAILY_WEATHER_COLUMNS[2:]}
    missing_station = required_station.difference(station_daily.columns)
    if missing_station:
        raise AemetObservationError(f"Faltan columnas de estaciones: {sorted(missing_station)}")
    required_grid = {"cell_id", "lat_centroid", "lon_centroid"}
    missing_grid = required_grid.difference(grid.columns)
    if missing_grid:
        raise AemetObservationError(f"Faltan columnas de rejilla: {sorted(missing_grid)}")

    grid_ids = grid["cell_id"].to_numpy()
    grid_xy = _project_coordinates(
        grid["lat_centroid"].to_numpy(float), grid["lon_centroid"].to_numpy(float)
    )
    rows: list[pd.DataFrame] = []
    variables = list(
        dict.fromkeys(
            ["tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia", *CANONICAL_DAILY_WEATHER_COLUMNS]
        )
    )
    for fecha, day in station_daily.groupby("fecha", sort=True):
        station_xy = _project_coordinates(day["lat"].to_numpy(float), day["lon"].to_numpy(float))
        output = pd.DataFrame({"cell_id": grid_ids, "fecha": fecha})
        nearest_distance = np.full(len(grid), np.nan)
        for variable in variables:
            values, distance = _idw_values(
                station_xy,
                pd.to_numeric(day[variable], errors="coerce").to_numpy(float),
                grid_xy,
                neighbors,
            )
            output[variable] = values.astype("float32")
            nearest_distance = np.fmin(nearest_distance, distance) if np.isfinite(nearest_distance).any() else distance
        coverage = pd.to_numeric(day.get("coverage_hours", 24.0), errors="coerce")
        output["coverage_hours"] = float(coverage.min()) if hasattr(coverage, "min") else 24.0
        output["source"] = source
        output["observation_station_count"] = int(day["station_id"].nunique())
        output["nearest_station_distance_km"] = nearest_distance.astype("float32")
        rows.append(output)

    if not rows:
        raise AemetObservationError("No hay días de observación para interpolar.")
    result = pd.concat(rows, ignore_index=True)
    required_output = list(dict.fromkeys([*DAILY_WEATHER_COLUMNS, *CANONICAL_DAILY_WEATHER_COLUMNS]))
    missing = result[required_output].isna().sum()
    if missing.any():
        raise AemetObservationError(
            "La interpolación dejó valores ausentes: "
            f"{missing[missing > 0].to_dict()}"
        )
    return result.sort_values(["fecha", "cell_id"]).reset_index(drop=True)


class AemetObservationClient:
    """Cliente AEMET para backfill diario y observación casi en tiempo real."""

    def __init__(self, config: AemetObservationConfig) -> None:
        self.config = config
        self.client = AemetClient(
            AemetForecastConfig(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                retry_backoff_seconds=config.retry_backoff_seconds,
                use_hourly_overlay=False,
            )
        )

    def fetch_daily_history(
        self,
        start_date: date | str,
        end_date: date | str,
        *,
        raw_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """Descarga, normaliza y filtra la climatología diaria de Galicia.

        AEMET solo acepta 15 días por consulta. La función divide rangos más
        largos, reutiliza el inventario de estaciones y concatena los bloques.
        """

        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        chunks = split_aemet_date_range(start, end)
        raw_path = Path(raw_dir) if raw_dir else None
        if raw_path:
            raw_path.mkdir(parents=True, exist_ok=True)
        try:
            inventory_payload = self.client.fetch_endpoint(AEMET_STATIONS_ENDPOINT)
            if raw_path:
                atomic_write_json(inventory_payload, raw_path / "aemet_station_inventory.json")
            station_frames: list[pd.DataFrame] = []
            for chunk_start, chunk_end in chunks:
                start_text = f"{chunk_start:%Y-%m-%d}T00:00:00UTC"
                end_text = f"{chunk_end:%Y-%m-%d}T23:59:59UTC"
                daily_endpoint = AEMET_DAILY_HISTORY_ENDPOINT.format(
                    start=start_text,
                    end=end_text,
                )
                daily_payload = self.client.fetch_endpoint(daily_endpoint)
                if raw_path:
                    atomic_write_json(
                        daily_payload,
                        raw_path / f"aemet_daily_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json",
                    )
                station_frames.append(
                    normalise_aemet_daily_payload(
                        daily_payload,
                        inventory_payload=inventory_payload,
                        start_date=chunk_start,
                        end_date=chunk_end,
                    )
                )
        except ForecastError as exc:
            raise AemetObservationError(
                "AEMET no pudo devolver la climatología diaria. "
                "Comprueba la clave, el rango de fechas y que el endpoint termine "
                "en '/'. Respuesta original: "
                f"{exc}"
            ) from exc
        result = (
            pd.concat(station_frames, ignore_index=True)
            .drop_duplicates(["station_id", "fecha"], keep="last")
            .sort_values(["fecha", "station_id"])
            .reset_index(drop=True)
        )
        expected_dates = pd.date_range(start, end, freq="D")
        actual_dates = pd.DatetimeIndex(result["fecha"].dropna().unique()).sort_values()
        missing_dates = expected_dates.difference(actual_dates)
        if len(missing_dates):
            missing_text = ", ".join(timestamp.strftime("%Y-%m-%d") for timestamp in missing_dates)
            raise AemetObservationError(
                "AEMET no devolvió datos para todas las fechas solicitadas. "
                f"Faltan {len(missing_dates)} fechas: {missing_text}. "
                "No se publicará un estado incompleto; espera a que AEMET cierre "
                "la climatología o solicita un rango que ya esté completo."
            )
        return result

    def fetch_current_observations(
        self,
        *,
        raw_dir: str | Path | None = None,
        downloaded_at: Any | None = None,
    ) -> pd.DataFrame:
        """Descarga la ventana móvil de observaciones horarias de AEMET.

        El endpoint oficial solo conserva las últimas horas. Por ello este
        método no pretende reconstruir por sí mismo un día completo: el script
        operativo acumula su salida en un Parquet local y agrega únicamente los
        días con cobertura suficiente.
        """

        raw_path = Path(raw_dir) if raw_dir else None
        if raw_path:
            raw_path.mkdir(parents=True, exist_ok=True)
        try:
            payload = self.client.fetch_endpoint(AEMET_CURRENT_OBSERVATIONS_ENDPOINT)
            if raw_path:
                timestamp = pd.Timestamp(downloaded_at or pd.Timestamp.now(tz="UTC"))
                stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
                atomic_write_json(payload, raw_path / f"aemet_current_{stamp}.json")
            return normalise_aemet_hourly_payload(payload)
        except ForecastError as exc:
            raise AemetObservationError(
                "AEMET no pudo devolver las observaciones horarias actuales: " f"{exc}"
            ) from exc


def build_aemet_client_from_env() -> AemetObservationClient:
    """Construye el cliente desde las variables de entorno del proyecto."""

    import os

    api_key = os.getenv("AEMET_API_KEY", "").strip()
    if not api_key:
        raise AemetObservationError("AEMET_API_KEY no está configurada.")
    return AemetObservationClient(
        AemetObservationConfig(
            api_key=api_key,
            base_url=os.getenv("AEMET_BASE_URL", AEMET_BASE_URL),
            timeout_seconds=int(os.getenv("AEMET_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("AEMET_MAX_RETRIES", "3")),
            idw_neighbors=int(os.getenv("AEMET_OBSERVATION_IDW_NEIGHBORS", "4")),
        )
    )


__all__ = [
    "AEMET_API_VERSION",
    "AEMET_CURRENT_OBSERVATIONS_ENDPOINT",
    "AemetObservationClient",
    "AemetObservationConfig",
    "AemetObservationError",
    "build_aemet_client_from_env",
    "interpolate_aemet_daily_to_grid",
    "aggregate_aemet_hourly_to_daily",
    "normalise_aemet_hourly_payload",
    "normalise_aemet_daily_payload",
    "parse_aemet_coordinate",
    "split_aemet_date_range",
]
