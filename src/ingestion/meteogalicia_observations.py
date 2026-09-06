"""Ingesta y normalización de observaciones diarias de MeteoGalicia (EMA).

MeteoGalicia opera una red de más de 140 Estaciones Meteorológicas Automáticas (EMA)
en el territorio gallego y ofrece un servicio Open Data en formato JSON:
``https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action``.

Este módulo:
1. Descarga el JSON diario para un rango de fechas (formato dd/MM/yyyy).
2. Extrae las medidas diarias de cada estación (temperatura, humedad, viento, lluvia).
3. Convierte las coordenadas UTM huso 29N (EPSG:25829) a WGS84 geográficas (EPSG:4326).
4. Normaliza las variables y calcula los derivados físicos (VPD, conversión de m/s a km/h).
5. Interpola espacialmente las estaciones a la rejilla de 1 km mediante IDW (Inverse Distance Weighting).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.features.operational_features import calculate_vpd
from src.ingestion.aemet_observations import (
    CANONICAL_DAILY_WEATHER_COLUMNS,
    DAILY_WEATHER_COLUMNS,
    GALICIA_BOUNDS,
    _idw_values,
    _project_coordinates,
)
from src.operational.artifacts import atomic_write_json

LOGGER = logging.getLogger(__name__)

GALICIA_TZ = ZoneInfo("Europe/Madrid")
DEFAULT_METEOGALICIA_DAILY_URL = (
    "https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action"
)

# Mapeo de parámetros de MeteoGalicia a variables canónicas
METEOGALICIA_PARAM_MAP = {
    "TA_MAX_1.5m": "tmax_c",
    "TA_MIN_1.5m": "tmin_c",
    "TA_AVG_1.5m": "tmean_c",
    "HR_MIN_1.5m": "rhmin_pct",
    "HR_MAX_1.5m": "rhmax_pct",
    "HR_AVG_1.5m": "rhmean_pct",
    "PP_SUM_1.5m": "precip_mm",
    "VV_AVG_10m": "wind_mean_ms",
    "VV_MAX_10m": "wind_max_ms",
}


class MeteoGaliciaObservationError(RuntimeError):
    """Error controlado en la descarga, parseo o interpolación de observaciones MeteoGalicia."""


@dataclass(frozen=True)
class MeteoGaliciaObservationConfig:
    """Configuración del cliente de observaciones Open Data de MeteoGalicia."""

    base_url: str = DEFAULT_METEOGALICIA_DAILY_URL
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0


def utm29n_to_wgs84(utmx: float, utmy: float) -> tuple[float, float]:
    """Convierte coordenadas UTM huso 29N (EPSG:25829) a lat/lon WGS84 (EPSG:4326).

    Utiliza ``pyproj.Transformer`` si está disponible. En su defecto, aplica
    la formulación analítica de la proyección transversal de Mercator para el elipsoide GRS80/WGS84.
    """
    if pd.isna(utmx) or pd.isna(utmy):
        return np.nan, np.nan

    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:25829", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(utmx, utmy)
        return float(lat), float(lon)
    except (ImportError, Exception):
        pass

    # Algoritmo analítico estándar de proyección inversa Transverse Mercator (GRS80/WGS84)
    a = 6378137.0
    f = 1 / 298.257222101
    e2 = 2 * f - f**2
    e_prime2 = e2 / (1 - e2)
    k0 = 0.9996
    lon0 = -9.0  # Meridiano central del huso 29N en radianes: -9 grados

    x = float(utmx) - 500000.0
    y = float(utmy)

    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512

    fp = (
        mu
        + j1 * math.sin(2 * mu)
        + j2 * math.sin(4 * mu)
        + j3 * math.sin(6 * mu)
        + j4 * math.sin(8 * mu)
    )

    c1 = e_prime2 * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e2) / ((1 - e2 * math.sin(fp) ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - e2 * math.sin(fp) ** 2)
    d = x / (n1 * k0)

    lat = fp - (n1 * math.tan(fp) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e_prime2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e_prime2 - 3 * c1**2) * d**6 / 720
    )
    lon = (
        math.radians(lon0)
        + (
            d
            - (1 + 2 * t1 + c1) * d**3 / 6
            + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e_prime2 + 24 * t1**2) * d**5 / 120
        )
        / math.cos(fp)
    )

    return math.degrees(lat), math.degrees(lon)


def _safe_float(val: Any) -> float:
    if val is None or val == "":
        return np.nan
    try:
        v = float(str(val).replace(",", "."))
        return v if np.isfinite(v) and v > -9990 else np.nan
    except (ValueError, TypeError):
        return np.nan


def normalise_meteogalicia_daily_payload(
    payload: Mapping[str, Any] | list[dict[str, Any]],
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Normaliza la respuesta JSON de datosDiariosEstacionsMeteo a un DataFrame de estaciones.

    Devuelve un registro por cada par (station_id, fecha) con variables físicas
    alineadas con el contrato canónico del proyecto.
    """
    if isinstance(payload, dict):
        raw_days = payload.get("listDatosDiarios", [])
        if not raw_days and "listaEstacions" in payload:
            raw_days = [payload]
    elif isinstance(payload, list):
        raw_days = payload
    else:
        raise MeteoGaliciaObservationError("Formato de payload MeteoGalicia no reconocido.")

    if not raw_days:
        raise MeteoGaliciaObservationError("El payload de MeteoGalicia no contiene días.")

    rows: list[dict[str, Any]] = []

    for day_entry in raw_days:
        raw_date_str = day_entry.get("data")
        if not raw_date_str:
            continue
        try:
            day_date = pd.to_datetime(raw_date_str, utc=True).tz_convert(GALICIA_TZ).normalize().tz_localize(None)
        except Exception:
            continue

        stations = day_entry.get("listaEstacions", [])
        for st in stations:
            st_id = st.get("idEstacion")
            st_name = st.get("estacion") or st.get("nomeEstacion")
            st_province = st.get("provincia")
            st_concello = st.get("concello")

            utmx = _safe_float(st.get("utmx"))
            utmy = _safe_float(st.get("utmy"))
            lat = _safe_float(st.get("lat"))
            lon = _safe_float(st.get("lon"))

            if pd.isna(lat) or pd.isna(lon):
                if pd.notna(utmx) and pd.notna(utmy):
                    lat, lon = utm29n_to_wgs84(utmx, utmy)
                else:
                    continue

            if not (
                GALICIA_BOUNDS["min_lat"] <= lat <= GALICIA_BOUNDS["max_lat"]
                and GALICIA_BOUNDS["min_lon"] <= lon <= GALICIA_BOUNDS["max_lon"]
            ):
                continue

            # Extraer medidas
            medidas_list = st.get("listaMedidas", [])
            medidas_dict: dict[str, float] = {}
            for m in medidas_list:
                param = m.get("codigoParametro")
                val = _safe_float(m.get("valor"))
                if param and pd.notna(val):
                    medidas_dict[param] = val

            tmax = medidas_dict.get("TA_MAX_1.5m", np.nan)
            tmin = medidas_dict.get("TA_MIN_1.5m", np.nan)
            tmean = medidas_dict.get("TA_AVG_1.5m", np.nan)
            rhmin = medidas_dict.get("HR_MIN_1.5m", np.nan)
            rhmean = medidas_dict.get("HR_AVG_1.5m", np.nan)
            precip = medidas_dict.get("PP_SUM_1.5m", 0.0)

            # Viento: MeteoGalicia entrega m/s. Convertimos a km/h (* 3.6).
            wind_mean_ms = medidas_dict.get("VV_AVG_10m", np.nan)
            wind_max_ms = medidas_dict.get("VV_MAX_10m", np.nan)

            wind_mean_kmh = wind_mean_ms * 3.6 if pd.notna(wind_mean_ms) else np.nan
            wind_max_kmh = (
                wind_max_ms * 3.6
                if pd.notna(wind_max_ms)
                else (wind_mean_kmh if pd.notna(wind_mean_kmh) else np.nan)
            )

            # Si faltan tmax o rhmin directos, usar tmean o rhmean como proxy
            if pd.isna(tmax) and pd.notna(tmean):
                tmax = tmean
            if pd.isna(rhmin) and pd.notna(rhmean):
                rhmin = rhmean

            # En un día completo de observaciones de estación, la tmax y rhmin
            # ocurren casi invariablemente en la ventana solar/crítica (12:00-18:00)
            tmax_vc = tmax
            rhmin_vc = rhmin
            vmax_vc = wind_max_kmh
            prec_dia = precip

            vpd_m = calculate_vpd(tmean, rhmean) if pd.notna(tmean) and pd.notna(rhmean) else np.nan
            vpd_crit = (
                calculate_vpd(tmax_vc, rhmin_vc) if pd.notna(tmax_vc) and pd.notna(rhmin_vc) else np.nan
            )

            rows.append(
                {
                    "station_id": str(st_id),
                    "fecha": day_date,
                    "lat": float(lat),
                    "lon": float(lon),
                    "station_name": str(st_name) if st_name else None,
                    "station_province": str(st_province) if st_province else None,
                    "station_concello": str(st_concello) if st_concello else None,
                    "tmax_vc": float(tmax_vc) if pd.notna(tmax_vc) else np.nan,
                    "rhmin_vc": float(rhmin_vc) if pd.notna(rhmin_vc) else np.nan,
                    "vmax_vc": float(vmax_vc) if pd.notna(vmax_vc) else np.nan,
                    "prec_dia": float(prec_dia) if pd.notna(prec_dia) else 0.0,
                    "vpd_vc": float(vpd_crit) if pd.notna(vpd_crit) else np.nan,
                    "temperature_mean": float(tmean) if pd.notna(tmean) else np.nan,
                    "temperature_min": float(tmin) if pd.notna(tmin) else np.nan,
                    "temperature_max": float(tmax) if pd.notna(tmax) else np.nan,
                    "temperature_max_12_18h": float(tmax_vc) if pd.notna(tmax_vc) else np.nan,
                    "relative_humidity_mean": float(rhmean) if pd.notna(rhmean) else np.nan,
                    "relative_humidity_min": float(rhmin) if pd.notna(rhmin) else np.nan,
                    "relative_humidity_min_12_18h": float(rhmin_vc) if pd.notna(rhmin_vc) else np.nan,
                    "wind_speed_mean": float(wind_mean_kmh) if pd.notna(wind_mean_kmh) else np.nan,
                    "wind_speed_max": float(wind_max_kmh) if pd.notna(wind_max_kmh) else np.nan,
                    "wind_speed_max_12_18h": float(wind_max_kmh) if pd.notna(wind_max_kmh) else np.nan,
                    "vpd_mean": float(vpd_m) if pd.notna(vpd_m) else np.nan,
                    "vpd_max_12_18h": float(vpd_crit) if pd.notna(vpd_crit) else np.nan,
                    "precipitation_sum": float(prec_dia) if pd.notna(prec_dia) else 0.0,
                    "coverage_hours": 24.0,
                    "source": "meteogalicia_ema",
                }
            )

    if not rows:
        raise MeteoGaliciaObservationError(
            "MeteoGalicia no devolvió estaciones válidas en Galicia para el periodo."
        )

    df = pd.DataFrame(rows)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.tz_localize(None)

    if start_date is not None:
        df = df[df["fecha"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["fecha"] <= pd.Timestamp(end_date)]

    df = df.dropna(subset=["station_id", "fecha"]).drop_duplicates(
        ["station_id", "fecha"], keep="last"
    )

    if df.empty:
        raise MeteoGaliciaObservationError("No hay registros en el rango de fechas solicitado.")

    return df.sort_values(["fecha", "station_id"]).reset_index(drop=True)


def interpolate_meteogalicia_daily_to_grid(
    station_daily: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    neighbors: int = 4,
    source: str = "meteogalicia_ema_idw",
) -> pd.DataFrame:
    """Interpola observaciones diarias de estaciones MeteoGalicia a la rejilla de 1 km."""
    required_station = {"station_id", "fecha", "lat", "lon", "tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia"}
    missing_st = required_station.difference(station_daily.columns)
    if missing_st:
        raise MeteoGaliciaObservationError(f"Faltan columnas en station_daily: {sorted(missing_st)}")

    required_grid = {"cell_id", "lat_centroid", "lon_centroid"}
    missing_grid = required_grid.difference(grid.columns)
    if missing_grid:
        raise MeteoGaliciaObservationError(f"Faltan columnas en la rejilla: {sorted(missing_grid)}")

    grid_ids = grid["cell_id"].to_numpy()
    grid_xy = _project_coordinates(
        grid["lat_centroid"].to_numpy(float), grid["lon_centroid"].to_numpy(float)
    )

    variables = list(
        dict.fromkeys(
            ["tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia", *CANONICAL_DAILY_WEATHER_COLUMNS]
        )
    )

    rows: list[pd.DataFrame] = []
    for fecha, day in station_daily.groupby("fecha", sort=True):
        station_xy = _project_coordinates(day["lat"].to_numpy(float), day["lon"].to_numpy(float))
        output = pd.DataFrame({"cell_id": grid_ids, "fecha": fecha})
        nearest_distance = np.full(len(grid), np.nan)

        for variable in variables:
            col_data = (
                pd.to_numeric(day[variable], errors="coerce").to_numpy(float)
                if variable in day.columns
                else np.full(len(day), np.nan)
            )
            values, distance = _idw_values(
                station_xy,
                col_data,
                grid_xy,
                neighbors,
            )
            output[variable] = values.astype("float32")
            nearest_distance = (
                np.fmin(nearest_distance, distance)
                if np.isfinite(nearest_distance).any()
                else distance
            )

        # Recomputar VPD en celdas interpoladas para coherencia física
        output["vpd_vc"] = calculate_vpd(output["tmax_vc"], output["rhmin_vc"]).astype("float32")
        output["vpd_max_12_18h"] = output["vpd_vc"]
        output["vpd_mean"] = calculate_vpd(output["temperature_mean"], output["relative_humidity_mean"]).astype("float32")

        coverage = pd.to_numeric(day.get("coverage_hours", 24.0), errors="coerce")
        output["coverage_hours"] = float(coverage.min()) if hasattr(coverage, "min") else 24.0
        output["source"] = source
        output["observation_station_count"] = int(day["station_id"].nunique())
        output["nearest_station_distance_km"] = nearest_distance.astype("float32")
        rows.append(output)

    if not rows:
        raise MeteoGaliciaObservationError("No hay días de observación para interpolar.")

    result = pd.concat(rows, ignore_index=True)
    required_output = list(dict.fromkeys([*DAILY_WEATHER_COLUMNS, *CANONICAL_DAILY_WEATHER_COLUMNS]))
    missing = result[required_output].isna().sum()
    if missing.any():
        raise MeteoGaliciaObservationError(
            f"La interpolación dejó valores ausentes: {missing[missing > 0].to_dict()}"
        )
    return result.sort_values(["fecha", "cell_id"]).reset_index(drop=True)


class MeteoGaliciaObservationClient:
    """Cliente HTTP para la API Open Data de observaciones de MeteoGalicia."""

    def __init__(self, config: MeteoGaliciaObservationConfig | None = None) -> None:
        self.config = config or MeteoGaliciaObservationConfig()

    def fetch_daily_observations(
        self,
        start_date: date | str,
        end_date: date | str,
        *,
        raw_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Descarga el JSON diario de MeteoGalicia para un rango de fechas."""
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()

        if end < start:
            raise MeteoGaliciaObservationError("end_date no puede ser anterior a start_date.")

        params = {
            # El servicio documenta el parámetro histórico como ``dataIni``.
            # ``datIni`` puede aparecer en una errata de la documentación, pero
            # el endpoint no aplica correctamente el rango cuando se envía así.
            "dataIni": start.strftime("%d/%m/%Y"),
            "dataFin": end.strftime("%d/%m/%Y"),
        }
        headers = {
            "User-Agent": "AnticipacionIncendiosForestalesTFM/1.0",
            "Accept": "application/json",
        }

        LOGGER.info(
            "Consultando observaciones MeteoGalicia: %s con params %s",
            self.config.base_url,
            params,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.get(
                    self.config.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()

                if raw_dir is not None:
                    raw_dir = Path(raw_dir)
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d_%H%M%S")
                    raw_file = raw_dir / f"meteogalicia_obs_{start}_{end}_{stamp}.json"
                    atomic_write_json(payload, raw_file)

                return payload
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Intento %s/%s fallido consultando MeteoGalicia: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )

        raise MeteoGaliciaObservationError(
            f"No se pudieron descargar observaciones de MeteoGalicia tras {self.config.max_retries} intentos."
        ) from last_error
