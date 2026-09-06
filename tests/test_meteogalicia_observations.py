"""Tests para la ingesta y normalización de observaciones de MeteoGalicia (EMA)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.aemet_observations import GALICIA_BOUNDS
from src.ingestion.meteogalicia_observations import (
    MeteoGaliciaObservationClient,
    MeteoGaliciaObservationConfig,
    interpolate_meteogalicia_daily_to_grid,
    normalise_meteogalicia_daily_payload,
    utm29n_to_wgs84,
)
from src.ingestion.weather_state import merge_weather_state


def test_utm29n_to_wgs84_accuracy():
    """Comprueba la transformación de coordenadas UTM huso 29N a lat/lon WGS84."""
    # Coordenadas de Mabegondo (A Coruña)
    utmx = 559899.0
    utmy = 4787883.0

    lat, lon = utm29n_to_wgs84(utmx, utmy)

    # Debe estar en torno a 43.24 N, -8.26 W
    assert pytest.approx(lat, abs=0.05) == 43.24
    assert pytest.approx(lon, abs=0.05) == -8.26

    # Debe caer dentro de los límites de Galicia
    assert GALICIA_BOUNDS["min_lat"] <= lat <= GALICIA_BOUNDS["max_lat"]
    assert GALICIA_BOUNDS["min_lon"] <= lon <= GALICIA_BOUNDS["max_lon"]


def test_normalise_meteogalicia_synthetic_payload():
    """Comprueba el mapeo de variables y conversión de unidades con un payload sintético."""
    payload = {
        "listDatosDiarios": [
            {
                "data": "2026-09-04T00:00:00",
                "listaEstacions": [
                    {
                        "idEstacion": 10045,
                        "estacion": "Mabegondo",
                        "concello": "ABEGONDO",
                        "provincia": "A Coruña",
                        "utmx": "559899.0",
                        "utmy": "4787883.0",
                        "listaMedidas": [
                            {"codigoParametro": "TA_MAX_1.5m", "valor": 28.5},
                            {"codigoParametro": "TA_MIN_1.5m", "valor": 12.0},
                            {"codigoParametro": "TA_AVG_1.5m", "valor": 20.0},
                            {"codigoParametro": "HR_MIN_1.5m", "valor": 35.0},
                            {"codigoParametro": "HR_MAX_1.5m", "valor": 95.0},
                            {"codigoParametro": "HR_AVG_1.5m", "valor": 65.0},
                            {"codigoParametro": "PP_SUM_1.5m", "valor": 0.0},
                            {"codigoParametro": "VV_AVG_10m", "valor": 2.0},  # 2 m/s = 7.2 km/h
                            {"codigoParametro": "VV_MAX_10m", "valor": 5.0},  # 5 m/s = 18.0 km/h
                        ],
                    }
                ],
            }
        ]
    }

    df = normalise_meteogalicia_daily_payload(payload)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["station_id"] == "10045"
    assert row["tmax_vc"] == 28.5
    assert row["rhmin_vc"] == 35.0
    assert row["prec_dia"] == 0.0
    assert row["wind_speed_mean"] == pytest.approx(7.2)
    assert row["vmax_vc"] == pytest.approx(18.0)
    assert row["wind_speed_max"] == pytest.approx(18.0)
    assert row["vpd_vc"] > 0
    assert row["coverage_hours"] == 24.0
    assert row["source"] == "meteogalicia_ema"


def test_client_uses_historical_date_parameter(monkeypatch):
    """Comprueba que la petición usa el parámetro histórico que entiende la API."""
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"listDatosDiarios": []}

    def fake_get(url, *, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "src.ingestion.meteogalicia_observations.requests.get",
        fake_get,
    )
    client = MeteoGaliciaObservationClient(
        MeteoGaliciaObservationConfig(max_retries=1)
    )

    client.fetch_daily_observations("2026-09-03", "2026-09-05")

    assert captured["params"] == {
        "dataIni": "03/09/2026",
        "dataFin": "05/09/2026",
    }


def test_normalise_meteogalicia_real_sample():
    """Valida la normalización contra el archivo de muestra de julio 2022 si existe."""
    sample_path = Path("misc/Datos/meteogalicia_diario_julio2022.json")
    if not sample_path.exists():
        pytest.skip("No existe misc/Datos/meteogalicia_diario_julio2022.json")

    with open(sample_path, encoding="utf-8") as f:
        payload = json.load(f)

    # Filtrar solo el primer día
    if "listDatosDiarios" in payload and len(payload["listDatosDiarios"]) > 1:
        payload = {"listDatosDiarios": payload["listDatosDiarios"][:1]}

    df = normalise_meteogalicia_daily_payload(payload)

    assert not df.empty
    assert df["station_id"].nunique() > 50
    assert "tmax_vc" in df.columns
    assert "rhmin_vc" in df.columns
    assert "vmax_vc" in df.columns
    valid_rh = df["rhmin_vc"].dropna()
    assert not valid_rh.empty
    assert (valid_rh >= 0).all() and (valid_rh <= 100).all()
    valid_wind = df["vmax_vc"].dropna()
    assert not valid_wind.empty
    assert (valid_wind >= 0).all()


def test_interpolate_meteogalicia_daily_to_grid():
    """Comprueba la interpolación IDW de estaciones a una rejilla sintética."""
    station_daily = pd.DataFrame(
        [
            {
                "station_id": "1",
                "fecha": pd.Timestamp("2026-09-04"),
                "lat": 43.0,
                "lon": -8.0,
                "tmax_vc": 25.0,
                "rhmin_vc": 40.0,
                "vmax_vc": 15.0,
                "prec_dia": 0.0,
                "temperature_mean": 20.0,
                "temperature_min": 15.0,
                "temperature_max": 25.0,
                "temperature_max_12_18h": 25.0,
                "relative_humidity_mean": 60.0,
                "relative_humidity_min": 40.0,
                "relative_humidity_min_12_18h": 40.0,
                "wind_speed_mean": 10.0,
                "wind_speed_max": 15.0,
                "wind_speed_max_12_18h": 15.0,
                "vpd_mean": 1.0,
                "vpd_max_12_18h": 1.5,
                "precipitation_sum": 0.0,
                "coverage_hours": 24.0,
            },
            {
                "station_id": "2",
                "fecha": pd.Timestamp("2026-09-04"),
                "lat": 42.5,
                "lon": -8.5,
                "tmax_vc": 30.0,
                "rhmin_vc": 30.0,
                "vmax_vc": 20.0,
                "prec_dia": 2.0,
                "temperature_mean": 22.0,
                "temperature_min": 17.0,
                "temperature_max": 30.0,
                "temperature_max_12_18h": 30.0,
                "relative_humidity_mean": 50.0,
                "relative_humidity_min": 30.0,
                "relative_humidity_min_12_18h": 30.0,
                "wind_speed_mean": 12.0,
                "wind_speed_max": 20.0,
                "wind_speed_max_12_18h": 20.0,
                "vpd_mean": 1.3,
                "vpd_max_12_18h": 2.0,
                "precipitation_sum": 2.0,
                "coverage_hours": 24.0,
            },
        ]
    )

    grid = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_b"],
            "lat_centroid": [42.9, 42.6],
            "lon_centroid": [-8.1, -8.4],
        }
    )

    interpolated = interpolate_meteogalicia_daily_to_grid(station_daily, grid, neighbors=2)

    assert len(interpolated) == 2
    assert set(interpolated["cell_id"]) == {"cell_a", "cell_b"}
    assert interpolated["tmax_vc"].notna().all()
    assert interpolated["rhmin_vc"].notna().all()
    assert interpolated["vpd_vc"].notna().all()
    assert (interpolated["source"] == "meteogalicia_ema_idw").all()


def test_integration_with_weather_state_merge():
    """Comprueba que la salida interpolada de MeteoGalicia es compatible con merge_weather_state."""
    grid_daily = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "fecha": [pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-04")],
            "tmax_vc": [26.0, 27.0],
            "rhmin_vc": [45.0, 40.0],
            "vmax_vc": [12.0, 14.0],
            "prec_dia": [0.0, 0.5],
            "vpd_vc": [1.5, 1.7],
            "temperature_mean": [20.0, 21.0],
            "temperature_min": [14.0, 15.0],
            "temperature_max": [26.0, 27.0],
            "temperature_max_12_18h": [26.0, 27.0],
            "relative_humidity_mean": [65.0, 60.0],
            "relative_humidity_min": [45.0, 40.0],
            "relative_humidity_min_12_18h": [45.0, 40.0],
            "wind_speed_mean": [8.0, 9.0],
            "wind_speed_max": [12.0, 14.0],
            "wind_speed_max_12_18h": [12.0, 14.0],
            "vpd_mean": [0.8, 0.9],
            "vpd_max_12_18h": [1.5, 1.7],
            "precipitation_sum": [0.0, 0.5],
            "coverage_hours": [24.0, 24.0],
            "source": ["meteogalicia_ema_idw", "meteogalicia_ema_idw"],
        }
    )

    merged = merge_weather_state(
        None,
        grid_daily,
        as_of=pd.Timestamp("2026-09-05", tz="Europe/Madrid"),
        retention_days=10,
    )

    assert len(merged) == 2
    assert "source" in merged.columns
    assert (merged["source"] == "meteogalicia_ema_idw").all()
    assert (merged["coverage_hours"] == 24.0).all()
