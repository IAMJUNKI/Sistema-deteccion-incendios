from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from src.features.operational_features import (
    aggregate_hourly_forecast,
    build_operational_features,
)
from src.ingestion.aemet_forecast import (
    AemetClient,
    AemetForecastConfig,
    AemetPoint,
    parse_aemet_payload,
)
from src.ingestion.aemet_observations import (
    AemetObservationClient,
    AemetObservationConfig,
    AemetObservationError,
    aggregate_aemet_hourly_to_daily,
    interpolate_aemet_daily_to_grid,
    normalise_aemet_hourly_payload,
    normalise_aemet_daily_payload,
    parse_aemet_coordinate,
    split_aemet_date_range,
)
from src.ingestion.meteogalicia_forecast import (
    ForecastConfig,
    ForecastPoint,
    ForecastValidationError,
    MeteoGaliciaClient,
    assign_forecast_to_grid,
    load_latest_forecast,
    parse_meteogalicia_payload,
    save_forecast_parquet,
    validate_hourly_forecast,
)
from src.ingestion.weather_state import WeatherStateError, validate_weather_state
from src.models.forecast_risk_model import add_risk_outputs
from src.operational.artifacts import RunLockError, run_lock


def _payload() -> dict:
    times = [
        "2026-08-17T12:00:00+02",
        "2026-08-17T13:00:00+02",
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [-8.0, 42.5]},
                "properties": {
                    "id": "p1",
                    "days": [
                        {
                            "variables": [
                                {"name": "temperature", "values": [{"timeInstant": t, "value": 30} for t in times]},
                                {"name": "relative_humidity", "values": [{"timeInstant": t, "value": 25} for t in times]},
                                {"name": "precipitation_amount", "values": [{"timeInstant": t, "value": 0.5} for t in times]},
                                {"name": "wind", "values": [{"timeInstant": t, "moduleValue": 20, "directionValue": 180} for t in times]},
                            ]
                        }
                    ],
                },
            }
        ],
    }


def _hourly_forecast() -> pd.DataFrame:
    rows = []
    for cell_id in [1, 2]:
        for date in pd.date_range("2026-08-17", periods=3, freq="D"):
            local_hours = pd.date_range(
                date,
                date + pd.Timedelta(hours=23),
                freq="h",
                tz="Europe/Madrid",
            )
            for timestamp in local_hours:
                rows.append(
                    {
                        "cell_id": cell_id,
                        "valid_time": timestamp.tz_convert("UTC"),
                        "forecast_lat": 42.5 + cell_id / 1000,
                        "forecast_lon": -8.0,
                        "temperature_c": 32.0 if timestamp.hour == 15 else 20.0,
                        "relative_humidity_pct": 25.0 if timestamp.hour == 15 else 60.0,
                        "precipitation_mm": 0.0,
                        "wind_speed_kmh": 20.0,
                        "forecast_run_at": pd.Timestamp("2026-08-16T03:00:00Z"),
                        "provider": "meteogalicia",
                        "model": "WRF",
                        "grid": "04km",
                    }
                )
    return pd.DataFrame(rows)


def _aemet_payload() -> list[dict]:
    days = []
    for date in pd.date_range("2026-08-17", periods=3, freq="D"):
        date_text = date.strftime("%Y-%m-%dT00:00:00")
        temperature = [
            {"periodo": f"{hour:02d}", "value": "30" if hour == 15 else "20"}
            for hour in range(24)
        ]
        humidity = [
            {"periodo": f"{hour:02d}", "value": "25" if hour == 15 else "60"}
            for hour in range(24)
        ]
        precipitation = [{"periodo": "00-24", "value": "24"}]
        wind = []
        for hour in range(24):
            wind.extend(
                [
                    {
                        "periodo": f"{hour:02d}",
                        "direccion": ["S"],
                        "velocidad": ["20"],
                    },
                    {"periodo": f"{hour:02d}", "value": "30"},
                ]
            )
        days.append(
            {
                "fecha": date_text,
                "temperatura": temperature,
                "humedadRelativa": humidity,
                "precipitacion": precipitation,
                "vientoAndRachaMax": wind,
            }
        )
    return [
        {
            "id": "15030",
            "version": "3.0",
            "elaborado": "2026-08-16T03:00:00",
            "prediccion": {"dia": days},
        }
    ]


def test_parse_meteogalicia_payload_normalises_variables_and_timezone() -> None:
    parsed = parse_meteogalicia_payload(_payload(), issued_at="2026-08-16T03:00:00Z")
    assert len(parsed) == 2
    assert parsed["valid_time"].dt.tz is not None
    assert parsed["temperature_c"].tolist() == [30.0, 30.0]
    assert parsed["relative_humidity_pct"].tolist() == [25.0, 25.0]
    assert parsed["wind_speed_kmh"].tolist() == [20.0, 20.0]
    assert parsed["wind_dir_deg"].tolist() == [180.0, 180.0]
    assert parsed["issued_at"].dt.tz is not None
    assert parsed["horizon_hours"].iloc[0] == 31.0
    assert parsed["model_version"].unique().tolist() == ["WRF"]


def test_parse_aemet_daily_payload_expands_72_hours_and_preserves_daily_rain() -> None:
    parsed = parse_aemet_payload(
        _aemet_payload(),
        AemetPoint("15030", 43.3623, -8.4115),
        start_time="2026-08-17T00:00:00+02:00",
        end_time="2026-08-19T23:00:00+02:00",
    )
    assert len(parsed) == 72
    assert parsed["valid_time"].dt.tz is not None
    assert parsed["temperature_c"].max() == 30.0
    assert parsed["relative_humidity_pct"].min() == 25.0
    assert parsed["wind_speed_kmh"].min() == 20.0
    assert parsed.groupby(parsed["valid_time"].dt.tz_convert("Europe/Madrid").dt.date)[
        "precipitation_mm"
    ].sum().tolist() == [24.0, 24.0, 24.0]


def test_parse_aemet_does_not_turn_missing_daily_precipitation_into_zero() -> None:
    payload = _aemet_payload()
    for day in payload[0]["prediccion"]["dia"]:
        day.pop("precipitacion")
    parsed = parse_aemet_payload(
        payload,
        AemetPoint("15030", 43.3623, -8.4115),
        start_time="2026-08-17T00:00:00+02:00",
        end_time="2026-08-19T23:00:00+02:00",
    )
    assert parsed["precipitation_mm"].isna().all()


def test_aemet_current_observation_parser_aggregates_local_day() -> None:
    payload = [
        {
            "idema": "15030",
            "lat": 43.3623,
            "lon": -8.4115,
            "fint": f"2026-08-16T{hour:02d}:00:00Z",
            "ta": 25 + hour / 10,
            "hr": 60 - hour,
            "vmax": 20 + hour,
            "prec": 0.5,
        }
        for hour in range(20)
    ]
    hourly = normalise_aemet_hourly_payload(payload)
    daily = aggregate_aemet_hourly_to_daily(hourly, min_coverage_hours=20)
    assert len(hourly) == 20
    assert daily["fecha"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-16"]
    assert daily["tmax_vc"].iloc[0] == 26.9
    assert daily["rhmin_vc"].iloc[0] == 41.0
    assert daily["vmax_vc"].iloc[0] == 39.0
    assert daily["prec_dia"].iloc[0] == 10.0
    assert daily["coverage_hours"].iloc[0] == 20.0


def test_aemet_client_uses_two_step_data_url_and_archives_payloads(tmp_path) -> None:
    calls: list[tuple[str, dict]] = []
    payload = _aemet_payload()

    class FakeResponse:
        def __init__(self, body: object) -> None:
            self.body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self.body

    def fake_get(url: str, *, params: dict, headers: dict, timeout: int) -> FakeResponse:
        calls.append((url, params))
        if url.endswith("/diaria/15030"):
            return FakeResponse({"estado": 200, "datos": "https://aemet.test/daily"})
        return FakeResponse(payload)

    client = AemetClient(
        AemetForecastConfig(api_key="test", use_hourly_overlay=False, max_retries=1),
        request_get=fake_get,
    )
    frame = client.fetch_points(
        [AemetPoint("15030", 43.3623, -8.4115)],
        start_time="2026-08-17T00:00:00+02:00",
        end_time="2026-08-19T23:00:00+02:00",
        raw_dir=tmp_path,
    )
    assert len(frame) == 72
    assert len(calls) == 2
    assert all(call[1]["api_key"] == "test" for call in calls)
    assert (tmp_path / "aemet_15030_daily.json").exists()


def test_aemet_observations_normalise_daily_records_and_coordinates() -> None:
    daily_payload = [
        {
            "indicativo": "1234A",
            "fecha": "2026-08-20",
            "tmax": "30,2",
            "hrMin": "25",
            "racha": "40",
            "prec": "Ip",
        }
    ]
    inventory_payload = [
        {
            "indicativo": "1234A",
            "latitud": "432000N",
            "longitud": "081000W",
            "nombre": "Estación sintética",
            "provincia": "A Coruña",
        }
    ]
    assert parse_aemet_coordinate("432000N") == pytest.approx(43.333333, rel=1e-5)
    assert parse_aemet_coordinate("081000W") == pytest.approx(-8.166666, rel=1e-5)
    parsed = normalise_aemet_daily_payload(
        daily_payload,
        inventory_payload=inventory_payload,
        start_date="2026-08-20",
        end_date="2026-08-20",
    )
    assert parsed.loc[0, "tmax_vc"] == pytest.approx(30.2)
    assert parsed.loc[0, "prec_dia"] == 0.0
    assert parsed.loc[0, "station_name"] == "Estación sintética"


def test_aemet_observations_interpolate_to_grid() -> None:
    station_daily = pd.DataFrame(
        [
            {
                "station_id": "a",
                "fecha": pd.Timestamp("2026-08-20"),
                "lat": 42.5,
                "lon": -8.0,
                "tmax_vc": 30.0,
                "rhmin_vc": 25.0,
                "vmax_vc": 40.0,
                "prec_dia": 0.0,
            },
            {
                "station_id": "b",
                "fecha": pd.Timestamp("2026-08-20"),
                "lat": 42.6,
                "lon": -8.1,
                "tmax_vc": 28.0,
                "rhmin_vc": 35.0,
                "vmax_vc": 20.0,
                "prec_dia": 2.0,
            },
        ]
    )
    grid = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lat_centroid": [42.5, 42.6],
            "lon_centroid": [-8.0, -8.1],
        }
    )
    result = interpolate_aemet_daily_to_grid(station_daily, grid, neighbors=2)
    assert len(result) == 2
    assert set(result["cell_id"]) == {1, 2}
    assert result["tmax_vc"].tolist() == [30.0, 28.0]
    assert result["prec_dia"].tolist() == [0.0, 2.0]


def test_aemet_observation_client_fetches_daily_history_and_inventory(tmp_path) -> None:
    daily_payload = [
        {
            "indicativo": "1234A",
            "fecha": "2026-08-20",
            "tmax": "30",
            "hrMin": "25",
            "racha": "40",
            "prec": "0",
        }
    ]
    inventory_payload = [
        {"indicativo": "1234A", "latitud": "432000N", "longitud": "081000W"}
    ]
    client = AemetObservationClient(AemetObservationConfig(api_key="test", max_retries=1))
    calls: list[str] = []

    def fake_endpoint(endpoint: str) -> object:
        calls.append(endpoint)
        return daily_payload if "todasestaciones" in endpoint and "diarios" in endpoint else inventory_payload

    client.client.fetch_endpoint = fake_endpoint  # type: ignore[method-assign]
    frame = client.fetch_daily_history("2026-08-20", "2026-08-20", raw_dir=tmp_path)
    assert len(frame) == 1
    assert any("fechaini/2026-08-20" in endpoint for endpoint in calls)
    assert any(endpoint.endswith("/todasestaciones/") for endpoint in calls)
    assert (tmp_path / "aemet_station_inventory.json").exists()


def test_aemet_observation_client_splits_ranges_over_fifteen_days(tmp_path) -> None:
    daily_payload = [
        {
            "indicativo": "1234A",
            "fecha": day.strftime("%Y-%m-%d"),
            "tmax": "30",
            "hrMin": "25",
            "racha": "40",
            "prec": "0",
        }
        for day in pd.date_range("2026-08-01", periods=30, freq="D")
    ]
    inventory_payload = [
        {"indicativo": "1234A", "latitud": "432000N", "longitud": "081000W"}
    ]
    client = AemetObservationClient(AemetObservationConfig(api_key="test", max_retries=1))
    daily_calls: list[str] = []

    def fake_endpoint(endpoint: str) -> object:
        if "inventarioestaciones" in endpoint:
            return inventory_payload
        daily_calls.append(endpoint)
        return daily_payload

    client.client.fetch_endpoint = fake_endpoint  # type: ignore[method-assign]
    frame = client.fetch_daily_history("2026-08-01", "2026-08-30", raw_dir=tmp_path)

    assert split_aemet_date_range("2026-08-01", "2026-08-30") == [
        (pd.Timestamp("2026-08-01").date(), pd.Timestamp("2026-08-15").date()),
        (pd.Timestamp("2026-08-16").date(), pd.Timestamp("2026-08-30").date()),
    ]
    assert len(daily_calls) == 2
    assert len(frame) == 30
    assert frame["fecha"].nunique() == 30
    assert len(list(tmp_path.glob("aemet_daily_*.json"))) == 2


def test_aemet_observation_client_rejects_missing_dates() -> None:
    daily_payload = [
        {
            "indicativo": "1234A",
            "fecha": day.strftime("%Y-%m-%d"),
            "tmax": "30",
            "hrMin": "25",
            "racha": "40",
            "prec": "0",
        }
        for day in pd.date_range("2026-08-01", periods=14, freq="D")
    ]
    inventory_payload = [
        {"indicativo": "1234A", "latitud": "432000N", "longitud": "081000W"}
    ]
    client = AemetObservationClient(AemetObservationConfig(api_key="test", max_retries=1))

    def fake_endpoint(endpoint: str) -> object:
        return inventory_payload if "inventarioestaciones" in endpoint else daily_payload

    client.client.fetch_endpoint = fake_endpoint  # type: ignore[method-assign]
    with pytest.raises(AemetObservationError, match="Faltan 1 fechas"):
        client.fetch_daily_history("2026-08-01", "2026-08-15")


def test_parser_keeps_provider_run_separate_from_download_time() -> None:
    payload = _payload()
    payload["metadata"] = {"referenceTime": "2026-08-16T03:00:00Z"}
    parsed = parse_meteogalicia_payload(
        payload,
        downloaded_at="2026-08-16T03:07:00Z",
    )
    assert parsed["forecast_run_source"].unique().tolist() == ["provider"]
    assert parsed["forecast_run_at"].iloc[0] == pd.Timestamp("2026-08-16T03:00:00Z")
    assert parsed["downloaded_at"].iloc[0] == pd.Timestamp("2026-08-16T03:07:00Z")


def test_parser_reads_model_run_from_each_v5_hourly_value() -> None:
    payload = _payload()
    variables = payload["features"][0]["properties"]["days"][0]["variables"]
    for variable in variables:
        variable["model"] = "WRF"
        variable["grid"] = "1km"
        for value in variable["values"]:
            value["modelRun"] = "2026-08-17T00:00:00Z"

    parsed = parse_meteogalicia_payload(payload, downloaded_at="2026-08-17T00:10:00Z")
    assert parsed["forecast_run_source"].unique().tolist() == ["provider"]
    assert parsed["forecast_run_at"].unique().tolist() == [pd.Timestamp("2026-08-17T00:00:00Z")]
    assert parsed["grid"].unique().tolist() == ["1km"]


def test_validate_hourly_forecast_rejects_missing_values() -> None:
    parsed = parse_meteogalicia_payload(_payload())
    with pytest.raises(ForecastValidationError):
        validate_hourly_forecast(
            parsed,
            expected_start="2026-08-17T10:00:00Z",
            expected_end="2026-08-17T12:00:00Z",
        )


def test_parser_treats_provider_sentinel_as_missing() -> None:
    payload = _payload()
    payload["features"][0]["properties"]["days"][0]["variables"][0]["values"][0]["value"] = -9999
    parsed = parse_meteogalicia_payload(payload)
    assert np.isnan(parsed.iloc[0]["temperature_c"])


def test_assign_forecast_to_grid_maps_each_cell_without_hourly_cartesian_search() -> None:
    forecast = _hourly_forecast().copy()
    forecast["forecast_lat"] = np.where(forecast["cell_id"] == 1, 42.5001, 42.5021)
    forecast = forecast.drop(columns=["cell_id"])
    grid = pd.DataFrame(
        {
            "cell_id": [10, 11, 12],
            "lat_centroid": [42.5000, 42.5002, 42.5020],
            "lon_centroid": [-8.0, -8.0, -8.0],
        }
    )
    assigned = assign_forecast_to_grid(forecast, grid)
    assert set(assigned["cell_id"]) == {10, 11, 12}
    assert assigned.groupby("cell_id")["valid_time"].nunique().min() == 72
    assert assigned.groupby("cell_id")["source_distance_km"].first().max() < 1.0


def test_injected_forecast_with_old_cell_ids_is_reassigned_to_operational_grid() -> None:
    import scripts.run_daily_inference as daily_inference

    forecast = _hourly_forecast()
    forecast["forecast_lat"] = 42.5
    grid = pd.DataFrame(
        {
            "cell_id": [10, 11],
            "lat_centroid": [42.501, 42.502],
            "lon_centroid": [-8.0, -8.0],
        }
    )

    aligned = daily_inference._align_forecast_to_grid(forecast, grid)

    assert set(aligned["cell_id"]) == {10, 11}
    assert len(aligned) == 2 * 72
    assert aligned.groupby("cell_id")["valid_time"].nunique().eq(72).all()


def test_client_batches_provider_requests_at_twenty_points(tmp_path) -> None:
    calls: list[dict] = []
    progress: list[tuple[int, int]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return _payload()

    def fake_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        calls.append(params)
        return FakeResponse()

    points = [ForecastPoint(f"p{i}", 42.5 + i / 10_000, -8.0) for i in range(21)]
    client = MeteoGaliciaClient(
        ForecastConfig(api_key="test", max_retries=1),
        request_get=fake_get,
    )
    client.fetch_points(
        points,
        start_time="2026-08-17T00:00:00+02:00",
        end_time="2026-08-17T01:00:00+02:00",
        issued_at="2026-08-16T03:00:00Z",
        raw_dir=tmp_path,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    assert len(calls) == 2
    assert all(len(params["coords"].split(";")) <= 20 for params in calls)
    assert len(list(tmp_path.glob("*.json"))) == 4
    assert '"units"' in next(tmp_path.glob("*.metadata.json")).read_text()
    assert progress == [(1, 2), (2, 2)]


def test_client_uses_meteosix_v5_and_auto_adjust_parameter() -> None:
    calls: list[tuple[str, dict]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return _payload()

    def fake_get(url: str, *, params: dict, timeout: int) -> FakeResponse:
        calls.append((url, params))
        return FakeResponse()

    client = MeteoGaliciaClient(
        ForecastConfig(api_key="test", auto_adjust_position=False),
        request_get=fake_get,
    )
    client.fetch_points(
        [ForecastPoint("p1", 42.5, -8.0)],
        start_time="2026-08-17T00:00:00+02:00",
        end_time="2026-08-17T01:00:00+02:00",
    )
    assert calls[0][0].endswith("/apiv5/getNumericForecastInfo")
    assert calls[0][1]["autoAdjustPosition"] == "false"
    assert "units" not in calls[0][1]
    assert calls[0][1]["startTime"] == "2026-08-17T00:00:00"
    assert calls[0][1]["endTime"] == "2026-08-17T01:00:00"


def test_meteosix_errors_redact_api_key_from_exception_chain() -> None:
    secret = "meteosix-secret-that-must-not-leak"

    def fake_get(url: str, *, params: dict, timeout: int) -> object:
        raise requests.ConnectionError(
            "connection failed for "
            f"https://servizos.meteogalicia.gal/apiv5/getNumericForecastInfo?API_KEY={secret}"
        )

    client = MeteoGaliciaClient(
        ForecastConfig(api_key=secret, max_retries=1),
        request_get=fake_get,
    )

    with pytest.raises(Exception) as raised:
        client._request_payload({"API_KEY": secret})

    rendered = " ".join(
        [str(raised.value), str(raised.value.__cause__), str(raised.value.__context__)]
    )
    assert secret not in rendered
    assert "API_KEY=<redacted>" in rendered


def test_latest_archive_is_explicitly_stale(tmp_path) -> None:
    forecast = _hourly_forecast()
    save_forecast_parquet(forecast, tmp_path / "forecast_20260816T030000Z.parquet")
    loaded, issued_at, stale = load_latest_forecast(
        tmp_path,
        required_start="2026-08-17T00:00:00Z",
        required_end="2026-08-19T21:00:00Z",
    )
    assert len(loaded) == len(forecast)
    assert issued_at == pd.Timestamp("2026-08-16T03:00:00Z")
    assert stale is True


def test_operational_fetch_falls_back_from_wrf_1km_to_04km(tmp_path, monkeypatch) -> None:
    import scripts.run_daily_inference as daily_inference

    class FakeClient:
        def __init__(self, config: ForecastConfig) -> None:
            self.config = config

        def fetch_points(self, points, **kwargs) -> pd.DataFrame:
            if self.config.grid == "1km":
                raise ForecastValidationError("WRF 1km sintético incompleto")
            frame = _hourly_forecast().drop(columns=["cell_id"]).copy()
            frame["grid"] = self.config.grid
            return frame

    monkeypatch.setenv("METEOGALICIA_API_KEY", "test")
    monkeypatch.setenv("METEOGALICIA_GRIDS", "1km,04km")
    monkeypatch.setattr(daily_inference, "MeteoGaliciaClient", FakeClient)
    grid = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lat_centroid": [42.501, 42.502],
            "lon_centroid": [-8.0, -8.0],
        }
    )

    forecast, _, quality, report = daily_inference.fetch_fresh_forecast(
        grid,
        issue_time="2026-08-16T05:00:00+02:00",
        forecast_dir=tmp_path,
    )
    assert quality == "fresh_fallback"
    assert report["selected_grid"] == "04km"
    assert forecast["forecast_grid"].unique().tolist() == ["04km"]


def test_operational_provider_can_use_aemet_without_meteogalicia_key(tmp_path, monkeypatch) -> None:
    import scripts.run_daily_inference as daily_inference

    class FakeAemetClient:
        def __init__(self, config) -> None:
            self.config = config

        def fetch_points(self, points, **kwargs) -> pd.DataFrame:
            frame = _hourly_forecast().drop(columns=["cell_id"]).copy()
            frame["provider"] = "aemet"
            frame["model"] = "AEMET-municipal"
            return frame

    monkeypatch.delenv("METEOGALICIA_API_KEY", raising=False)
    monkeypatch.setenv("AEMET_API_KEY", "test")
    monkeypatch.setenv("FORECAST_PROVIDER", "auto")
    monkeypatch.setattr(daily_inference, "AemetClient", FakeAemetClient)
    grid = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lat_centroid": [42.501, 42.502],
            "lon_centroid": [-8.0, -8.0],
        }
    )

    forecast, _, quality, report = daily_inference.fetch_fresh_forecast(
        grid,
        issue_time="2026-08-16T05:00:00+02:00",
        forecast_dir=tmp_path,
    )

    assert len(forecast) == 144
    assert quality == "fresh_aemet_degraded"
    assert report["provider"] == "aemet"
    assert forecast["forecast_grid"].unique().tolist() == ["municipal"]


def test_critical_window_uses_local_galicia_hours() -> None:
    forecast = _hourly_forecast()
    daily = aggregate_hourly_forecast(forecast)
    row = daily[(daily.cell_id == 1) & (daily.fecha == pd.Timestamp("2026-08-17"))].iloc[0]
    assert row.tmax_vc == 32.0
    assert row.rhmin_vc == 25.0
    assert row.vmax_vc == 20.0
    assert row.prec_dia == 0.0


def test_operational_features_respect_issue_time_and_build_three_horizons() -> None:
    forecast = _hourly_forecast()
    history_rows = []
    for cell_id in [1, 2]:
        for date in pd.date_range("2026-08-09", "2026-08-16", freq="D"):
            history_rows.append(
                {
                    "cell_id": cell_id,
                    "fecha": date,
                    "tmax_vc": 18.0,
                    "rhmin_vc": 70.0,
                    "vmax_vc": 10.0,
                    "prec_dia": 2.0,
                }
            )
    # Este registro futuro no puede contaminar ninguna feature de producción.
    history_rows.append(
        {
            "cell_id": 1,
            "fecha": pd.Timestamp("2026-08-17"),
            "tmax_vc": 99.0,
            "rhmin_vc": 1.0,
            "vmax_vc": 99.0,
            "prec_dia": 999.0,
        }
    )
    history = pd.DataFrame(history_rows)
    grid = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lat_centroid": [42.5, 42.501],
            "lon_centroid": [-8.0, -8.0],
            "altitud_media": [100.0, 200.0],
            "pendiente_media": [5.0, 6.0],
            "orientacion_media": [180.0, 180.0],
            "combustible_pct_forestal": [50.0, 60.0],
        }
    )
    features = build_operational_features(
        forecast,
        history,
        grid,
        issue_time="2026-08-16T05:00:00+02:00",
    )
    assert set(features.horizon_days) == {1, 2, 3}
    t1 = features[(features.cell_id == 1) & (features.horizon_days == 1)].iloc[0]
    assert t1.tmax_vc == 32.0
    # La fila diaria del día de emisión (16/08) queda fuera: aún no está
    # completa a las 05:00. Solo entran 14/08 y 15/08 del histórico.
    assert t1.prec_acum_3d == 4.0
    assert t1.prec_dia == 0.0
    assert t1.tmax_media_7d == 18.0


def test_risk_outputs_are_monotonic_and_labelled() -> None:
    frame = pd.DataFrame({"cell_id": np.arange(100)})
    scored = add_risk_outputs(frame, np.linspace(0.01, 0.9, 100))
    assert scored["prob_risk"].is_monotonic_increasing
    assert scored["percentil_riesgo"].is_monotonic_increasing
    assert set(scored["risk_level"]) == {"bajo", "moderado", "alto", "extremo"}
    assert "recommended_action" in scored.columns
    assert "preposicion_prioritaria" in set(scored["recommended_action"])


def test_weather_state_requires_complete_recent_history() -> None:
    rows = [
        {
            "cell_id": 1,
            "fecha": date,
            "tmax_vc": 25.0,
            "rhmin_vc": 40.0,
            "vmax_vc": 20.0,
            "prec_dia": 0.0,
        }
        for date in pd.date_range("2026-07-17", "2026-08-15", freq="D")
    ]
    state = pd.DataFrame(rows)
    report = validate_weather_state(
        state,
        issue_time="2026-08-16T05:00:00+02:00",
        required_cells=[1],
        history_days=30,
    )
    assert report["covered_cells"] == 1

    incomplete = state.iloc[:-2]
    with pytest.raises(WeatherStateError):
        validate_weather_state(
            incomplete,
            issue_time="2026-08-16T05:00:00+02:00",
            required_cells=[1],
            history_days=30,
        )

    with pytest.raises(WeatherStateError):
        validate_weather_state(
            state.iloc[:-1],
            issue_time="2026-08-16T05:00:00+02:00",
            required_cells=[1],
            history_days=30,
        )


def test_weather_state_rejects_missing_values_and_sentinels() -> None:
    state = pd.DataFrame(
        [
            {
                "cell_id": 1,
                "fecha": date,
                "tmax_vc": -9999.0 if date.day == 17 else 25.0,
                "rhmin_vc": 40.0,
                "vmax_vc": 20.0,
                "prec_dia": 0.0,
            }
            for date in pd.date_range("2026-07-17", "2026-08-15", freq="D")
        ]
    )
    with pytest.raises(WeatherStateError):
        validate_weather_state(
            state,
            issue_time="2026-08-16T05:00:00+02:00",
            required_cells=[1],
            history_days=30,
        )


def test_run_lock_is_exclusive_and_released(tmp_path) -> None:
    lock_path = tmp_path / "run.lock"
    with run_lock(lock_path):
        with pytest.raises(RunLockError):
            with run_lock(lock_path):
                pass
    assert not lock_path.exists()


def test_operational_health_check_validates_manifest_and_output(tmp_path) -> None:
    from scripts.check_operational_run import check_run
    from src.operational.artifacts import atomic_write_json, sha256_file

    output = tmp_path / "predictions.parquet"
    manifest = tmp_path / "predictions.manifest.json"
    frame = pd.DataFrame(
        {
            "horizon_days": [1, 1, 2, 2, 3, 3],
            "cell_id": [1, 2, 1, 2, 1, 2],
            "prob_risk": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    frame.to_parquet(output, index=False)
    atomic_write_json(
        {
            "horizons": [1, 2, 3],
            "forecast_quality": "fresh",
            "forecast_age_hours": 2.0,
            "forecast_coverage": {"minimum_ratio": 1.0},
            "output_sha256": sha256_file(output),
        },
        manifest,
    )
    report = check_run(output, manifest_path=manifest)
    assert report["status"] == "ok"
    assert report["rows"] == 6


def test_local_state_simulation_is_explicit_and_local_only(monkeypatch) -> None:
    import scripts.run_daily_inference as daily_inference

    issue_time = daily_inference._issue_timestamp("2026-08-30T05:00:00+02:00", None)
    monkeypatch.setenv("PIPELINE_ENVIRONMENT", "local")
    monkeypatch.setenv("LOCAL_SIMULATION_MODE", "true")
    monkeypatch.setenv("LOCAL_SIMULATION_AS_OF_DATE", "2026-08-26")

    settings = daily_inference._local_simulation_settings(issue_time)
    assert settings["enabled"] is True
    assert settings["mode"] == "local_state_simulation"
    assert settings["as_of_date"] == pd.Timestamp("2026-08-26")
    assert settings["simulated_issue_time"] == pd.Timestamp("2026-08-27T05:00:00+02:00")

    monkeypatch.setenv("PIPELINE_ENVIRONMENT", "production")
    with pytest.raises(daily_inference.ForecastError, match="solo puede activarse"):
        daily_inference._local_simulation_settings(issue_time)


def test_synthetic_operational_inference_is_reproducible(tmp_path, monkeypatch) -> None:
    import scripts.run_daily_inference as daily_inference

    class FakeModel:
        def __init__(self, horizon: int) -> None:
            self.horizon = horizon

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            probability = np.clip(
                features["tmax_vc"].to_numpy(float) / 100.0 + self.horizon / 1000.0,
                0.0,
                1.0,
            )
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(
        daily_inference,
        "load_horizon_model",
        lambda horizon, model_dir: FakeModel(horizon),
    )
    history = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "fecha": date,
                "tmax_vc": 18.0,
                "rhmin_vc": 70.0,
                "vmax_vc": 10.0,
                "prec_dia": 2.0,
            }
            for cell_id in [1, 2]
            for date in pd.date_range("2026-08-09", "2026-08-15", freq="D")
        ]
    )
    grid = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lat_centroid": [42.5, 42.501],
            "lon_centroid": [-8.0, -8.0],
            "altitud_media": [100.0, 200.0],
            "pendiente_media": [5.0, 6.0],
            "orientacion_media": [180.0, 180.0],
            "combustible_pct_forestal": [50.0, 60.0],
        }
    )
    kwargs = {
        "issue_time": "2026-08-16T05:00:00+02:00",
        "forecast_df": _hourly_forecast(),
        "history_df": history,
        "grid_df": grid,
        "model_dir": tmp_path,
    }
    first = daily_inference.run_daily_inference_pipeline(
        output_path=tmp_path / "first.parquet", **kwargs
    )
    second = daily_inference.run_daily_inference_pipeline(
        output_path=tmp_path / "second.parquet", **kwargs
    )
    columns = ["cell_id", "fecha", "horizon_days", "prob_risk", "forecast_quality"]
    first_core = first[columns].sort_values(["horizon_days", "cell_id"]).reset_index(drop=True)
    second_core = second[columns].sort_values(["horizon_days", "cell_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_core, second_core)
    assert len(first) == 6
    assert set(first["horizon_days"]) == {1, 2, 3}
    assert set(first["forecast_quality"]) == {"fresh"}
