from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.features.canonical_contract import (
    CANONICAL_FEATURES,
    CANONICAL_FEATURE_SCHEMA_VERSION,
)
from src.features.operational_features import (
    aggregate_hourly_forecast,
    build_historical_horizon_dataset,
)
from src.ingestion.aemet_forecast import AemetPoint, parse_aemet_payload
from src.ingestion.weather_state import normalise_weather_state
from src.models.forecast_risk_model import ForecastRiskModel, load_horizon_model


def _canonical_rows(days: int = 35, cells: tuple[int, ...] = (1,)) -> pd.DataFrame:
    rows = []
    for cell_id in cells:
        for index, fecha in enumerate(pd.date_range("2026-01-01", periods=days, freq="D")):
            row = {
                "cell_id": cell_id,
                "fecha": fecha,
                "target_ignicion": int(index == days - 1 and cell_id == cells[0]),
            }
            for position, column in enumerate(CANONICAL_FEATURES):
                row[column] = float(position + index / 100)
            rows.append(row)
    return pd.DataFrame(rows)


def test_canonical_contract_is_versioned_and_has_50_predictors() -> None:
    assert CANONICAL_FEATURE_SCHEMA_VERSION == "egif-2d-v1"
    assert len(CANONICAL_FEATURES) == 50
    assert len(set(CANONICAL_FEATURES)) == 50


def test_historical_targets_are_explicit_per_horizon() -> None:
    data = _canonical_rows()
    outputs = build_historical_horizon_dataset(data, horizons=(1, 2, 3))
    assert set(outputs) == {1, 2, 3}
    for horizon, output in outputs.items():
        assert output["feature_schema_version"].eq(CANONICAL_FEATURE_SCHEMA_VERSION).all()
        assert output["target_t%s" % horizon].equals(output["target"])
        assert (output["target_date"] - output["issue_date"]).eq(
            pd.Timedelta(days=horizon)
        ).all()


def test_historical_targets_are_shifted_to_the_future_day() -> None:
    data = _canonical_rows(days=4, cells=(1,)).copy()
    data["target_ignicion"] = [0, 1, 0, 0]
    outputs = build_historical_horizon_dataset(data, horizons=(1, 2))
    t1 = outputs[1].set_index("issue_date")
    t2 = outputs[2].set_index("issue_date")
    assert t1.loc[pd.Timestamp("2026-01-01"), "target"] == 1
    assert t2.loc[pd.Timestamp("2026-01-01"), "target"] == 0


def test_forecast_aggregation_uses_local_12_to_18_window() -> None:
    rows = []
    local_times = pd.date_range(
        "2026-08-17", "2026-08-17 23:00", freq="h", tz="Europe/Madrid"
    )
    for local_time in local_times:
        critical = 12 <= local_time.hour <= 18
        rows.append(
            {
                "cell_id": 1,
                "valid_time": local_time.tz_convert("UTC"),
                "temperature_c": 35.0 if critical else 10.0,
                "relative_humidity_pct": 20.0 if critical else 80.0,
                "precipitation_mm": 1.0,
                "wind_speed_kmh": 40.0 if critical else 5.0,
            }
        )
    result = aggregate_hourly_forecast(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["temperature_max_12_18h"] == 35.0
    assert row["relative_humidity_min_12_18h"] == 20.0
    assert row["wind_speed_max_12_18h"] == 40.0
    assert row["precipitation_sum"] == 24.0


def test_old_weather_state_is_marked_as_proxy_for_canonical_features() -> None:
    state = normalise_weather_state(
        pd.DataFrame(
            {
                "cell_id": [1],
                "fecha": ["2026-08-17"],
                "tmax_vc": [30.0],
                "rhmin_vc": [25.0],
                "vmax_vc": [35.0],
                "prec_dia": [0.0],
            }
        )
    )
    assert state.loc[0, "state_feature_quality"] == "legacy_proxy"
    assert state.loc[0, "temperature_max_12_18h"] == 30.0
    assert state.loc[0, "precipitation_sum"] == 0.0


def test_aemet_probability_of_rain_is_not_millimetres() -> None:
    payload = [
        {
            "id": "15030",
            "prediccion": {
                "dia": [
                    {
                        "fecha": "2026-08-17T00:00:00",
                        "temperatura": [{"periodo": "00-24", "value": 25}],
                        "humedadRelativa": [{"periodo": "00-24", "value": 50}],
                        "probPrecipitacion": [{"periodo": "00-24", "value": 90}],
                        "vientoAndRachaMax": [
                            {"periodo": "00-24", "direccion": ["S"], "velocidad": [10]}
                        ],
                    }
                ]
            },
        }
    ]
    result = parse_aemet_payload(
        payload,
        AemetPoint("15030", 43.36, -8.41),
        start_time="2026-08-17T00:00:00+02:00",
        end_time="2026-08-17T23:00:00+02:00",
    )
    assert result["precipitation_mm"].isna().all()


def test_canonical_model_uses_its_stored_feature_order() -> None:
    class FakeModel:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            frame = frame[list(CANONICAL_FEATURES)]
            assert list(frame.columns) == list(CANONICAL_FEATURES)
            return np.tile([[0.75, 0.25]], (len(frame), 1))

    class Identity:
        def transform(self, values: np.ndarray) -> np.ndarray:
            return values

    model = ForecastRiskModel(
        base_model=FakeModel(),
        calibrator=Identity(),
        feature_columns=list(CANONICAL_FEATURES),
        horizon_days=1,
        metadata={"feature_schema_version": CANONICAL_FEATURE_SCHEMA_VERSION},
        feature_schema_version=CANONICAL_FEATURE_SCHEMA_VERSION,
        model_family="canonical-egif-2d",
    )
    features = pd.DataFrame({column: [1.0] for column in CANONICAL_FEATURES})
    probabilities = model.predict_proba(features)
    assert probabilities.tolist() == [[0.75, 0.25]]


def test_canonical_training_serialises_three_independent_models(tmp_path, monkeypatch) -> None:
    import src.models.canonical_training as training

    if training.LGBMClassifier is None:
        return
    dataset_dir = tmp_path / "egif"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text(
        json.dumps({"predictor_columns": list(CANONICAL_FEATURES)}), encoding="utf-8"
    )
    for year in (2019, 2020, 2021, 2022, 2023):
        data = _canonical_rows(days=12, cells=(1, 2, 3, 4)).copy()
        data["fecha"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(
            data.groupby("cell_id").cumcount(), unit="D"
        )
        data["target_ignicion"] = (np.arange(len(data)) % 7 == 0).astype("int8")
        data.to_parquet(dataset_dir / f"dataset_{year}.parquet", index=False)

    reports = training.train_all_canonical_horizons(
        dataset_dir,
        output_dir=tmp_path / "models",
        n_estimators=5,
        batch_size=100,
        bootstrap_samples=2,
        negative_ratio=2,
    )
    assert set(reports) == {1, 2, 3}
    for horizon in (1, 2, 3):
        assert (tmp_path / "models" / f"forecast_risk_egif_t{horizon}.joblib").exists()
    monkeypatch.setenv("FORECAST_MODEL_FAMILY", "canonical")
    loaded = load_horizon_model(1, model_dir=tmp_path / "models")
    assert loaded.feature_schema_version == CANONICAL_FEATURE_SCHEMA_VERSION


def test_canonical_inference_is_reproducible_with_synthetic_forecast(tmp_path, monkeypatch) -> None:
    import scripts.run_daily_inference as daily

    class FakeCanonicalModel:
        feature_schema_version = CANONICAL_FEATURE_SCHEMA_VERSION
        model_family = "canonical-egif-2d-test"
        metadata = {
            "feature_schema_version": CANONICAL_FEATURE_SCHEMA_VERSION,
            "feature_columns": list(CANONICAL_FEATURES),
        }

        def __init__(self, horizon: int) -> None:
            self.horizon_days = horizon

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            frame = frame[list(CANONICAL_FEATURES)]
            assert list(frame.columns) == list(CANONICAL_FEATURES)
            score = np.full(len(frame), 0.05 + self.horizon_days / 100.0)
            return np.column_stack([1.0 - score, score])

    issue_time = pd.Timestamp("2026-08-16T05:00:00+02:00")
    grid = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lat_centroid": [42.50, 42.51],
            "lon_centroid": [-8.00, -8.01],
            **{
                column: [1.0, 2.0]
                for column in set(CANONICAL_FEATURES) - set(
                    (
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
                        "precipitation_sum_3d",
                        "precipitation_sum_7d",
                        "precipitation_sum_14d",
                        "precipitation_sum_30d",
                        "temperature_mean_7d",
                        "relative_humidity_mean_7d",
                        "wind_speed_mean_7d",
                        "relative_humidity_mean_14d",
                        "consecutive_dry_days",
                    )
                )
            },
        }
    )
    history = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "fecha": fecha,
                "tmax_vc": 25.0,
                "rhmin_vc": 50.0,
                "vmax_vc": 10.0,
                "prec_dia": 0.0,
            }
            for cell_id in (1, 2)
            for fecha in pd.date_range("2026-07-17", "2026-08-15", freq="D")
        ]
    )
    rows = []
    for cell_id in (1, 2):
        for local_time in pd.date_range(
            "2026-08-17", "2026-08-19 23:00", freq="h", tz="Europe/Madrid"
        ):
            rows.append(
                {
                    "cell_id": cell_id,
                    "forecast_lat": 42.49 + cell_id / 100.0,
                    "forecast_lon": -8.00,
                    "valid_time": local_time.tz_convert("UTC"),
                    "temperature_c": 32.0,
                    "relative_humidity_pct": 25.0,
                    "precipitation_mm": 0.0,
                    "wind_speed_kmh": 20.0,
                }
            )
    forecast = pd.DataFrame(rows)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for horizon in (1, 2, 3):
        (model_dir / f"forecast_risk_egif_t{horizon}.joblib").touch()
    monkeypatch.setenv("CANONICAL_GRID_CELLS", "2")
    monkeypatch.setenv("FORECAST_MODEL_FAMILY", "canonical")
    monkeypatch.setattr(
        daily,
        "load_horizon_model",
        lambda horizon, model_dir: FakeCanonicalModel(horizon),
    )

    first = daily.run_daily_inference_pipeline(
        issue_time=issue_time,
        forecast_df=forecast,
        history_df=history,
        grid_df=grid,
        model_dir=model_dir,
        output_path=tmp_path / "predictions.parquet",
        use_lock=False,
    )
    second = daily.run_daily_inference_pipeline(
        issue_time=issue_time,
        forecast_df=forecast,
        history_df=history,
        grid_df=grid,
        model_dir=model_dir,
        output_path=tmp_path / "predictions.parquet",
        use_lock=False,
    )
    columns = ["cell_id", "horizon_days", "prob_risk", "feature_schema_version"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(["horizon_days", "cell_id"]).reset_index(drop=True),
        second[columns].sort_values(["horizon_days", "cell_id"]).reset_index(drop=True),
    )
    assert set(first["horizon_days"]) == {1, 2, 3}
    assert first["feature_schema_version"].eq(CANONICAL_FEATURE_SCHEMA_VERSION).all()
