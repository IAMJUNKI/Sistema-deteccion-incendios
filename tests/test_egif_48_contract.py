from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entrenamiento.calibracion import PlattProbabilityCalibrator
from src.features.canonical_contract import (
    CANONICAL_FEATURES,
    EGIF_48_EXCLUDED_FEATURES,
    EGIF_48_FEATURE_CONTRACT_VERSION,
    EGIF_48_FEATURES,
    load_feature_columns_for_contract,
    validate_feature_contract,
)
from src.features.operational_benchmark import (
    MEMORY_COLUMNS,
    OPERATIONAL_ALIGNMENT_VERSION,
    OPERATIONAL_TEMPORAL_SEMANTICS,
)
from src.features.operational_features import build_historical_horizon_dataset
from src.models.forecast_risk_model import load_horizon_model


def _rows() -> pd.DataFrame:
    rows = []
    for index, fecha in enumerate(pd.date_range("2020-01-01", periods=5, freq="D")):
        row = {
            "cell_id": 1,
            "fecha": fecha,
            "target_ignicion": int(index == 1),
        }
        for position, column in enumerate(CANONICAL_FEATURES):
            row[column] = float(position + index)
        rows.append(row)
    return pd.DataFrame(rows)


def test_egif_48_contract_is_exactly_the_50_variable_contract_minus_two() -> None:
    assert EGIF_48_FEATURE_CONTRACT_VERSION == "egif-2d-48-v1"
    assert len(EGIF_48_FEATURES) == 48
    assert set(EGIF_48_FEATURES) == set(CANONICAL_FEATURES) - EGIF_48_EXCLUDED_FEATURES
    assert "fire_weather_index" not in EGIF_48_FEATURES
    validate_feature_contract(EGIF_48_FEATURES, EGIF_48_FEATURE_CONTRACT_VERSION)
    assert load_feature_columns_for_contract(EGIF_48_FEATURE_CONTRACT_VERSION) == list(
        EGIF_48_FEATURES
    )


def test_egif_48_rejects_removed_predictors() -> None:
    with pytest.raises(ValueError, match="no coincide"):
        validate_feature_contract(
            [*EGIF_48_FEATURES, "precipitation_sum"],
            EGIF_48_FEATURE_CONTRACT_VERSION,
        )


def test_perfect_benchmark_uses_target_day_features_and_backdated_issue() -> None:
    data = _rows()
    data.loc[data["fecha"] == pd.Timestamp("2020-01-02"), "temperature_max"] = 999.0
    output = build_historical_horizon_dataset(data, horizons=(1,))[1]
    row = output.loc[output["target_date"] == pd.Timestamp("2020-01-02")].iloc[0]
    assert row["issue_date"] == pd.Timestamp("2020-01-01")
    assert row["target"] == 1
    assert row["temperature_max"] == 999.0
    assert row["target_alignment"] == "target_day_features_issue_date_minus_horizon"


def test_platt_calibrator_is_fitted_separately_from_prior_correction() -> None:
    y = np.array([0, 0, 0, 0, 1, 1], dtype=np.int8)
    raw = np.array([0.01, 0.03, 0.08, 0.15, 0.40, 0.70])
    calibrator = PlattProbabilityCalibrator(neg_sampling_rate=0.01).fit(y, raw)
    transformed = calibrator.transform(raw)
    assert transformed.shape == raw.shape
    assert np.all((transformed >= 0.0) & (transformed <= 1.0))
    assert np.all(np.diff(transformed) >= 0.0)


def test_incomplete_feature_rows_are_excluded_from_training() -> None:
    from src.models.canonical_training import _has_complete_feature_row

    frame = pd.DataFrame([{column: 1.0 for column in EGIF_48_FEATURES} for _ in range(2)])
    frame.loc[1, "precipitation_sum_3d"] = np.nan
    mask = _has_complete_feature_row(
        frame, list(EGIF_48_FEATURES), EGIF_48_FEATURE_CONTRACT_VERSION
    )
    assert mask.tolist() == [True, False]


def test_egif_48_training_writes_three_contract_bound_artifacts(tmp_path) -> None:
    import json

    import src.models.canonical_training as training

    if training.LGBMClassifier is None:
        pytest.skip("LightGBM no está instalado")
    dataset_dir = tmp_path / "egif"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "predictor_columns": list(CANONICAL_FEATURES),
                "alignment_version": OPERATIONAL_ALIGNMENT_VERSION,
                "training_temporal_semantics": OPERATIONAL_TEMPORAL_SEMANTICS,
                "operational_feature_contract_version": EGIF_48_FEATURE_CONTRACT_VERSION,
                "memory_columns_recomputed": list(MEMORY_COLUMNS),
            }
        ),
        encoding="utf-8",
    )
    for year in (2019, 2020, 2021, 2022, 2023):
        frame = _rows().copy()
        frame["fecha"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(
            frame.groupby("cell_id").cumcount(), unit="D"
        )
        frame.to_parquet(dataset_dir / f"dataset_{year}.parquet", index=False)

    reports = training.train_all_egif48_horizons(
        dataset_dir,
        output_dir=tmp_path / "models",
        n_estimators=3,
        batch_size=10,
        bootstrap_samples=1,
        negative_ratio=2,
    )
    assert set(reports) == {1, 2, 3}
    for horizon in (1, 2, 3):
        artifact = tmp_path / "models" / f"forecast_risk_egif_48_t{horizon}.joblib"
        metadata = tmp_path / "models" / f"forecast_risk_egif_48_t{horizon}.json"
        assert artifact.exists()
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        assert payload["feature_contract_version"] == EGIF_48_FEATURE_CONTRACT_VERSION
        assert payload["calibration_method"] == "prior_correction+platt"
        assert payload["calibration_year"] == 2021
    loaded = load_horizon_model(1, model_dir=tmp_path / "models", model_family="egif_48")
    assert loaded.feature_schema_version == EGIF_48_FEATURE_CONTRACT_VERSION
    assert len(loaded.feature_columns) == 48


def test_aligned_50_control_is_separate_from_published_50_family(tmp_path) -> None:
    import json

    import src.models.canonical_training as training

    if training.LGBMClassifier is None:
        pytest.skip("LightGBM no está instalado")
    dataset_dir = tmp_path / "egif_operational"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text(
        json.dumps(
            {
                "predictor_columns": list(CANONICAL_FEATURES),
                "alignment_version": OPERATIONAL_ALIGNMENT_VERSION,
                "training_temporal_semantics": OPERATIONAL_TEMPORAL_SEMANTICS,
                "operational_feature_contract_version": EGIF_48_FEATURE_CONTRACT_VERSION,
                "memory_columns_recomputed": list(MEMORY_COLUMNS),
            }
        ),
        encoding="utf-8",
    )
    for year in (2019, 2020, 2021, 2022, 2023):
        frame = _rows().copy()
        frame["fecha"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(
            frame.groupby("cell_id").cumcount(), unit="D"
        )
        frame.to_parquet(dataset_dir / f"dataset_{year}.parquet", index=False)

    training.train_all_aligned50_horizons(
        dataset_dir,
        output_dir=tmp_path / "models",
        n_estimators=3,
        batch_size=10,
        bootstrap_samples=1,
        negative_ratio=2,
    )
    for horizon in (1, 2, 3):
        assert (
            tmp_path / "models" / f"forecast_risk_egif_50_aligned_t{horizon}.joblib"
        ).exists()
