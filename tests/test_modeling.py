import numpy as np
import pytest

from src.modeling.data import sample_years_for_training, split_name_for_year
from src.modeling.evaluation import evaluate_binary_predictions
from src.modeling.features import resolve_feature_set


def test_feature_sets_remove_only_candidates_and_reject_leakage() -> None:
    predictors = ["slope_mean", "roughness_mean", "month", "temperature_max"]

    assert resolve_feature_set(predictors, "sin_topografia_redundante") == [
        "month",
        "slope_mean",
        "temperature_max",
    ]
    with pytest.raises(ValueError, match="prohibidos"):
        resolve_feature_set(["temperature_max", "year"])


def test_temporal_protocol_is_fixed() -> None:
    assert split_name_for_year(2020) == "train"
    assert split_name_for_year(2022) == "validation"
    assert split_name_for_year(2023) == "test"
    with pytest.raises(ValueError):
        split_name_for_year(2024)


def test_metrics_include_recall_at_top_fraction() -> None:
    metrics = evaluate_binary_predictions(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), top_fraction=0.5
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["recall_at_top_fraction"] == 1.0


def test_sampling_rejects_an_invalid_modulus() -> None:
    with pytest.raises(ValueError, match="negative_cell_modulus"):
        sample_years_for_training(None, [], [], negative_cell_modulus=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="negative_cell_remainder"):
        sample_years_for_training(None, [], [], negative_cell_remainder=25)  # type: ignore[arg-type]
