from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_egif48_test import (
    _parse_years,
    _roc_auc_in_season,
    _roc_auc_within_day,
    _risk_report,
)


def test_parse_years_accepts_single_year_and_ranges() -> None:
    assert _parse_years("2023") == (2023,)
    assert _parse_years("2019-2021") == (2019, 2020, 2021)
    assert _parse_years("2019, 2021") == (2019, 2021)
    with pytest.raises(ValueError):
        _parse_years("2021-2019")


def test_roc_helpers_measure_season_and_spatial_signal() -> None:
    dates = np.array(["2023-01-01", "2023-07-01", "2023-07-01", "2023-08-01"], dtype="datetime64[D]")
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    score = np.array([0.1, 0.2, 0.8, 0.9])
    days = dates.astype(np.int64)

    assert _roc_auc_in_season(y, score, dates) == pytest.approx(1.0)
    assert _roc_auc_within_day(y, score, days) == 1.0


def test_risk_report_has_four_ordered_levels() -> None:
    probability = np.linspace(0.01, 0.99, 200)
    y = (probability > 0.8).astype(np.int8)

    report = _risk_report(y, probability)

    assert list(report["thresholds"]) == ["moderado", "alto", "extremo"]
    assert [row["nivel"] for row in report["levels"]] == [
        "Bajo",
        "Moderado",
        "Alto",
        "Extremo",
    ]
