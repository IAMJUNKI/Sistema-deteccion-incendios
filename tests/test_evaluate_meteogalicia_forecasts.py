from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.evaluate_meteogalicia_forecasts import evaluate_forecasts, main
from scripts.plot_meteogalicia_evaluation import plot_direct_case


def _hourly_forecast() -> pd.DataFrame:
    times = pd.date_range("2026-09-01 00:00", periods=72, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "cell_id": 1,
            "valid_time": times,
            "temperature_c": np.full(len(times), 20.0),
            "relative_humidity_pct": np.full(len(times), 50.0),
            "precipitation_mm": np.full(len(times), 2.0 / 24.0),
            "wind_speed_kmh": np.full(len(times), 10.0),
            "downloaded_at": pd.Timestamp("2026-08-31 10:00", tz="UTC"),
            "forecast_run_at": pd.Timestamp("2026-08-31 00:00", tz="UTC"),
        }
    )


def test_evaluate_forecast_against_later_observations(tmp_path):
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    _hourly_forecast().to_parquet(forecast_dir / "forecast_20260831T000000Z_1km.parquet")

    dates = pd.date_range("2026-09-01", periods=3, freq="D")
    state = pd.DataFrame(
        {
            "cell_id": 1,
            "fecha": dates,
            "source": "meteogalicia_ema_idw",
            "tmax_vc": 19.0,
            "rhmin_vc": 55.0,
            "vmax_vc": 8.0,
            "prec_dia": 2.0,
        }
    )
    state_path = tmp_path / "state.parquet"
    state.to_parquet(state_path, index=False)

    report = evaluate_forecasts(forecast_dir, state_path)

    assert report["forecast_files_found"] == 1
    assert report["comparison_cases"] == 3
    assert report["forecast_inventory"][0]["closed_horizons"] == [1, 2, 3]
    assert report["forecast_inventory"][0]["closed_cells"] == {"1": 1, "2": 1, "3": 1}
    t1_temperature = next(
        row for row in report["metrics"]
        if row["horizon_days"] == 1 and row["variable"] == "tmax_vc"
    )
    assert t1_temperature["n_pairs"] == 1
    assert t1_temperature["mae"] == 1.0
    assert t1_temperature["bias"] == 1.0


def test_cli_writes_json_and_csv(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    _hourly_forecast().to_parquet(forecast_dir / "forecast_20260831T000000Z_1km.parquet")
    state = pd.DataFrame(
        {
            "cell_id": 1,
            "fecha": pd.date_range("2026-09-01", periods=3, freq="D"),
            "source": "meteogalicia_ema_idw",
            "tmax_vc": 19.0,
            "rhmin_vc": 55.0,
            "vmax_vc": 8.0,
            "prec_dia": 2.0,
        }
    )
    state_path = tmp_path / "state.parquet"
    state.to_parquet(state_path, index=False)
    output_dir = tmp_path / "evaluation"

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_meteogalicia_forecasts.py",
            "--forecast-dir",
            str(forecast_dir),
            "--state",
            str(state_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert main() == 0
    payload = json.loads(
        (output_dir / "meteogalicia_forecast_metrics.json").read_text()
    )
    assert payload["comparison_cases"] == 3
    assert (output_dir / "meteogalicia_forecast_metrics.csv").exists()
    pairs_path = output_dir / "meteogalicia_forecast_observation_pairs.parquet"
    assert payload["pair_rows"] == 3
    assert pairs_path.exists()
    pairs = pd.read_parquet(pairs_path)
    assert len(pairs) == 3
    assert pairs["tmax_vc_forecast"].tolist() == [20.0, 20.0, 20.0]
    assert pairs["tmax_vc_observed"].tolist() == [19.0, 19.0, 19.0]

    direct_path = output_dir / "direct.png"
    selected = plot_direct_case(pairs, direct_path)
    assert selected == ("2026-08-31", "2026-09-03", 3)
    assert direct_path.exists()
