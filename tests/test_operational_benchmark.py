from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features.canonical_contract import CANONICAL_FEATURES
from src.features.operational_benchmark import (
    MEMORY_COLUMNS,
    OPERATIONAL_ALIGNMENT_VERSION,
    OPERATIONAL_TEMPORAL_SEMANTICS,
    build_operational_dataset,
    validate_operational_dataset,
)


def _source_rows(start: str, days: int, year: int) -> pd.DataFrame:
    rows = []
    for cell_id in (1, 2):
        for offset, fecha in enumerate(pd.date_range(start, periods=days, freq="D")):
            row = {
                "cell_id": cell_id,
                "fecha": fecha,
                "target_ignicion": int(cell_id == 1 and offset == days - 1),
                "fire_weather_index": float(offset),
                "year": year,
            }
            row.update({column: 0.0 for column in CANONICAL_FEATURES})
            row["temperature_mean"] = 10.0 + offset
            row["relative_humidity_mean"] = 50.0
            row["wind_speed_max"] = 5.0
            row["precipitation_sum"] = float(offset + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_source(root, year: int, frame: pd.DataFrame) -> None:
    destination = root / f"year={year}"
    destination.mkdir(parents=True)
    frame.to_parquet(destination / f"dataset_{year}.parquet", index=False)


def test_operational_dataset_recomputes_memory_without_current_day(tmp_path) -> None:
    source = tmp_path / "canonical"
    source.mkdir()
    _write_source(source, 2016, _source_rows("2016-12-01", 31, 2016))
    _write_source(source, 2017, _source_rows("2017-01-01", 5, 2017))
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "predictor_columns": sorted(CANONICAL_FEATURES),
                "time_contract": "No temporal shift; meteorological accumulations include date T.",
            }
        ),
        encoding="utf-8",
    )

    destination = build_operational_dataset(
        source,
        tmp_path / "operational",
        years=(2016, 2017),
        batch_size=7,
        static_layer_quality="provisional_import",
    )
    result = pd.concat(
        [
            pd.read_parquet(destination / "year=2016/dataset_2016.parquet"),
            pd.read_parquet(destination / "year=2017/dataset_2017.parquet"),
        ],
        ignore_index=True,
    )
    row = result[(result.cell_id == 1) & (result.fecha == pd.Timestamp("2016-12-05"))].iloc[0]
    assert row["precipitation_sum"] == 5.0
    assert row["precipitation_sum_3d"] == 9.0
    assert row["temperature_mean_7d"] == pytest.approx((10 + 0 + 10 + 1 + 10 + 2 + 10 + 3) / 4)

    january = result[(result.cell_id == 1) & (result.fecha == pd.Timestamp("2017-01-01"))].iloc[0]
    assert january["precipitation_sum_3d"] == 90.0
    assert len(result) == 2 * 36

    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["alignment_version"] == OPERATIONAL_ALIGNMENT_VERSION
    assert metadata["training_temporal_semantics"] == OPERATIONAL_TEMPORAL_SEMANTICS
    assert set(metadata["memory_columns_recomputed"]) == set(MEMORY_COLUMNS)
    assert metadata["active_cells"] == 2
    assert metadata["row_counts"] == {"2016": 62, "2017": 10}
    assert validate_operational_dataset(destination)["static_layer_quality"] == "provisional_import"


def test_canonical_dataset_is_rejected_as_operational_dataset(tmp_path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "predictor_columns": sorted(CANONICAL_FEATURES),
                "time_contract": "No temporal shift; meteorological accumulations include date T.",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="alineación temporal"):
        validate_operational_dataset(tmp_path)


def test_operational_dataset_recalculates_dry_streak(tmp_path) -> None:
    source = tmp_path / "canonical"
    source.mkdir()
    frame = _source_rows("2016-01-01", 5, 2016)
    frame.loc[frame.cell_id.eq(1), "precipitation_sum"] = [0.0, 0.0, 2.0, 0.0, 0.0]
    frame.loc[frame.cell_id.eq(2), "precipitation_sum"] = 0.0
    _write_source(source, 2016, frame)
    (source / "metadata.json").write_text(
        json.dumps({"predictor_columns": sorted(CANONICAL_FEATURES)}), encoding="utf-8"
    )

    destination = build_operational_dataset(source, tmp_path / "operational", years=(2016,))
    result = pd.read_parquet(destination / "year=2016/dataset_2016.parquet")
    cell = result[result.cell_id == 1].sort_values("fecha")
    assert cell["consecutive_dry_days"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]
