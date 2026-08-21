"""Tests del cubo temporal diario."""

import numpy as np

from src.features.time import TIME_VARIABLES, crear_datacubo_temporal


def test_cubo_temporal_por_defecto_cubre_contexto_y_periodo_historico() -> None:
    """El rango incluye diciembre de contexto y todos los días de 2019-2023."""
    dataset = crear_datacubo_temporal()

    assert len(dataset.time) == 1857
    assert str(dataset.time.min().values)[:10] == "2018-12-01"
    assert str(dataset.time.max().values)[:10] == "2023-12-31"
    assert set(TIME_VARIABLES).issubset(dataset.data_vars)
    assert int(dataset.sel(time="2020-02-29")["day_of_year"]) == 60
    assert int(dataset.sel(time="2023-12-31")["is_weekend"]) == 1


def test_variables_ciclicas_estan_acotadas() -> None:
    """Las codificaciones seno/coseno permanecen entre -1 y 1."""
    dataset = crear_datacubo_temporal("2020-01-01", "2020-12-31")

    for variable in ("day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos"):
        assert np.abs(dataset[variable]).max() <= 1.0
