"""Pruebas del protocolo temporal del pipeline definitivo de modelado."""

import argparse

import numpy as np
import pytest

from scripts.pipeline_definitivo import CalibradorPlatt, configurar_desde_argumentos


def test_protocolo_por_defecto_aprovecha_historico_y_reserva_2023() -> None:
    config = configurar_desde_argumentos(argparse.Namespace(train_years=None))

    assert config.años_entrenamiento == (2016, 2017, 2018, 2019, 2020)
    assert config.año_calibracion == 2021
    assert config.año_validacion == 2022
    assert config.año_reservado == 2023
    assert config.filas_puntuacion == 100_000


def test_protocolo_rechaza_ano_de_calibracion_en_entrenamiento() -> None:
    with pytest.raises(SystemExit, match="distintos"):
        configurar_desde_argumentos(argparse.Namespace(train_years=[2019, 2020, 2021]))


def test_calibrador_platt_muestrea_negativos_sin_perder_el_orden() -> None:
    probabilidades = np.linspace(0.001, 0.9, 20)
    objetivo = np.array([0] * 17 + [1] * 3)
    calibrador = CalibradorPlatt(0.01, max_puntos_ajuste=8).ajustar(
        probabilidades, objetivo, semilla=42
    )

    resultado = calibrador.aplicar(probabilidades)
    assert np.all(np.isfinite(resultado))
    assert np.all(np.diff(resultado) >= 0)
    assert np.any(np.diff(resultado) > 0)
