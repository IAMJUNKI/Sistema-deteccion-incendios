"""Corrección de prior y calibración isotónica."""

from __future__ import annotations

import numpy as np
import pytest

from src.entrenamiento import calibracion


class TestPriorCorrection:
    def test_tasa_uno_no_cambia_nada(self):
        p = np.array([0.01, 0.3, 0.9])
        assert calibracion.prior_correction(p, 1.0) == pytest.approx(p)

    def test_reduce_las_probabilidades(self):
        """Submuestrear negativos infla la probabilidad; corregirlo tiene que bajarla."""
        p = np.array([0.05, 0.2, 0.5, 0.9])
        assert (calibracion.prior_correction(p, 0.04) < p).all()

    def test_conserva_el_orden(self):
        """Es una transformación monótona: no puede alterar el ranking de riesgo."""
        p = np.sort(np.random.default_rng(4).random(500))
        corregida = calibracion.prior_correction(p, 0.04)
        assert (np.diff(corregida) >= -1e-12).all()

    def test_recupera_la_prevalencia_real(self):
        """Con una tasa de 1/25, una probabilidad del 50 % en la muestra son unas odds de 1,
        que corregidas quedan en 1/25 y por tanto en p = 1/26."""
        assert calibracion.prior_correction(np.array([0.5]), 1 / 25)[0] == pytest.approx(1 / 26)

    @pytest.mark.parametrize("tasa", [0.0, -0.1, 1.5])
    def test_tasa_invalida_es_error(self, tasa):
        with pytest.raises(ValueError, match="neg_sampling_rate"):
            calibracion.prior_correction(np.array([0.5]), tasa)


class TestProbabilityCalibrator:
    def test_la_media_calibrada_se_acerca_a_la_prevalencia(self):
        """El objetivo de calibrar: que la probabilidad media prediga la frecuencia real."""
        rng = np.random.default_rng(6)
        n, prevalencia = 200_000, 0.0015
        y = (rng.random(n) < prevalencia).astype(np.int8)
        # Puntuación inflada, como la de un modelo entrenado con negativos submuestreados.
        p = np.clip(rng.beta(2, 8, n) + y * 0.3, 1e-6, 1 - 1e-6)

        calibrador = calibracion.ProbabilityCalibrator(0.04).fit(y, p)
        calibrada = calibrador.transform(p)

        assert calibrada.mean() == pytest.approx(y.mean(), abs=5e-4)
        assert calibrada.mean() < p.mean(), "calibrar tiene que corregir hacia abajo"

    def test_conserva_el_orden_del_modelo(self):
        rng = np.random.default_rng(8)
        y = (rng.random(5_000) < 0.02).astype(np.int8)
        p = np.clip(rng.beta(2, 8, 5_000) + y * 0.3, 0, 1)
        calibrador = calibracion.ProbabilityCalibrator(0.04).fit(y, p)
        c = calibrador.transform(p)
        orden = np.argsort(p)
        assert (np.diff(c[orden]) >= -1e-9).all()

    def test_sin_ajustar_solo_corrige_el_prior(self):
        calibrador = calibracion.ProbabilityCalibrator(0.04)
        p = np.array([0.5])
        assert calibrador.transform(p) == pytest.approx(
            calibracion.prior_correction(p, 0.04)
        )

    def test_submuestreo_interno_respeta_el_tope(self):
        """Con más filas que `max_fit_points` se submuestrean negativos con pesos; el ajuste
        debe seguir siendo válido y no romperse."""
        rng = np.random.default_rng(10)
        n = 50_000
        y = (rng.random(n) < 0.01).astype(np.int8)
        p = np.clip(rng.beta(2, 8, n) + y * 0.3, 0, 1)
        calibrador = calibracion.ProbabilityCalibrator(0.04, max_fit_points=5_000).fit(y, p)
        assert calibrador.isotonic is not None
        assert np.isfinite(calibrador.transform(p)).all()


class TestNivelesDeRiesgo:
    def test_los_cortes_son_crecientes_y_reparten_como_se_pide(self):
        p = np.random.default_rng(12).random(10_000)
        niveles, umbrales = calibracion.risk_levels(p, (0.90, 0.98, 0.995))
        assert umbrales["moderado"] < umbrales["alto"] < umbrales["extremo"]
        assert set(np.unique(niveles)) <= {0, 1, 2, 3}
        assert (niveles == 0).mean() == pytest.approx(0.90, abs=0.01)
        assert (niveles == 3).mean() == pytest.approx(0.005, abs=0.005)
