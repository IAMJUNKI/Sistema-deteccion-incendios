"""Métricas de evaluación con desbalanceo extremo.

Varios de estos tests son **regresiones de fallos reales** que costaron resultados en este
proyecto. Están marcados como tal para que nadie los simplifique sin saber qué protegen.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.entrenamiento import metricas


class TestRecallAtFpr:
    @pytest.mark.parametrize("objetivo", [0.01, 0.05, 0.10])
    def test_el_presupuesto_de_falsos_positivos_nunca_se_supera(
        self, prediccion_desbalanceada, objetivo
    ):
        """Garantía dura: si se promete vigilar el 5 % del territorio, no se vigila el 5,003 %.

        La versión anterior fijaba el umbral con `np.quantile`, que interpola entre
        observaciones y podía devolver una tasa marginalmente por encima del objetivo.
        """
        y, p, _ = prediccion_desbalanceada
        _, _, fpr = metricas.recall_at_fpr(y, p, fpr_max=objetivo)
        assert fpr <= objetivo

    def test_el_umbral_es_el_mas_generoso_que_cabe(self, prediccion_desbalanceada):
        """No basta con respetar el presupuesto: hay que agotarlo, o se pierde recall gratis."""
        y, p, _ = prediccion_desbalanceada
        neg = p[y == 0]
        umbral = metricas.umbral_at_fpr(neg, 0.05)
        menores = neg[neg < umbral]
        if len(menores):
            siguiente = menores.max()
            assert float((neg >= siguiente).mean()) > 0.05, (
                "bajar al siguiente valor debería salirse del presupuesto"
            )

    def test_presupuesto_cero_no_alerta_nada(self):
        neg = np.linspace(0, 1, 100)
        assert (neg >= metricas.umbral_at_fpr(neg, 0.0)).sum() == 0

    def test_todo_empatado_en_el_maximo(self):
        """Caso degenerado: si todos los negativos valen lo mismo, no hay umbral que separe."""
        neg = np.full(1000, 0.7)
        assert (neg >= metricas.umbral_at_fpr(neg, 0.05)).sum() == 0

    def test_empates_masivos_no_inflan_el_recall(self):
        """REGRESIÓN. La calibración isotónica es escalonada y crea millones de empates.

        Con comparación `>=` el umbral se lleva la meseta entera, el FPR real se dispara muy
        por encima del objetivo y el recall sale espectacular y falso: así apareció un
        Recall@FPR5% del 72 % junto a un ROC-AUC de 0,72, que es imposible.
        """
        # Puntuación con solo tres valores: el 90 % de las filas empatadas en 0,10.
        y = np.array([0] * 900 + [0] * 50 + [1] * 50, dtype=np.int8)
        p = np.array([0.10] * 900 + [0.10] * 50 + [0.10] * 30 + [0.90] * 20)

        recall, _, fpr = metricas.recall_at_fpr(y, p, fpr_max=0.05)
        assert fpr <= 0.05 + 1e-9, "el FPR real se disparó por encima del objetivo"
        assert recall <= 0.5, f"recall inflado por empates: {recall}"

    def test_aciertos_promedian_al_recall(self, prediccion_desbalanceada):
        """El vector por incendio debe reproducir exactamente el recall reportado."""
        y, p, _ = prediccion_desbalanceada
        recall, _, _ = metricas.recall_at_fpr(y, p, 0.05)
        aciertos = metricas.aciertos_at_fpr(y, p, 0.05)
        assert aciertos.mean() == pytest.approx(recall)

    def test_sin_positivos_no_revienta(self):
        y = np.zeros(100, dtype=np.int8)
        recall, _, fpr = metricas.recall_at_fpr(y, np.random.random(100))
        assert recall == 0.0 and fpr == 0.0


class TestRecallDiario:
    def test_coincide_con_la_media_de_capturas(self, prediccion_desbalanceada):
        y, p, dia = prediccion_desbalanceada
        capturas = metricas.capturas_top_k_daily(y, p, dia, 0.01)
        assert metricas.recall_at_top_k_daily(y, p, dia, 0.01) == pytest.approx(capturas.mean())

    def test_una_captura_por_incendio(self, prediccion_desbalanceada):
        y, p, dia = prediccion_desbalanceada
        assert len(metricas.capturas_top_k_daily(y, p, dia, 0.01)) == int(y.sum())

    def test_prediccion_perfecta_captura_todo(self):
        dia = np.repeat(np.arange(10), 100)
        y = (np.arange(1000) % 100 == 0).astype(np.int8)   # 1 incendio por día
        assert metricas.recall_at_top_k_daily(y, y.astype(float), dia, 0.01) == 1.0

    def test_k_mayor_captura_al_menos_lo_mismo(self, prediccion_desbalanceada):
        """Alertar más celdas nunca puede detectar menos incendios."""
        y, p, dia = prediccion_desbalanceada
        assert (metricas.recall_at_top_k_daily(y, p, dia, 0.05)
                >= metricas.recall_at_top_k_daily(y, p, dia, 0.01))

    def test_intervalo_contiene_el_valor_puntual(self, prediccion_desbalanceada):
        y, p, dia = prediccion_desbalanceada
        punto = metricas.recall_at_top_k_daily(y, p, dia, 0.01)
        bajo, alto = metricas.bootstrap_top_k_daily_ci(y, p, dia, 0.01, n_boot=500)
        assert bajo <= punto <= alto


class TestEvaluate:
    def test_expone_intervalo_para_la_metrica_diaria(self, prediccion_desbalanceada):
        """REGRESIÓN. Sin intervalo propio, la curva de selección dibujaba el de otra métrica:
        el punto en 0,06 y las barras en 0,36, sin avisar de nada."""
        y, p, dia = prediccion_desbalanceada
        r = metricas.evaluate(y, p, p, dia, n_boot=200)
        assert "recall_at_top1%_daily_ci90_low" in r
        assert "recall_at_top1%_daily_ci90_high" in r
        assert r["recall_at_top1%_daily_ci90_low"] <= r["recall_at_top1%_daily"]

    def test_las_metricas_de_ordenacion_ignoran_una_transformacion_monotona(
        self, prediccion_desbalanceada
    ):
        """ROC-AUC y PR-AUC solo dependen del orden, así que calibrar no debe moverlas."""
        y, p, dia = prediccion_desbalanceada
        base = metricas.evaluate(y, p, None, dia, n_boot=100)
        reescalado = metricas.evaluate(y, np.sqrt(p) * 0.5, None, dia, n_boot=100)
        assert base["roc_auc"] == pytest.approx(reescalado["roc_auc"])
        assert base["pr_auc"] == pytest.approx(reescalado["pr_auc"])

    def test_lift_relativo_a_la_prevalencia(self, prediccion_desbalanceada):
        y, p, _ = prediccion_desbalanceada
        r = metricas.evaluate(y, p, n_boot=100)
        assert r["lift_vs_azar"] == pytest.approx(r["pr_auc"] / r["prevalence"])
        assert r["lift_vs_azar"] > 1, "el modelo de prueba debería batir al azar"


class TestCompararPareado:
    def test_modelos_identicos_nunca_tienen_ganador(self):
        a = np.random.default_rng(3).random(500) < 0.4
        r = metricas.comparar_pareado(a, a.copy(), n_boot=500)
        assert r["diferencia"] == 0.0
        assert not r["concluyente"]
        assert r["mcnemar_p"] == 1.0

    def test_dominancia_estricta_se_detecta(self):
        """B caza todo lo de A y 40 incendios más: no puede salir 'no concluyente'."""
        rng = np.random.default_rng(5)
        a = rng.random(1000) < 0.35
        b = a.copy()
        b[rng.choice(np.flatnonzero(~a), 40, replace=False)] = True
        r = metricas.comparar_pareado(a, b, "A", "B", n_boot=1000)
        assert r["concluyente"] and r["veredicto"] == "B mejor"
        assert r["solo_A"] == 0 and r["solo_B"] == 40

    def test_es_antisimetrico(self):
        rng = np.random.default_rng(9)
        a, b = rng.random(400) < 0.4, rng.random(400) < 0.3
        ab = metricas.comparar_pareado(a, b, "A", "B", n_boot=800, seed=1)
        ba = metricas.comparar_pareado(b, a, "B", "A", n_boot=800, seed=1)
        assert ab["diferencia"] == pytest.approx(-ba["diferencia"])
        assert ab["dif_ci90_low"] == pytest.approx(-ba["dif_ci90_high"])
        assert ab["mcnemar_p"] == pytest.approx(ba["mcnemar_p"])

    def test_es_mas_potente_que_comparar_intervalos_marginales(self):
        """El motivo de existir de esta función.

        Dos modelos muy correlacionados con una diferencia pequeña: los intervalos marginales
        se solapan y el test antiguo diría 'no concluyente', mientras que el pareado detecta
        la diferencia. Ocurrió con datos reales entre Random Forest y la regresión logística.
        """
        rng = np.random.default_rng(13)
        base = rng.random(1659)          # los incendios reales de 2022
        a, b = base < 0.390, base < 0.414

        def ic_marginal(v, seed=1):
            r = np.random.default_rng(seed)
            m = r.choice(v.astype(float), size=(2000, len(v)), replace=True).mean(axis=1)
            return np.percentile(m, 5), np.percentile(m, 95)

        (la, ha), (lb, hb) = ic_marginal(a), ic_marginal(b)
        assert not (ha < lb or hb < la), "el caso construido exige que se solapen"

        r = metricas.comparar_pareado(a, b, "A", "B", n_boot=2000)
        assert r["concluyente"], "el test pareado debería resolver lo que el marginal no"

    def test_longitudes_distintas_es_error(self):
        with pytest.raises(ValueError, match="mismos incendios"):
            metricas.comparar_pareado(np.array([True]), np.array([True, False]))

    def test_sin_incendios_es_error(self):
        with pytest.raises(ValueError, match="No hay incendios"):
            metricas.comparar_pareado(np.array([], bool), np.array([], bool))


class TestCompararModelos:
    def test_una_fila_por_par_y_metrica(self, prediccion_desbalanceada):
        y, p, dia = prediccion_desbalanceada
        rng = np.random.default_rng(2)
        puntuaciones = {n: np.clip(p + rng.normal(0, 0.02, len(p)), 0, 1)
                        for n in ("a", "b", "c")}
        tabla = metricas.comparar_modelos(y, puntuaciones, dia, n_boot=200)
        assert len(tabla) == 3 * 2      # 3 pares x 2 métricas
        assert set(tabla["metrica"]) == {"recall@fpr5", "recall@top1%/día"}
        assert tabla["veredicto"].notna().all()

    def test_sin_indice_de_dia_solo_una_metrica(self, prediccion_desbalanceada):
        y, p, _ = prediccion_desbalanceada
        tabla = metricas.comparar_modelos(y, {"a": p, "b": p * 0.9}, n_boot=200)
        assert set(tabla["metrica"]) == {"recall@fpr5"}
