"""Poda automática, diagnóstico y curva de compromiso."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entrenamiento import seleccion


@pytest.fixture
def marco():
    rng = np.random.default_rng(21)
    n = 4_000
    base = rng.normal(0, 1, n)
    return pd.DataFrame({
        "util": base,
        "gemela": base * 2 + rng.normal(0, 0.005, n),   # duplicada de `util`
        "ruido": rng.normal(0, 1, n),
        "constante": np.ones(n),
        "casi_nula": np.where(rng.random(n) < 0.8, np.nan, rng.normal(0, 1, n)),
        "target": (rng.random(n) < 1 / (1 + np.exp(-base))).astype(np.int8),
    })


class TestPodaAutomatica:
    def test_descarta_constantes_y_exceso_de_nulos(self, marco):
        vivas, descartes = seleccion.poda_automatica(
            marco, ["util", "ruido", "constante", "casi_nula"], "target"
        )
        motivos = dict(zip(descartes["variable"], descartes["motivo"]))
        assert motivos["constante"] == "constante"
        assert motivos["casi_nula"] == "exceso de nulos"
        assert "util" in vivas and "ruido" in vivas

    def test_de_un_par_duplicado_conserva_el_de_mas_senal(self, marco):
        vivas, descartes = seleccion.poda_automatica(marco, ["util", "gemela"], "target")
        assert len(vivas) == 1
        assert descartes.iloc[0]["motivo"] == "duplicada"
        assert "Spearman" in descartes.iloc[0]["evidencia"]

    def test_cada_descarte_lleva_evidencia_numerica(self, marco):
        """Auditable: en la defensa hay que responder con un dato, no con una intuición."""
        _, descartes = seleccion.poda_automatica(
            marco, ["util", "gemela", "constante", "casi_nula"], "target"
        )
        assert not descartes.empty
        assert set(descartes.columns) >= {"variable", "motivo", "evidencia", "valor"}
        assert descartes["evidencia"].str.len().gt(0).all()
        assert descartes["valor"].notna().all()

    def test_no_descarta_si_el_control_la_desmiente(self, marco):
        """REGRESIÓN. Una variable salió marcada como constante en una ejecución y con más de
        un millón de valores distintos en otra, sin poder reproducirlo. Un descarte silencioso
        e irreproducible no puede sostener un capítulo de resultados: si los dos conjuntos no
        coinciden, se conserva la variable y se deja constancia."""
        control = marco.copy()
        control["constante"] = np.random.default_rng(1).normal(0, 1, len(control))

        vivas, descartes = seleccion.poda_automatica(
            marco, ["util", "constante"], "target", marco_control=control
        )
        assert "constante" in vivas, "no debe descartarse si el control la desmiente"
        assert "constante" not in set(descartes.get("variable", []))

    def test_con_control_coherente_si_descarta(self, marco):
        vivas, _ = seleccion.poda_automatica(
            marco, ["util", "constante"], "target", marco_control=marco
        )
        assert "constante" not in vivas


class TestDiagnostico:
    def test_la_senal_util_supera_a_la_del_ruido(self, marco):
        senal = seleccion.senal_univariante(marco, ["util", "ruido"], "target")
        fuerza = senal.set_index("variable")["fuerza"]
        assert fuerza["util"] > fuerza["ruido"]

    def test_una_variable_invertida_tiene_la_misma_fuerza(self, marco):
        """Un AUC de 0,2 discrimina tanto como uno de 0,8: el modelo invierte el signo."""
        marco = marco.assign(util_invertida=-marco["util"])
        senal = seleccion.senal_univariante(
            marco, ["util", "util_invertida"], "target"
        ).set_index("variable")
        assert senal.loc["util", "fuerza"] == pytest.approx(
            senal.loc["util_invertida", "fuerza"], abs=1e-9
        )
        assert senal.loc["util", "auc"] == pytest.approx(1 - senal.loc["util_invertida", "auc"])

    def test_las_familias_son_transitivas(self):
        """Si A se parece a B y B a C, las tres van al mismo grupo aunque A y C no."""
        rng = np.random.default_rng(31)
        a = np.sort(rng.normal(0, 1, 2_000))
        df = pd.DataFrame({
            "a": a,
            "b": a + rng.normal(0, 0.05, 2_000),
            "c": a + rng.normal(0, 0.10, 2_000),
            "suelta": rng.normal(0, 1, 2_000),
        })
        familias = seleccion.clusters_correlacion(df, ["a", "b", "c", "suelta"], umbral=0.9)
        assert len(familias) == 1
        assert set(familias.iloc[0]["variables"].split(", ")) == {"a", "b", "c"}


class TestResolverIC:
    def test_prefiere_el_intervalo_propio_de_la_metrica(self):
        resultado = {"m": 0.5, "m_ci90_low": 0.4, "m_ci90_high": 0.6,
                     "recall_ci90_low": 0.1, "recall_ci90_high": 0.2}
        assert seleccion._resolver_ic(resultado, "m") == ("m_ci90_low", "m_ci90_high")

    def test_admite_el_alias_historico_del_recall(self):
        resultado = {"recall_at_fpr5": 0.4, "recall_ci90_low": 0.3, "recall_ci90_high": 0.5}
        assert seleccion._resolver_ic(resultado, "recall_at_fpr5") == (
            "recall_ci90_low", "recall_ci90_high"
        )

    def test_falla_si_la_metrica_no_tiene_intervalo(self):
        """REGRESIÓN. Antes se cogía el intervalo de otra métrica y se dibujaba el punto en
        0,06 con las barras en 0,36. Es preferible fallar a mentir en silencio."""
        with pytest.raises(KeyError, match="no tiene intervalo"):
            seleccion._resolver_ic({"pr_auc": 0.1, "recall_ci90_low": 0.3}, "pr_auc")


class TestCurvaYRecomendacion:
    @staticmethod
    def _evaluador(mapa):
        def f(variables):
            k = len(variables)
            return {"recall_at_fpr5": mapa[k], "recall_ci90_low": mapa[k] - 0.02,
                    "recall_ci90_high": mapa[k] + 0.02, "roc_auc": 0.8, "pr_auc": 0.01}
        return f

    def test_guarda_todas_las_metricas_de_cada_escalon(self):
        """REGRESIÓN. Guardando solo la métrica que gobierna hubo que repetir la escalera
        entera para descubrir que el recall diario ordenaba los conjuntos al revés."""
        curva = seleccion.curva_compromiso(
            [f"v{i}" for i in range(10)],
            self._evaluador({3: 0.30, 6: 0.35}), [3, 6],
        )
        assert "metrica_roc_auc" in curva.columns
        assert "metrica_pr_auc" in curva.columns
        assert list(curva["k"]) == [3, 6]

    def test_los_escalones_se_recortan_al_tamano_disponible(self):
        curva = seleccion.curva_compromiso(
            ["a", "b", "c"], self._evaluador({1: 0.1, 3: 0.3}), [1, 3, 50]
        )
        assert list(curva["k"]) == [1, 3]

    def test_recomienda_el_mas_pequeno_que_no_es_peor(self):
        curva = seleccion.curva_compromiso(
            [f"v{i}" for i in range(30)],
            self._evaluador({5: 0.20, 10: 0.36, 20: 0.37, 30: 0.38}), [5, 10, 20, 30],
        )
        r = seleccion.recomendar(curva)
        assert r["k_mejor"] == 30
        assert r["k_recomendado"] == 10, "10 se solapa con 30, así que gana el simple"
        assert r["variables_ahorradas"] == 20
        assert not r["concluyente"]

    def test_si_el_grande_gana_de_verdad_lo_dice(self):
        curva = seleccion.curva_compromiso(
            [f"v{i}" for i in range(30)],
            self._evaluador({5: 0.10, 30: 0.50}), [5, 30],
        )
        r = seleccion.recomendar(curva)
        assert r["k_recomendado"] == 30 and r["concluyente"]

    def test_curva_vacia_es_error(self):
        with pytest.raises(ValueError, match="vacía"):
            seleccion.recomendar(pd.DataFrame())


class TestSubmuestraEstratificada:
    def test_conserva_todos_los_positivos(self):
        rng = np.random.default_rng(41)
        df = pd.DataFrame({"x": rng.random(10_000),
                           "y": (rng.random(10_000) < 0.01).astype(np.int8)})
        reducida = seleccion.submuestra_estratificada(df, "y", n_maximo=1_000)
        assert reducida["y"].sum() == df["y"].sum()
        assert len(reducida) <= 1_000

    def test_si_ya_cabe_no_toca_nada(self):
        df = pd.DataFrame({"y": np.zeros(10, dtype=np.int8)})
        assert seleccion.submuestra_estratificada(df, "y", 100) is df
