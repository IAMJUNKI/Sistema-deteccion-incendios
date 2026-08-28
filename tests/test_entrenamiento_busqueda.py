"""Validación cruzada temporal y búsqueda de hiperparámetros."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entrenamiento import busqueda


@pytest.fixture
def anios():
    return np.repeat([2019, 2020, 2021, 2022], 250)


class TestVentanaExpansiva:
    def test_un_pliegue_menos_que_anios(self, anios):
        """El primer año no se valida nunca: no tiene pasado con el que entrenar."""
        v = busqueda.VentanaExpansiva(anios)
        assert v.get_n_splits() == 3

    def test_siempre_predice_hacia_adelante(self, anios):
        """La propiedad que hace válida esta partición: ningún año de entrenamiento puede ser
        posterior al de validación, o estaríamos usando el futuro para predecir el pasado."""
        v = busqueda.VentanaExpansiva(anios)
        for entrena, valida in v.split():
            anio_val = np.unique(anios[valida])
            assert len(anio_val) == 1, "cada pliegue valida un único año"
            assert anios[entrena].max() < anio_val[0]

    def test_la_ventana_se_expande(self, anios):
        v = busqueda.VentanaExpansiva(anios)
        tamanos = [len(entrena) for entrena, _ in v.split()]
        assert tamanos == sorted(tamanos) and len(set(tamanos)) == len(tamanos)

    def test_no_hay_solape_entre_entrenamiento_y_validacion(self, anios):
        v = busqueda.VentanaExpansiva(anios)
        for entrena, valida in v.split():
            assert not (set(entrena) & set(valida))

    def test_descarta_pliegues_sin_positivos_suficientes(self, anios):
        """Un pliegue con cuatro incendios da una métrica que es ruido puro y contaminaría la
        media entre pliegues."""
        y = np.zeros(len(anios), dtype=np.int8)
        y[anios == 2020] = (np.arange((anios == 2020).sum()) < 3)   # solo 3 positivos
        y[anios == 2021] = (np.arange((anios == 2021).sum()) < 50)
        y[anios == 2022] = (np.arange((anios == 2022).sum()) < 50)

        v = busqueda.VentanaExpansiva(anios, minimo_positivos=20, target=y)
        validados = [int(np.unique(anios[val])[0]) for _, val in v.split()]
        assert 2020 not in validados
        assert validados == [2021, 2022]

    def test_un_solo_anio_es_error(self):
        with pytest.raises(ValueError, match="ningún pliegue"):
            busqueda.VentanaExpansiva(np.repeat([2019], 100))

    def test_describir_menciona_los_anios(self, anios):
        texto = busqueda.VentanaExpansiva(anios).describir()
        assert "2020" in texto and "valida" in texto


class TestScorer:
    def test_puntua_con_el_recall_del_proyecto(self):
        """Debe coincidir con `recall_at_fpr`, no con la métrica por defecto de scikit-learn:
        prediciendo siempre 'no incendio' se acierta el 99,99 % de las veces."""
        from src.entrenamiento import metricas

        rng = np.random.default_rng(3)
        y = (rng.random(5_000) < 0.01).astype(np.int8)
        p = np.clip(rng.beta(2, 20, 5_000) + y * 0.3, 0, 1)

        class Falso:
            def predict_proba(self, X):
                return np.column_stack([1 - p, p])

        esperado, _, _ = metricas.recall_at_fpr(y, p, 0.05)
        assert busqueda.scorer_recall_at_fpr(0.05)(Falso(), None, y) == pytest.approx(esperado)

    def test_el_nombre_refleja_el_coste(self):
        assert busqueda.scorer_recall_at_fpr(0.10).__name__ == "recall_at_fpr10"


class TestEspacios:
    @pytest.mark.parametrize("modelo", ["lightgbm", "xgboost", "random_forest",
                                        "logistic_regression"])
    def test_hay_espacio_para_cada_modelo(self, modelo):
        from src.entrenamiento import modelos

        assert modelo in busqueda.ESPACIOS
        assert modelo in modelos.MODELOS_SOPORTADOS

    def test_las_distribuciones_muestrean_dentro_de_rango(self):
        for modelo, espacio in busqueda.ESPACIOS.items():
            for parametro, dist in espacio.items():
                muestras = dist.rvs(50, random_state=0)
                assert np.isfinite(muestras).all(), f"{modelo}.{parametro}"
                assert (muestras > 0).all(), f"{modelo}.{parametro} debe ser positivo"

    def test_el_desbalanceo_es_un_parametro_a_buscar(self):
        """No se «arregla» aparte: entra en el espacio junto a los demás."""
        for modelo in ("lightgbm", "xgboost"):
            assert "scale_pos_weight" in busqueda.ESPACIOS[modelo]

    def test_lightgbm_fija_subsample_freq(self):
        """Sin `subsample_freq`, LightGBM ignora `subsample` en silencio y la búsqueda
        exploraría un parámetro que no hace nada."""
        assert busqueda.FIJOS["lightgbm"]["subsample_freq"] == 1


class TestElegir:
    @staticmethod
    def _tabla():
        return pd.DataFrame({
            "configuracion": [0, 1, 2],
            "recall_medio": [0.40, 0.39, 0.30],
            "recall_desv": [0.06, 0.01, 0.01],   # la 0 gana en media pero es inestable
            "num_leaves": [100, 31, 15],
            "learning_rate": [0.1, 0.05, 0.02],
        })

    def test_por_defecto_penaliza_la_inestabilidad(self):
        """El sesgo del ganador: la de mayor media suele serlo por haber tenido suerte en un
        pliegue. Se prefiere la que funciona en los tres años."""
        elegida = busqueda.elegir(self._tabla(), ["num_leaves", "learning_rate"])
        assert elegida["num_leaves"] == 31

    def test_se_puede_pedir_la_de_mayor_media(self):
        elegida = busqueda.elegir(self._tabla(), ["num_leaves"],
                                  penalizar_inestabilidad=False)
        assert elegida["num_leaves"] == 100

    def test_devuelve_tipos_nativos(self):
        """Los numpy scalars rompen la serialización a YAML."""
        tabla = self._tabla()
        tabla["num_leaves"] = tabla["num_leaves"].astype(np.int64)
        elegida = busqueda.elegir(tabla, ["num_leaves"])
        assert type(elegida["num_leaves"]) is int

    def test_ignora_parametros_que_no_estan(self):
        elegida = busqueda.elegir(self._tabla(), ["num_leaves", "inventado"])
        assert set(elegida) == {"num_leaves"}

    def test_tabla_vacia_es_error(self):
        with pytest.raises(ValueError, match="vacía"):
            busqueda.elegir(pd.DataFrame(), ["x"])


class TestMismasConfiguracionesQueRandomizedSearchCV:
    """La búsqueda se ejecuta por bloques con `GridSearchCV` en vez de con
    `RandomizedSearchCV`, para poder guardar el avance. Estas pruebas verifican que eso no
    cambia *qué* se explora: si las configuraciones difirieran, el cambio dejaría de ser una
    mejora de robustez y pasaría a ser un experimento distinto."""

    def test_parametersampler_reproduce_el_muestreo(self):
        from sklearn.model_selection import ParameterSampler

        espacio = busqueda.ESPACIOS["lightgbm"]
        a = list(ParameterSampler(espacio, 10, random_state=42))
        b = list(ParameterSampler(espacio, 10, random_state=42))
        assert a == b, "con la misma semilla deben salir las mismas configuraciones"

    def test_semillas_distintas_dan_configuraciones_distintas(self):
        from sklearn.model_selection import ParameterSampler

        espacio = busqueda.ESPACIOS["lightgbm"]
        a = list(ParameterSampler(espacio, 10, random_state=42))
        b = list(ParameterSampler(espacio, 10, random_state=7))
        assert a != b

    def test_gridsearchcv_evalua_exactamente_la_lista_dada(self):
        """El mecanismo del que depende el troceado: una rejilla con un solo valor por
        parámetro equivale a una configuración concreta."""
        from sklearn.model_selection import ParameterGrid

        candidatas = [{"num_leaves": 7, "n_estimators": 5},
                      {"num_leaves": 3, "n_estimators": 9}]
        rejilla = [{k: [v] for k, v in p.items()} for p in candidatas]
        assert list(ParameterGrid(rejilla)) == candidatas


class TestBusquedaCompleta:
    @staticmethod
    def _datos(dataset_falso):
        from src.entrenamiento import contrato as mod_contrato, datos

        c = mod_contrato.cargar(dataset_falso)
        marco, _ = datos.muestrear_entrenamiento(c, c.anios, modulo=2)
        marco["__anio"] = pd.to_datetime(marco["fecha"]).dt.year.to_numpy()
        y = marco["target_ignicion"]
        particionador = busqueda.VentanaExpansiva(
            marco["__anio"].to_numpy(), minimo_positivos=1, target=y.to_numpy()
        )
        return marco[c.predictores], y, particionador

    ESPACIO = {"n_estimators": busqueda.randint(5, 12), "num_leaves": busqueda.randint(3, 8)}

    def test_recorre_el_ciclo_entero(self, dataset_falso):
        X, y, particionador = self._datos(dataset_falso)
        tabla = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=3,
                                espacio=self.ESPACIO, tam_bloque=2)

        assert len(tabla) == 3
        assert {"recall_medio", "recall_desv", "recall_minimo", "pliegue_1"} <= set(tabla.columns)
        assert tabla["recall_medio"].between(0, 1).all()
        assert (tabla["recall_medio"].diff().dropna() <= 1e-12).all(), "debe venir ordenada"
        assert sorted(tabla["configuracion"]) == [0, 1, 2]

    def test_el_troceado_no_cambia_el_resultado(self, dataset_falso):
        """Lo esencial: partir la búsqueda en bloques es una decisión de robustez y no puede
        alterar los números. Un bloque de 4 evalúa lo mismo que dos de 2."""
        X, y, particionador = self._datos(dataset_falso)
        entero = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=4,
                                 espacio=self.ESPACIO, tam_bloque=4)
        troceado = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=4,
                                   espacio=self.ESPACIO, tam_bloque=2)
        columnas = ["configuracion", "recall_medio", "recall_desv", "n_estimators",
                    "num_leaves"]
        pd.testing.assert_frame_equal(
            entero[columnas].sort_values("configuracion").reset_index(drop=True),
            troceado[columnas].sort_values("configuracion").reset_index(drop=True),
        )

    def test_guarda_el_avance_en_disco(self, dataset_falso, tmp_path):
        X, y, particionador = self._datos(dataset_falso)
        destino = tmp_path / "avance.csv"
        busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=4,
                        espacio=self.ESPACIO, tam_bloque=2, ruta_checkpoint=destino)
        assert destino.exists()
        assert len(pd.read_csv(destino)) == 4
        assert not destino.with_suffix(".parcial").exists(), "no debe quedar el temporal"

    def test_reanuda_desde_un_avance_previo(self, dataset_falso, tmp_path):
        """El motivo de todo el cambio: una interrupción no debe costar la búsqueda entera."""
        X, y, particionador = self._datos(dataset_falso)
        destino = tmp_path / "avance.csv"

        # Primera pasada: solo dos configuraciones, como si se hubiera cortado ahí.
        parcial = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=2,
                                  espacio=self.ESPACIO, tam_bloque=2, ruta_checkpoint=destino)
        assert len(parcial) == 2

        completa = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=4,
                                   espacio=self.ESPACIO, tam_bloque=2, ruta_checkpoint=destino)
        assert len(completa) == 4
        # Las dos primeras no se recalculan: conservan su puntuación original.
        for _, fila in parcial.iterrows():
            previa = completa[completa["configuracion"] == fila["configuracion"]].iloc[0]
            assert previa["recall_medio"] == pytest.approx(fila["recall_medio"])

    def test_desde_cero_ignora_el_avance(self, dataset_falso, tmp_path):
        X, y, particionador = self._datos(dataset_falso)
        destino = tmp_path / "avance.csv"
        busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=4,
                        espacio=self.ESPACIO, tam_bloque=2, ruta_checkpoint=destino)
        tabla = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=4,
                                espacio=self.ESPACIO, tam_bloque=2,
                                ruta_checkpoint=destino, reanudar=False)
        assert len(tabla) == 4, "no debe duplicar filas al repetir la búsqueda"

    def test_descarta_un_checkpoint_de_otra_busqueda(self, dataset_falso, tmp_path):
        """Un avance con más configuraciones que las pedidas viene de otro experimento;
        reanudarlo mezclaría dos búsquedas distintas en la misma tabla."""
        X, y, particionador = self._datos(dataset_falso)
        destino = tmp_path / "avance.csv"
        pd.DataFrame({"configuracion": [0, 99], "recall_medio": [0.5, 0.6]}).to_csv(
            destino, index=False
        )
        tabla = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=2,
                                espacio=self.ESPACIO, tam_bloque=2, ruta_checkpoint=destino)
        assert len(tabla) == 2
        assert 99 not in set(tabla["configuracion"])

    def test_tolera_un_checkpoint_corrupto(self, dataset_falso, tmp_path):
        X, y, particionador = self._datos(dataset_falso)
        destino = tmp_path / "avance.csv"
        destino.write_text("esto no es un csv valido\x00\x00", encoding="utf-8")
        tabla = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=2,
                                espacio=self.ESPACIO, tam_bloque=2, ruta_checkpoint=destino)
        assert len(tabla) == 2

    def test_una_configuracion_invalida_no_tira_la_busqueda(self, dataset_falso):
        """Convive con las buenas en el mismo bloque: `error_score=nan` la absorbe."""
        X, y, particionador = self._datos(dataset_falso)
        espacio = {"num_leaves": [3, -5]}   # -5 es invalido para LightGBM
        tabla = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=2,
                                espacio=espacio, tam_bloque=2)
        assert len(tabla) == 2
        assert tabla["recall_medio"].isna().sum() == 1
        assert np.isnan(tabla["recall_medio"].iloc[-1]), "la fallida va al final"
        assert busqueda.elegir(tabla.dropna(subset=["recall_medio"]),
                               ["num_leaves"])["num_leaves"] == 3

    def test_un_bloque_entero_invalido_tampoco_la_tira(self, dataset_falso):
        """El caso límite del troceado.

        `error_score=nan` absorbe los fallos sueltos, pero scikit-learn vuelve a lanzar si
        TODAS las configuraciones de una llamada fallaron. Con bloques pequeños eso es fácil
        que ocurra, y sin red mataría la búsqueda entera: justo lo que el troceado venía a
        evitar.
        """
        X, y, particionador = self._datos(dataset_falso)
        tabla = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=2,
                                espacio={"num_leaves": [3, -5]}, tam_bloque=1)
        assert len(tabla) == 2, "la busqueda debe completarse pese al bloque fallido"
        assert tabla["recall_medio"].isna().sum() == 1
        assert not np.isnan(tabla["recall_medio"].iloc[0]), "la buena debe seguir ahi"

    def test_todas_invalidas_devuelve_tabla_completa_de_fallos(self, dataset_falso):
        X, y, particionador = self._datos(dataset_falso)
        tabla = busqueda.buscar("lightgbm", X, y, particionador, n_configuraciones=2,
                                espacio={"num_leaves": [-5, -3]}, tam_bloque=2)
        assert len(tabla) == 2
        assert tabla["recall_medio"].isna().all()
        assert sorted(tabla["configuracion"]) == [0, 1]
