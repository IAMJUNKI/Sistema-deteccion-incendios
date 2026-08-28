"""Variables derivadas y su adaptación automática al esquema del dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entrenamiento import contrato as mod_contrato, datos, derivadas


@pytest.fixture(scope="module")
def contrato(dataset_falso):
    return mod_contrato.cargar(dataset_falso)


@pytest.fixture(scope="module")
def preparado(contrato, tmp_path_factory, monkeypatch_session=None):
    """Muestra con contexto ajustado y derivadas añadidas."""
    muestra, _ = datos.muestrear_entrenamiento(contrato, [2019], modulo=2)
    contexto = derivadas.ajustar_contexto(contrato, [2019], [2019], usar_cache=False)
    return muestra, contexto


class TestDeficitPresionVapor:
    def test_es_cero_con_aire_saturado(self):
        vpd = derivadas.deficit_presion_vapor(pd.Series([25.0]), pd.Series([100.0]))
        assert vpd.iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_crece_con_la_temperatura(self):
        vpd = derivadas.deficit_presion_vapor(pd.Series([10.0, 25.0, 40.0]),
                                              pd.Series([50.0, 50.0, 50.0]))
        assert (np.diff(vpd) > 0).all()

    def test_decrece_con_la_humedad(self):
        vpd = derivadas.deficit_presion_vapor(pd.Series([30.0] * 3),
                                              pd.Series([20.0, 50.0, 80.0]))
        assert (np.diff(vpd) < 0).all()

    def test_valor_conocido(self):
        """A 20 °C la presión de saturación son ~2,339 kPa; con 50 % de humedad, ~1,17."""
        vpd = derivadas.deficit_presion_vapor(pd.Series([20.0]), pd.Series([50.0]))
        assert vpd.iloc[0] == pytest.approx(1.169, abs=0.01)

    def test_humedad_fuera_de_rango_se_recorta(self):
        assert derivadas.deficit_presion_vapor(pd.Series([25.0]), pd.Series([150.0])).iloc[0] >= 0


class TestAdaptacionAlEsquema:
    """El pipeline no puede duplicar lo que el cubo ya trae ni dejar huecos si lo retira."""

    def test_usa_el_vpd_del_cubo_si_existe(self, contrato):
        assert contrato.tiene("vpd_mean")
        assert derivadas.columna_vpd(contrato) == "vpd_mean"
        assert "vpd" not in derivadas.columnas_derivadas(contrato)

    def test_construye_forestal_si_falta_la_suma(self, contrato):
        """REGRESIÓN. Con `forest_cover_fraction` presente se creaba además una `forestal`
        idéntica: el árbol repartía los cortes entre las dos y la importancia de la cobertura
        forestal aparecía dividida por la mitad."""
        assert not contrato.tiene("forest_cover_fraction")
        assert derivadas.columna_forestal(contrato) == "forestal"
        assert "forestal" in derivadas.columnas_derivadas(contrato)

    def test_no_duplica_el_calendario_que_ya_esta(self, contrato):
        assert set(derivadas.calendario_ausente(contrato)) == set(derivadas.CALENDARIO)

    def test_no_genera_lo_que_depende_de_columnas_ausentes(self, contrato):
        """Sin `consecutive_dry_days` no puede haber interacción con la sequía."""
        assert not contrato.tiene("consecutive_dry_days")
        assert "sequia_x_forestal" not in derivadas.columnas_derivadas(contrato)


class TestCalendario:
    def test_es_finde_en_fechas_conocidas(self):
        # 2022-08-06 es sábado, 07 domingo, 08 lunes.
        marco = pd.DataFrame({"fecha": pd.to_datetime(
            ["2022-08-05", "2022-08-06", "2022-08-07", "2022-08-08"])})
        salida = derivadas.anadir_calendario(marco, ["es_finde"])
        assert list(salida["es_finde"]) == [0, 1, 1, 0]

    def test_el_ciclo_anual_cierra(self):
        """REGRESIÓN de diseño: con 365 en vez de 365,25, en un bisiesto el 31 de diciembre
        se desfasa del 1 de enero y la estacionalidad tiene un salto artificial."""
        marco = pd.DataFrame({"fecha": pd.to_datetime(["2020-12-31", "2021-01-01"])})
        salida = derivadas.anadir_calendario(marco, ["dia_anio_sin", "dia_anio_cos"])
        distancia = np.hypot(
            salida["dia_anio_sin"].iloc[0] - salida["dia_anio_sin"].iloc[1],
            salida["dia_anio_cos"].iloc[0] - salida["dia_anio_cos"].iloc[1],
        )
        assert distancia < 0.05, "fin y principio de año deben quedar juntos en el círculo"

    def test_esta_acotado(self):
        marco = pd.DataFrame({"fecha": pd.date_range("2021-01-01", periods=400)})
        salida = derivadas.anadir_calendario(marco, ["dia_anio_sin", "dia_anio_cos"])
        for col in ("dia_anio_sin", "dia_anio_cos"):
            assert salida[col].between(-1, 1).all()

    def test_lista_vacia_no_toca_nada(self):
        marco = pd.DataFrame({"fecha": pd.to_datetime(["2022-01-01"])})
        assert derivadas.anadir_calendario(marco, []) is marco


class TestAnadirDerivadas:
    def test_genera_exactamente_lo_declarado(self, contrato, preparado):
        """`columnas_derivadas` es un contrato: si promete algo, tiene que aparecer."""
        muestra, contexto = preparado
        salida = derivadas.anadir_derivadas(muestra, contexto, contrato)
        for nombre in derivadas.columnas_derivadas(contrato):
            assert nombre in salida.columns, f"declarada pero no generada: {nombre}"

    def test_no_deja_columnas_auxiliares(self, contrato, preparado):
        """Los `merge` del contexto traen medias y desviaciones que no son predictoras."""
        muestra, contexto = preparado
        salida = derivadas.anadir_derivadas(muestra, contexto, contrato)
        residuo = [c for c in salida.columns
                   if c.endswith(("_media", "_desv", "_media_dia", "_desv_dia")) or c == "__mes"]
        assert not residuo, f"columnas auxiliares filtradas: {residuo}"

    def test_no_introduce_nulos(self, contrato, preparado):
        muestra, contexto = preparado
        salida = derivadas.anadir_derivadas(muestra, contexto, contrato)
        generadas = derivadas.columnas_derivadas(contrato)
        assert salida[generadas].isna().sum().sum() == 0

    def test_no_pierde_ni_duplica_filas(self, contrato, preparado):
        """Un `merge` mal planteado multiplica filas en silencio."""
        muestra, contexto = preparado
        assert len(derivadas.anadir_derivadas(muestra, contexto, contrato)) == len(muestra)

    def test_no_modifica_el_marco_original(self, contrato, preparado):
        muestra, contexto = preparado
        columnas = list(muestra.columns)
        derivadas.anadir_derivadas(muestra, contexto, contrato)
        assert list(muestra.columns) == columnas

    def test_los_z_scores_estan_estandarizados(self, contrato, preparado):
        """Media cero y desviación uno por construcción: si no, el contexto está mal ajustado."""
        muestra, contexto = preparado
        salida = derivadas.anadir_derivadas(muestra, contexto, contrato)
        for variable in contexto.variables:
            z = salida[f"{variable}_z_dia"]
            assert abs(z.mean()) < 0.15
            assert z.std() == pytest.approx(1.0, abs=0.15)

    def test_es_sin_estado(self, contrato, preparado):
        """Procesar por lotes exige que el resultado no dependa de qué más se procesó antes."""
        muestra, contexto = preparado
        entero = derivadas.anadir_derivadas(muestra, contexto, contrato)
        mitad = derivadas.anadir_derivadas(muestra.iloc[:200].copy(), contexto, contrato)
        generadas = derivadas.columnas_derivadas(contrato)
        np.testing.assert_allclose(
            mitad[generadas].to_numpy(), entero[generadas].iloc[:200].to_numpy(),
            rtol=1e-5, atol=1e-6,
        )

    def test_dia_uniforme_da_z_score_cero(self, contrato, preparado):
        """Si toda Galicia tiene el mismo valor un día, ninguna celda destaca: el z-score no
        está definido y debe valer cero, no infinito ni NaN."""
        muestra, contexto = preparado
        plano = contexto.diarios.copy()
        for variable in contexto.variables:
            plano[f"{variable}_desv_dia"] = 0.0
        sin_dispersion = derivadas.Contexto(
            contexto.climatologia, plano, contexto.variables, contexto.anios_climatologia
        )
        salida = derivadas.anadir_derivadas(muestra, sin_dispersion, contrato)
        for variable in contexto.variables:
            z = salida[f"{variable}_z_dia"]
            assert (z == 0).all() and np.isfinite(z).all()


class TestContexto:
    def test_la_climatologia_cubre_las_celdas_y_meses_vistos(self, contrato, preparado):
        _, contexto = preparado
        assert {"cell_id", "mes", "n"} <= set(contexto.climatologia.columns)
        assert (contexto.climatologia["n"] > 0).all()

    def test_hay_un_estadistico_por_dia(self, contrato, preparado):
        _, contexto = preparado
        assert contexto.diarios["fecha"].is_unique

    def test_las_desviaciones_no_son_negativas(self, contrato, preparado):
        _, contexto = preparado
        for variable in contexto.variables:
            assert (contexto.climatologia[f"{variable}_desv"] >= 0).all()
            assert (contexto.diarios[f"{variable}_desv_dia"] >= 0).all()
