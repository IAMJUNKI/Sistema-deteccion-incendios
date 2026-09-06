"""Variables de contexto espacial.

La prueba que manda es la de fuga: estas variables se construyen a partir del propio objetivo,
así que si la ventana temporal incluyera el día en curso el modelo estaría leyendo la respuesta.
El datacubo ya trae una variable con ese defecto —`is_near_ignition_25x25_10d`, que abarca el
día del evento— y por eso el contrato la prohíbe como predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entrenamiento import contrato as mod_contrato, experimento, vecindad


def malla(n_filas=12, n_columnas=12, n_dias=40, igniciones=(), semilla=0) -> pd.DataFrame:
    """Población sintética: rejilla regular completa con las igniciones que se indiquen.

    `igniciones` son tripletas (dia, fila, columna).
    """
    filas = []
    fechas = pd.date_range("2022-06-01", periods=n_dias, freq="D")
    marcadas = set(igniciones)
    for d, fecha in enumerate(fechas):
        for f in range(n_filas):
            for c in range(n_columnas):
                filas.append({
                    "cell_id": f * n_columnas + c,
                    "x": 3_000_000.0 + c * vecindad.LADO_CELDA,
                    "y": 2_000_000.0 - f * vecindad.LADO_CELDA,
                    "fecha": fecha,
                    "target_ignicion": int((d, f, c) in marcadas),
                    "broadleaf_forest": 0.5,
                    "coniferous_forest": 0.1,
                    "mixed_forest": 0.1,
                })
    return pd.DataFrame(filas)


class TestRejilla:
    def test_deduce_la_geometria(self):
        m = malla(n_filas=7, n_columnas=9, n_dias=1)
        r = vecindad.construir_rejilla(m.x.to_numpy(), m.y.to_numpy())
        assert r.forma == (7, 9)

    def test_traduce_coordenadas_a_indices(self):
        m = malla(n_filas=5, n_columnas=5, n_dias=1)
        r = vecindad.construir_rejilla(m.x.to_numpy(), m.y.to_numpy())
        f, c = r.indices(m.x.to_numpy(), m.y.to_numpy())
        # La celda 0 es la esquina superior izquierda: y maximo, x minimo.
        assert (f[0], c[0]) == (0, 0)
        assert f.max() == 4 and c.max() == 4

    def test_cada_celda_tiene_un_indice_distinto(self):
        m = malla(n_filas=6, n_columnas=8, n_dias=1)
        r = vecindad.construir_rejilla(m.x.to_numpy(), m.y.to_numpy())
        f, c = r.indices(m.x.to_numpy(), m.y.to_numpy())
        assert len({(a, b) for a, b in zip(f, c)}) == len(m)


class TestSumaVentana:
    def test_una_sola_celda_encendida(self):
        matriz = np.zeros((7, 7))
        matriz[3, 3] = 1
        suma = vecindad._suma_ventana(matriz, radio=1)
        # Las nueve celdas de su entorno la ven, el resto no.
        assert suma[3, 3] == 1 and suma[2, 2] == 1 and suma[4, 4] == 1
        assert suma[1, 3] == 0 and suma[3, 5] == 0
        assert suma.sum() == 9

    def test_coincide_con_el_calculo_directo(self):
        rng = np.random.default_rng(3)
        matriz = rng.integers(0, 4, (11, 13)).astype(float)
        for radio in (1, 2, 4):
            esperado = np.zeros_like(matriz)
            for f in range(matriz.shape[0]):
                for c in range(matriz.shape[1]):
                    esperado[f, c] = matriz[max(0, f - radio):f + radio + 1,
                                            max(0, c - radio):c + radio + 1].sum()
            np.testing.assert_allclose(vecindad._suma_ventana(matriz, radio), esperado)

    def test_el_borde_no_da_la_vuelta(self):
        matriz = np.zeros((6, 6))
        matriz[0, 0] = 1
        suma = vecindad._suma_ventana(matriz, radio=2)
        assert suma[5, 5] == 0, "la esquina opuesta no puede verla"


class TestSinFugaTemporal:
    """El grupo de pruebas que justifica que estas variables sean legítimas."""

    def test_una_ignicion_no_aparece_en_su_propio_historial(self):
        """REGRESIÓN conceptual. Si la ventana incluyera el día en curso, la variable sería la
        respuesta disfrazada de pregunta: exactamente el defecto por el que el contrato prohíbe
        `is_near_ignition_25x25_10d`."""
        m = malla(igniciones=[(20, 6, 6)])
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        salida = vecindad.anadir_vecindad(m, ctx)

        fila = salida[(salida.target_ignicion == 1)]
        assert len(fila) == 1
        assert fila["igniciones_2km_7d"].iloc[0] == 0, (
            "la unica ignicion se esta viendo a si misma"
        )

    def test_la_ignicion_aparece_al_dia_siguiente(self):
        m = malla(igniciones=[(20, 6, 6)])
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        salida = vecindad.anadir_vecindad(m, ctx)
        salida["dia"] = (pd.to_datetime(salida.fecha).astype("int64")
                         // 86_400_000_000_000) - ctx.dia_minimo

        misma_celda = salida[(salida.x == 3_000_000.0 + 6 * vecindad.LADO_CELDA) &
                             (salida.y == 2_000_000.0 - 6 * vecindad.LADO_CELDA)]
        por_dia = misma_celda.set_index("dia")["igniciones_2km_7d"]
        assert por_dia[20] == 0, "el dia propio no cuenta"
        assert por_dia[21] == 1, "al dia siguiente si"
        assert por_dia[27] == 1, "sigue dentro de la ventana de 7 dias"
        assert por_dia[28] == 0, "fuera de la ventana ya no"

    def test_nada_antes_de_la_ignicion(self):
        m = malla(igniciones=[(20, 6, 6)])
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(3,), ventanas=(30,))
        salida = vecindad.anadir_vecindad(m, ctx)
        salida["dia"] = (pd.to_datetime(salida.fecha).astype("int64")
                         // 86_400_000_000_000) - ctx.dia_minimo
        assert salida[salida.dia <= 20]["igniciones_3km_30d"].sum() == 0

    def test_los_dias_transcurridos_tampoco_miran_el_dia_propio(self):
        m = malla(igniciones=[(20, 6, 6)])
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,),
                                                 dias_maximo=90)
        salida = vecindad.anadir_vecindad(m, ctx)
        salida["dia"] = (pd.to_datetime(salida.fecha).astype("int64")
                         // 86_400_000_000_000) - ctx.dia_minimo
        celda = salida[(salida.x == 3_000_000.0 + 6 * vecindad.LADO_CELDA) &
                       (salida.y == 2_000_000.0 - 6 * vecindad.LADO_CELDA)]
        por_dia = celda.set_index("dia")["dias_desde_ignicion_2km"]
        assert por_dia[20] == 90, "el dia de la ignicion aun no la conoce"
        assert por_dia[21] == 1
        assert por_dia[25] == 5


class TestAlcanceEspacial:
    def test_solo_las_celdas_dentro_del_radio_la_ven(self):
        m = malla(igniciones=[(10, 6, 6)])
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        salida = vecindad.anadir_vecindad(m, ctx)
        salida["dia"] = (pd.to_datetime(salida.fecha).astype("int64")
                         // 86_400_000_000_000) - ctx.dia_minimo
        dia11 = salida[salida.dia == 11].copy()
        f, c = ctx.rejilla.indices(dia11.x.to_numpy(), dia11.y.to_numpy())
        dia11["dist"] = np.maximum(np.abs(f - 6), np.abs(c - 6))

        assert (dia11[dia11.dist <= 2]["igniciones_2km_7d"] == 1).all()
        assert (dia11[dia11.dist > 2]["igniciones_2km_7d"] == 0).all()

    def test_un_radio_mayor_nunca_ve_menos(self):
        rng = np.random.default_rng(11)
        ign = [(int(d), int(f), int(c)) for d, f, c in
               zip(rng.integers(0, 40, 25), rng.integers(0, 12, 25), rng.integers(0, 12, 25))]
        m = malla(igniciones=ign)
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(1, 4), ventanas=(30,))
        salida = vecindad.anadir_vecindad(m, ctx)
        assert (salida["igniciones_4km_30d"] >= salida["igniciones_1km_30d"]).all()

    def test_una_ventana_mayor_nunca_ve_menos(self):
        rng = np.random.default_rng(12)
        ign = [(int(d), int(f), int(c)) for d, f, c in
               zip(rng.integers(0, 40, 25), rng.integers(0, 12, 25), rng.integers(0, 12, 25))]
        m = malla(igniciones=ign)
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(3,), ventanas=(5, 20))
        salida = vecindad.anadir_vecindad(m, ctx)
        assert (salida["igniciones_3km_20d"] >= salida["igniciones_3km_5d"]).all()


class TestContinuidadCombustible:
    def test_promedia_las_tres_fracciones_de_bosque(self):
        m = malla(n_dias=3)
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        salida = vecindad.anadir_vecindad(m, ctx)
        # Todas las celdas valen 0,5 + 0,1 + 0,1 = 0,7
        assert salida["forestal_vecindad_2km"].dropna().round(4).eq(0.7).all()

    def test_es_estatica_en_el_tiempo(self):
        m = malla(n_dias=5, igniciones=[(2, 5, 5)])
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        salida = vecindad.anadir_vecindad(m, ctx)
        por_celda = salida.groupby("cell_id")["forestal_vecindad_2km"].nunique()
        assert (por_celda <= 1).all(), "la cobertura no puede variar de un dia a otro"


class TestContrato:
    def test_declara_lo_que_genera(self):
        m = malla(n_dias=5)
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2, 5), ventanas=(7, 30))
        salida = vecindad.anadir_vecindad(m, ctx)
        for nombre in vecindad.columnas_vecindad(radios=(2, 5), ventanas=(7, 30)):
            assert nombre in salida.columns, f"declarada y no generada: {nombre}"

    def test_no_pierde_ni_duplica_filas(self):
        m = malla(n_dias=6)
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        assert len(vecindad.anadir_vecindad(m, ctx)) == len(m)

    def test_no_modifica_el_marco_original(self):
        m = malla(n_dias=4)
        columnas = list(m.columns)
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        vecindad.anadir_vecindad(m, ctx)
        assert list(m.columns) == columnas

    def test_es_sin_estado(self):
        """Procesar por lotes exige que el resultado no dependa de qué más se procesó antes."""
        m = malla(n_dias=10, igniciones=[(3, 4, 4), (5, 8, 8)])
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        entero = vecindad.anadir_vecindad(m, ctx)
        trozo = vecindad.anadir_vecindad(m.iloc[300:500].copy(), ctx)
        columnas = vecindad.columnas_vecindad(radios=(2,), ventanas=(7,))
        columnas = [c for c in columnas if c in trozo.columns]
        np.testing.assert_allclose(trozo[columnas].to_numpy(),
                                   entero[columnas].iloc[300:500].to_numpy())

    def test_avisa_si_faltan_columnas(self):
        with pytest.raises(KeyError, match="columnas ausentes"):
            vecindad.ajustar_contexto_espacial(pd.DataFrame({"cualquiera": [1, 2]}))

    def test_desde_el_contrato_deduplica_los_dias_de_borde(self, dataset_falso):
        """`ajustar_desde_contrato` reúne los positivos y dos días completos con consultas que
        se solapan; el recuento tiene que salir igual que contando las igniciones únicas."""
        c = mod_contrato.cargar(dataset_falso)
        ctx = vecindad.ajustar_desde_contrato(c, [2019], radios=(2,), ventanas=(7,))
        assert ctx.rejilla.forma == (5, 8)

    def test_avisa_si_hay_igniciones_repetidas(self):
        """REGRESIÓN. Al reunir el marco del contexto con dos consultas solapadas —todos los
        positivos, más dos días completos para fijar el rango— las igniciones que caían en un
        día de borde entraban dos veces, y el historial las contaba como dos incendios
        distintos. Eran 2 de 5.659, lo bastante pocas para no notarse en ninguna métrica y lo
        bastante graves para que las variables dejaran de significar lo que dicen."""
        m = malla(n_dias=8, igniciones=[(3, 5, 5)])
        repetido = pd.concat([m, m[m.target_ignicion == 1]], ignore_index=True)
        with pytest.raises(ValueError, match="igniciones repetidas"):
            vecindad.ajustar_contexto_espacial(repetido, radios=(2,), ventanas=(7,))

    def test_avisa_si_el_lote_cae_fuera_del_rango(self):
        m = malla(n_dias=5)
        ctx = vecindad.ajustar_contexto_espacial(m, radios=(2,), ventanas=(7,))
        futuro = m.copy()
        futuro["fecha"] = pd.to_datetime(futuro["fecha"]) + pd.Timedelta(days=400)
        with pytest.raises(ValueError, match="fuera del rango temporal"):
            vecindad.anadir_vecindad(futuro, ctx)


class TestIntegracionConElExperimento:
    """La bandera `usar_vecindad` tiene que hacer exactamente lo que dice: ni menos, ni —lo que
    importa más— más. Está apagada por defecto porque encenderla sola cambiaría en silencio
    cifras ya publicadas."""

    def test_apagada_por_defecto(self):
        assert experimento.Configuracion().usar_vecindad is False

    def test_apagada_no_anade_ninguna_columna(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        cfg = experimento.Configuracion(
            anios_train=[2019], anios_validacion=[2020], usar_derivadas=False)
        prep = experimento.preparar(c, cfg)
        assert prep.contexto_espacial is None
        assert not [v for v in prep.variables if v.startswith(("igniciones_", "dias_desde_"))]

    def test_encendida_anade_las_variables_a_la_lista(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        cfg = experimento.Configuracion(
            anios_train=[2019], anios_validacion=[2020], usar_derivadas=False,
            usar_vecindad=True, radios_vecindad=(2,), ventanas_vecindad=(7,))
        prep = experimento.preparar(c, cfg)

        assert prep.contexto_espacial is not None
        esperadas = vecindad.columnas_vecindad(radios=(2,), ventanas=(7,))
        for nombre in esperadas:
            assert nombre in prep.train.columns, f"no generada: {nombre}"
            assert nombre in prep.variables, f"generada pero no declarada: {nombre}"

    def test_las_coordenadas_no_entran_como_predictoras(self, dataset_falso):
        """`x` e `y` sitúan la fila en la rejilla; usarlas como variables sería memorizar
        coordenadas, que es otra intervención distinta y no generaliza a un año nuevo."""
        c = mod_contrato.cargar(dataset_falso)
        cfg = experimento.Configuracion(
            anios_train=[2019], anios_validacion=[2020], usar_derivadas=False,
            usar_vecindad=True, radios_vecindad=(2,), ventanas_vecindad=(7,))
        prep = experimento.preparar(c, cfg)

        assert "x" in prep.train.columns and "y" in prep.train.columns
        assert "x" not in prep.variables and "y" not in prep.variables
