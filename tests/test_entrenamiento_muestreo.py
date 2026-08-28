"""Estrategias de submuestreo de negativos y su compatibilidad con la corrección de prior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entrenamiento import calibracion, muestreo


@pytest.fixture
def poblacion():
    """Población sintética con estructura: estratos de tamaño muy distinto."""
    rng = np.random.default_rng(17)
    n = 60_000
    cell_id = rng.integers(0, 500, n)
    dia = rng.integers(0, 365, n)
    mes = (dia // 30 + 1).clip(1, 12)
    # Combustible desequilibrado a propósito: `humedal` es rarísimo, y es justo el estrato que
    # el muestreo uniforme puede hacer desaparecer.
    combustible = rng.choice(["matorral", "bosque", "cultivo", "humedal"],
                             size=n, p=[0.45, 0.35, 0.19, 0.01])
    objetivo = (rng.random(n) < 0.004).astype(np.int8)
    marco = pd.DataFrame({
        "cell_id": cell_id, "dia": dia, "mes": mes, "combustible": combustible,
        "target": objetivo,
        "temperature_max": rng.normal(25, 6, n),
        "relative_humidity_min": rng.uniform(15, 90, n),
        "vpd_mean": rng.gamma(2, 0.5, n),
        "precipitation_sum_7d": rng.gamma(1.5, 4, n),
    })
    return marco


class TestUniforme:
    def test_conserva_todos_los_positivos(self, poblacion):
        m = muestreo.uniforme(poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
                              poblacion.dia.to_numpy(), modulo=25)
        assert poblacion.loc[m.mascara, "target"].sum() == poblacion.target.sum()

    def test_la_tasa_se_acerca_al_inverso_del_modulo(self, poblacion):
        """La tasa se **mide** sobre lo conservado, no se supone: el hash no reparte
        exactamente 1/modulo, y usar el valor nominal descalibraría esas filas."""
        m = muestreo.uniforme(poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
                              poblacion.dia.to_numpy(), modulo=25)
        negativos = (poblacion.loc[m.mascara, "target"] == 0).to_numpy()
        assert m.tasas[negativos].mean() == pytest.approx(0.04, rel=0.05)

        # Y coincide exactamente con la proporción real conservada.
        real = negativos.sum() / (poblacion.target == 0).sum()
        assert m.tasas[negativos][0] == pytest.approx(real, rel=1e-9)

    def test_los_positivos_llevan_tasa_uno(self, poblacion):
        m = muestreo.uniforme(poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
                              poblacion.dia.to_numpy())
        positivos = (poblacion.loc[m.mascara, "target"] == 1).to_numpy()
        assert (m.tasas[positivos] == 1.0).all()

    def test_es_reproducible(self, poblacion):
        args = (poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
                poblacion.dia.to_numpy())
        assert (muestreo.uniforme(*args).mascara == muestreo.uniforme(*args).mascara).all()

    def test_restos_distintos_son_disjuntos(self, poblacion):
        args = (poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
                poblacion.dia.to_numpy())
        a = muestreo.uniforme(*args, resto=0).mascara & (poblacion.target == 0).to_numpy()
        b = muestreo.uniforme(*args, resto=1).mascara & (poblacion.target == 0).to_numpy()
        assert not (a & b).any()


class TestDosEtapasDeMuestreo:
    """REGRESIÓN. El estudio de muestreo carga un embalse por hash y luego muestrea dentro.

    Reutilizando el mismo hash en las dos etapas las tasas **no se multiplican**: comparten
    divisores, así que pedir `% 5` y después `% 25` no descarta nada extra respecto de `% 5`.
    El primer barrido salió con más negativos en el ratio 1:25 que en el 1:10, y comparaba
    estrategias con volúmenes de datos distintos sin que se notara.
    """

    @staticmethod
    def _poblacion(n=200_000, semilla=5):
        rng = np.random.default_rng(semilla)
        return (np.zeros(n, dtype=np.int8),
                rng.integers(0, 3_000, n), rng.integers(0, 1_500, n))

    def test_el_mismo_hash_no_compone_las_tasas(self):
        """Deja constancia del comportamiento defectuoso, para que no se reintroduzca."""
        objetivo, celda, dia = self._poblacion()
        embalse = muestreo.uniforme(objetivo, celda, dia, modulo=5).mascara
        dentro = muestreo.uniforme(objetivo[embalse], celda[embalse], dia[embalse],
                                   modulo=5, independiente=False).mascara
        assert dentro.all(), "con el mismo hash, el segundo %5 no descarta nada"

    @pytest.mark.parametrize("interno", [2, 4, 5, 8])
    def test_la_seleccion_aleatoria_compone_las_tasas(self, interno):
        objetivo, celda, dia = self._poblacion()
        embalse = muestreo.uniforme(objetivo, celda, dia, modulo=5).mascara
        m = muestreo.uniforme(objetivo[embalse], celda[embalse], dia[embalse],
                              modulo=interno, independiente=True)
        conservados = int(m.mascara.sum())
        esperado = int(embalse.sum()) / interno
        assert conservados == pytest.approx(esperado, rel=0.02), (
            f"con embalse 1:5 e interno 1:{interno} el efectivo debe ser 1:{5 * interno}"
        )

    def test_ratios_mayores_dan_menos_filas(self):
        """La comprobación que habría cazado el primer fallo de un vistazo."""
        objetivo, celda, dia = self._poblacion()
        embalse = muestreo.uniforme(objetivo, celda, dia, modulo=5).mascara
        tamanos = [
            int(muestreo.uniforme(objetivo[embalse], celda[embalse], dia[embalse],
                                  modulo=i, independiente=True).mascara.sum())
            for i in (2, 4, 8, 16)
        ]
        assert tamanos == sorted(tamanos, reverse=True), tamanos

    @pytest.mark.parametrize("interno", [4, 5, 8])
    def test_no_degenera_en_un_muestreo_espacial(self, interno):
        """REGRESIÓN del fallo grave, que el recuento de filas NO delataba.

        Un segundo hash sobre las mismas claves puede reducirse a `celda % k == 0`: conserva el
        número correcto de filas pero elige **celdas enteras con todos sus días**, de modo que
        el modelo solo ve una fracción del territorio. En el barrido eso hundió el recall de
        0,39 a 0,24 sin que nada fallara ni el conteo cuadrara mal.

        La firma de la degeneración es inconfundible: la cobertura de celdas cae a `1/interno`
        y las celdas supervivientes conservan **todos** sus días. Un muestreo sano toca casi
        todas las celdas y se queda con una fracción de los días de cada una.
        """
        objetivo, celda, dia = self._poblacion()
        embalse = muestreo.uniforme(objetivo, celda, dia, modulo=5).mascara
        celda_pool = celda[embalse]

        m = muestreo.uniforme(objetivo[embalse], celda[embalse], dia[embalse],
                              modulo=interno, independiente=True)
        cobertura = len(np.unique(celda_pool[m.mascara])) / len(np.unique(celda_pool))

        assert cobertura > 3 / interno, (
            f"cobertura {cobertura:.0%}, cerca del {1 / interno:.0%} que daría un muestreo "
            f"puramente espacial: la selección degeneró"
        )

        # Y las celdas que sobreviven no se llevan todos sus días.
        dias_pool = pd.Series(celda_pool).value_counts()
        dias_muestra = pd.Series(celda_pool[m.mascara]).value_counts()
        fraccion = (dias_muestra / dias_pool.reindex(dias_muestra.index)).mean()
        assert fraccion < 0.6, (
            f"cada celda conserva el {fraccion:.0%} de sus días: parece selección de celdas"
        )

    def test_la_tasa_declarada_coincide_con_la_real(self):
        """La tasa se mide sobre lo conservado; si se supusiera 1/modulo, la corrección de
        prior quedaría mal cuando el redondeo del cupo no cuadra."""
        objetivo, celda, dia = self._poblacion()
        embalse = muestreo.uniforme(objetivo, celda, dia, modulo=5).mascara
        m = muestreo.uniforme(objetivo[embalse], celda[embalse], dia[embalse],
                              modulo=7, independiente=True)
        real = int(m.mascara.sum()) / int(embalse.sum())
        assert m.meta["tasa_global"] == pytest.approx(real, rel=1e-9)


class TestEstratificado:
    def test_ningun_estrato_desaparece(self, poblacion):
        """El motivo de existir de la estrategia: `humedal` es el 1 % de la población."""
        m = muestreo.estratificado(
            poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
            poblacion.dia.to_numpy(), poblacion[["mes", "combustible"]], modulo=25,
        )
        muestra = poblacion.loc[m.mascara]
        assert set(muestra["combustible"]) == set(poblacion["combustible"])
        for mes in poblacion["mes"].unique():
            assert (muestra["mes"] == mes).any(), f"desapareció el mes {mes}"

    def test_cada_fila_lleva_la_tasa_de_su_estrato(self, poblacion):
        """Sin esto la corrección de prior sería incorrecta para los estratos rescatados."""
        m = muestreo.estratificado(
            poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
            poblacion.dia.to_numpy(), poblacion[["mes", "combustible"]], modulo=25,
        )
        assert len(np.unique(m.tasas)) > 2, "debería haber varias tasas distintas"
        assert ((m.tasas > 0) & (m.tasas <= 1)).all()

    def test_los_estratos_escasos_se_conservan_enteros(self, poblacion):
        m = muestreo.estratificado(
            poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
            poblacion.dia.to_numpy(), poblacion[["mes", "combustible"]],
            modulo=25, minimo_por_estrato=100_000,   # fuerza el rescate de todos
        )
        assert m.mascara.all(), "con el mínimo por las nubes no se descarta nada"
        assert (m.tasas == 1.0).all()

    def test_la_tasa_declarada_coincide_con_la_real(self, poblacion):
        """La tasa se mide, no se supone: el hash no reparte exactamente 1/modulo."""
        m = muestreo.estratificado(
            poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
            poblacion.dia.to_numpy(), poblacion[["combustible"]], modulo=10,
        )
        muestra = poblacion.loc[m.mascara].assign(tasa=m.tasas, estrato=m.estrato)
        negativos = muestra[muestra.target == 0]
        for combustible, grupo in negativos.groupby("combustible"):
            total = int(((poblacion.combustible == combustible) & (poblacion.target == 0)).sum())
            assert grupo["tasa"].iloc[0] == pytest.approx(len(grupo) / total, rel=1e-9)


class TestPorClusters:
    def test_cubre_todos_los_grupos(self, poblacion):
        variables = poblacion[["temperature_max", "relative_humidity_min", "vpd_mean"]]
        m = muestreo.por_clusters(poblacion.target.to_numpy(), variables,
                                  modulo=10, n_clusters=8, submuestra_ajuste=5_000)
        conservados = poblacion.loc[m.mascara]
        assert (conservados.target == 1).sum() == poblacion.target.sum()
        assert len(np.unique(m.estrato[(conservados.target == 0).to_numpy()])) == 8

    def test_respeta_aproximadamente_el_presupuesto(self, poblacion):
        variables = poblacion[["temperature_max", "relative_humidity_min", "vpd_mean"]]
        m = muestreo.por_clusters(poblacion.target.to_numpy(), variables,
                                  modulo=10, n_clusters=8, submuestra_ajuste=5_000)
        negativos_totales = int((poblacion.target == 0).sum())
        conservados = int((poblacion.loc[m.mascara, "target"] == 0).sum())
        assert conservados == pytest.approx(negativos_totales / 10, rel=0.15)

    def test_las_tasas_son_validas(self, poblacion):
        variables = poblacion[["temperature_max", "relative_humidity_min", "vpd_mean"]]
        m = muestreo.por_clusters(poblacion.target.to_numpy(), variables,
                                  modulo=10, n_clusters=8, submuestra_ajuste=5_000)
        assert ((m.tasas > 0) & (m.tasas <= 1)).all()

    def test_proporcional_deja_las_tasas_casi_iguales(self, poblacion):
        """Reparto proporcional = reducción de varianza, no cambio de composición. Si las
        tasas salieran muy distintas es que no está repartiendo proporcionalmente."""
        variables = poblacion[["temperature_max", "relative_humidity_min", "vpd_mean"]]
        m = muestreo.por_clusters(poblacion.target.to_numpy(), variables, modulo=10,
                                  n_clusters=8, submuestra_ajuste=5_000,
                                  asignacion="proporcional")
        assert m.meta["tasa_maxima"] / m.meta["tasa_minima"] < 3

    def test_equilibrado_sobrerrepresenta_los_grupos_pequenos(self):
        """La variante que sí cambia la composición.

        Se construye un espacio **desigual** a propósito, que es como es el real: la inmensa
        mayoría de los días son suaves y las condiciones extremas de incendio son rarísimas. Con
        un espacio uniforme las dos asignaciones convergen y el test no probaría nada.
        """
        rng = np.random.default_rng(31)
        n_denso, n_raro = 19_000, 1_000
        variables = pd.DataFrame({
            "temperature_max": np.concatenate([rng.normal(15, 2, n_denso),
                                               rng.normal(38, 2, n_raro)]),
            "relative_humidity_min": np.concatenate([rng.normal(75, 5, n_denso),
                                                     rng.normal(18, 4, n_raro)]),
        })
        objetivo = np.zeros(len(variables), dtype=np.int8)
        objetivo[rng.choice(len(variables), 40, replace=False)] = 1

        m = muestreo.por_clusters(objetivo, variables, modulo=10, n_clusters=6,
                                  submuestra_ajuste=5_000, asignacion="equilibrado")

        assert m.meta["tasa_maxima"] / m.meta["tasa_minima"] > 3, (
            "los grupos raros deben ceder una fracción mucho mayor de sus filas"
        )
        assert len(np.unique(m.tasas)) > 2, "cada grupo debe tener su propia tasa"

        # Y la comprobación que importa: la región rara queda sobrerrepresentada frente a su
        # peso real en la población.
        conservados = variables.loc[m.mascara & (objetivo == 0)]
        peso_real = n_raro / (n_denso + n_raro)
        peso_muestra = (conservados["temperature_max"] > 28).mean()
        assert peso_muestra > peso_real * 2

    def test_equilibrado_reparte_de_forma_mas_pareja(self, poblacion):
        """Comprobación directa de la diferencia entre las dos asignaciones."""
        variables = poblacion[["temperature_max", "relative_humidity_min", "vpd_mean"]]
        tamanos = {}
        for asignacion in ("proporcional", "equilibrado"):
            m = muestreo.por_clusters(poblacion.target.to_numpy(), variables, modulo=10,
                                      n_clusters=8, submuestra_ajuste=5_000,
                                      asignacion=asignacion)
            negativos = (poblacion.loc[m.mascara, "target"] == 0).to_numpy()
            _, cuentas = np.unique(m.estrato[negativos], return_counts=True)
            tamanos[asignacion] = cuentas.std() / cuentas.mean()
        assert tamanos["equilibrado"] < tamanos["proporcional"], (
            "el equilibrado debe dar grupos de tamaño más parecido entre sí"
        )

    def test_asignacion_desconocida_es_error(self, poblacion):
        variables = poblacion[["temperature_max", "relative_humidity_min", "vpd_mean"]]
        with pytest.raises(ValueError, match="proporcional"):
            muestreo.por_clusters(poblacion.target.to_numpy(), variables,
                                  asignacion="inventada")


class TestDuros:
    def test_prioriza_los_negativos_dificiles(self, poblacion):
        """Control negativo: se comprueba que hace lo que dice, no que funcione bien."""
        dificultad = muestreo.dificultad_climatica(poblacion)
        m = muestreo.duros(poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
                           poblacion.dia.to_numpy(), dificultad, modulo=25, fraccion_dura=0.8)
        muestra = poblacion.loc[m.mascara]
        negativos = (muestra.target == 0).to_numpy()
        assert dificultad[m.mascara][negativos].mean() > dificultad[poblacion.target == 0].mean()

    def test_avisa_de_que_su_calibracion_no_es_fiable(self, poblacion):
        dificultad = muestreo.dificultad_climatica(poblacion)
        m = muestreo.duros(poblacion.target.to_numpy(), poblacion.cell_id.to_numpy(),
                           poblacion.dia.to_numpy(), dificultad)
        assert "aviso" in m.meta

    def test_la_dificultad_necesita_meteorologia(self):
        with pytest.raises(ValueError, match="dificultad"):
            muestreo.dificultad_climatica(pd.DataFrame({"cualquier_cosa": [1, 2, 3]}))


class TestPriorCorrectionConTasasPorFila:
    def test_acepta_un_vector_de_tasas(self):
        p = np.array([0.5, 0.5, 0.5])
        tasas = np.array([1 / 10, 1 / 25, 1 / 50])
        corregidas = calibracion.prior_correction(p, tasas)
        # Con la misma probabilidad de partida, menor tasa implica mayor corrección a la baja.
        assert corregidas[0] > corregidas[1] > corregidas[2]

    def test_coincide_con_el_escalar_cuando_todas_son_iguales(self):
        p = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(
            calibracion.prior_correction(p, np.full(3, 0.04)),
            calibracion.prior_correction(p, 0.04),
        )

    def test_rechaza_un_vector_de_forma_distinta(self):
        with pytest.raises(ValueError, match="forma de y_prob"):
            calibracion.prior_correction(np.array([0.5, 0.5]), np.array([0.1, 0.2, 0.3]))

    def test_rechaza_tasas_fuera_de_rango(self):
        with pytest.raises(ValueError, match="en \\(0, 1\\]"):
            calibracion.prior_correction(np.array([0.5, 0.5]), np.array([0.1, 1.5]))

    def test_recupera_la_prevalencia_con_tasas_heterogeneas(self):
        """La prueba que de verdad importa: si el muestreo fue estratificado y se corrige con
        las tasas correctas, la probabilidad media debe volver a la prevalencia real."""
        rng = np.random.default_rng(23)
        n = 40_000
        estrato = rng.integers(0, 3, n)
        tasas_estrato = np.array([1 / 5, 1 / 25, 1 / 100])
        tasas = tasas_estrato[estrato]

        prevalencia_real = 0.001
        # Probabilidad inflada exactamente por el factor del muestreo de cada estrato.
        odds_real = prevalencia_real / (1 - prevalencia_real)
        odds_muestra = odds_real / tasas
        p_muestra = odds_muestra / (1 + odds_muestra)

        corregidas = calibracion.prior_correction(p_muestra, tasas)
        np.testing.assert_allclose(corregidas, prevalencia_real, rtol=1e-9)

        # Y con la tasa global equivocada, la calibración se va: este es el fallo que la rama
        # vectorial evita.
        con_tasa_global = calibracion.prior_correction(p_muestra, float(tasas.mean()))
        assert abs(con_tasa_global.mean() - prevalencia_real) > prevalencia_real * 0.1
