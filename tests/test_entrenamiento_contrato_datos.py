"""Contrato del dataset, carga y muestreo.

Se apoyan en `dataset_falso`, un datacubo EGIF en miniatura con la misma estructura que el real.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.entrenamiento import contrato as mod_contrato, datos


class TestClasificacionPorPatron:
    """El pipeline debe sobrevivir a que el dataset gane o pierda columnas."""

    @pytest.mark.parametrize("variable,grupo", [
        ("temperature_max", "meteorologia"),
        ("precipitation_sum_60d", "meteorologia"),      # ventana que aún no existe
        ("vpd_max_12_18h", "meteorologia"),
        ("elevation_mean", "topografia"),
        ("aspect_315_360_fraction", "topografia"),
        ("broadleaf_forest", "cobertura"),
        ("road_length_main_km", "actividad_humana"),
        ("building_area_fraction", "actividad_humana"),
        ("day_of_year_sin", "calendario"),
        ("inventada_xyz", "otras"),
    ])
    def test_clasifica_por_nombre(self, variable, grupo):
        assert mod_contrato.clasificar(variable) == grupo


class TestCargaDelContrato:
    def test_lee_el_esquema_real(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        assert len(c.predictores) == 15
        assert c.anios == [2019, 2020]
        assert c.filas(c.anios) == sum(c.filas_por_anio.values())
        assert "No temporal shift" in c.contrato_temporal

    def test_agrupa_todas_las_variables(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        assert sum(len(v) for v in c.grupos.values()) == len(c.predictores)
        assert "otras" not in c.grupos, "toda variable real debe caer en un grupo temático"

    def test_tiene_y_del_grupo(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        assert c.tiene("temperature_max", "vpd_mean")
        assert not c.tiene("no_existe")
        assert "elevation_mean" in c.del_grupo("topografia")

    def test_falta_el_metadato(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            mod_contrato.cargar(tmp_path)

    def test_rechaza_predictores_prohibidos(self, dataset_falso, tmp_path):
        """Barrera contra fuga de información: `is_near_ignition_25x25_10d` describe si hubo
        una ignición cerca en los diez días anteriores. Usarla sería mirar la respuesta."""
        destino = tmp_path / "fuga"
        destino.mkdir()
        meta = json.loads((dataset_falso / "metadata.json").read_text(encoding="utf-8"))
        meta["predictor_columns"] += ["is_near_ignition_25x25_10d"]
        (destino / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(ValueError, match="prohibidas"):
            mod_contrato.cargar(destino, verificar_esquema=False)

    def test_detecta_metadato_que_miente(self, dataset_falso, tmp_path):
        """Mejor fallar al arrancar que a los veinte minutos de entrenamiento."""
        destino = tmp_path / "desfasado"
        destino.mkdir()
        for anio in (2019, 2020):
            enlace = destino / f"year={anio}"
            enlace.mkdir()
            (enlace / f"dataset_{anio}.parquet").write_bytes(
                (dataset_falso / f"year={anio}" / f"dataset_{anio}.parquet").read_bytes()
            )
        meta = json.loads((dataset_falso / "metadata.json").read_text(encoding="utf-8"))
        meta["predictor_columns"] += ["columna_que_no_existe"]
        (destino / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(ValueError, match="no están en"):
            mod_contrato.cargar(destino)

    def test_particion_ausente(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        with pytest.raises(FileNotFoundError, match="partición anual"):
            c.ruta(2099)


class TestMuestreo:
    def test_conserva_todos_los_positivos(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        muestra, meta = datos.muestrear_entrenamiento(c, [2019], modulo=5)
        completo = pd.read_parquet(c.ruta(2019), columns=["target_ignicion"])
        assert meta["positivos"] == int(completo["target_ignicion"].sum())
        assert int(muestra["target_ignicion"].sum()) == meta["positivos"]

    def test_la_tasa_de_negativos_se_acerca_al_inverso_del_modulo(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        _, meta = datos.muestrear_entrenamiento(c, [2019, 2020], modulo=5)
        assert meta["tasa_negativos"] == pytest.approx(0.2, abs=0.03)

    def test_la_muestra_infla_la_prevalencia(self, dataset_falso):
        """El motivo de que exista la corrección de prior."""
        c = mod_contrato.cargar(dataset_falso)
        _, meta = datos.muestrear_entrenamiento(c, [2019], modulo=5)
        assert meta["prevalencia_muestra"] > meta["prevalencia_real"]

    def test_restos_distintos_dan_muestras_distintas(self, dataset_falso):
        """Es lo que permite comprobar si un resultado se sostiene o depende de la muestra."""
        c = mod_contrato.cargar(dataset_falso)
        a, _ = datos.muestrear_entrenamiento(c, [2019], modulo=5, resto=0)
        b, _ = datos.muestrear_entrenamiento(c, [2019], modulo=5, resto=1)
        neg_a = set(map(tuple, a.loc[a.target_ignicion == 0, ["cell_id", "fecha"]].values))
        neg_b = set(map(tuple, b.loc[b.target_ignicion == 0, ["cell_id", "fecha"]].values))
        assert neg_a and neg_b and not (neg_a & neg_b), "los restos deben ser disjuntos"

    def test_es_reproducible(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        a, _ = datos.muestrear_entrenamiento(c, [2019], modulo=5)
        b, _ = datos.muestrear_entrenamiento(c, [2019], modulo=5)
        pd.testing.assert_frame_equal(a, b)

    @pytest.mark.parametrize("modulo,resto", [(0, 0), (5, 5), (5, -1)])
    def test_parametros_invalidos(self, dataset_falso, modulo, resto):
        c = mod_contrato.cargar(dataset_falso)
        with pytest.raises(ValueError):
            datos.muestrear_entrenamiento(c, [2019], modulo=modulo, resto=resto)

    def test_variables_inexistentes(self, dataset_falso):
        c = mod_contrato.cargar(dataset_falso)
        with pytest.raises(ValueError, match="no están en el dataset"):
            datos.muestrear_entrenamiento(c, [2019], ["inventada"])


class TestEvaluacionCompleta:
    def test_recorre_todas_las_filas(self, dataset_falso):
        """Evaluar sobre la población completa es la diferencia entre una cifra publicable
        y una medida sobre una prevalencia que no existe."""
        c = mod_contrato.cargar(dataset_falso)
        vistas = sum(len(lote) for lote in datos.iter_evaluacion(c, [2019], filas_por_lote=97))
        assert vistas == c.filas_por_anio[2019]

    def test_guardian_de_cobertura(self, dataset_falso):
        """REGRESIÓN. Una ejecución perdió un bloque entero de celdas sin lanzar excepción y
        las métricas salieron MEJORES, porque faltaba una parte difícil del mapa."""
        c = mod_contrato.cargar(dataset_falso)
        datos.verificar_cobertura(c, [2019], c.filas_por_anio[2019])
        with pytest.raises(RuntimeError, match="Cobertura incompleta"):
            datos.verificar_cobertura(c, [2019], c.filas_por_anio[2019] - 1)


class TestIndiceDia:
    def test_es_constante_dentro_del_dia_y_crece_entre_dias(self):
        fechas = pd.Series(pd.to_datetime(
            ["2022-08-01", "2022-08-01", "2022-08-02", "2022-09-01"]
        ))
        indice = datos.indice_dia(fechas)
        assert indice[0] == indice[1]
        assert indice[2] == indice[0] + 1
        assert indice[3] == indice[0] + 31
