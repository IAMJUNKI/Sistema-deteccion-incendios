"""Fixtures compartidas para la suite de `src/entrenamiento`.

La pieza central es `dataset_falso`: un datacubo EGIF en miniatura, escrito en disco con la
misma estructura que el real (`metadata.json` más `year=YYYY/dataset_YYYY.parquet`). Permite
probar la carga, el muestreo y los guardianes de cobertura sin depender de los 6,5 GB del
dataset de verdad, que ni están en integración continua ni deben estarlo.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: Módulos de prueba del pipeline geoespacial. Importan `geopandas`, `rasterio` y `netcdf4`,
#: que solo están en el entorno `incendios-forestales`. Los tests de `src/entrenamiento` no
#: dependen de ninguno de ellos, así que se omiten estos en vez de dejar que un
#: `ModuleNotFoundError` durante la recolección impida ejecutar la suite entera.
_TESTS_GEOESPACIALES = (
    "test_grid.py", "test_human_activity.py", "test_ingest_egif.py",
    "test_ingestion_pipeline.py", "test_landcover.py", "test_meteorology.py",
    "test_topography.py", "test_workflow.py", "test_pipeline.py", "test_era5.py",
    "test_datacube_profile.py", "test_webapp_components.py", "test_fwi.py",
)

collect_ignore = (
    [] if importlib.util.find_spec("geopandas") else list(_TESTS_GEOESPACIALES)
)

CELDAS = 40
DIAS_POR_ANIO = 30
ANIOS = (2019, 2020)

#: Las 40 celdas se disponen en una rejilla de 8 x 5, con el mismo paso de 1 km del datacubo.
COLUMNAS_REJILLA = 8
X0, Y0 = 3_000_000.0, 2_500_000.0

#: Predictores del dataset falso, uno por grupo temático para que la clasificación por
#: patrón quede ejercitada de verdad.
PREDICTORES = [
    "temperature_max", "relative_humidity_min", "wind_speed_max", "precipitation_sum",
    "precipitation_sum_7d", "vpd_mean",
    "elevation_mean", "slope_mean", "aspect_000_045_fraction",
    "broadleaf_forest", "coniferous_forest", "mixed_forest", "scrub",
    "road_length_km", "road_length_main_km",
]


@pytest.fixture(autouse=True)
def _disable_local_simulation_from_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los tests no deben depender del `.env` local de quien los ejecuta."""

    monkeypatch.setenv("LOCAL_SIMULATION_MODE", "false")
    monkeypatch.setenv("FORECAST_MODEL_FAMILY", "auto")
    monkeypatch.setenv("SHADOW_50_MODEL", "false")


def _marco(anio: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(f"{anio}-06-01", periods=DIAS_POR_ANIO, freq="D")
    celdas = np.arange(CELDAS, dtype=np.int32)
    malla = pd.MultiIndex.from_product([fechas, celdas], names=["fecha", "cell_id"])
    df = pd.DataFrame(index=malla).reset_index()

    # Coordenadas proyectadas, como en el datacubo real (EPSG:3035, paso de 1 km). No son
    # predictoras —el contrato no las declara— pero sin ellas no se puede probar el contexto
    # espacial de extremo a extremo, porque `cell_id` no revela dónde está la celda.
    df["x"] = (X0 + (df.cell_id % COLUMNAS_REJILLA) * 1000.0).astype(np.float64)
    df["y"] = (Y0 - (df.cell_id // COLUMNAS_REJILLA) * 1000.0).astype(np.float64)

    n = len(df)
    df["temperature_max"] = rng.normal(28, 5, n).astype(np.float32)
    df["relative_humidity_min"] = rng.uniform(10, 90, n).astype(np.float32)
    df["wind_speed_max"] = rng.gamma(2, 6, n).astype(np.float32)
    df["precipitation_sum"] = rng.gamma(0.4, 3, n).astype(np.float32)
    df["precipitation_sum_7d"] = rng.gamma(1.5, 5, n).astype(np.float32)
    df["vpd_mean"] = rng.gamma(2, 0.5, n).astype(np.float32)

    # Estáticas: constantes por celda, como en el datacubo real.
    por_celda = rng.random((CELDAS, 4)).astype(np.float32)
    df["elevation_mean"] = (por_celda[df.cell_id, 0] * 1200).astype(np.float32)
    df["slope_mean"] = (por_celda[df.cell_id, 1] * 30).astype(np.float32)
    df["aspect_000_045_fraction"] = por_celda[df.cell_id, 2]
    df["road_length_km"] = (por_celda[df.cell_id, 3] * 20).astype(np.float32)
    df["road_length_main_km"] = (df["road_length_km"] * 0.3).astype(np.float32)

    fracciones = rng.dirichlet([1, 1, 1, 1], CELDAS).astype(np.float32)
    for i, nombre in enumerate(["broadleaf_forest", "coniferous_forest", "mixed_forest",
                                "scrub"]):
        df[nombre] = fracciones[df.cell_id, i]

    # Target con señal real: calor y sequedad suben la probabilidad, para que un modelo de
    # prueba pueda discriminar y las métricas no se evalúen sobre ruido puro.
    riesgo = (df.temperature_max - 28) / 5 - (df.relative_humidity_min - 50) / 20
    prob = 1 / (1 + np.exp(-(riesgo - 3.0)))
    df["target_ignicion"] = (rng.random(n) < prob).astype(np.uint8)
    df["burned_area_ha"] = (df.target_ignicion * rng.gamma(2, 30, n)).astype(np.float32)
    df["year"] = np.int16(anio)
    return df


@pytest.fixture(scope="session")
def dataset_falso(tmp_path_factory) -> Path:
    """Escribe un dataset EGIF en miniatura y devuelve su directorio."""
    raiz = tmp_path_factory.mktemp("egif_falso")
    filas = {}
    for i, anio in enumerate(ANIOS):
        df = _marco(anio, semilla=100 + i)
        destino = raiz / f"year={anio}"
        destino.mkdir(parents=True)
        df.to_parquet(destino / f"dataset_{anio}.parquet", index=False)
        filas[str(anio)] = len(df)

    (raiz / "metadata.json").write_text(json.dumps({
        "row_definition": "celda-dia de prueba",
        "target": "target_ignicion (EGIF-MITECO)",
        "outcome_columns_not_predictors": ["target_ignicion", "burned_area_ha"],
        "predictor_columns": PREDICTORES,
        "partitioning": "year=YYYY/dataset_YYYY.parquet",
        "annual_files": filas,
        "dropped_rows_incomplete_predictors": 0,
        "time_contract": "No temporal shift; meteorological accumulations include date T.",
    }, indent=2), encoding="utf-8")
    return raiz


@pytest.fixture
def prediccion_desbalanceada():
    """Etiquetas, puntuaciones e índice de día con desbalanceo realista.

    Devuelve `(y, p, dia)` con prevalencia del orden de 10⁻³ y una señal moderada, que es el
    régimen en el que las métricas del proyecto tienen que comportarse bien.
    """
    rng = np.random.default_rng(11)
    n_dias, por_dia = 60, 500
    dia = np.repeat(np.arange(n_dias), por_dia)
    y = (rng.random(n_dias * por_dia) < 0.004).astype(np.int8)
    p = np.clip(rng.beta(2, 30, len(y)) + y * rng.uniform(0.1, 0.4, len(y)), 0, 1)
    return y, p, dia
