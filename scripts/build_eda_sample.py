"""Construye la muestra de trabajo para el análisis exploratorio de variables.

Un año completo son 11,2 millones de filas: no se puede explorar de forma interactiva. Este
script recorre el dataset por bloques, calcula las variables derivadas y guarda una muestra
manejable con **todos los incendios** y una fracción de los días sin incendio.

La climatología de referencia y los estadísticos diarios se aprenden **solo con los años de
entrenamiento** y luego se aplican a validación. Es la misma disciplina que se aplica a
cualquier parámetro ajustado: si se calcularan sobre el total, la muestra de exploración
llevaría dentro información del año de validación.

Uso:
    python -m scripts.build_eda_sample
    python -m scripts.build_eda_sample --neg-ratio 200 --mode nowcast

Salida:
    data/processed/eda_sample_<mode>.parquet   (ignorado por Git)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.derived import (
    BASE_VARS,
    add_derived_features,
    derived_columns,
    fit_cell_climatology,
    fit_daily_stats,
)
from src.models.dataset import (
    TARGET_COL,
    feature_columns,
    iter_blocks,
    resolve_data_root,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_eda_sample")

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera la muestra para el análisis exploratorio de variables."
    )
    parser.add_argument("--mode", choices=["nowcast", "t1"], default="t1")
    parser.add_argument("--train-years", nargs="+", type=int, default=[2019, 2020, 2021])
    parser.add_argument("--val-years", nargs="+", type=int, default=[2022])
    parser.add_argument("--neg-ratio", type=int, default=150,
                        help="Días sin incendio por cada incendio en la muestra.")
    parser.add_argument("--block-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = resolve_data_root()
    cols = feature_columns(cyclic=True)
    todos_los_años = args.train_years + args.val_years
    logger.info("Datos: %s | modo: %s", root, args.mode)

    def bloques(años):
        for año in años:
            yield from iter_blocks(año, cols, args.mode, args.block_size, root, cyclic=True)

    # ── 1. Estadísticos de referencia, SOLO con los años de entrenamiento ──
    logger.info("Aprendiendo climatología por celda y mes con %s...", args.train_years)
    clim = fit_cell_climatology(bloques(args.train_years), BASE_VARS)

    logger.info("Aprendiendo estadísticos espaciales diarios con %s...", args.train_years)
    # Los estadísticos diarios son por fecha, así que para los años de validación hay que
    # calcularlos también: no existe "el 14 de agosto de 2022" en el histórico de entrenamiento.
    # Se calculan por año, pero nunca usan la variable objetivo, así que no filtran información
    # sobre qué celdas ardieron.
    stats = fit_daily_stats(bloques(todos_los_años), BASE_VARS)

    # ── 2. Recorrido y muestreo ──
    rng = np.random.default_rng(args.seed)
    piezas = []
    n_pos_total = n_neg_total = 0

    for año in todos_los_años:
        n_pos = n_neg = 0
        for bloque in iter_blocks(año, cols, args.mode, args.block_size, root, cyclic=True):
            bloque = add_derived_features(bloque, clim, stats, BASE_VARS)

            es_pos = bloque[TARGET_COL].to_numpy() == 1
            # Se conservan todos los incendios y una fracción fija de los días sin incendio.
            # El muestreo es uniforme a propósito: estratificarlo hacia días de riesgo fue lo
            # que rompió el entrenamiento (ver MODEL_DOCUMENTATION.md §3.2).
            frac = min(1.0, args.neg_ratio * max(es_pos.sum(), 1) / max((~es_pos).sum(), 1))
            conservar = es_pos | ((rng.random(len(bloque)) < frac) & ~es_pos)

            sel = bloque.loc[conservar].copy()
            sel["anio"] = año
            sel["particion"] = "train" if año in args.train_years else "val"
            piezas.append(sel)
            n_pos += int(es_pos.sum())
            n_neg += int(conservar.sum() - es_pos.sum())

        logger.info("  %d: %s incendios | %s días sin incendio muestreados",
                    año, f"{n_pos:,}", f"{n_neg:,}")
        n_pos_total += n_pos
        n_neg_total += n_neg

    muestra = pd.concat(piezas, ignore_index=True)

    salida = REPO_ROOT / "data" / "processed" / f"eda_sample_{args.mode}.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    muestra.to_parquet(salida, index=False)

    logger.info("=" * 70)
    logger.info("Muestra guardada en %s", salida)
    logger.info("  filas: %s | incendios: %s | columnas: %d",
                f"{len(muestra):,}", f"{n_pos_total:,}", muestra.shape[1])
    logger.info("  derivadas añadidas: %s", ", ".join(derived_columns(BASE_VARS)))
    logger.info("  ATENCIÓN: la prevalencia de esta muestra NO es la real (1:%d por diseño). "
                "Sirve para explorar relaciones, no para estimar rendimiento.", args.neg_ratio)


if __name__ == "__main__":
    main()
