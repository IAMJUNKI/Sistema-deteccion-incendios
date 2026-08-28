"""Carga de datos: muestreo para entrenar, población completa para evaluar.

La distinción entre esas dos cosas es el núcleo de este módulo, y es deliberada.

**Al entrenar** se submuestrean los negativos. Con una prevalencia de 1,2·10⁻⁴ un modelo que
diga siempre «no» acierta el 99,99 % de las veces, así que hay que reequilibrar para que el
gradiente aprenda algo.

**Al evaluar, no.** Se predice sobre el año entero, con su prevalencia real intacta. Evaluar
sobre negativos submuestreados infla la prevalencia por el mismo factor del muestreo, y como
PR-AUC depende de la prevalencia, cualquier cifra medida así es incomparable con el despliegue
real. Cuesta unos segundos más por año y es la diferencia entre un número publicable y uno que
no lo es.

El muestreo usa el mismo hash celda-día que `src/modeling/data.py` para que las filas
seleccionadas coincidan exactamente con las del resto del equipo: así, cualquier diferencia de
resultados es atribuible al modelo o al protocolo, nunca a que cada uno mire filas distintas.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from src.entrenamiento.contrato import COL_CELDA, COL_FECHA, COL_TARGET, Contrato

logger = logging.getLogger(__name__)

#: Primo del hash celda-día. Debe coincidir con `src/modeling/data.py` del pipeline común.
PRIMO_HASH = 1_000_003

#: Filas por lote al recorrer un año completo. 500 000 filas × 60 columnas float32 son ~120 MB.
FILAS_POR_LOTE = 500_000


def _numero_de_dia(fechas: pd.Series) -> np.ndarray:
    """Convierte fechas a días enteros desde época, que es el índice que usan hash y métricas."""
    return pd.to_datetime(fechas).astype("int64").to_numpy() // 86_400_000_000_000


def _columnas(contrato: Contrato, predictores: Optional[Sequence[str]]) -> list[str]:
    """Resuelve qué columnas hay que leer, validando que existan en el contrato."""
    if predictores is None:
        return list(contrato.predictores)
    disponibles = set(contrato.predictores)
    faltan = [c for c in predictores if c not in disponibles]
    if faltan:
        raise ValueError(f"Variables pedidas que no están en el dataset: {faltan}")
    return list(predictores)


def muestrear_entrenamiento(
    contrato: Contrato,
    anios: Sequence[int],
    predictores: Optional[Sequence[str]] = None,
    modulo: int = 25,
    resto: int = 0,
    columnas_extra: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict]:
    """Carga todos los positivos y una fracción `1/modulo` de los negativos.

    Args:
        contrato: Contrato del dataset.
        anios: Años de entrenamiento.
        predictores: Subconjunto de variables. `None` usa todas las del contrato.
        modulo: Se conserva un negativo de cada `modulo`.
        resto: Residuo del hash. Cambiarlo da una muestra distinta e independiente, que es lo
            que permite comprobar si un resultado se sostiene o depende de la muestra.
        columnas_extra: Columnas adicionales a arrastrar (por ejemplo `burned_area_ha` para
            análisis posteriores). No entran como predictoras.

    Returns:
        Tupla con el DataFrame muestreado y un diccionario de metadatos del muestreo, del que
        `tasa_negativos` es el que necesita la calibración.
    """
    if modulo < 1:
        raise ValueError("modulo debe ser mayor que cero.")
    if not 0 <= resto < modulo:
        raise ValueError(f"resto debe estar en [0, {modulo}).")

    predictores = _columnas(contrato, predictores)
    columnas = [COL_FECHA, COL_CELDA, COL_TARGET, *predictores, *columnas_extra]
    columnas = list(dict.fromkeys(columnas))  # sin duplicados, conservando orden

    dataset = pads.dataset([str(p) for p in contrato.rutas(anios)], format="parquet")
    trozos: list[pd.DataFrame] = []
    vistas = 0
    positivos = 0

    for lote in dataset.scanner(columns=columnas, batch_size=100_000).to_batches():
        marco = lote.to_pandas()
        vistas += len(marco)
        objetivo = marco[COL_TARGET].to_numpy()
        positivos += int(objetivo.sum())

        hash_fila = (
            marco[COL_CELDA].to_numpy(dtype=np.int64) * PRIMO_HASH + _numero_de_dia(marco[COL_FECHA])
        ) % modulo
        conservar = (objetivo == 1) | (hash_fila == resto)
        if conservar.any():
            trozos.append(marco.loc[conservar])

    if not trozos:
        raise ValueError(f"El muestreo sobre {list(anios)} no produjo ninguna fila.")

    muestra = pd.concat(trozos, ignore_index=True)
    negativos_muestra = int((muestra[COL_TARGET] == 0).sum())
    negativos_totales = vistas - positivos

    meta = {
        "anios": list(anios),
        "modulo": modulo,
        "resto": resto,
        "filas_totales": vistas,
        "filas_muestra": len(muestra),
        "positivos": positivos,
        "negativos_muestra": negativos_muestra,
        "negativos_totales": negativos_totales,
        # Fracción de negativos conservada. Es el parámetro exacto que deshace el sesgo de
        # prevalencia en `calibracion.prior_correction`.
        "tasa_negativos": negativos_muestra / negativos_totales if negativos_totales else 1.0,
        "prevalencia_real": positivos / vistas if vistas else 0.0,
        "prevalencia_muestra": positivos / len(muestra),
    }

    logger.info(
        "Entrenamiento %s: %s filas (%s positivos, %s negativos de %s) — "
        "prevalencia real %.6f%%, en la muestra %.4f%%",
        list(anios), f"{len(muestra):,}", f"{positivos:,}", f"{negativos_muestra:,}",
        f"{negativos_totales:,}", meta["prevalencia_real"] * 100, meta["prevalencia_muestra"] * 100,
    )
    return muestra, meta


def iter_evaluacion(
    contrato: Contrato,
    anios: Sequence[int],
    predictores: Optional[Sequence[str]] = None,
    filas_por_lote: int = FILAS_POR_LOTE,
) -> Iterator[pd.DataFrame]:
    """Recorre la población completa de los años pedidos, en lotes acotados en memoria.

    No submuestrea nada: es el recorrido sobre el que se calculan las métricas publicables.
    """
    predictores = _columnas(contrato, predictores)
    columnas = list(dict.fromkeys([COL_FECHA, COL_CELDA, COL_TARGET, *predictores]))
    dataset = pads.dataset([str(p) for p in contrato.rutas(anios)], format="parquet")

    for lote in dataset.scanner(columns=columnas, batch_size=filas_por_lote).to_batches():
        yield lote.to_pandas()


def indice_dia(fechas: pd.Series) -> np.ndarray:
    """Índice entero de día, tal como lo espera `metricas.recall_at_top_k_daily`."""
    return _numero_de_dia(fechas)


def verificar_cobertura(contrato: Contrato, anios: Sequence[int], filas_vistas: int) -> None:
    """Comprueba que la predicción cubrió el año entero.

    Existe porque ya pasó: una ejecución perdió un bloque completo de celdas sin lanzar ninguna
    excepción, y las métricas salieron *mejores* porque faltaba una parte difícil del mapa. Un
    fallo silencioso que mejora los resultados es el peor tipo de fallo posible en un TFM.
    """
    esperadas = contrato.filas(anios)
    if filas_vistas != esperadas:
        raise RuntimeError(
            f"Cobertura incompleta al evaluar {list(anios)}: se predijo sobre "
            f"{filas_vistas:,} filas y el contrato declara {esperadas:,} "
            f"(faltan {esperadas - filas_vistas:,}). No se publican métricas parciales."
        )
    logger.info("Cobertura verificada: %s filas, las %s declaradas.",
                f"{filas_vistas:,}", f"{esperadas:,}")
