"""Orquestación de un experimento completo sobre el dataset EGIF.

Este módulo no sabe leer un Parquet ni entrenar nada: llama a los demás en orden y vigila que
no se hagan trampas. Los dos guardianes que implementa son:

- **Cobertura**: se comprueba que la predicción cubrió exactamente las filas que el contrato
  declara para ese año. Está aquí porque ya ocurrió una vez que una ejecución perdió un bloque
  entero de celdas sin lanzar ninguna excepción, y las métricas salieron *mejores* porque
  faltaba una parte difícil del mapa.
- **Test ciego**: 2023 no se toca salvo que se pida explícitamente. Evaluar sobre el test para
  «echar un vistazo» lo quema: a partir de ese momento cualquier decisión posterior está
  contaminada por haberlo visto.

## El protocolo, y en qué se aparta del actual

Se entrena sobre negativos submuestreados y **se evalúa sobre la población completa**. El
pipeline común del equipo submuestrea también la validación, lo que multiplica su prevalencia
por el módulo del muestreo —con módulo 25, la prevalencia de 2022 pasa de 0,0154 % a 0,383 %—.
Para comparar conjuntos de variables entre sí eso da igual, porque el sesgo se cancela; pero
cualquier cifra absoluta medida así describe una población que no existe, y PR-AUC en concreto
depende directamente de la prevalencia. Evaluar sobre el año entero cuesta unos segundos más y
es la diferencia entre un número publicable y uno que no lo es.

La parada temprana sí usa una muestra de validación, no el año completo: elegir el número de
árboles requiere evaluar en cada iteración, y hacerlo sobre diez millones de filas multiplicaría
el tiempo de entrenamiento sin cambiar la decisión.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.entrenamiento import calibracion, datos, derivadas, metricas, modelos
from src.entrenamiento.contrato import COL_FECHA, COL_TARGET, Contrato

logger = logging.getLogger(__name__)


@dataclass
class Configuracion:
    """Parámetros de un experimento. Se rellena desde el YAML o desde un notebook."""

    anios_train: Sequence[int] = (2016, 2017, 2018, 2019, 2020)
    anios_validacion: Sequence[int] = (2022,)
    anios_test: Sequence[int] = (2023,)
    modelos: Sequence[str] = ("lightgbm", "xgboost", "logistic_regression", "random_forest")
    hiperparametros: dict[str, dict] = field(default_factory=dict)
    variables: Optional[Sequence[str]] = None       # None = todas las del contrato
    #: Subconjunto final, aplicado DESPUÉS de generar las derivadas. Es el que rellena
    #: `--conjunto`, porque un conjunto seleccionado puede mezclar columnas del cubo con
    #: derivadas nuestras, y estas no existen hasta que se han calculado.
    variables_finales: Optional[Sequence[str]] = None
    usar_derivadas: bool = True
    modulo_negativos: int = 25
    resto_negativos: int = 0
    rondas_parada: int = 100
    fpr_objetivo: float = 0.05
    top_k_diario: float = 0.01
    n_bootstrap: int = 1000
    semilla: int = 42
    evaluar_test: bool = False
    etiqueta: str = "egif_v1"


@dataclass
class Preparacion:
    """Todo lo que hace falta para entrenar, calculado una sola vez y reutilizado."""

    train: pd.DataFrame
    parada: pd.DataFrame
    variables: list[str]
    contexto: Optional[derivadas.Contexto]
    meta_muestreo: dict


def preparar(contrato: Contrato, cfg: Configuracion) -> Preparacion:
    """Construye el conjunto de entrenamiento una vez para todos los modelos.

    Reconstruirlo por modelo costaba cuatro pasadas sobre decenas de millones de filas para
    obtener exactamente el mismo resultado, y además introducía la posibilidad de que dos
    modelos vieran muestras distintas y la comparación dejara de ser justa.
    """
    base = list(cfg.variables) if cfg.variables else list(contrato.predictores)

    inicio = time.perf_counter()
    train, meta = datos.muestrear_entrenamiento(
        contrato, cfg.anios_train, base,
        modulo=cfg.modulo_negativos, resto=cfg.resto_negativos,
    )
    parada, _ = datos.muestrear_entrenamiento(
        contrato, cfg.anios_validacion, base,
        modulo=cfg.modulo_negativos, resto=cfg.resto_negativos,
    )

    contexto = None
    variables = list(base)
    if cfg.usar_derivadas:
        contexto = derivadas.ajustar_contexto(
            contrato,
            anios_climatologia=cfg.anios_train,
            anios_diarios=list(cfg.anios_train) + list(cfg.anios_validacion),
        )
        train = derivadas.anadir_derivadas(train, contexto, contrato)
        parada = derivadas.anadir_derivadas(parada, contexto, contrato)
        variables += derivadas.columnas_derivadas(contrato)

    faltan = [v for v in variables if v not in train.columns]
    if faltan:
        raise KeyError(f"Variables declaradas que no se generaron: {faltan}")

    if cfg.variables_finales is not None:
        pedidas = list(cfg.variables_finales)
        ausentes = [v for v in pedidas if v not in train.columns]
        if ausentes:
            raise KeyError(
                f"El conjunto pide variables que no existen tras la preparación: {ausentes}. "
                "Si son derivadas, comprueba que usar_derivadas esté activado."
            )
        logger.info("Conjunto aplicado: %s de %s variables", len(pedidas), len(variables))
        variables = pedidas

    logger.info(
        "Preparación lista en %.1f s — %s filas de entrenamiento, %s variables",
        time.perf_counter() - inicio, f"{len(train):,}", len(variables),
    )
    return Preparacion(train, parada, variables, contexto, meta)


def puntuar(
    modelo: Any,
    contrato: Contrato,
    anios: Sequence[int],
    variables: Sequence[str],
    prep: Preparacion,
    cfg: Configuracion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predice sobre la población completa de los años pedidos, lote a lote.

    Returns:
        Tupla (y_true, puntuación cruda, índice de día).
    """
    etiquetas: list[np.ndarray] = []
    puntuaciones: list[np.ndarray] = []
    dias: list[np.ndarray] = []
    vistas = 0

    base = list(cfg.variables) if cfg.variables else list(contrato.predictores)
    for lote in datos.iter_evaluacion(contrato, anios, base):
        vistas += len(lote)
        if prep.contexto is not None:
            lote = derivadas.anadir_derivadas(lote, prep.contexto, contrato)
        X = modelos.matriz(lote, list(variables))
        puntuaciones.append(modelo.predict_proba(X)[:, 1].astype(np.float32))
        etiquetas.append(lote[COL_TARGET].to_numpy(dtype=np.int8))
        dias.append(datos.indice_dia(lote[COL_FECHA]))

    datos.verificar_cobertura(contrato, anios, vistas)
    return (
        np.concatenate(etiquetas),
        np.concatenate(puntuaciones),
        np.concatenate(dias),
    )


def ejecutar(contrato: Contrato, cfg: Configuracion) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Entrena y evalúa todos los modelos de la configuración.

    Returns:
        Tupla con la tabla de resultados y el diccionario de modelos entrenados.
    """
    if cfg.evaluar_test:
        logger.warning(
            "TEST CIEGO ACTIVADO: se va a evaluar sobre %s. A partir de ahora ese año está "
            "quemado y no puede volver a usarse para tomar ninguna decisión.", list(cfg.anios_test)
        )

    prep = preparar(contrato, cfg)
    tasa_negativos = prep.meta_muestreo["tasa_negativos"]

    X_train = modelos.matriz(prep.train, prep.variables)
    y_train = prep.train[COL_TARGET]
    X_parada = modelos.matriz(prep.parada, prep.variables)
    y_parada = prep.parada[COL_TARGET]

    filas: list[dict] = []
    entrenados: dict[str, Any] = {}
    # Se conservan las puntuaciones de validación de cada modelo para poder compararlos entre
    # sí de forma pareada al terminar. Son ~43 MB por modelo y evitan tener que recargar los
    # modelos y volver a predecir sobre diez millones de filas solo para compararlos.
    puntuaciones: dict[str, np.ndarray] = {}
    y_val_comun: Optional[np.ndarray] = None
    dia_val_comun: Optional[np.ndarray] = None

    for nombre in cfg.modelos:
        logger.info("=" * 70)
        logger.info("MODELO %s", nombre.upper())
        inicio = time.perf_counter()

        modelo = modelos.construir(nombre, cfg.hiperparametros.get(nombre, {}), cfg.semilla)
        modelo, mejor_iter = modelos.entrenar(
            modelo, nombre, X_train, y_train, X_parada, y_parada, cfg.rondas_parada
        )
        if mejor_iter is not None:
            logger.info("  parada temprana en la iteración %s", mejor_iter)

        auc_train = metricas.roc_auc_score(y_train, modelo.predict_proba(X_train)[:, 1])
        logger.info("  ROC-AUC en entrenamiento: %.4f", auc_train)

        # ── Validación sobre el año completo, sin submuestrear ────────────────────────────
        logger.info("  puntuando %s completo...", list(cfg.anios_validacion))
        y_val, p_val, dia_val = puntuar(
            modelo, contrato, cfg.anios_validacion, prep.variables, prep, cfg
        )

        puntuaciones[nombre] = p_val
        y_val_comun, dia_val_comun = y_val, dia_val

        calibrador = calibracion.ProbabilityCalibrator(tasa_negativos).fit(
            y_val, p_val, seed=cfg.semilla
        )
        resultado = metricas.evaluate(
            y_val, p_val, calibrador.transform(p_val), dia_val,
            fpr_max=cfg.fpr_objetivo, top_k=cfg.top_k_diario,
            n_boot=cfg.n_bootstrap, seed=cfg.semilla,
        )
        fila = {
            "modelo": nombre,
            "n_variables": len(prep.variables),
            "derivadas": cfg.usar_derivadas,
            "anio_validacion": list(cfg.anios_validacion)[0],
            "train_roc_auc": float(auc_train),
            "mejor_iteracion": mejor_iter,
            "segundos": round(time.perf_counter() - inicio, 1),
            **{f"val_{k}": v for k, v in resultado.items()},
        }
        fila["gap_train_val"] = fila["train_roc_auc"] - fila["val_roc_auc"]

        _registrar(nombre, resultado, prep.variables, modelo)

        if cfg.evaluar_test:
            logger.info("  puntuando el test ciego %s...", list(cfg.anios_test))
            y_test, p_test, dia_test = puntuar(
                modelo, contrato, cfg.anios_test, prep.variables, prep, cfg
            )
            resultado_test = metricas.evaluate(
                y_test, p_test, calibrador.transform(p_test), dia_test,
                fpr_max=cfg.fpr_objetivo, top_k=cfg.top_k_diario,
                n_boot=cfg.n_bootstrap, seed=cfg.semilla,
            )
            fila["anio_test"] = list(cfg.anios_test)[0]
            fila.update({f"test_{k}": v for k, v in resultado_test.items()})

        filas.append(fila)
        entrenados[nombre] = {"modelo": modelo, "calibrador": calibrador}

    tabla = metricas.summarize(filas)

    comparacion = pd.DataFrame()
    if len(puntuaciones) > 1 and y_val_comun is not None:
        logger.info("=" * 70)
        logger.info("COMPARACIÓN PAREADA entre modelos (mismos %s incendios)",
                    f"{int(y_val_comun.sum()):,}")
        comparacion = metricas.comparar_modelos(
            y_val_comun, puntuaciones, dia_val_comun,
            fpr_max=cfg.fpr_objetivo, top_k=cfg.top_k_diario, seed=cfg.semilla,
        )
        for _, fila in comparacion.iterrows():
            logger.info(
                "  [%s] %s (%.2f%%) vs %s (%.2f%%): %+.2f pp "
                "[IC90 %+.2f, %+.2f] McNemar p=%.4g -> %s",
                fila["metrica"], fila["modelo_a"], fila["recall_a"] * 100,
                fila["modelo_b"], fila["recall_b"] * 100, fila["diferencia"] * 100,
                fila["dif_ci90_low"] * 100, fila["dif_ci90_high"] * 100,
                fila["mcnemar_p"], fila["veredicto"],
            )

    return tabla, entrenados, comparacion


def _registrar(nombre: str, resultado: dict, variables: list[str], modelo: Any) -> None:
    """Vuelca al log las cifras que interesa ver mientras corre."""
    clave_recall = next(k for k in resultado if k.startswith("recall_at_fpr"))
    clave_diario = next((k for k in resultado if "daily" in k), None)
    logger.info(
        "  VALIDACIÓN — %s incendios sobre %s filas (prevalencia %.6f%%)",
        f"{resultado['n_positives']:,}", f"{resultado['n_rows']:,}",
        resultado["prevalence"] * 100,
    )
    logger.info("    PR-AUC        %.6e  (lift x%.1f sobre el azar)",
                resultado["pr_auc"], resultado["lift_vs_azar"])
    logger.info("    ROC-AUC       %.4f", resultado["roc_auc"])
    logger.info("    %s  %.2f%%  (IC90 %.2f%% – %.2f%%)  [FPR real %.3f%%]",
                clave_recall, resultado[clave_recall] * 100,
                resultado["recall_ci90_low"] * 100, resultado["recall_ci90_high"] * 100,
                resultado["fpr_achieved"] * 100)
    if clave_diario:
        logger.info("    %s  %.2f%%", clave_diario, resultado[clave_diario] * 100)

    importantes = modelos.importancias(modelo, variables, nombre)
    if not importantes.empty:
        logger.info("    top 8: %s", ", ".join(importantes.head(8)["variable"]))
