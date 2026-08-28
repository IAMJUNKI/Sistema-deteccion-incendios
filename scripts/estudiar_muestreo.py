"""Barrido de estrategias de submuestreo de negativos por ratio.

Responde a dos preguntas que el proyecto arrastra sin respuesta escrita:

1. **¿Por qué 1:25?** El módulo se heredó del pipeline común sin justificación, y el borrador de
   la memoria describe otra cosa —minado de negativos duros a 1:50—. Nunca se midió.
2. **¿Importa *cuáles* negativos se conservan, o solo cuántos?**

El barrido es bidimensional a propósito. Comparar estrategias a ratio fijo respondería a la
segunda pregunta ignorando que la respuesta probablemente **depende** de la primera: con 1:25 se
conservan más de un millón de negativos, un presupuesto tan holgado que el muestreo aleatorio ya
cubre bien el espacio de variables. La hipótesis es que el muestreo por diversidad solo compensa
cuando el presupuesto aprieta, y eso solo se ve mirando las dos dimensiones a la vez.

Todas las combinaciones se evalúan sobre **la población completa** del año de validación, así que
las cifras son directamente comparables entre sí y con el resto del proyecto.

Ejemplos:
    python scripts/estudiar_muestreo.py
    python scripts/estudiar_muestreo.py --ratios 25 200 --estrategias uniforme clusters
    python scripts/estudiar_muestreo.py --incluir-duros    # anade el control negativo
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.entrenamiento import (  # noqa: E402
    calibracion, contrato as mod_contrato, datos, derivadas, metricas, modelos, muestreo,
)
from src.entrenamiento.contrato import COL_CELDA, COL_FECHA, COL_TARGET  # noqa: E402
from src.entrenamiento.experimento import Configuracion, Preparacion, puntuar  # noqa: E402

DIR_RESULTADOS = RAIZ / "docs" / "technical"

#: Modelo ligero y fijo para todo el barrido: se comparan estrategias de muestreo, no modelos.
#: Si cada celda de la rejilla usara hiperparámetros distintos no sabríamos a qué atribuir las
#: diferencias.
PARAMS = {
    "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "reg_lambda": 10.0, "n_jobs": -1, "verbosity": -1,
    "importance_type": "gain",
}


def titulo(texto: str) -> None:
    print(f"\n{'=' * 88}\n{texto}\n{'=' * 88}")


def clase_combustible(marco: pd.DataFrame) -> pd.Series:
    """Etiqueta de combustible dominante, para estratificar.

    El cubo trae fracciones continuas de cobertura y no una clase. Para estratificar hace falta
    una variable categórica, así que se toma la fracción dominante de cada celda.
    """
    candidatas = [c for c in ("broadleaf_forest", "coniferous_forest", "mixed_forest",
                              "scrub", "agriculture", "artificial", "open_spaces")
                  if c in marco.columns]
    if not candidatas:
        raise ValueError("No hay fracciones de cobertura con las que estratificar.")
    return pd.Series(marco[candidatas].to_numpy().argmax(axis=1), index=marco.index).map(
        dict(enumerate(candidatas))
    )


def construir_muestra(estrategia: str, marco: pd.DataFrame, variables: list[str],
                      ratio: int, semilla: int) -> muestreo.Muestra:
    """Aplica la estrategia pedida sobre el marco completo de entrenamiento."""
    objetivo = marco[COL_TARGET].to_numpy()
    celda = marco[COL_CELDA].to_numpy()
    dia = datos.indice_dia(marco[COL_FECHA])

    # `independiente=True` es imprescindible: el embalse ya seleccionó con el hash primario, y
    # volver a usarlo aquí no compone las tasas —comparten divisores— sino que produce un ratio
    # efectivo distinto del anunciado. Con el segundo juego de primos las dos etapas sí se
    # multiplican.
    if estrategia == "uniforme":
        return muestreo.uniforme(objetivo, celda, dia, modulo=ratio, independiente=True)

    if estrategia == "estratificado":
        claves = pd.DataFrame({
            "mes": pd.to_datetime(marco[COL_FECHA]).dt.month.to_numpy(),
            "combustible": clase_combustible(marco).to_numpy(),
        })
        return muestreo.estratificado(objetivo, celda, dia, claves, modulo=ratio,
                                      independiente=True)

    if estrategia.startswith("clusters"):
        # Se agrupa sobre un subconjunto meteorológico y estructural: usar las 61 columnas
        # haría los centroides dominados por las que más varían, no por las que más importan.
        base = [c for c in ("temperature_max", "relative_humidity_min", "precipitation_sum_7d",
                            "vpd_mean", "elevation_mean", "road_length_km")
                if c in marco.columns]
        asignacion = "equilibrado" if estrategia.endswith("equilibrado") else "proporcional"
        return muestreo.por_clusters(objetivo, marco[base], modulo=ratio, semilla=semilla,
                                     asignacion=asignacion)

    if estrategia == "duros":
        return muestreo.duros(objetivo, celda, dia,
                              muestreo.dificultad_climatica(marco), modulo=ratio)

    raise ValueError(f"Estrategia desconocida: {estrategia!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datos", default=str(mod_contrato.DIR_DATASET))
    parser.add_argument("--config", default=str(RAIZ / "configs" / "egif_v1.yaml"))
    parser.add_argument("--ratios", nargs="+", type=int, default=[10, 25, 100, 200],
                        help="Ratios EFECTIVOS sobre la poblacion real. Deben ser multiplos "
                             "de --modulo-embalse.")
    parser.add_argument("--modulo-embalse", type=int, default=5,
                        help="La poblacion de entrenamiento completa son 32 millones de filas: "
                             "cargarla entera pasaria de 20 GB. Se carga un embalse uniforme "
                             "de 1 de cada N y las estrategias eligen dentro de el. Como el "
                             "embalse es insesgado, la tasa efectiva es el producto de las dos "
                             "y la correccion de prior sigue siendo exacta.")
    parser.add_argument("--estrategias", nargs="+",
                        default=["uniforme", "estratificado",
                                 "clusters_proporcional", "clusters_equilibrado"],
                        help="clusters_proporcional reduce varianza sin cambiar la "
                             "composicion; clusters_equilibrado sobrerrepresenta a proposito "
                             "las regiones raras del espacio de variables.")
    parser.add_argument("--incluir-duros", action="store_true",
                        help="Anade el control negativo. Empeora el modelo: esta para "
                             "documentar por que se descarto, no para competir.")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger("muestreo")
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    estrategias = list(args.estrategias) + (["duros"] if args.incluir_duros else [])

    malos = [r for r in args.ratios if r % args.modulo_embalse or r < args.modulo_embalse]
    if malos:
        log.error("Ratios %s no son múltiplos de --modulo-embalse=%s.",
                  malos, args.modulo_embalse)
        return 1

    contrato = mod_contrato.cargar(args.datos)
    print(contrato.resumen())

    crudo = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    crudo.pop("seleccion", None)
    cfg = Configuracion(**{k: v for k, v in crudo.items()
                           if k in Configuracion.__dataclass_fields__})

    # ── Embalse: una muestra uniforme de la que beben todas las estrategias ─────────────
    # Se carga una sola vez. Al ser uniforme e insesgado, las estrategias pueden repartir
    # dentro de él y la tasa efectiva de cada fila es el producto de las dos tasas.
    titulo(f"EMBALSE 1:{args.modulo_embalse} SOBRE {list(cfg.anios_train)}")
    inicio = time.perf_counter()
    base = list(contrato.predictores)
    completo, meta_embalse = datos.muestrear_entrenamiento(
        contrato, cfg.anios_train, base, modulo=args.modulo_embalse,
    )
    tasa_embalse = meta_embalse["tasa_negativos"]
    print(f"  {len(completo):,} filas, {int(completo[COL_TARGET].sum()):,} igniciones "
          f"({time.perf_counter() - inicio:.0f} s)")
    print(f"  tasa del embalse: {tasa_embalse:.5f}  "
          f"(de {meta_embalse['negativos_totales']:,} negativos reales)")

    contexto = None
    variables = list(base)
    if cfg.usar_derivadas:
        contexto = derivadas.ajustar_contexto(
            contrato, cfg.anios_train,
            list(cfg.anios_train) + list(cfg.anios_validacion),
        )
        completo = derivadas.anadir_derivadas(completo, contexto, contrato)
        variables += derivadas.columnas_derivadas(contrato)

    filas: list[dict] = []
    destino = DIR_RESULTADOS / "estudio_muestreo.csv"
    for ratio in args.ratios:
        for estrategia in estrategias:
            titulo(f"1:{ratio} · {estrategia.upper()}")
            reloj = time.perf_counter()

            interno = ratio // args.modulo_embalse
            m = construir_muestra(estrategia, completo, variables, interno, args.semilla)
            entrena = m.aplicar(completo)
            # Tasa efectiva sobre la población real: la del embalse por la de la estrategia.
            # Los positivos se conservan siempre, así que su tasa sigue siendo 1.
            es_pos = entrena[COL_TARGET] == 1
            entrena.loc[~es_pos, "tasa_muestreo"] *= tasa_embalse
            negativos = int((~es_pos).sum())
            print(f"  1:{ratio} = embalse 1:{args.modulo_embalse} x interno 1:{interno}")
            print(f"  {len(entrena):,} filas · {negativos:,} negativos · "
                  f"{m.meta['n_estratos']} estratos · "
                  f"tasa efectiva media {entrena.loc[~es_pos, 'tasa_muestreo'].mean():.6f}")

            modelo = modelos.construir("lightgbm", PARAMS, args.semilla)
            modelo, _ = modelos.entrenar(
                modelo, "lightgbm", modelos.matriz(entrena, variables), entrena[COL_TARGET]
            )
            segundos_entreno = time.perf_counter() - reloj

            prep = Preparacion(entrena, entrena, variables, contexto, {})
            y_val, p_val, dia_val = puntuar(
                modelo, contrato, cfg.anios_validacion, variables, prep, cfg
            )

            # Calibración con las tasas del propio muestreo. En las estrategias no uniformes
            # la tasa varía por fila, y usar una global descalibraría en silencio.
            tasa_negativos = entrena.loc[entrena[COL_TARGET] == 0, "tasa_muestreo"]
            calibrador = calibracion.ProbabilityCalibrator(
                float(tasa_negativos.mean())
            ).fit(y_val, p_val, seed=args.semilla)

            resultado = metricas.evaluate(
                y_val, p_val, calibrador.transform(p_val), dia_val,
                fpr_max=cfg.fpr_objetivo, top_k=cfg.top_k_diario,
                n_boot=cfg.n_bootstrap, seed=args.semilla,
            )
            clave_recall = f"recall_at_fpr{int(cfg.fpr_objetivo * 100)}"
            n_pos = int(y_val.sum())

            filas.append({
                "ratio": ratio, "estrategia": estrategia,
                "filas_entreno": len(entrena), "negativos": negativos,
                "n_estratos": m.meta["n_estratos"],
                "roc_auc": resultado["roc_auc"],
                "recall": resultado[clave_recall],
                "recall_ci_bajo": resultado["recall_ci90_low"],
                "recall_ci_alto": resultado["recall_ci90_high"],
                "incendios": round(resultado[clave_recall] * n_pos),
                "top1_diario": resultado.get("recall_at_top1%_daily", np.nan),
                "brier": resultado["brier_score"],
                "segundos": round(segundos_entreno, 1),
            })
            print(f"  recall {resultado[clave_recall]:.4f} "
                  f"[{resultado['recall_ci90_low']:.4f}–{resultado['recall_ci90_high']:.4f}] · "
                  f"{filas[-1]['incendios']:,} de {n_pos:,} incendios · "
                  f"top1%/dia {filas[-1]['top1_diario']:.4f} · {segundos_entreno:.0f} s")

            # Se guarda tras cada combinación: son dieciséis entrenamientos y casi una hora,
            # y una interrupción no debe costar más que la combinación en curso. Escritura
            # atómica para que un corte a mitad no deje un CSV truncado.
            temporal = destino.with_suffix(".parcial")
            pd.DataFrame(filas).to_csv(temporal, index=False)
            temporal.replace(destino)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(destino, index=False)

    titulo("RESULTADOS")
    print(tabla[["ratio", "estrategia", "negativos", "recall", "recall_ci_bajo",
                 "recall_ci_alto", "incendios", "top1_diario", "segundos"]]
          .round(4).to_string(index=False))

    titulo("RECALL POR RATIO Y ESTRATEGIA")
    print(tabla.pivot(index="ratio", columns="estrategia", values="recall").round(4).to_string())
    print("\nIncendios detectados de los que hay en el año de validación:")
    print(tabla.pivot(index="ratio", columns="estrategia", values="incendios").to_string())

    print(f"\n  {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
