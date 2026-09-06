"""Mide si las variables de contexto espacial mejoran el modelo.

Compara el mismo modelo con y sin las variables de vecindad, entrenado sobre las mismas filas
y evaluado sobre la población completa del año de validación. El veredicto lo da el test
pareado, no la diferencia de cifras: con 1.659 igniciones dos puntos pueden ser ruido.

El interés está en el recall **dentro del día**. La debilidad medida del sistema es que acierta
el día pero no la celda, y estas variables atacan exactamente eso, así que una mejora del recall
global sin mejora intradía sería una señal de que están funcionando por otro motivo.

Uso:
    python archive/vecindad/evaluar_vecindad.py
    python archive/vecindad/evaluar_vecindad.py --radios 3 10 --ventanas 7 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# parents[2] y no [1]: este script vive en archive/vecindad/, dos niveles por debajo de la raíz.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as pads  # noqa: E402
import yaml  # noqa: E402

from src.entrenamiento import (  # noqa: E402
    calibracion, contrato as mod_contrato, datos, derivadas, metricas, modelos, vecindad,
)
from src.entrenamiento.contrato import COL_FECHA, COL_TARGET  # noqa: E402
from src.entrenamiento.experimento import Configuracion  # noqa: E402

#: Los resultados se escriben junto a este script, no en docs/technical/, que es donde vive la
#: bitácora de experimentos adoptados. Esto no lo está.
DIR_RESULTADOS = Path(__file__).resolve().parent
GEOMETRIA = ["x", "y", COL_FECHA, COL_TARGET, "broadleaf_forest", "coniferous_forest",
             "mixed_forest"]


def titulo(texto: str) -> None:
    print(f"\n{'=' * 88}\n{texto}\n{'=' * 88}")


def marco_para_contexto(contrato, anios: list[int]) -> pd.DataFrame:
    """Reúne lo mínimo necesario para construir el contexto espacial.

    No hace falta leer los 43 millones de filas: el historial solo necesita las igniciones, y
    la geometría de la rejilla se obtiene de un único día. Se añaden el primer y el último día
    completos para que el rango temporal quede bien delimitado.
    """
    rutas = [str(p) for p in contrato.rutas(anios)]
    conjunto = pads.dataset(rutas, format="parquet")

    positivos = conjunto.to_table(
        columns=GEOMETRIA, filter=(pads.field(COL_TARGET) == 1)
    ).to_pandas()

    bordes = []
    for anio, extremo in ((min(anios), "min"), (max(anios), "max")):
        uno = pads.dataset([str(contrato.ruta(anio))], format="parquet")
        fechas = uno.to_table(columns=[COL_FECHA]).to_pandas()[COL_FECHA]
        limite = fechas.min() if extremo == "min" else fechas.max()
        bordes.append(uno.to_table(columns=GEOMETRIA,
                                   filter=(pads.field(COL_FECHA) == limite)).to_pandas())

    # Sin deduplicar, una ignición que caiga en uno de los dos días de borde aparecería dos
    # veces —una por el filtro de positivos y otra por el día completo— y el historial la
    # contaría doble. Con los datos actuales son 2 de 5.659, pero el recuento tiene que ser
    # exacto o las variables dejan de significar lo que dicen.
    marco = pd.concat([positivos, *bordes], ignore_index=True).drop_duplicates(
        subset=["x", "y", COL_FECHA], keep="first", ignore_index=True)

    logger = logging.getLogger("vecindad")
    logger.info("Contexto a partir de %s filas únicas (%s igniciones y 2 días completos)",
                f"{len(marco):,}", f"{int(marco[COL_TARGET].sum()):,}")
    return marco


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default=str(mod_contrato.DIR_DATASET))
    parser.add_argument("--config", default=str(RAIZ / "configs" / "egif_v1.yaml"))
    parser.add_argument("--radios", nargs="+", type=int, default=list(vecindad.RADIOS))
    parser.add_argument("--ventanas", nargs="+", type=int, default=list(vecindad.VENTANAS))
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger("vecindad")

    contrato = mod_contrato.cargar(args.datos)
    print(contrato.resumen())

    crudo = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    crudo.pop("seleccion", None)
    cfg = Configuracion(**{k: v for k, v in crudo.items()
                           if k in Configuracion.__dataclass_fields__})
    anios = list(cfg.anios_train) + list(cfg.anios_validacion)

    # ── Contexto espacial ─────────────────────────────────────────────────────────────
    titulo(f"CONTEXTO ESPACIAL · radios {args.radios} km · ventanas {args.ventanas} días")
    inicio = time.perf_counter()
    contexto = vecindad.ajustar_contexto_espacial(
        marco_para_contexto(contrato, anios), radios=args.radios, ventanas=args.ventanas
    )
    print(f"  {contexto.resumen()}   ({time.perf_counter() - inicio:.0f} s)")

    # ── Preparación, común a los dos modelos ──────────────────────────────────────────
    titulo("PREPARACIÓN")
    # `x` e `y` viajan como columnas_extra, no como predictoras: sirven para situar la fila
    # en la rejilla. Darlas al modelo como variables sería otra intervención distinta —y peor,
    # porque memorizar coordenadas no generaliza a un año nuevo.
    base = list(contrato.predictores)
    train, meta = datos.muestrear_entrenamiento(
        contrato, cfg.anios_train, base, columnas_extra=("x", "y"),
        modulo=cfg.modulo_negativos, resto=cfg.resto_negativos,
    )
    parada, _ = datos.muestrear_entrenamiento(
        contrato, cfg.anios_validacion, base, columnas_extra=("x", "y"),
        modulo=cfg.modulo_negativos, resto=cfg.resto_negativos,
    )

    contexto_derivadas = None
    variables_base = list(base)
    if cfg.usar_derivadas:
        contexto_derivadas = derivadas.ajustar_contexto(contrato, cfg.anios_train, anios)
        train = derivadas.anadir_derivadas(train, contexto_derivadas, contrato)
        parada = derivadas.anadir_derivadas(parada, contexto_derivadas, contrato)
        variables_base += derivadas.columnas_derivadas(contrato)

    train = vecindad.anadir_vecindad(train, contexto)
    parada = vecindad.anadir_vecindad(parada, contexto)
    nuevas = [c for c in vecindad.columnas_vecindad(args.radios, args.ventanas)
              if c in train.columns]

    print(f"  {len(train):,} filas · {len(variables_base)} variables base · "
          f"{len(nuevas)} de vecindad")
    print(f"\n  Variables nuevas: {', '.join(nuevas)}")

    print("\n  Media en las filas con ignición frente al resto:\n")
    for c in nuevas:
        con = train.loc[train[COL_TARGET] == 1, c].mean()
        sin = train.loc[train[COL_TARGET] == 0, c].mean()
        print(f"    {c:<32} {con:>9.3f}  frente a {sin:>9.3f}   "
              f"({con / sin if sin else float('nan'):.2f}x)")

    # ── Los dos modelos ───────────────────────────────────────────────────────────────
    resultados, puntuaciones = {}, {}
    y_val = dia_val = None

    for etiqueta, variables in (("sin vecindad", variables_base),
                                ("con vecindad", variables_base + nuevas)):
        titulo(f"{etiqueta.upper()} · {len(variables)} variables")
        reloj = time.perf_counter()

        modelo = modelos.construir("lightgbm", cfg.hiperparametros.get("lightgbm", {}),
                                   cfg.semilla)
        modelo, iteracion = modelos.entrenar(
            modelo, "lightgbm", modelos.matriz(train, variables), train[COL_TARGET],
            modelos.matriz(parada, variables), parada[COL_TARGET], cfg.rondas_parada,
        )
        log.info("  entrenado en %.0f s (parada en %s)", time.perf_counter() - reloj,
                 iteracion)

        y_val, p_val, dia_val = _puntuar_con_vecindad(
            modelo, contrato, cfg, variables, contexto_derivadas, contexto, base
        )
        puntuaciones[etiqueta] = p_val

        calibrador = calibracion.ProbabilityCalibrator(meta["tasa_negativos"]).fit(
            y_val, p_val, seed=cfg.semilla)
        r = metricas.evaluate(y_val, p_val, calibrador.transform(p_val), dia_val,
                              fpr_max=cfg.fpr_objetivo, top_k=cfg.top_k_diario,
                              n_boot=cfg.n_bootstrap, seed=cfg.semilla)
        # La métrica que estas variables existen para mover: aísla la señal espacial
        # eliminando por completo la contribución del calendario.
        r["roc_auc_dentro_del_dia"] = roc_dentro_del_dia(y_val, p_val, dia_val)
        resultados[etiqueta] = r

        n = int(y_val.sum())
        print(f"  recall@5%          {r['recall_at_fpr5']:.4f}  "
              f"[{r['recall_ci90_low']:.4f}, {r['recall_ci90_high']:.4f}]  "
              f"→ {round(r['recall_at_fpr5'] * n):,} de {n:,} igniciones")
        print(f"  top 1 %/día        {r['recall_at_top1%_daily']:.4f}")
        print(f"  ROC-AUC global     {r['roc_auc']:.4f}")
        print(f"  ROC-AUC intradía   {r['roc_auc_dentro_del_dia']:.4f}   <- el objetivo")

        importancias = modelos.importancias(modelo, variables, "lightgbm")
        if not importancias.empty and etiqueta == "con vecindad":
            print("\n  Posición de las variables nuevas por importancia:")
            for c in nuevas:
                pos = importancias.index[importancias.variable == c]
                if len(pos):
                    print(f"    {int(pos[0]) + 1:>3} de {len(variables)}   {c}")

    # ── Veredicto ─────────────────────────────────────────────────────────────────────
    titulo("VEREDICTO · comparación pareada sobre las mismas igniciones")
    comparacion = metricas.comparar_modelos(
        y_val, puntuaciones, dia_val, fpr_max=cfg.fpr_objetivo,
        top_k=cfg.top_k_diario, seed=cfg.semilla,
    )
    n = int(y_val.sum())
    for _, f in comparacion.iterrows():
        print(f"\n  [{f['metrica']}]")
        print(f"    {f['modelo_a']:<14} {f['recall_a'] * 100:>6.2f} %   "
              f"({round(f['recall_a'] * n):,} igniciones)")
        print(f"    {f['modelo_b']:<14} {f['recall_b'] * 100:>6.2f} %   "
              f"({round(f['recall_b'] * n):,} igniciones)")
        # `comparar_modelos` devuelve a - b. Se imprime b - a y se dice de quién, porque un
        # signo negativo junto a un veredicto de «mejor» se lee al revés de lo que dice.
        print(f"    {f['modelo_b']} - {f['modelo_a']}:  "
              f"{-f['diferencia'] * 100:>+6.2f} pp   "
              f"IC90 [{-f['dif_ci90_high'] * 100:+.2f}, {-f['dif_ci90_low'] * 100:+.2f}]   "
              f"McNemar p = {f['mcnemar_p']:.4g}")
        print(f"    -> {f['veredicto'].upper()}")

    tabla = pd.DataFrame([{"variante": k, "n_variables": len(variables_base) + (
        len(nuevas) if k == "con vecindad" else 0), **v} for k, v in resultados.items()])
    tabla.to_csv(DIR_RESULTADOS / "resultados.csv", index=False)
    comparacion.to_csv(DIR_RESULTADOS / "pareado.csv", index=False)
    print(f"\n  {DIR_RESULTADOS / 'resultados.csv'}")
    return 0


def roc_dentro_del_dia(y: np.ndarray, p: np.ndarray, dia: np.ndarray) -> float:
    """ROC-AUC calculado día a día y promediado: aísla la señal espacial.

    Responde a la pregunta operativa —dado que hoy es 12 de agosto, ¿acierta *qué celdas*?—
    eliminando por completo la contribución del calendario. Es la definición que usa
    `scripts/pipeline_definitivo.py`, para que las cifras sean comparables.
    """
    from sklearn.metrics import roc_auc_score

    orden = np.argsort(dia, kind="stable")
    y, p, dia = y[orden], p[orden], dia[orden]
    cortes = np.flatnonzero(np.diff(dia)) + 1
    valores = [roc_auc_score(yy, pp)
               for yy, pp in zip(np.split(y, cortes), np.split(p, cortes))
               if yy.min() != yy.max()]
    return float(np.mean(valores)) if valores else float("nan")


def _puntuar_con_vecindad(modelo, contrato, cfg, variables, contexto_derivadas, contexto,
                          base):
    """Puntúa la población completa añadiendo derivadas y vecindad lote a lote."""
    etiquetas, puntuaciones, dias, vistas = [], [], [], 0
    for lote in datos.iter_evaluacion(contrato, cfg.anios_validacion, base,
                                      columnas_extra=("x", "y")):
        vistas += len(lote)
        if contexto_derivadas is not None:
            lote = derivadas.anadir_derivadas(lote, contexto_derivadas, contrato)
        lote = vecindad.anadir_vecindad(lote, contexto)
        puntuaciones.append(
            modelo.predict_proba(modelos.matriz(lote, variables))[:, 1].astype(np.float32))
        etiquetas.append(lote[COL_TARGET].to_numpy(dtype=np.int8))
        dias.append(datos.indice_dia(lote[COL_FECHA]))
    datos.verificar_cobertura(contrato, cfg.anios_validacion, vistas)
    return np.concatenate(etiquetas), np.concatenate(puntuaciones), np.concatenate(dias)


if __name__ == "__main__":
    raise SystemExit(main())
