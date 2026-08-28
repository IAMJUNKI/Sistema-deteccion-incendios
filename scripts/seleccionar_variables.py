"""Selección de variables en cuatro fases, con la decisión final en manos de quien ejecuta.

El diseño separa deliberadamente tres cosas que suelen ir mezcladas en un «feature selection»:

**Lo que se quita sin preguntar.** Solo entra aquí lo demostrable: constantes, columnas con más
nulos que datos y duplicados con |Spearman| >= 0,99. A esa correlación no hay ningún corte de
árbol que distinga las dos variables, así que conservar ambas no añade información y sí reparte
la importancia entre ellas hasta volver ilegible la tabla. Cada descarte se registra con su
motivo y su evidencia numérica, porque en la defensa hay que poder contestar «¿por qué quitasteis
esta?» con un dato y no con una intuición.

**Lo que se mide pero no se decide.** Señal univariante, familias de variables correlacionadas,
importancia por permutación y su estabilidad. Son diagnósticos: informan, no mandan. En
particular, el AUC univariante es un mal juez para modelos de árboles, porque no ve
interacciones: una variable inútil en solitario puede ser decisiva combinada con otra.

**Lo que se ofrece a elegir.** La curva de compromiso entrena con las `k` mejores variables para
una escalera de valores de `k` y mide el resultado con intervalos de confianza. Convierte
«¿cuántas variables uso?» en una curva que permite afirmar que 20 rinden igual que 54 —o que
no—. La recomendación automática aplica una regla explícita: el conjunto más pequeño cuyo
intervalo se solape con el del mejor. Queda como opción por defecto, no como imposición.

Salidas:
    docs/technical/seleccion_*.csv      diagnósticos y curva
    configs/conjuntos_variables.yaml    conjuntos con nombre, listos para --conjunto

Ejemplos:
    python scripts/seleccionar_variables.py
    python scripts/seleccionar_variables.py --rapido          # curva sobre submuestra
    python scripts/seleccionar_variables.py --sin-curva       # solo diagnóstico, minutos
    python scripts/seleccionar_variables.py --escalones 5 10 20 40 66
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.entrenamiento import (  # noqa: E402
    calibracion, contrato as mod_contrato, datos, derivadas, metricas, modelos, seleccion,
)
from src.entrenamiento.contrato import COL_FECHA, COL_TARGET  # noqa: E402
from src.entrenamiento.experimento import Configuracion, preparar, puntuar  # noqa: E402

DIR_RESULTADOS = RAIZ / "docs" / "technical"
RUTA_CONJUNTOS = RAIZ / "configs" / "conjuntos_variables.yaml"

#: Hiperparámetros de la curva. Modestos a propósito: se comparan conjuntos de variables entre
#: sí, no se busca el mejor modelo posible, y un modelo más ligero hace la escalera asequible.
PARAMS_CURVA = {
    "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "reg_lambda": 10.0, "n_jobs": -1, "verbosity": -1,
}


def titulo(texto: str) -> None:
    print(f"\n{'=' * 86}\n{texto}\n{'=' * 86}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datos", default=str(mod_contrato.DIR_DATASET))
    parser.add_argument("--escalones", nargs="+", type=int,
                        default=[5, 10, 15, 20, 25, 30, 40, 50, 66])
    parser.add_argument("--metrica", default="recall_at_fpr5")
    parser.add_argument("--umbral-duplicado", type=float, default=seleccion.UMBRAL_DUPLICADO)
    parser.add_argument("--umbral-familia", type=float, default=seleccion.UMBRAL_DISCUTIBLE)
    parser.add_argument("--permutacion-filas", type=int, default=1_000_000)
    parser.add_argument("--permutacion-repeticiones", type=int, default=3)
    parser.add_argument("--rapido", action="store_true",
                        help="La curva se mide sobre una submuestra de validación en vez del "
                             "año completo. Es defendible para recall@FPR —que es un cuantil "
                             "de la distribución de negativos y se conserva al submuestrear— "
                             "pero NO para PR-AUC, que depende de la prevalencia.")
    parser.add_argument("--sin-curva", action="store_true",
                        help="Solo poda y diagnóstico. Minutos en vez de una hora.")
    parser.add_argument("--sin-derivadas", action="store_true")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger("seleccion")
    marca = datetime.now().strftime("%Y%m%d_%H%M")
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    contrato = mod_contrato.cargar(args.datos)
    print(contrato.resumen())

    cfg = Configuracion(usar_derivadas=not args.sin_derivadas, semilla=args.semilla)
    prep = preparar(contrato, cfg)
    train, parada = prep.train, prep.parada
    partida = list(prep.variables)

    # ── FASE 0 · Poda automática ─────────────────────────────────────────────────────────
    titulo("FASE 0 · PODA AUTOMÁTICA — solo lo demostrablemente prescindible")
    vivas, descartes = seleccion.poda_automatica(
        train, partida, COL_TARGET, umbral_duplicado=args.umbral_duplicado,
        marco_control=parada,
    )
    if descartes.empty:
        print("  No se descartó ninguna variable.")
    else:
        print(f"  {len(partida)} variables -> {len(vivas)}\n")
        print(descartes.to_string(index=False))
        descartes.to_csv(DIR_RESULTADOS / "seleccion_poda.csv", index=False)

    # ── FASE 1 · Diagnóstico ─────────────────────────────────────────────────────────────
    titulo("FASE 1 · DIAGNÓSTICO — se mide, no se decide")

    senal = seleccion.senal_univariante(train, vivas, COL_TARGET)
    print("\nSeñal univariante (ROC-AUC de cada variable en solitario)")
    print("  Un AUC por debajo de 0,5 ordena al revés y discrimina igual: cuenta la fuerza.")
    print(senal.head(12).round(4).to_string(index=False))
    print("  ...")
    print(senal.tail(6).round(4).to_string(index=False))

    familias = seleccion.clusters_correlacion(train, vivas, umbral=args.umbral_familia)
    print(f"\nFamilias de variables correlacionadas (|Spearman| >= {args.umbral_familia})")
    if familias.empty:
        print("  Ninguna familia por encima del umbral.")
    else:
        print("  Quitar todas menos una de cada familia es una DECISIÓN, no una obviedad:")
        for _, fila in familias.iterrows():
            print(f"    [{fila['n']} vars, r>={fila['correlacion_min']:.3f}] {fila['variables']}")

    log.info("Entrenando el modelo de referencia para la permutación...")
    X_train = modelos.matriz(train, vivas)
    y_train = train[COL_TARGET]
    modelo_ref = modelos.construir("lightgbm", PARAMS_CURVA, args.semilla)
    modelo_ref, _ = modelos.entrenar(
        modelo_ref, "lightgbm", X_train, y_train,
        modelos.matriz(parada, vivas), parada[COL_TARGET], rondas_parada=50,
    )

    muestra = seleccion.submuestra_estratificada(
        parada, COL_TARGET, args.permutacion_filas, args.semilla
    )
    log.info("Permutación sobre %s filas (%s positivos)...",
             f"{len(muestra):,}", f"{int(muestra[COL_TARGET].sum()):,}")
    permutacion = seleccion.importancia_permutacion(
        modelo_ref, modelos.matriz(muestra, vivas).copy(), muestra[COL_TARGET].to_numpy(),
        vivas, repeticiones=args.permutacion_repeticiones, semilla=args.semilla,
    )
    print("\nImportancia por permutación (caída de ROC-AUC al barajar cada variable)")
    print(permutacion.head(15).round(5).to_string(index=False))
    print("  ...")
    print(permutacion.tail(6).round(5).to_string(index=False))

    tabla = seleccion.resumen(senal, permutacion, descartes if not descartes.empty else None)
    tabla.to_csv(DIR_RESULTADOS / "seleccion_diagnostico.csv", index=False)

    orden = permutacion["variable"].tolist()

    # ── FASE 2 · Curva de compromiso ─────────────────────────────────────────────────────
    curva = pd.DataFrame()
    recomendacion: dict = {}
    if not args.sin_curva:
        titulo("FASE 2 · CURVA DE COMPROMISO — qué se gana y qué se pierde con cada tamaño")
        modo = ("submuestra de validación" if args.rapido
                else f"año {list(cfg.anios_validacion)[0]} COMPLETO")
        print(f"  Cada punto es un entrenamiento evaluado sobre {modo}.\n")

        def entrenar_y_evaluar(variables: list[str]) -> dict:
            inicio = time.perf_counter()
            modelo = modelos.construir("lightgbm", PARAMS_CURVA, args.semilla)
            modelo, _ = modelos.entrenar(
                modelo, "lightgbm", modelos.matriz(train, variables), y_train,
                modelos.matriz(parada, variables), parada[COL_TARGET], rondas_parada=50,
            )
            if args.rapido:
                y = muestra[COL_TARGET].to_numpy()
                p = modelo.predict_proba(modelos.matriz(muestra, variables))[:, 1]
                dia = datos.indice_dia(muestra[COL_FECHA])
            else:
                y, p, dia = puntuar(
                    modelo, contrato, cfg.anios_validacion, variables, prep, cfg
                )
            resultado = metricas.evaluate(y, p, None, dia, n_boot=500, seed=args.semilla)
            resultado["segundos"] = round(time.perf_counter() - inicio, 1)
            return resultado

        curva = seleccion.curva_compromiso(
            orden, entrenar_y_evaluar, args.escalones, metrica=args.metrica
        )
        curva.to_csv(DIR_RESULTADOS / "seleccion_curva.csv", index=False)

        print("\n" + _dibujar_curva(curva, args.metrica))

        recomendacion = seleccion.recomendar(curva, args.metrica)
        titulo("FASE 3 · RECOMENDACIÓN — regla explícita, decisión tuya")
        print(f"  Mejor resultado absoluto ....... k = {recomendacion['k_mejor']:>3}  "
              f"({args.metrica} = {recomendacion['metrica_mejor']:.4f})")
        print(f"  Conjunto más pequeño no peor ... k = {recomendacion['k_recomendado']:>3}  "
              f"({args.metrica} = {recomendacion['metrica_recomendada']:.4f})")
        print(f"\n  Coste de la simplificación: {recomendacion['coste_pp']:+.2f} puntos "
              f"a cambio de {recomendacion['variables_ahorradas']} variables menos.")
        if recomendacion["concluyente"]:
            print("  Los intervalos NO se solapan: aquí sí compensa el conjunto grande.")
        else:
            print("  Los intervalos se solapan: la diferencia no es concluyente y, por el "
                  "criterio del proyecto, gana el conjunto más simple.")

    # ── FASE 4 · Conjuntos con nombre ────────────────────────────────────────────────────
    titulo("FASE 4 · CONJUNTOS DISPONIBLES — elígelos con --conjunto")
    conjuntos = _construir_conjuntos(contrato, partida, vivas, orden, curva, recomendacion,
                                     train, senal, args)

    for nombre, variables in conjuntos.items():
        print(f"  {nombre:<24} {len(variables):>3} variables")

    RUTA_CONJUNTOS.write_text(
        yaml.safe_dump(
            {"generado": marca,
             "metrica": args.metrica,
             "recomendado": recomendacion.get("k_recomendado"),
             "conjuntos": {k: sorted(v) for k, v in conjuntos.items()}},
            allow_unicode=True, sort_keys=False, width=100,
        ),
        encoding="utf-8",
    )
    print(f"\n  guardado en {RUTA_CONJUNTOS}")
    print(f"\n  Uso:  python scripts/entrenar_egif.py --conjunto <nombre>")
    return 0


def _dibujar_curva(curva: pd.DataFrame, metrica: str) -> str:
    """Curva de compromiso en texto, con el intervalo de confianza dibujado."""
    lineas = [f"  {'k':>4} {metrica:>12} {'IC90':>18}   {'':<42}",
              f"  {'-' * 4} {'-' * 12} {'-' * 18}   {'-' * 42}"]
    # La escala abarca el punto y su intervalo: si se tomara solo del intervalo, un valor
    # fuera de él saldría del dibujo en vez de avisar.
    bajo = float(min(curva["ic_bajo"].min(), curva[metrica].min()))
    alto = float(max(curva["ic_alto"].max(), curva[metrica].max()))
    rango = max(alto - bajo, 1e-9)
    mejor = float(curva[metrica].max())
    ancho = 40

    def celda(valor: float) -> int:
        return int(np.clip((valor - bajo) / rango * ancho, 0, ancho))

    for _, fila in curva.iterrows():
        ini, fin, punto = celda(fila["ic_bajo"]), celda(fila["ic_alto"]), celda(fila[metrica])
        barra = list(" " * (ancho + 2))
        for i in range(min(ini, fin), max(ini, fin) + 1):
            barra[i] = "─"
        barra[punto] = "●"
        marca = "  <- mejor" if np.isclose(fila[metrica], mejor) else ""
        lineas.append(
            f"  {int(fila['k']):>4} {fila[metrica]:>12.4f} "
            f"{fila['ic_bajo']:>8.4f}–{fila['ic_alto']:<8.4f}   {''.join(barra)}{marca}"
        )
    lineas.append("\n  Donde dos intervalos se solapan, la diferencia no es concluyente.")
    return "\n".join(lineas)


def _construir_conjuntos(contrato, partida, vivas, orden, curva, recomendacion,
                         train, senal, args) -> dict[str, list[str]]:
    """Arma el catálogo de conjuntos con nombre que se ofrecen a quien ejecute."""
    conjuntos: dict[str, list[str]] = {
        "completo": list(partida),
        "tras_poda": list(vivas),
        "solo_dataset": [v for v in partida if v in set(contrato.predictores)],
    }

    pares = seleccion.redundancia(train, vivas, umbral=0.95)
    sin_redundancia, _ = seleccion.podar_redundantes(vivas, pares, senal)
    conjuntos["sin_redundancia_095"] = sin_redundancia

    if not curva.empty:
        for k in curva["k"]:
            conjuntos[f"top{int(k)}"] = orden[:int(k)]
    if recomendacion:
        conjuntos["recomendado"] = recomendacion["variables"]

    # Ablaciones temáticas: para responder «¿y este bloque, para qué está?».
    for grupo in contrato.grupos:
        restantes = [v for v in vivas if v not in set(contrato.grupos[grupo])]
        if restantes and len(restantes) < len(vivas):
            conjuntos[f"sin_{grupo}"] = restantes

    derivadas_generadas = set(derivadas.columnas_derivadas(contrato))
    solo_derivadas = [v for v in vivas if v not in derivadas_generadas]
    if len(solo_derivadas) < len(vivas):
        conjuntos["sin_derivadas"] = solo_derivadas

    return conjuntos


if __name__ == "__main__":
    raise SystemExit(main())
