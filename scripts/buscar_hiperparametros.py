"""Búsqueda de hiperparámetros con validación cruzada temporal de ventana expansiva.

El ganador de la búsqueda **no se acepta por serlo**: se reentrena con el protocolo normal, se
evalúa sobre la población completa del año de validación y se compara con la configuración
actual mediante el test pareado. Si la mejora no supera ese test, se rechaza.

Ese último paso es el que evita el sesgo del ganador. Con 1.659 incendios el ruido de la métrica
ronda los dos puntos, así que probar cuarenta configuraciones y quedarse con la mejor encuentra
algo por encima del óptimo real sin que el modelo sea mejor. La única defensa es confirmarlo
contra el baseline con un test que sepa distinguir una diferencia real de la suerte.

Ejemplos:
    python scripts/buscar_hiperparametros.py --modelos lightgbm
    python scripts/buscar_hiperparametros.py --modelos lightgbm xgboost --n-configuraciones 60
    python scripts/buscar_hiperparametros.py --modelos lightgbm --n-configuraciones 8 --rapido
"""

from __future__ import annotations

import argparse
import json
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
    busqueda, calibracion, contrato as mod_contrato, datos, derivadas, metricas, modelos,
)
from src.entrenamiento.contrato import COL_FECHA, COL_TARGET  # noqa: E402
from src.entrenamiento.experimento import Configuracion, preparar, puntuar  # noqa: E402

DIR_RESULTADOS = RAIZ / "docs" / "technical"
RUTA_GANADORES = RAIZ / "configs" / "hiperparametros_encontrados.yaml"


def titulo(texto: str) -> None:
    print(f"\n{'=' * 86}\n{texto}\n{'=' * 86}")


def preparar_ventana(contrato, cfg: Configuracion, anios: list[int]):
    """Carga y prepara todos los años de la ventana en un solo marco, con su columna de año."""
    base = list(cfg.variables) if cfg.variables else list(contrato.predictores)
    marco, meta = datos.muestrear_entrenamiento(
        contrato, anios, base, modulo=cfg.modulo_negativos, resto=cfg.resto_negativos
    )
    variables = list(base)
    contexto = None
    if cfg.usar_derivadas:
        # La climatología se ajusta solo con los años de entrenamiento del protocolo, no con
        # toda la ventana: definir «lo normal» con años que después se validan sería fuga.
        contexto = derivadas.ajustar_contexto(contrato, cfg.anios_train, anios)
        marco = derivadas.anadir_derivadas(marco, contexto, contrato)
        variables += derivadas.columnas_derivadas(contrato)

    marco["__anio"] = pd.to_datetime(marco[COL_FECHA]).dt.year.to_numpy()
    return marco, variables, contexto, meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datos", default=str(mod_contrato.DIR_DATASET))
    parser.add_argument("--modelos", nargs="+", default=["lightgbm"])
    parser.add_argument("--n-configuraciones", type=int, default=40)
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument("--config", default=str(RAIZ / "configs" / "egif_v1.yaml"))
    parser.add_argument("--sin-derivadas", action="store_true")
    parser.add_argument("--rapido", action="store_true",
                        help="Omite la confirmación sobre la población completa. Para "
                             "comprobar que la maquinaria funciona, no para decidir nada.")
    parser.add_argument("--tam-bloque", type=int, default=5,
                        help="Configuraciones por bloque. El avance se guarda tras cada uno, "
                             "asi que una interrupcion cuesta como mucho un bloque.")
    parser.add_argument("--desde-cero", action="store_true",
                        help="Ignora el avance guardado y repite la busqueda entera.")
    parser.add_argument("--por-media", action="store_true",
                        help="Elige por media entre pliegues en vez de penalizar la "
                             "dispersión. Más expuesto al sesgo del ganador.")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger("busqueda")
    marca = datetime.now().strftime("%Y%m%d_%H%M")
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    contrato = mod_contrato.cargar(args.datos)
    print(contrato.resumen())

    crudo = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    crudo.pop("seleccion", None)
    cfg = Configuracion(**{k: v for k, v in crudo.items()
                           if k in Configuracion.__dataclass_fields__})
    if args.sin_derivadas:
        cfg.usar_derivadas = False

    ventana = list(cfg.anios_train) + list(cfg.anios_validacion)
    titulo(f"PREPARACIÓN — ventana {ventana}")
    marco, variables, contexto, meta = preparar_ventana(contrato, cfg, ventana)
    X, y = marco[variables], marco[COL_TARGET]
    print(f"  {len(marco):,} filas, {len(variables)} variables, "
          f"{int(y.sum()):,} igniciones")

    particionador = busqueda.VentanaExpansiva(marco["__anio"].to_numpy(), target=y.to_numpy())
    print(f"\n  Pliegues ({particionador.get_n_splits()}):")
    print(particionador.describir())

    resumen: dict[str, dict] = {}
    ganadores: dict[str, dict] = {}

    for nombre in args.modelos:
        titulo(f"BÚSQUEDA · {nombre.upper()} · {args.n_configuraciones} configuraciones")
        inicio = time.perf_counter()
        tabla = busqueda.buscar(
            nombre, X, y, particionador,
            n_configuraciones=args.n_configuraciones,
            fpr_max=args.fpr, semilla=args.semilla,
            # El checkpoint es el propio fichero de resultados: si la ejecución se corta, al
            # relanzarla con los mismos parámetros continúa donde se quedó.
            ruta_checkpoint=DIR_RESULTADOS / f"busqueda_{nombre}.csv",
            reanudar=not args.desde_cero,
            tam_bloque=args.tam_bloque,
        )
        minutos = (time.perf_counter() - inicio) / 60
        # Reescritura final, ya ordenada de mejor a peor. Durante la búsqueda el fichero
        # existía como checkpoint sin ordenar.
        tabla.to_csv(DIR_RESULTADOS / f"busqueda_{nombre}.csv", index=False)

        columnas = ["configuracion", "recall_medio", "recall_desv", "recall_minimo"]
        columnas += [c for c in tabla.columns if c.startswith("pliegue_")]
        print(f"\n  Mejores 8 de {len(tabla)} (en {minutos:.1f} min):\n")
        print(tabla[columnas].head(8).round(4).to_string(index=False))

        peor, mejor = tabla["recall_medio"].min(), tabla["recall_medio"].max()
        print(f"\n  Dispersión: de {peor:.4f} a {mejor:.4f} "
              f"({(mejor - peor) * 100:.2f} puntos entre la mejor y la peor)")

        parametros = busqueda.elegir(
            tabla, busqueda.ESPACIOS[nombre], penalizar_inestabilidad=not args.por_media
        )
        criterio = "mayor media" if args.por_media else "media menos desviación"
        print(f"\n  Elegida por {criterio}:")
        for clave, valor in parametros.items():
            print(f"    {clave:<22} {valor}")

        ganadores[nombre] = parametros
        resumen[nombre] = {
            "n_configuraciones": len(tabla),
            "minutos": round(minutos, 1),
            "recall_cv_mejor": float(mejor),
            "recall_cv_peor": float(peor),
            "parametros": parametros,
        }

        if not args.rapido:
            resumen[nombre]["confirmacion"] = _confirmar(
                contrato, cfg, nombre, parametros, variables, contexto, marco, meta, log
            )

    # Se fusiona con lo que ya hubiera: lanzar la búsqueda de un solo modelo no puede borrar
    # los hiperparámetros encontrados para los demás en ejecuciones anteriores.
    acumulados: dict[str, dict] = {}
    if RUTA_GANADORES.exists():
        try:
            previo = yaml.safe_load(RUTA_GANADORES.read_text(encoding="utf-8")) or {}
            acumulados = dict(previo.get("hiperparametros") or {})
        except yaml.YAMLError:
            log.warning("No se pudo leer %s; se reescribe desde cero.", RUTA_GANADORES)
    conservados = [m for m in acumulados if m not in ganadores]
    if conservados:
        log.info("Se conservan los hiperparámetros previos de: %s", ", ".join(conservados))
    acumulados.update(ganadores)

    RUTA_GANADORES.write_text(
        yaml.safe_dump({"generado": marca, "fpr": args.fpr,
                        "n_configuraciones": args.n_configuraciones,
                        "modelos_de_esta_ejecucion": list(ganadores),
                        "hiperparametros": acumulados},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (DIR_RESULTADOS / "busqueda_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    titulo("VEREDICTO")
    for nombre, datos_modelo in resumen.items():
        c = datos_modelo.get("confirmacion")
        if not c:
            print(f"  {nombre}: sin confirmar (--rapido)")
            continue
        print(f"\n  {nombre}")
        print(f"    baseline actual .... {c['recall_baseline'] * 100:>6.2f} %  "
              f"({c['incendios_baseline']:,} de {c['n_incendios']:,} igniciones)")
        print(f"    tras la busqueda ... {c['recall_nuevo'] * 100:>6.2f} %  "
              f"({c['incendios_nuevo']:,} igniciones)")
        print(f"    diferencia ......... {c['diferencia'] * 100:>+6.2f} puntos  "
              f"({c['incendios_nuevo'] - c['incendios_baseline']:+,} igniciones)")
        print(f"    IC90 de la dif ..... [{c['ic_bajo'] * 100:+.2f}, {c['ic_alto'] * 100:+.2f}]"
              f"   McNemar p = {c['mcnemar_p']:.4g}")
        print(f"    -> {c['veredicto'].upper()}")

    print(f"\n  hiperparametros: {RUTA_GANADORES}")
    print(f"  tablas:          {DIR_RESULTADOS}/busqueda_*.csv")
    return 0


def _confirmar(contrato, cfg, nombre, parametros, variables, contexto, marco, meta, log) -> dict:
    """Reentrena baseline y ganador y los compara sobre la población completa, en pareado."""
    from src.entrenamiento.experimento import Preparacion

    titulo(f"CONFIRMACIÓN · {nombre.upper()} · población completa de "
           f"{list(cfg.anios_validacion)[0]}")

    entrena = marco[marco["__anio"].isin(cfg.anios_train)]
    parada = marco[marco["__anio"].isin(cfg.anios_validacion)]
    prep = Preparacion(entrena, parada, variables, contexto, meta)

    X_tr, y_tr = modelos.matriz(entrena, variables), entrena[COL_TARGET]
    X_pa, y_pa = modelos.matriz(parada, variables), parada[COL_TARGET]

    puntuaciones: dict[str, np.ndarray] = {}
    y_val = dia_val = None

    candidatos = {
        "baseline": cfg.hiperparametros.get(nombre, {}),
        "buscado": {**busqueda.FIJOS.get(nombre, {}), **parametros},
    }
    for etiqueta, params in candidatos.items():
        log.info("Entrenando %s (%s)...", nombre, etiqueta)
        modelo = modelos.construir(nombre, params, cfg.semilla)
        # El baseline conserva su parada temprana; el buscado lleva `n_estimators` fijado por
        # la búsqueda, así que se entrena tal cual y la comparación es entre configuraciones
        # completas, cada una con su propia forma de decidir el número de árboles.
        rondas = cfg.rondas_parada if etiqueta == "baseline" else None
        modelo, _ = modelos.entrenar(modelo, nombre, X_tr, y_tr, X_pa, y_pa, rondas)
        y_val, p_val, dia_val = puntuar(
            modelo, contrato, cfg.anios_validacion, variables, prep, cfg
        )
        puntuaciones[etiqueta] = p_val
        resultado = metricas.evaluate(y_val, p_val, None, dia_val,
                                      fpr_max=cfg.fpr_objetivo, n_boot=500, seed=cfg.semilla)
        log.info("  %s: recall %.4f, top1%%/dia %.4f", etiqueta,
                 resultado[f"recall_at_fpr{int(cfg.fpr_objetivo * 100)}"],
                 resultado.get("recall_at_top1%_daily", float("nan")))

    aciertos = {e: metricas.aciertos_at_fpr(y_val, p, cfg.fpr_objetivo)
                for e, p in puntuaciones.items()}
    comparacion = metricas.comparar_pareado(
        aciertos["buscado"], aciertos["baseline"], "buscado", "baseline", seed=cfg.semilla
    )
    n = int(y_val.sum())
    return {
        "n_incendios": n,
        "recall_baseline": comparacion["recall_b"],
        "recall_nuevo": comparacion["recall_a"],
        "incendios_baseline": round(comparacion["recall_b"] * n),
        "incendios_nuevo": round(comparacion["recall_a"] * n),
        "diferencia": comparacion["diferencia"],
        "ic_bajo": comparacion["dif_ci90_low"],
        "ic_alto": comparacion["dif_ci90_high"],
        "mcnemar_p": comparacion["mcnemar_p"],
        "veredicto": ("mejora confirmada" if comparacion["concluyente"]
                      and comparacion["diferencia"] > 0
                      else "empeora" if comparacion["concluyente"]
                      else "no concluyente: se mantiene el baseline"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
