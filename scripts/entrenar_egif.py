"""Punto de entrada del pipeline de modelado sobre el datacubo EGIF.

Ejemplos:

    # Inspeccionar el contrato del dataset sin entrenar nada
    python scripts/entrenar_egif.py --solo-contrato

    # Baseline: las 54 variables del dataset, sin derivadas
    python scripts/entrenar_egif.py --sin-derivadas --etiqueta baseline

    # Con las variables derivadas (anomalías y z-scores diarios)
    python scripts/entrenar_egif.py --etiqueta derivadas

    # Un solo modelo, para iterar rápido
    python scripts/entrenar_egif.py --modelos lightgbm

    # Test ciego: solo cuando el modelo esté decidido. Es irreversible.
    python scripts/entrenar_egif.py --evaluar-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import joblib  # noqa: E402
import yaml  # noqa: E402

from src.entrenamiento import contrato as mod_contrato, registro  # noqa: E402
from src.entrenamiento.experimento import Configuracion, ejecutar  # noqa: E402

DIR_RESULTADOS = RAIZ / "docs" / "technical"
#: Detalle por corrida. Queda fuera de Git: se reproduce relanzando el experimento y, si se
#: versiona, cada ejecución añade ficheros nuevos hasta que el historial deja de ser legible.
DIR_DETALLE = DIR_RESULTADOS / "corridas"
#: `data/models/` y no `models/`: es la ruta que el .gitignore del equipo ya cubre
#: (líneas 129-130). Un Random Forest serializado ocupa 70-80 MB y cuatro modelos por
#: ejecución suman cientos de megas; escribirlos fuera de una ruta ignorada los deja a un
#: `git add .` de distancia del repositorio.
DIR_MODELOS = RAIZ / "data" / "models"


def configurar_log(verboso: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verboso else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("py4j").setLevel(logging.WARNING)


RUTA_CONJUNTOS = RAIZ / "configs" / "conjuntos_variables.yaml"


def _conjunto(nombre: str) -> list[str]:
    """Recupera un conjunto con nombre de los que generó `seleccionar_variables.py`."""
    if not RUTA_CONJUNTOS.exists():
        raise SystemExit(
            f"No existe {RUTA_CONJUNTOS}.\n"
            "Genera los conjuntos antes:  python scripts/seleccionar_variables.py"
        )
    catalogo = yaml.safe_load(RUTA_CONJUNTOS.read_text(encoding="utf-8"))["conjuntos"]
    if nombre not in catalogo:
        raise SystemExit(
            f"Conjunto desconocido: {nombre!r}.\nDisponibles: {', '.join(sorted(catalogo))}"
        )
    return list(catalogo[nombre])


def cargar_configuracion(ruta: Path, args: argparse.Namespace) -> Configuracion:
    """Lee el YAML y le aplica encima las opciones de línea de comandos."""
    crudo = yaml.safe_load(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    crudo.pop("seleccion", None)  # Lo consume el notebook de selección, no el entrenamiento.

    cfg = Configuracion(**{k: v for k, v in crudo.items()
                           if k in Configuracion.__dataclass_fields__})

    if args.modelos:
        cfg.modelos = args.modelos
    if args.sin_derivadas:
        cfg.usar_derivadas = False
    if args.conjunto:
        cfg.variables_finales = _conjunto(args.conjunto)
        if not cfg.etiqueta.endswith(args.conjunto):
            cfg.etiqueta = f"{cfg.etiqueta}_{args.conjunto}"
    if args.con_vecindad:
        cfg.usar_vecindad = True
        if not cfg.etiqueta.endswith("vecindad"):
            cfg.etiqueta = f"{cfg.etiqueta}_vecindad"
    if args.resto is not None:
        cfg.resto_negativos = args.resto
    if args.etiqueta:
        cfg.etiqueta = args.etiqueta
    cfg.evaluar_test = args.evaluar_test
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(RAIZ / "configs" / "egif_v1.yaml"))
    parser.add_argument("--datos", default=str(mod_contrato.DIR_DATASET))
    parser.add_argument("--modelos", nargs="+", metavar="MODELO")
    parser.add_argument("--sin-derivadas", action="store_true",
                        help="Entrena solo con las variables del dataset.")
    parser.add_argument("--con-vecindad", action="store_true",
                        help="Añade las variables de contexto espacial (historial de "
                             "igniciones del entorno hasta D-1 y continuidad del "
                             "combustible). La etiqueta se sufija para que la bitácora "
                             "distinga las dos variantes.")
    parser.add_argument("--conjunto", metavar="NOMBRE",
                        help="Conjunto de variables de configs/conjuntos_variables.yaml, "
                             "generado por scripts/seleccionar_variables.py.")
    parser.add_argument("--resto", type=int, metavar="N",
                        help="Residuo del hash de muestreo (0..modulo-1). "
                             "Cambiarlo da una muestra independiente.")
    parser.add_argument("--etiqueta", help="Sufijo de los ficheros de salida.")
    parser.add_argument("--evaluar-test", action="store_true",
                        help="Evalúa sobre el test ciego. IRREVERSIBLE: a partir de ahí ese "
                             "año no puede sustentar ninguna decisión.")
    parser.add_argument("--solo-contrato", action="store_true",
                        help="Muestra el esquema del dataset y termina.")
    parser.add_argument("-v", "--verboso", action="store_true")
    args = parser.parse_args()

    configurar_log(args.verboso)
    log = logging.getLogger("entrenar_egif")

    try:
        contrato = mod_contrato.cargar(args.datos)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    print(contrato.resumen())
    if args.solo_contrato:
        for grupo in sorted(contrato.grupos):
            print(f"\n{grupo.upper()}")
            for variable in contrato.grupos[grupo]:
                print(f"    {variable}")
        return 0

    cfg = cargar_configuracion(Path(args.config), args)

    faltan = [a for a in list(cfg.anios_train) + list(cfg.anios_validacion)
              if a not in contrato.filas_por_anio]
    if faltan:
        log.error("El dataset no contiene los años %s. Disponibles: %s", faltan, contrato.anios)
        return 1

    log.info("Entrenamiento %s | validación %s | derivadas: %s",
             list(cfg.anios_train), list(cfg.anios_validacion), cfg.usar_derivadas)

    tabla, entrenados, comparacion = ejecutar(contrato, cfg)

    marca = datetime.now().strftime("%Y%m%d_%H%M")
    DIR_DETALLE.mkdir(parents=True, exist_ok=True)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    # Bitácora acumulativa: nombre estable, va a Git, una fila por experimento y modelo.
    registro.registrar(tabla, cfg.etiqueta, {
        "dataset": Path(args.datos).name,
        "n_predictores_dataset": len(contrato.predictores),
        "derivadas": cfg.usar_derivadas,
        "conjunto": args.conjunto or "(todas)",
        "anios_train": "-".join(str(a) for a in cfg.anios_train),
        "anio_validacion": list(cfg.anios_validacion)[0],
        "modulo_negativos": cfg.modulo_negativos,
        "resto_negativos": cfg.resto_negativos,
        "test_evaluado": cfg.evaluar_test,
    })

    # Detalle de esta corrida: reproducible relanzando el experimento, así que no va a Git.
    detalle = DIR_DETALLE / f"{cfg.etiqueta}_{marca}.csv"
    tabla.to_csv(detalle, index=False)

    csv_comparacion = None
    if not comparacion.empty:
        csv_comparacion = DIR_RESULTADOS / f"comparacion_pareada_{cfg.etiqueta}.csv"
        comparacion.to_csv(csv_comparacion, index=False)

    meta = {
        "etiqueta": cfg.etiqueta,
        "generado": marca,
        "commit": registro.commit_actual(),
        "dataset": str(args.datos),
        "anios_train": list(cfg.anios_train),
        "anios_validacion": list(cfg.anios_validacion),
        "test_evaluado": cfg.evaluar_test,
        "usar_derivadas": cfg.usar_derivadas,
        "conjunto": args.conjunto,
        "modulo_negativos": cfg.modulo_negativos,
        "resto_negativos": cfg.resto_negativos,
        "contrato_temporal": contrato.contrato_temporal,
        "n_predictores_dataset": len(contrato.predictores),
    }
    (DIR_DETALLE / f"{cfg.etiqueta}_{marca}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for nombre, artefactos in entrenados.items():
        destino = DIR_MODELOS / f"{cfg.etiqueta}_{nombre}_{marca}.joblib"
        joblib.dump(artefactos, destino)

    print("\n" + "=" * 78)
    print("RESUMEN — validación")
    print("=" * 78)
    columnas = [c for c in ("modelo", "val_roc_auc", "val_pr_auc", "val_recall_at_fpr5",
                            "val_recall_ci90_low", "val_recall_ci90_high",
                            "val_recall_at_top1%_daily", "gap_train_val", "segundos")
                if c in tabla.columns]
    print(tabla[columnas].round(4).to_string(index=False))

    if not comparacion.empty:
        print("\n" + "=" * 78)
        print("COMPARACIÓN PAREADA — mismos incendios para todos los modelos")
        print("=" * 78)
        print("  El intervalo es el de la DIFERENCIA. Si no contiene el cero, es concluyente.")
        for metrica, bloque in comparacion.groupby("metrica", sort=False):
            print(f"\n  {metrica}")
            for _, f in bloque.iterrows():
                print(f"    {f['modelo_a']:<20} {f['recall_a'] * 100:>6.2f}%  vs  "
                      f"{f['modelo_b']:<20} {f['recall_b'] * 100:>6.2f}%   "
                      f"{f['diferencia'] * 100:>+6.2f} pp "
                      f"[{f['dif_ci90_low'] * 100:>+6.2f}, {f['dif_ci90_high'] * 100:>+6.2f}]  "
                      f"p={f['mcnemar_p']:<9.3g} {f['veredicto']}")

    print(f"\nbitacora:   {registro.BITACORA}")
    print(f"detalle:    {detalle}")
    if csv_comparacion is not None:
        print(f"pareado:    {csv_comparacion}")
    print(f"modelos:    {DIR_MODELOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
