"""Entrenamiento, calibración y evaluación de los modelos de riesgo de ignición.

Ejecuta la matriz experimental del TFM: cada modelo se entrena dos veces, una con la
meteorología del propio día del incendio (``nowcast``) y otra con la del día anterior (``t1``).
La diferencia entre ambas mide el coste real de anticipar 24 horas, que es la pregunta central
del trabajo.

A diferencia del pipeline anterior del repositorio, aquí **entrenar y predecir están separados**:
este script produce artefactos serializados (`data/models/*.joblib`) que el pipeline de
inferencia carga, en lugar de reentrenar el modelo en cada ejecución.

Uso:
    python -m scripts.train_model --config configs/model_v1.yaml
    python -m scripts.train_model --config configs/model_v1.yaml --models lightgbm --modes t1
    python -m scripts.train_model --config configs/model_v1.yaml --protocol expanding

El conjunto de test permanece ciego salvo que se pase ``--evaluate-test``.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.derived import fit_context
from src.models.calibration import ProbabilityCalibrator, risk_levels
from src.models.dataset import (
    DATE_REAL_COL,
    PARQUET_COLS,
    TARGET_COL,
    build_training_set,
    expected_rows,
    feature_columns,
    iter_blocks,
    load_eval_year,
    resolve_data_root,
)
from sklearn.metrics import roc_auc_score

from src.models.estimators import build_model, feature_importances, fit_model, prepare_features
from src.models.evaluate import evaluate, reliability_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_model")

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(ruta: Path) -> dict:
    """Carga el fichero YAML de configuración."""
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_years(
    model: Any,
    years: list[int],
    mode: str,
    feature_cols: list[str],
    model_name: str,
    cfg: dict,
    data_root: Optional[Path],
    derived: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predice sobre uno o varios años completos, bloque a bloque.

    La evaluación nunca se submuestrea: hacerlo cambiaría la prevalencia y las métricas dejarían
    de ser comparables entre experimentos.

    Args:
        model: Modelo entrenado.
        years: Años a predecir.
        mode: Variante del dataset.
        feature_cols: Variables explicativas.
        model_name: Nombre del modelo (decide el preprocesado).
        cfg: Configuración completa.
        data_root: Directorio de los parquets.

    Returns:
        Tupla (y_true, y_prob, day_index), donde `day_index` numera los días de forma correlativa.
    """
    y_true, y_prob, dias = [], [], []
    esperadas = 0
    for year in years:
        # Las filas de margen del año anterior se descartan dentro de `iter_blocks`, así que lo
        # recorrido debe coincidir exactamente con el tamaño del fichero de ese año.
        esperadas += expected_rows(year, data_root)
        t0, filas = time.time(), 0
        for bloque in load_eval_year(
            year,
            mode,
            block_size=cfg["data"]["block_size"],
            data_root=data_root,
            cyclic=cfg["data"]["cyclic_features"],
            derived=derived,
            columns=feature_cols,
        ):
            X = prepare_features(bloque, feature_cols, model_name)
            y_true.append(bloque[TARGET_COL].to_numpy(dtype=np.int8))
            y_prob.append(model.predict_proba(X)[:, 1].astype(np.float32))
            dias.append(bloque[DATE_REAL_COL].values.astype("datetime64[D]").astype(np.int32))
            filas += len(bloque)
        # La inferencia sobre un año completo es la fase más costosa del pipeline (11,2 M de
        # filas). Se registra el ritmo para que una espera larga no parezca un cuelgue.
        seg = time.time() - t0
        logger.info(
            "    %d: %s filas en %.0f s (%.0f k filas/s)",
            year, f"{filas:,}", seg, filas / max(seg, 1e-9) / 1000,
        )

    y_true = np.concatenate(y_true)

    # Comprobación de cobertura. Se observó la pérdida transitoria de un bloque entero (4.000
    # celdas, 46 incendios) durante una ejecución, sin que se lanzase ninguna excepción: las
    # métricas salieron calculadas sobre una población incompleta y con aspecto perfectamente
    # normal, y solo se detectó al comparar el recuento de filas entre modelos. Evaluar de menos
    # en silencio es peor que fallar, así que aquí se falla.
    if len(y_true) != esperadas:
        raise RuntimeError(
            f"Cobertura incompleta al predecir {years} (modo {mode}): se recorrieron "
            f"{len(y_true):,} filas de {esperadas:,} esperadas — faltan "
            f"{esperadas - len(y_true):,}. Las métricas serían inválidas. Relanza la ejecución."
        )

    return y_true, np.concatenate(y_prob), np.concatenate(dias)


def run_one(
    model_name: str,
    mode: str,
    cfg: dict,
    train: pd.DataFrame,
    meta: dict,
    train_years: list[int],
    val_years: list[int],
    test_years: Optional[list[int]],
    data_root: Optional[Path],
    outdir_models: Path,
    early_stop_set: Optional[pd.DataFrame] = None,
    derived_ctx: Optional[dict] = None,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """Entrena, calibra y evalúa una combinación (modelo, variante del dataset).

    Args:
        model_name: Modelo a entrenar.
        mode: ``"nowcast"`` o ``"t1"``.
        cfg: Configuración.
        train: Conjunto de entrenamiento ya muestreado. Se construye una sola vez por variante
            en `main`, porque depende del modo y no del modelo.
        meta: Metadatos del muestreo, con la tasa de conservación de negativos.
        train_years: Años de entrenamiento.
        val_years: Años de validación (calibración y umbral).
        test_years: Años de test ciego, o `None` para no tocarlos.
        data_root: Directorio de los parquets.
        outdir_models: Carpeta donde serializar el artefacto.

    Returns:
        Diccionario con las métricas de validación y, si procede, de test.
    """
    seed = cfg["seed"]
    feature_cols = feature_cols or feature_columns(
        cyclic=cfg["data"]["cyclic_features"], derived=derived_ctx is not None
    )
    m_cfg = cfg["metrics"]
    t0 = time.time()

    logger.info("=" * 78)
    logger.info("MODELO %s | VARIANTE %s", model_name.upper(), mode.upper())
    logger.info("=" * 78)

    # ── 1. Entrenamiento ──
    logger.info("Entrenando %s sobre %s filas...", model_name, f"{len(train):,}")
    model = build_model(model_name, cfg["models"][model_name]["params"], feature_cols, seed)
    X_train = prepare_features(train, feature_cols, model_name)

    X_es = y_es = None
    if early_stop_set is not None:
        X_es = prepare_features(early_stop_set, feature_cols, model_name)
        y_es = early_stop_set[TARGET_COL]

    model, mejor_iter = fit_model(
        model,
        model_name,
        X_train,
        train[TARGET_COL],
        X_eval=X_es,
        y_eval=y_es,
        early_stopping_rounds=cfg["models"][model_name].get("early_stopping_rounds"),
    )
    if mejor_iter is not None:
        logger.info(
            "  parada temprana en la iteración %s de %s",
            mejor_iter, cfg["models"][model_name]["params"].get("n_estimators"),
        )

    # Rendimiento sobre el propio entrenamiento. Es la medida directa del sobreajuste: el
    # ROC-AUC es invariante a la prevalencia, así que la distancia entre este valor y el de
    # validación es comparable pese a que ambos conjuntos tengan proporciones distintas.
    roc_train = float(roc_auc_score(train[TARGET_COL], model.predict_proba(X_train)[:, 1]))
    logger.info("  ROC-AUC en entrenamiento: %.4f", roc_train)

    # ── 2. Validación y calibración ──
    logger.info("Prediciendo sobre validación %s (año completo)...", val_years)
    y_val, p_val_raw, dias_val = predict_years(
        model, val_years, mode, feature_cols, model_name, cfg, data_root, derived_ctx
    )

    calibrador = None
    p_val = p_val_raw
    if cfg["calibration"]["enabled"]:
        logger.info("Calibrando (corrección de prior + isotónica sobre validación)...")
        calibrador = ProbabilityCalibrator(
            neg_sampling_rate=meta["neg_sampling_rate"],
            max_fit_points=cfg["calibration"]["max_fit_points"],
        ).fit(y_val, p_val_raw, seed=seed)
        p_val = calibrador.transform(p_val_raw)

    # Las métricas de ordenación se calculan sobre la puntuación cruda y las de calibración
    # sobre la probabilidad calibrada (ver `evaluate`).
    metricas_val = evaluate(
        y_val,
        p_val_raw,
        y_prob_calibrated=p_val if calibrador else None,
        day_index=dias_val,
        fpr_max=m_cfg["fpr_max"],
        top_k=m_cfg["top_k_daily"],
        n_boot=m_cfg["n_bootstrap"],
        seed=seed,
    )
    _log_metrics("VALIDACIÓN", val_years, metricas_val, m_cfg)

    _, umbrales_riesgo = risk_levels(p_val, tuple(cfg["calibration"]["risk_quantiles"]))

    # ── 3. Test ciego (solo si se pide explícitamente) ──
    metricas_test = None
    if test_years:
        logger.info("Prediciendo sobre TEST CIEGO %s...", test_years)
        y_test, p_test_raw, dias_test = predict_years(
            model, test_years, mode, feature_cols, model_name, cfg, data_root, derived_ctx
        )
        p_test = calibrador.transform(p_test_raw) if calibrador else None
        metricas_test = evaluate(
            y_test,
            p_test_raw,
            y_prob_calibrated=p_test,
            day_index=dias_test,
            fpr_max=m_cfg["fpr_max"],
            top_k=m_cfg["top_k_daily"],
            n_boot=m_cfg["n_bootstrap"],
            seed=seed,
        )
        _log_metrics("TEST CIEGO", test_years, metricas_test, m_cfg)

    # ── 4. Serialización ──
    outdir_models.mkdir(parents=True, exist_ok=True)
    # La etiqueta incluye los años porque un mismo modelo se reentrena con distintos cortes
    # temporales (protocolo fijo y cada pliegue de la ventana expansiva). Sin los años en el
    # nombre, los pliegues se sobrescriben entre sí y el artefacto que queda en disco no
    # corresponde necesariamente al experimento que se está reportando.
    etiqueta = (
        f"{cfg['version']}_{model_name}_{mode}"
        f"_tr{train_years[0]}-{train_years[-1]}_val{val_years[0]}"
    )
    artefacto = {
        "model": model,
        "calibrator": calibrador,
        "feature_cols": feature_cols,
        "mode": mode,
        "model_name": model_name,
        "sampling_meta": meta,
        "risk_thresholds": umbrales_riesgo,
        "threshold_at_fpr": metricas_val["threshold_at_fpr"],
        "config": cfg,
        "train_years": train_years,
        "val_years": val_years,
    }
    ruta_modelo = outdir_models / f"{etiqueta}.joblib"
    joblib.dump(artefacto, ruta_modelo, compress=3)
    logger.info("Artefacto guardado en %s", ruta_modelo)

    imp = feature_importances(model, feature_cols, model_name)
    if not imp.empty:
        imp.to_csv(outdir_models / f"{etiqueta}_importancias.csv", index=False)
        logger.info("Top 8 variables: %s", ", ".join(imp.variable.head(8)))

    fila = {"modelo": model_name, "variante": mode, "segundos": round(time.time() - t0, 1)}
    fila["train_roc_auc"] = roc_train
    fila["gap_sobreajuste"] = roc_train - metricas_val["roc_auc"]
    fila["best_iteration"] = mejor_iter if mejor_iter is not None else ""
    fila.update({f"val_{k}": v for k, v in metricas_val.items()})
    if metricas_test:
        fila.update({f"test_{k}": v for k, v in metricas_test.items()})
    return fila


def _log_metrics(titulo: str, years: list[int], m: dict, m_cfg: dict) -> None:
    """Imprime el bloque de métricas de forma legible."""
    clave_recall = f"recall_at_fpr{int(m_cfg['fpr_max'] * 100)}"
    clave_top = f"recall_at_top{m_cfg['top_k_daily']:.0%}_daily"
    logger.info("  %s %s — %s incendios sobre %s filas",
                titulo, years, f"{m['n_positives']:,}", f"{m['n_rows']:,}")
    logger.info("    PR-AUC          %.6e   (lift x%.1f sobre el azar)", m["pr_auc"], m["lift_vs_azar"])
    logger.info("    ROC-AUC         %.4f", m["roc_auc"])
    logger.info("    Recall@FPR%.0f%%    %.2f%%   (IC90 %.2f%% – %.2f%%)  [FPR real %.3f%%]",
                m_cfg["fpr_max"] * 100, m[clave_recall] * 100,
                m["recall_ci90_low"] * 100, m["recall_ci90_high"] * 100,
                m["fpr_achieved"] * 100)
    if clave_top in m:
        logger.info("    Recall@top%.0f%%/día %.2f%%", m_cfg["top_k_daily"] * 100, m[clave_top] * 100)
    if not np.isnan(m["brier_score"]):
        logger.info("    Brier           %.3e", m["brier_score"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa los modelos de riesgo de ignición forestal."
    )
    parser.add_argument("--config", type=str, default="configs/model_v1.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subconjunto de modelos a entrenar.")
    parser.add_argument("--modes", nargs="+", default=None, choices=["nowcast", "t1"],
                        help="Variantes del dataset a entrenar.")
    parser.add_argument("--protocol", choices=["fixed", "expanding"], default="fixed",
                        help="fixed: split 2019-21/2022/2023-24. expanding: ventana expansiva.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evalúa sobre el test ciego. Solo con el modelo congelado.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--hard-fraction", type=float, default=None,
                        help="Sobrescribe sampling.hard_fraction. 0 = submuestreo aleatorio "
                             "puro; ~0.18 reproduce la proporción natural de días de riesgo.")
    parser.add_argument("--neg-ratio", type=int, default=None,
                        help="Sobrescribe sampling.neg_ratio.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Sufijo para el CSV de resultados, para no pisar barridos previos.")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config if not Path(args.config).is_absolute()
                      else Path(args.config))

    data_root = Path(args.data_root) if args.data_root else (
        Path(cfg["data"]["root"]) if cfg["data"].get("root") else None
    )
    data_root = data_root or resolve_data_root()
    logger.info("Datos: %s", data_root)

    # Sobrescrituras de línea de comandos, para barrer parámetros de muestreo sin duplicar
    # ficheros de configuración. Quedan registradas en el CSV de resultados.
    if args.hard_fraction is not None:
        cfg["sampling"]["hard_fraction"] = args.hard_fraction
        logger.info("Sobrescrito hard_fraction = %s", args.hard_fraction)
    if args.neg_ratio is not None:
        cfg["sampling"]["neg_ratio"] = args.neg_ratio
        logger.info("Sobrescrito neg_ratio = %s", args.neg_ratio)

    modelos = args.models or [n for n, c in cfg["models"].items() if c.get("enabled")]
    modos = args.modes or cfg["data"]["modes"]

    # Lista de variables explicativas. Si el config trae una lista explícita, manda esa; es el
    # resultado del análisis de notebooks/02_feature_engineering.ipynb.
    f_cfg = cfg.get("features", {}) or {}
    feature_cols_cfg = feature_columns(
        cyclic=cfg["data"]["cyclic_features"],
        derived=bool(f_cfg.get("derived")),
        explicit=f_cfg.get("columns"),
    )
    logger.info("Variables explicativas: %d%s", len(feature_cols_cfg),
                " (lista explícita del config)" if f_cfg.get("columns") else "")

    outdir_models = REPO_ROOT / cfg["output"]["models_dir"]
    outdir_res = REPO_ROOT / cfg["output"]["results_dir"]
    outdir_res.mkdir(parents=True, exist_ok=True)

    # Construye la lista de (train, val, test) según el protocolo elegido.
    if args.protocol == "fixed":
        tandas = [(cfg["split"]["train"], cfg["split"]["validation"],
                   cfg["split"]["test"] if args.evaluate_test else None)]
    else:
        ew = cfg["expanding_window"]
        # La ventana expansiva evalúa sobre años que en el protocolo fijo son test ciego. No
        # puede lanzarse por descuido: exige el mismo consentimiento explícito que el test.
        if not args.evaluate_test:
            logger.error(
                "--protocol expanding evalúa sobre %s, que incluye los años de test ciego %s. "
                "Relánzalo con --evaluate-test si es lo que quieres. El gesto debe ser "
                "deliberado: una vez mirado, el test deja de ser ciego.",
                ew["test_years"], cfg["split"]["test"],
            )
            sys.exit(2)
        tandas = []
        for año_test in ew["test_years"]:
            previos = list(range(ew["first_train_year"], año_test))
            # El año inmediatamente anterior hace de validación; el resto, de entrenamiento.
            # Con menos de dos años previos no hay forma de separar ambos papeles, y reutilizar
            # el mismo año para entrenar y calibrar invalidaría la calibración.
            if len(previos) < 2:
                logger.warning(
                    "Se omite el test %d: solo hay %d año(s) previo(s), insuficientes para "
                    "separar entrenamiento y validación.", año_test, len(previos),
                )
                continue
            tandas.append((previos[:-1], previos[-1:], [año_test]))

    filas = []
    for train_years, val_years, test_years in tandas:
        for mode in modos:
            # Contexto de las variables derivadas: climatología por celda (aprendida solo con
            # los años de entrenamiento) y estadísticos espaciales diarios. Se ajusta antes que
            # nada porque todo lo demás lo necesita, y se cachea en disco porque exige dos
            # recorridos completos del histórico.
            derived_ctx = None
            if cfg.get("features", {}).get("derived"):
                años_eval = list(train_years) + list(val_years) + list(test_years or [])
                bs = cfg["data"]["block_size"]
                derived_ctx = fit_context(
                    iter_blocks_fn=lambda a, _m=mode, _bs=bs: iter_blocks(
                        a, PARQUET_COLS, _m, _bs, data_root, cyclic=True
                    ),
                    train_years=train_years,
                    all_years=sorted(set(años_eval)),
                    mode=mode,
                    cache_dir=REPO_ROOT / "data" / "processed" / "derived_context",
                )

            # El conjunto de entrenamiento depende de la variante y de los años, no del modelo:
            # se construye una vez y lo comparten los cuatro modelos. Reconstruirlo por modelo
            # multiplicaría por cuatro el recorrido de 33 millones de filas sin cambiar nada.
            train, meta = build_training_set(
                years=train_years,
                mode=mode,
                neg_ratio=cfg["sampling"]["neg_ratio"],
                hard_fraction=cfg["sampling"]["hard_fraction"],
                seed=cfg["seed"],
                block_size=cfg["data"]["block_size"],
                data_root=data_root,
                cyclic=cfg["data"]["cyclic_features"],
                hard_kwargs=cfg["sampling"].get("hard_thresholds"),
                derived=derived_ctx,
                columns=feature_cols_cfg,
            )

            # Conjunto de parada temprana: se muestrea el año de VALIDACIÓN con el mismo
            # criterio que el entrenamiento, para que la métrica de parada sea comparable. No
            # entra nunca en el cálculo del gradiente, solo decide cuántos árboles se conservan.
            early_stop_set = None
            if any(cfg["models"][m].get("early_stopping_rounds") for m in modelos):
                early_stop_set, _ = build_training_set(
                    years=val_years,
                    mode=mode,
                    neg_ratio=cfg["sampling"]["neg_ratio"],
                    hard_fraction=cfg["sampling"]["hard_fraction"],
                    seed=cfg["seed"],
                    block_size=cfg["data"]["block_size"],
                    data_root=data_root,
                    cyclic=cfg["data"]["cyclic_features"],
                    hard_kwargs=cfg["sampling"].get("hard_thresholds"),
                    derived=derived_ctx,
                    columns=feature_cols_cfg,
                )

            for model_name in modelos:
                try:
                    fila = run_one(model_name, mode, cfg, train, meta, train_years,
                                   val_years, test_years, data_root, outdir_models,
                                   early_stop_set=early_stop_set,
                                   derived_ctx=derived_ctx,
                                   feature_cols=feature_cols_cfg)
                    fila["protocolo"] = args.protocol
                    fila["train_years"] = str(train_years)
                    # Dos columnas separadas: las métricas val_* se refieren a `val_year` y las
                    # test_* a `test_year`. Una sola columna "eval_year" resultaba engañosa,
                    # porque etiquetaba con el año de test unas métricas que eran de validación.
                    fila["val_year"] = str(val_years)
                    fila["test_year"] = str(test_years) if test_years else ""
                    fila["hard_fraction"] = cfg["sampling"]["hard_fraction"]
                    fila["neg_ratio"] = cfg["sampling"]["neg_ratio"]
                    filas.append(fila)
                except ImportError as e:
                    logger.error("Se omite %s: %s", model_name, e)
                except Exception:
                    logger.exception("Fallo entrenando %s / %s", model_name, mode)

    if not filas:
        logger.error("No se completó ningún entrenamiento.")
        sys.exit(1)

    df = pd.DataFrame(filas)
    prefijo = cfg["output"]["results_prefix"]
    sufijo = f"_{args.tag}" if args.tag else ""
    ruta_csv = outdir_res / f"{prefijo}_{args.protocol}{sufijo}_resultados.csv"
    df.to_csv(ruta_csv, index=False)
    logger.info("Resultados guardados en %s", ruta_csv)

    clave = f"val_recall_at_fpr{int(cfg['metrics']['fpr_max'] * 100)}"
    columnas = ["modelo", "variante", "val_pr_auc", "val_roc_auc", clave,
                "val_recall_ci90_low", "val_recall_ci90_high"]
    print("\n" + "=" * 78)
    print("RESUMEN — validación")
    print("=" * 78)
    print(df[[c for c in columnas if c in df.columns]].to_string(index=False))
    print("\nLectura: si los intervalos de confianza de dos filas se solapan, la diferencia")
    print("entre ellas NO es concluyente y se prefiere el modelo más simple.")


if __name__ == "__main__":
    main()
