"""
ENTRENAMIENTO COMPARATIVO DEL 2D: features actuales vs enriquecidas  (Enrique)
==============================================================================

Pregunta que responde
---------------------
¿Aportan algo las variables que faltaban en el dataset 2D del equipo?
Se entrena EXACTAMENTE el mismo modelo (LightGBM) con el mismo protocolo, cambiando
solo el juego de variables, para que la comparacion sea limpia:

  A) BASE        -> las 15 variables que usa hoy el pipeline del equipo
  B) ENRIQUECIDO -> las mismas + dias sin lluvia, VPD, memorias de humedad y viento,
                    acumulados 3d/14d, combustible desglosado y estacionalidad ciclica

Protocolo (identico al del equipo, para que los numeros sean comparables)
------------------------------------------------------------------------
  * Train: 2019-2022   |   Test ciego: 2023
  * Hard Negative Mining 1:50 en entrenamiento (todos los positivos + 50 negativos
    por positivo). El test se evalua SOBRE EL ANIO COMPLETO, sin submuestrear.
  * Metricas: PR-AUC, ROC-AUC, Recall a FPR<=5%, Brier.
    Se anade el LIFT (PR-AUC / prevalencia): con 1 incendio por cada ~75.000 filas,
    un PR-AUC de 0,00007 suena a fracaso pero es 5 veces mejor que el azar. El lift
    evita esa lectura equivocada.

Uso
---
  conda activate incendios-forestales
  python entrena_2d_comparativa.py

Salidas
-------
  Datos/resultados_2d/comparativa_features_2023.csv   (metricas de los dos modelos)
  Datos/resultados_2d/importancia_enriquecido.csv     (que variables usa el modelo)
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parent
DIR_DATOS = BASE / "Datos" / "dataset_2d_enriquecido"
DIR_SALIDA = BASE / "Datos" / "resultados_2d"

ANIOS_TRAIN = [2019, 2020, 2021, 2022]
ANIO_TEST = 2023
NEGATIVOS_POR_POSITIVO = 50
SEMILLA = 42

# --- Juegos de variables a comparar -------------------------------------------
FEATURES_BASE = [
    "tmax_vc_t1", "rhmin_vc_t1", "vmax_vc_t1", "prec_dia_t1",
    "prec_acum_7d", "prec_acum_30d", "tmax_media_7d",
    "altitud_media", "pendiente_media", "orientacion_media",
    "combustible_pct_forestal", "mes", "dia_semana", "es_finde",
]

FEATURES_NUEVAS = [
    "dias_sin_lluvia",            # dias secos encadenados
    "vpd_t1",                     # deficit de presion de vapor
    "rhmin_media_7d",             # memoria de humedad
    "vmax_media_7d",              # memoria de viento
    "prec_acum_3d", "prec_acum_14d",
    "dia_anio_sin", "dia_anio_cos",   # estacionalidad continua
    "comb_matorral", "comb_bosque_coniferas", "comb_bosque_frondosas",
    "comb_bosque_mixto", "comb_pastizal", "comb_agricola",
    "comb_urbano", "comb_agua_humedal",
    "orient_norte", "orient_sur", "orient_este", "orient_oeste",
]

FEATURES_ENRIQUECIDO = FEATURES_BASE + FEATURES_NUEVAS


# ---------------------------------------------------------------- carga de datos
def bloques(anio: int) -> list[str]:
    return sorted(glob.glob(str(DIR_DATOS / str(anio) / "bloque_*.parquet")))


def carga_entrenamiento(columnas: list[str]) -> pd.DataFrame:
    """
    Carga los anios de entrenamiento quedandose con TODOS los positivos y una
    muestra aleatoria de negativos (1:50).

    Se hace bloque a bloque para no cargar 45 millones de filas en memoria: de cada
    bloque se guardan sus positivos y una fraccion pequenia de sus negativos.
    """
    rng = np.random.default_rng(SEMILLA)
    cols = list(dict.fromkeys(columnas + ["target"]))

    # Primera pasada: contar positivos y negativos para calibrar la fraccion
    n_pos = n_neg = 0
    for anio in ANIOS_TRAIN:
        for f in bloques(anio):
            t = pd.read_parquet(f, columns=["target"])
            n_pos += int(t.target.sum())
            n_neg += len(t) - int(t.target.sum())
    fraccion = min(1.0, (NEGATIVOS_POR_POSITIVO * n_pos) / max(n_neg, 1))
    print(f"  positivos: {n_pos:,} | negativos: {n_neg:,} -> se muestrea "
          f"{fraccion:.4%} de los negativos")

    piezas = []
    for anio in ANIOS_TRAIN:
        for f in bloques(anio):
            df = pd.read_parquet(f, columns=cols)
            pos = df[df.target == 1]
            neg = df[df.target == 0]
            if len(neg):
                mascara = rng.random(len(neg)) < fraccion
                neg = neg[mascara]
            piezas.append(pd.concat([pos, neg], ignore_index=True))
    train = pd.concat(piezas, ignore_index=True).sample(frac=1, random_state=SEMILLA)
    print(f"  dataset de entrenamiento: {len(train):,} filas "
          f"({int(train.target.sum())} positivos)")
    return train.reset_index(drop=True)


def predice_test(modelo, columnas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Evalua sobre el anio de test COMPLETO (11,2 millones de filas), leyendo y
    prediciendo bloque a bloque para no agotar la memoria.
    """
    y_true, y_prob = [], []
    for f in bloques(ANIO_TEST):
        df = pd.read_parquet(f, columns=list(dict.fromkeys(columnas + ["target"])))
        y_true.append(df.target.to_numpy())
        y_prob.append(modelo.predict_proba(df[columnas].fillna(0))[:, 1])
    return np.concatenate(y_true), np.concatenate(y_prob)


# ---------------------------------------------------------------- metricas
def evalua(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    prevalencia = y_true.mean()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.where(fpr <= 0.05)[0]
    recall_fpr5 = tpr[idx[-1]] if len(idx) else 0.0

    return {
        "pr_auc": pr_auc,
        "lift_vs_azar": pr_auc / prevalencia,      # cuantas veces mejor que el azar
        "roc_auc": roc_auc_score(y_true, y_prob),
        "recall_at_fpr5": recall_fpr5,
        "brier": brier_score_loss(y_true, y_prob),
    }


def entrena_y_evalua(nombre: str, columnas: list[str], train: pd.DataFrame) -> dict:
    print(f"\n--- {nombre} ({len(columnas)} variables) ---")
    modelo = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=SEMILLA, n_jobs=-1, verbose=-1,
    )
    modelo.fit(train[columnas].fillna(0), train.target)

    y_true, y_prob = predice_test(modelo, columnas)
    m = evalua(y_true, y_prob)
    m["modelo"] = nombre
    m["n_features"] = len(columnas)
    print(f"  PR-AUC {m['pr_auc']:.6f} (x{m['lift_vs_azar']:.1f} vs azar) | "
          f"ROC-AUC {m['roc_auc']:.4f} | Recall@FPR5% {m['recall_at_fpr5']:.2%}")

    if nombre.startswith("ENRIQUECIDO"):
        imp = pd.DataFrame({"variable": columnas,
                            "importancia": modelo.feature_importances_}
                           ).sort_values("importancia", ascending=False)
        DIR_SALIDA.mkdir(parents=True, exist_ok=True)
        imp.to_csv(DIR_SALIDA / "importancia_enriquecido.csv", index=False)
        print("\n  Top 12 variables mas usadas por el modelo:")
        print(imp.head(12).to_string(index=False))
    return m


if __name__ == "__main__":
    print("=" * 70)
    print("COMPARATIVA 2D: features actuales del equipo vs enriquecidas")
    print(f"Train {ANIOS_TRAIN} | Test ciego {ANIO_TEST} | HNM 1:{NEGATIVOS_POR_POSITIVO}")
    print("=" * 70)

    print("\nCargando entrenamiento...")
    train = carga_entrenamiento(FEATURES_ENRIQUECIDO)

    resultados = [
        entrena_y_evalua("BASE (equipo)", FEATURES_BASE, train),
        entrena_y_evalua("ENRIQUECIDO (Enrique)", FEATURES_ENRIQUECIDO, train),
    ]

    df_res = pd.DataFrame(resultados)[
        ["modelo", "n_features", "pr_auc", "lift_vs_azar", "roc_auc", "recall_at_fpr5", "brier"]]
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(DIR_SALIDA / "comparativa_features_2023.csv", index=False)

    print("\n" + "=" * 70)
    print("RESULTADOS (test ciego 2023, anio completo)")
    print("=" * 70)
    print(df_res.to_string(index=False))
    print(f"\nGuardado en {DIR_SALIDA}")
