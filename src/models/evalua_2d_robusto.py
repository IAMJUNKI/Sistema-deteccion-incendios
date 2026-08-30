"""
EVALUACION ROBUSTA DEL 2D: multi-anio, ablacion por grupos e intervalos de confianza
===================================================================================

Por que hace falta este script
------------------------------
La primera comparativa (entrena_2d_comparativa.py) dio un resultado sorprendente:
anadir 20 variables EMPEORO el modelo. Al mirar el detalle aparece el problema de
fondo: el test de 2023 solo tiene 102 incendios, asi que un Recall del 20,6% frente
a otro del 19,6% son 21 incendios frente a 20. UN incendio de diferencia.

Con esa muestra no se puede ordenar modelos: las diferencias que vemos caen dentro
del ruido. Este script corrige tres cosas:

  1. MULTI-ANIO: en vez de un unico test, se evalua en 2021, 2022 y 2023 por separado
     con validacion temporal expansiva (entrenar siempre solo con el pasado).
  2. ABLACION POR GRUPOS: en vez de meter las 20 variables nuevas de golpe (lo que
     provoca sobreajuste con ~1.500 positivos), se prueban por bloques tematicos para
     ver cual aporta de verdad.
  3. INTERVALOS DE CONFIANZA: bootstrap sobre los incendios del test, para saber
     cuando una diferencia es real y cuando es azar.

Uso
---
  conda activate incendios-forestales
  python evalua_2d_robusto.py

Salida
------
  Datos/resultados_2d/evaluacion_robusta.csv
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parent
DIR_DATOS = BASE / "Datos" / "dataset_2d_enriquecido"
DIR_SALIDA = BASE / "Datos" / "resultados_2d"

ANIOS_TEST = [2021, 2022, 2023]      # cada uno se evalua entrenando solo con anios previos
NEG_POR_POS = 50
SEMILLA = 42
N_BOOTSTRAP = 200

# --- Grupos de variables ------------------------------------------------------
G_BASE = [
    "tmax_vc_t1", "rhmin_vc_t1", "vmax_vc_t1", "prec_dia_t1",
    "prec_acum_7d", "prec_acum_30d", "tmax_media_7d",
    "altitud_media", "pendiente_media", "orientacion_media",
    "combustible_pct_forestal", "mes", "dia_semana", "es_finde",
]
G_SEQUEDAD = ["dias_sin_lluvia", "vpd_t1", "prec_acum_3d", "prec_acum_14d"]
G_MEMORIAS = ["rhmin_media_7d", "vmax_media_7d"]
G_COMBUSTIBLE = ["comb_matorral", "comb_bosque_coniferas", "comb_bosque_frondosas",
                 "comb_bosque_mixto", "comb_pastizal", "comb_agricola",
                 "comb_urbano", "comb_agua_humedal"]
G_ORIENTACION = ["orient_norte", "orient_sur", "orient_este", "orient_oeste"]
G_ESTACIONAL = ["dia_anio_sin", "dia_anio_cos"]

CONFIGURACIONES = {
    "BASE (equipo)": G_BASE,
    "BASE + sequedad (dias sin lluvia, VPD)": G_BASE + G_SEQUEDAD,
    "BASE + memorias humedad/viento": G_BASE + G_MEMORIAS,
    "BASE + combustible desglosado": G_BASE + G_COMBUSTIBLE,
    "BASE + orientacion desglosada": G_BASE + G_ORIENTACION,
    "BASE + estacionalidad ciclica": G_BASE + G_ESTACIONAL,
    "TODO (enriquecido)": (G_BASE + G_SEQUEDAD + G_MEMORIAS + G_COMBUSTIBLE
                           + G_ORIENTACION + G_ESTACIONAL),
}


def bloques(anio: int) -> list[str]:
    return sorted(glob.glob(str(DIR_DATOS / str(anio) / "bloque_*.parquet")))


def carga_train(anios: list[int], columnas: list[str]) -> pd.DataFrame:
    """Todos los positivos de esos anios + una muestra 1:50 de negativos."""
    rng = np.random.default_rng(SEMILLA)
    cols = list(dict.fromkeys(columnas + ["target"]))

    n_pos = n_neg = 0
    for a in anios:
        for f in bloques(a):
            t = pd.read_parquet(f, columns=["target"])
            p = int(t.target.sum())
            n_pos += p
            n_neg += len(t) - p
    fraccion = min(1.0, (NEG_POR_POS * n_pos) / max(n_neg, 1))

    piezas = []
    for a in anios:
        for f in bloques(a):
            df = pd.read_parquet(f, columns=cols)
            pos = df[df.target == 1]
            neg = df[df.target == 0]
            neg = neg[rng.random(len(neg)) < fraccion]
            piezas.append(pd.concat([pos, neg], ignore_index=True))
    return pd.concat(piezas, ignore_index=True).sample(frac=1, random_state=SEMILLA)


def predice(modelo, anio: int, columnas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predice sobre el anio completo, bloque a bloque (11,2 millones de filas)."""
    y_true, y_prob = [], []
    for f in bloques(anio):
        df = pd.read_parquet(f, columns=list(dict.fromkeys(columnas + ["target"])))
        y_true.append(df.target.to_numpy())
        y_prob.append(modelo.predict_proba(df[columnas].fillna(0))[:, 1])
    return np.concatenate(y_true), np.concatenate(y_prob)


def recall_a_fpr(y_true: np.ndarray, y_prob: np.ndarray, fpr_max: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.where(fpr <= fpr_max)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def intervalo_bootstrap(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Intervalo de confianza del 90% del Recall@FPR5%, remuestreando los incendios.

    Se remuestrean solo los positivos (que son los escasos): es lo que determina
    la incertidumbre real de la metrica cuando hay ~100 fuegos en el test.
    """
    rng = np.random.default_rng(SEMILLA)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)
    prob_neg = y_prob[idx_neg]
    # umbral que deja FPR = 5% (fijo, no depende de los positivos)
    umbral = np.quantile(prob_neg, 0.95)

    valores = []
    for _ in range(N_BOOTSTRAP):
        muestra = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        valores.append((y_prob[muestra] >= umbral).mean())
    return float(np.percentile(valores, 5)), float(np.percentile(valores, 95))


if __name__ == "__main__":
    print("=" * 78)
    print("EVALUACION ROBUSTA DEL 2D — multi-anio + ablacion por grupos + bootstrap")
    print("=" * 78)

    filas = []
    for anio_test in ANIOS_TEST:
        anios_train = [a for a in range(2019, anio_test)]
        print(f"\n### TEST {anio_test} (entrenando con {anios_train})")

        train_completo = carga_train(anios_train, CONFIGURACIONES["TODO (enriquecido)"])
        print(f"    entrenamiento: {len(train_completo):,} filas | "
              f"{int(train_completo.target.sum())} incendios")

        for nombre, columnas in CONFIGURACIONES.items():
            modelo = LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                random_state=SEMILLA, n_jobs=-1, verbose=-1,
            )
            modelo.fit(train_completo[columnas].fillna(0), train_completo.target)

            y_true, y_prob = predice(modelo, anio_test, columnas)
            rec = recall_a_fpr(y_true, y_prob)
            lo, hi = intervalo_bootstrap(y_true, y_prob)
            filas.append({
                "anio_test": anio_test,
                "configuracion": nombre,
                "n_features": len(columnas),
                "incendios_test": int(y_true.sum()),
                "roc_auc": roc_auc_score(y_true, y_prob),
                "pr_auc": pr_auc(y_true, y_prob),
                "recall_fpr5": rec,
                "ic90_bajo": lo,
                "ic90_alto": hi,
            })
            print(f"    {nombre:42s} ROC {filas[-1]['roc_auc']:.4f} | "
                  f"Recall@5% {rec:6.2%}  (IC90 {lo:5.2%}-{hi:5.2%})")

    df = pd.DataFrame(filas)
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR_SALIDA / "evaluacion_robusta.csv", index=False)

    print("\n" + "=" * 78)
    print("RESUMEN — media de los 3 anios de test")
    print("=" * 78)
    resumen = (df.groupby("configuracion")
                 .agg(roc_auc_medio=("roc_auc", "mean"),
                      recall_medio=("recall_fpr5", "mean"),
                      n_features=("n_features", "first"))
                 .sort_values("recall_medio", ascending=False))
    print(resumen.to_string())
    print(f"\nGuardado en {DIR_SALIDA / 'evaluacion_robusta.csv'}")
    print("\nComo leerlo: si los intervalos de confianza de dos configuraciones se")
    print("solapan, la diferencia entre ellas NO es concluyente.")
