"""Métricas de evaluación para clasificación con desbalanceo extremo.

Sustituye a `src/models/metrics.py`, que se conserva intacto para poder reproducir los
resultados anteriores. Las diferencias respecto a aquel módulo están documentadas en
`MODEL_DOCUMENTATION.md` §3.3, y son tres:

1. **PR-AUC con `average_precision_score`** en lugar de `auc(recall, precision)`. El segundo
   interpola linealmente la curva Precision-Recall, lo que introduce un sesgo optimista.
2. **Lift sobre la prevalencia**, porque un PR-AUC de 7·10⁻⁵ es ilegible sin normalizar: parece
   un fracaso y en realidad puede ser ocho veces mejor que el azar.
3. **Intervalos de confianza por bootstrap**. Con ~100-200 incendios en el conjunto de test, un
   punto porcentual de recall son dos incendios: sin intervalos no se pueden ordenar modelos.

Se añade además `recall_at_top_k_daily`, que responde a la pregunta operativa real: si cada día
se patrulla el 1 % del territorio, ¿cuántos incendios se cazan?
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def recall_at_fpr(
    y_true: np.ndarray, y_prob: np.ndarray, fpr_max: float = 0.05
) -> tuple[float, float, float]:
    """Recall alcanzable manteniendo la tasa de falsos positivos en `fpr_max`.

    El umbral se fija en el cuantil `1 - fpr_max` de las puntuaciones de la clase negativa, lo
    que evita ordenar los 11 millones de filas que exigiría `roc_curve`.

    **Tratamiento de empates.** A esta escala hay millones de filas con la misma puntuación: un
    modelo de árboles asigna el mismo valor a todas las que caen en la misma hoja, y la
    regresión isotónica lo agrava porque es una función escalonada. Si el umbral cae sobre una
    meseta, la comparación `>=` se lleva la meseta entera y el FPR real se dispara muy por
    encima del objetivo, inflando el recall de forma espectacular y falsa. Cuando se detecta ese
    caso se pasa a comparación estricta, que es la opción conservadora. En ambos casos se
    devuelve el FPR **realmente alcanzado**, para que el punto de operación sea auditable.

    Args:
        y_true: Etiquetas reales (0/1).
        y_prob: Puntuación o probabilidad predicha.
        fpr_max: Tasa de falsos positivos objetivo.

    Returns:
        Tupla (recall, umbral, fpr_alcanzado).
    """
    neg = y_prob[y_true == 0]
    pos = y_prob[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return 0.0, float("inf"), 0.0

    umbral = float(np.quantile(neg, 1.0 - fpr_max))
    fpr_inclusivo = float((neg >= umbral).mean())

    if fpr_inclusivo > fpr_max:
        fpr_alcanzado = float((neg > umbral).mean())
        recall = float((pos > umbral).mean())
        logger.debug(
            "Empates en el umbral: FPR inclusivo %.4f > objetivo %.4f; se usa comparación "
            "estricta (FPR alcanzado %.4f).", fpr_inclusivo, fpr_max, fpr_alcanzado,
        )
    else:
        fpr_alcanzado = fpr_inclusivo
        recall = float((pos >= umbral).mean())

    return recall, umbral, fpr_alcanzado


def recall_at_top_k_daily(
    y_true: np.ndarray, y_prob: np.ndarray, day_index: np.ndarray, k: float = 0.01
) -> float:
    """Recall alertando cada día únicamente el `k` por uno de celdas con más riesgo.

    Es la métrica más honesta operativamente. Un FPR del 5 % sobre Galicia son 1.535 km², que
    ninguna brigada patrulla; el 1 % son 307 celdas, que sí. Al aplicarse día a día, refleja que
    la decisión de despliegue se toma cada mañana sobre el mapa de ese día.

    Args:
        y_true: Etiquetas reales (0/1).
        y_prob: Puntuación predicha.
        day_index: Identificador entero del día de cada fila.
        k: Fracción de celdas que se alertan cada día.

    Returns:
        Fracción de incendios que caen dentro de la selección diaria.
    """
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return 0.0

    orden = np.argsort(day_index, kind="stable")
    dias = day_index[orden]
    probs = y_prob[orden]
    reales = y_true[orden]

    cortes = np.flatnonzero(np.diff(dias)) + 1
    capturados = 0
    for ini, fin in zip(
        np.concatenate([[0], cortes]), np.concatenate([cortes, [len(dias)]])
    ):
        p_dia = probs[ini:fin]
        y_dia = reales[ini:fin]
        if y_dia.sum() == 0:
            continue
        n_alerta = max(1, int(round(len(p_dia) * k)))
        umbral = np.partition(p_dia, -n_alerta)[-n_alerta]
        capturados += int(((y_dia == 1) & (p_dia >= umbral)).sum())

    return float(capturados / total_pos)


def bootstrap_recall_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_max: float = 0.05,
    n_boot: int = 1000,
    alpha: float = 0.10,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalo de confianza del recall remuestreando los incendios.

    Se remuestrean solo los positivos porque son los escasos: con ~100 fuegos son ellos, y no
    los millones de negativos, los que determinan la incertidumbre de la métrica. El umbral se
    mantiene fijo (calculado sobre los negativos, que no se remuestrean).

    Args:
        y_true: Etiquetas reales.
        y_prob: Puntuación predicha.
        fpr_max: Tasa de falsos positivos que fija el umbral.
        n_boot: Número de réplicas bootstrap.
        alpha: Nivel de significación (0.10 → intervalo del 90 %).
        seed: Semilla.

    Returns:
        Tupla (límite inferior, límite superior).
    """
    idx_pos = np.flatnonzero(y_true == 1)
    if len(idx_pos) == 0:
        return 0.0, 0.0

    recall_puntual, umbral, _ = recall_at_fpr(y_true, y_prob, fpr_max)
    # Se replica el mismo criterio de empates que usó `recall_at_fpr`, para que la media de las
    # réplicas bootstrap sea coherente con el valor puntual que acompaña al intervalo.
    inclusivo = (y_prob[idx_pos] >= umbral).astype(np.float64)
    estricto = (y_prob[idx_pos] > umbral).astype(np.float64)
    aciertos = inclusivo if np.isclose(inclusivo.mean(), recall_puntual) else estricto

    rng = np.random.default_rng(seed)
    muestras = rng.choice(aciertos, size=(n_boot, len(aciertos)), replace=True).mean(axis=1)
    return (
        float(np.percentile(muestras, 100 * alpha / 2)),
        float(np.percentile(muestras, 100 * (1 - alpha / 2))),
    )


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    day_index: Optional[np.ndarray] = None,
    fpr_max: float = 0.05,
    top_k: float = 0.01,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Calcula el cuadro completo de métricas del proyecto.

    Separa deliberadamente dos familias de métricas que se calculan sobre entradas distintas:

    - **Métricas de ordenación** (PR-AUC, ROC-AUC, recalls): sobre la puntuación **cruda** del
      modelo. La calibración es una transformación monótona, así que en teoría no altera el
      orden; en la práctica la isotónica es escalonada y crea millones de empates que sí
      distorsionan cualquier métrica basada en umbral.
    - **Métricas de calibración** (Brier): sobre la probabilidad **calibrada**, que es la única
      que tiene significado como probabilidad. Antes de recalibrar, el Brier mide el sesgo del
      submuestreo y no la calidad del modelo, así que se omite si no se aporta.

    Args:
        y_true: Etiquetas reales (0/1).
        y_score: Puntuación cruda del modelo, tal como sale de `predict_proba`.
        y_prob_calibrated: Probabilidad recalibrada. Opcional.
        day_index: Identificador entero del día, para el recall diario. Opcional.
        fpr_max: Tasa de falsos positivos objetivo.
        top_k: Fracción de celdas alertadas cada día.
        n_boot: Réplicas bootstrap.
        seed: Semilla.

    Returns:
        Diccionario de métricas.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int(y_true.sum())
    prevalencia = float(n_pos / len(y_true)) if len(y_true) else 0.0

    ap = float(average_precision_score(y_true, y_score)) if n_pos else float("nan")
    rec_fpr, umbral, fpr_real = recall_at_fpr(y_true, y_score, fpr_max)
    lo, hi = bootstrap_recall_ci(y_true, y_score, fpr_max, n_boot, seed=seed)

    resultado = {
        "n_rows": int(len(y_true)),
        "n_positives": n_pos,
        "prevalence": prevalencia,
        "pr_auc": ap,
        "lift_vs_azar": float(ap / prevalencia) if prevalencia > 0 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if n_pos else float("nan"),
        f"recall_at_fpr{int(fpr_max * 100)}": rec_fpr,
        "recall_ci90_low": lo,
        "recall_ci90_high": hi,
        "threshold_at_fpr": umbral,
        "fpr_achieved": fpr_real,
    }

    if day_index is not None:
        resultado[f"recall_at_top{top_k:.0%}_daily"] = recall_at_top_k_daily(
            y_true, y_score, day_index, top_k
        )

    resultado["brier_score"] = (
        float(brier_score_loss(y_true, np.asarray(y_prob_calibrated, dtype=np.float64)))
        if y_prob_calibrated is not None
        else float("nan")
    )
    return resultado


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 12
) -> pd.DataFrame:
    """Tabla de fiabilidad: probabilidad predicha frente a frecuencia observada.

    Los cortes se hacen por cuantiles y no en tramos iguales porque a esta prevalencia casi
    todas las predicciones se acumulan cerca de cero y los tramos uniformes quedarían vacíos.

    Args:
        y_true: Etiquetas reales.
        y_prob: Probabilidad predicha (recalibrada).
        n_bins: Número de tramos.

    Returns:
        DataFrame con la predicción media, la frecuencia observada y el tamaño de cada tramo.
    """
    bordes = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(bordes) < 3:
        return pd.DataFrame(columns=["bin", "pred_media", "obs_frecuencia", "n"])
    tramo = np.clip(np.digitize(y_prob, bordes[1:-1]), 0, len(bordes) - 2)
    df = pd.DataFrame({"bin": tramo, "p": y_prob, "y": y_true})
    return (
        df.groupby("bin")
        .agg(pred_media=("p", "mean"), obs_frecuencia=("y", "mean"), n=("y", "size"))
        .reset_index()
    )


def summarize(resultados: list[dict]) -> pd.DataFrame:
    """Ordena una lista de resultados en una tabla comparable, de mayor a menor recall."""
    df = pd.DataFrame(resultados)
    col_recall = next((c for c in df.columns if c.startswith("recall_at_fpr")), None)
    return df.sort_values(col_recall, ascending=False) if col_recall else df
