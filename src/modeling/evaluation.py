"""Métricas adecuadas para el problema desbalanceado de ignición."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_binary_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    top_fraction: float = 0.01,
) -> dict[str, float]:
    """Calcula métricas globales y recall al inspeccionar el riesgo más alto."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true y probabilities deben tener la misma forma.")
    if y_true.ndim != 1 or len(y_true) == 0 or not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true debe ser un vector binario no vacío.")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction debe estar entre 0 y 1.")
    if y_true.min() == y_true.max():
        raise ValueError("Se requieren ambas clases para evaluar.")

    top_n = max(1, int(np.ceil(len(y_true) * top_fraction)))
    selected = np.argsort(probabilities)[-top_n:]
    positives = int(y_true.sum())
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "recall_at_top_fraction": float(y_true[selected].sum() / positives),
        "top_fraction": float(top_fraction),
    }
