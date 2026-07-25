"""
Modulo de Métricas de Evaluación y Calibración para Clasificación Desbalanceada.

Proporciona metricas adaptadas a baja prevalencia: PR-AUC (Precision-Recall),
Recall a Tasa Fija de Falsa Alarma (FPR <= 5%), Brier Score y Calibración Isotónica.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression


def evaluate_imbalanced_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Calcula el conjunto completo de métricas recomendadas para el TFM.
    """
    # 1. PR-AUC (Precision-Recall Area Under Curve)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # 2. ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5

    # 3. Recall a FPR <= 5% (Tasa de Falsos Positivos fija)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx_5pct = np.where(fpr <= 0.05)[0]
    recall_at_fpr5 = tpr[idx_5pct[-1]] if len(idx_5pct) > 0 else 0.0

    # 4. Brier Score (Calidad de la calibración de probabilidad)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "recall_at_fpr5": float(recall_at_fpr5),
        "brier_score": float(brier)
    }


def calibrate_probabilities_isotonic(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    y_prob_test: np.ndarray
) -> np.ndarray:
    """
    Aplica Regresión Isotónica entrenada sobre el conjunto de validación
    para re-calibrar las probabilidades devueltas por modelos entrenados con submuestreo.
    """
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(y_prob_val, y_true_val)
    calibrated_probs = iso_reg.transform(y_prob_test)
    return calibrated_probs


if __name__ == "__main__":
    print("Módulo metrics listo para importar.")
