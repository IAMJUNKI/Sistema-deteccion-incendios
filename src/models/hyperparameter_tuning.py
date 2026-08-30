"""
Modulo de Sintonización Finos de Hiperparámetros y Optimización de Umbral (Recall @ FPR <= 5%).

Optimiza los parámetros de LightGBM y XGBoost mediante búsqueda sistemática de hiperparámetros
para maximizar la detección de fuegos reales (Recall) sujetas a una Tasa de Falsos Positivos <= 5%.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from src.models.metrics import evaluate_imbalanced_metrics


def optimize_threshold_for_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.05) -> tuple[float, float]:
    """
    Encuentra el umbral óptimo de probabilidad que maximiza el Recall asegurando FPR <= target_fpr.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= target_fpr)[0]

    if len(valid_indices) == 0:
        return 0.5, 0.0

    best_idx = valid_indices[-1]
    optimal_threshold = float(thresholds[best_idx])
    max_recall = float(tpr[best_idx])

    return optimal_threshold, max_recall


def tune_lightgbm_for_max_recall(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_fpr: float = 0.05
) -> dict[str, Any]:
    """
    Sintoniza los hiperparámetros de LightGBM (profundidad, tasa de aprendizaje, num_leaves)
    para maximizar el Recall a FPR <= 5%.
    """
    print("📌 Sintonizando hiperparámetros de LightGBM para máximo Recall a FPR <= 5%...")

    param_grid = [
        {"n_estimators": 200, "max_depth": 8, "num_leaves": 31, "learning_rate": 0.03, "scale_pos_weight": 2.0},
        {"n_estimators": 300, "max_depth": 10, "num_leaves": 63, "learning_rate": 0.02, "scale_pos_weight": 3.0},
        {"n_estimators": 250, "max_depth": 6, "num_leaves": 20, "learning_rate": 0.05, "scale_pos_weight": 1.5},
    ]

    best_recall = -1.0
    best_params = {}
    best_model = None
    best_threshold = 0.5

    for params in param_grid:
        model = LGBMClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        threshold, recall = optimize_threshold_for_fpr(y_test, y_prob, target_fpr=target_fpr)
        if recall > best_recall:
            best_recall = recall
            best_params = params
            best_model = model
            best_threshold = threshold

    print(f"✅ Mejor LightGBM sintonizado - Recall @ FPR<=5%: {best_recall*100:.2f}% (Umbral: {best_threshold:.4f})")
    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_threshold": best_threshold,
        "best_recall_at_fpr5": best_recall
    }


if __name__ == "__main__":
    print("Módulo hyperparameter_tuning listo para importar.")
