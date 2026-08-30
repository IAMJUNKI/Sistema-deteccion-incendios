"""
Pipeline de Entrenamiento y Evaluación Comparativa de Modelos Baseline.

Modelos evaluados:
1. Logistic Regression (Baseline Lineal L2)
2. Random Forest (Baseline Ensamble Embolsado)
3. XGBoost (Baseline Gradiente Potenciado SOTA)
4. LightGBM (Baseline Gradiente Potenciado Ligero)

Aplica dividisión temporal estricta: Train (2019-2022) / Test (2023),
Hard Negative Mining (1:50), métricas PR-AUC, Recall@FPR5% y Calibración Isotónica.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.models.metrics import evaluate_imbalanced_metrics, calibrate_probabilities_isotonic


def hard_negative_sampling(df: pd.DataFrame, target_col: str = "target_ignicion", ratio: int = 50, random_state: int = 42) -> pd.DataFrame:
    """
    Conserva el 100% de la clase positiva (Y=1) y realiza un submuestreo
    inteligente de la clase negativa (Y=0) en una proporcion 1:ratio.
    """
    positives = df[df[target_col] == 1]
    negatives = df[df[target_col] == 0]

    n_positives = len(positives)
    if n_positives == 0:
        return df

    n_negatives_wanted = min(len(negatives), n_positives * ratio)
    negatives_sampled = negatives.sample(n=n_negatives_wanted, random_state=random_state)

    balanced_df = pd.concat([positives, negatives_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df


def train_and_evaluate_baselines(
    df_master: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_ignicion",
    date_col: str = "fecha",
    train_years: list[int] = [2019, 2020, 2021, 2022],
    test_year: int = 2023,
    negative_ratio: int = 50
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de entrenamiento y evaluacion comparativa de modelos.
    """
    print("📌 1. Realizando split temporal de datos...")
    df_master["year"] = pd.to_datetime(df_master[date_col]).dt.year

    train_mask = df_master["year"].isin(train_years)
    test_mask = df_master["year"] == test_year

    df_train_raw = df_master[train_mask].copy()
    df_test = df_master[test_mask].copy()

    print(f"   - Entrenando con años {train_years}: {len(df_train_raw):,} filas ({df_train_raw[target_col].sum()} fuegos)")
    print(f"   - Evaluando con año {test_year}: {len(df_test):,} filas ({df_test[target_col].sum()} fuegos)")

    print(f"📌 2. Aplicando Hard Negative Mining (1:{negative_ratio}) en Train...")
    df_train = hard_negative_sampling(df_train_raw, target_col=target_col, ratio=negative_ratio)

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target_col]

    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test[target_col].values

    results = []

    # --- 1. Regresión Logística ---
    print("\n🚀 Entrenando Regresión Logística...")
    lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    lr_model.fit(X_train, y_train)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    metrics_lr = evaluate_imbalanced_metrics(y_test, y_prob_lr)
    metrics_lr["modelo"] = "Regresión Logística (L2)"
    results.append(metrics_lr)

    # --- 2. Random Forest ---
    print("🚀 Entrenando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    metrics_rf = evaluate_imbalanced_metrics(y_test, y_prob_rf)
    metrics_rf["modelo"] = "Random Forest"
    results.append(metrics_rf)

    # --- 3. XGBoost ---
    if HAS_XGB:
        print("🚀 Entrenando XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        metrics_xgb = evaluate_imbalanced_metrics(y_test, y_prob_xgb)
        metrics_xgb["modelo"] = "XGBoost Classifier"
        results.append(metrics_xgb)

    # --- 4. LightGBM ---
    if HAS_LGBM:
        print("🚀 Entrenando LightGBM...")
        lgbm_model = LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        lgbm_model.fit(X_train, y_train)
        y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
        metrics_lgbm = evaluate_imbalanced_metrics(y_test, y_prob_lgbm)
        metrics_lgbm["modelo"] = "LightGBM Classifier"
        results.append(metrics_lgbm)

    df_results = pd.DataFrame(results)[["modelo", "pr_auc", "roc_auc", "recall_at_fpr5", "brier_score"]]
    return df_results


if __name__ == "__main__":
    print("Módulo train_baseline listo para importar.")
