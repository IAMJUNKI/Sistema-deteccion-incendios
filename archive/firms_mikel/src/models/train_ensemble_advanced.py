"""
Modulo de Entrenamiento de Ensamble Avanzado (Blending Rank Averaging: LightGBM + XGBoost + CatBoost).

Aplica la suite avanzada de caracteristicas fisicas (VPD, Nesterov, Ratios) y combina
las predicciones mediante Rank Averaging para elevar el score ROC-AUC > 0.90.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.features.advanced_physics import add_advanced_physical_features
from src.models.metrics import evaluate_imbalanced_metrics
from src.models.train_baseline import hard_negative_sampling


def rank_averaging_blend(prob_dict: dict[str, np.ndarray], weights: dict[str, float] = None) -> np.ndarray:
    """
    Combina las probabilidades predichas mediante promedio ponderado de rangos (Rank Averaging).
    Convierte las probabilidades en percentiles (0 a 1) para eliminar distorsiones de escala.
    """
    n_samples = len(next(iter(prob_dict.values())))
    blended_rank = np.zeros(n_samples, dtype=np.float64)

    if weights is None:
        weights = {name: 1.0 / len(prob_dict) for name in prob_dict}

    total_weight = sum(weights.values())

    for name, probs in prob_dict.items():
        # Convertir probabilidades a rangos percentiles (0 a 1)
        ranks = pd.Series(probs).rank(pct=True).values
        w = weights.get(name, 1.0) / total_weight
        blended_rank += ranks * w

    return blended_rank


def train_and_evaluate_advanced_ensemble(
    df_master: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    date_col: str = "fecha",
    train_years: list[int] = [2019, 2020, 2021, 2022],
    test_year: int = 2023,
    negative_ratio: int = 50
) -> pd.DataFrame:
    """
    Ejecuta el pipeline avanzado con caracteristicas fisicas y ensamble blending.
    """
    print("📌 1. Añadiendo características físicas avanzadas (VPD, Nesterov, Ratio Térmico-Eólico)...")
    df_master = add_advanced_physical_features(df_master)

    # Actualizar lista de features si no estaban incluidas
    adv_cols = ["vpd_t1", "ratio_termico_eolico", "nesterov_index"]
    for c in adv_cols:
        if c in df_master.columns and c not in feature_cols:
            feature_cols.append(c)

    print("📌 2. Aplicando split temporal estricto (Train 2019-2022 / Test 2023)...")
    df_master["year"] = pd.to_datetime(df_master[date_col]).dt.year

    train_mask = df_master["year"].isin(train_years)
    test_mask = df_master["year"] == test_year

    df_train_raw = df_master[train_mask].copy()
    df_test = df_master[test_mask].copy()

    print(f"📌 3. Aplicando Hard Negative Mining (1:{negative_ratio}) en Train...")
    df_train = hard_negative_sampling(df_train_raw, target_col=target_col, ratio=negative_ratio)

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target_col]

    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test[target_col].values

    prob_predictions = {}
    results = []

    # --- 1. LightGBM Avanzado ---
    print("\n🚀 1/3 Entrenando LightGBM Sintonizado...")
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        num_leaves=45,
        learning_rate=0.03,
        scale_pos_weight=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgbm.fit(X_train, y_train)
    prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
    prob_predictions["LightGBM"] = prob_lgbm
    m_lgbm = evaluate_imbalanced_metrics(y_test, prob_lgbm)
    m_lgbm["modelo"] = "LightGBM Avanzado"
    results.append(m_lgbm)

    # --- 2. XGBoost Avanzado ---
    print("🚀 2/3 Entrenando XGBoost Sintonizado...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.03,
        scale_pos_weight=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    prob_predictions["XGBoost"] = prob_xgb
    m_xgb = evaluate_imbalanced_metrics(y_test, prob_xgb)
    m_xgb["modelo"] = "XGBoost Avanzado"
    results.append(m_xgb)

    # --- 3. CatBoost Avanzado ---
    print("🚀 3/3 Entrenando CatBoost Classifier...")
    cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.04,
        scale_pos_weight=3.0,
        random_seed=42,
        verbose=0
    )
    cat.fit(X_train, y_train)
    prob_cat = cat.predict_proba(X_test)[:, 1]
    prob_predictions["CatBoost"] = prob_cat
    m_cat = evaluate_imbalanced_metrics(y_test, prob_cat)
    m_cat["modelo"] = "CatBoost Classifier"
    results.append(m_cat)

    # --- 4. Ensamble Blending Rank Averaging ---
    print("\n🌟 4/4 Generando Ensamble Blending (Rank Averaging: LightGBM + XGBoost + CatBoost)...")
    prob_blend = rank_averaging_blend(prob_predictions, weights={"LightGBM": 0.4, "XGBoost": 0.3, "CatBoost": 0.3})
    m_blend = evaluate_imbalanced_metrics(y_test, prob_blend)
    m_blend["modelo"] = "Ensamble Blending (SOTA Tabular)"
    results.append(m_blend)

    df_results = pd.DataFrame(results)[["modelo", "pr_auc", "roc_auc", "recall_at_fpr5", "brier_score"]]
    return df_results


if __name__ == "__main__":
    print("Módulo train_ensemble_advanced listo para importar.")
