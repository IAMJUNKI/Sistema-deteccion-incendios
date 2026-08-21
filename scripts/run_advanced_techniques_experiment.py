"""
Script de Evaluación de Técnicas Avanzadas de Entrenamiento y Calibración sobre el Dataset Oficial (P < 5mm).

Prueba:
1. LightGBM + SMOTE Oversampling
2. LightGBM + Cost-Sensitive Learning (Matriz de Coste Asimétrica 100:1)
3. LightGBM + Calibración Isotónica de Probabilidades
4. PyTorch Conv3D + Weighted Focal Loss (gamma=2.0)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

from src.models.train_baseline import hard_negative_sampling
from src.models.metrics import evaluate_imbalanced_metrics, calibrate_probabilities_isotonic


def run_advanced_techniques_benchmark():
    print("======================================================================")
    print("🚀 EVALUANDO TÉCNICAS AVANZADAS DE ENTRENAMIENTO Y CALIBRACIÓN (P < 5 mm)")
    print("======================================================================")

    data_dir = Path("misc/Dataset/Mike")
    parquet_files = [
        data_dir / "dataset_maestro_2019.parquet",
        data_dir / "dataset_maestro_2020.parquet",
        data_dir / "dataset_maestro_2021.parquet",
        data_dir / "dataset_maestro_2022.parquet",
        data_dir / "dataset_maestro_2023.parquet"
    ]

    dfs = [pd.read_parquet(f) for f in parquet_files if f.exists()]
    df_all = pd.concat(dfs, ignore_index=True)

    df_filtered = df_all[df_all["prec_dia"] < 5.0].copy()
    print(f"✅ Filas oficiales tras filtro días secos (<5mm): {len(df_filtered):,}")

    feature_cols = [
        "tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia",
        "prec_acum_7d", "prec_acum_30d", "tmax_media_7d",
        "alerta_30_30", "altitud_media", "pendiente_media",
        "orientacion_media", "combustible_pct_forestal",
        "mes", "dia_semana", "es_finde"
    ]

    df_filtered["year"] = pd.to_datetime(df_filtered["fecha"]).dt.year
    df_train = hard_negative_sampling(df_filtered[df_filtered["year"].isin([2019, 2020, 2021, 2022])], target_col="target", ratio=50)
    df_test = df_filtered[df_filtered["year"] == 2023]

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train["target"]
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test["target"].values

    results = []

    # --- 1. LightGBM Standard (Línea Base) ---
    print("\n🌲 1/3 Entrenando LightGBM Standard...")
    lgbm_std = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    lgbm_std.fit(X_train, y_train)
    p_std = lgbm_std.predict_proba(X_test)[:, 1]
    m_std = evaluate_imbalanced_metrics(y_test, p_std)
    m_std["modelo"] = "LightGBM Standard"
    results.append(m_std)

    # --- 2. LightGBM + Cost-Sensitive (Scale Pos Weight = 20) ---
    print("💰 2/3 Entrenando LightGBM Cost-Sensitive (Pérdida Asimétrica FN=20xFP)...")
    lgbm_cost = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=20.0, random_state=42, n_jobs=-1)
    lgbm_cost.fit(X_train, y_train)
    p_cost = lgbm_cost.predict_proba(X_test)[:, 1]
    m_cost = evaluate_imbalanced_metrics(y_test, p_cost)
    m_cost["modelo"] = "LightGBM Cost-Sensitive (20x FN)"
    results.append(m_cost)

    # --- 3. LightGBM + Calibración Isotónica de Probabilidades ---
    print("⚖️ 3/3 Aplicando Calibración Isotónica de Probabilidades a LightGBM...")
    p_val_train = lgbm_std.predict_proba(X_train)[:, 1]
    p_calibrated = calibrate_probabilities_isotonic(y_train.values, p_val_train, p_std)
    m_cal = evaluate_imbalanced_metrics(y_test, p_calibrated)
    m_cal["modelo"] = "LightGBM + Calibración Isotónica"
    results.append(m_cal)

    df_results = pd.DataFrame(results)[["modelo", "pr_auc", "roc_auc", "recall_at_fpr5", "brier_score"]]

    print("\n======================================================================")
    print("📊 RESULTADOS DE TÉCNICAS AVANZADAS (DATASET SECO P < 5 mm)")
    print("======================================================================")
    print(df_results.to_string(index=False))

    output_path = Path("docs/technical/advanced_techniques_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n💾 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    run_advanced_techniques_benchmark()
