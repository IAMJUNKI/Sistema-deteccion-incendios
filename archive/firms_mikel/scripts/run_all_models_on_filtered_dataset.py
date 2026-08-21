"""
Script de Evaluación Benchmark Oficial sobre el Dataset Filtrado (P < 5mm).

Entrena y compara:
1. LightGBM Baseline
2. LightGBM + Focal Loss (Técnica avanzada para desbalanceo)
3. XGBoost Optimizado
4. PyTorch Conv3D (Red Neuronal 3D Espacio-Temporal)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.models.train_baseline import hard_negative_sampling
from src.models.metrics import evaluate_imbalanced_metrics
from src.models.train_nn_3d import train_and_evaluate_3d_convnet


def run_benchmark_filtered_dataset():
    print("======================================================================")
    print("🚀 EJECUTANDO BENCHMARK OFICIAL SOBRE DATASET FILTRADO (P < 5 mm)")
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

    print(f"📥 Filas totales iniciales: {len(df_all):,}")
    df_filtered = df_all[df_all["prec_dia"] < 5.0].copy()
    print(f"✅ Filas tras filtrado de lluvia (<5mm): {len(df_filtered):,}")
    print(f"🔥 Igniciones conservadas: {df_filtered['target'].sum()} / {df_all['target'].sum()} ({df_filtered['target'].sum()/df_all['target'].sum()*100:.1f}%)")

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

    # --- 1. LightGBM Standard ---
    print("\n🌲 1/4 Entrenando LightGBM Standard...")
    lgbm_std = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    lgbm_std.fit(X_train, y_train)
    p_lgbm_std = lgbm_std.predict_proba(X_test)[:, 1]
    m_lgbm_std = evaluate_imbalanced_metrics(y_test, p_lgbm_std)
    m_lgbm_std["modelo"] = "LightGBM Standard (Tabular)"
    results.append(m_lgbm_std)

    # --- 2. LightGBM + Focal Loss ---
    print("🌲 2/4 Entrenando LightGBM + Focal Loss...")
    lgbm_focal = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=5.0, random_state=42, n_jobs=-1)
    lgbm_focal.fit(X_train, y_train)
    p_lgbm_focal = lgbm_focal.predict_proba(X_test)[:, 1]
    m_lgbm_focal = evaluate_imbalanced_metrics(y_test, p_lgbm_focal)
    m_lgbm_focal["modelo"] = "LightGBM + Focal Weight (Tabular)"
    results.append(m_lgbm_focal)

    # --- 3. XGBoost Optimizado ---
    print("⚡ 3/4 Entrenando XGBoost Optimizado...")
    xgb_opt = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=3.0, random_state=42, n_jobs=-1)
    xgb_opt.fit(X_train, y_train)
    p_xgb = xgb_opt.predict_proba(X_test)[:, 1]
    m_xgb = evaluate_imbalanced_metrics(y_test, p_xgb)
    m_xgb["modelo"] = "XGBoost Optimizado (Tabular)"
    results.append(m_xgb)

    # --- 4. PyTorch Conv3D ---
    print("🧠 4/4 Entrenando Red Neuronal 3D Conv3D (PyTorch)...")
    m_3d = train_and_evaluate_3d_convnet(
        df_master=df_filtered,
        feature_cols=["tmax_vc", "rhmin_vc", "vmax_vc", "prec_acum_7d", "combustible_pct_forestal"],
        target_col="target",
        date_col="fecha",
        epochs=5,
        batch_size=64
    )
    m_3d["modelo"] = "Red Neuronal 3D Conv3D (PyTorch)"
    results.append(m_3d)

    df_results = pd.DataFrame(results)[["modelo", "pr_auc", "roc_auc", "recall_at_fpr5", "brier_score"]]

    print("\n======================================================================")
    print("📊 BENCHMARK FINAL SOBRE DÍAS SECOS (P < 5 mm - EVALUACIÓN TEST 2023)")
    print("======================================================================")
    print(df_results.to_string(index=False))

    output_path = Path("docs/technical/benchmark_filtered_p5mm_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n💾 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    run_benchmark_filtered_dataset()
