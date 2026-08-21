"""
Script de Ejecución del Experimento de Red Neuronal 3D (PyTorch Conv3D).

Carga los datos históricos (2019-2023), genera parches espacio-temporales 3D,
entrena el modelo PyTorch Conv3D y compara sus métricas frente a LightGBM.
"""

from pathlib import Path
import pandas as pd

from src.models.train_nn_3d import train_and_evaluate_3d_convnet


def run_3d_experiment():
    print("======================================================================")
    print("🧠 EJECUTANDO EXPERIMENTO DE RED NEURONAL 3D CONV3D (PYTORCH)")
    print("======================================================================")

    data_dir = Path("misc/Dataset/Mike")
    parquet_files = [
        data_dir / "dataset_maestro_2019.parquet",
        data_dir / "dataset_maestro_2020.parquet",
        data_dir / "dataset_maestro_2021.parquet",
        data_dir / "dataset_maestro_2022.parquet",
        data_dir / "dataset_maestro_2023.parquet"
    ]

    dfs = []
    for p_file in parquet_files:
        if p_file.exists():
            print(f"📥 Cargando {p_file.name}...")
            df = pd.read_parquet(p_file)
            dfs.append(df)

    if not dfs:
        print("❌ No se encontraron archivos de dataset.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"✅ Dataset cargado correctamente: {len(df_all):,} filas totales.")

    feature_cols = [
        "tmax_vc", "rhmin_vc", "vmax_vc",
        "prec_acum_7d", "combustible_pct_forestal"
    ]

    metrics_3d = train_and_evaluate_3d_convnet(
        df_master=df_all,
        feature_cols=feature_cols,
        target_col="target",
        date_col="fecha",
        epochs=5,
        batch_size=64
    )

    print("\n======================================================================")
    print("📊 COMPARATIVA METODOLÓGICA FINAL (MODELO TABULAR VS RED NEURONAL 3D)")
    print("======================================================================")

    results_table = [
        {"modelo": "LightGBM Tabular 2D (Baseline)", "pr_auc": 0.000071, "roc_auc": 0.8377, "recall_at_fpr5": 0.3235, "brier_score": 0.0010},
        {"modelo": "XGBoost Tabular 2D", "pr_auc": 0.000049, "roc_auc": 0.8297, "recall_at_fpr5": 0.2745, "brier_score": 0.0007},
        {"modelo": metrics_3d["modelo"], "pr_auc": metrics_3d["pr_auc"], "roc_auc": metrics_3d["roc_auc"], "recall_at_fpr5": metrics_3d["recall_at_fpr5"], "brier_score": metrics_3d["brier_score"]}
    ]

    df_comp = pd.DataFrame(results_table)
    print(df_comp.to_string(index=False))

    output_path = Path("docs/technical/nn_3d_vs_tabular_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(output_path, index=False)
    print(f"\n💾 Resultados comparativos guardados en: {output_path}")


if __name__ == "__main__":
    run_3d_experiment()
