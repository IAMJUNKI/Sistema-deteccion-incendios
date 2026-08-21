"""
Script de Ejecución del Experimento Baseline del TFM.

Carga el dataset maestro de Galicia (2019-2023), entrena y evalua cuantitativamente los 4 modelos
baseline (Logistic Regression, Random Forest, XGBoost, LightGBM) con dividision temporal estricta
(Train 2019-2022 / Test 2023) y Hard Negative Mining.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.models.train_baseline import train_and_evaluate_baselines


def run_experiment():
    print("======================================================================")
    print("🔥 EJECUTANDO EXPERIMENTO BASELINE COMPLETO DE PREDICCIÓN DE INCENDIOS")
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
    print(f"🔥 Total igniciones reales en dataset: {df_all['target'].sum():,} eventos.")

    # Selección de variables explicativas continuas y categóricas
    feature_cols = [
        "tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia",
        "prec_acum_7d", "prec_acum_30d", "tmax_media_7d",
        "alerta_30_30", "altitud_media", "pendiente_media",
        "orientacion_media", "combustible_pct_forestal",
        "mes", "dia_semana", "es_finde"
    ]

    print("\n🧠 Iniciando entrenamiento de modelos baseline...")
    df_results = train_and_evaluate_baselines(
        df_master=df_all,
        feature_cols=feature_cols,
        target_col="target",
        date_col="fecha",
        train_years=[2019, 2020, 2021, 2022],
        test_year=2023,
        negative_ratio=50
    )

    print("\n======================================================================")
    print("📊 RESULTADOS EMPÍRICOS DE MODELOS BASELINE (EVALUACIÓN AÑO 2023)")
    print("======================================================================")
    print(df_results.to_string(index=False))

    output_path = Path("docs/technical/baseline_results_2023.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n💾 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    run_experiment()
