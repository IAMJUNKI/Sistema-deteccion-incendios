"""
Script de Ejecución del Experimento Avanzado de Ensamble y Características Físicas.

Carga el dataset maestro de Galicia (2019-2023), añade VPD, Nesterov Index y Ratios,
entrena LightGBM, XGBoost y CatBoost, y genera el Ensamble Blending (Rank Averaging).
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.models.train_ensemble_advanced import train_and_evaluate_advanced_ensemble


def run_advanced_experiment():
    print("======================================================================")
    print("🚀 EJECUTANDO EXPERIMENTO TABULAR AVANZADO + ENSAMBLE BLENDING")
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

    feature_cols = [
        "tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia",
        "prec_acum_7d", "prec_acum_30d", "tmax_media_7d",
        "alerta_30_30", "altitud_media", "pendiente_media",
        "orientacion_media", "combustible_pct_forestal",
        "mes", "dia_semana", "es_finde"
    ]

    df_results = train_and_evaluate_advanced_ensemble(
        df_master=df_all,
        feature_cols=feature_cols,
        target_col="target",
        date_col="fecha",
        train_years=[2019, 2020, 2021, 2022],
        test_year=2023,
        negative_ratio=50
    )

    print("\n======================================================================")
    print("📊 RESULTADOS DEL EXPERIMENTO TABULAR AVANZADO (EVALUACIÓN AÑO 2023)")
    print("======================================================================")
    print(df_results.to_string(index=False))

    output_path = Path("docs/technical/advanced_ensemble_results_2023.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n💾 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    run_advanced_experiment()
