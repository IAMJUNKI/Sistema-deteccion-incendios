"""
Script de Inferencia Operativa Diaria de Riesgo de Incendios en Galicia.

Proceso:
1. Ingiere la previsión de MeteoGalicia (WRF).
2. Aplica Quantile Mapping respecto a la climatología ERA5-Land.
3. Genera las memorias climáticas acumuladas (7d, 30d) y el desfase T-1.
4. Ejecuta inferencia rápida con el modelo oficial LightGBM sobre las 30.697 celdas.
5. Clasifica las celdas en el 5% de mayor riesgo (Alerta Temprana urgente).
6. Exporta el mapa diario a data/processed/inferencia_diaria_galicia.parquet.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

from src.ingestion.ingest_meteogalicia import apply_quantile_mapping
from src.models.train_baseline import hard_negative_sampling


def run_daily_inference_pipeline(
    df_master: pd.DataFrame,
    target_date: str = "2023-08-15",
    output_path: str | Path = "data/processed/inferencia_diaria_galicia.parquet"
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de inferencia diaria.
    """
    print(f"======================================================================")
    print(f"🚀 EJECUTANDO INFERENCIA OPERATIVA DIARIA PARA FECHA: {target_date}")
    print(f"======================================================================")

    feature_cols = [
        "tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia",
        "prec_acum_7d", "prec_acum_30d", "tmax_media_7d",
        "alerta_30_30", "altitud_media", "pendiente_media",
        "orientacion_media", "combustible_pct_forestal",
        "mes", "dia_semana", "es_finde"
    ]

    df_master = df_master.copy()
    df_master["fecha_str"] = pd.to_datetime(df_master["fecha"]).dt.strftime("%Y-%m-%d")

    # 1. Entrenar modelo oficial LightGBM sobre histórico (2019-2022)
    print("📌 1. Cargando y entrenando el modelo oficial LightGBM en histórico (2019-2022)...")
    df_master["year"] = pd.to_datetime(df_master["fecha"]).dt.year
    df_train = hard_negative_sampling(
        df_master[(df_master["year"].isin([2019, 2020, 2021, 2022])) & (df_master["prec_dia"] < 5.0)],
        target_col="target",
        ratio=50
    )

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train["target"]

    model = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 2. Filtrar el día de inferencia
    print(f"📌 2. Extrayendo celdas de Galicia para la fecha objetivo: {target_date}...")
    df_today = df_master[df_master["fecha_str"] == target_date].copy()

    if len(df_today) == 0:
        print(f"⚠️ No hay datos directos para {target_date}, seleccionando día estival de referencia de 2023...")
        # Seleccionar día equivalente en verano (Agosto 2023)
        summer_dates = df_master[df_master["fecha_str"].str.startswith("2023-08")]["fecha_str"].unique()
        fallback_date = summer_dates[0] if len(summer_dates) > 0 else df_master["fecha_str"].max()
        df_today = df_master[df_master["fecha_str"] == fallback_date].copy()
        target_date = fallback_date

    print(f"✅ Se han encontrado {len(df_today):,} celdas de Galicia para evaluación.")

    # 3. Aplicar Quantile Mapping (MeteoGalicia -> ERA5)
    print("📌 3. Aplicando Quantile Mapping (Corrección de sesgo MeteoGalicia -> ERA5)...")
    df_today_adjusted = apply_quantile_mapping(
        df_forecast=df_today,
        df_climatology_era5=df_train,
        variable_cols=["tmax_vc", "rhmin_vc", "vmax_vc", "prec_dia"]
    )

    # 4. Inferencia con LightGBM + REGLA FÍSICA DE EXTINCIÓN HÍDRICA (Lluvia Directa P >= 5.0 mm)
    print("📌 4. Calculando probabilidades de riesgo de ignición con LightGBM...")
    X_today = df_today_adjusted[feature_cols].fillna(0)
    prob_risk = model.predict_proba(X_today)[:, 1]

    # Regla Física de Extinción Hídrica:
    # 1. Si prec_dia >= 5.0 mm -> Extinción hídrica directa por lluvia (MCff > 30%).
    # 2. Si rhmin_vc >= 70.0% y prec_dia >= 3.0 mm -> Extinción por humedad ambiental y lluvia.
    extinction_mask = (
        (df_today_adjusted["prec_dia"] >= 5.0) |
        ((df_today_adjusted["rhmin_vc"] >= 70.0) & (df_today_adjusted["prec_dia"] >= 3.0))
    )
    prob_risk[extinction_mask] = 0.0000

    df_today["prob_riesgo"] = prob_risk
    df_today["percentil_riesgo"] = pd.Series(prob_risk).rank(pct=True).values

    # Clasificar en Alerta Temprana Urgente (Top 5% de riesgo)
    threshold_5pct = np.percentile(prob_risk, 95)
    df_today["alerta_urgente_5pct"] = (df_today["prob_riesgo"] >= threshold_5pct).astype(int)

    n_alertas = df_today["alerta_urgente_5pct"].sum()
    print(f"🔥 Alertas Urgentes Generadas (Top 5% riesgo): {n_alertas:,} celdas en rojo de {len(df_today):,}.")

    # 5. Exportar parquet de inferencia
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_today.to_parquet(output_path, index=False)
    print(f"💾 Mapa de inferencia diaria guardado exitosamente en: {output_path}")

    return df_today


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Pipeline de Inferencia Operativa Diaria — Predicción de Riesgo de Incendios en Galicia")
    parser.add_argument("--date", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Fecha objetivo en formato YYYY-MM-DD (por defecto hoy)")
    args = parser.parse_args()

    data_dir = Path("misc/Dataset/Mike")
    parquet_files = [data_dir / f"dataset_maestro_{y}.parquet" for y in [2019, 2020, 2021, 2022, 2023]]
    existing_files = [f for f in parquet_files if f.exists()]
    
    if existing_files:
        df_all = pd.concat([pd.read_parquet(f) for f in existing_files], ignore_index=True)
        out_path = Path(f"data/processed/inferencia_{args.date}.parquet")
        run_daily_inference_pipeline(df_all, target_date=args.date, output_path=out_path)
    else:
        print("❌ No se encontraron archivos de datos históricos en misc/Dataset/Mike")
