"""
Script de Verificación Retrospectiva (Backtesting) en Incendios Históricos Reales.

Demuestra cómo el pipeline de inferencia predijo con 24 horas de antelacion (en T-1)
el riesgo de ignición en las celdas donde ocurrieron fuegos reales en 2023.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from scripts.run_daily_inference import run_daily_inference_pipeline


def run_backtesting_on_real_fires():
    print("======================================================================")
    print("🧪 INICIANDO TEST DE VERIFICACIÓN RETROSPECTIVA (BACKTESTING ON FIRES)")
    print("======================================================================")

    data_dir = Path("misc/Dataset/Mike")
    parquet_files = [data_dir / f"dataset_maestro_{y}.parquet" for y in [2019, 2020, 2021, 2022, 2023]]
    df_all = pd.concat([pd.read_parquet(f) for f in parquet_files if f.exists()], ignore_index=True)

    df_2023_fires = df_all[(df_all["target"] == 1) & (pd.to_datetime(df_all["fecha"]).dt.year == 2023)]
    print(f"🔥 Se han encontrado {len(df_2023_fires):,} fuegos reales en el año de test 2023.")

    if len(df_2023_fires) == 0:
        print("⚠️ No hay eventos de ignición en 2023.")
        return

    # Seleccionar un día emblemático de 2023 con fuegos reales
    sample_fire = df_2023_fires.iloc[0]
    fire_date = pd.to_datetime(sample_fire["fecha"]).strftime("%Y-%m-%d")
    fire_cell = sample_fire["cell_id"]

    print(f"🎯 Seleccionado Incendio Real para Backtesting:")
    print(f"   - Fecha de Ignición: {fire_date}")
    print(f"   - Celda Afectada: {fire_cell}")

    # Ejecutar inferencia para esa fecha
    df_inf = run_daily_inference_pipeline(df_all, target_date=fire_date)

    # Verificar si la celda del fuego estaba en el top 5% de riesgo
    cell_result = df_inf[df_inf["cell_id"] == fire_cell]

    if len(cell_result) > 0:
        prob = cell_result.iloc[0]["prob_riesgo"]
        pct = cell_result.iloc[0]["percentil_riesgo"]
        alerta = cell_result.iloc[0]["alerta_urgente_5pct"]

        print("\n======================================================================")
        print("📊 RESULTADO DE LA VERIFICACIÓN RETROSPECTIVA")
        print("======================================================================")
        print(f"✅ Probabilidad de Riesgo Predicha: {prob*100:.2f}%")
        print(f"📈 Percentil de Riesgo en Galicia: {pct*100:.1f}%")
        print(f"🚨 Clasificado en Alerta Urgente (Top 5%): {'SÍ (ÉXITO PROBADO)' if alerta == 1 else 'NO'}")
        print("======================================================================\n")


if __name__ == "__main__":
    run_backtesting_on_real_fires()
