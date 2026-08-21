"""
Pipeline Principal de Construccion del Dataset Maestro Tabular (Parquet 2D).

Combina la rejilla estatica geoespacial (topografía, distancias), la meteorología ERA5
con shift temporal (T-1), memoria climática y la variable objetivo del EGIF MITECO.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.features.temporal_shift import apply_temporal_shift, compute_climate_memory


def assemble_tabular_dataset(
    geo_features_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    target_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensambla el dataset maestro realizando los joins por cell_id y fecha.
    """
    print("📌 1. Calculando memoria climática y aplicando Shift T-1...")
    weather_df = compute_climate_memory(weather_df)
    
    weather_vars = ["tmax", "tmean", "rh_min", "wind_speed", "precip"]
    weather_df = apply_temporal_shift(weather_df, weather_vars)

    print("📌 2. Cruzando variables geoespaciales estáticas...")
    master_df = weather_df.merge(geo_features_df, on="cell_id", how="left")

    print("📌 3. Asignando la variable objetivo (Target EGIF)...")
    if not target_df.empty:
        master_df = master_df.merge(target_df, on=["cell_id", "fecha"], how="left")
        master_df["target_ignicion"] = master_df["target_ignicion"].fillna(0).astype(int)
    else:
        master_df["target_ignicion"] = 0

    print("📌 4. Generando variables cíclicas temporales...")
    dates = pd.to_datetime(master_df["fecha"])
    master_df["mes"] = dates.dt.month
    master_df["dia_semana"] = dates.dt.dayofweek
    master_df["es_finde"] = master_df["dia_semana"].isin([5, 6]).astype(int)
    master_df["dia_año_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    master_df["dia_año_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)

    return master_df


if __name__ == "__main__":
    print("Módulo build_tabular_dataset listo para importar.")
