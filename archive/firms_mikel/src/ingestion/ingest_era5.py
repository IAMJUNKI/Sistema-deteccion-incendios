"""
Modulo de Ingesta y Procesamiento de Meteorologia ERA5-Land (Reanalisis Copernicus).

Carga la meteorologia diaria downscaled a la rejilla de 1km de Galicia (temperatura maxima,
temperatura media, humedad relativa, velocidad del viento, precipitacion diaria).
"""

from pathlib import Path
import pandas as pd


def load_era5_daily(parquet_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset meteorologico diario ERA5-Land en formato Parquet.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo ERA5 en: {path}")

    df_era5 = pd.read_parquet(path)
    required_cols = {"cell_id", "fecha"}
    if not required_cols.issubset(df_era5.columns):
        raise ValueError(f"El dataset ERA5 debe contener las columnas: {required_cols}")

    df_era5["fecha"] = pd.to_datetime(df_era5["fecha"]).dt.strftime("%Y-%m-%d")
    return df_era5


def compute_fwi_proxy(df_era5: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un proxy simplificado del Fire Weather Index (FWI) basado en la regla de negocio
    y combinacion de temperatura alta, baja humedad y ráfagas de viento.
    """
    df = df_era5.copy()
    if "tmax" in df.columns and "rh_min" in df.columns:
        # Flag de alerta 30-30-30 (T > 30ºC, RH < 30%)
        df["alerta_30_30"] = ((df["tmax"] >= 30.0) & (df["rh_min"] <= 30.0)).astype(int)
    else:
        df["alerta_30_30"] = 0
    return df


if __name__ == "__main__":
    print("Módulo ingest_era5 listo para importar.")
