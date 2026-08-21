"""
Modulo de Ingesta Operativa de Previsiones de MeteoGalicia y Quantile Mapping (Producción 2025+).

Ingiere las previsiones numéricas del modelo WRF de MeteoGalicia (API / JSON) y aplica
un filtrado no paramétrico de Quantile Mapping respecto a la climatología histórica de ERA5-Land
para eliminar el sesgo de distribución (Domain Shift).
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np


def parse_meteogalicia_json(json_path: str | Path) -> pd.DataFrame:
    """
    Parsea el archivo JSON de observaciones/previsiones diarias de MeteoGalicia.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo JSON de MeteoGalicia en: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Si es una lista directa de diccionarios de estaciones/celdas
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and "features" in data:
        records = [feat["properties"] for feat in data["features"]]
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame([data])

    return df


def apply_quantile_mapping(
    df_forecast: pd.DataFrame,
    df_climatology_era5: pd.DataFrame,
    variable_cols: list[str]
) -> pd.DataFrame:
    """
    Aplica el mapeo empírico de cuantiles (Quantile Mapping) para igualar la distribución
    de la previsión operativa de MeteoGalicia con la distribución climática de entrenamiento ERA5.
    
    X_era5 = Q_era5( Q_meteogalicia_inv( X_meteogalicia ) )
    """
    print("📌 Aplicando Quantile Mapping para eliminar el Domain Shift MeteoGalicia -> ERA5...")
    df_adjusted = df_forecast.copy()

    for col in variable_cols:
        if col in df_forecast.columns and col in df_climatology_era5.columns:
            source_vals = df_forecast[col].dropna().values
            ref_vals = df_climatology_era5[col].dropna().values

            if len(source_vals) > 0 and len(ref_vals) > 0:
                # Calcular percentiles de la previsión de MeteoGalicia
                percentiles = pd.Series(source_vals).rank(pct=True).values
                # Interpolar los valores correspondientes en la distribución de ERA5
                adjusted_vals = np.percentile(ref_vals, percentiles * 100)
                df_adjusted[col] = adjusted_vals

    return df_adjusted


if __name__ == "__main__":
    print("Módulo ingest_meteogalicia listo para importar.")
