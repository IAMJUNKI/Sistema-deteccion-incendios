"""
Modulo de Transformacion Temporal y Prevencion de Fuga de Datos (Anti-Data Leakage).

Aplica el shift temporal de 24 horas (T-1) a las variables meteorológicas explicativas
y calcula las ventanas moviles de memoria climatica (7 dias y 30 dias).
"""

import pandas as pd


def apply_temporal_shift(df: pd.DataFrame, weather_cols: list[str], time_col: str = "fecha", cell_col: str = "cell_id") -> pd.DataFrame:
    """
    Desplaza las variables meteorológicas un día hacia atrás (T-1) dentro de cada celda.
    Garantiza que la probabilidad de ignición del día T se prediga únicamente con información
    antecedente del día T-1.
    """
    df = df.sort_values(by=[cell_col, time_col]).reset_index(drop=True)
    
    shifted_cols = {}
    for col in weather_cols:
        if col in df.columns:
            shifted_cols[f"{col}_t1"] = df.groupby(cell_col)[col].shift(1)
            
    df_shifted = pd.concat([df, pd.DataFrame(shifted_cols, index=df.index)], axis=1)
    return df_shifted


def compute_climate_memory(df: pd.DataFrame, cell_col: str = "cell_id", time_col: str = "fecha") -> pd.DataFrame:
    """
    Calcula ventanas móviles de memoria climática acumulada:
    - Precipitacion acumulada a 7 dias (prec_acum_7d)
    - Precipitacion acumulada a 30 dias (prec_acum_30d)
    - Temperatura maxima media a 7 dias (tmax_media_7d)
    """
    df = df.sort_values(by=[cell_col, time_col]).reset_index(drop=True)

    if "precip" in df.columns:
        df["prec_acum_7d"] = df.groupby(cell_col)["precip"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
        df["prec_acum_30d"] = df.groupby(cell_col)["precip"].transform(lambda x: x.shift(1).rolling(30, min_periods=1).sum())

    if "tmax" in df.columns:
        df["tmax_media_7d"] = df.groupby(cell_col)["tmax"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())

    return df


if __name__ == "__main__":
    print("Módulo temporal_shift listo para importar.")
