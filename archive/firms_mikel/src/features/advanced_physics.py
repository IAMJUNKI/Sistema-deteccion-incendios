"""
Modulo de Ingeniería de Características Físicas Avanzadas e Índices Meteorológicos de Riesgo.

Calcula:
1. VPD (Vapor Pressure Deficit - Deficiencia de Presion de Vapor)
2. Nesterov Index (Indice de Nesterov acumulado)
3. Ratio de Aceleración Térmico-Eólica (Tmax * Vwind / RHmin)
"""

import numpy as np
import pandas as pd


def compute_vapor_pressure_deficit(tmax_c: pd.Series | np.ndarray, rhmin_pct: pd.Series | np.ndarray) -> np.ndarray:
    """
    Calcula el Deficit de Presion de Vapor (VPD en kPa).
    e_s = 0.61078 * exp((17.27 * T) / (T + 237.3))
    VPD = e_s * (1 - RH / 100)
    """
    tmax_c = np.asarray(tmax_c, dtype=np.float64)
    rhmin_pct = np.clip(np.asarray(rhmin_pct, dtype=np.float64), 1.0, 100.0)

    e_s = 0.61078 * np.exp((17.27 * tmax_c) / (tmax_c + 237.3))
    vpd = e_s * (1.0 - (rhmin_pct / 100.0))
    return np.clip(vpd, 0.0, None)


def compute_nesterov_index(df: pd.DataFrame, cell_col: str = "cell_id", date_col: str = "fecha") -> pd.Series:
    """
    Calcula el Indice Nesterov diario acumulando (Tmax - Tdew) * Tmax en dias sin lluvia (P < 5mm).
    Se reinicia a 0 cuando la lluvia acumulada supera los 5mm.
    """
    df = df.sort_values(by=[cell_col, date_col]).reset_index(drop=True)

    tmax = df["tmax_vc"].values
    rhmin = np.clip(df["rhmin_vc"].values, 1.0, 100.0)
    precip = df["prec_dia"].values

    # Aproximacion de temperatura de punto de rocio (Tdew) por formula de Magnus
    a, b = 17.27, 237.3
    alpha = ((a * tmax) / (b + tmax)) + np.log(rhmin / 100.0)
    tdew = (b * alpha) / (a - alpha)

    diff_t = np.clip(tmax - tdew, 0.0, None) * np.clip(tmax, 0.0, None)

    nesterov = np.zeros(len(df), dtype=np.float64)
    current_val = 0.0
    prev_cell = None

    for i in range(len(df)):
        cell = df.at[i, cell_col]
        if cell != prev_cell:
            current_val = 0.0
            prev_cell = cell

        if precip[i] >= 5.0:
            current_val = 0.0
        else:
            current_val += diff_t[i]

        nesterov[i] = current_val

    return pd.Series(nesterov, index=df.index)


def compute_thermal_wind_ratio(tmax_c: pd.Series | np.ndarray, vmax_kmh: pd.Series | np.ndarray, rhmin_pct: pd.Series | np.ndarray) -> np.ndarray:
    """
    Calcula el Ratio de Aceleración Térmico-Eólica: (Tmax * Vwind) / max(RHmin, 5.0)
    """
    tmax = np.asarray(tmax_c, dtype=np.float64)
    vmax = np.asarray(vmax_kmh, dtype=np.float64)
    rh = np.clip(np.asarray(rhmin_pct, dtype=np.float64), 5.0, 100.0)

    ratio = (tmax * vmax) / rh
    return ratio


def add_advanced_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade todas las características físicas avanzadas al DataFrame.
    """
    df = df.copy()

    tmax_col = "tmax_vc" if "tmax_vc" in df.columns else "tmax"
    rhmin_col = "rhmin_vc" if "rhmin_vc" in df.columns else "rh_min"
    vmax_col = "vmax_vc" if "vmax_vc" in df.columns else "wind_speed"

    if tmax_col in df.columns and rhmin_col in df.columns:
        df["vpd_t1"] = compute_vapor_pressure_deficit(df[tmax_col], df[rhmin_col])

    if tmax_col in df.columns and vmax_col in df.columns and rhmin_col in df.columns:
        df["ratio_termico_eolico"] = compute_thermal_wind_ratio(df[tmax_col], df[vmax_col], df[rhmin_col])

    if tmax_col in df.columns and rhmin_col in df.columns and "prec_dia" in df.columns:
        df["nesterov_index"] = compute_nesterov_index(df)

    return df


if __name__ == "__main__":
    print("Módulo advanced_physics listo para importar.")
