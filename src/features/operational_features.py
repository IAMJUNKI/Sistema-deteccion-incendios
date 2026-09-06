"""Features diarias para inferencia operativa T+1/T+2/T+3.

Las funciones de este módulo reciben un forecast horario normalizado y un
histórico diario observado/reanalizado. La fecha de emisión es la frontera
temporal: nunca se consultan observaciones posteriores a ella. Para T+2 y T+3,
los días intermedios del forecast sí pueden alimentar las memorias acumuladas,
igual que ocurriría en producción.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.canonical_contract import (
    CANONICAL_FEATURE_SCHEMA_VERSION,
    CANONICAL_FEATURES,
)

GALICIA_TZ = ZoneInfo("Europe/Madrid")
CRITICAL_START_HOUR = 12
CRITICAL_END_HOUR = 18
RAIN_THRESHOLD_MM = 1.0

OPERATIONAL_FEATURES = [
    "tmax_vc",
    "rhmin_vc",
    "vmax_vc",
    "prec_dia",
    "prec_acum_3d",
    "prec_acum_7d",
    "prec_acum_14d",
    "prec_acum_30d",
    "tmax_media_7d",
    "rhmin_media_7d",
    "vmax_media_7d",
    "dias_sin_lluvia",
    "vpd_vc",
    "alerta_30_30",
    "altitud_media",
    "pendiente_media",
    "orientacion_media",
    "combustible_pct_forestal",
    "mes",
    "dia_semana",
    "es_finde",
    "dia_anio_sin",
    "dia_anio_cos",
]

BASE_WEATHER_COLUMNS = [
    "tmax_vc",
    "rhmin_vc",
    "vmax_vc",
    "prec_dia",
]

DAILY_CANONICAL_WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "temperature_max_12_18h",
    "relative_humidity_mean",
    "relative_humidity_min",
    "relative_humidity_min_12_18h",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_speed_max_12_18h",
    "vpd_mean",
    "vpd_max_12_18h",
    "precipitation_sum",
]


def calculate_vpd(temperature_c: pd.Series | np.ndarray, rh_pct: pd.Series | np.ndarray) -> np.ndarray:
    """Calcula el déficit de presión de vapor en kPa."""

    temperature = np.asarray(temperature_c, dtype=float)
    humidity = np.asarray(rh_pct, dtype=float)
    saturation_pressure = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    return saturation_pressure * (1.0 - np.clip(humidity, 0.0, 100.0) / 100.0)


def _as_local_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(GALICIA_TZ)


def aggregate_hourly_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el forecast horario a variables diarias por celda.

    La ventana crítica es inclusiva: 12:00, 13:00, ..., 18:00 hora local.
    ``prec_dia`` es la suma de las horas disponibles del día; la validación de
    cobertura completa se realiza antes de esta función en producción.
    """

    required = {
        "cell_id",
        "valid_time",
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
    }
    missing = required.difference(forecast_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas horarias: {sorted(missing)}")
    if forecast_df.empty:
        return pd.DataFrame()

    hourly = forecast_df.copy()
    hourly["valid_time"] = pd.to_datetime(hourly["valid_time"], utc=True, errors="coerce")
    if hourly["valid_time"].isna().any():
        raise ValueError("El forecast contiene timestamps inválidos.")
    hourly["local_time"] = hourly["valid_time"].dt.tz_convert(GALICIA_TZ)
    hourly["fecha"] = hourly["local_time"].dt.normalize().dt.tz_localize(None)
    hourly["local_hour"] = hourly["local_time"].dt.hour
    hourly["vpd_hourly"] = calculate_vpd(
        hourly["temperature_c"], hourly["relative_humidity_pct"]
    )

    keys = ["cell_id", "fecha"]
    critical = hourly[hourly["local_hour"].between(CRITICAL_START_HOUR, CRITICAL_END_HOUR)]
    if critical.empty:
        return pd.DataFrame()
    daily = hourly.groupby(keys, as_index=False).agg(
        temperature_mean=("temperature_c", "mean"),
        temperature_min=("temperature_c", "min"),
        temperature_max=("temperature_c", "max"),
        relative_humidity_mean=("relative_humidity_pct", "mean"),
        relative_humidity_min=("relative_humidity_pct", "min"),
        wind_speed_mean=("wind_speed_kmh", "mean"),
        wind_speed_max=("wind_speed_kmh", "max"),
        vpd_mean=("vpd_hourly", "mean"),
    )
    critical_daily = critical.groupby(keys, as_index=False).agg(
        temperature_max_12_18h=("temperature_c", "max"),
        relative_humidity_min_12_18h=("relative_humidity_pct", "min"),
        wind_speed_max_12_18h=("wind_speed_kmh", "max"),
        vpd_max_12_18h=("vpd_hourly", "max"),
    )
    daily = daily.merge(critical_daily, on=keys, validate="one_to_one")
    precipitation = hourly.groupby(keys, as_index=False)["precipitation_mm"].sum(
        min_count=1
    )
    daily = daily.merge(
        precipitation.rename(columns={"precipitation_mm": "precipitation_sum"}),
        on=keys,
        validate="one_to_one",
    )
    # Aliases del contrato operativo antiguo. Se conservan para que los
    # artefactos de rollback puedan convivir durante la migración a EGIF.
    daily["tmax_vc"] = daily["temperature_max_12_18h"]
    daily["rhmin_vc"] = daily["relative_humidity_min_12_18h"]
    daily["vmax_vc"] = daily["wind_speed_max_12_18h"]
    daily["prec_dia"] = daily["precipitation_sum"]
    daily["vpd_vc"] = daily["vpd_max_12_18h"]
    for column in (
        "forecast_run_at",
        "downloaded_at",
        "forecast_run_source",
        "forecast_quality",
        "forecast_selection",
        "forecast_api_version",
        "forecast_grid",
        "forecast_query_resolution_km",
        "forecast_requested_points",
        "provider",
        "model",
        "model_version",
        "grid",
    ):
        if column in hourly.columns:
            metadata = hourly.groupby(keys, as_index=False)[column].first()
            daily = daily.merge(metadata, on=keys, how="left")
    return daily.sort_values(keys).reset_index(drop=True)


def _normalise_history(history_df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres del histórico ERA5/observacional."""

    aliases = {
        "fecha": "fecha",
        "date": "fecha",
        "tmax": "tmax_vc",
        "rh_min": "rhmin_vc",
        "rhmin": "rhmin_vc",
        "wind_speed": "vmax_vc",
        "vmax": "vmax_vc",
        "precip": "prec_dia",
        "precip_mm": "prec_dia",
    }
    history = history_df.copy()
    for old, new in aliases.items():
        if old in history.columns and new not in history.columns:
            history = history.rename(columns={old: new})

    # La tienda histórica anterior solo conserva agregados de la ventana
    # crítica. Se proyectan como proxy para mantener el rollback operativo;
    # las nuevas ingestas conservan las estadísticas horarias reales.
    canonical_aliases = {
        "temperature_max": "tmax_vc",
        "temperature_max_12_18h": "tmax_vc",
        "temperature_mean": "tmax_vc",
        "temperature_min": "tmax_vc",
        "relative_humidity_min": "rhmin_vc",
        "relative_humidity_min_12_18h": "rhmin_vc",
        "relative_humidity_mean": "rhmin_vc",
        "wind_speed_max": "vmax_vc",
        "wind_speed_max_12_18h": "vmax_vc",
        "wind_speed_mean": "vmax_vc",
        "precipitation_sum": "prec_dia",
    }
    for canonical, legacy in canonical_aliases.items():
        if canonical not in history.columns and legacy in history.columns:
            history[canonical] = history[legacy]
    legacy_aliases = {
        "tmax_vc": "temperature_max_12_18h",
        "rhmin_vc": "relative_humidity_min_12_18h",
        "vmax_vc": "wind_speed_max_12_18h",
        "prec_dia": "precipitation_sum",
    }
    for legacy, canonical in legacy_aliases.items():
        if legacy not in history.columns and canonical in history.columns:
            history[legacy] = history[canonical]
    if "vpd_mean" not in history.columns:
        history["vpd_mean"] = calculate_vpd(
            history["temperature_mean"], history["relative_humidity_mean"]
        )
    if "vpd_max_12_18h" not in history.columns:
        history["vpd_max_12_18h"] = calculate_vpd(
            history["temperature_max_12_18h"], history["relative_humidity_min_12_18h"]
        )

    required = {
        "cell_id",
        "fecha",
        *BASE_WEATHER_COLUMNS,
        *DAILY_CANONICAL_WEATHER_COLUMNS,
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Faltan columnas históricas: {sorted(missing)}")
    history["fecha"] = pd.to_datetime(history["fecha"], errors="coerce").dt.tz_localize(None)
    return history.dropna(subset=["cell_id", "fecha"]).sort_values(["cell_id", "fecha"])


def _window_values(
    combined_cell: pd.DataFrame,
    target_date: pd.Timestamp,
    column: str,
    window_days: int,
) -> pd.Series:
    start = target_date - pd.Timedelta(days=window_days)
    values = combined_cell[
        (combined_cell["fecha"] >= start) & (combined_cell["fecha"] < target_date)
    ][column]
    return values


def _dry_streak(combined_cell: pd.DataFrame, target_date: pd.Timestamp) -> int:
    """Cuenta días secos consecutivos antes del día objetivo."""

    previous = combined_cell[combined_cell["fecha"] < target_date].sort_values("fecha", ascending=False)
    streak = 0
    for value in previous["prec_dia"].tolist():
        if pd.isna(value):
            break
        if float(value) < RAIN_THRESHOLD_MM:
            streak += 1
        else:
            break
    return streak


def _add_static_and_calendar(features: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    if grid_df is not None and not grid_df.empty:
        static_columns = [
            column
            for column in grid_df.columns
            if column not in {"geometry", "cell_id"} and column not in features.columns
        ]
        if static_columns:
            features = features.merge(grid_df[["cell_id", *static_columns]], on="cell_id", how="left")

    dates = pd.to_datetime(features["fecha"])
    features["mes"] = dates.dt.month.astype("int8")
    features["dia_semana"] = dates.dt.dayofweek.astype("int8")
    features["es_finde"] = dates.dt.dayofweek.isin([5, 6]).astype("int8")
    features["dia_anio_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25).astype("float32")
    features["dia_anio_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25).astype("float32")
    return features


def build_operational_features(
    forecast_hourly: pd.DataFrame,
    history_daily: pd.DataFrame,
    grid_df: pd.DataFrame,
    *,
    issue_time: datetime | pd.Timestamp | str,
    horizons: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Construye features operativas para los horizontes diarios pedidos.

    ``history_daily`` es un agregado diario y solo se usa hasta el día local
    anterior al issue time. Incluir la fila del día de emisión implicaría usar
    observaciones posteriores a una ejecución de las 05:00.
    Para un horizonte superior a uno, los días forecast intermedios participan
    en las ventanas de memoria, pero nunca se mezclan datos posteriores al día
    objetivo con sus features.
    """

    issue = pd.Timestamp(issue_time)
    if issue.tzinfo is None:
        issue = issue.tz_localize(GALICIA_TZ)
    issue_local = issue.tz_convert(GALICIA_TZ)
    issue_date = issue_local.normalize().tz_localize(None)

    daily_forecast = aggregate_hourly_forecast(forecast_hourly)
    if daily_forecast.empty:
        raise ValueError("No se pudieron generar features diarias del forecast.")
    daily_forecast["fecha"] = pd.to_datetime(daily_forecast["fecha"]).dt.tz_localize(None)
    daily_forecast = daily_forecast[daily_forecast["fecha"] > issue_date].copy()
    if daily_forecast.empty:
        raise ValueError("El forecast no contiene días posteriores a la emisión.")

    history = _normalise_history(history_daily)
    history = history[history["fecha"] < issue_date].copy()
    # El forecast diario y el histórico tienen un contrato común. Los nombres
    # de origen se conservan en columnas separadas para trazabilidad.
    weather_columns = [
        "cell_id",
        "fecha",
        *BASE_WEATHER_COLUMNS,
        *DAILY_CANONICAL_WEATHER_COLUMNS,
    ]
    history_weather = history[weather_columns].copy()
    forecast_weather = daily_forecast[weather_columns].copy()
    combined = pd.concat([history_weather, forecast_weather], ignore_index=True)
    combined = (
        combined.drop_duplicates(["cell_id", "fecha"], keep="last")
        .sort_values(["cell_id", "fecha"])
        .reset_index(drop=True)
    )

    horizon_set = tuple(sorted(set(int(h) for h in horizons)))
    if not horizon_set:
        raise ValueError("Debe solicitarse al menos un horizonte operativo.")

    # Las ventanas de memoria se calculan de una vez para todas las celdas.
    # La implementación anterior filtraba el histórico por celda y por
    # horizonte (92.091 filtros para la rejilla completa), lo que hacía que
    # una inferencia pareciese bloqueada durante muchos minutos. ``closed``
    # evita incluir el propio día objetivo y reproduce la semántica temporal
    # de _window_values sin usar observaciones futuras.
    rolling_source = combined.set_index("fecha")
    rolling_features: dict[str, pd.Series] = {}
    for name, column, window_days, aggregation in (
        ("prec_acum_3d", "precipitation_sum", 3, "sum"),
        ("prec_acum_7d", "precipitation_sum", 7, "sum"),
        ("prec_acum_14d", "precipitation_sum", 14, "sum"),
        ("prec_acum_30d", "precipitation_sum", 30, "sum"),
        ("tmax_media_7d", "temperature_max", 7, "mean"),
        ("rhmin_media_7d", "relative_humidity_min", 7, "mean"),
        ("vmax_media_7d", "wind_speed_max", 7, "mean"),
        ("precipitation_sum_3d", "precipitation_sum", 3, "sum"),
        ("precipitation_sum_7d", "precipitation_sum", 7, "sum"),
        ("precipitation_sum_14d", "precipitation_sum", 14, "sum"),
        ("precipitation_sum_30d", "precipitation_sum", 30, "sum"),
        ("temperature_mean_7d", "temperature_mean", 7, "mean"),
        ("relative_humidity_mean_7d", "relative_humidity_mean", 7, "mean"),
        ("wind_speed_mean_7d", "wind_speed_mean", 7, "mean"),
        ("relative_humidity_mean_14d", "relative_humidity_mean", 14, "mean"),
    ):
        rolling_features[name] = (
            rolling_source.groupby("cell_id", sort=False)[column]
            .rolling(
                f"{window_days}D",
                closed="left",
                min_periods=1,
            )
            .agg(aggregation)
            .rename(name)
        )
    memory_features = pd.concat(rolling_features, axis=1).reset_index()

    # La racha seca se expresa como segmentos de filas secas separados por
    # lluvia o por un valor faltante. El desplazamiento final excluye el día
    # objetivo, igual que el bucle histórico original.
    dry = combined["precipitation_sum"].notna() & combined["precipitation_sum"].lt(RAIN_THRESHOLD_MM)
    dry_segments = (~dry).groupby(combined["cell_id"], sort=False).cumsum()
    dry_streak = dry.astype("int64").groupby(
        [combined["cell_id"], dry_segments], sort=False
    ).cumsum()
    combined["dias_sin_lluvia"] = (
        dry_streak.groupby(combined["cell_id"], sort=False)
        .shift(1)
        .fillna(0)
        .astype("float64")
    )
    memory_features = memory_features.merge(
        combined[["cell_id", "fecha", "dias_sin_lluvia"]],
        on=["cell_id", "fecha"],
        how="left",
        validate="one_to_one",
    )

    target_dates = [issue_date + pd.Timedelta(days=horizon) for horizon in horizon_set]
    targets = daily_forecast[daily_forecast["fecha"].isin(target_dates)].copy()
    targets["horizon_days"] = (targets["fecha"] - issue_date).dt.days.astype("int16")
    missing_horizons = set(horizon_set).difference(targets["horizon_days"].unique())
    if missing_horizons:
        missing = ", ".join(f"T+{h}" for h in sorted(missing_horizons))
        raise ValueError(f"El forecast no contiene los días: {missing}")

    features = targets.merge(
        memory_features,
        on=["cell_id", "fecha"],
        how="left",
        validate="one_to_one",
    )
    features["issue_time"] = issue.tz_convert("UTC")
    features["issue_date_local"] = issue_date
    features["source_type"] = "forecast"
    features["alerta_30_30"] = (
        (features["temperature_max_12_18h"] >= 30)
        & (features["relative_humidity_min_12_18h"] <= 30)
    ).astype("int8")
    features["consecutive_dry_days"] = features["dias_sin_lluvia"]
    features = features.sort_values(["horizon_days", "cell_id"]).reset_index(drop=True)
    features = _add_static_and_calendar(features, grid_df)
    for column in OPERATIONAL_FEATURES + list(CANONICAL_FEATURES):
        if column not in features.columns:
            features[column] = np.nan
    features["feature_schema_version"] = CANONICAL_FEATURE_SCHEMA_VERSION
    return features


def build_historical_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla de features históricas sin duplicarla por horizonte.

    El cálculo es común a T+1, T+2 y T+3 en el benchmark actual. Separarlo de
    ``build_historical_horizon_dataset`` permite entrenar por años y mantener
    el consumo de memoria acotado sin cambiar las ventanas temporales.
    """

    data = _normalise_history(daily_df)
    data = data.sort_values(["cell_id", "fecha"]).reset_index(drop=True)
    grouped = data.groupby("cell_id", group_keys=False)
    for window in (3, 7, 14, 30):
        data[f"prec_acum_{window}d"] = grouped["prec_dia"].transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).sum()
        )
    data["tmax_media_7d"] = grouped["tmax_vc"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    data["rhmin_media_7d"] = grouped["rhmin_vc"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    data["vmax_media_7d"] = grouped["vmax_vc"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    # Memorias con los nombres del contrato EGIF. Se calculan a partir de la
    # meteorología previa al día de la fila, incluso si el datacubo ya traía
    # una columna con el mismo nombre. Así la construcción histórica y la
    # inferencia operativa tienen exactamente la misma semántica temporal.
    data["precipitation_sum_3d"] = grouped["precipitation_sum"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).sum()
    )
    data["precipitation_sum_7d"] = grouped["precipitation_sum"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).sum()
    )
    data["precipitation_sum_14d"] = grouped["precipitation_sum"].transform(
        lambda values: values.shift(1).rolling(14, min_periods=1).sum()
    )
    data["precipitation_sum_30d"] = grouped["precipitation_sum"].transform(
        lambda values: values.shift(1).rolling(30, min_periods=1).sum()
    )
    data["temperature_mean_7d"] = grouped["temperature_mean"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    data["relative_humidity_mean_7d"] = grouped["relative_humidity_mean"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    data["relative_humidity_mean_14d"] = grouped["relative_humidity_mean"].transform(
        lambda values: values.shift(1).rolling(14, min_periods=1).mean()
    )
    data["wind_speed_mean_7d"] = grouped["wind_speed_mean"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    data["vpd_vc"] = calculate_vpd(data["tmax_vc"], data["rhmin_vc"])
    data["vpd_mean"] = calculate_vpd(data["temperature_mean"], data["relative_humidity_mean"])
    data["vpd_max_12_18h"] = calculate_vpd(
        data["temperature_max_12_18h"], data["relative_humidity_min_12_18h"]
    )
    data["alerta_30_30"] = ((data["tmax_vc"] >= 30) & (data["rhmin_vc"] <= 30)).astype("int8")

    # El contador se calcula sobre valores previos, nunca sobre el propio día.
    data["dias_sin_lluvia"] = 0.0
    for _, indices in data.groupby("cell_id", sort=False).groups.items():
        previous = data.loc[indices, "prec_dia"].shift(1)
        streak = []
        counter = 0
        for value in previous:
            counter = counter + 1 if pd.notna(value) and value < RAIN_THRESHOLD_MM else 0
            streak.append(counter)
        data.loc[indices, "dias_sin_lluvia"] = streak
    data["consecutive_dry_days"] = data["dias_sin_lluvia"]

    dates = pd.to_datetime(data["fecha"])
    data["mes"] = dates.dt.month.astype("int8")
    data["dia_semana"] = dates.dt.dayofweek.astype("int8")
    data["es_finde"] = dates.dt.dayofweek.isin([5, 6]).astype("int8")
    data["dia_anio_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    data["dia_anio_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)

    return data


def build_historical_horizon_dataset(
    daily_df: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 2, 3),
) -> dict[int, pd.DataFrame]:
    """Construye el benchmark ERA5-perfect con semántica de día objetivo.

    Cada fila contiene la meteorología del propio ``target_date`` y sus
    memorias previas. La fecha de emisión se reconstruye como
    ``target_date - horizon``. Es deliberadamente un escenario perfecto: en
    producción esa meteorología será un forecast y no se dispone de históricos
    de vintages de MeteoGalicia para medir todavía la degradación del tiempo.
    """

    is_canonical = "target_ignicion" in daily_df.columns
    data = build_historical_features(daily_df)
    if is_canonical:
        data["target_ignicion"] = pd.to_numeric(
            data["target_ignicion"], errors="coerce"
        ).astype("Int8")

    outputs: dict[int, pd.DataFrame] = {}
    for horizon in sorted(set(int(h) for h in horizons)):
        if is_canonical:
            # La fila representa el día objetivo. En el benchmark ERA5-perfect
            # sus variables se conocen retrospectivamente; la inferencia
            # operativa sustituirá exactamente esa fila por el forecast del día
            # objetivo y conservará las memorias observadas/previstas previas.
            output = data.copy()
            output["target_date"] = output["fecha"]
            output["issue_date"] = output["target_date"] - pd.Timedelta(days=horizon)
            output["target"] = output["target_ignicion"].astype("int8")
            output["target_ignicion"] = output["target"]
            output[f"target_t{horizon}"] = output["target"]
            output["target_alignment"] = "target_day_features_issue_date_minus_horizon"
            output["weather_benchmark"] = "era5_perfect_benchmark"
        else:
            output = data.copy()
            output["horizon_days"] = horizon
            output["issue_date"] = output["fecha"] - pd.Timedelta(days=horizon)
            output["target_date"] = output["fecha"]
            output[f"target_t{horizon}"] = output["target"].astype("int8")
        output["source_type"] = "era5_perfect"
        output["horizon_days"] = horizon
        output["feature_schema_version"] = (
            CANONICAL_FEATURE_SCHEMA_VERSION if is_canonical else "operational-risk-v1"
        )
        outputs[horizon] = output
    return outputs


def ensure_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Devuelve una matriz numérica estable para el modelo serializado."""

    columns = list(feature_columns or OPERATIONAL_FEATURES)
    matrix = df.copy()
    for column in columns:
        if column not in matrix.columns:
            matrix[column] = 0.0
    matrix = matrix[columns].copy()
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
