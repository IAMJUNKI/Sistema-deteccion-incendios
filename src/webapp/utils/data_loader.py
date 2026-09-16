"""Cargador de datos resiliente y optimizado para el dashboard de inferencia."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from src.models.forecast_risk_model import load_horizon_model
from src.webapp.utils.geo_helpers import assign_approx_province, assign_comarca_or_distrito

DEFAULT_CANONICAL_GRID_PATH = "data/processed/grid/galicia_grid_1km_egif.parquet"
DEFAULT_PREDICTIONS_OUTPUT_PATH = "data/processed/predicciones_operativas.parquet"
DEFAULT_DASHBOARD_CACHE_SUFFIX = ".dashboard.parquet"
DASHBOARD_CACHE_VERSION = 1
LEGACY_GRID_PATHS = (
    "data/processed/grid/galicia_grid_1km_2018.parquet",
    "data/processed/grid/galicia_grid_1km_2012.parquet",
)


def _configured_predictions_output_path() -> Path:
    """Devuelve la ruta del artefacto operativo publicado."""
    return Path(
        os.getenv("PREDICTIONS_OUTPUT_PATH", DEFAULT_PREDICTIONS_OUTPUT_PATH)
    )


def _resolve_inference_dataset(selected_file: str | None = None) -> Path | None:
    """Resuelve una ruta de inferencia sin hacer trabajo de enriquecimiento."""
    if selected_file and Path(selected_file).exists():
        return Path(selected_file)

    primary_path = _configured_predictions_output_path()
    if primary_path.exists():
        return primary_path

    candidates = sorted(glob.glob("data/processed/inferencia_*.parquet"), reverse=True)
    return Path(candidates[0]) if candidates else None


def _dashboard_cache_path(source_path: Path) -> Path:
    """Devuelve la ruta del Parquet preparado para el dashboard."""
    configured = os.getenv("PREDICTIONS_DASHBOARD_CACHE_PATH", "").strip()
    primary_path = _configured_predictions_output_path()
    if configured and source_path == primary_path:
        return Path(configured)
    return source_path.with_name(f"{source_path.stem}{DEFAULT_DASHBOARD_CACHE_SUFFIX}")


def _cache_is_fresh(cache_path: Path, source_path: Path) -> bool:
    """Comprueba que el artefacto enriquecido procede del Parquet publicado actual."""
    try:
        return (
            cache_path.exists()
            and cache_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
        )
    except OSError:
        return False


@st.cache_data(ttl=300)
def list_available_inference_datasets() -> dict[str, str]:
    """Lista los archivos de inferencia disponibles en data/processed/."""
    files_map = {}

    # Archivo operativo primario
    primary_path = Path(
        os.getenv("PREDICTIONS_OUTPUT_PATH", DEFAULT_PREDICTIONS_OUTPUT_PATH)
    )
    if primary_path.exists():
        files_map["Predicción Operativa Actual (Producción)"] = str(primary_path)

    # Buscar otros parquets de inferencia en data/processed
    candidates = sorted(glob.glob("data/processed/inferencia_*.parquet"), reverse=True)
    for c in candidates:
        name = Path(c).stem.replace("inferencia_", "Inferencia Histórica ").replace("_", " ")
        files_map[name] = c

    return files_map


@st.cache_data(ttl=300)
def _load_operational_predictions_from_path(
    source_path: str,
    dashboard_cache_path: str,
    source_mtime_ns: int,
    dashboard_cache_mtime_ns: int,
) -> pd.DataFrame:
    """Carga una fuente de predicciones, reutilizando su versión preparada."""
    source = Path(source_path)
    dashboard_cache = Path(dashboard_cache_path)
    try:
        read_path = source
        use_dashboard_cache = _cache_is_fresh(dashboard_cache, source)
        if use_dashboard_cache:
            cached_df = pd.read_parquet(dashboard_cache)
            if (
                "dashboard_cache_version" in cached_df.columns
                and not cached_df.empty
                and cached_df["dashboard_cache_version"].eq(DASHBOARD_CACHE_VERSION).all()
            ):
                return cached_df

        df = pd.read_parquet(read_path)
        return enrich_dataset_metadata(df)
    except Exception:
        return pd.DataFrame()


def load_operational_predictions(selected_file: str | None = None) -> pd.DataFrame:
    """Carga las predicciones usando un artefacto de dashboard precalculado cuando existe.

    La ruta se normaliza antes de entrar en ``st.cache_data``. Así, la llamada
    preliminar con ``None`` y la llamada posterior con el archivo seleccionado
    comparten la misma entrada de caché y no vuelven a enriquecer el Parquet.
    """
    path = _resolve_inference_dataset(selected_file)
    if path is None:
        return pd.DataFrame()

    cache_path = _dashboard_cache_path(path)
    try:
        source_mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return pd.DataFrame()
    try:
        dashboard_cache_mtime_ns = cache_path.stat().st_mtime_ns
    except OSError:
        dashboard_cache_mtime_ns = 0

    return _load_operational_predictions_from_path(
        str(path),
        str(cache_path),
        source_mtime_ns,
        dashboard_cache_mtime_ns,
    )


def enrich_dataset_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece el dataframe con columnas de agrupación territorial y métricas de soporte."""
    if df.empty:
        return df

    df = df.copy()

    # Normalizar nombres de columnas de coordenadas si vienen como lat/lon
    if "lat" in df.columns and "lat_centroid" not in df.columns:
        df["lat_centroid"] = df["lat"]
    if "latitude" in df.columns and "lat_centroid" not in df.columns:
        df["lat_centroid"] = df["latitude"]
    if "lon" in df.columns and "lon_centroid" not in df.columns:
        df["lon_centroid"] = df["lon"]
    if "longitude" in df.columns and "lon_centroid" not in df.columns:
        df["lon_centroid"] = df["longitude"]

    # Si faltan lat_centroid o lon_centroid, intentar recuperar de los centroides de la rejilla
    if ("lat_centroid" not in df.columns or "lon_centroid" not in df.columns) and "cell_id" in df.columns:
        centroids_path = Path("data/processed/grid_galicia_centroids.parquet")
        grid_path = Path(os.getenv("GRID_PATH", DEFAULT_CANONICAL_GRID_PATH))
        if centroids_path.exists():
            try:
                cdf = pd.read_parquet(centroids_path)
                merge_cols = [c for c in ["cell_id", "lat_centroid", "lon_centroid"] if c in cdf.columns]
                df = df.merge(cdf[merge_cols], on="cell_id", how="left")
            except Exception:
                pass
        elif grid_path.exists():
            try:
                gdf_tmp = gpd.read_parquet(grid_path)
                if gdf_tmp.crs is not None and str(gdf_tmp.crs) != "EPSG:4326":
                    gdf_tmp = gdf_tmp.to_crs("EPSG:4326")
                if "lat_centroid" not in gdf_tmp.columns and "geometry" in gdf_tmp.columns:
                    gdf_tmp["lat_centroid"] = gdf_tmp.geometry.centroid.y
                    gdf_tmp["lon_centroid"] = gdf_tmp.geometry.centroid.x
                cols_to_use = [c for c in ["cell_id", "lat_centroid", "lon_centroid"] if c in gdf_tmp.columns]
                df = df.merge(gdf_tmp[cols_to_use], on="cell_id", how="left")
            except Exception:
                pass

    # Asegurar horizonte temporal si falta
    if "horizon_days" not in df.columns:
        df["horizon_days"] = 1

    # Asegurar campos temporales
    if "fecha" not in df.columns:
        df["fecha"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    if "issue_time" not in df.columns:
        df["issue_time"] = pd.Timestamp.now().strftime("%Y-%m-%dT05:00:00Z")

    # Asegurar métricas de riesgo
    if "prob_riesgo" not in df.columns and "prob_risk" in df.columns:
        df["prob_riesgo"] = df["prob_risk"]
    elif "prob_riesgo" not in df.columns:
        df["prob_riesgo"] = 0.01

    if "percentil_riesgo" not in df.columns:
        df["percentil_riesgo"] = df["prob_riesgo"].rank(pct=True)

    # Asignar provincia y distrito forestal oficial PLADIGA (I a XIX)
    if "provincia" not in df.columns or "distrito_forestal" not in df.columns:
        egif_admin_path = Path("data/processed/grid/galicia_grid_1km_egif_admin.parquet")
        legacy_admin_path = Path("data/processed/grid_galicia_centroids_admin.parquet")
        admin_df = None
        if "cell_id" in df.columns:
            max_cid = df["cell_id"].max()
            if max_cid > 30696 and egif_admin_path.exists():
                try:
                    admin_df = pd.read_parquet(egif_admin_path)
                except Exception:
                    pass
            elif legacy_admin_path.exists():
                try:
                    admin_df = pd.read_parquet(legacy_admin_path)
                except Exception:
                    pass
            elif egif_admin_path.exists():
                try:
                    admin_df = pd.read_parquet(egif_admin_path)
                except Exception:
                    pass

        if admin_df is not None and "cell_id" in df.columns:
            admin_map = admin_df.drop_duplicates(subset=["cell_id"]).set_index("cell_id")
            if "provincia" not in df.columns and "provincia" in admin_df.columns:
                df["provincia"] = df["cell_id"].map(admin_map["provincia"])
            if "distrito_forestal" not in df.columns and "distrito_forestal" in admin_df.columns:
                df["distrito_forestal"] = df["cell_id"].map(admin_map["distrito_forestal"])

        # Fallback exacto con polígonos oficiales de Galicia
        if "provincia" not in df.columns or df["provincia"].isna().any():
            missing_mask = df["provincia"].isna() if "provincia" in df.columns else pd.Series([True] * len(df))
            if "lat_centroid" in df.columns and "lon_centroid" in df.columns:
                df.loc[missing_mask, "provincia"] = [
                    assign_approx_province(lat, lon) if pd.notna(lat) and pd.notna(lon) else "Ourense"
                    for lat, lon in zip(df.loc[missing_mask, "lat_centroid"], df.loc[missing_mask, "lon_centroid"])
                ]
            else:
                df["provincia"] = "Ourense"

        if "distrito_forestal" not in df.columns or df["distrito_forestal"].isna().any():
            missing_mask = df["distrito_forestal"].isna() if "distrito_forestal" in df.columns else pd.Series([True] * len(df))
            if "lat_centroid" in df.columns and "lon_centroid" in df.columns:
                df.loc[missing_mask, "distrito_forestal"] = [
                    assign_comarca_or_distrito(lat, lon) if pd.notna(lat) and pd.notna(lon) else "Distrito XII — Miño - Arnoia"
                    for lat, lon in zip(df.loc[missing_mask, "lat_centroid"], df.loc[missing_mask, "lon_centroid"])
                ]
            else:
                df["distrito_forestal"] = "Distrito XII — Miño - Arnoia"

    # Normalizar variables meteorológicas y topográficas (canónicas EGIF <-> legacy)
    if "temperature_max_12_18h" in df.columns and "tmax_vc" not in df.columns:
        df["tmax_vc"] = df["temperature_max_12_18h"]
    elif "tmax_vc" in df.columns and "temperature_max_12_18h" not in df.columns:
        df["temperature_max_12_18h"] = df["tmax_vc"]

    if "relative_humidity_min_12_18h" in df.columns and "rhmin_vc" not in df.columns:
        df["rhmin_vc"] = df["relative_humidity_min_12_18h"]
    elif "rhmin_vc" in df.columns and "relative_humidity_min_12_18h" not in df.columns:
        df["relative_humidity_min_12_18h"] = df["rhmin_vc"]

    if "wind_speed_max_12_18h" in df.columns and "vmax_vc" not in df.columns:
        df["vmax_vc"] = df["wind_speed_max_12_18h"]
    elif "vmax_vc" in df.columns and "wind_speed_max_12_18h" not in df.columns:
        df["wind_speed_max_12_18h"] = df["vmax_vc"]

    if "precipitation_sum_30d" in df.columns and "prec_acum_30d" not in df.columns:
        df["prec_acum_30d"] = df["precipitation_sum_30d"]
    elif "prec_acum_30d" in df.columns and "precipitation_sum_30d" not in df.columns:
        df["precipitation_sum_30d"] = df["prec_acum_30d"]

    if "slope_mean" in df.columns and "pendiente_media" not in df.columns:
        df["pendiente_media"] = df["slope_mean"]
    elif "pendiente_media" in df.columns and "slope_mean" not in df.columns:
        df["slope_mean"] = df["pendiente_media"]

    # Sintetizar cobertura de combustible/vegetación a partir de fracciones CORINE
    forest_cols = [c for c in ["broadleaf_forest", "coniferous_forest", "mixed_forest"] if c in df.columns]
    if forest_cols:
        computed_forest = (df[forest_cols].fillna(0.0).sum(axis=1) * 100).clip(0, 100)
        if "combustible_pct_forestal" not in df.columns:
            df["combustible_pct_forestal"] = computed_forest
        else:
            # Si la columna existía pero con NaNs (rellenada como nula en el pipeline de features),
            # reconstruir con la suma real de las fracciones de arbolado de CORINE
            df["combustible_pct_forestal"] = (
                pd.to_numeric(df["combustible_pct_forestal"], errors="coerce")
                .fillna(computed_forest)
                .fillna(0.0)
            )
    else:
        if "combustible_pct_forestal" not in df.columns:
            df["combustible_pct_forestal"] = 0.0
        else:
            df["combustible_pct_forestal"] = pd.to_numeric(df["combustible_pct_forestal"], errors="coerce").fillna(0.0)

    if "combustible_clase" not in df.columns:
        # El orden conserva la prioridad de desempate de la implementación
        # original: matorral, coníferas, frondosas, mixto y agrícola.
        fuel_columns = [
            "scrub",
            "coniferous_forest",
            "broadleaf_forest",
            "mixed_forest",
            "agriculture",
        ]
        fuel_values = (
            df.reindex(columns=fuel_columns, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        max_fuel = fuel_values.max(axis=1)
        dominant_fuel = fuel_values.idxmax(axis=1).map(
            {
                "scrub": "Matorral / Brezal",
                "coniferous_forest": "Pinar / Coníferas",
                "broadleaf_forest": "Frondosas Caducifolias",
                "mixed_forest": "Bosque Mixto",
                "agriculture": "Agrícola / Mosaico",
            }
        )
        df["combustible_clase"] = dominant_fuel.fillna("Matorral / Monte Bajo")
        df.loc[max_fuel <= 0.05, "combustible_clase"] = "Matorral / Monte Bajo"

    # Calcular indicador de la Regla Crítica 30-30-30. El contrato EGIF usa
    # nombres canónicos; los aliases legacy se conservan para rollback.
    tmax = pd.to_numeric(
        df.get(
            "temperature_max_12_18h",
            df.get("tmax_vc", pd.Series([20.0] * len(df))),
        ),
        errors="coerce",
    ).fillna(20.0)
    rhmin = pd.to_numeric(
        df.get(
            "relative_humidity_min_12_18h",
            df.get("rhmin_vc", pd.Series([50.0] * len(df))),
        ),
        errors="coerce",
    ).fillna(50.0)
    vmax = pd.to_numeric(
        df.get(
            "wind_speed_max_12_18h",
            df.get("vmax_vc", pd.Series([15.0] * len(df))),
        ),
        errors="coerce",
    ).fillna(15.0)
    df["regla_30_30_activa"] = (tmax >= 30.0) & (rhmin <= 30.0) & (vmax >= 30.0)
    # Alerta termo-higrométrica 30-30 (desecación crítica del combustible fino)
    if "alerta_30_30" in df.columns:
        df["alerta_30_30_activa"] = df["alerta_30_30"].astype(bool) | ((tmax >= 30.0) & (rhmin <= 30.0))
    else:
        df["alerta_30_30_activa"] = (tmax >= 30.0) & (rhmin <= 30.0)

    # Acción recomendada
    if "recommended_action" not in df.columns:
        percentiles = pd.to_numeric(df["percentil_riesgo"], errors="coerce").fillna(0.0)
        df["recommended_action"] = np.select(
            [
                percentiles >= 0.995,
                percentiles >= 0.95,
                percentiles >= 0.80,
            ],
            [
                "preposicion_helitransportada",
                "vigilancia_aerea_reforzada",
                "patrullaje_terrestre_preventivo",
            ],
            default="monitoreo_rutinario",
        )

    return df


@st.cache_data(ttl=300)
def load_operational_manifest() -> dict:
    """Carga el manifiesto de trazabilidad del pipeline."""
    manifest_path = Path(
        os.getenv(
            "PREDICTIONS_MANIFEST_PATH",
            "data/processed/predicciones_operativas.manifest.json",
        )
    )
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_resource
def load_dashboard_model(horizon: int):
    """Carga el artefacto serializado del modelo LightGBM para el horizonte especificado."""
    return load_horizon_model(
        horizon,
        model_dir=Path(os.getenv("MODEL_DIR", "data/models")),
    )


@st.cache_data(ttl=3600)
def load_grid_geometries() -> pd.DataFrame:
    """Carga las geometrías y centroides reales de las celdas de la rejilla de 1 km."""
    configured_grid = Path(os.getenv("GRID_PATH", DEFAULT_CANONICAL_GRID_PATH))
    candidates = [configured_grid, Path(DEFAULT_CANONICAL_GRID_PATH)] + [
        Path(path) for path in LEGACY_GRID_PATHS
    ]
    grid_path = next(
        (candidate for candidate in dict.fromkeys(candidates) if candidate.exists()),
        None,
    )
    if grid_path is None:
        centroids_path = Path("data/processed/grid_galicia_centroids.parquet")
        if centroids_path.exists():
            try:
                return pd.read_parquet(centroids_path)
            except Exception:
                return pd.DataFrame(columns=["cell_id", "geometry", "lat_centroid", "lon_centroid"])
        return pd.DataFrame(columns=["cell_id", "geometry", "lat_centroid", "lon_centroid"])

    try:
        grid = gpd.read_parquet(grid_path)
        if grid.crs is not None and str(grid.crs) != "EPSG:4326":
            grid = grid.to_crs("EPSG:4326")

        # Asegurar lat_centroid y lon_centroid si faltan
        if "lat_centroid" not in grid.columns and "geometry" in grid.columns:
            grid["lat_centroid"] = grid.geometry.centroid.y
            grid["lon_centroid"] = grid.geometry.centroid.x

        ret_cols = [c for c in ["cell_id", "geometry", "lat_centroid", "lon_centroid"] if c in grid.columns]
        return pd.DataFrame(grid[ret_cols])
    except Exception:
        return pd.DataFrame(columns=["cell_id", "geometry", "lat_centroid", "lon_centroid"])


@st.cache_data(ttl=300)
def load_operational_data_for_horizon(
    selected_file: str | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Carga y fusiona las predicciones con geometrías de celda para un horizonte concreto con caché."""
    predictions = load_operational_predictions(selected_file)
    if predictions.empty:
        return predictions

    if "horizon_days" in predictions.columns:
        df_data = predictions[predictions["horizon_days"] == horizon].copy()
        if df_data.empty:
            df_data = predictions.copy()
    else:
        df_data = predictions.copy()

    geometries = load_grid_geometries()
    if not geometries.empty and "cell_id" in df_data.columns:
        cols_to_merge = [c for c in geometries.columns if c not in df_data.columns or c == "cell_id"]
        if len(cols_to_merge) > 1:
            df_data = df_data.merge(geometries[cols_to_merge], on="cell_id", how="left")

    return df_data
