"""Cargador de datos resiliente y optimizado para el dashboard de inferencia."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from src.models.forecast_risk_model import load_horizon_model
from src.webapp.utils.geo_helpers import assign_approx_province, assign_comarca_or_distrito

DEFAULT_CANONICAL_GRID_PATH = "data/processed/grid/galicia_grid_1km_egif.parquet"
LEGACY_GRID_PATHS = (
    "data/processed/grid/galicia_grid_1km_2018.parquet",
    "data/processed/grid/galicia_grid_1km_2012.parquet",
)


@st.cache_data(ttl=300)
def list_available_inference_datasets() -> dict[str, str]:
    """Lista los archivos de inferencia disponibles en data/processed/."""
    files_map = {}

    # Archivo operativo primario
    primary_path = Path(
        os.getenv("PREDICTIONS_OUTPUT_PATH", "data/processed/predicciones_operativas.parquet")
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
def load_operational_predictions(selected_file: str | None = None) -> pd.DataFrame:
    """Carga el conjunto de predicciones operativo o el dataset seleccionado."""
    if selected_file and Path(selected_file).exists():
        path = Path(selected_file)
    else:
        path = Path(
            os.getenv("PREDICTIONS_OUTPUT_PATH", "data/processed/predicciones_operativas.parquet")
        )
        if not path.exists():
            # Fallback automático al parquet más reciente disponible
            candidates = sorted(glob.glob("data/processed/inferencia_*.parquet"), reverse=True)
            if candidates:
                path = Path(candidates[0])
            else:
                return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
        return enrich_dataset_metadata(df)
    except Exception:
        return pd.DataFrame()


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

    # Asignar provincia y distrito comarcal si no existen
    if "provincia" not in df.columns and "lat_centroid" in df.columns and "lon_centroid" in df.columns:
        df["provincia"] = [
            assign_approx_province(lat, lon) if pd.notna(lat) and pd.notna(lon) else "Ourense"
            for lat, lon in zip(df["lat_centroid"], df["lon_centroid"])
        ]
    elif "provincia" not in df.columns:
        df["provincia"] = "Ourense"

    if "distrito_forestal" not in df.columns and "lat_centroid" in df.columns and "lon_centroid" in df.columns:
        df["distrito_forestal"] = [
            assign_comarca_or_distrito(lat, lon) if pd.notna(lat) and pd.notna(lon) else "Galicia Sur"
            for lat, lon in zip(df["lat_centroid"], df["lon_centroid"])
        ]
    elif "distrito_forestal" not in df.columns:
        df["distrito_forestal"] = "Galicia Sur"

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

    # Acción recomendada
    if "recommended_action" not in df.columns:
        def _get_action(pct):
            if pct >= 0.995:
                return "preposicion_helitransportada"
            elif pct >= 0.95:
                return "vigilancia_aerea_reforzada"
            elif pct >= 0.80:
                return "patrullaje_terrestre_preventivo"
            return "monitoreo_rutinario"
        df["recommended_action"] = df["percentil_riesgo"].apply(_get_action)

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
