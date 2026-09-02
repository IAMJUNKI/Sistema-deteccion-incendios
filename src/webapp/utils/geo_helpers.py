"""Utilidades geoespaciales y presets territoriales de Galicia."""

from __future__ import annotations

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
import streamlit as st

# Presets de navegación territorial en Galicia
ZOOM_PRESETS = {
    "Galicia Completa": {"center": [42.60, -7.85], "zoom": 8},
    "Ourense Sur — Monterrei": {"center": [41.95, -7.50], "zoom": 10},
    "Macizo Central — Trevinca": {"center": [42.25, -7.15], "zoom": 10},
    "O Ribeiro — Carballiño": {"center": [42.35, -8.15], "zoom": 10},
    "Rías Baixas — Pontevedra": {"center": [42.40, -8.60], "zoom": 9},
    "Costa da Morte — Barbanza": {"center": [42.80, -9.00], "zoom": 9},
    "Lugo Interior — Terra Chá": {"center": [43.15, -7.60], "zoom": 9},
    "Mariña Lucense": {"center": [43.55, -7.45], "zoom": 9},
}

# Municipios de referencia de Galicia
CONCELLOS_GALICIA = {
    "Ourense": {"lat": 42.3358, "lon": -7.8639, "provincia": "Ourense"},
    "Verín": {"lat": 41.9408, "lon": -7.4378, "provincia": "Ourense"},
    "Xinzo de Limia": {"lat": 42.0633, "lon": -7.7236, "provincia": "Ourense"},
    "A Gudiña": {"lat": 42.0617, "lon": -7.1378, "provincia": "Ourense"},
    "O Barco de Valdeorras": {"lat": 42.4167, "lon": -6.9833, "provincia": "Ourense"},
    "A Pobra de Trives": {"lat": 42.3417, "lon": -7.2583, "provincia": "Ourense"},
    "Ribadavia": {"lat": 42.2872, "lon": -8.1436, "provincia": "Ourense"},
    "Allariz": {"lat": 42.1906, "lon": -7.8028, "provincia": "Ourense"},
    "Celanova": {"lat": 42.1528, "lon": -7.9575, "provincia": "Ourense"},
    "Santiago de Compostela": {"lat": 42.8782, "lon": -8.5448, "provincia": "A Coruña"},
    "A Coruña": {"lat": 43.3623, "lon": -8.4115, "provincia": "A Coruña"},
    "Ferrol": {"lat": 43.4832, "lon": -8.2369, "provincia": "A Coruña"},
    "Carballo": {"lat": 43.2131, "lon": -8.6911, "provincia": "A Coruña"},
    "Ribeira": {"lat": 42.5539, "lon": -8.9931, "provincia": "A Coruña"},
    "Noia": {"lat": 42.7847, "lon": -8.8872, "provincia": "A Coruña"},
    "Muros": {"lat": 42.7758, "lon": -9.0578, "provincia": "A Coruña"},
    "Lugo": {"lat": 43.0097, "lon": -7.5568, "provincia": "Lugo"},
    "Monforte de Lemos": {"lat": 42.5217, "lon": -7.5142, "provincia": "Lugo"},
    "Sarria": {"lat": 42.7806, "lon": -7.4147, "provincia": "Lugo"},
    "Vilalba": {"lat": 43.2981, "lon": -7.6811, "provincia": "Lugo"},
    "Viveiro": {"lat": 43.6628, "lon": -7.5956, "provincia": "Lugo"},
    "Ribadeo": {"lat": 43.5361, "lon": -7.0408, "provincia": "Lugo"},
    "Pontevedra": {"lat": 42.4310, "lon": -8.6444, "provincia": "Pontevedra"},
    "Vigo": {"lat": 42.2406, "lon": -8.7207, "provincia": "Pontevedra"},
    "Vilagarcía de Arousa": {"lat": 42.5969, "lon": -8.7636, "provincia": "Pontevedra"},
    "Lalín": {"lat": 42.6617, "lon": -8.1136, "provincia": "Pontevedra"},
    "A Estrada": {"lat": 42.6889, "lon": -8.4897, "provincia": "Pontevedra"},
    "Ponteareas": {"lat": 42.1764, "lon": -8.5042, "provincia": "Pontevedra"},
    "Tui": {"lat": 42.0461, "lon": -8.6447, "provincia": "Pontevedra"},
}


@st.cache_data(ttl=3600)
def load_galicia_focus_layers() -> tuple[dict | None, dict | None]:
    """Carga la máscara inversa optimizada para atenuar el resto del país de forma ultrarrápida."""
    configured_path = os.getenv("GALICIA_BOUNDARY_PATH", "").strip()
    candidates = [
        Path(configured_path) if configured_path else None,
        Path("data/external/galicia_boundary.geojson"),
        Path("data/raw/igm/galicia_boundary.geojson"),
    ]
    boundary_path = next((path for path in candidates if path and path.exists()), None)
    if boundary_path is None:
        return None, None
    try:
        gdf = gpd.read_file(boundary_path)
        if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        if hasattr(gdf.geometry, "union_all"):
            galicia_geom = gdf.geometry.union_all()
        else:
            galicia_geom = gdf.unary_union

        # Simplificar suavemente el contorno para reducir el payload del GeoJSON al navegador
        galicia_geom_simplified = galicia_geom.simplify(0.002, preserve_topology=True)

        # Polígono exterior que cubre el entorno ibérico y europeo circundante
        outer_box = box(-22.0, 30.0, 8.0, 49.0)
        mask_geom = outer_box.difference(galicia_geom_simplified)

        return mask_geom.__geo_interface__, galicia_geom_simplified.__geo_interface__
    except Exception:
        return None, None


def build_complex_dissolved_shapes(df_subset: pd.DataFrame) -> gpd.GeoDataFrame:
    """Construye geometrías disueltas continuas optimizadas para celdas de igual nivel de riesgo."""
    if df_subset.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    has_real_geometry = (
        "geometry" in df_subset.columns
        and df_subset["geometry"].notna().all()
        and df_subset["geometry"].map(
            lambda value: hasattr(value, "__geo_interface__")
        ).all()
    )

    if has_real_geometry:
        gdf = gpd.GeoDataFrame(df_subset.copy(), geometry="geometry", crs="EPSG:4326")
        unified_geom = unary_union(gdf.geometry)
    else:
        dlat = 0.00449
        dlon = 0.00615
        geoms = [
            box(
                row["lon_centroid"] - dlon,
                row["lat_centroid"] - dlat,
                row["lon_centroid"] + dlon,
                row["lat_centroid"] + dlat,
            )
            for _, row in df_subset.iterrows()
            if pd.notna(row.get("lon_centroid")) and pd.notna(row.get("lat_centroid"))
        ]
        if not geoms:
            return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        unified_geom = unary_union(geoms)

    return gpd.GeoDataFrame(geometry=[unified_geom], crs="EPSG:4326")


def assign_approx_province(lat: float, lon: float) -> str:
    """Asigna la provincia gallega correspondiente por coordenadas aproximadas."""
    if lat < 42.45:
        if lon > -8.25:
            return "Ourense"
        return "Pontevedra"
    else:
        if lon > -7.95:
            return "Lugo"
        return "A Coruña"


def assign_comarca_or_distrito(lat: float, lon: float) -> str:
    """Asigna un distrito forestal / comarca representativa según coordenadas."""
    if lat < 42.15 and lon > -7.65:
        return "Distrito XIV — Verín - Viana"
    elif lat < 42.45 and lon > -7.50:
        return "Distrito XV — A Limia / Valdeorras"
    elif lat < 42.45 and lon > -8.20:
        return "Distrito XI — O Ribeiro - Arenteiro"
    elif lat < 42.50 and lon <= -8.20:
        return "Distrito XIX — Caldas - O Salnés"
    elif lat >= 42.45 and lat < 42.90 and lon <= -8.30:
        return "Distrito IV — Barbanza"
    elif lat >= 42.90 and lon <= -8.30:
        return "Distrito V — Bergantiños - Mariñas"
    elif lat >= 42.45 and lat < 43.10 and lon > -7.90:
        return "Distrito VIII — Terra de Lemos"
    elif lat >= 43.10 and lon > -7.90:
        return "Distrito VII — A Fonsagrada - Os Ancares"
    return "Galicia Central / Deza"
