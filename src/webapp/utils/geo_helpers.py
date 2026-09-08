"""Utilidades geoespaciales y presets territoriales de Galicia."""

from __future__ import annotations

import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.affinity import scale
from shapely.geometry import Point, box
from shapely.ops import unary_union

# Directorio de recursos vectoriales integrados en el paquete webapp
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# Presets de navegación territorial en Galicia
ZOOM_PRESETS = {
    "Galicia Completa": {
        "center": [42.60, -7.85],
        "zoom": 8,
    },
    "Ourense Sur — Monterrei": {
        "center": [41.95, -7.50],
        "zoom": 10,
        "bounds": [[41.78, -7.95], [42.18, -7.05]],
    },
    "Macizo Central — Trevinca": {
        "center": [42.25, -7.15],
        "zoom": 10,
        "bounds": [[42.02, -7.55], [42.50, -6.75]],
    },
    "O Ribeiro — Carballiño": {
        "center": [42.35, -8.15],
        "zoom": 10,
        "bounds": [[42.15, -8.45], [42.58, -7.85]],
    },
    "Rías Baixas — Pontevedra": {
        "center": [42.40, -8.60],
        "zoom": 9,
        "bounds": [[42.00, -8.95], [42.70, -8.30]],
    },
    "Costa da Morte — Barbanza": {
        "center": [42.80, -9.00],
        "zoom": 9,
        "bounds": [[42.45, -9.35], [43.15, -8.60]],
    },
    "Lugo Interior — Terra Chá": {
        "center": [43.15, -7.60],
        "zoom": 9,
        "bounds": [[42.80, -7.95], [43.45, -7.25]],
    },
    "Mariña Lucense": {
        "center": [43.55, -7.45],
        "zoom": 9,
        "bounds": [[43.35, -7.80], [43.78, -7.00]],
    },
}

# Asignación de distritos forestales canónicos del PLADIGA por sector territorial
SECTOR_PLADIGA_DISTRITOS = {
    "Ourense Sur — Monterrei": [
        "Distrito XIV — Verín - Viana",
        "Distrito XV — A Limia",
        "Distrito X — Terra de Celanova - Baixa Limia",
    ],
    "Macizo Central — Trevinca": [
        "Distrito XIII — Valdeorras - Trives",
        "Distrito VIII — Terra de Lemos",
    ],
    "O Ribeiro — Carballiño": [
        "Distrito XI — O Ribeiro - Arenteiro",
        "Distrito XII — Miño - Arnoia",
    ],
    "Rías Baixas — Pontevedra": [
        "Distrito XVIII — Vigo - Baixo Miño",
        "Distrito XVII — O Condado - Paradanta",
        "Distrito XIX — Caldas - O Salnés",
    ],
    "Costa da Morte — Barbanza": [
        "Distrito IV — Barbanza",
        "Distrito V — Fisterra",
        "Distrito II — Bergantiños - As Mariñas",
    ],
    "Lugo Interior — Terra Chá": [
        "Distrito IX — Lugo - Sarria",
        "Distrito VII — A Fonsagrada - Os Ancares",
        "Distrito XVI — Deza - Tabeirós",
    ],
    "Mariña Lucense": [
        "Distrito VI — A Mariña - Terra Chá",
        "Distrito I — Ferrol",
    ],
}

# Municipios de referencia de Galicia (Cabeceras comarcales y concellos de alta relevancia operativa)
CONCELLOS_GALICIA = {
    # Ourense
    "Ourense": {"lat": 42.3358, "lon": -7.8639, "provincia": "Ourense"},
    "Verín": {"lat": 41.9408, "lon": -7.4378, "provincia": "Ourense"},
    "Xinzo de Limia": {"lat": 42.0633, "lon": -7.7236, "provincia": "Ourense"},
    "A Gudiña": {"lat": 42.0617, "lon": -7.1378, "provincia": "Ourense"},
    "O Barco de Valdeorras": {"lat": 42.4167, "lon": -6.9833, "provincia": "Ourense"},
    "A Pobra de Trives": {"lat": 42.3417, "lon": -7.2583, "provincia": "Ourense"},
    "Ribadavia": {"lat": 42.2872, "lon": -8.1436, "provincia": "Ourense"},
    "Allariz": {"lat": 42.1906, "lon": -7.8028, "provincia": "Ourense"},
    "Celanova": {"lat": 42.1528, "lon": -7.9575, "provincia": "Ourense"},
    "O Carballiño": {"lat": 42.4303, "lon": -8.0778, "provincia": "Ourense"},
    "Bande": {"lat": 42.0306, "lon": -7.9753, "provincia": "Ourense"},
    "Viana do Bolo": {"lat": 42.1806, "lon": -7.1147, "provincia": "Ourense"},
    "A Rúa": {"lat": 42.3958, "lon": -7.1158, "provincia": "Ourense"},
    # A Coruña
    "Santiago de Compostela": {"lat": 42.8782, "lon": -8.5448, "provincia": "A Coruña"},
    "A Coruña": {"lat": 43.3623, "lon": -8.4115, "provincia": "A Coruña"},
    "Ferrol": {"lat": 43.4832, "lon": -8.2369, "provincia": "A Coruña"},
    "Carballo": {"lat": 43.2131, "lon": -8.6911, "provincia": "A Coruña"},
    "Ribeira": {"lat": 42.5539, "lon": -8.9931, "provincia": "A Coruña"},
    "Noia": {"lat": 42.7847, "lon": -8.8872, "provincia": "A Coruña"},
    "Muros": {"lat": 42.7758, "lon": -9.0578, "provincia": "A Coruña"},
    "Betanzos": {"lat": 43.2811, "lon": -8.2114, "provincia": "A Coruña"},
    "Fisterra": {"lat": 42.9083, "lon": -9.2639, "provincia": "A Coruña"},
    "Cee": {"lat": 42.9567, "lon": -9.1894, "provincia": "A Coruña"},
    "As Pontes de García Rodríguez": {"lat": 43.4497, "lon": -7.8528, "provincia": "A Coruña"},
    "Melide": {"lat": 42.9150, "lon": -8.0153, "provincia": "A Coruña"},
    "Arzúa": {"lat": 42.9272, "lon": -8.1636, "provincia": "A Coruña"},
    "Ordes": {"lat": 43.0767, "lon": -8.4078, "provincia": "A Coruña"},
    # Lugo
    "Lugo": {"lat": 43.0097, "lon": -7.5568, "provincia": "Lugo"},
    "Monforte de Lemos": {"lat": 42.5217, "lon": -7.5142, "provincia": "Lugo"},
    "Sarria": {"lat": 42.7806, "lon": -7.4147, "provincia": "Lugo"},
    "Vilalba": {"lat": 43.2981, "lon": -7.6811, "provincia": "Lugo"},
    "Viveiro": {"lat": 43.6628, "lon": -7.5956, "provincia": "Lugo"},
    "Ribadeo": {"lat": 43.5361, "lon": -7.0408, "provincia": "Lugo"},
    "Chantada": {"lat": 42.6089, "lon": -7.7686, "provincia": "Lugo"},
    "Quiroga": {"lat": 42.4758, "lon": -7.2725, "provincia": "Lugo"},
    "A Fonsagrada": {"lat": 43.1256, "lon": -7.0689, "provincia": "Lugo"},
    "Becerreá": {"lat": 42.8542, "lon": -7.1611, "provincia": "Lugo"},
    "Mondoñedo": {"lat": 43.4286, "lon": -7.3628, "provincia": "Lugo"},
    # Pontevedra
    "Pontevedra": {"lat": 42.4310, "lon": -8.6444, "provincia": "Pontevedra"},
    "Vigo": {"lat": 42.2406, "lon": -8.7207, "provincia": "Pontevedra"},
    "Vilagarcía de Arousa": {"lat": 42.5969, "lon": -8.7636, "provincia": "Pontevedra"},
    "Lalín": {"lat": 42.6617, "lon": -8.1136, "provincia": "Pontevedra"},
    "A Estrada": {"lat": 42.6889, "lon": -8.4897, "provincia": "Pontevedra"},
    "Ponteareas": {"lat": 42.1764, "lon": -8.5042, "provincia": "Pontevedra"},
    "Tui": {"lat": 42.0461, "lon": -8.6447, "provincia": "Pontevedra"},
    "Redondela": {"lat": 42.2831, "lon": -8.6086, "provincia": "Pontevedra"},
    "Cangas do Morrazo": {"lat": 42.2644, "lon": -8.7836, "provincia": "Pontevedra"},
    "Sanxenxo": {"lat": 42.4003, "lon": -8.8078, "provincia": "Pontevedra"},
    "Cambados": {"lat": 42.5147, "lon": -8.8147, "provincia": "Pontevedra"},
    "Caldas de Reis": {"lat": 42.6042, "lon": -8.6417, "provincia": "Pontevedra"},
    "Silleda": {"lat": 42.6975, "lon": -8.2483, "provincia": "Pontevedra"},
    "A Guarda": {"lat": 41.9014, "lon": -8.8744, "provincia": "Pontevedra"},
}


@st.cache_data(ttl=3600)
def load_galicia_focus_layers() -> tuple[dict | None, dict | None]:
    """Carga la máscara inversa optimizada para atenuar el resto del país de forma ultrarrápida."""
    configured_path = os.getenv("GALICIA_BOUNDARY_PATH", "").strip()
    candidates = [
        Path(configured_path) if configured_path else None,
        ASSETS_DIR / "galicia_boundary.geojson",
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


@st.cache_data(ttl=3600)
def load_galicia_sectors_geojson() -> dict | None:
    """Carga las geometrías orgánicas oficiales de los sectores territoriales de Galicia (PLADIGA)."""
    candidates = [
        ASSETS_DIR / "galicia_sectors.geojson",
        Path("data/external/galicia_sectors.geojson"),
        Path("data/processed/galicia_sectors.geojson"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        import json

        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=3600)

def build_concello_spotlight_mask(lat: float, lon: float, radius_km: float = 9.0) -> dict | None:
    """Construye una máscara que atenúa el entorno exterior dejando el concello en foco luminoso."""
    try:
        deg_lat = radius_km / 111.0
        deg_lon = radius_km / 81.7

        circle = Point(lon, lat).buffer(deg_lat)
        scaled_circle = scale(circle, xfact=deg_lat / deg_lon, yfact=1.0, origin=(lon, lat))
        outer_box = box(-22.0, 30.0, 8.0, 49.0)
        spotlight = outer_box.difference(scaled_circle)
        return spotlight.__geo_interface__
    except Exception:
        return None


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


# Centros de referencia de los 19 Distritos Forestales del PLADIGA (Xunta de Galicia)
DISTRITOS_PLADIGA_CENTROIDES = {
    "Distrito I — Ferrol": (43.52, -8.05),
    "Distrito II — Bergantiños - As Mariñas": (43.26, -8.45),
    "Distrito III — Santiago - Meseta Interior": (42.92, -8.40),
    "Distrito IV — Barbanza": (42.68, -8.90),
    "Distrito V — Fisterra": (43.05, -9.05),
    "Distrito VI — A Mariña - Terra Chá": (43.40, -7.45),
    "Distrito VII — A Fonsagrada - Os Ancares": (43.05, -7.05),
    "Distrito VIII — Terra de Lemos": (42.50, -7.45),
    "Distrito IX — Lugo - Sarria": (42.90, -7.55),
    "Distrito X — Terra de Celanova - Baixa Limia": (42.05, -8.05),
    "Distrito XI — O Ribeiro - Arenteiro": (42.35, -8.20),
    "Distrito XII — Miño - Arnoia": (42.30, -7.75),
    "Distrito XIII — Valdeorras - Trives": (42.38, -7.10),
    "Distrito XIV — Verín - Viana": (41.98, -7.35),
    "Distrito XV — A Limia": (42.08, -7.65),
    "Distrito XVI — Deza - Tabeirós": (42.66, -8.30),
    "Distrito XVII — O Condado - Paradanta": (42.18, -8.45),
    "Distrito XVIII — Vigo - Baixo Miño": (42.15, -8.70),
    "Distrito XIX — Caldas - O Salnés": (42.54, -8.70),
}


@st.cache_data(ttl=3600)
def _load_province_polygons():
    """Carga los polígonos oficiales de las 4 provincias gallegas."""
    candidates = [
        ASSETS_DIR / "galicia_provinces.geojson",
        Path("data/external/galicia_provinces.geojson"),
        Path("data/raw/igm/galicia_provinces.geojson"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                gdf = gpd.read_file(candidate)
                return [(str(row["provincia"]), row["geometry"]) for _, row in gdf.iterrows()]
            except Exception:
                pass
    return None


def assign_approx_province(lat: float, lon: float) -> str:
    """Asigna la provincia gallega correspondiente con exactitud geométrica oficial."""
    provinces = _load_province_polygons()
    if provinces:
        p = Point(lon, lat)
        for name, geom in provinces:
            if geom.contains(p):
                return name
        # Si cae en aguas costeras o justo en la frontera, asignar al polígono más cercano
        return min(provinces, key=lambda item: item[1].distance(p))[0]

    # Fallback con cortes ajustados a la morfología real gallega
    if lat < 42.45:
        if lon > -8.15:
            return "Ourense"
        return "Pontevedra"
    else:
        if lon > -7.95:
            return "Lugo"
        if lat < 42.75 and lon < -8.0 and lon > -8.5:
            return "Pontevedra"
        return "A Coruña"


def assign_comarca_or_distrito(lat: float, lon: float) -> str:
    """Asigna el distrito forestal oficial PLADIGA (I a XIX) más cercano."""
    cos_lat = math.cos(math.radians(42.6))
    best_dist = float("inf")
    best_name = "Distrito XII — Miño - Arnoia"
    for name, (d_lat, d_lon) in DISTRITOS_PLADIGA_CENTROIDES.items():
        dist = (lat - d_lat) ** 2 + ((lon - d_lon) * cos_lat) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name
