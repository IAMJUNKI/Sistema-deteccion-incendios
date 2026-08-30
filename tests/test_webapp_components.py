"""Tests unitarios para los componentes y utilidades del dashboard webapp."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest

from src.webapp.components.concello_lookup import get_risk_level_info
from src.webapp.components.operational_protocols import generate_executive_briefing
from src.webapp.styles import CUSTOM_CSS, apply_custom_styles
from src.webapp.utils.data_loader import enrich_dataset_metadata, load_grid_geometries
from src.webapp.utils.geo_helpers import (
    assign_approx_province,
    assign_comarca_or_distrito,
    build_complex_dissolved_shapes,
    load_galicia_focus_layers,
)


def test_assign_approx_province():
    # Sur-este de Galicia -> Ourense
    assert assign_approx_province(42.0, -7.5) == "Ourense"
    # Sur-oeste de Galicia -> Pontevedra
    assert assign_approx_province(42.2, -8.6) == "Pontevedra"
    # Nor-este de Galicia -> Lugo
    assert assign_approx_province(43.2, -7.5) == "Lugo"
    # Nor-oeste de Galicia -> A Coruña
    assert assign_approx_province(43.3, -8.5) == "A Coruña"


def test_assign_comarca_or_distrito():
    distrito = assign_comarca_or_distrito(42.0, -7.4)
    assert "Verín" in distrito or "Distrito" in distrito


def test_get_risk_level_info():
    extremo = get_risk_level_info(0.15)
    assert "Nivel 5" in extremo["level"]
    assert extremo["badge_color"] == "#800026"

    muy_alto = get_risk_level_info(0.08)
    assert "Nivel 4" in muy_alto["level"]

    bajo = get_risk_level_info(0.002)
    assert "Nivel 1" in bajo["level"]


def test_enrich_dataset_metadata_with_coordinates():
    raw_df = pd.DataFrame(
        {
            "cell_id": [1, 2, 3],
            "lat_centroid": [42.0, 42.5, 43.0],
            "lon_centroid": [-7.5, -8.5, -8.0],
            "tmax_vc": [35.0, 22.0, 18.0],
            "rhmin_vc": [20.0, 60.0, 75.0],
            "vmax_vc": [35.0, 10.0, 5.0],
            "prob_riesgo": [0.15, 0.03, 0.005],
        }
    )

    enriched = enrich_dataset_metadata(raw_df)

    assert "provincia" in enriched.columns
    assert "distrito_forestal" in enriched.columns
    assert "regla_30_30_activa" in enriched.columns
    assert "recommended_action" in enriched.columns
    assert "percentil_riesgo" in enriched.columns

    # Primera celda cumple la regla 30-30-30 (T=35, RH=20, V=35)
    assert enriched.loc[0, "regla_30_30_activa"] == True
    # Segunda celda no la cumple
    assert enriched.loc[1, "regla_30_30_activa"] == False


def test_enrich_dataset_metadata_without_coordinates_resilience():
    raw_df = pd.DataFrame(
        {
            "cell_id": [6818, 6819],
            "tmax_vc": [32.0, 25.0],
            "rhmin_vc": [28.0, 45.0],
            "vmax_vc": [31.0, 12.0],
            "prob_risk": [0.12, 0.02],
        }
    )

    enriched = enrich_dataset_metadata(raw_df)
    assert "prob_riesgo" in enriched.columns
    assert "percentil_riesgo" in enriched.columns
    assert "regla_30_30_activa" in enriched.columns
    assert "recommended_action" in enriched.columns


def test_build_complex_dissolved_shapes():
    df = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lat_centroid": [42.0, 42.01],
            "lon_centroid": [-7.5, -7.51],
        }
    )
    gdf = build_complex_dissolved_shapes(df)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0] is not None


def test_load_galicia_focus_layers():
    mask_json, boundary_json = load_galicia_focus_layers()
    if mask_json is not None:
        assert "type" in mask_json
        assert "coordinates" in mask_json
    if boundary_json is not None:
        assert "type" in boundary_json
        assert "coordinates" in boundary_json


def test_generate_executive_briefing():
    df = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "provincia": ["Ourense", "Lugo"],
            "distrito_forestal": ["Distrito XIV", "Distrito VII"],
            "prob_riesgo": [0.18, 0.02],
            "percentil_riesgo": [0.999, 0.50],
            "regla_30_30_activa": [True, False],
            "recommended_action": ["preposicion_helitransportada", "monitoreo_rutinario"],
            "issue_time": ["2026-08-30T05:00:00Z", "2026-08-30T05:00:00Z"],
        }
    )
    manifest = {"forecast_provider": "MeteoGalicia WRF 1km"}

    briefing = generate_executive_briefing(
        df,
        manifest,
        selected_horizon=1,
        target_date="2026-08-31",
    )

    assert "INFORME EJECUTIVO DE SITUACIÓN OPERATIVA" in briefing
    assert "MeteoGalicia WRF 1km" in briefing
    assert "2026-08-31" in briefing
    assert "PLADIGA" in briefing


def test_custom_css_structure():
    assert ".command-header" in CUSTOM_CSS
    assert ".kpi-card" in CUSTOM_CSS
    assert "material-symbols-outlined" in CUSTOM_CSS
