"""Tests unitarios para los componentes y utilidades del dashboard webapp."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest

from src.webapp.components.concello_lookup import get_risk_level_info
from src.webapp.components.operational_protocols import generate_executive_briefing
from src.webapp.styles import CUSTOM_CSS
from src.webapp.utils.data_loader import enrich_dataset_metadata
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
    assert bool(enriched.loc[0, "regla_30_30_activa"]) is True
    # Segunda celda no la cumple
    assert bool(enriched.loc[1, "regla_30_30_activa"]) is False


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


def test_enrich_dataset_canonical_egif_48():
    raw_df = pd.DataFrame(
        {
            "cell_id": [101, 102],
            "lat_centroid": [42.0, 43.1],
            "lon_centroid": [-7.5, -8.0],
            "temperature_max_12_18h": [34.5, 21.0],
            "relative_humidity_min_12_18h": [22.0, 65.0],
            "wind_speed_max_12_18h": [32.0, 14.0],
            "precipitation_sum_30d": [2.5, 45.0],
            "slope_mean": [16.5, 4.2],
            "coniferous_forest": [0.6, 0.0],
            "broadleaf_forest": [0.2, 0.1],
            "scrub": [0.1, 0.8],
            "prob_riesgo": [0.14, 0.01],
        }
    )

    enriched = enrich_dataset_metadata(raw_df)

    # Aliases normalizados automáticamente
    assert enriched.loc[0, "tmax_vc"] == 34.5
    assert enriched.loc[0, "rhmin_vc"] == 22.0
    assert enriched.loc[0, "vmax_vc"] == 32.0
    assert enriched.loc[0, "prec_acum_30d"] == 2.5
    assert enriched.loc[0, "pendiente_media"] == 16.5

    # Regla 30-30-30 activa en celda 0 (T=34.5, HR=22.0, V=32.0)
    assert bool(enriched.loc[0, "regla_30_30_activa"]) is True
    assert bool(enriched.loc[1, "regla_30_30_activa"]) is False

    # Síntesis de cobertura vegetal
    assert enriched.loc[0, "combustible_pct_forestal"] == pytest.approx(80.0)
    assert enriched.loc[0, "combustible_clase"] == "Pinar / Coníferas"
    assert enriched.loc[1, "combustible_clase"] == "Matorral / Brezal"


def test_get_color_gradient_modes():
    from src.webapp.components.map_view import get_color_gradient

    # Modo Absoluto P(Y=1)
    color_abs_extremo = get_color_gradient(0.15, 0.999, "Riesgo Absoluto Calibrado P(Y=1)")
    assert color_abs_extremo == "#800026"

    color_abs_bajo = get_color_gradient(0.001, 0.10, "Riesgo Absoluto Calibrado P(Y=1)")
    assert color_abs_bajo == "#FED976"

    # Modo Relativo Percentil
    color_rel_top = get_color_gradient(0.001, 0.999, "Priorización Relativa por Percentil (%)")
    assert color_rel_top == "#800026"

    color_rel_muy_alto = get_color_gradient(0.001, 0.996, "Priorización Relativa por Percentil (%)")
    assert color_rel_muy_alto == "#BD0026"

    color_rel_low = get_color_gradient(0.15, 0.50, "Priorización Relativa por Percentil (%)")
    assert color_rel_low == "#FED976"


def test_build_map_legend_html_modes():
    from src.webapp.components.map_view import build_map_legend_html

    legend_abs = build_map_legend_html("Riesgo Absoluto Calibrado P(Y=1)")
    assert "Probabilidad P(Y=1)" in legend_abs
    assert "12.0%" in legend_abs
    assert "Severidad física calibrada" in legend_abs

    legend_rel = build_map_legend_html("Priorización Relativa por Percentil (%)")
    assert "Priorización Relativa" in legend_rel
    assert "Top 0.5%" in legend_rel
    assert "Ranking relativo para despacho" in legend_rel

    legend_tac = build_map_legend_html("Niveles Tácticos Discretos (Top %)")
    assert "Niveles Tácticos" in legend_tac
    assert "Nivel 5: Crítico" in legend_tac
    assert "Nivel 4: Muy Alto" in legend_tac


def test_get_day_severity_info():
    from src.webapp.components.header_kpis import get_day_severity_info

    # 1.88% debe ser Nivel 2 - Moderado
    mod = get_day_severity_info(1.88)
    assert "Nivel 2" in mod["level"]
    assert mod["card_class"] == "alert-warning"

    # Invierno 0.05% debe ser Nivel 1 - Bajo / Nominal
    bajo = get_day_severity_info(0.05)
    assert "Nivel 1" in bajo["level"]
    assert bajo["card_class"] == "alert-success"

    # 4.5% debe ser Nivel 3 - Alto
    alto = get_day_severity_info(4.5)
    assert "Nivel 3" in alto["level"]
    assert alto["card_class"] == "alert-warning"

    # Verano extremo 15.0% debe ser Nivel 5 - Extremo
    extremo = get_day_severity_info(15.0)
    assert "Nivel 5" in extremo["level"]
    assert extremo["card_class"] == "alert-critical"


def test_pladiga_19_distritos_structure():
    from src.webapp.utils.geo_helpers import DISTRITOS_PLADIGA_CENTROIDES

    # Verificar que los 19 distritos oficiales del PLADIGA están presentes
    assert len(DISTRITOS_PLADIGA_CENTROIDES) == 19
    for name, (lat, lon) in DISTRITOS_PLADIGA_CENTROIDES.items():
        assert 41.5 <= lat <= 44.0
        assert -9.5 <= lon <= -6.5
        assert len(name) > 0


def test_admin_lookup_enrichment_exact_provinces():
    raw_df = pd.DataFrame(
        {
            "cell_id": [325, 45197],
            "temperature_max_12_18h": [31.0, 24.0],
            "relative_humidity_min_12_18h": [25.0, 50.0],
            "wind_speed_max_12_18h": [15.0, 10.0],
            "prob_riesgo": [0.03, 0.005],
        }
    )
    enriched = enrich_dataset_metadata(raw_df)
    assert "provincia" in enriched.columns
    assert "distrito_forestal" in enriched.columns
    assert "alerta_30_30_activa" in enriched.columns
    # Celda 0 tiene T=31, RH=25, V=15 -> Alerta 30-30 activa pero regla 30-30-30 inactiva
    assert bool(enriched.loc[0, "alerta_30_30_activa"]) is True
    assert bool(enriched.loc[0, "regla_30_30_activa"]) is False
    assert bool(enriched.loc[1, "alerta_30_30_activa"]) is False


def test_explain_tree_prediction_contract():
    from src.models.explainability import explain_tree_prediction
    from src.webapp.utils.data_loader import load_dashboard_model

    model = load_dashboard_model(1)
    # Crear una fila representativa
    raw_row = pd.DataFrame(
        {
            "temperature_mean": [26.0],
            "temperature_max_12_18h": [32.5],
            "relative_humidity_mean": [40.0],
            "relative_humidity_min_12_18h": [22.0],
            "wind_speed_mean": [14.0],
            "wind_speed_max_12_18h": [28.0],
            "precipitation_sum_30d": [0.5],
            "slope_mean": [18.0],
        }
    )
    df_shap = explain_tree_prediction(model, raw_row)
    assert isinstance(df_shap, pd.DataFrame)
    assert len(df_shap) == 48
    assert "feature" in df_shap.columns
    assert "contribution" in df_shap.columns
    assert "value" in df_shap.columns

