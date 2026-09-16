"""Dashboard Interactivo de Predicción Diaria de Riesgo de Incendios Forestales en Galicia.

Centro de Mando de Alerta Temprana (Emergency Operations Center - EOC) con resolución a 1 km²,
perímetros disueltos GIS, consulta municipal por concellos, diagnóstico en lenguaje claro,
simulador What-If y protocolos operativos de despacho PLADIGA.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.webapp.components.concello_lookup import render_concello_lookup_tab
from src.webapp.components.header_kpis import render_header_and_kpis
from src.webapp.components.map_view import render_map_tab
from src.webapp.components.operational_protocols import render_operational_protocols_tab
from src.webapp.components.shap_simulator import render_shap_and_simulator_tab
from src.webapp.components.sidebar import render_sidebar
from src.webapp.components.system_status import render_system_status_tab
from src.webapp.components.territorial_analytics import render_territorial_analytics_tab
from src.webapp.styles import apply_custom_styles
from src.webapp.utils.data_loader import (
    load_dashboard_model,
    load_operational_data_for_horizon,
    load_operational_manifest,
    load_operational_predictions,
)

# 1. Configuración de página Streamlit
st.set_page_config(
    page_title="Sistema de Alerta Temprana de Incendios — Galicia",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inyección de estilos visuales profesionales con Google Material Symbols
apply_custom_styles()


def main() -> None:
    load_dotenv()
    # 3. Carga preliminar de datos para inicializar el sidebar
    predictions_raw = load_operational_predictions()

    # 4. Renderizar panel lateral
    sidebar_params = render_sidebar(predictions_raw)
    selected_file = sidebar_params["selected_dataset_file"]
    selected_horizon = sidebar_params["selected_horizon"]
    map_style = sidebar_params["map_style"]
    color_mode = sidebar_params["color_mode"]
    filter_risk = sidebar_params["filter_risk"]

    # 5. Cargar dataset efectivo para el horizonte con caché de geometrías (carga instantánea)
    predictions = load_operational_predictions(selected_file)
    df_data = load_operational_data_for_horizon(selected_file, selected_horizon)
    if df_data.empty:
        st.error(
            "No se encontraron predicciones disponibles en data/processed/. Ejecuta "
            "`scripts/run_daily_inference.py` para generar el pronóstico operativo."
        )
        st.stop()

    manifest = load_operational_manifest()

    forecast_quality = manifest.get("forecast_quality", "unknown")
    if forecast_quality in {"incomplete", "invalid", "unavailable"}:
        st.error(
            "No hay un forecast completo y válido para publicar. El resultado mostrado "
            "no debe utilizarse para movilización preventiva."
        )
    elif forecast_quality in {"stale", "fresh_fallback", "fresh_aemet_degraded", "fresh_aemet_proxy", "fresh_aemet"}:
        st.warning(
            f"Calidad meteorológica: {forecast_quality}. Consulta la pestaña de auditoría "
            "antes de interpretar el mapa como escenario WRF 1 km."
        )

    target_date = str(pd.to_datetime(df_data.get("fecha", pd.Series(["hoy"]))).dt.strftime("%Y-%m-%d").iloc[0])

    # Cargar modelo serializado para el horizonte
    try:
        dashboard_model = load_dashboard_model(selected_horizon)
    except Exception:
        dashboard_model = None

    # 6. Renderizar Cabecera de Mando y Resumen Ejecutivo Matinal
    render_header_and_kpis(df_data, manifest, selected_horizon, target_date)

    # 7. Navegación operativa con renderizado diferido.
    #
    # ``st.tabs`` solo organiza visualmente el contenido: Streamlit ejecuta el
    # cuerpo de las seis pestañas en cada sesión y en cada rerun. Eso fuerza a
    # construir el mapa, las tablas, TreeSHAP y el simulador aunque el usuario
    # solo necesite la vista inicial. Un selector horizontal conserva la
    # navegación compacta y permite renderizar únicamente la sección activa.
    sections = [
        "Centro de Mando Cartográfico",
        "Consulta por Concello",
        "Situación Territorial y Rankings",
        "Diagnóstico y Simulador",
        "Medidas y Despacho (PLADIGA)",
        "Auditoría del Sistema",
    ]
    active_section = st.radio(
        "Sección del dashboard",
        sections,
        horizontal=True,
        label_visibility="collapsed",
    )

    if active_section == sections[0]:
        render_map_tab(
            df_data=df_data,
            selected_horizon=selected_horizon,
            map_style=map_style,
            color_mode=color_mode,
            filter_risk=filter_risk,
        )
    elif active_section == sections[1]:
        render_concello_lookup_tab(
            df_data=df_data,
            target_date=target_date,
        )
    elif active_section == sections[2]:
        render_territorial_analytics_tab(
            df_data=df_data,
            all_predictions=predictions,
        )
    elif active_section == sections[3]:
        render_shap_and_simulator_tab(
            df_data=df_data,
            dashboard_model=dashboard_model,
            selected_horizon=selected_horizon,
        )
    elif active_section == sections[4]:
        render_operational_protocols_tab(
            df_data=df_data,
            manifest=manifest,
            selected_horizon=selected_horizon,
            target_date=target_date,
        )
    else:
        render_system_status_tab(
            df_data=df_data,
            manifest=manifest,
            selected_horizon=selected_horizon,
        )


if __name__ == "__main__":
    main()
