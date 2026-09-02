"""Componente del panel lateral (Sidebar) con Google Material Symbols."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.webapp.utils.data_loader import list_available_inference_datasets


def render_sidebar(predictions: pd.DataFrame) -> dict:
    """Renderiza el panel lateral y devuelve los parámetros de configuración seleccionados."""
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.5rem;">
            <span class="material-symbols-outlined" style="font-size:20px; color:#60a5fa;">settings</span>
            <span style="font-weight:700; font-size:1.05rem; letter-spacing:-0.01em;">Configuración Operativa</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Selector de Dataset / Ejecución
    available_datasets = list_available_inference_datasets()
    selected_dataset_name = st.sidebar.selectbox(
        "Conjunto de Inferencia:",
        list(available_datasets.keys()),
        index=0,
    )
    selected_dataset_file = available_datasets.get(selected_dataset_name)

    # 2. Selector de Horizonte Temporal
    if not predictions.empty and "horizon_days" in predictions.columns:
        available_horizons = sorted(predictions["horizon_days"].dropna().astype(int).unique())
    else:
        available_horizons = [1, 2, 3]

    horizon_labels = {h: f"T+{h} ({h * 24} horas)" for h in available_horizons}
    selected_horizon = st.sidebar.selectbox(
        "Horizonte de Previsión:",
        available_horizons,
        format_func=lambda value: horizon_labels.get(value, f"T+{value}"),
    )

    st.sidebar.markdown("---")

    # 3. Estilo del Mapa Base
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.35rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:18px; color:#94a3b8;">layers</span>
            <span style="font-weight:600; font-size:0.85rem; text-transform:uppercase; color:#94a3b8;">Capa Base Cartográfica</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    map_style = st.sidebar.selectbox(
        "Seleccionar Estilo:",
        [
            "Esri Gris Claro (Lienzo Táctico)",
            "Esri Gris Oscuro (Lienzo Táctico)",
            "Esri Satellite (Satelital)",
            "IGN España — PNOA Ortofoto (Oficial)",
            "OpenTopoMap (Topográfico)",
            "OpenStreetMap",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # 4. Modo de Visualización de Color
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.35rem; margin-top:0.75rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:18px; color:#94a3b8;">palette</span>
            <span style="font-weight:600; font-size:0.85rem; text-transform:uppercase; color:#94a3b8;">Criterio Térmico</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    color_mode = st.sidebar.radio(
        "Criterio Térmico:",
        [
            "Gradiente Continuo por Probabilidad P(Y=1)",
            "Gradiente Continuo por Percentil Relativo (%)",
            "Selección Táctica por Niveles Fijos (Top %)",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # 5. Filtro de Celdas por Umbral de Riesgo
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.35rem; margin-top:0.75rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:18px; color:#94a3b8;">filter_alt</span>
            <span style="font-weight:600; font-size:0.85rem; text-transform:uppercase; color:#94a3b8;">Filtro Espacial</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filter_risk = st.sidebar.selectbox(
        "Filtro Espacial:",
        [
            "Top 1.0% Celdas Prioritarias",
            "Top 5.0% Celdas en Riesgo Elevado",
            "Top 2.0% Celdas Críticas",
            "Top 0.5% Riesgo Extremo",
            "Top 10.0% Celdas de Vigilancia",
            "Mostrar Todas las Celdas con Riesgo > 0.5%",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # 6. Botón de Recarga de Caché
    if st.sidebar.button("Recargar Datos"):
        st.cache_data.clear()
        st.rerun()

    return {
        "selected_dataset_file": selected_dataset_file,
        "selected_horizon": selected_horizon,
        "map_style": map_style,
        "color_mode": color_mode,
        "filter_risk": filter_risk,
    }
