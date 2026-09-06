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

    # 1. Selector de Horizonte Temporal (Control Primario)
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

    # 2. Modo de Visualización de Riesgo (Simbología)
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.35rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:18px; color:#94a3b8;">palette</span>
            <span style="font-weight:600; font-size:0.85rem; text-transform:uppercase; color:#94a3b8;">Simbología de Riesgo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    color_mode = st.sidebar.radio(
        "Simbología de Riesgo:",
        [
            "Riesgo Absoluto Calibrado P(Y=1)",
            "Priorización Relativa por Percentil (%)",
            "Niveles Tácticos Discretos (Top %)",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.caption(
        "• **Absoluto:** severidad física real del día.\n• **Relativo:** prioriza las celdas más calientes de Galicia."
    )

    # 3. Filtro Espacial de Celdas
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.35rem; margin-top:0.75rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:18px; color:#94a3b8;">filter_alt</span>
            <span style="font-weight:600; font-size:0.85rem; text-transform:uppercase; color:#94a3b8;">Filtro de Celdas</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filter_risk = st.sidebar.selectbox(
        "Filtro Espacial:",
        [
            "Top 0.5%",
            "Top 1.0%",
            "Top 2.0%",
            "Top 5.0%",
            "Top 10.0%",
            "Top 20.0%",
        ],
        index=3,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # 4. Configuración Avanzada y Cartografía (Colapsado para reducir carga cognitiva)
    with st.sidebar.expander("⚙️ Opciones Avanzadas / Capas", expanded=False):
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.35rem; margin-bottom:0.25rem;">
                <span class="material-symbols-outlined" style="font-size:18px; color:#94a3b8;">layers</span>
                <span style="font-weight:600; font-size:0.85rem; text-transform:uppercase; color:#94a3b8;">Capa Base Cartográfica</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        map_style = st.selectbox(
            "Capa Base Cartográfica:",
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

        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.35rem; margin-top:0.6rem; margin-bottom:0.25rem;">
                <span class="material-symbols-outlined" style="font-size:18px; color:#94a3b8;">database</span>
                <span style="font-weight:600; font-size:0.85rem; text-transform:uppercase; color:#94a3b8;">Dataset de Inferencia</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        available_datasets = list_available_inference_datasets()
        selected_dataset_name = st.selectbox(
            "Dataset:",
            list(available_datasets.keys()),
            index=0,
            label_visibility="collapsed",
        )
        selected_dataset_file = available_datasets.get(selected_dataset_name)

        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        if st.button("Recargar Datos de Caché", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return {
        "selected_dataset_file": selected_dataset_file,
        "selected_horizon": selected_horizon,
        "map_style": map_style,
        "color_mode": color_mode,
        "filter_risk": filter_risk,
    }
