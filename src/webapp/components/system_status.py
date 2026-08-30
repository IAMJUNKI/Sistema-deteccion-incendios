"""Componente de Salud del Sistema, Trazabilidad y Auditoría de Modelos."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_system_status_tab(
    df_data: pd.DataFrame,
    manifest: dict,
    selected_horizon: int,
) -> None:
    """Renderiza la pestaña de auditoría, gobernanza y salud del pipeline operativo."""
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:22px; color:#60a5fa;">dns</span>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Salud del Pipeline, Trazabilidad y Auditoría Metodológica</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Verificación de gobernanza de datos, calibración de modelos y procedencia de forecasts.")

    col_man, col_mod = st.columns(2)

    with col_man:
        st.markdown("#### Trazabilidad de la Emisión Meteorológica")
        provider = manifest.get("forecast_provider", "MeteoGalicia")
        grid_res = manifest.get("forecast_grid", "1km")
        api_ver = manifest.get("forecast_api_version", "MeteoSIX v5")
        issue_time = manifest.get("issue_time", df_data.get("issue_time", pd.Series(["-"])).iloc[0])
        mode = manifest.get("pipeline_run_mode", "live")
        coverage = manifest.get("forecast_coverage", {})

        st.markdown(
            f"""
            <div class="action-box">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#60a5fa;">cloud_sync</span>
                    Parámetros de Ingesta Operativa
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
                    <b>Proveedor Meteorológico:</b> {provider}<br/>
                    <b>Malla Numérica:</b> WRF {grid_res}<br/>
                    <b>Versión API:</b> {api_ver}<br/>
                    <b>Fecha/Hora de Emisión:</b> <code>{issue_time}</code><br/>
                    <b>Modo de Ejecución:</b> <code>{mode}</code><br/>
                    <b>Cobertura de Celdas:</b> {coverage.get('complete_cells', len(df_data)):,} / {coverage.get('cells', len(df_data)):,} celdas completas ({coverage.get('minimum_ratio', 1.0):.1%})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mod:
        st.markdown("#### Arquitectura y Calibración del Modelo")
        st.markdown(
            f"""
            <div class="action-box">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#34d399;">model_training</span>
                    Especificación del Algoritmo Supervisado
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
                    <b>Algoritmo Base:</b> LightGBM Classifier (Gradient Boosting Decision Trees)<br/>
                    <b>Calibración de Probabilidades:</b> Regresión Isotónica (Isotonic Regression sobre año completo)<br/>
                    <b>Esquema de Features:</b> <code>operational-risk-v1</code> (23 variables biofísicas)<br/>
                    <b>Estrategia de Validación:</b> Bloques temporales anuales completos (Anti-Data Leakage)<br/>
                    <b>Horizonte Operativo Activo:</b> T+{selected_horizon} ({selected_horizon*24} horas)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Referencias del Estado del Arte y Memoria TFM
    st.markdown("#### Marco Metodológico y Alineación con el Estado del Arte")
    
    st.markdown(
        """
        - **IberFire (Ercibengoa et al., 2025):** Validación metodológica y benchmark de resolución espacial de 1 km × 1 km. A diferencia de IberFire (que se orienta a la construcción de datasets históricos), este sistema implementa un pipeline operativo *end-to-end* en tiempo real con MeteoGalicia.
        - **IPIF de AEMET (2026):** Integra variables multimodales biofísicas de combustible (CORINE), estrés hídrico acumulado y topografía DEM de Copernicus, superando las limitaciones del FWI clásico mediante Machine Learning supervisado y calibrado.
        - **Copernicus EFFIS / GEFF:** Complementa los modelos continentales europeos de baja resolución (10–25 km) mediante reducción de escala local a 1 km² adaptada al régimen pirogénico específico de Galicia.
        """
    )

    # Manifiesto JSON Crudo
    with st.expander("Ver Manifiesto Técnico Completo (JSON)", expanded=False):
        st.json(manifest if manifest else {"status": "offline_mode", "dataset_cells": len(df_data)})
