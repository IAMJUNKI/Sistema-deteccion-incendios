"""Componente de Salud del Sistema, Trazabilidad y Auditoría de Modelos."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.features.canonical_contract import EGIF_48_FEATURE_CONTRACT_VERSION


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
        quality = manifest.get("forecast_quality", "unknown")
        age = manifest.get("forecast_age_hours")
        source_resolution = manifest.get("forecast_source_resolution", "unknown")
        valid_start = manifest.get("valid_start", "-")
        valid_end = manifest.get("valid_end", "-")

        st.markdown(
            f"""
            <div class="action-box">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#60a5fa;">cloud_sync</span>
                    Parámetros de Ingesta Operativa
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
                    <b>Proveedor Meteorológico:</b> {provider}<br/>
                    <b>Malla / resolución:</b> {grid_res or 'no disponible'}<br/>
                    <b>Resolución de fuente:</b> {source_resolution}<br/>
                    <b>Versión API:</b> {api_ver}<br/>
                    <b>Calidad:</b> <code>{quality}</code> · <b>Antigüedad:</b> {age if age is not None else '-'} h<br/>
                    <b>Fecha/Hora de Emisión:</b> <code>{issue_time}</code><br/>
                    <b>Validez UTC:</b> <code>{valid_start}</code> → <code>{valid_end}</code><br/>
                    <b>Modo de Ejecución:</b> <code>{mode}</code><br/>
                    <b>Cobertura de Celdas:</b> {coverage.get('complete_cells', len(df_data)):,} / {coverage.get('cells', len(df_data)):,} celdas completas ({coverage.get('minimum_ratio', 1.0):.1%})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if quality in {"fresh_aemet", "fresh_aemet_degraded", "fresh_aemet_proxy", "stale", "incomplete", "invalid", "unavailable"}:
            st.warning(
                "Este mapa no usa el escenario WRF 1 km nominal: "
                f"estado meteorológico = {quality}. Revisa la cobertura antes de movilizar medios."
            )
        state_quality = manifest.get("state", {}).get("feature_quality_counts", {})
        if isinstance(state_quality, dict) and state_quality.get("legacy_proxy", 0):
            st.warning(
                "El estado meteorológico reciente contiene filas legacy_proxy: "
                "las memorias históricas no proceden de una serie horaria completa."
            )

    with col_mod:
        selected_model = manifest.get("models", {}).get(str(selected_horizon), {})
        model_metadata = selected_model.get("metadata", {}) if isinstance(selected_model, dict) else {}
        feature_version = manifest.get("feature_contract_version") or model_metadata.get(
            "feature_schema_version", "desconocido"
        )
        feature_count = len(model_metadata.get("feature_columns", [])) or (
            48
            if feature_version == EGIF_48_FEATURE_CONTRACT_VERSION
            else 50
            if feature_version == "egif-2d-v1"
            else 23
        )
        model_family = selected_model.get("model_family", "desconocido") if isinstance(selected_model, dict) else "desconocido"
        calibration_method = model_metadata.get(
            "calibration_method",
            "prior_correction+platt"
            if feature_version == EGIF_48_FEATURE_CONTRACT_VERSION
            else "prior_correction+isotonic"
            if feature_version == "egif-2d-v1"
            else "legacy",
        )
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
                    <b>Calibración de Probabilidades:</b> <code>{calibration_method}</code><br/>
                    <b>Familia:</b> <code>{model_family}</code><br/>
                    <b>Esquema de Features:</b> <code>{feature_version}</code> ({feature_count} variables)<br/>
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
