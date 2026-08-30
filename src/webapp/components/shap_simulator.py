"""Componente de Diagnóstico Biofísico y Simulador Interactivo What-If Accesible."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st

from src.features.operational_features import calculate_vpd
from src.models.explainability import explain_tree_prediction
from src.webapp.components.concello_lookup import get_risk_level_info


def render_html_safely(html_str: str) -> None:
    """Renderiza HTML de forma segura eliminando cualquier sangría que Markdown interprete como código."""
    clean_html = re.sub(r"^[ \t]+", "", html_str, flags=re.MULTILINE).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


def render_shap_and_simulator_tab(
    df_data: pd.DataFrame,
    dashboard_model: object | None,
    selected_horizon: int,
) -> None:
    """Renderiza el panel de diagnóstico biofísico claro y el simulador de escenarios What-If."""
    render_html_safely(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:22px; color:#60a5fa;">psychology</span>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Diagnóstico Biofísico: ¿Por qué esta zona está en peligro?</h3>
        </div>
        """
    )
    st.caption(
        "Explicación transparente de las causas físicas que elevan o reducen el riesgo en cualquier punto de Galicia."
    )

    if df_data.empty:
        st.warning("No hay datos disponibles para el diagnóstico.")
        return

    # Selector de Celda para Diagnóstico
    top_cells = (
        df_data.sort_values(by="percentil_riesgo", ascending=False)["cell_id"]
        .head(30)
        .astype(int)
        .tolist()
    )

    col_sel, col_info_box = st.columns([1.1, 1.9])

    with col_sel:
        selected_cid = st.selectbox("Seleccionar Celda Prioritaria:", top_cells)
        c_info = df_data[df_data["cell_id"] == selected_cid].iloc[0]

        prob = float(c_info.get("prob_riesgo", 0.02))
        risk_info = get_risk_level_info(prob)
        action = str(c_info.get("recommended_action", "vigilancia_reforzada")).replace("_", " ").title()
        distrito = str(c_info.get("distrito_forestal", "Galicia"))
        provincia = str(c_info.get("provincia", "Ourense"))
        r30_flag = "Activa" if c_info.get("regla_30_30_activa", False) else "Inactiva"

        action_box_html = f"""
        <div class="action-box" style="border-top: 3px solid {risk_info['border_color']};">
            <div class="action-title">
                <span class="material-symbols-outlined" style="font-size:18px; color:{risk_info['border_color']};">location_on</span>
                Celda #{selected_cid} · {provincia}
            </div>
            <div style="font-size:1.25rem; font-weight:800; color:{risk_info['badge_color']}; margin:0.25rem 0;">
                {risk_info['level']}
            </div>
            <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
                <b>Distrito:</b> {distrito}<br/>
                <b>Acción Asignada:</b> {action}<br/>
                <b>Condición 30-30-30:</b> {r30_flag}
            </div>
        </div>
        """
        render_html_safely(action_box_html)

    with col_info_box:
        tmax = float(c_info.get("tmax_vc", 25.0))
        rhmin = float(c_info.get("rhmin_vc", 35.0))
        vmax = float(c_info.get("vmax_vc", 15.0))
        prec30 = float(c_info.get("prec_acum_30d", 5.0))
        raw_fuel = c_info.get("combustible_clase")
        fuel_type = (
            str(raw_fuel).capitalize()
            if pd.notna(raw_fuel) and str(raw_fuel).strip().lower() not in ["", "nan", "none"]
            else "Matorral / Monte Bajo"
        )
        forest_pct = float(c_info.get("combustible_pct_forestal", 0.0))
        slope = float(c_info.get("pendiente_media", 12.0))

        st.markdown("#### Factores Clave Observados en el Terreno")
        
        fc1, fc2 = st.columns(2)
        with fc1:
            box1_html = f"""
            <div style="background:#0f172a; padding:0.65rem 0.85rem; border-radius:6px; border:1px solid #1e293b; margin-bottom:0.5rem;">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Calor en Ventana Crítica</div>
                <div style="font-size:1.15rem; font-weight:700; color:{'#ef4444' if tmax>=30 else '#f8fafc'};">{tmax:.1f} °C</div>
                <div style="font-size:0.75rem; color:#94a3b8;">{'Supera el umbral de 30°C' if tmax>=30 else 'Temperatura moderada'}</div>
            </div>
            <div style="background:#0f172a; padding:0.65rem 0.85rem; border-radius:6px; border:1px solid #1e293b;">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Humedad del Aire</div>
                <div style="font-size:1.15rem; font-weight:700; color:{'#ef4444' if rhmin<=30 else '#f8fafc'};">{rhmin:.1f} %</div>
                <div style="font-size:0.75rem; color:#94a3b8;">{'Extremadamente seco (<30%)' if rhmin<=30 else 'Humedad relativa favorable'}</div>
            </div>
            """
            render_html_safely(box1_html)
        with fc2:
            box2_html = f"""
            <div style="background:#0f172a; padding:0.65rem 0.85rem; border-radius:6px; border:1px solid #1e293b; margin-bottom:0.5rem;">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Viento Máximo Previsto</div>
                <div style="font-size:1.15rem; font-weight:700; color:{'#f59e0b' if vmax>=25 else '#f8fafc'};">{vmax:.1f} km/h</div>
                <div style="font-size:0.75rem; color:#94a3b8;">{'Favorece avance rápido en monte' if vmax>=25 else 'Viento moderado'}</div>
            </div>
            <div style="background:#0f172a; padding:0.65rem 0.85rem; border-radius:6px; border:1px solid #1e293b;">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Tipo de Combustible</div>
                <div style="font-size:1.15rem; font-weight:700; color:#f8fafc;">{fuel_type}</div>
                <div style="font-size:0.75rem; color:#94a3b8;">{forest_pct:.0f}% arbolado · Pendiente {slope:.0f}°</div>
            </div>
            """
            render_html_safely(box2_html)

    st.markdown("---")

    # Explicación en Lenguaje Natural
    st.markdown("#### Conclusión del Diagnóstico")
    
    explanation_df = None
    if dashboard_model is not None:
        try:
            explanation_df = explain_tree_prediction(dashboard_model, c_info.to_frame().T)
        except Exception:
            pass

    reasons = []
    if tmax >= 30:
        reasons.append(f"temperatura elevada de **{tmax:.1f}°C** que deseca el combustible fino")
    if rhmin <= 30:
        reasons.append(f"humedad relativa muy baja (**{rhmin:.1f}%**) que facilita la ignición")
    if vmax >= 25:
        reasons.append(f"viento de **{vmax:.1f} km/h** que aceleraría la propagación")
    if prec30 < 10:
        reasons.append(f"fuerte sequedad acumulada (**{prec30:.1f} mm en 30 días**)")

    if reasons:
        narrative_text = "El nivel de peligro en esta celda se debe principalmente a: " + ", ".join(reasons) + "."
    else:
        narrative_text = "Las condiciones meteorológicas y de terreno se encuentran en rangos moderados o habituales para la estación."

    st.info(narrative_text)

    # Vista técnica expandible
    if explanation_df is not None and not explanation_df.empty:
        with st.expander("Ver Desglose de Contribuciones Técnicas (TreeSHAP)", expanded=False):
            top_exp = explanation_df.head(8).copy()
            chart_series = top_exp.set_index("feature")["contribution"]
            st.bar_chart(chart_series, color="#dc2626")

    st.markdown("---")

    # =========================================================================
    # SIMULADOR ACCESIBLE "WHAT-IF"
    # =========================================================================
    render_html_safely(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:22px; color:#60a5fa;">tune</span>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Simulador ¿Qué pasaría si cambia el tiempo? (What-If)</h3>
        </div>
        """
    )
    st.caption("Ajusta los controles deslizantes para ver cómo cambiaría el nivel de peligro ante una ola de calor o un cambio de viento.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        delta_temp = st.slider("Ajuste de Temperatura (°C):", -5.0, 10.0, 0.0, 0.5)
    with sim_col2:
        delta_rh = st.slider("Ajuste de Humedad (%):", -25.0, 25.0, 0.0, 1.0)
    with sim_col3:
        delta_wind = st.slider("Ajuste de Viento (km/h):", -15.0, 30.0, 0.0, 1.0)

    # Crear vector de features simulado
    sim_row = c_info.copy()
    sim_row["tmax_vc"] = float(np.clip(c_info.get("tmax_vc", 25.0) + delta_temp, 5.0, 50.0))
    sim_row["rhmin_vc"] = float(np.clip(c_info.get("rhmin_vc", 35.0) + delta_rh, 5.0, 100.0))
    sim_row["vmax_vc"] = float(np.clip(c_info.get("vmax_vc", 15.0) + delta_wind, 0.0, 100.0))
    sim_row["vpd_vc"] = float(calculate_vpd(np.array([sim_row["tmax_vc"]]), np.array([sim_row["rhmin_vc"]]))[0])

    if dashboard_model is not None and hasattr(dashboard_model, "predict_proba"):
        try:
            df_sim_eval = sim_row.to_frame().T
            sim_proba_mat = dashboard_model.predict_proba(df_sim_eval)
            sim_prob_val = float(sim_proba_mat[0, 1])
        except Exception:
            base_p = float(c_info.get("prob_riesgo", 0.02))
            sim_prob_val = float(np.clip(base_p * (1.0 + delta_temp * 0.08 - delta_rh * 0.03 + delta_wind * 0.04), 0.001, 0.99))
    else:
        base_p = float(c_info.get("prob_riesgo", 0.02))
        sim_prob_val = float(np.clip(base_p * (1.0 + delta_temp * 0.08 - delta_rh * 0.03 + delta_wind * 0.04), 0.001, 0.99))

    sim_risk_info = get_risk_level_info(sim_prob_val)

    st.markdown("#### Resultado de la Simulación")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        res1_html = f"""
        <div style="background:#0f172a; padding:1rem; border-radius:6px; border:1px solid #1e293b;">
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Nivel Actual Previsto</div>
            <div style="font-size:1.3rem; font-weight:700; color:{risk_info['badge_color']};">{risk_info['level']}</div>
            <div style="font-size:0.8rem; color:#94a3b8;">T: {tmax:.1f}°C · HR: {rhmin:.1f}% · Viento: {vmax:.1f} km/h</div>
        </div>
        """
        render_html_safely(res1_html)

    with r_col2:
        res2_html = f"""
        <div style="background:#0f172a; padding:1rem; border-radius:6px; border:2px solid {sim_risk_info['border_color']};">
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Nuevo Nivel Simulado</div>
            <div style="font-size:1.3rem; font-weight:700; color:{sim_risk_info['badge_color']};">{sim_risk_info['level']}</div>
            <div style="font-size:0.8rem; color:#94a3b8;">T: {sim_row['tmax_vc']:.1f}°C · HR: {sim_row['rhmin_vc']:.1f}% · Viento: {sim_row['vmax_vc']:.1f} km/h</div>
        </div>
        """
        render_html_safely(res2_html)
