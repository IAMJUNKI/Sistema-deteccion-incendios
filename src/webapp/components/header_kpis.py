"""Componente de cabecera y telemetría de KPIs operativos con Google Material Symbols."""

from __future__ import annotations

import re
import pandas as pd
import streamlit as st


def render_html_safely(html_str: str) -> None:
    """Renderiza HTML de forma segura eliminando cualquier sangría que Markdown interprete como código."""
    clean_html = re.sub(r"^[ \t]+", "", html_str, flags=re.MULTILINE).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


def render_header_and_kpis(
    df_data: pd.DataFrame,
    manifest: dict,
    selected_horizon: int,
    target_date: str,
) -> None:
    """Renderiza la cabecera del centro de mando y las tarjetas de telemetría KPI."""
    if df_data.empty:
        return

    # Extraer variables clave de calidad y procedencia
    issue_time = str(df_data.get("issue_time", pd.Series(["-"])).iloc[0])
    forecast_quality = str(df_data.get("forecast_quality", pd.Series(["unknown"])).iloc[0])
    forecast_age = float(df_data.get("forecast_age_hours", pd.Series([float("nan")])).iloc[0])
    forecast_provider = str(
        df_data.get("forecast_provider", pd.Series([manifest.get("forecast_provider", "MeteoGalicia")])).iloc[0]
    )
    forecast_grid = str(df_data.get("forecast_grid", pd.Series(["1km"])).iloc[0])

    # Determinar badge de calidad del forecast
    if forecast_quality in {"stale"}:
        badge_class = "badge-stale"
        badge_icon = "warning"
        badge_text = f"Forecast Reutilizado ({forecast_age:.1f}h)"
    elif forecast_quality in {"fresh_fallback"}:
        badge_class = "badge-fallback"
        badge_icon = "sync_problem"
        badge_text = f"WRF 04km Fallback ({forecast_provider})"
    elif forecast_quality in {"fresh_aemet", "fresh_aemet_proxy"}:
        badge_class = "badge-fallback"
        badge_icon = "location_city"
        badge_text = "AEMET Municipal (Contingencia)"
    else:
        badge_class = "badge-fresh"
        badge_icon = "verified"
        badge_text = f"WRF {forecast_grid.upper()} Nominal ({forecast_provider})"

    # Cálculos para los KPIs
    total_cells = len(df_data)
    max_prob = float(df_data["prob_riesgo"].max() * 100) if "prob_riesgo" in df_data.columns else 1.0
    mean_prob = float(df_data["prob_riesgo"].mean() * 100) if "prob_riesgo" in df_data.columns else 0.5

    num_top5 = int((df_data["percentil_riesgo"] >= 0.95).sum()) if "percentil_riesgo" in df_data.columns else 0
    num_top05 = int((df_data["percentil_riesgo"] >= 0.995).sum()) if "percentil_riesgo" in df_data.columns else 0

    # Indicador de Regla 30-30-30
    num_r30 = int(df_data["regla_30_30_activa"].sum()) if "regla_30_30_activa" in df_data.columns else 0
    pct_r30 = (num_r30 / total_cells * 100) if total_cells > 0 else 0.0

    # Distrito con mayor riesgo
    distrito_top = "Galicia Sur"
    if "distrito_forestal" in df_data.columns:
        top_group = df_data.groupby("distrito_forestal")["prob_riesgo"].mean()
        if not top_group.empty:
            distrito_top = str(top_group.idxmax())

    # Determinar severidad general del día
    if num_top05 > 100 or max_prob > 12.0:
        alert_card_class = "alert-critical"
        alert_label = "Nivel 5 — Extremo"
        briefing_color = "#dc2626"
    elif num_top5 > 1000 or max_prob > 6.0:
        alert_card_class = "alert-warning"
        alert_label = "Nivel 4 — Muy Alto"
        briefing_color = "#ea580c"
    elif max_prob > 2.5:
        alert_card_class = "alert-warning"
        alert_label = "Nivel 3 — Alto"
        briefing_color = "#d97706"
    else:
        alert_card_class = "alert-success"
        alert_label = "Nivel 1-2 — Moderado/Bajo"
        briefing_color = "#059669"

    # Renderizar Banner Superior Institucional
    header_html = f"""
    <div class="command-header">
        <div>
            <h1 class="command-title">
                <span class="material-symbols-outlined" style="color:#ef4444; font-size:26px;">local_fire_department</span>
                SISTEMA DE ALERTA TEMPRANA DE INCENDIOS FORESTALES
            </h1>
            <div class="command-subtitle">
                Centro de Mando Operativo a 1 km² · Pronóstico para <b>{target_date}</b> (Horizonte <b>T+{selected_horizon}</b> / {selected_horizon*24}h)
            </div>
        </div>
        <div>
            <span class="badge {badge_class}">
                <span class="material-symbols-outlined" style="font-size:16px;">{badge_icon}</span>
                {badge_text}
            </span>
        </div>
    </div>
    """
    render_html_safely(header_html)

    # Resumen Ejecutivo de Situación (Lectura en 30 segundos)
    max_t = float(df_data["tmax_vc"].max()) if "tmax_vc" in df_data.columns else 28.0
    max_v = float(df_data["vmax_vc"].max()) if "vmax_vc" in df_data.columns else 20.0

    briefing_html = f"""
    <div style="background: #111827; border-left: 4px solid {briefing_color}; border-radius: 6px; padding: 0.85rem 1.2rem; margin-bottom: 1rem; border-top: 1px solid #1f2937; border-right: 1px solid #1f2937; border-bottom: 1px solid #1f2937;">
        <div style="display:flex; align-items:center; gap:0.4rem; font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#94a3b8; letter-spacing:0.04em;">
            <span class="material-symbols-outlined" style="font-size:18px; color:{briefing_color};">campaign</span>
            Resumen Ejecutivo Matinal para Mandos y Protección Civil
        </div>
        <div style="font-size:0.88rem; color:#f1f5f9; margin-top:0.35rem; line-height:1.5;">
            Hoy la atención prioritaria debe concentrarse en el <b>{distrito_top}</b> durante la ventana crítica de <b>13:00 a 18:00 h</b>, con máximas previstas de hasta <b>{max_t:.1f} °C</b> y vientos de <b>{max_v:.1f} km/h</b>. 
            Estado de severidad autonómica: <b style="color:{briefing_color};">{alert_label}</b> ({num_top5:,} km² en vigilancia prioritaria).
        </div>
    </div>
    """
    render_html_safely(briefing_html)

    # Renderizar Tarjetas de Telemetría
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card1_html = f"""
        <div class="kpi-card {alert_card_class}">
            <div class="kpi-label">
                <span class="material-symbols-outlined">speed</span>
                Nivel de Peligro del Día
            </div>
            <div class="kpi-value" style="font-size:1.35rem;">{alert_label}</div>
            <div class="kpi-sub">Prob. Máx: <b>{max_prob:.2f}%</b> · Media: {mean_prob:.2f}%</div>
        </div>
        """
        render_html_safely(card1_html)

    with c2:
        card2_html = f"""
        <div class="kpi-card alert-critical">
            <div class="kpi-label">
                <span class="material-symbols-outlined">warning</span>
                Celdas de Alerta Urgente
            </div>
            <div class="kpi-value">{num_top5:,} <span style="font-size:0.9rem; color:#94a3b8;">km²</span></div>
            <div class="kpi-sub"><b>{num_top05:,}</b> celdas en Máxima Prioridad (Top 0.5%)</div>
        </div>
        """
        render_html_safely(card2_html)

    with c3:
        r30_style = "alert-warning" if num_r30 > 0 else "alert-info"
        card3_html = f"""
        <div class="kpi-card {r30_style}">
            <div class="kpi-label">
                <span class="material-symbols-outlined">thermostat</span>
                Condición Crítica 30-30-30
            </div>
            <div class="kpi-value">{num_r30:,} <span style="font-size:0.9rem;color:#94a3b8;">({pct_r30:.1f}%)</span></div>
            <div class="kpi-sub">T &gt; 30°C · HR &lt; 30% · V &gt; 30 km/h</div>
        </div>
        """
        render_html_safely(card3_html)

    with c4:
        card4_html = f"""
        <div class="kpi-card alert-info">
            <div class="kpi-label">
                <span class="material-symbols-outlined">explore</span>
                Sector de Mayor Atención
            </div>
            <div class="kpi-value" style="font-size:1.05rem;padding-top:0.3rem;">{distrito_top}</div>
            <div class="kpi-sub">Superficie evaluada: {total_cells:,} km²</div>
        </div>
        """
        render_html_safely(card4_html)
