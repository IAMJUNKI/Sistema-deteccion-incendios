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


def get_day_severity_info(max_prob: float, num_critical_cells: int = 0) -> dict:
    """Determina la severidad autonómica global basada en el riesgo físico calibrado P(Y=1).

    max_prob: Probabilidad máxima en porcentaje (ej. 1.88 para 1.88%).
    num_critical_cells: Número de celdas con probabilidad crítica (>= 12.0%).
    """
    if max_prob >= 12.0 or num_critical_cells >= 50:
        return {
            "level": "Nivel 5 — Extremo",
            "card_class": "alert-critical",
            "briefing_color": "#dc2626",
            "badge_color": "#800026",
            "icon": "emergency",
            "description": "Jornada de peligro crítico generalizado. Máxima alerta autonómica.",
        }
    elif max_prob >= 6.0 or num_critical_cells >= 20:
        return {
            "level": "Nivel 4 — Muy Alto",
            "card_class": "alert-critical",
            "briefing_color": "#ea580c",
            "badge_color": "#dc2626",
            "icon": "warning",
            "description": "Focos probables de rápida propagación. Atención reforzada.",
        }
    elif max_prob >= 2.5:
        return {
            "level": "Nivel 3 — Alto",
            "card_class": "alert-warning",
            "briefing_color": "#ea580c",
            "badge_color": "#ea580c",
            "icon": "report_problem",
            "description": "Condiciones desfavorables en sectores específicos.",
        }
    elif max_prob >= 1.0:
        return {
            "level": "Nivel 2 — Moderado",
            "card_class": "alert-warning",
            "briefing_color": "#d97706",
            "badge_color": "#d97706",
            "icon": "info",
            "description": "Riesgo moderado controlado. Condiciones habituales de vigilancia.",
        }
    else:
        return {
            "level": "Nivel 1 — Bajo / Nominal",
            "card_class": "alert-success",
            "briefing_color": "#059669",
            "badge_color": "#059669",
            "icon": "verified",
            "description": "Régimen nominal. Baja probabilidad de ignición.",
        }


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
    elif forecast_quality in {"fresh_aemet", "fresh_aemet_proxy", "fresh_aemet_degraded"}:
        badge_class = "badge-fallback"
        badge_icon = "location_city"
        badge_text = "AEMET Municipal (Contingencia degradada)"
    elif forecast_quality in {"incomplete", "invalid", "unavailable"}:
        badge_class = "badge-stale"
        badge_icon = "cloud_off"
        badge_text = f"Forecast no publicable ({forecast_quality})"
    else:
        badge_class = "badge-fresh"
        badge_icon = "verified"
        badge_text = f"WRF {forecast_grid.upper()} Nominal ({forecast_provider})"

    # Cálculos para los KPIs
    total_cells = len(df_data)
    max_prob = float(df_data["prob_riesgo"].max() * 100) if "prob_riesgo" in df_data.columns else 1.0
    mean_prob = float(df_data["prob_riesgo"].mean() * 100) if "prob_riesgo" in df_data.columns else 0.5

    num_crit_12 = int((df_data["prob_riesgo"] >= 0.12).sum()) if "prob_riesgo" in df_data.columns else 0
    num_elev_25 = int((df_data["prob_riesgo"] >= 0.025).sum()) if "prob_riesgo" in df_data.columns else 0
    num_mod_10 = int((df_data["prob_riesgo"] >= 0.010).sum()) if "prob_riesgo" in df_data.columns else 0

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

    # Determinar severidad general del día basada en riesgo físico calibrado
    severity_info = get_day_severity_info(max_prob, num_crit_12)
    alert_card_class = severity_info["card_class"]
    alert_label = severity_info["level"]
    briefing_color = severity_info["briefing_color"]

    # Renderizar Banner Superior Institucional
    header_html = f"""
    <div class="command-header">
        <div>
            <div class="command-title">
                <span class="material-symbols-outlined" style="color:#ef4444; font-size:26px; line-height:1;">local_fire_department</span>
                <span>SISTEMA DE ALERTA TEMPRANA DE INCENDIOS FORESTALES</span>
            </div>
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
    temperature_column = next(
        (column for column in ("temperature_max_12_18h", "temperature_max", "tmax_vc") if column in df_data),
        None,
    )
    wind_column = next(
        (column for column in ("wind_speed_max_12_18h", "wind_speed_max", "vmax_vc") if column in df_data),
        None,
    )
    max_t = float(df_data[temperature_column].max()) if temperature_column else 28.0
    max_v = float(df_data[wind_column].max()) if wind_column else 20.0

    # Resumen territorial de severidad
    if max_prob >= 2.5 and num_elev_25 > 0:
        briefing_scope = f"{num_elev_25:,} km² en riesgo elevado P &ge; 2.5%"
    elif max_prob >= 1.0 and num_mod_10 > 0:
        briefing_scope = f"{num_mod_10:,} km² en riesgo activo P &ge; 1.0%"
    else:
        briefing_scope = f"{num_top05:,} km² en máxima prioridad de despacho táctico"

    briefing_html = f"""
    <div style="background: #111827; border-left: 4px solid {briefing_color}; border-radius: 6px; padding: 0.85rem 1.2rem; margin-bottom: 1rem; border-top: 1px solid #1f2937; border-right: 1px solid #1f2937; border-bottom: 1px solid #1f2937;">
            <div style="display:flex; align-items:center; gap:0.4rem; font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#94a3b8; letter-spacing:0.04em;">
                <span class="material-symbols-outlined" style="font-size:18px; color:{briefing_color};">campaign</span>
                Resumen Ejecutivo Matinal para Mandos y Protección Civil
            </div>
            <div style="font-size:0.88rem; color:#f1f5f9; margin-top:0.35rem; line-height:1.5;">
                Hoy la atención prioritaria debe concentrarse en el <b>{distrito_top}</b> durante la ventana crítica de <b>13:00 a 18:00 h</b>, con máximas previstas de hasta <b>{max_t:.1f} °C</b> y vientos de <b>{max_v:.1f} km/h</b>.
                Estado de severidad autonómica: <b style="color:{briefing_color};">{alert_label}</b> ({briefing_scope}).
            </div>
        </div>
        """
    render_html_safely(briefing_html)

    # Configuración dinámica de Card 2 según severidad real del pronóstico
    if max_prob >= 2.5 and num_elev_25 > 0:
        c2_class = "alert-critical" if max_prob >= 6.0 else "alert-warning"
        c2_label = "Superficie en Riesgo Elevado"
        c2_value = f"{num_elev_25:,} <span style='font-size:0.9rem; color:#94a3b8;'>km²</span>"
        c2_sub = f"<b>{num_top05:,}</b> cuadrículas en Máxima Prioridad (Top 0.5%)"
    elif max_prob >= 1.0:
        c2_class = "alert-warning"
        c2_label = "Superficie en Riesgo Activo"
        c2_value = f"{num_mod_10:,} <span style='font-size:0.9rem; color:#94a3b8;'>km²</span>"
        c2_sub = f"<b>{num_top05:,}</b> cuadrículas en Despacho Preventivo (Top 0.5%)"
    else:
        c2_class = "alert-success"
        c2_label = "Superficie en Vigilancia Nominal"
        c2_value = f"{num_top05:,} <span style='font-size:0.9rem; color:#94a3b8;'>km²</span>"
        c2_sub = "Sin cuadrículas por encima de umbral de riesgo (P &ge; 1.0%)"

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
            <div class="kpi-sub">Prob. Máx: <b>{max_prob:.2f}%</b> (Media: {mean_prob:.2f}% · <span title="Frecuencia basal diaria en Galicia: ~0.02%. Probabilidades > 5% representan riesgo extremo.">Base: ~0.02%</span>)</div>
        </div>
        """
        render_html_safely(card1_html)

    with c2:
        card2_html = f"""
        <div class="kpi-card {c2_class}">
            <div class="kpi-label">
                <span class="material-symbols-outlined">warning</span>
                {c2_label}
            </div>
            <div class="kpi-value">{c2_value}</div>
            <div class="kpi-sub">{c2_sub}</div>
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
