"""Componente de Consulta Rápida por Municipio (Ficha Semafórica para Ayuntamientos)."""

from __future__ import annotations

import re
import pandas as pd
import streamlit as st

from src.webapp.utils.geo_helpers import CONCELLOS_GALICIA


def render_html_safely(html_str: str) -> None:
    """Renderiza HTML de forma segura eliminando cualquier sangría que Markdown interprete como código."""
    clean_html = re.sub(r"^[ \t]+", "", html_str, flags=re.MULTILINE).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


def get_risk_level_info(prob: float) -> dict:
    """Traduce la probabilidad matemática a una escala operativa accesible (Nivel 1 a 5)."""
    p_pct = prob * 100
    if p_pct >= 12.0:
        return {
            "level": "Nivel 5 — Extremo",
            "badge_color": "#800026",
            "bg_color": "rgba(128, 0, 38, 0.25)",
            "border_color": "#800026",
            "icon": "emergency",
            "description": "Peligro crítico de incendios de rápida propagación. Condiciones meteorológicas extremas.",
            "recommendation": "Preposición inmediata de brigadas helitransportadas, prohibición total de quemas y vigilancia en pistas forestales.",
        }
    elif p_pct >= 6.0:
        return {
            "level": "Nivel 4 — Muy Alto",
            "badge_color": "#dc2626",
            "bg_color": "rgba(220, 38, 38, 0.25)",
            "border_color": "#dc2626",
            "icon": "warning",
            "description": "Alta probabilidad de ignición y propagación rápida ante cualquier foco.",
            "recommendation": "Patrullas preventivas en solanas, vigilancia aérea reforzada y aviso a Protección Civil local.",
        }
    elif p_pct >= 2.5:
        return {
            "level": "Nivel 3 — Alto",
            "badge_color": "#ea580c",
            "bg_color": "rgba(234, 88, 12, 0.25)",
            "border_color": "#ea580c",
            "icon": "report_problem",
            "description": "Condiciones propicias para conatos de incendio en zonas de matorral o monte bajo.",
            "recommendation": "Supervisión de actividades agrícolas y retén preparado para salida rápida.",
        }
    elif p_pct >= 1.0:
        return {
            "level": "Nivel 2 — Moderado",
            "badge_color": "#d97706",
            "bg_color": "rgba(217, 119, 6, 0.25)",
            "border_color": "#d97706",
            "icon": "info",
            "description": "Riesgo moderado controlado. Sin factores extremos inmediatos.",
            "recommendation": "Monitoreo rutinario y control habitual de quemas.",
        }
    else:
        return {
            "level": "Nivel 1 — Bajo / Nominal",
            "badge_color": "#059669",
            "bg_color": "rgba(5, 150, 105, 0.25)",
            "border_color": "#059669",
            "icon": "check_circle",
            "description": "Sin riesgo relevante previsto. Humedad favorable o temperaturas suaves.",
            "recommendation": "Operatividad estándar sin restricciones adicionales.",
        }


def render_concello_lookup_tab(df_data: pd.DataFrame, target_date: str) -> None:
    """Renderiza la pestaña de búsqueda y ficha semafórica municipal."""
    render_html_safely(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:22px; color:#60a5fa;">location_city</span>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Consulta y Ficha de Situación por Concello</h3>
        </div>
        """
    )
    st.caption("Herramienta directa para alcaldes, técnicos de Protección Civil y jefes de retén local.")

    if df_data.empty:
        st.warning("No hay datos disponibles para la consulta municipal.")
        return

    col_search, col_spacer = st.columns([1.5, 1.0])

    with col_search:
        municipio_list = sorted(CONCELLOS_GALICIA.keys())
        selected_muni = st.selectbox(
            "Selecciona o busca un municipio de Galicia:",
            municipio_list,
            index=municipio_list.index("Verín") if "Verín" in municipio_list else 0,
        )

    muni_coords = CONCELLOS_GALICIA.get(selected_muni, {"lat": 42.6, "lon": -7.8, "provincia": "Ourense"})

    # Buscar la celda más cercana al municipio
    df_with_coords = df_data.dropna(subset=["lat_centroid", "lon_centroid"]).copy()
    if not df_with_coords.empty:
        dists = (df_with_coords["lat_centroid"] - muni_coords["lat"]) ** 2 + (
            df_with_coords["lon_centroid"] - muni_coords["lon"]
        ) ** 2
        nearest_row = df_with_coords.loc[dists.idxmin()]
    else:
        nearest_row = df_data.iloc[0]

    prob_val = float(nearest_row.get("prob_riesgo", 0.02))
    risk_info = get_risk_level_info(prob_val)

    tmax = float(nearest_row.get("tmax_vc", 28.0))
    rhmin = float(nearest_row.get("rhmin_vc", 40.0))
    vmax = float(nearest_row.get("vmax_vc", 18.0))
    prec30 = float(nearest_row.get("prec_acum_30d", 5.0))
    distrito = str(nearest_row.get("distrito_forestal", "Distrito Forestal Galicia"))

    # Renderizar Tarjeta Semafórica Municipal de forma segura
    card_html = f"""
    <div style="background: #111827; border: 2px solid {risk_info['border_color']}; border-radius: 8px; padding: 1.4rem; margin-top: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">
                    Ficha Municipal de Emergencias · {muni_coords['provincia']} · {distrito}
                </span>
                <h2 style="margin: 0.2rem 0; font-size: 1.6rem; color: #f8fafc;">
                    Concello de {selected_muni}
                </h2>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 0.25rem;">
                    Pronóstico válido para: <b>{target_date}</b> (Ventana crítica: <b>13:00 h – 18:00 h</b>)
                </div>
            </div>
            <div style="background: {risk_info['bg_color']}; border: 1px solid {risk_info['border_color']}; padding: 0.6rem 1.1rem; border-radius: 6px; text-align: right;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #cbd5e1; font-weight: 600;">Nivel de Amenaza</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {risk_info['badge_color']}; display: flex; align-items: center; gap: 0.4rem;">
                    <span class="material-symbols-outlined">{risk_info['icon']}</span>
                    {risk_info['level']}
                </div>
            </div>
        </div>

        <hr style="border: 0; border-top: 1px solid #1f2937; margin: 1rem 0;" />

        <div style="font-size: 0.92rem; color: #e2e8f0; margin-bottom: 1rem; line-height: 1.5;">
            <b>Diagnóstico de Situación:</b> {risk_info['description']}
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; margin-bottom: 1.25rem;">
            <div style="background: #0f172a; padding: 0.75rem; border-radius: 6px; border: 1px solid #1e293b;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Temperatura Máxima</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {'#ef4444' if tmax>=30 else '#f8fafc'};">{tmax:.1f} °C</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">{'Supera umbral de 30°C' if tmax>=30 else 'Temperatura moderada'}</div>
            </div>
            <div style="background: #0f172a; padding: 0.75rem; border-radius: 6px; border: 1px solid #1e293b;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Humedad Mínima</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {'#ef4444' if rhmin<=30 else '#f8fafc'};">{rhmin:.1f} %</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">{'Aire muy seco (<30%)' if rhmin<=30 else 'Humedad suficiente'}</div>
            </div>
            <div style="background: #0f172a; padding: 0.75rem; border-radius: 6px; border: 1px solid #1e293b;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Rachas de Viento</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {'#f59e0b' if vmax>=25 else '#f8fafc'};">{vmax:.1f} km/h</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">{'Viento relevante en ladera' if vmax>=25 else 'Viento en calma'}</div>
            </div>
            <div style="background: #0f172a; padding: 0.75rem; border-radius: 6px; border: 1px solid #1e293b;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Lluvia Últimos 30 Días</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {'#ef4444' if prec30<10 else '#f8fafc'};">{prec30:.1f} mm</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">{'Déficit hídrico severo' if prec30<10 else 'Suelo húmedo'}</div>
            </div>
        </div>

        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; border-radius: 4px; padding: 0.9rem 1.1rem;">
            <div style="font-weight: 600; color: #f8fafc; font-size: 0.88rem; display: flex; align-items: center; gap: 0.35rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; color: #38bdf8;">verified_user</span>
                Recomendaciones para Protección Civil y Autoridades Locales
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 0.35rem; line-height: 1.5;">
                {risk_info['recommendation']}
            </div>
        </div>
    </div>
    """
    render_html_safely(card_html)
