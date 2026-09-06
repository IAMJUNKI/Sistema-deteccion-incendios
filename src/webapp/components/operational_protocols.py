"""Componente de Protocolos Operativos PLADIGA y Generador de Informes con Google Material Symbols."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.webapp.components.header_kpis import get_day_severity_info


def generate_executive_briefing(
    df_data: pd.DataFrame,
    manifest: dict,
    selected_horizon: int,
    target_date: str,
) -> str:
    """Genera el texto estructurado del informe ejecutivo diario para mando."""
    total_cells = len(df_data)
    max_prob = df_data["prob_riesgo"].max() * 100
    mean_prob = df_data["prob_riesgo"].mean() * 100
    num_top05 = int((df_data["percentil_riesgo"] >= 0.995).sum())
    num_r30 = int(df_data["regla_30_30_activa"].sum())
    num_crit_12 = int((df_data["prob_riesgo"] >= 0.12).sum()) if "prob_riesgo" in df_data.columns else 0

    severity_info = get_day_severity_info(max_prob, num_crit_12)

    provider = manifest.get("forecast_provider", "MeteoGalicia WRF")
    issue_time = df_data.get("issue_time", pd.Series(["-"])).iloc[0]

    # Distrito con mayor riesgo
    distrito_top = "Galicia Sur"
    if "distrito_forestal" in df_data.columns:
        distrito_top = str(df_data.groupby("distrito_forestal")["prob_riesgo"].mean().idxmax())

    top_cells = (
        df_data.sort_values("prob_riesgo", ascending=False)
        .head(10)[["cell_id", "provincia", "distrito_forestal", "prob_riesgo", "recommended_action"]]
    )

    # Medidas operativas adaptadas a la severidad real
    if max_prob >= 6.0:
        aerea_text = f"Reforzar pasadas de reconocimiento continuo sobre el sector **{distrito_top}** en la ventana crítica de ignición (12:00–18:00 h)."
        helitrans_text = "Declarar nivel de alerta máxima y preposición en bases estratégicas (BRIF Laza, bases comarcales)."
        quemas_text = "Suspensión total e inmediata de permisos de quema agrícola y forestal en toda la comunidad."
    elif max_prob >= 2.5:
        aerea_text = f"Vigilancia aérea programada en solanas y áreas de interfaz del sector **{distrito_top}** (13:00–18:00 h)."
        helitrans_text = "Pre-alerta operativa y retenes preparados en bases comarcales próximas."
        quemas_text = "Suspensión de quemas agrícolas y forestales en los distritos en Nivel 3 y 4."
    else:
        aerea_text = f"Vigilancia rutinaria con atención preventiva al sector **{distrito_top}** en horas centrales."
        helitrans_text = "Retenes en régimen de disponibilidad ordinaria en sus bases habituales."
        quemas_text = "Monitoreo rutinario de autorizaciones de quema según normativa PLADIGA vigente."

    briefing = f"""# INFORME EJECUTIVO DE SITUACIÓN OPERATIVA — INCENDIOS FORESTALES
**Área Territorial:** Comunidad Autónoma de Galicia (Resolución 1 km × 1 km)
**Fecha Objetivo del Pronóstico:** {target_date} (Horizonte T+{selected_horizon} / {selected_horizon*24}h)
**Emisión Meteorológica:** {issue_time} (Proveedor: {provider})
**Sistema:** Plataforma Predictiva de Alerta Temprana (TFM)

---

## 1. RESUMEN DE AMENAZA Y TELEMETRÍA GLOBAL
- **Nivel de Severidad Global:** {severity_info['level']}
- **Probabilidad Máxima Calibrada P(Y=1):** {max_prob:.2f}% (Media Autonómica: {mean_prob:.2f}%)
- **Cuadrículas de Máxima Prioridad de Despacho (Top 0.5%):** {num_top05:,} km²
- **Superficie con Condición Crítica 30-30-30:** {num_r30:,} km² ({(num_r30/total_cells*100):.1f}% del territorio)
- **Sector de Máxima Atención Comarcal:** {distrito_top}

---

## 2. MEDIDAS Y PROTOCOLOS PREVENTIVOS RECOMENDADOS (PLADIGA)

### A. Preposición y Despacho de Medios
- **Vigilancia Aérea:** {aerea_text}
- **Brigadas Helitransportadas:** {helitrans_text}
- **Patrullaje Terrestre:** Presencia preventiva de agentes ambientales en pistas forestales de las celdas clasificadas en el Top 0.5%.

### B. Restricciones a la Población y Actividades en Monte
- **Permisos de Quema:** {quemas_text}
- **Trabajos con Maquinaria:** Restricción de desbrozadoras y maquinaria pesada forestal no asistida entre las 13:00 y las 18:00 h en áreas con condición 30-30-30 activa.

---

## 3. CELDAS PRIORITARIAS DE INTERVENCIÓN (TOP 10)
| Celda ID | Provincia | Distrito Forestal | Probabilidad P(Y=1) | Acción Táctica Asignada |
| :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in top_cells.iterrows():
        briefing += f"| #{int(row['cell_id'])} | {row['provincia']} | {row['distrito_forestal']} | {row['prob_riesgo']*100:.2f}% | {row['recommended_action'].replace('_', ' ').title()} |\n"

    briefing += """
---
*Nota de Gobernanza: Las probabilidades reportadas representan el valor calibrado del modelo supervisado LightGBM validado por bloques temporales anuales completos. La prioridad orienta la preposición de medios y no sustituye el juicio de la dirección técnica de extinción.*
"""
    return briefing


def render_operational_protocols_tab(
    df_data: pd.DataFrame,
    manifest: dict,
    selected_horizon: int,
    target_date: str,
) -> None:
    """Renderiza la pestaña de protocolos de despacho e informes de situación."""
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:22px; color:#60a5fa;">shield</span>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Protocolos de Intervención y Despacho Operativo (PLADIGA)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Directrices de actuación táctica coordinadas con el Plan de Prevención y Defensa contra "
        "los Incendios Forestales de Galicia."
    )

    if df_data.empty:
        st.warning("No hay datos disponibles para protocolos operativos.")
        return

    # Matriz de Acción Preventiva
    st.markdown("#### Matriz de Activación de Recursos por Nivel de Riesgo")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(
            """
            <div class="action-box" style="border-left: 3px solid #800026; margin-bottom: 0.75rem;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#800026;">emergency</span>
                    Nivel 5 — Riesgo Extremo (P &ge; 12.0%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Preposición Aérea:</b> Alerta máxima y despliegue preventivo de medios helitransportados.<br/>
                    • <b>Prohibición Total:</b> Suspensión fulminante de todo permiso de quema agrícola o forestal.<br/>
                    • <b>Restricción de Accesos:</b> Cierre de pistas forestales y vigilancia fija al 100% con cámaras térmicas.
                </div>
            </div>

            <div class="action-box" style="border-left: 3px solid #dc2626; margin-bottom: 0.75rem;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#dc2626;">warning</span>
                    Nivel 4 — Riesgo Muy Alto (P: 6.0% &ndash; 12.0%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Vigilancia Aérea:</b> Vuelos de patrulla continua en solanas durante la ventana crítica (12:00–18:00 h).<br/>
                    • <b>Patrullaje Terrestre:</b> Presencia disuasoria reforzada de agentes ambientales en celdas prioritarias.<br/>
                    • <b>Protección Civil:</b> Preaviso a corporaciones locales y retenes en alerta de salida inmediata.
                </div>
            </div>

            <div class="action-box" style="border-left: 3px solid #ea580c;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#ea580c;">report_problem</span>
                    Nivel 3 — Riesgo Alto (P: 2.5% &ndash; 6.0%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Vigilancia Programada:</b> Rondas terrestres preventivas en pistas y áreas de interfaz urbano-forestal.<br/>
                    • <b>Restricción de Quemas:</b> Suspensión temporal de quemas en distritos afectados.<br/>
                    • <b>Retenes en Base:</b> Brigadas en régimen de prealerta operativa preparadas para conatos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class="action-box" style="border-left: 3px solid #d97706; margin-bottom: 0.75rem;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#d97706;">info</span>
                    Nivel 2 — Riesgo Moderado (P: 1.0% &ndash; 2.5%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Monitoreo Preventivo:</b> Seguimiento horario de rachas de viento y temperaturas en horas centrales.<br/>
                    • <b>Control Habitual:</b> Supervisión estándar de quemas según normativa PLADIGA vigente.<br/>
                    • <b>Disponibilidad Ordinaria:</b> Retenes y parques comarcales en turnos regulares de vigilancia.
                </div>
            </div>

            <div class="action-box" style="border-left: 3px solid #059669;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#059669;">check_circle</span>
                    Nivel 1 — Riesgo Bajo / Nominal (P &lt; 1.0%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Operatividad Estándar:</b> Sin restricciones extraordinarias a la población ni a trabajos en monte.<br/>
                    • <b>Mantenimiento Preventivo:</b> Selvicultura preventiva y limpieza de fajas de protección secundarias.<br/>
                    • <b>Seguimiento Remoto:</b> Detección satelital rutinaria de anomalías térmicas (NASA FIRMS).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Generador de Briefing Ejecutivo
    st.markdown("#### Generador de Informe de Situación (Briefing Diario)")
    st.caption("Documento estructurado listo para reunión del comité de coordinación o exportación.")

    briefing_text = generate_executive_briefing(df_data, manifest, selected_horizon, target_date)

    with st.expander("Vista Previa del Informe de Situación", expanded=True):
        st.markdown(briefing_text)

    col_btn1, _ = st.columns([1, 2])
    with col_btn1:
        st.download_button(
            label="Descargar Informe de Situación (Markdown)",
            data=briefing_text,
            file_name=f"informe_situacion_incendios_{target_date}_T+{selected_horizon}.md",
            mime="text/markdown",
        )
