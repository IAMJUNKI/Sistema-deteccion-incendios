"""Componente de Protocolos Operativos PLADIGA y Generador de Informes con Google Material Symbols."""

from __future__ import annotations

import pandas as pd
import streamlit as st


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
    num_top5 = int((df_data["percentil_riesgo"] >= 0.95).sum())
    num_top05 = int((df_data["percentil_riesgo"] >= 0.995).sum())
    num_r30 = int(df_data["regla_30_30_activa"].sum())

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

    briefing = f"""# INFORME EJECUTIVO DE SITUACIÓN OPERATIVA — INCENDIOS FORESTALES
**Área Territorial:** Comunidad Autónoma de Galicia (Resolución 1 km × 1 km)
**Fecha Objetivo del Pronóstico:** {target_date} (Horizonte T+{selected_horizon} / {selected_horizon*24}h)
**Emisión Meteorológica:** {issue_time} (Proveedor: {provider})
**Sistema:** Plataforma Predictiva de Alerta Temprana (TFM)

---

## 1. RESUMEN DE AMENAZA Y TELEMETRÍA GLOBAL
- **Nivel de Severidad Global:** {'Nivel 3 — Extremo' if num_top05 > 50 or max_prob > 12 else 'Nivel 2 — Alto' if max_prob > 6 else 'Nivel 1 — Moderado'}
- **Probabilidad Máxima Calibrada P(Y=1):** {max_prob:.2f}% (Media Autonómica: {mean_prob:.2f}%)
- **Celdas Críticas en Alerta Urgente (Top 5%):** {num_top5:,} km²
- **Celdas de Máxima Prioridad (Top 0.5% Extremo):** {num_top05:,} km²
- **Superficie con Condición Crítica 30-30-30:** {num_r30:,} km² ({(num_r30/total_cells*100):.1f}% del territorio)
- **Sector de Máxima Atención Comarcal:** {distrito_top}

---

## 2. MEDIDAS Y PROTOCOLOS PREVENTIVOS RECOMENDADOS (PLADIGA)

### A. Preposición y Despacho de Medios
- **Vigilancia Aérea:** Reforzar pasadas de reconocimiento sobre el sector **{distrito_top}** en la ventana crítica de ignición (12:00–18:00 h).
- **Brigadas Helitransportadas:** Declarar nivel de pre-alerta operativa en bases estratégicas del cuadrante sur-oriental (BRIF Laza, bases comarcales).
- **Patrullaje Terrestre:** Presencia preventiva de agentes ambientales en pistas forestales de las celdas clasificadas en el Top 0.5%.

### B. Restricciones a la Población y Actividades en Monte
- **Permisos de Quema:** Suspensión temporal de quemas agrícolas y forestales en los distritos en Nivel 2 y 3.
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
            <div class="action-box" style="border-left: 3px solid #800026;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#800026;">emergency</span>
                    Nivel 3 — Riesgo Extremo (Top 0.5% / P &gt; 15%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Preposición Helitransportada:</b> Alerta inmediata a bases del cuadrante sur.<br/>
                    • <b>Restricción de Accesos:</b> Cierre preventivo de pistas forestales no esenciales.<br/>
                    • <b>Vigilancia Fija:</b> Operación 100% de torretas de observación con cámaras térmicas.<br/>
                    • <b>Prohibición Absoluta:</b> Quemas agrícolas y desbroces mecánicos no asistidos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="action-box" style="border-left: 3px solid #dc2626;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#dc2626;">warning</span>
                    Nivel 2 — Riesgo Alto (Top 5.0% / P &gt; 5%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Vigilancia Aérea:</b> Vuelos de patrulla preventiva en ventana crítica (12:00–18:00 h).<br/>
                    • <b>Patrullas Terrestres:</b> Presencia disuasoria de agentes ambientales en solanas.<br/>
                    • <b>Aviso a Protección Civil:</b> Comunicación a corporaciones locales en alerta.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class="action-box" style="border-left: 3px solid #d97706;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#d97706;">info</span>
                    Nivel 1 — Riesgo Moderado-Alto (Top 10%)
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Control de Permisos:</b> Supervisión de quemas autorizadas.<br/>
                    • <b>Monitoreo Meteorológico:</b> Seguimiento horario de rachas de viento y humedad.<br/>
                    • <b>Medios en Guardia:</b> Retenes en parque preparados para salida inmediata.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="action-box" style="border-left: 3px solid #059669;">
                <div class="action-title">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#059669;">check_circle</span>
                    Nivel 0 — Riesgo Basal / Moderado
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    • <b>Monitoreo Rutinario:</b> Registro satelital de anomalías térmicas (FIRMS).<br/>
                    • <b>Mantenimiento Preventivo:</b> Limpieza de fajas de protección secundarias.
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
