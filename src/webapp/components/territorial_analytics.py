"""Componente de Analítica Territorial y Estadísticas Comarcales con Google Material Symbols."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_territorial_analytics_tab(df_data: pd.DataFrame, all_predictions: pd.DataFrame) -> None:
    """Renderiza el cuadro de mando de analítica territorial agregada."""
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
            <span class="material-symbols-outlined" style="font-size:22px; color:#60a5fa;">analytics</span>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Analítica Territorial y Estadísticas Comarcales</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Agregación espacial de riesgo por Provincias, Distritos Forestales y Municipios.")

    if df_data.empty:
        st.warning("No hay datos disponibles para el análisis territorial.")
        return

    # 1. Resumen por Provincias
    st.markdown("#### Distribución de Riesgo por Provincias")
    
    prov_summary = (
        df_data.groupby("provincia")
        .agg(
            total_celdas=("cell_id", "count"),
            prob_media=("prob_riesgo", lambda s: s.mean() * 100),
            prob_maxima=("prob_riesgo", lambda s: s.max() * 100),
            celdas_criticas=("percentil_riesgo", lambda s: (s >= 0.95).sum()),
            regla_30_30=("regla_30_30_activa", "sum"),
        )
        .reset_index()
        .sort_values("prob_media", ascending=False)
    )

    col_prov_table, col_prov_chart = st.columns([1.2, 1.0])

    with col_prov_table:
        display_prov = prov_summary.rename(
            columns={
                "provincia": "Provincia",
                "total_celdas": "Celdas (km²)",
                "prob_media": "Prob. Media (%)",
                "prob_maxima": "Prob. Máx (%)",
                "celdas_criticas": "Top 5% Críticas",
                "regla_30_30": "Condición 30-30-30",
            }
        )
        st.dataframe(
            display_prov.style.format(
                {
                    "Prob. Media (%)": "{:.2f}%",
                    "Prob. Máx (%)": "{:.2f}%",
                    "Celdas (km²)": "{:,}",
                    "Top 5% Críticas": "{:,}",
                    "Condición 30-30-30": "{:,}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_prov_chart:
        chart_data = prov_summary.set_index("provincia")[["prob_media", "prob_maxima"]]
        chart_data.columns = ["Riesgo Medio (%)", "Riesgo Máximo (%)"]
        st.bar_chart(chart_data)

    st.markdown("---")

    # 2. Ranking de Distritos Forestales
    st.markdown("#### Ranking de Distritos Forestales por Severidad")
    
    if "distrito_forestal" in df_data.columns:
        dist_summary = (
            df_data.groupby("distrito_forestal")
            .agg(
                provincia=("provincia", "first"),
                celdas=("cell_id", "count"),
                prob_media=("prob_riesgo", lambda s: s.mean() * 100),
                prob_max=("prob_riesgo", lambda s: s.max() * 100),
                celdas_top5=("percentil_riesgo", lambda s: (s >= 0.95).sum()),
                celdas_top05=("percentil_riesgo", lambda s: (s >= 0.995).sum()),
            )
            .reset_index()
            .sort_values("prob_media", ascending=False)
        )

        st.dataframe(
            dist_summary.rename(
                columns={
                    "distrito_forestal": "Distrito Forestal / Comarca",
                    "provincia": "Provincia",
                    "celdas": "Área (km²)",
                    "prob_media": "Riesgo Medio (%)",
                    "prob_max": "Riesgo Máx (%)",
                    "celdas_top5": "Alerta Urgente (Top 5%)",
                    "celdas_top05": "Extremo (Top 0.5%)",
                }
            ).style.format(
                {
                    "Riesgo Medio (%)": "{:.2f}%",
                    "Riesgo Máx (%)": "{:.2f}%",
                    "Área (km²)": "{:,}",
                    "Alerta Urgente (Top 5%)": "{:,}",
                    "Extremo (Top 0.5%)": "{:,}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # 3. Top Celdas Críticas y Exportación de Datos
    st.markdown("#### Tabla de Celdas Prioritarias de Intervención")
    
    c_filter_prov, c_filter_limit = st.columns([1, 1])
    with c_filter_prov:
        filter_p = st.selectbox("Filtrar por Provincia:", ["Todas"] + sorted(df_data["provincia"].unique()))
    with c_filter_limit:
        top_n = st.slider("Número de celdas a listar:", min_value=20, max_value=500, value=50, step=10)

    df_filtered = df_data.copy()
    if filter_p != "Todas":
        df_filtered = df_filtered[df_filtered["provincia"] == filter_p]

    top_table = (
        df_filtered.sort_values("prob_riesgo", ascending=False)
        .head(top_n)[
            [
                "cell_id",
                "provincia",
                "distrito_forestal",
                "prob_riesgo",
                "percentil_riesgo",
                "tmax_vc",
                "rhmin_vc",
                "vmax_vc",
                "prec_acum_30d",
                "recommended_action",
            ]
        ]
        .copy()
    )

    top_table["prob_riesgo"] = top_table["prob_riesgo"] * 100
    top_table["percentil_riesgo"] = top_table["percentil_riesgo"] * 100
    top_table["recommended_action"] = top_table["recommended_action"].str.replace("_", " ").str.title()

    st.dataframe(
        top_table.rename(
            columns={
                "cell_id": "Celda ID",
                "provincia": "Provincia",
                "distrito_forestal": "Distrito Forestal",
                "prob_riesgo": "P(Y=1) (%)",
                "percentil_riesgo": "Percentil (%)",
                "tmax_vc": "T. Máx (°C)",
                "rhmin_vc": "HR Mín (%)",
                "vmax_vc": "Viento (km/h)",
                "prec_acum_30d": "Lluvia 30d (mm)",
                "recommended_action": "Acción Preventiva Sugerida",
            }
        ).style.format(
            {
                "P(Y=1) (%)": "{:.2f}%",
                "Percentil (%)": "{:.1f}%",
                "T. Máx (°C)": "{:.1f}",
                "HR Mín (%)": "{:.1f}",
                "Viento (km/h)": "{:.1f}",
                "Lluvia 30d (mm)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Botón de Descarga CSV
    csv_data = top_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar Celdas Prioritarias (CSV)",
        data=csv_data,
        file_name=f"priorizacion_celdas_incendios_galicia_top{top_n}.csv",
        mime="text/csv",
    )

    # 4. Comparativa Multi-Horizonte (si hay T+1, T+2, T+3)
    if not all_predictions.empty and "horizon_days" in all_predictions.columns:
        unique_horizons = sorted(all_predictions["horizon_days"].dropna().unique())
        if len(unique_horizons) > 1:
            st.markdown("---")
            st.markdown("#### Evolución Temporal del Riesgo a 72 Horas (T+1 vs T+2 vs T+3)")
            
            evo_data = (
                all_predictions.groupby(["horizon_days", "provincia"])["prob_riesgo"]
                .mean()
                .unstack(level=1)
                * 100
            )
            evo_data.index = [f"T+{int(h)} ({int(h)*24}h)" for h in evo_data.index]
            
            st.line_chart(evo_data)
