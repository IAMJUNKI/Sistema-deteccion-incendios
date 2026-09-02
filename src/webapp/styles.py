"""Sistema de diseño visual profesional e institucional con Google Material Symbols."""

import streamlit as st

CUSTOM_CSS = """
<style>
/* ==========================================================================
   TIPOGRAFÍA E ICONOGRAFÍA PROFESIONAL (GOOGLE MATERIAL SYMBOLS + INTER)
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    vertical-align: -3px;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #e2e8f0;
}

/* Reducir padding de la página principal para sensación de software de escritorio */
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ==========================================================================
   CABECERA INSTITUCIONAL (COMMAND CENTER HEADER)
   ========================================================================== */
.command-header {
    background: #0d1322;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

.command-title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0;
}

.command-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* ==========================================================================
   TARJETAS KPI Y TELEMETRÍA (STAT METRIC CARDS)
   ========================================================================== */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.kpi-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    position: relative;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    transition: border-color 0.15s ease;
}

.kpi-card:hover {
    border-color: #374151;
}

.kpi-card.alert-critical {
    border-top: 3px solid #dc2626;
}

.kpi-card.alert-warning {
    border-top: 3px solid #d97706;
}

.kpi-card.alert-info {
    border-top: 3px solid #2563eb;
}

.kpi-card.alert-success {
    border-top: 3px solid #059669;
}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f9fafb;
    line-height: 1.15;
    font-family: 'Inter', sans-serif;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

/* ==========================================================================
   BADGES DE ESTADO Y ETIQUETAS DISCRETAS
   ========================================================================== */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-fresh {
    background-color: rgba(5, 150, 105, 0.15);
    color: #34d399;
    border: 1px solid rgba(5, 150, 105, 0.35);
}

.badge-fallback {
    background-color: rgba(217, 119, 6, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(217, 119, 6, 0.35);
}

.badge-stale {
    background-color: rgba(220, 38, 38, 0.15);
    color: #f87171;
    border: 1px solid rgba(220, 38, 38, 0.35);
}

/* ==========================================================================
   ESTILIZACIÓN DE PESTAÑAS (TABS)
   ========================================================================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #0b0f19;
    padding: 4px;
    border-radius: 6px;
    border: 1px solid #1e293b;
    margin-bottom: 0.75rem;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 4px;
    color: #94a3b8;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0 14px;
    border: none !important;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #60a5fa !important;
    font-weight: 600 !important;
}

/* ==========================================================================
   PANELES Y CAJAS DE INFORMACIÓN TÉCNICA
   ========================================================================== */
.action-box {
    background: #111827;
    border-radius: 6px;
    border: 1px solid #1f2937;
    padding: 1.1rem;
    margin-bottom: 0.85rem;
}

.action-title {
    font-weight: 600;
    font-size: 0.92rem;
    color: #f3f4f6;
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ==========================================================================
   CONTENEDOR DEL MAPA Y ELEMENTOS GIS
   ========================================================================== */
iframe {
    border-radius: 6px !important;
    border: 1px solid #1e293b !important;
}

/* ==========================================================================
   TABLAS Y ELEMENTOS NATIVOS STREAMLIT
   ========================================================================== */
.stDataFrame {
    border-radius: 6px !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
}

</style>
"""


def apply_custom_styles() -> None:
    """Inyecta los estilos CSS profesionales y la librería Material Symbols."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
