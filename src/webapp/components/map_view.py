"""Componente del Mapa GIS Táctico Interactivo con Google Material Symbols y Alto Rendimiento."""

from __future__ import annotations

import textwrap
from collections import OrderedDict

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import folium
    from folium import plugins
except ImportError:  # pragma: no cover - se prueba en instalaciones sin extras GIS
    folium = None
    plugins = None

from src.webapp.components.sidebar import MAP_STYLES
from src.webapp.utils.geo_helpers import (
    SECTOR_PLADIGA_DISTRITOS,
    ZOOM_PRESETS,
    build_complex_dissolved_shapes,
    load_galicia_focus_layers,
    load_galicia_sectors_geojson,
)

# Folium genera cientos de objetos Python y serializa hasta 250 popups por
# mapa. El HTML final se comparte entre sesiones del mismo proceso de
# Streamlit para que un segundo usuario no tenga que reconstruirlo ni volver a
# serializar Folium. Guardar HTML, en vez del objeto Folium, evita reutilizar
# los identificadores JavaScript internos de GeoJson/Leaflet entre reruns.
_MAP_CACHE: OrderedDict[tuple[object, ...], str] = OrderedDict()
_MAP_CACHE_MAX_ENTRIES = 24


def get_color_gradient(prob: float, pct: float, mode: str) -> str:
    """Devuelve un color de la escala cromática de riesgo según el modo de visualización."""
    if "Probabilidad" in mode or "Absoluto" in mode:
        val = prob * 100
        if val >= 12.0:
            return "#800026"  # Nivel 5 — Extremo (P >= 12.0%)
        if val >= 6.0:
            return "#BD0026"  # Nivel 4 — Muy Alto (6.0% - 12.0%)
        if val >= 2.5:
            return "#E31A1C"  # Nivel 3 — Alto (2.5% - 6.0%)
        if val >= 1.0:
            return "#FD8D3C"  # Nivel 2 — Moderado (1.0% - 2.5%)
        return "#FED976"      # Nivel 1 — Bajo / Nominal (< 1.0%)
    elif "Percentil" in mode or "Relativa" in mode:
        val = pct * 100
        if val >= 99.8:
            return "#800026"  # Top 0.2% Crítico
        if val >= 99.5:
            return "#BD0026"  # Top 0.5% Muy Alto
        if val >= 98.0:
            return "#E31A1C"  # Top 2.0% Prioritario
        if val >= 95.0:
            return "#FC4E2A"  # Top 5.0% Despacho
        if val >= 90.0:
            return "#FD8D3C"  # Top 10.0% Vigilancia
        if val >= 80.0:
            return "#FEB24C"  # Top 20.0%
        return "#FED976"
    else:
        if pct >= 0.998:
            return "#800026"  # Nivel 5: Crítico (Top 0.2%)
        if pct >= 0.995:
            return "#BD0026"  # Nivel 4: Muy Alto (Top 0.5%)
        if pct >= 0.980:
            return "#E31A1C"  # Nivel 3: Alto (Top 2.0%)
        if pct >= 0.950:
            return "#FD8D3C"  # Nivel 2: Moderado (Top 5.0%)
        return "#FED976"      # Nivel 1: Bajo / Nominal (< 95.0%)


def build_map_legend_html(color_mode: str) -> str:
    """Genera la leyenda cartográfica adaptada dinámicamente al modo de simbología seleccionado."""
    if "Probabilidad" in color_mode or "Absoluto" in color_mode:
        title = "Escala: Probabilidad P(Y=1)"
        items = """
            <span style="background:#800026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 5 — Extremo (&ge; 12.0%)<br/>
            <span style="background:#BD0026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 4 — Muy Alto (6.0% &ndash; 12.0%)<br/>
            <span style="background:#E31A1C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 3 — Alto (2.5% &ndash; 6.0%)<br/>
            <span style="background:#FD8D3C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 2 — Moderado (1.0% &ndash; 2.5%)<br/>
            <span style="background:#FED976;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 1 — Bajo / Nominal (&lt; 1.0%)
        """
        subtitle = "Severidad física calibrada (5 Niveles)"
    elif "Percentil" in color_mode or "Relativa" in color_mode:
        title = "Escala: Priorización Relativa"
        items = """
            <span style="background:#800026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Top 0.2% Crítico (&ge; 99.8%)<br/>
            <span style="background:#BD0026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Top 0.5% Muy Alto (&ge; 99.5%)<br/>
            <span style="background:#E31A1C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Top 2.0% Prioritario (&ge; 98.0%)<br/>
            <span style="background:#FC4E2A;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Top 5.0% Despacho (&ge; 95.0%)<br/>
            <span style="background:#FD8D3C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Top 10.0% Vigilancia (&ge; 90.0%)<br/>
            <span style="background:#FEB24C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Top 20.0% / Resto
        """
        subtitle = "Ranking relativo para despacho"
    else:
        title = "Escala: Niveles Tácticos (Top %)"
        items = """
            <span style="background:#800026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 5: Crítico (Top 0.2% / &ge; 99.8%)<br/>
            <span style="background:#BD0026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 4: Muy Alto (Top 0.5% / &ge; 99.5%)<br/>
            <span style="background:#E31A1C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 3: Alto (Top 2.0% / &ge; 98.0%)<br/>
            <span style="background:#FD8D3C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 2: Moderado (Top 5.0% / &ge; 95.0%)<br/>
            <span style="background:#FED976;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Nivel 1: Bajo / Nominal (&lt; 95.0%)
        """
        subtitle = "Tramos discretos de intervención (5 Niveles)"

    return f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background: rgba(15, 23, 42, 0.95); color: #f8fafc; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.4); font-family: 'Inter', sans-serif; font-size: 11px;
                border: 1px solid #334155; min-width: 175px;">
        <div style="font-weight:700; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:5px; color:#cbd5e1;">{title}</div>
        <div style="line-height: 1.6;">
            {items}
        </div>
        <div style="font-size:9px; color:#94a3b8; margin-top:5px; border-top:1px solid #334155; padding-top:4px;">
            {subtitle}
        </div>
    </div>
    """


def _render_map_html(map_html: str) -> None:
    """Muestra un mapa Folium en un iframe aislado sin rerenderizar Leaflet."""
    components.html(map_html, height=620, scrolling=False)


def render_map_tab(
    df_data: pd.DataFrame,
    selected_horizon: int,
    map_style: str = "IGN España (Ortofoto Oficial)",
    color_mode: str = "Riesgo Absoluto Calibrado P(Y=1)",
    filter_risk: str = "Top 5.0% Celdas en Riesgo Elevado",
) -> None:
    """Renderiza la pestaña del Centro de Mando Cartográfico con rendimiento optimizado."""
    st.markdown(
        textwrap.dedent(
            """
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.75rem;">
                <span class="material-symbols-outlined" style="font-size:22px; color:#60a5fa;">map</span>
                <h3 style="margin:0; font-size:1.15rem; font-weight:700;">Visualizador Geoespacial de Riesgo a 1 km²</h3>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    if df_data.empty:
        st.warning("No hay datos para renderizar el mapa.")
        return
    if folium is None:
        st.error(
            "El mapa requiere folium. Activa el entorno del proyecto y "
            "verifica las dependencias de `environment.yml`."
        )
        return

    # Asegurar coordenadas si no están presentes
    df_ready = df_data.copy()
    if "lat_centroid" not in df_ready.columns and "lat" in df_ready.columns:
        df_ready["lat_centroid"] = df_ready["lat"]
    if "lon_centroid" not in df_ready.columns and "lon" in df_ready.columns:
        df_ready["lon_centroid"] = df_ready["lon"]
    if ("lat_centroid" not in df_ready.columns or "lon_centroid" not in df_ready.columns) and "geometry" in df_ready.columns:
        try:
            df_ready["lat_centroid"] = df_ready["geometry"].map(lambda g: g.centroid.y if hasattr(g, "centroid") else None)
            df_ready["lon_centroid"] = df_ready["geometry"].map(lambda g: g.centroid.x if hasattr(g, "centroid") else None)
        except Exception:
            pass

    # Barra de Controles Rápidos del Mapa
    col_preset, col_map_style, col_search = st.columns([1.3, 1.4, 0.9])

    with col_preset:
        preset_name = st.selectbox(
            "Sector Territorial:",
            list(ZOOM_PRESETS.keys()),
            index=0,
            help="Enfoca una comarca o provincia con demarcación oficial PLADIGA.",
        )
        selected_preset = ZOOM_PRESETS[preset_name]

    with col_map_style:
        default_style_idx = MAP_STYLES.index(map_style) if map_style in MAP_STYLES else 0
        active_map_style = st.selectbox(
            "Capa Base Cartográfica:",
            MAP_STYLES,
            index=default_style_idx,
            help="Selecciona el estilo de mapa base (relieve físico, satélite, lienzo claro/oscuro o topográfico).",
        )

    with col_search:
        search_cell = st.text_input(
            "Localizar Celda ID:",
            value="",
            placeholder="Ej. 6818",
            help="Introduce el ID de una cuadrícula para marcarla en el mapa.",
        )

    # La caché de datos de Streamlit ya evita releer el parquet, pero el mapa
    # Folium se construye fuera de esa caché. Este fingerprint solo usa campos
    # que pueden cambiar con una nueva emisión y evita incluir geometrías
    # Shapely en la clave.
    fingerprint_columns = [
        column
        for column in [
            "dashboard_cache_version",
            "fecha",
            "issue_time",
            "horizon_days",
            "cell_id",
            "prob_riesgo",
            "percentil_riesgo",
            "lat_centroid",
            "lon_centroid",
            "recommended_action",
            "combustible_clase",
        ]
        if column in df_ready.columns
    ]
    try:
        data_fingerprint = int(
            pd.util.hash_pandas_object(
                df_ready[fingerprint_columns], index=True
            ).sum()
        )
    except (TypeError, ValueError):
        data_fingerprint = (len(df_ready), tuple(df_ready.columns))
    map_cache_key = (
        data_fingerprint,
        preset_name,
        active_map_style,
        color_mode,
        filter_risk,
        search_cell.strip(),
    )
    cached_map = _MAP_CACHE.get(map_cache_key)
    if cached_map is not None:
        _MAP_CACHE.move_to_end(map_cache_key)
        _render_map_html(cached_map)
        return

    # Orientación táctica contextual si se combina filtro estrecho con modo percentil
    if ("Top 0.5%" in filter_risk or "Top 1.0%" in filter_risk) and ("Percentil" in color_mode or "Relativa" in color_mode):
        st.info(
            "💡 **Guía Táctica de Interpretación:** Al filtrar por el Top 0.5% en modo *'Priorización Relativa'*, "
            "la escala muestra la sub-graduación del percentil superior. Para evaluar la probabilidad física real "
            "calibrada (ej. de 1.2% a 1.88%), activa **'Riesgo Absoluto Calibrado P(Y=1)'** en la barra lateral.",
            icon="ℹ️",
        )

    # Determinar centro y zoom según preset seleccionado
    center_lat, center_lon = selected_preset["center"]
    initial_zoom = selected_preset["zoom"]

    # Mapas base 100% libres y sin marcas de agua ni API keys requeridas
    tiles_configs = {
        "IGN España (Ortofoto Oficial)": {
            "url": "https://www.ign.es/wmts/pnoa-ma?request=GetTile&service=WMTS&version=1.0.0&layer=OI.OrthoimageCoverage&style=default&format=image/jpeg&tilematrixset=GoogleMapsCompatible&tilematrix={z}&tilerow={y}&tilecol={x}",
            "attr": "© Instituto Geográfico Nacional de España / PNOA",
        },
        "Relieve Topográfico": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, DeLorme, NAVTEQ, TomTom, Intermap, IPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, METI",
        },
        "Lienzo Claro": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, HERE, Garmin, © OpenStreetMap contributors",
        },
        "Lienzo Oscuro": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, HERE, Garmin, © OpenStreetMap contributors",
        },
        "Satélite": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, Maxar, Earthstar Geographics",
        },
        "Callejero": {
            "url": "OpenStreetMap",
            "attr": "© OpenStreetMap contributors",
        },
    }

    selected_config = tiles_configs.get(
        active_map_style,
        tiles_configs["IGN España (Ortofoto Oficial)"],
    )

    # Crear mapa base Folium centrado y acotado al territorio gallego
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=initial_zoom,
        min_zoom=7,
        max_zoom=16,
        tiles=selected_config["url"],
        attr=selected_config["attr"] if selected_config["url"] != "OpenStreetMap" else None,
        prefer_canvas=True,
    )

    # Cargar y aplicar máscara de atenuación territorial para oscurecer el exterior de Galicia (sin contorno azul)
    mask_geojson, _ = load_galicia_focus_layers()

    # Determinar opacidad exterior según si la capa base es fotográfica/oscura o clara
    is_dark_or_photo = any(k in active_map_style for k in ["Oscuro", "Satélite", "Ortofoto", "IGN"])

    if mask_geojson:
        # Atenuación exterior para que fuera de Galicia sea menos visible
        mask_opacity = 0.80 if is_dark_or_photo else 0.65
        folium.GeoJson(
            mask_geojson,
            name="Atenuación Exterior (Fuera de Galicia)",
            style_function=lambda _, op=mask_opacity: {
                "fillColor": "#060913",
                "fillOpacity": op,
                "color": "#060913",
                "weight": 0,
                "opacity": 0,
                "interactive": False,
            },
        ).add_to(m)

    # Demarcación de sector territorial si se ha seleccionado uno específico (overlay sutil sin bordes)
    if preset_name != "Galicia Completa":
        sector_bounds = selected_preset.get("bounds")
        sectors_data = load_galicia_sectors_geojson()
        sector_feature = None
        if sectors_data and "features" in sectors_data:
            for feat in sectors_data["features"]:
                if feat.get("properties", {}).get("sector") == preset_name:
                    sector_feature = feat
                    break

        sector_color = "#0284c7" if is_dark_or_photo else "#0369a1"

        if sector_feature:
            folium.GeoJson(
                sector_feature,
                name=f"Sector Operativo: {preset_name}",
                style_function=lambda _, c=sector_color: {
                    "fillColor": c,
                    "fillOpacity": 0.08,
                    "color": "transparent",
                    "weight": 0,
                    "opacity": 0,
                    "interactive": False,
                },
            ).add_to(m)
        elif sector_bounds:
            folium.Rectangle(
                bounds=sector_bounds,
                color="transparent",
                weight=0,
                opacity=0,
                fill=True,
                fill_color=sector_color,
                fill_opacity=0.06,
                interactive=False,
            ).add_to(m)

    # Plugins Folium para emergencias
    plugins.Fullscreen(position="topright", title="Pantalla Completa", title_cancel="Salir").add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # Filtrado y presupuesto de celdas para renderizar con máxima fluidez y relevancia táctica
    if "lat_centroid" in df_ready.columns and "lon_centroid" in df_ready.columns:
        df_map = df_ready.dropna(subset=["lat_centroid", "lon_centroid"]).copy()
    else:
        df_map = df_ready.copy()

    if preset_name != "Galicia Completa":
        # Filtrado estrictamente alineado con los distritos oficiales del sector (0 celdas fuera del límite)
        target_distritos = SECTOR_PLADIGA_DISTRITOS.get(preset_name, [])
        if "distrito_forestal" in df_map.columns and target_distritos:
            df_sector = df_map[df_map["distrito_forestal"].isin(target_distritos)].copy()
        else:
            b = selected_preset.get("bounds", [[41.8, -9.0], [43.8, -6.8]])
            df_sector = df_map[
                (df_map["lat_centroid"] >= b[0][0])
                & (df_map["lat_centroid"] <= b[1][0])
                & (df_map["lon_centroid"] >= b[0][1])
                & (df_map["lon_centroid"] <= b[1][1])
            ].copy()

        if not df_sector.empty:
            df_render = df_sector.nlargest(max(150, int(len(df_sector) * 0.15)), "prob_riesgo").copy()
        else:
            df_render = df_map.nlargest(200, "prob_riesgo").copy()
    else:
        # Presupuesto de celdas general optimizado para vista autonómica completa
        if "0.5%" in filter_risk:
            n_cutoff = max(80, int(len(df_map) * 0.005))
            df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
        elif "1.0%" in filter_risk or "1%" in filter_risk:
            n_cutoff = max(80, int(len(df_map) * 0.01))
            df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
        elif "2.0%" in filter_risk or "2%" in filter_risk:
            n_cutoff = max(180, int(len(df_map) * 0.02))
            df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
        elif "5.0%" in filter_risk or "5%" in filter_risk:
            n_cutoff = max(350, int(len(df_map) * 0.05))
            df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
        elif "10" in filter_risk:
            n_cutoff = max(600, int(len(df_map) * 0.10))
            df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
        elif "20" in filter_risk:
            n_cutoff = max(900, int(len(df_map) * 0.20))
            df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
        else:
            n_cutoff = max(350, int(len(df_map) * 0.05))
            df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()

    # Si hay celdas para renderizar
    if not df_render.empty and "lat_centroid" in df_render.columns and "lon_centroid" in df_render.columns:
        # Calcular color
        df_render["cell_color"] = [
            get_color_gradient(row.get("prob_riesgo", 0.01), row.get("percentil_riesgo", 0.5), color_mode)
            for _, row in df_render.iterrows()
        ]

        # 1. Capa Poligonal Disuelta (Formas complejas continuas)
        try:
            for color_hex, df_group in df_render.groupby("cell_color"):
                gdf_dissolved = build_complex_dissolved_shapes(df_group)
                folium.GeoJson(
                    gdf_dissolved,
                    name="Perímetros de Riesgo Disueltos",
                    style_function=lambda x, c=color_hex: {
                        "fillColor": c,
                        "color": c,
                        "weight": 1.2,
                        "fillOpacity": 0.55,
                        "interactive": False,
                    },
                ).add_to(m)
        except Exception:
            pass

        # 2. Capa de Inspección Interactiva (Popups detallados para las celdas más críticas)
        dlat = 0.00449
        dlon = 0.00615

        # Limitar celdas interactivas con popup a las más críticas para máxima fluidez
        df_interactive = df_render.head(250)

        for _, row in df_interactive.iterrows():
            lat = row["lat_centroid"]
            lon = row["lon_centroid"]
            if pd.isna(lat) or pd.isna(lon):
                continue

            pct = row.get("percentil_riesgo", 0.5)
            prob = row.get("prob_riesgo", 0.01)
            cid = int(row.get("cell_id", 0))
            c = row.get("cell_color", "#FEB24C")
            action = str(row.get("recommended_action", "vigilancia_reforzada")).replace("_", " ").title()

            raw_fuel = row.get("combustible_clase")
            fuel_type = (
                str(raw_fuel).capitalize()
                if pd.notna(raw_fuel) and str(raw_fuel).strip().lower() not in ["", "nan", "none"]
                else "Matorral / Monte Bajo"
            )
            forest_pct = row.get("combustible_pct_forestal", 0.0)
            if pd.isna(forest_pct):
                forest_pct = 0.0

            r30_flag = "Activa" if row.get("regla_30_30_activa", False) else "Inactiva"
            tmax_val = row.get("temperature_max_12_18h", row.get("tmax_vc", 25.0))
            rhmin_val = row.get("relative_humidity_min_12_18h", row.get("rhmin_vc", 35.0))
            vmax_val = row.get("wind_speed_max_12_18h", row.get("vmax_vc", 15.0))
            prec30d = row.get("precipitation_sum_30d", row.get("prec_acum_30d", 5.0))
            distrito = row.get("distrito_forestal", "Galicia")

            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; min-width: 240px; font-size: 12px; color: #1e293b;">
                <div style="background:#0f172a; color:#fff; padding:6px 10px; border-radius:4px 4px 0 0; font-weight:600; font-size:12px;">
                    CELDA #{cid} · {distrito}
                </div>
                <div style="padding:9px; background:#f8fafc; border:1px solid #cbd5e1; border-radius:0 0 4px 4px;">
                    <div style="margin-bottom:5px;">
                        <b>Probabilidad P(Y=1):</b> <span style="color:#b91c1c; font-weight:700;">{prob*100:.2f}%</span><br/>
                        <b>Percentil de Riesgo:</b> {pct*100:.1f}% (Top {(1-pct)*100:.1f}%)<br/>
                        <b>Acción:</b> <span style="color:#1d4ed8; font-weight:600;">{action}</span>
                    </div>
                    <hr style="margin:5px 0; border:0; border-top:1px solid #e2e8f0;"/>
                    <div style="font-size:11px; color:#475569; line-height:1.4;">
                        <b>Combustible:</b> {fuel_type} ({forest_pct:.0f}% arb.)<br/>
                        <b>T. Máx:</b> {tmax_val:.1f} ºC · <b>HR Mín:</b> {rhmin_val:.1f}%<br/>
                        <b>Viento:</b> {vmax_val:.1f} km/h · <b>Lluvia 30d:</b> {prec30d:.1f} mm<br/>
                        <b>Regla 30-30-30:</b> {r30_flag}
                    </div>
                </div>
            </div>
            """
            popup = folium.Popup(popup_html, max_width=280)
            tooltip = f"Celda #{cid} | P: {prob*100:.2f}% | {fuel_type}"

            geometry = row.get("geometry")
            if hasattr(geometry, "__geo_interface__"):
                folium.GeoJson(
                    geometry.__geo_interface__,
                    style_function=lambda _feature, c=c: {
                        "color": c,
                        "weight": 0.3,
                        "fillColor": c,
                        "fillOpacity": 0.08,
                    },
                    highlight_function=lambda _feature, c=c: {
                        "color": c,
                        "weight": 2.0,
                        "fillOpacity": 0.3,
                    },
                    popup=popup,
                    tooltip=tooltip,
                ).add_to(m)
            else:
                bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]
                folium.Rectangle(
                    bounds=bounds,
                    color=c,
                    weight=0.2,
                    fill=True,
                    fill_color=c,
                    fill_opacity=0.05,
                    popup=popup,
                    tooltip=tooltip,
                ).add_to(m)

    # Marcador de búsqueda
    if search_cell.strip() and "lat_centroid" in df_map.columns:
        try:
            cid = int(search_cell.strip())
            srow = df_map[df_map["cell_id"] == cid]
            if not srow.empty:
                r_info = srow.iloc[0]
                folium.CircleMarker(
                    location=[r_info["lat_centroid"], r_info["lon_centroid"]],
                    radius=10,
                    color="#ef4444",
                    fill=True,
                    fill_color="#ef4444",
                    fill_opacity=0.8,
                    popup=f"Celda #{cid} | P: {r_info['prob_riesgo']*100:.2f}%",
                ).add_to(m)
        except ValueError:
            pass

    # Leyenda flotante dinámica adaptada al modo de simbología seleccionado
    leyenda_html = build_map_legend_html(color_mode)
    m.get_root().html.add_child(folium.Element(leyenda_html))

    map_html = m.get_root().render()
    _MAP_CACHE[map_cache_key] = map_html
    _MAP_CACHE.move_to_end(map_cache_key)
    while len(_MAP_CACHE) > _MAP_CACHE_MAX_ENTRIES:
        _MAP_CACHE.popitem(last=False)

    _render_map_html(map_html)
