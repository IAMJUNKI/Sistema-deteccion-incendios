"""Componente del Mapa GIS Táctico Interactivo con Google Material Symbols y Alto Rendimiento."""

from __future__ import annotations

import textwrap

import pandas as pd
import streamlit as st

try:
    import folium
    from folium import plugins
    from streamlit_folium import st_folium
except ImportError:  # pragma: no cover - se prueba en instalaciones sin extras GIS
    folium = None
    plugins = None
    st_folium = None

from src.webapp.utils.geo_helpers import (
    CONCELLOS_GALICIA,
    ZOOM_PRESETS,
    build_complex_dissolved_shapes,
    load_galicia_focus_layers,
)


def get_color_gradient(prob: float, pct: float, mode: str) -> str:
    """Devuelve un color continuo en gradiente térmico adaptativo."""
    if "Probabilidad" in mode:
        val = prob * 100
        if val >= 15.0:
            return "#800026"  # Púrpura / Granate Extremo
        if val >= 10.0:
            return "#BD0026"  # Rojo Oscuro
        if val >= 5.0:
            return "#E31A1C"  # Rojo Vivo
        if val >= 2.5:
            return "#FC4E2A"  # Naranja Intenso
        if val >= 1.0:
            return "#FD8D3C"  # Naranja Claro
        if val >= 0.5:
            return "#FEB24C"  # Amarillo Dorado
        return "#FED976"      # Amarillo Suave
    elif "Percentil" in mode:
        val = pct * 100
        if val >= 99.5:
            return "#800026"  # Top 0.5% (Extremo)
        if val >= 98.0:
            return "#BD0026"  # Top 2.0%
        if val >= 95.0:
            return "#E31A1C"  # Top 5.0%
        if val >= 90.0:
            return "#FC4E2A"  # Top 10.0%
        if val >= 80.0:
            return "#FD8D3C"  # Top 20.0%
        return "#FEB24C"
    else:
        if pct >= 0.995:
            return "#800026"
        if pct >= 0.985:
            return "#E31A1C"
        if pct >= 0.950:
            return "#FD8D3C"
        return "#FED976"


def render_map_tab(
    df_data: pd.DataFrame,
    selected_horizon: int,
    map_style: str = "Esri Gris Claro (Lienzo Táctico)",
    color_mode: str = "Gradiente Continuo por Probabilidad P(Y=1)",
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
    if folium is None or st_folium is None:
        st.error(
            "El mapa requiere folium y streamlit-folium. Activa el entorno del proyecto y "
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
    col_preset, col_concello, col_search = st.columns([1.2, 1.2, 1.0])

    with col_preset:
        preset_name = st.selectbox(
            "Sector Territorial:",
            list(ZOOM_PRESETS.keys()),
            index=0,
        )
        selected_preset = ZOOM_PRESETS[preset_name]

    with col_concello:
        concello_options = ["-- Vista General --"] + sorted(CONCELLOS_GALICIA.keys())
        selected_concello = st.selectbox(
            "Centrar en Concello:",
            concello_options,
            index=0,
        )

    with col_search:
        search_cell = st.text_input("Localizar Celda ID:", value="", placeholder="Ej. 6818")

    # Determinar centro y zoom según preset o concello seleccionado
    center_lat, center_lon = selected_preset["center"]
    initial_zoom = selected_preset["zoom"]

    if selected_concello != "-- Vista General --":
        c_info = CONCELLOS_GALICIA[selected_concello]
        center_lat = c_info["lat"]
        center_lon = c_info["lon"]
        initial_zoom = 12

    # Mapas base 100% libres y sin marcas de agua ni API keys requeridas
    tiles_configs = {
        "Esri Gris Claro (Lienzo Táctico)": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, HERE, Garmin, © OpenStreetMap contributors",
        },
        "Esri Gris Oscuro (Lienzo Táctico)": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, HERE, Garmin, © OpenStreetMap contributors",
        },
        "Esri Satellite (Satelital)": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, Maxar, Earthstar Geographics",
        },
        "IGN España — PNOA Ortofoto (Oficial)": {
            "url": "https://www.ign.es/wmts/pnoa-ma?request=GetTile&service=WMTS&version=1.0.0&layer=OI.OrthoimageCoverage&style=default&format=image/jpeg&tilematrixset=GoogleMapsCompatible&tilematrix={z}&tilerow={y}&tilecol={x}",
            "attr": "© Instituto Geográfico Nacional de España / PNOA",
        },
        "OpenTopoMap (Topográfico)": {
            "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "attr": "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap",
        },
        "OpenStreetMap": {
            "url": "OpenStreetMap",
            "attr": "© OpenStreetMap contributors",
        },
    }

    selected_config = tiles_configs.get(
        map_style,
        tiles_configs["Esri Gris Claro (Lienzo Táctico)"],
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

    # Cargar y aplicar máscara de atenuación territorial para oscurecer el exterior de Galicia
    mask_geojson, _ = load_galicia_focus_layers()

    if mask_geojson:
        mask_opacity = 0.65 if "Oscuro" in map_style or "Satellite" in map_style else 0.48
        folium.GeoJson(
            mask_geojson,
            name="Atenuación Exterior",
            style_function=lambda _, op=mask_opacity: {
                "fillColor": "#0b0f19",
                "fillOpacity": op,
                "color": "#0b0f19",
                "weight": 0,
                "opacity": 0,
                "interactive": False,
            },
        ).add_to(m)

    # Plugins Folium para emergencias
    plugins.Fullscreen(position="topright", title="Pantalla Completa", title_cancel="Salir").add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # Filtrado de celdas para renderizar de forma segura
    if "lat_centroid" in df_ready.columns and "lon_centroid" in df_ready.columns:
        df_map = df_ready.dropna(subset=["lat_centroid", "lon_centroid"]).copy()
    else:
        df_map = df_ready.copy()

    # Presupuesto de celdas optimizado para carga instantánea
    if "0.3%" in filter_risk or "0.5%" in filter_risk:
        n_cutoff = max(80, int(len(df_map) * 0.005))
        df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
    elif "1.0%" in filter_risk or "1%" in filter_risk:
        n_cutoff = max(80, int(len(df_map) * 0.01))
        df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
    elif "1.5%" in filter_risk or "2.0%" in filter_risk:
        n_cutoff = max(180, int(len(df_map) * 0.02))
        df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
    elif "5.0%" in filter_risk or "5%" in filter_risk:
        n_cutoff = max(350, int(len(df_map) * 0.05))
        df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
    elif "10%" in filter_risk:
        n_cutoff = max(600, int(len(df_map) * 0.10))
        df_render = df_map.nlargest(n_cutoff, "prob_riesgo").copy()
    else:
        df_render = df_map.nlargest(400, "prob_riesgo").copy()

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

    # Leyenda flotante sobria
    leyenda_html = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background: rgba(15, 23, 42, 0.95); color: #f8fafc; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.4); font-family: 'Inter', sans-serif; font-size: 11px;
                border: 1px solid #334155;">
        <div style="font-weight:600; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:5px; color:#cbd5e1;">Escala de Riesgo</div>
        <div style="line-height: 1.6;">
            <span style="background:#800026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Extremo (&ge; 15% / Top 0.5%)<br/>
            <span style="background:#BD0026;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Muy Alto (&ge; 10% / Top 2%)<br/>
            <span style="background:#E31A1C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Alto (&ge; 5% / Top 5%)<br/>
            <span style="background:#FC4E2A;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Moderado-Alto (&ge; 2.5%)<br/>
            <span style="background:#FEB24C;width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Moderado / Bajo
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(leyenda_html))

    st_folium(m, use_container_width=True, height=620, returned_objects=[])
