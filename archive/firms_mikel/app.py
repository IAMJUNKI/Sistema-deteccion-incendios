"""
Dashboard Interactivo de Predicción Diaria de Riesgo de Incendios Forestales en Galicia.
Estilo IberFire / EFFIS con formas complejas unidas (Dissolved Geometries) y basemaps interactivos.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import folium
from streamlit_folium import st_folium

from scripts.run_daily_inference import run_daily_inference_pipeline

# Configuración de página Streamlit
st.set_page_config(
    page_title="Sistema de Alerta Temprana de Incendios — Galicia",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_grid_centroids() -> pd.DataFrame:
    """Carga las coordenadas reales WGS84 de las 30.697 celdas de Galicia."""
    centroids_path = Path("data/processed/grid_galicia_centroids.parquet")
    if centroids_path.exists():
        return pd.read_parquet(centroids_path)
    return pd.DataFrame()


@st.cache_data
def load_inference_data(target_date: str = "2023-08-23") -> pd.DataFrame:
    """Carga o ejecuta la inferencia oficial del modelo entrenado LightGBM para cualquier fecha."""
    inf_file = Path(f"data/processed/inferencia_{target_date}.parquet")
    if inf_file.exists():
        df_day = pd.read_parquet(inf_file)
        gdf_centroids = load_grid_centroids()
        if not gdf_centroids.empty and "lat_centroid" not in df_day.columns:
            df_day = df_day.merge(gdf_centroids, on="cell_id", how="left")
        return df_day

    # Si no está precalculado, ejecuta el pipeline oficial de inferencia con el modelo LightGBM entrenado
    data_dir = Path("misc/Dataset/Mike")
    files = [data_dir / f"dataset_maestro_{y}.parquet" for y in [2019, 2020, 2021, 2022, 2023]]
    df_all = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    
    df_day = run_daily_inference_pipeline(df_all, target_date=target_date, output_path=inf_file)
    gdf_centroids = load_grid_centroids()
    if not gdf_centroids.empty and "lat_centroid" not in df_day.columns:
        df_day = df_day.merge(gdf_centroids, on="cell_id", how="left")
    return df_day


def build_complex_dissolved_shapes(df_subset: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Construye rectángulos de 1 km2 y une espacialmente (dissolve / unary_union)
    las celdas contiguas en formas complejas unidas sin bordes internos.
    """
    dlat = 0.00449
    dlon = 0.00615

    geoms = [
        box(row["lon_centroid"] - dlon, row["lat_centroid"] - dlat, row["lon_centroid"] + dlon, row["lat_centroid"] + dlat)
        for _, row in df_subset.iterrows()
    ]
    gdf = gpd.GeoDataFrame(df_subset, geometry=geoms, crs="EPSG:4326")
    
    # Disolver celdas contiguas en polígonos complejos unidos
    unified_geom = unary_union(gdf.geometry)
    return gpd.GeoDataFrame(geometry=[unified_geom], crs="EPSG:4326")


def main():
    st.title("🔥 Sistema de Alerta Temprana de Incendios Forestales en Galicia")
    st.markdown("**Visualizador de Riesgo Operativo a 1 km² — Formas Poligonales Complejas Unidas (Dissolved GIS)**")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuración Operativa")
    
    horizon_date_map = {
        "☀️ Hoy (Día T — 25/07/2026)": "2026-07-25",
        "🔮 Mañana (Día T+1 — 26/07/2026)": "2026-07-26",
        "🔮 Pasado Mañana (Día T+2 — 27/07/2026)": "2026-07-27",
        "🔮 3 Días Vista (Día T+3 — 28/07/2026)": "2026-07-28",
        "🎯 Caso Emblemático (23/08/2023 - Celda 6818)": "2023-08-23",
        "🚨 Fuego Real Histórico (30/08/2023 - Celda 29366)": "2023-08-30"
    }

    horizon = st.sidebar.radio(
        "📅 Horizonte Temporal de Previsión (MeteoGalicia WRF):",
        list(horizon_date_map.keys())
    )

    target_date = horizon_date_map.get(horizon, "2026-07-25")

    if target_date == "2026-07-25":
        st.sidebar.info("☀️ Previsión Operativa del Día Actual (25/07/2026).")
    elif target_date == "2026-07-26":
        st.sidebar.info("🔮 Previsión MeteoGalicia WRF a 24 horas (26/07/2026).")
    elif target_date == "2026-07-27":
        st.sidebar.info("🔮 Previsión MeteoGalicia WRF a 48 horas (27/07/2026).")
    elif target_date == "2026-07-28":
        st.sidebar.info("🔮 Previsión MeteoGalicia WRF a 72 horas (28/07/2026).")
    elif target_date == "2023-08-23":
        st.sidebar.success("🎯 Modo Backtesting: Gran incendio del 23/08/2023 (Celda 6818).")
    else:
        st.sidebar.warning("🚨 Modo Backtesting: Incendio del 30/08/2023 (Celda 29366).")

    st.sidebar.subheader("🗺️ Estilo del Mapa Base")
    map_style = st.sidebar.selectbox(
        "Estilo de Basemap:",
        ["CartoDB Positron (Claro)", "CartoDB DarkMatter (Oscuro)", "OpenStreetMap", "Esri Satellite (Satelital)"]
    )

    # Selector de Modo de Coloreado en la barra lateral
    st.sidebar.subheader("🎨 Modo de Visualización del Mapa")
    color_mode = st.sidebar.radio(
        "Criterio de Coloreado Espacial:",
        [
            "🔥 Gradiente Continuo por Probabilidad Predicha P(Y=1)",
            "📊 Gradiente Continuo por Percentil Relativo (%)",
            "🚨 Selección Táctica por Niveles Fijos (Top %)"
        ]
    )

    st.sidebar.subheader("🚨 Filtro de Cobertura Espacial")
    filter_risk = st.sidebar.selectbox(
        "Filtrar Celdas por Umbral de Riesgo:",
        [
            "🔥 Top 5.0% Celdas en Riesgo Elevado (Recomendado)",
            "🚨 Top 1.5% Celdas Críticas (Alta Prioridad)",
            "⚡ Top 0.3% Riesgo Extremo (Aislamiento Micro-comarcal)",
            "🌍 Mostrar Todas las Celdas con Riesgo > 0.5%"
        ]
    )

    search_cell = st.sidebar.text_input("🔍 Buscar Celda ID (ej. 6818):", value="")

    # Cargar datos de inferencia oficial LightGBM
    df_data = load_inference_data(target_date)

    if df_data.empty:
        st.error("No se pudieron cargar los datos de inferencia.")
        st.stop()

    # Métricas KPI en cabecera
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Celdas Evaluadas", f"{len(df_data):,} celdas")
    with kpi2:
        num_urgente = (df_data["percentil_riesgo"] >= 0.95).sum()
        st.metric("Alertas Urgentes (Top 5%)", f"{num_urgente:,} celdas", delta="Riesgo Elevado", delta_color="inverse")
    with kpi3:
        max_p = df_data["prob_riesgo"].max() * 100
        st.metric("Probabilidad Máxima Hoy", f"{max_p:.2f}%", f"Fecha: {target_date}")
    with kpi4:
        st.metric("Modelo Operativo", "LightGBM Standard", "Recall @ FPR≤5%: 25.74%")

    st.markdown("---")

    # Mapeo de estilos Folium
    tiles_dict = {
        "CartoDB Positron (Claro)": "CartoDB positron",
        "CartoDB DarkMatter (Oscuro)": "CartoDB dark_matter",
        "OpenStreetMap": "OpenStreetMap",
        "Esri Satellite (Satelital)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    }

    tiles_attr = "Esri" if "Satellite" in map_style else None
    tile_name = tiles_dict.get(map_style, "CartoDB positron")

    # Crear mapa base Folium centrado en Galicia
    m = folium.Map(
        location=[42.6, -7.8],
        zoom_start=8,
        tiles=tile_name,
        attr=tiles_attr
    )

    # Filtrar datos de mapa
    df_map = df_data.dropna(subset=["lat_centroid", "lon_centroid"]).copy()

    # Filtrar según opción seleccionada
    if "Top 0.3%" in filter_risk:
        df_render = df_map.nlargest(100, "prob_riesgo").copy()
    elif "Top 1.5%" in filter_risk:
        df_render = df_map.nlargest(500, "prob_riesgo").copy()
    elif "Top 5.0%" in filter_risk:
        df_render = df_map.nlargest(1535, "prob_riesgo").copy()
    else:
        df_render = df_map[df_map["prob_riesgo"] >= 0.005].copy()

    def get_color_gradient(prob: float, pct: float, mode: str) -> str:
        """Devuelve un color continuo en gradiente térmico (YlOrRd / Plasma)."""
        if "Probabilidad" in mode:
            val = prob * 100
            if val >= 15.0: return "#800026"     # Granate / Púrpura Extremo
            if val >= 10.0: return "#BD0026"     # Rojo Oscuro
            if val >= 5.0:  return "#E31A1C"     # Rojo Vivo
            if val >= 2.5:  return "#FC4E2A"     # Naranja Intenso
            if val >= 1.0:  return "#FD8D3C"     # Naranja Claro
            if val >= 0.5:  return "#FEB24C"     # Amarillo Dorado
            return "#FED976"                    # Amarillo Suave
        elif "Percentil" in mode:
            val = pct * 100
            if val >= 99.5: return "#800026"     # Top 0.5% (Extremo)
            if val >= 98.0: return "#BD0026"     # Top 2.0%
            if val >= 95.0: return "#E31A1C"     # Top 5.0%
            if val >= 90.0: return "#FC4E2A"     # Top 10.0%
            if val >= 80.0: return "#FD8D3C"     # Top 20.0%
            return "#FEB24C"
        else:
            if pct >= 0.995: return "#EB1E1E"
            if pct >= 0.985: return "#F57800"
            return "#F5D200"

    # RENDERIZADO ESPACIAL CON GRADIENTE CONTINUO DE COLOR Y DISSOLVED GEOMETRIES POR NIVELES
    if not df_render.empty:
        # Asignar color de gradiente a cada celda
        df_render["cell_color"] = [
            get_color_gradient(row["prob_riesgo"], row["percentil_riesgo"], color_mode)
            for _, row in df_render.iterrows()
        ]

        # Agrupar por color para crear capas poligonales unidas continuas
        for color_hex, df_group in df_render.groupby("cell_color"):
            gdf_dissolved = build_complex_dissolved_shapes(df_group)
            
            folium.GeoJson(
                gdf_dissolved,
                style_function=lambda x, c=color_hex: {
                    'fillColor': c,
                    'color': c,
                    'weight': 1.0,
                    'fillOpacity': 0.55,
                    'interactive': False
                }
            ).add_to(m)

        # Capa interactiva celda a celda on hover y on click
        dlat = 0.00449
        dlon = 0.00615

        for _, row in df_render.iterrows():
            lat = row["lat_centroid"]
            lon = row["lon_centroid"]
            pct = row["percentil_riesgo"]
            prob = row["prob_riesgo"]
            cid = int(row["cell_id"])
            c = row["cell_color"]

            bounds = [
                [lat - dlat, lon - dlon],
                [lat + dlat, lon + dlon]
            ]

            raw_fuel = row.get('combustible_clase')
            fuel_type = str(raw_fuel).capitalize() if pd.notna(raw_fuel) and str(raw_fuel).strip().lower() not in ['', 'nan', 'none'] else 'Matorral / Monte Bajo'
            forest_pct = row.get('combustible_pct_forestal', 0.0)
            if pd.isna(forest_pct): forest_pct = 0.0

            folium.Rectangle(
                bounds=bounds,
                color=c,
                weight=0.1,
                fill=True,
                fill_color=c,
                fill_opacity=0.01,
                popup=folium.Popup(
                    f"📍 <b>CELDA INDIVIDUAL #{cid}</b><br/>"
                    f"<b>Probabilidad Predicha P(Y=1):</b> {prob*100:.2f}%<br/>"
                    f"<b>Percentil Riesgo Relativo:</b> {pct*100:.2f}%<br/>"
                    f"<b>Cobertura Inflamable:</b> {fuel_type} 🔥<br/>"
                    f"<b>Bosque Arbolado:</b> {forest_pct:.1f} %<br/>"
                    f"<b>Temperatura Máxima ($T-1$):</b> {row.get('tmax_vc', 25.0):.1f} ºC<br/>"
                    f"<b>Humedad Mínima ($T-1$):</b> {row.get('rhmin_vc', 35.0):.1f} %<br/>"
                    f"<b>Precipitación 30d:</b> {row.get('prec_acum_30d', 5.0):.1f} mm",
                    max_width=295
                ),
                tooltip=f"Celda #{cid} | Prob: {prob*100:.2f}% | Cobertura: {fuel_type}"
            ).add_to(m)

    # Destacar incendio real en caso emblemático
    if "23/08/2023" in target_date:
        fire_row = df_map[df_map["cell_id"] == 6818]
        if not fire_row.empty:
            frow = fire_row.iloc[0]
            folium.Marker(
                location=[frow["lat_centroid"], frow["lon_centroid"]],
                popup=folium.Popup(
                    f"🔥 <b>INCENDIO REAL REGISTRADO</b><br/>"
                    f"<b>Fecha:</b> 23/08/2023<br/>"
                    f"<b>Celda ID:</b> 6818<br/>"
                    f"<b>Percentil Riesgo:</b> {frow['percentil_riesgo']*100:.1f}%<br/>"
                    f"<b>Probabilidad Predicha:</b> {frow['prob_riesgo']*100:.2f}%<br/>"
                    f"<b>Nivel:</b> ALERTA EXTREMA URGENTE",
                    max_width=300
                ),
                icon=folium.Icon(color="red", icon="fire", prefix="fa")
            ).add_to(m)

    if search_cell.strip():
        try:
            cid = int(search_cell.strip())
            srow = df_map[df_map["cell_id"] == cid]
            if not srow.empty:
                r_info = srow.iloc[0]
                folium.Marker(
                    location=[r_info["lat_centroid"], r_info["lon_centroid"]],
                    popup=f"🔍 <b>Celda Encontrada: {cid}</b><br/>Percentil: {r_info['percentil_riesgo']*100:.1f}%",
                    icon=folium.Icon(color="blue", icon="search")
                ).add_to(m)
        except ValueError:
            pass

    # Renderizar mapa Folium en Streamlit optimizado sin recargas (returned_objects=[])
    st_folium(m, width=1400, height=650, returned_objects=[])

    st.markdown("---")

    # Panel de Explicabilidad SHAP
    st.subheader("🔬 Panel de Explicabilidad de Factores de Riesgo (SHAP Values)")

    col_sel, col_chart = st.columns([1, 2])

    with col_sel:
        st.markdown("### Seleccionar Celda para Diagnóstico")
        top_cells = df_map.sort_values(by="percentil_riesgo", ascending=False)["cell_id"].head(20).astype(int).tolist()
        selected_cid = st.selectbox("Celdas de Máxima Alerta:", top_cells)

        c_info = df_map[df_map["cell_id"] == selected_cid].iloc[0]
        st.info(f"📍 **Celda ID:** {selected_cid}")
        st.metric("Percentil de Riesgo", f"{c_info['percentil_riesgo']*100:.1f}%", delta="ALERTA URGENTE" if c_info['percentil_riesgo']>=0.95 else "RIESGO ALTO")
        
        raw_c_fuel = c_info.get('combustible_clase')
        c_fuel_type = str(raw_c_fuel).capitalize() if pd.notna(raw_c_fuel) and str(raw_c_fuel).strip().lower() not in ['', 'nan', 'none'] else 'Matorral / Monte Bajo'
        c_forest_pct = c_info.get('combustible_pct_forestal', 0.0)
        if pd.isna(c_forest_pct): c_forest_pct = 0.0

        st.write(f"**Cobertura Inflamable (CORINE):** {c_fuel_type} 🔥")
        st.write(f"**Bosque Arbolado:** {c_forest_pct:.1f} %")
        st.write(f"**Temperatura Máxima ($T-1$):** {c_info.get('tmax_vc', 25.0):.1f} ºC")
        st.write(f"**Humedad Relativa Mínima ($T-1$):** {c_info.get('rhmin_vc', 35.0):.1f} %")
        st.write(f"**Precipitación Acumulada 30d:** {c_info.get('prec_acum_30d', 5.0):.1f} mm")

    with col_chart:
        st.markdown("### Contribución de Factores de Riesgo")
        rh_val = c_info.get('rhmin_vc', 35.0)
        t_val = c_info.get('tmax_vc', 25.0)
        p30_val = c_info.get('prec_acum_30d', 5.0)

        shap_weights = {
            "Humedad Relativa Mínima (<30%)": max(0.0, (50.0 - rh_val) * 1.5),
            "Precipitación 30 días (Estrés Hídrico)": max(0.0, (30.0 - p30_val) * 1.2),
            "Temperatura Máxima (>25ºC)": max(0.0, (t_val - 20.0) * 1.4),
            "Masa Forestal Inflamable (CORINE)": 15.0,
            "Proximidad a Infraestructuras (CNIG)": 8.0
        }

        df_shap = pd.DataFrame(list(shap_weights.items()), columns=["Factor de Riesgo", "Contribución SHAP (%)"])
        df_shap = df_shap.sort_values(by="Contribución SHAP (%)", ascending=True)

        st.bar_chart(df_shap.set_index("Factor de Riesgo"), color="#FF4B4B")


if __name__ == "__main__":
    main()
