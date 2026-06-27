---
name: webapp-dashboard
description: >
  Instrucciones para el desarrollo del dashboard interactivo Streamlit y el pipeline de 
  inferencia diaria del proyecto: mapas de riesgo con Folium/PyDeck, filtros interactivos, 
  panel de explicabilidad SHAP, y script de producción para generación automática diaria 
  de mapas de riesgo a partir de previsiones AEMET. Activar en la Fase 5 del proyecto.
---

# Skill: WebApp y Dashboard — Fase 5

## Contexto

La webapp transforma las predicciones del modelo en una herramienta operativa para gestores forestales y protección civil. El sistema tiene dos componentes:

1. **Dashboard Streamlit**: interfaz de consulta para el mapa de riesgo actualizado.
2. **Pipeline de inferencia diaria**: script automatizado que descarga previsiones AEMET, genera predicciones y actualiza la webapp.

---

## Stack Tecnológico

```python
import streamlit as st
import folium
from streamlit_folium import st_folium
import pydeck as pdk
import pandas as pd
import geopandas as gpd
import xgboost as xgb
import shap
import json
import os
from pathlib import Path
from dotenv import load_dotenv
```

---

## 1. Estructura del Dashboard Streamlit

### Organización de archivos

```
src/webapp/
├── __init__.py
├── app.py               # Punto de entrada de Streamlit
├── pages/
│   ├── 01_mapa_riesgo.py    # Mapa interactivo principal
│   ├── 02_analisis.py       # Análisis por zona y tendencias
│   └── 03_metodologia.py    # Explicación técnica del sistema
├── components/
│   ├── mapa.py              # Componentes de mapa Folium/PyDeck
│   ├── sidebar.py           # Filtros laterales
│   └── shap_panel.py        # Panel de explicabilidad
└── utils/
    ├── carga_modelo.py      # Carga de modelo y umbrales
    └── inferencia.py        # Función de predicción
```

### Punto de entrada (app.py)

```python
import streamlit as st

st.set_page_config(
    page_title="Sistema de Predicción de Incendios — Galicia",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔥 Sistema Predictivo de Incendios Forestales")
st.caption("Galicia · Actualizado diariamente a las 06:00 AM")
```

---

## 2. Mapa de Riesgo Interactivo (Folium)

```python
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import pandas as pd

# Paleta de colores por nivel de riesgo
COLORES_RIESGO = {
    "bajo":     "#2ecc71",   # Verde
    "moderado": "#f39c12",   # Naranja
    "alto":     "#e74c3c",   # Rojo
    "extremo":  "#8e44ad",   # Púrpura
}

def crear_mapa_riesgo(gdf_predicciones: gpd.GeoDataFrame, 
                       centro: list = [42.8, -8.0],
                       zoom: int = 8) -> folium.Map:
    """Crea un mapa Folium con las celdas coloreadas por nivel de riesgo.
    
    Args:
        gdf_predicciones: GeoDataFrame con columnas 'geometry', 'nivel_riesgo', 
                           'proba_incendio' y 'cell_id'.
        centro: [lat, lon] del centro del mapa.
        zoom: Nivel de zoom inicial.
    
    Returns:
        Objeto folium.Map listo para renderizar en Streamlit.
    """
    m = folium.Map(
        location=centro,
        zoom_start=zoom,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )
    
    # Añadir celdas coloreadas por nivel de riesgo
    for _, row in gdf_predicciones.iterrows():
        color = COLORES_RIESGO.get(row["nivel_riesgo"], "#95a5a6")
        
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda x, c=color: {
                "fillColor": c,
                "color": "none",
                "fillOpacity": 0.6,
                "weight": 0,
            },
            tooltip=folium.Tooltip(
                f"""
                <b>Nivel de riesgo:</b> {row['nivel_riesgo'].upper()}<br>
                <b>Probabilidad:</b> {row['proba_incendio']:.1%}<br>
                <b>Municipio:</b> {row.get('municipio', 'N/D')}<br>
                <b>Cell ID:</b> {row['cell_id']}
                """,
                sticky=False,
            ),
        ).add_to(m)
    
    # Leyenda
    _añadir_leyenda(m, COLORES_RIESGO)
    
    return m

def _añadir_leyenda(m: folium.Map, colores: dict) -> None:
    """Añade una leyenda al mapa."""
    leyenda_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: sans-serif;">
        <b>Nivel de Riesgo</b><br>
    """
    for nivel, color in colores.items():
        leyenda_html += f'<span style="background:{color};width:16px;height:16px;display:inline-block;border-radius:3px;margin-right:6px;"></span>{nivel.capitalize()}<br>'
    leyenda_html += "</div>"
    m.get_root().html.add_child(folium.Element(leyenda_html))
```

### Integrar el mapa en Streamlit

```python
from streamlit_folium import st_folium

st.subheader("🗺️ Mapa de Riesgo — Próximas 24h")
mapa = crear_mapa_riesgo(gdf_predicciones)
st_folium(mapa, width=None, height=600, returned_objects=[])
```

---

## 3. Sidebar con Filtros

```python
def render_sidebar() -> dict:
    """Renderiza el panel lateral con filtros interactivos.
    
    Returns:
        Diccionario con los valores seleccionados por el usuario.
    """
    with st.sidebar:
        st.header("⚙️ Filtros")
        
        # Horizonte temporal
        horizonte = st.selectbox(
            "Horizonte de predicción",
            options=["24 horas", "48 horas", "72 horas"],
            index=0,
        )
        
        # Filtro por provincia
        provincias = ["Todas", "A Coruña", "Lugo", "Ourense", "Pontevedra"]
        provincia = st.selectbox("Provincia", options=provincias)
        
        # Filtro por nivel de riesgo mínimo
        nivel_minimo = st.select_slider(
            "Mostrar celdas con riesgo ≥",
            options=["bajo", "moderado", "alto", "extremo"],
            value="moderado",
        )
        
        st.divider()
        st.caption(f"🕐 Datos actualizados: {ultima_actualizacion()}")
    
    return {
        "horizonte": horizonte,
        "provincia": provincia,
        "nivel_minimo": nivel_minimo,
    }
```

---

## 4. Panel SHAP — Explicabilidad

```python
import shap
import matplotlib.pyplot as plt

def panel_explicabilidad(modelo, X_celda: pd.DataFrame, 
                          feature_names: list[str]) -> None:
    """Muestra el panel de explicabilidad SHAP para una celda específica.
    
    Args:
        X_celda: DataFrame con las features de la celda seleccionada (1 fila).
    """
    st.subheader("🔍 ¿Por qué esta celda está en riesgo?")
    
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_celda)
    
    # Top 5 factores más influyentes
    importancias = pd.DataFrame({
        "variable": feature_names,
        "shap": shap_values[0],
        "valor": X_celda.iloc[0].values,
    }).sort_values("shap", key=abs, ascending=False).head(5)
    
    for _, row in importancias.iterrows():
        direccion = "⬆️ Aumenta" if row["shap"] > 0 else "⬇️ Reduce"
        st.metric(
            label=f"{row['variable']}",
            value=f"{row['valor']:.2f}",
            delta=f"{direccion} el riesgo ({row['shap']:+.3f})",
        )
```

---

## 5. Pipeline de Inferencia Diaria

Este script se ejecuta cada madrugada (cron job 05:00 AM) para generar las predicciones del día siguiente.

```python
#!/usr/bin/env python3
"""
pipeline_inferencia_diaria.py

Script de producción para generar el mapa de riesgo de incendios de mañana.
Ejecutar cada día a las 05:00 AM via cron job:
    0 5 * * * /path/to/conda/envs/incendios-forestales/bin/python /path/to/pipeline_inferencia_diaria.py
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import geopandas as gpd
import xgboost as xgb
import requests
import json
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

AEMET_API_KEY = os.getenv("AEMET_API_KEY")
MODELO_PATH = Path("data/models/modelo_xgboost_v1.json")
UMBRALES_PATH = Path("data/models/umbrales_calibrados_v1.json")
GRID_PATH = Path("data/processed/grid/galicia_grid_1km_v1.parquet")
OUTPUT_PATH = Path("data/processed/predicciones")

def main():
    fecha_prediccion = datetime.today() + timedelta(days=1)
    logger.info(f"🔥 Generando mapa de riesgo para: {fecha_prediccion.strftime('%Y-%m-%d')}")
    
    # 1. Cargar modelo y umbrales
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(MODELO_PATH))
    with open(UMBRALES_PATH) as f:
        umbrales = json.load(f)
    
    # 2. Cargar rejilla base
    gdf_grid = gpd.read_parquet(GRID_PATH)
    
    # 3. Descargar previsión AEMET para mañana
    df_prevision = descargar_prevision_aemet(fecha_prediccion, AEMET_API_KEY)
    
    # 4. Interpolar previsión a la rejilla de celdas
    df_features = interpolar_prevision_a_rejilla(df_prevision, gdf_grid)
    
    # 5. Inferencia
    X = df_features[FEATURES]
    probabilidades = modelo.predict_proba(X)[:, 1]
    
    # 6. Clasificar en niveles de riesgo
    df_features["proba_incendio"] = probabilidades
    df_features["nivel_riesgo"] = df_features["proba_incendio"].apply(
        lambda p: clasificar_riesgo(p, umbrales)
    )
    
    # 7. Guardar predicciones
    fecha_str = fecha_prediccion.strftime("%Y%m%d")
    output_file = OUTPUT_PATH / f"predicciones_{fecha_str}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_file, index=False)
    
    logger.info(f"✅ Predicciones guardadas: {output_file}")
    logger.info(f"   Celdas en riesgo ALTO+EXTREMO: {(df_features['nivel_riesgo'].isin(['alto', 'extremo'])).sum()}")

def clasificar_riesgo(probabilidad: float, umbrales: dict) -> str:
    """Convierte una probabilidad en nivel de riesgo usando umbrales calibrados."""
    if probabilidad >= umbrales["extremo"]["proba_min"]:
        return "extremo"
    elif probabilidad >= umbrales["alto"]["proba_min"]:
        return "alto"
    elif probabilidad >= umbrales["moderado"]["proba_min"]:
        return "moderado"
    else:
        return "bajo"

if __name__ == "__main__":
    main()
```

---

## 6. Análisis de Degradación ERA5 vs. AEMET

Un objetivo científico del TFM es cuantificar cuánto se degrada el rendimiento al usar previsiones AEMET (con error) frente a datos ERA5 reales (perfectos).

```python
def analisis_degradacion(modelo, df_test: pd.DataFrame, 
                          df_test_aemet: pd.DataFrame) -> pd.DataFrame:
    """Compara el rendimiento del modelo con ERA5 real vs. previsión AEMET.
    
    Args:
        df_test: Dataset de test con meteorología ERA5 (datos perfectos).
        df_test_aemet: Dataset de test con meteorología interpolada de AEMET.
    
    Returns:
        DataFrame comparativo de métricas.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    resultados = []
    for nombre, df in [("ERA5 (real)", df_test), ("AEMET (previsión)", df_test_aemet)]:
        X = df[FEATURES]
        y = df[TARGET]
        y_proba = modelo.predict_proba(X)[:, 1]
        
        resultados.append({
            "fuente_meteo": nombre,
            "auc_roc": roc_auc_score(y, y_proba),
            "pr_auc": average_precision_score(y, y_proba),
        })
    
    df_comparacion = pd.DataFrame(resultados)
    degradacion_auc = (df_comparacion.iloc[0]["auc_roc"] - df_comparacion.iloc[1]["auc_roc"])
    print(f"\n📉 Degradación AUC-ROC por uso de previsión AEMET: {degradacion_auc:.4f}")
    
    return df_comparacion
```

---

## Despliegue en Streamlit Cloud

```bash
# Lanzar en local
streamlit run src/webapp/app.py

# Para despliegue en Streamlit Cloud:
# 1. Subir el código al repositorio GitHub
# 2. Conectar en https://share.streamlit.io/
# 3. Configurar secrets en Settings > Secrets (equivalente al .env)
# 4. El modelo serializado debe ser accesible (Git LFS o descarga automática)
```

---

## Checklist de Calidad — Fase 5

- [ ] El mapa carga y muestra celdas coloreadas correctamente.
- [ ] Los filtros por provincia y nivel de riesgo funcionan.
- [ ] El panel SHAP muestra los 5 factores principales para la celda seleccionada.
- [ ] El script de inferencia diaria se ejecuta sin errores con las claves AEMET configuradas.
- [ ] El análisis de degradación ERA5 vs. AEMET está calculado y documentado.
- [ ] La webapp se despliega correctamente en local con `streamlit run src/webapp/app.py`.
