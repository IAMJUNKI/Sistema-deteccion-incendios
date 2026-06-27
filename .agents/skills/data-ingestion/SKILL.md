---
name: data-ingestion
description: >
  Instrucciones para la descarga e ingesta de datos del proyecto: NASA FIRMS (focos de calor históricos), 
  Copernicus ERA5-Land (meteorología horaria), AEMET OpenData (previsión en tiempo real) y límites 
  administrativos CNIG/IGN. Incluye autenticación con APIs, formatos de archivo y construcción del 
  target (variable objetivo). Activar en la Fase 2 del proyecto.
---

# Skill: Ingesta de Datos — Fase 2

## Contexto

La Fase 2 descarga y procesa el histórico de incendios (NASA FIRMS) y meteorología (ERA5-Land) para construir la variable objetivo `target` (1 = inicio de incendio, 0 = no incendio). El rango temporal del MVP es **2019-2024**, región **Galicia**.

---

## 1. NASA FIRMS — Focos de Calor Históricos

### Obtener API Key

1. Registrarse en: https://firms.modaps.eosdis.nasa.gov/api/area/
2. Guardar la clave como `NASA_FIRMS_MAP_KEY` en `.env`.

### Descarga vía API (recomendado para automatización)

```python
import requests
import pandas as pd
from pathlib import Path

MAP_KEY = os.getenv("NASA_FIRMS_MAP_KEY")
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Bounding box de Galicia (WGS84): lon_min, lat_min, lon_max, lat_max
GALICIA_BBOX = "-9.3,41.8,-6.7,43.8"

def descargar_firms(sensor: str, año: int, bbox: str, api_key: str) -> pd.DataFrame:
    """Descarga focos de calor de NASA FIRMS para un año y sensor dados.
    
    Args:
        sensor: 'MODIS_NRT' o 'VIIRS_SNPP_NRT' o 'VIIRS_NOAA20_NRT'
        año: Año a descargar (2019-2024)
        bbox: Bounding box en formato 'lon_min,lat_min,lon_max,lat_max'
        api_key: Clave de la API NASA FIRMS
    
    Returns:
        DataFrame con focos de calor del año especificado.
    """
    url = f"{BASE_URL}/{sensor}/{api_key}/{bbox}/{año}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    return df

# Descarga de todos los años
for año in range(2019, 2025):
    df = descargar_firms("VIIRS_SNPP_NRT", año, GALICIA_BBOX, MAP_KEY)
    df.to_parquet(f"data/raw/firms/firms_galicia_viirs_{año}.parquet", index=False)
    print(f"✅ {año}: {len(df)} focos descargados")
```

### Descarga masiva (alternativa)

Para históricos largos, usar la [descarga directa](https://firms.modaps.eosdis.nasa.gov/download/) con archivos anuales en formato CSV o SHP.

### Columnas relevantes de FIRMS

| Columna | Descripción |
|---|---|
| `latitude`, `longitude` | Coordenadas del foco |
| `acq_date` | Fecha de adquisición (`YYYY-MM-DD`) |
| `acq_time` | Hora UTC |
| `frp` | Fire Radiative Power (MW) — intensidad del foco |
| `confidence` | Confianza de la detección (`l`=low, `n`=nominal, `h`=high) |

**Filtrar solo focos con confianza `n` o `h`.**

---

## 2. Construcción del Target — Clustering Espacio-Temporal

El objetivo es identificar **eventos únicos de incendio** a partir de los focos FIRMS, evitando contar el mismo incendio múltiples veces.

### Algoritmo de clustering (DBSCAN)

```python
from sklearn.cluster import DBSCAN
import numpy as np

def clustear_incendios(df_firms: pd.DataFrame, 
                        dist_km: float = 5.0, 
                        ventana_dias: int = 4) -> pd.DataFrame:
    """Agrupa focos FIRMS en eventos únicos de incendio usando DBSCAN.
    
    Parámetros configurables:
        dist_km: Radio máximo de agrupación espacial (km). Default: 5km.
        ventana_dias: Ventana temporal máxima de agrupación (días). Default: 4 días.
    
    Returns:
        DataFrame con columna 'evento_id' y columna 'es_inicio' (True solo en T0).
    """
    # Normalizar espacio y tiempo en la misma métrica
    # 1 grado lat ≈ 111 km → convertir días a km equivalentes
    ESCALA_TIEMPO = 111.0 / 1  # 1 día = 111 km (equivalencia para DBSCAN)
    
    coords = np.column_stack([
        df_firms["latitude"].values,
        df_firms["longitude"].values,
        df_firms["dia_del_año"].values * (dist_km / ventana_dias) / 111.0
    ])
    
    epsilon = dist_km / 111.0  # Convertir km a grados
    
    db = DBSCAN(eps=epsilon, min_samples=1, metric="euclidean")
    df_firms["evento_id"] = db.fit_predict(coords)
    
    # Identificar fecha de inicio (T0) de cada evento
    idx_inicio = df_firms.groupby("evento_id")["acq_date"].idxmin()
    df_firms["es_inicio"] = False
    df_firms.loc[idx_inicio, "es_inicio"] = True
    
    return df_firms

# Solo los inicios son positivos (Y=1)
df_positivos = df_firms[df_firms["es_inicio"]].copy()
df_positivos["target"] = 1
```

---

## 3. Muestreo de Negativos Difíciles (Y=0)

Los negativos no se generan aleatoriamente — deben ser "difíciles" para que el modelo aprenda patrones reales.

```python
def generar_negativos(df_positivos: pd.DataFrame, 
                       gdf_grid: gpd.GeoDataFrame,
                       ratio_negativos: int = 5) -> pd.DataFrame:
    """Genera negativos difíciles estratificados.
    
    Tipos de negativos:
        1. Vecinos espaciales: celdas contiguas al incendio, mismo día.
        2. Vecinos temporales: misma celda, ±10-14 días del incendio.
        3. Aleatorios estratificados: celdas aleatorias, distribuidas por estaciones.
    
    Regla de exclusión:
        Ningún negativo puede estar a ±5 días de un incendio real en la misma celda.
    """
    negativos = []
    
    for _, incendio in df_positivos.iterrows():
        fecha_t0 = incendio["acq_date"]
        celda_t0 = incendio["cell_id"]
        
        # 1. Vecinos espaciales (celdas contiguas, mismo día)
        vecinas = obtener_celdas_vecinas(gdf_grid, celda_t0, radio_km=5)
        for celda_v in vecinas[:3]:  # Máximo 3 vecinos espaciales por incendio
            negativos.append({"cell_id": celda_v, "fecha": fecha_t0, "target": 0})
        
        # 2. Vecinos temporales (misma celda, días cercanos)
        for offset in [-14, -10, 10, 14]:
            fecha_neg = fecha_t0 + pd.Timedelta(days=offset)
            negativos.append({"cell_id": celda_t0, "fecha": fecha_neg, "target": 0})
    
    df_negativos = pd.DataFrame(negativos)
    
    # Aplicar regla de exclusión: eliminar negativos a ±5 días de un incendio real
    df_negativos = aplicar_exclusion_temporal(df_negativos, df_positivos, ventana_dias=5)
    
    return df_negativos
```

---

## 4. ERA5-Land — Meteorología Histórica (Copernicus CDS)

### Configuración de la API

```bash
# Instalar cliente CDS
pip install cdsapi
```

Crear `~/.cdsapirc`:

```ini
url: https://cds.climate.copernicus.eu/api
key: {UID}:{API_KEY}
```

O usar variables de entorno:

```python
import cdsapi
import os

c = cdsapi.Client(
    url=os.getenv("COPERNICUS_CDS_API_URL"),
    key=f"{os.getenv('COPERNICUS_CDS_UID')}:{os.getenv('COPERNICUS_CDS_API_KEY')}"
)
```

### Descarga de ERA5-Land

```python
def descargar_era5(año: int, mes: int, variables: list[str], 
                    area: list[float], ruta_salida: Path) -> None:
    """Descarga datos ERA5-Land para un mes y área dados.
    
    Args:
        año: Año a descargar.
        mes: Mes (1-12).
        variables: Lista de variables CDS. Ej: ['2m_temperature', '10m_u_component_of_wind'].
        area: [lat_max, lon_min, lat_min, lon_max] en WGS84. Ej para Galicia: [43.8, -9.3, 41.8, -6.7].
        ruta_salida: Ruta del archivo NetCDF de salida.
    """
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": variables,
            "year": str(año),
            "month": f"{mes:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],  # Datos horarios
            "area": area,
            "format": "netcdf",
        },
        str(ruta_salida),
    )

VARIABLES_ERA5 = [
    "2m_temperature",              # Temperatura a 2m
    "2m_dewpoint_temperature",     # Punto de rocío → humedad relativa
    "10m_u_component_of_wind",     # Viento U
    "10m_v_component_of_wind",     # Viento V
    "total_precipitation",         # Precipitación total
    "surface_solar_radiation_downwards",  # Radiación solar
]

GALICIA_AREA = [43.8, -9.3, 41.8, -6.7]  # [lat_max, lon_min, lat_min, lon_max]
```

### Variables derivadas clave

A partir de los datos brutos ERA5, calcular durante la **ventana crítica 12h-18h UTC**:

| Variable derivada | Cálculo |
|---|---|
| `temp_max_12_18h` | `max(t2m[12:18])` convertido de K a °C |
| `humedad_min_12_18h` | Humedad relativa mínima derivada del punto de rocío |
| `viento_max_12_18h` | `max(sqrt(u10² + v10²))[12:18]` en km/h |
| `dir_viento_dominante` | Dirección predominante en la ventana |
| `precipitacion_24h_ant` | Suma de precipitación del día anterior completo |

---

## 5. AEMET OpenData — Previsión en Tiempo Real (Fase 5)

```python
def descargar_prevision_aemet(api_key: str, 
                               codigo_municipio: str = "15030") -> dict:
    """Descarga la previsión horaria de AEMET para un municipio.
    
    Args:
        api_key: Clave de AEMET OpenData.
        codigo_municipio: Código INE del municipio (ej: '15030' = A Coruña).
    
    Returns:
        Diccionario con la previsión meteorológica para las próximas 48/72h.
    """
    BASE_URL = "https://opendata.aemet.es/opendata/api"
    headers = {"api_key": api_key}
    
    # Paso 1: obtener URL de datos
    r = requests.get(
        f"{BASE_URL}/prediccion/especifica/municipio/horaria/{codigo_municipio}",
        headers=headers, verify=True
    )
    datos_url = r.json()["datos"]
    
    # Paso 2: descargar datos reales
    r_datos = requests.get(datos_url, headers=headers)
    return r_datos.json()
```

---

## Checklist de Calidad — Fase 2

- [ ] FIRMS descargado para Galicia, años 2019-2024, sensor VIIRS (confianza n/h).
- [ ] Clustering DBSCAN genera eventos únicos: verificar que el número de eventos es coherente con estadísticas oficiales de Galicia (~1.000-3.000 incendios/año).
- [ ] Dataset de positivos: `df_positivos["target"].sum()` > 0.
- [ ] Dataset de negativos: sin solapamiento temporal ±5 días con positivos.
- [ ] ERA5-Land descargado en formato NetCDF para todos los meses de 2019-2024.
- [ ] Dataset integrado final tiene columnas: `cell_id`, `fecha`, `target`, variables meteorológicas.
