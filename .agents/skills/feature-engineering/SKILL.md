---
name: feature-engineering
description: >
  Instrucciones para la construcción del Dataset Maestro: variables meteorológicas acumuladas 
  (ventana crítica 12h-18h), días secos, historial de incendios, proximidad humana, y auditoría 
  estricta contra el data leakage temporal. Activar en la Fase 3 del proyecto o cuando se 
  modifiquen las features del modelo.
---

# Skill: Ingeniería de Características — Fase 3

## Contexto

El Dataset Maestro es el corazón del sistema. Su calidad determina el 80% del rendimiento del modelo. Hay dos riesgos críticos que deben evitarse:

1. **Data leakage temporal**: usar datos del día T para predecir T (trampa imposible en producción).
2. **Negativos triviales**: si los negativos son demasiado fáciles, el modelo aprende patrones superficiales (verano = riesgo) en lugar de señales reales de peligro.

---

## Reglas Anti-Data-Leakage (NO NEGOCIABLES)

```
Para predecir el riesgo del día T:
  ✅ Usar datos hasta T-1 (inclusive)
  ❌ NUNCA usar datos de T o T+1, T+2...

Esto incluye:
  ✅ Variables meteorológicas: extraídas del día T-1 (ventana 12-18h de T-1)
  ✅ Acumulados: calculados hasta T-1
  ✅ Historial de incendios: solo incendios hasta T-1
  ❌ target: nunca usarlo como feature (obvio, pero verificar)
```

### Implementación de la barrera temporal

```python
def calcular_features_para_dia(df_meteo: pd.DataFrame, 
                                 cell_id: int, 
                                 fecha_prediccion: pd.Timestamp) -> dict:
    """Calcula todas las features para predecir el riesgo del día `fecha_prediccion`.
    
    CRÍTICO: Solo usa datos hasta fecha_prediccion - 1 día.
    """
    fecha_corte = fecha_prediccion - pd.Timedelta(days=1)
    
    # Filtrar datos hasta T-1
    df_celda = df_meteo[
        (df_meteo["cell_id"] == cell_id) & 
        (df_meteo["fecha"] <= fecha_corte)
    ].copy()
    
    return extraer_features(df_celda, fecha_corte)
```

---

## 1. Variables Meteorológicas — Ventana Crítica 12h-18h

En lugar de promediar las 24 horas, extraemos valores en la **franja de máxima vulnerabilidad** (12:00-18:00 UTC), cuando coinciden el pico de temperatura, mínima humedad y máximo viento.

```python
import xarray as xr
import numpy as np

def extraer_ventana_critica(ds: xr.Dataset, fecha: pd.Timestamp) -> dict:
    """Extrae variables meteorológicas de la ventana crítica 12h-18h del día T-1.
    
    Args:
        ds: Dataset xarray de ERA5-Land con datos horarios.
        fecha: Fecha de predicción T (se usarán datos de T-1).
    
    Returns:
        Diccionario con variables de la ventana crítica.
    """
    fecha_anterior = fecha - pd.Timedelta(days=1)
    
    # Seleccionar horas 12-18 del día T-1
    ds_ventana = ds.sel(
        time=ds.time.dt.date == fecha_anterior.date()
    ).isel(time=slice(12, 19))  # Horas 12, 13, 14, 15, 16, 17, 18
    
    # Temperatura máxima en la ventana (K → °C)
    t2m = ds_ventana["t2m"]
    temp_max = float((t2m.max("time") - 273.15).values)
    
    # Humedad relativa mínima (derivada del punto de rocío)
    td2m = ds_ventana["d2m"]
    t_celsius = t2m - 273.15
    td_celsius = td2m - 273.15
    # Aproximación de Magnus: HR = 100 * exp(17.625*td/(243.04+td)) / exp(17.625*t/(243.04+t))
    hr = 100 * np.exp(17.625 * td_celsius / (243.04 + td_celsius)) / \
              np.exp(17.625 * t_celsius / (243.04 + t_celsius))
    humedad_min = float(hr.min("time").values)
    
    # Velocidad máxima del viento (m/s → km/h)
    u10 = ds_ventana["u10"]
    v10 = ds_ventana["v10"]
    velocidad = np.sqrt(u10**2 + v10**2)
    viento_max_kmh = float((velocidad.max("time") * 3.6).values)
    
    return {
        "temp_max_12_18h": temp_max,
        "humedad_min_12_18h": humedad_min,
        "viento_max_kmh_12_18h": viento_max_kmh,
    }
```

---

## 2. Variables de Precipitación Acumulada

El estrés hídrico de la vegetación se mide a través de la precipitación acumulada en múltiples ventanas temporales.

```python
def calcular_acumulados_precipitacion(df_precip: pd.DataFrame, 
                                       cell_id: int, 
                                       fecha_corte: pd.Timestamp) -> dict:
    """Calcula la precipitación acumulada en múltiples ventanas temporales.
    
    Ventanas: 1, 3, 7, 14 y 30 días anteriores a fecha_corte.
    
    Args:
        df_precip: DataFrame con columnas ['cell_id', 'fecha', 'precip_mm'].
        cell_id: Identificador de la celda.
        fecha_corte: Último día disponible (T-1).
    
    Returns:
        Diccionario con precipitación acumulada por ventana.
    """
    df_celda = df_precip[df_precip["cell_id"] == cell_id].copy()
    df_celda = df_celda[df_celda["fecha"] <= fecha_corte]
    
    acumulados = {}
    for ventana_dias in [1, 3, 7, 14, 30]:
        fecha_inicio = fecha_corte - pd.Timedelta(days=ventana_dias - 1)
        precip_ventana = df_celda[df_celda["fecha"] >= fecha_inicio]["precip_mm"].sum()
        acumulados[f"precip_acum_{ventana_dias}d"] = float(precip_ventana)
    
    return acumulados

def calcular_dias_sin_lluvia(df_precip: pd.DataFrame, 
                              cell_id: int, 
                              fecha_corte: pd.Timestamp,
                              umbral_mm: float = 1.0) -> int:
    """Cuenta los días consecutivos sin lluvia significativa hasta fecha_corte.
    
    Args:
        umbral_mm: Precipitación mínima para considerar un día con lluvia (mm).
    """
    df_celda = df_precip[
        (df_precip["cell_id"] == cell_id) & 
        (df_precip["fecha"] <= fecha_corte)
    ].sort_values("fecha", ascending=False)
    
    dias_secos = 0
    for _, row in df_celda.iterrows():
        if row["precip_mm"] < umbral_mm:
            dias_secos += 1
        else:
            break
    
    return dias_secos
```

---

## 3. Historial de Incendios en el Vecindario

```python
def calcular_historial_incendios(df_incendios: pd.DataFrame,
                                  gdf_grid: gpd.GeoDataFrame,
                                  cell_id: int,
                                  fecha_corte: pd.Timestamp,
                                  radio_km: float = 10.0,
                                  ventana_años: int = 3) -> dict:
    """Calcula el número de incendios históricos en el vecindario de una celda.
    
    Args:
        radio_km: Radio de búsqueda de vecindario en km.
        ventana_años: Años de histórico a considerar.
    
    Returns:
        Diccionario con frecuencia de incendios en el vecindario.
    """
    fecha_inicio = fecha_corte - pd.DateOffset(years=ventana_años)
    
    # Incendios en la ventana temporal
    incendios_periodo = df_incendios[
        (df_incendios["fecha"] >= fecha_inicio) & 
        (df_incendios["fecha"] < fecha_corte)  # < para evitar leakage
    ]
    
    # Celdas vecinas en radio de búsqueda
    celda_geom = gdf_grid[gdf_grid["cell_id"] == cell_id].geometry.iloc[0]
    buffer = celda_geom.buffer(radio_km * 1000)  # metros (CRS en UTM)
    celdas_vecinas = gdf_grid[gdf_grid.geometry.intersects(buffer)]["cell_id"].tolist()
    
    n_incendios_vecindario = len(incendios_periodo[
        incendios_periodo["cell_id"].isin(celdas_vecinas)
    ])
    
    n_incendios_misma_celda = len(incendios_periodo[
        incendios_periodo["cell_id"] == cell_id
    ])
    
    return {
        f"n_incendios_vecindario_{radio_km}km_{ventana_años}años": n_incendios_vecindario,
        f"n_incendios_celda_{ventana_años}años": n_incendios_misma_celda,
    }
```

---

## 4. Variables Temporales y Estacionales

```python
def extraer_variables_temporales(fecha: pd.Timestamp) -> dict:
    """Extrae variables temporales e índices estacionales.
    
    Incluye codificación cíclica para mes y día del año (evitar discontinuidades).
    """
    dia_del_año = fecha.day_of_year
    mes = fecha.month
    
    return {
        "mes": mes,
        "dia_del_año": dia_del_año,
        "estacion": _estacion(mes),
        # Codificación cíclica (mantiene la continuidad dic→ene)
        "mes_sin": np.sin(2 * np.pi * mes / 12),
        "mes_cos": np.cos(2 * np.pi * mes / 12),
        "dia_sin": np.sin(2 * np.pi * dia_del_año / 365),
        "dia_cos": np.cos(2 * np.pi * dia_del_año / 365),
    }

def _estacion(mes: int) -> str:
    if mes in [12, 1, 2]: return "invierno"
    elif mes in [3, 4, 5]: return "primavera"
    elif mes in [6, 7, 8]: return "verano"
    else: return "otoño"
```

---

## 5. Variables de Proximidad Humana (OpenStreetMap)

```python
from shapely.ops import nearest_points

def calcular_distancia_carretera(gdf_carreteras: gpd.GeoDataFrame, 
                                  celda_centroide: Point) -> float:
    """Calcula la distancia en metros a la carretera más cercana.
    
    El factor humano explica >80% de los inicios de incendio.
    """
    distancias = gdf_carreteras.geometry.distance(celda_centroide)
    return float(distancias.min())
```

---

## 6. Esquema Final del Dataset Maestro

```python
COLUMNAS_DATASET_MAESTRO = {
    # Identificadores
    "cell_id": int,
    "fecha": "datetime64[ns]",
    "target": int,            # 0 o 1
    
    # Variables estáticas (Fase 1)
    "altitud_media": float,
    "pendiente_media": float,
    "orientacion_media": float,
    "combustible_clase": str,
    "lat_centroid": float,
    "lon_centroid": float,
    
    # Meteorología — ventana crítica 12-18h (T-1)
    "temp_max_12_18h": float,
    "humedad_min_12_18h": float,
    "viento_max_kmh_12_18h": float,
    
    # Acumulados de precipitación
    "precip_acum_1d": float,
    "precip_acum_3d": float,
    "precip_acum_7d": float,
    "precip_acum_14d": float,
    "precip_acum_30d": float,
    "dias_sin_lluvia": int,
    
    # Historial de incendios
    "n_incendios_celda_3años": int,
    "n_incendios_vecindario_10km_3años": int,
    
    # Proximidad humana
    "dist_carretera_m": float,
    "dist_nucleo_urbano_m": float,
    
    # Temporales (codificación cíclica)
    "mes_sin": float,
    "mes_cos": float,
    "dia_sin": float,
    "dia_cos": float,
    "estacion": str,
}
```

---

## Checklist de Calidad — Fase 3

- [ ] Ninguna feature usa datos del día T o posterior.
- [ ] Los acumulados de precipitación calculan correctamente ventanas de 1, 3, 7, 14 y 30 días.
- [ ] `dias_sin_lluvia` nunca es negativo.
- [ ] El dataset no tiene filas donde el negativo esté a ±5 días de un incendio real en la misma celda.
- [ ] El ratio positivos/negativos es verificado: típicamente 1:5 a 1:10.
- [ ] El dataset está guardado en `.parquet` con metadatos de fecha de generación.
- [ ] Ejecutar: `df.isnull().sum()` — documentar columnas con NaN y justificar o imputar.

## Guardado del Dataset Maestro

```python
import pandas as pd
from datetime import datetime

df_maestro.to_parquet(
    f"data/processed/features/dataset_maestro_{datetime.now().strftime('%Y%m%d')}.parquet",
    index=False,
    engine="pyarrow",
    compression="snappy"
)
print(f"✅ Dataset Maestro guardado: {len(df_maestro)} filas, {len(df_maestro.columns)} columnas")
print(f"   Positivos: {df_maestro['target'].sum()} ({df_maestro['target'].mean()*100:.2f}%)")
print(f"   Negativos: {(df_maestro['target'] == 0).sum()}")
```
