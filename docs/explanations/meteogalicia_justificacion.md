# Por qué usar MeteoGalicia como fuente meteorológica principal

## TFM: Sistema Predictivo de Anticipación de Incendios Forestales — Galicia MVP

---

## El problema con AEMET sola en Galicia rural

AEMET opera **~35 estaciones meteorológicas** en Galicia, concentradas principalmente en capitales de provincia (A Coruña, Vigo, Santiago, Lugo, Ourense) y cabeceras de comarca. Para una rejilla de ~30.000 celdas de 1km² que incluye las Sierras Orientales, el interior de Lugo y las zonas de eucalipto y pino del litoral... 35 puntos de interpolación son insuficientes.

El problema es especialmente grave para el propósito del proyecto: **los incendios forestales de Galicia no empiezan en A Coruña ciudad**. Empiezan en zonas rurales, en montes del interior, donde la cobertura de AEMET es la más débil.

---

## MeteoGalicia: la solución que ya existe

MeteoGalicia es el **Servicio de Meteorología de la Xunta de Galicia**, y dispone de una infraestructura propia diseñada específicamente para el territorio gallego.

### Red de Estaciones Meteorológicas Automáticas (EMA)

| Característica | AEMET (en Galicia) | MeteoGalicia |
|---|---|---|
| **Número de estaciones** | ~35 | **~170 EMA activas** |
| **Cobertura rural/forestal** | Muy limitada | Alta — incluye zonas forestales |
| **Resolución temporal** | Horaria | Horaria (algunas cada 10 min) |
| **Variables disponibles** | Estándar | Temperatura, humedad, viento, precipitación, presión, radiación solar, temperatura del suelo |
| **API abierta y gratuita** | ✅ `opendata.aemet.es` | ✅ `servizos.meteogalicia.gal` |
| **Datos históricos** | ✅ | ✅ (disponibles en `abertos.xunta.gal`) |
| **Librería Python** | No oficial | `meteoGalicia-api` (PyPI) |
| **Latencia de observaciones** | ~1-2 horas | ~1 hora |

### Densidad de cobertura

Con 170 estaciones sobre ~30.000 km², MeteoGalicia tiene una estación cada **~175 km²** de media. Para la interpolación IDW o Kriging a una rejilla de 1km×1km en zonas forestales, esto produce resultados significativamente más fiables que los ~857 km² por estación de AEMET.

---

## Acceso a la API

### Observaciones en tiempo real y histórico

```python
import requests

BASE_URL = "https://servizos.meteogalicia.gal/mf/observacion"

# Lista de todas las estaciones con coordenadas
def obtener_estaciones() -> list[dict]:
    r = requests.get(f"{BASE_URL}/listEstacions/json", params={"request": "getEstacionsInfo"})
    return r.json()["listEstacions"]

# Datos horarios históricos de una estación
def obtener_datos_historicos(id_estacion: str, fecha_ini: str, fecha_fin: str) -> dict:
    """
    Args:
        id_estacion: ID numérico de la estación (obtenido de listEstacions).
        fecha_ini: Fecha inicio formato 'YYYY-MM-DDTHH:MM:SSZ'.
        fecha_fin: Fecha fin formato 'YYYY-MM-DDTHH:MM:SSZ'.
    """
    r = requests.get(
        f"{BASE_URL}/observacionHoraria/json",
        params={
            "request": "getAWXData",
            "idEstacion": id_estacion,
            "dataIni": fecha_ini,
            "dataFin": fecha_fin,
            "variable": "temperatura,humidadeRelativa,velocidadeVento,precipitacion,radiacSolar",
        }
    )
    return r.json()

# Librería de terceros (más cómoda)
# pip install meteoGalicia-api
from meteogalicia_api.obtencion_datos_red_meteogalicia import Datos
datos = Datos(id_estacion=10045, fecha_inicio="2024-06-01", fecha_fin="2024-06-30")
df = datos.get_data()
```

### Previsión meteorológica de MeteoGalicia

```python
# Previsión por punto geográfico (más útil que por municipio para nuestra rejilla)
def obtener_prevision_punto(lat: float, lon: float, dias: int = 3) -> dict:
    """Descarga la previsión puntual para coordenadas concretas."""
    r = requests.get(
        "https://servizos.meteogalicia.gal/mf/predicion/puntual/diaria/json",
        params={
            "request": "getPredicionPuntual",
            "lat": lat,
            "lon": lon,
            "numDias": dias,
        }
    )
    return r.json()
```

### Portal de datos abiertos — Xunta de Galicia

Datos históricos completos también disponibles en:
- `https://abertos.xunta.gal/catalogo/medio-abiente/-/dataset/0689/datos-horarios-historicos-estacions-meteoroloxicas`

---

## Ventaja adicional: modelo de previsión propio

MeteoGalicia opera su propio **modelo de predicción numérica del tiempo** (WRF)
para el territorio gallego. La documentación MeteoSIX v5 expone una malla de
**1 km × 1 km** como opción preferente, además de mallas de 4, 12 y 36 km. La
malla de 4 km se mantiene como fallback cuando la ejecución de 1 km aún no
está disponible o no supera la validación de cobertura.

Las salidas tienen frecuencia horaria y horizonte aproximado de 96 horas para
WRF 1 km. La hora de disponibilidad de cada ejecución es variable, por lo que
el pipeline decide mediante `modelRun` y cobertura validada, no solo mediante
un horario fijo.

Para el pipeline de inferencia diaria, usar la previsión de MeteoGalicia en lugar de AEMET supone partir de datos **más ajustados a la orografía local de Galicia**, que es precisamente donde importa más la precisión (valles entre sierras, zonas de foehn, microclimas costeros).

### AEMET como alternativa de desarrollo

La posesión de una clave AEMET permite probar el circuito antes de recibir la
credencial de MeteoGalicia. El adaptador utiliza predicciones por municipio y
asigna cada celda al punto municipal más próximo. La predicción diaria permite
completar los tres horizontes y la horaria mejora las horas coincidentes, pero
esta combinación no crea una malla meteorológica de 1 km. Por ello el sistema
la etiqueta `fresh_aemet`, la muestra en el dashboard y la separa de los
resultados WRF en la evaluación.

La alternativa es útil para comprobar autenticación, descarga, normalización,
agregación 12:00–18:00, generación de modelos y publicación del dashboard. No
debe utilizarse para justificar precisión espacial ni para movilizar medios sin
confirmación independiente.

---

## Estrategia de fuentes recomendada para el MVP

```
ENTRENAMIENTO (histórico 2019-2024):
  Variables meteorológicas base → ERA5-Land (cobertura homogénea, consistente)
  Gap de últimos 5 días ERA5   → MeteoGalicia EMA (observaciones horarias)

PRODUCCIÓN (ejecución provisional y refresco posterior):
  Ventana crítica 12-18h T+1/T+2/T+3 → WRF 1km; fallback explícito WRF 04km
  Sin clave MeteoGalicia (pruebas)                  → AEMET municipal `fresh_aemet`
  Acumulados precipitación recientes  → Observaciones MeteoGalicia EMA
  Acumulados largo plazo (>7 días)    → ERA5 histórico + MeteoGalicia reciente
```

---

## Acción en el proyecto

Añadir al `.env`:

```bash
FORECAST_PROVIDER=auto  # meteogalicia, aemet o auto
# MeteoGalicia MeteoSIX v5 (requiere API_KEY privada)
METEOGALICIA_BASE_URL=https://servizos.meteogalicia.gal/apiv5
METEOGALICIA_GRIDS=1km,04km
METEOGALICIA_DATOS_ABIERTOS_URL=https://abertos.xunta.gal
# Alternativa temporal
AEMET_API_KEY=clave_aemet
AEMET_BASE_URL=https://opendata.aemet.es/opendata/api
```

Y actualizar el `data-ingestion` skill para incluir MeteoGalicia como **fuente primaria de meteorología en Galicia**, con ERA5 como fuente de calibración y relleno de gaps.
