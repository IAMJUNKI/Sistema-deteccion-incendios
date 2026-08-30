# Pipeline operativo detallado de predicción de peligro de incendio

## 1. Propósito y decisión que debe soportar

Este documento describe, de extremo a extremo, cómo el sistema transforma una
previsión meteorológica y el estado del territorio en tres mapas diarios de
riesgo de inicio de incendio forestal para Galicia.

La salida no pretende responder a “¿dónde habrá humo con certeza?” ni a
“¿cómo se propagará un incendio que ya ha comenzado?”. La decisión operativa es
más concreta:

> Para cada celda de 1 km × 1 km, ¿cuál es la probabilidad calibrada de que se
> produzca una nueva ignición durante el día objetivo, dadas la meteorología
> prevista, la topografía, el combustible y la memoria ambiental disponible en
> el momento de la ejecución?

El producto inicial publica tres horizontes independientes:

| Horizonte | Significado | Ejemplo si la ejecución es el 16 de agosto a las 05:00 CEST |
|---|---|---|
| T+1 | Riesgo durante el día siguiente | 17 de agosto |
| T+2 | Riesgo durante el segundo día | 18 de agosto |
| T+3 | Riesgo durante el tercer día | 19 de agosto |

Separar los horizontes es importante: el error de una previsión a 24 horas no
es el mismo que el de una previsión a 72 horas y las probabilidades se calibran
por separado.

## 2. Qué problema resuelve y qué problema no resuelve

El sistema resuelve una clasificación espacial y temporal de riesgo de
ignición. Cada observación lógica es la pareja cell_id, fecha, y el target
histórico indica si se produjo una ignición en esa celda durante el día.

El sistema no construye una predicción meteorológica propia a partir de la
temperatura actual. La previsión meteorológica ya es el resultado de modelos
numéricos que resuelven ecuaciones físicas de la atmósfera, asimilan
observaciones y producen valores futuros por hora y por punto de la malla. El
modelo de este proyecto recibe esos valores como variables explicativas y
aprende el riesgo de incendio condicionado a ellos:

~~~text
modelo numérico meteorológico
        ↓
forecast horario de temperatura, humedad, lluvia y viento
        ↓
features meteorológicas, de terreno, combustible y memoria
        ↓
P(ignición | información disponible en la emisión)
~~~

La separación evita mezclar dos tareas con escalas y errores distintos:

1. Predicción del estado físico de la atmósfera.
2. Predicción del riesgo de ignición condicionado a ese estado.

### 2.1 Objetivo preventivo y utilidad para la movilización de medios

El target científico se mantiene como una variable binaria de nueva ignición o
incendio detectado en una celda y día. No se intenta predecir la intención de
una persona ni simular la propagación de un incendio ya activo. Esta elección
evita que el modelo aprenda de forma trivial que una celda que ya arde seguirá
ardiendo.

Para la operación, la probabilidad calibrada se transforma además en un
ranking de prioridad preventiva. El ranking permite decidir dónde reforzar la
vigilancia o preposicionar medios cuando los recursos son limitados. La acción
no se automatiza: debe combinar el riesgo meteorológico con accesibilidad,
exposición, medios disponibles y validación del centro de mando.

| Nivel relativo | Acción preventiva orientativa |
|---|---|
| Bajo | Vigilancia rutinaria |
| Moderado | Vigilancia reforzada |
| Alto | Preposición de medios |
| Extremo | Preposición prioritaria y confirmación operativa |

La métrica de éxito más útil para este objetivo no es únicamente accuracy. Se
evaluará cuántos incendios quedan cubiertos por el 1 %, 5 % y 10 % de celdas
priorizadas, el recall a una tasa de falsas alarmas fijada y la estabilidad del
ranking entre horizontes.

## 3. Arquitectura completa

~~~mermaid
flowchart TD
    A[Scheduler y disponibilidad de modelRun] --> B[Actualizar estado meteorológico reciente]
    B --> C{Proveedor configurado}
    C -- MeteoGalicia --> D[Descargar MeteoSIX WRF]
    D --> E{WRF 1 km válido y completo}
    E -- Sí --> F[Archivar WRF 1 km]
    E -- No --> G[Intentar WRF 04 km]
    G --> H{WRF 04 km válido y completo}
    H -- Sí --> I[Marcar fresh_fallback]
    H -- No --> J[Cargar último forecast válido]
    J --> K[Marcar calidad stale]
    C -- AEMET --> L[Descargar diaria y horaria mediante datos]
    L --> M[Expandir 72 h y marcar fresh_aemet]
    F --> X[Asignar puntos del proveedor a rejilla 1 km]
    I --> X
    K --> X
    M --> X
    X --> K[Convertir UTC a Europe/Madrid]
    K --> L[Agregación 12:00–18:00 y acumulados]
    B --> L
    L --> M[Features T+1]
    L --> N[Features T+2]
    L --> O[Features T+3]
    M --> P[Modelo serializado T+1 + calibrador]
    N --> Q[Modelo serializado T+2 + calibrador]
    O --> R[Modelo serializado T+3 + calibrador]
    P --> S[prob_risk, percentil, acción preventiva, metadatos]
    Q --> S
    R --> S
    S --> T[Parquet de resultados]
    S --> U[JSON de metadatos]
    S --> V[Manifest con checksums]
    T --> W[Dashboard Streamlit]
    V --> W
~~~

| Responsabilidad | Módulo |
|---|---|
| Cliente y parser MeteoSIX | src/ingestion/meteogalicia_forecast.py |
| Ingesta histórica AEMET | src/ingestion/aemet_observations.py |
| Estado meteorológico reciente | src/ingestion/weather_state.py |
| Publicación de observaciones y estado | scripts/ingest_aemet_weather_state.py |
| Actualización genérica del estado | scripts/update_weather_state.py |
| Agregación y features | src/features/operational_features.py |
| Artefactos atómicos y lock | src/operational/artifacts.py |
| Modelos T+1/T+2/T+3 | src/models/forecast_risk_model.py |
| Explicabilidad | src/models/explainability.py |
| Entrenamiento offline | scripts/train_forecast_models.py |
| Inferencia diaria | scripts/run_daily_inference.py |
| Health check | scripts/check_operational_run.py |
| Visualización | app.py |

## 4. Fuentes de datos y responsabilidades

### 4.1 Rejilla espacial

La rejilla de producción contiene una fila por celda de aproximadamente 1 km ×
1 km. Su identificador estable es cell_id. Como mínimo, la inferencia exige:

| Campo | Descripción |
|---|---|
| cell_id | Identificador persistente de celda |
| lat_centroid | Latitud WGS84 del centroide |
| lon_centroid | Longitud WGS84 del centroide |
| geometry | Geometría opcional para el dashboard |
| altitud_media | Elevación media |
| pendiente_media | Pendiente media |
| orientacion_media | Orientación media |
| combustible_pct_forestal | Porcentaje de superficie forestal o combustible |

La rejilla no se consulta a MeteoGalicia celda a celda. Se reduce primero a
puntos representativos de la malla meteorológica, en bloques de hasta 20
localizaciones por petición, y posteriormente se asigna cada celda al punto
meteorológico más cercano.

### 4.2 ERA5-Land

ERA5-Land es un reanálisis histórico horario. Sirve para reconstruir el pasado
de forma homogénea y entrenar o evaluar el benchmark histórico. No es una
previsión de lo que ocurrirá mañana y no debe sustituir silenciosamente a un
forecast fallido en la ejecución de producción.

En el benchmark, la meteorología del día objetivo es conocida porque proviene
del reanálisis. Por eso se etiqueta como era5_perfect: permite comprobar la
alineación de targets, features y años, pero no mide todavía el error real de
la previsión meteorológica.

### 4.3 MeteoGalicia MeteoSIX v5

MeteoGalicia es la fuente operativa principal. El cliente solicita variables
horarias de la previsión numérica WRF y conserva tanto la respuesta original
como el formato normalizado. La selección de malla es explícita: WRF `1km` es
la fuente preferida y WRF `04km` es un fallback controlado.

| Variable normalizada | Unidad esperada | Uso |
|---|---:|---|
| temperature_c | °C | Máxima en ventana crítica y VPD |
| relative_humidity_pct | % | Mínima en ventana crítica y VPD |
| precipitation_mm | mm | Lluvia diaria y memoria de sequedad |
| wind_speed_kmh | km/h | Máxima en ventana crítica |
| wind_dir_deg | grados | Trazabilidad; aún no entra en el esquema base |

La API v5 limita cada petición a 20 localizaciones y permite solicitar hasta
7 días de información numérica. Cada valor puede declarar `modelRun`,
`timeInstant`, `model`, `grid`, unidades y la geometría real utilizada. La
geometría real se conserva porque `autoAdjustPosition` puede desplazar el punto
solicitado en determinadas zonas costeras.

La ejecución WRF de 1 km de las 00:00 UTC termina aproximadamente a las 07:30
UTC. Por tanto, las 05:00 hora local no garantizan que esté publicada la nueva
ejecución de 1 km. El pipeline prueba la cobertura real y no asume que el
forecast más nuevo está disponible solo por la hora del scheduler.

La solicitud registra proveedor, versión API, modelo, versión del modelo,
rejilla, unidades, puntos consultados, intervalo válido y timestamps del
sistema.

### 4.3.1 AEMET OpenData como proveedor alternativo

Mientras no esté disponible la clave de MeteoGalicia, el pipeline puede
seleccionar AEMET mediante `FORECAST_PROVIDER=aemet` o automáticamente mediante
`FORECAST_PROVIDER=auto`. AEMET OpenData requiere una doble petición: primero se
obtiene la URL temporal de `datos` y después se descarga el JSON real. El
adaptador conserva los JSON diarios y horarios en `data/raw/meteogalicia/raw/aemet/`.

AEMET no devuelve una malla regular WRF. La configuración recibe una lista
auditable de municipios con el formato:

~~~text
AEMET_MUNICIPALITIES=codigo:lat:lon,codigo:lat:lon
~~~

El forecast diario se expande al contrato horario para cubrir los tres días y
las horas coincidentes con el endpoint horario se superponen sobre esa
expansión. Esta operación mantiene la compatibilidad técnica con las features,
pero no crea detalle espacial inexistente. La salida se marca como
`fresh_aemet`, `grid=municipal` y `source_resolution=daily_expansion` o
`hourly`. Es válida para probar el circuito y como contingencia visible, no como
equivalente científico de WRF 1 km.

Si AEMET no aporta precipitación cuantitativa para una fecha, el parser deja el
valor ausente y la validación detiene la ejecución. El valor
`AEMET_MISSING_PRECIPITATION_FALLBACK=0` solo debe usarse en pruebas técnicas;
en ese caso la calidad pasa a `fresh_aemet_proxy` y el resultado no debe
utilizarse para evaluar el riesgo ni para movilizar recursos.

### 4.4 Estado meteorológico reciente

Las memorias de 3, 7, 14 y 30 días no pueden calcularse leyendo una vez el
histórico 2019–2024 si la ejecución ocurre en 2026. El pipeline mantiene por
eso un Parquet compacto:

~~~text
data/processed/state/weather_daily_state.parquet
~~~

| Campo | Significado |
|---|---|
| cell_id, fecha | Clave espacial y fecha local |
| tmax_vc, rhmin_vc, vmax_vc, prec_dia | Variables diarias comunes |
| vpd_vc | VPD derivado de temperatura y humedad |
| source | Observación, análisis, bootstrap, etc. |
| coverage_hours | Horas válidas del agregado |
| state_as_of | Momento en que se incorporó la fila |

El estado debe contener para cada celda los últimos 30 días completos anteriores
a la ejecución. A las 05:00 no se trata el día de emisión como completo: sus
observaciones todavía pueden ser parciales y no deben contaminar la memoria.

La actualización está desacoplada de la inferencia. El comando recibe un CSV o
Parquet diario ya producido por una ingesta de observaciones o análisis:

~~~bash
PYTHONPATH=. python scripts/update_weather_state.py \
  --input data/processed/observations/weather_daily_latest.parquet \
  --state data/processed/state/weather_daily_state.parquet \
  --source meteogalicia_observation
~~~

Esta frontera es intencionada. El colector AEMET actual ya fija el contrato,
autenticación, resolución, frecuencia, unidades y política de revisiones. Se
ejecuta con:

~~~bash
PYTHONPATH=. python scripts/ingest_aemet_current_observations.py \
  --loop --interval-hours 6
~~~

La API actual ofrece una ventana móvil, por lo que el colector acumula las
capturas en `aemet_hourly_observations.parquet` y solo cierra un día cuando
alcanza la cobertura mínima. Si la actualización no llega, la inferencia falla
con un error explícito; no convierte un estado desconocido en ceros.

Para el primer arranque con la clave AEMET se puede recuperar el rango completo
sin esperar treinta días:

~~~bash
PYTHONPATH=. python scripts/ingest_aemet_weather_state.py
~~~

Este comando consulta la climatología diaria AEMET para los últimos 30 días
publicados, por defecto hasta `hoy - 4 días`. Como la API limita cada rango a
15 días, realiza bloques consecutivos, obtiene una vez el inventario de estaciones, filtra Galicia,
interpola las variables a la rejilla mediante IDW de cuatro vecinos y publica
el Parquet de observaciones y el estado. Los JSON originales de ambos bloques
se conservan en
`data/raw/aemet/observations/`. El estado queda marcado por su fuente y no
debe confundirse con observaciones WRF.

La ingesta comprueba además que el rango solicitado esté completo. Si AEMET
solo ha publicado hasta una fecha anterior, el proceso se detiene antes de
publicar `weather_daily_latest.parquet` o actualizar el estado; no rellena los
días ausentes con ceros ni con datos históricos de otro año.

La operación numérica de MeteoSIX v5 no sirve como backfill: sus límites
temporales empiezan en el día actual y la API expone como máximo siete días de
forecast. La variable `precipitation_amount` es precipitación prevista durante
la hora anterior, no un histórico observado. Con MeteoGalicia se mantendrá por
tanto una capa de observaciones separada.

## 5. Tiempo, emisión, validez y horizonte

El pipeline distingue cuatro conceptos:

| Campo conceptual | Significado |
|---|---|
| issue_time | Momento en que el sistema decide y ejecuta |
| forecast_run_at / issued_at | Ejecución del modelo numérico declarada por el proveedor |
| downloaded_at | Momento en que nuestro sistema recibió la respuesta |
| valid_time | Hora futura a la que se refiere cada valor |

Internamente los instantes se normalizan a UTC. Para ventanas de negocio se
convierten a Europe/Madrid, incluyendo los cambios de horario de Galicia.

Ejemplo para una ejecución el 16 de agosto a las 05:00 CEST:

~~~text
issue_time       = 2026-08-16 05:00 Europe/Madrid
T+1 target date  = 2026-08-17 local
T+2 target date  = 2026-08-18 local
T+3 target date  = 2026-08-19 local
critical window  = 12:00, 13:00, ..., 18:00 local del día objetivo
~~~

Una fila a las 12:00 local se almacena con su equivalente UTC, pero la
agregación usa la hora local. Esto evita desplazar la ventana crítica por
confundir UTC con hora peninsular.

El campo horizon_hours se calcula como:

~~~text
horizon_hours = (valid_time - forecast_run_at) / 1 hora
~~~

Si MeteoSIX no expone el run meteorológico, el sistema usa la hora de descarga
como referencia y lo marca con forecast_run_source = download_time. Este hecho
aparece en el manifiesto y el dashboard.

## 6. Descarga y conservación del forecast

### 6.1 Selección de puntos

sample_provider_points() agrupa los centroides de la rejilla de 1 km en
una resolución de consulta configurable. Por defecto se consultan puntos
representativos cada 4 km, suficiente para limitar el número de peticiones en
la primera versión. El resultado contiene un point_id para cada localización
representativa y después se asigna al grid de 1 km.

El objetivo es evitar unas 30.697 peticiones independientes. El cliente divide
los puntos en lotes de 20, límite configurado para MeteoSIX.

### 6.2 Petición

Cada lote solicita:

1. coordenadas de los puntos;
2. temperatura, humedad relativa, precipitación y viento;
3. modelo WRF y rejilla configurada;
4. unidades explícitas;
5. idioma y zona horaria;
6. startTime y endTime de los tres días locales;
7. la clave API desde la variable de entorno.

La clave no se escribe en el JSON bruto ni en el manifiesto. Sí se guardan la
URL del endpoint y los parámetros no secretos necesarios para reproducir.

### 6.3 Reintentos y fallback

El cliente reintenta errores HTTP, JSON inválido y errores controlados con
backoff exponencial. Los parámetros se ajustan en .env:

~~~text
METEOGALICIA_TIMEOUT_SECONDS=60
METEOGALICIA_MAX_RETRIES=3
~~~

Si la malla 1 km falla por descarga, ausencia de variables u horas, se repite
el proceso con la malla 04 km. Ese resultado se etiqueta
`forecast_quality = fresh_fallback` y se conserva `forecast_grid = 04km`.
Únicamente si también falla 04 km la inferencia intenta cargar el último
Parquet archivado que cubra completamente T+1, T+2 y T+3. Ese resultado se
etiqueta `forecast_quality = stale`.

No se hace lo siguiente:

- usar una fila histórica de df_master como forecast futuro;
- inventar horas ausentes con ceros;
- copiar el último día histórico para ocultar el fallo;
- aplicar Quantile Mapping con los valores ordenados de una sola fecha.

Por cada descarga se conservan:

~~~text
data/raw/meteogalicia/raw/forecast_<downloaded>_batch_<n>.json
data/raw/meteogalicia/raw/forecast_<downloaded>_batch_<n>.metadata.json
data/raw/meteogalicia/forecast_<forecast_run_at>.parquet
~~~

En la implementación actual los JSON de cada malla se separan en
`raw/1km/` y `raw/04km/`, de forma que un fallback no sobreescribe la
evidencia de la fuente preferida.

El JSON es la evidencia del proveedor. El Parquet es el contrato interno
normalizado y asignado a la rejilla de 1 km.

## 7. Parser y validación de calidad

El parser aplana la respuesta GeoJSON. Cada valor de variable se convierte en
una fila intermedia y después las variables se pivotan a una fila por
point_id, valid_time y forecast_run_at.

Los valores -9999, infinitos, cadenas vacías y null se convierten en NaN. Esto
es normalización, no imputación: la validación debe rechazar el forecast si
esos ausentes afectan a la ventana requerida.

validate_hourly_forecast() exige:

- columnas horarias obligatorias;
- al menos un punto;
- timestamps válidos;
- cobertura dentro de la ventana;
- temperatura, humedad, precipitación y viento presentes;
- humedad entre 0 y 100 %;
- precipitación y viento no negativos;
- horas completas cuando require_hourly=True;
- distancia máxima razonable al punto WRF.

Después de asignar a la rejilla, _validate_grid_coverage() repite la
comprobación por cell_id. Es necesario validar por celda porque una cobertura
global puede ocultar una celda sin horas.

La distancia espacial queda en source_distance_km. El límite por defecto es
10 km y se ajusta con:

~~~text
FORECAST_MAX_SOURCE_DISTANCE_KM=10
~~~

El límite no convierte una malla de 4 km en observación puntual de 1 km: evita
asignaciones absurdamente alejadas y conserva trazabilidad de la aproximación.

## 8. Conversión de horas a variables diarias

aggregate_hourly_forecast() recibe el forecast horario asignado y produce una
fila por cell_id y fecha local.

### 8.1 Ventana crítica

La ventana crítica es inclusiva:

~~~text
12:00 ≤ hora local ≤ 18:00
~~~

Sobre esas siete horas se calcula:

~~~text
tmax_vc = max(temperature_c)
rhmin_vc = min(relative_humidity_pct)
vmax_vc = max(wind_speed_kmh)
vpd_vc = max(VPD horario)
~~~

La precipitación diaria usa todas las horas locales del día:

~~~text
prec_dia = Σ precipitation_mm de las horas del día
~~~

La validación trabaja con el número real de horas que tenga el día al cambiar
el horario; no se debe asumir que un día UTC y un día local son idénticos.

### 8.2 VPD

El déficit de presión de vapor aproxima la demanda evaporativa. Con
temperatura en °C y humedad en porcentaje:

~~~text
e_s(T) = 0.6108 × exp(17.27 × T / (T + 237.3))
VPD    = e_s(T) × (1 - RH / 100)
~~~

El resultado queda en kPa. Temperaturas altas y humedades bajas elevan el VPD,
representando una atmósfera con mayor capacidad de extraer agua del combustible
fino.

## 9. Construcción de features y control de leakage

build_operational_features() combina:

1. meteorología del día objetivo del forecast;
2. memoria meteorológica previa del estado y, para T+2/T+3, de los días
   intermedios previstos;
3. atributos estáticos y calendario.

Para un objetivo target_date, una ventana de n días usa:

~~~text
[target_date - n días, target_date)
~~~

El extremo derecho es abierto. Nunca se incluye el propio día objetivo.

| Feature | Cálculo |
|---|---|
| prec_acum_3d | suma de lluvia previa en 3 días |
| prec_acum_7d | suma de lluvia previa en 7 días |
| prec_acum_14d | suma de lluvia previa en 14 días |
| prec_acum_30d | suma de lluvia previa en 30 días |
| tmax_media_7d | media de máximas previas |
| rhmin_media_7d | media de humedades mínimas previas |
| vmax_media_7d | media de viento máximo previo |
| dias_sin_lluvia | contador mientras prec_dia < 1 mm |

Para T+1 las memorias terminan en observaciones anteriores a issue_time. Para
T+2 el día T+1 puede alimentar la memoria y para T+3 pueden participar T+1 y
T+2. Así se refleja la información realmente disponible en cada horizonte.

El esquema operativo actual incluye:

~~~text
tmax_vc, rhmin_vc, vmax_vc, prec_dia, vpd_vc
prec_acum_3d, prec_acum_7d, prec_acum_14d, prec_acum_30d
tmax_media_7d, rhmin_media_7d, vmax_media_7d, dias_sin_lluvia
alerta_30_30
altitud_media, pendiente_media, orientacion_media
combustible_pct_forestal
mes, dia_semana, es_finde, dia_anio_sin, dia_anio_cos
~~~

alerta_30_30 vale uno cuando la máxima alcanza al menos 30 °C y la humedad
mínima es como máximo 30 %. El calendario usa codificación cíclica.

El modelo recibe exactamente OPERATIONAL_FEATURES. FEATURE_SCHEMA_VERSION =
operational-risk-v1 se guarda en los artefactos y se valida al cargarlos. Si
cambia el orden, nombre o significado de una feature, hay que incrementar la
versión y reentrenar.

La meteorología horaria ausente debe provocar un fallo antes de crear
features. El fillna(0) de ensure_feature_matrix() es una defensa final para
columnas estáticas opcionales, no una forma válida de ocultar un forecast
incompleto.

## 10. Estado reciente y actualización diaria

La operación tiene dos entradas temporales:

~~~text
estado observado/analizado hasta D-1
forecast MeteoGalicia desde D+1 hasta D+3
~~~

La fecha D se excluye porque a las 05:00 no está cerrada. Esta decisión evita
usar información que solo aparecería al final del día.

validate_weather_state() comprueba:

- cobertura por cada celda;
- 30 fechas previas completas;
- ausencia de valores meteorológicos;
- humedad en [0, 100];
- viento y lluvia no negativos;
- cobertura diaria no superior a 24 horas;
- temperatura dentro de un rango físico amplio.

Si falta una fecha, la inferencia no continúa. No se debe convertir la ausencia
en prec_dia = 0, porque eso haría parecer que hubo un día seco.

## 11. Entrenamiento de modelos

build_historical_horizon_dataset() genera tres DataFrames, uno por horizonte.
Cada fila contiene:

~~~text
fecha         = día objetivo
issue_date    = fecha - horizon_days
horizon_days  = 1, 2 o 3
target        = ignición del día objetivo
source_type   = era5_perfect
~~~

La meteorología del día objetivo es conocida en ERA5-Land. Este benchmark
evalúa la relación entre meteorología real y riesgo; todavía no mide el error
de MeteoGalicia. Los tres DataFrames se entrenan y calibran por separado.

El siguiente paso es construir pares forecast-observación. Para cada emisión se
conserva la predicción, se espera la observación consolidada y se une por:

~~~text
(cell_id, issued_at, valid_time, variable)
~~~

La configuración temporal es:

| Partición | Años |
|---|---|
| Entrenamiento | 2019–2021 |
| Validación | 2022 |
| Test | 2023–2024 |

No se usa K-Fold aleatorio. Mezclar días del mismo año entre train y test
produce una estimación demasiado optimista.

Para hacer entrenable LightGBM se muestrean negativos en entrenamiento con una
relación por defecto de 50 negativos por positivo. El entrenamiento local de
gran volumen procesa los años secuencialmente y aplica un límite de filas por
año después de construir las ventanas temporales; en ese modo las métricas
describen la muestra conservada y deben etiquetarse como desarrollo. El
benchmark final debe ejecutarse con evaluación sobre los años completos. Las
métricas prioritarias son PR-AUC, ROC-AUC y Brier Score.

Cada horizonte se guarda como:

~~~text
data/models/forecast_risk_t1.joblib
data/models/forecast_risk_t2.joblib
data/models/forecast_risk_t3.joblib
~~~

El artefacto contiene LightGBM, calibrador, features, horizonte y metadatos.
El calibrador isotónico se ajusta en validación y nunca en inferencia.

| Campo de salida | Interpretación |
|---|---|
| prob_risk / prob_riesgo | probabilidad calibrada |
| percentil_riesgo | posición relativa de la celda |
| risk_level / nivel_riesgo | etiqueta basada en percentiles |

El percentil no es una probabilidad: estar en el percentil 99 no significa
tener 99 % de riesgo.

## 12. Inferencia diaria paso a paso

### Paso 0: adquisición del estado

Antes de las 05:00 una tarea de ingesta incorpora observaciones o análisis
cerrados. El estado resultante debe tener 30 días completos por celda.

### Paso 1: lock

run_daily_inference_pipeline() crea:

~~~text
data/processed/.daily_inference.lock
~~~

Si otro proceso está ejecutándose, la segunda instancia termina. El lock no se
elimina automáticamente tras una caída: el operador debe comprobar PID y
timestamp antes de retirar un lock abandonado.

### Paso 2: rejilla

Se valida existencia, lectura Parquet y columnas espaciales. Una rejilla
corrupta detiene el proceso antes de publicar.

### Paso 3: estado

Se carga weather_daily_state.parquet y se valida la ventana reciente. La ruta
de producción no usa el histórico completo de df_master; ese argumento queda
para compatibilidad explícita y el manifiesto lo marca como legacy_master_argument.

### Paso 4: forecast

Se consulta MeteoSIX, se guardan JSON y metadatos, se parsea y se asigna a la
rejilla. La hora de descarga se separa de la hora de run del proveedor.

### Paso 5: validación

Se exige la ventana completa de tres días, variables obligatorias y cobertura
por cada cell_id.

### Paso 6: selección de malla y fallback

Se prueba WRF 1 km y se valida descarga, variables, horas y cobertura por
celda. Si no supera la validación, se intenta WRF 04 km. Solo si las dos
mallas fallan y `allow_stale=True`, se carga el último Parquet que cubra el
intervalo. Los estados quedan marcados como `fresh`, `fresh_fallback` o
`stale`; nunca se sustituyen por una fila histórica.

### Paso 7: features

Se agregan las horas a días locales y se combinan estado y días futuros
previstos. Se generan las filas de los tres horizontes.

### Paso 8: modelos

Se carga el joblib de cada horizonte. No se reentrena LightGBM. El loader
verifica horizonte, esquema operational-risk-v1 y lista de features.

### Paso 9: scoring y prioridad preventiva

El modelo produce un score crudo, el calibrador lo transforma y se limita a
[0, 1]. Después se calculan percentiles, niveles y una acción preventiva
orientativa (`vigilancia_rutinaria`, `vigilancia_reforzada`,
`preposicion_medios` o `preposicion_prioritaria`). La acción se interpreta como
ranking para recursos limitados, no como una orden autónoma.

### Paso 10: publicación

La salida se escribe en un temporal y se publica con os.replace solo tras una
escritura correcta. El dashboard nunca debe observar un Parquet a medio escribir.

### Paso 11: manifiesto

Se crean:

~~~text
data/processed/predicciones_operativas.parquet
data/processed/predicciones_operativas.json
data/processed/predicciones_operativas.manifest.json
~~~

El manifiesto incluye run ID, tiempos, calidad, cobertura, estado, modelos,
checksums, filas, celdas y duración.

## 13. Contrato de salida

Cada fila conserva las features y metadatos usados:

~~~text
cell_id, fecha, horizon_days
issue_time, issue_date_local, generated_at
forecast_run_at, forecast_downloaded_at
forecast_quality, forecast_selection, forecast_age_hours
forecast_grid, forecast_api_version, forecast_query_resolution_km
forecast_requested_points
provider, model, model_version, grid
forecast_source_distance_km
tmax_vc, rhmin_vc, vmax_vc, prec_dia, vpd_vc
prec_acum_3d, prec_acum_7d, prec_acum_14d, prec_acum_30d
prob_risk, prob_riesgo, percentil_riesgo, operational_priority
risk_level, nivel_riesgo, recommended_action
~~~

Los campos estáticos se conservan para que el dashboard pueda mostrar
combustible y topografía sin otra unión obligatoria.

## 14. Dashboard y lectura correcta

El dashboard carga el Parquet publicado, ofrece T+1/T+2/T+3, calcula la fecha
desde los datos de ejecución, muestra calidad, edad y cobertura, colorea el
mapa y permite inspeccionar una celda.

El ámbito geográfico actual es exclusivamente Galicia. El mapa atenúa el
territorio exterior y dibuja el límite de la comunidad para evitar que el
mapa base se interprete como una predicción nacional. Las provincias y celdas
que aparecen fuera de Galicia no forman parte del resultado; para ampliar el
producto a España habría que crear una rejilla, histórico, fuente meteorológica,
etiquetas y validación específicos para cada nueva región.

La explicación local usa src/models/explainability.py y el mismo
ForecastRiskModel que hizo la predicción. Las contribuciones están en la
escala nativa del árbol, normalmente log-odds. Un valor positivo empuja el
score hacia mayor riesgo y uno negativo hacia menor riesgo. No son porcentajes
ni causalidad física.

Si shap no está instalado o el artefacto no es explicable, el dashboard muestra
explicabilidad no disponible; no inventa pesos para temperatura, humedad o
combustible.

## 15. Calidad, stale y políticas

| Metadato | Qué responde |
|---|---|
| forecast_quality | ¿WRF 1 km fresco, fallback 04 km, AEMET alternativo/proxy o archivo reutilizado? |
| forecast_grid | ¿Qué malla o soporte espacial se utilizó realmente? |
| forecast_selection | ¿Primaria, fallback, archivo o prueba inyectada? |
| forecast_provider | ¿MeteoGalicia o AEMET respondió? |
| forecast_api_version | ¿Qué versión de MeteoSIX/OpenData respondió? |
| forecast_run_source | ¿emisión declarada por proveedor o aproximada a descarga? |
| forecast_age_hours | ¿qué edad tiene la emisión? |
| forecast_coverage | ¿qué proporción horaria cubre cada celda? |
| source_distance_km | ¿qué distancia tiene la asignación espacial? |

Política conservadora:

- fresh y cobertura completa: publicar normalmente;
- fresh_fallback y cobertura completa: publicar con advertencia de resolución
  degradada y confirmación humana para movilizaciones críticas;
- fresh_aemet y cobertura completa: publicar solo como prueba o contingencia
  visible, con advertencia de resolución municipal y posibilidad de bloquearlo;
- fresh con edad o distancia anómala: advertir o bloquear;
- stale completo: publicar solo como contingencia visible;
- stale incompleto: no publicar;
- sin forecast válido: no generar un mapa que parezca actual.

Health check:

~~~bash
PYTHONPATH=. python scripts/check_operational_run.py
PYTHONPATH=. python scripts/check_operational_run.py --allow-stale
~~~

El segundo comando acepta stale explícitamente para una contingencia controlada.

## 16. Observabilidad y auditoría

Cada ejecución debe contestar:

1. ¿Cuándo se ejecutó la decisión?
2. ¿Qué forecast se utilizó y cuándo se emitió/descargó?
3. ¿Qué versión de modelo y features produjo cada horizonte?
4. ¿Qué filas y celdas se publicaron?
5. ¿El archivo visible coincide con el generado?

El checksum SHA-256 del output y de los modelos se guarda en el manifiesto.

Indicadores recomendados:

| Indicador | Umbral inicial |
|---|---:|
| duración total | inferior al intervalo operativo |
| celdas publicadas | igual a la rejilla esperada |
| cobertura mínima | 100 % normal |
| edad del forecast | inferior a 30 h |
| forecasts stale consecutivos | 0 ideal |
| probabilidades | sin NaN y en [0, 1] |
| estado reciente | 30 días por celda |
| distancia máxima | menor que el límite configurado |

En una instalación real conviene añadir logs estructurados, métricas de API,
número de reintentos, HTTP status, tiempos por etapa y alertas.

## 17. Seguridad, retención y reproducibilidad

Las claves API se cargan desde .env o un gestor de secretos y no se incluyen
en artefactos. Los JSON brutos deben tener una política de retención coherente
con el uso del proyecto.

La retención del estado reciente es de 60 días por defecto. El forecast bruto
se conserva para la futura evaluación forecast-observación. Los modelos se
versionan junto con métricas y checksums.

Para reproducir un resultado se deben conservar:

~~~text
run_id
issue_time
forecast_run_at
forecast_downloaded_at
JSON bruto de MeteoGalicia
Parquet normalizado/asignado
estado meteorológico usado
checksum de los tres modelos
versión del esquema de features
~~~

## 18. Pruebas y criterios de aceptación

La suite sintética cubre:

- parser de variables y unidades;
- conversión UTC a hora local;
- ventana crítica 12:00–18:00;
- valores -9999;
- columnas meteorológicas ausentes;
- asignación espacial y distancia;
- batching de 20 puntos;
- archivado stale;
- separación de emisión y descarga;
- prevención de leakage respecto a issue_time;
- estado incompleto o inválido;
- lock exclusivo;
- inferencia sintética reproducible;
- tres horizontes.

Criterios de aceptación:

1. una respuesta sintética produce los tres horizontes reproduciblemente;
2. una hora ausente detiene la publicación;
3. una variable obligatoria ausente detiene la publicación;
4. -9999 no se transforma en riesgo meteorológico cero;
5. ninguna observación posterior entra en features;
6. un forecast antiguo queda visible como stale;
7. dos ejecuciones no publican simultáneamente;
8. el manifiesto permite verificar checksums;
9. el dashboard muestra fecha dinámica, edad y calidad;
10. el dashboard no etiqueta heurísticas como SHAP.

## 19. Procedimientos

### Entrenamiento offline

~~~bash
conda activate incendios-forestales
PYTHONPATH=. python scripts/train_forecast_models.py \
  --dataset-dir misc/Dataset/Mike \
  --output-dir data/models
~~~

El comando anterior usa el entrenamiento acotado por memoria. El límite se
puede ajustar con `--rows-per-year 200000` o `--rows-per-year 500000` según la
RAM disponible. `--max-rows` solo sirve para smoke tests y toma las primeras
filas de cada fichero; no debe emplearse para generar el modelo que se reporte
como benchmark final.

Antes de promover modelos se revisan positivos/negativos, PR-AUC, Brier Score,
prevalencia real de test, esquema, estabilidad por años, distribución de
probabilidades y checksum.

### Operación diaria

~~~bash
# Observaciones cerradas hasta D-1 y estado móvil de 30 días
PYTHONPATH=. python scripts/ingest_aemet_weather_state.py

# Forecast, features, scoring, publicación y manifiesto
PYTHONPATH=. python scripts/run_daily_inference.py

# Health check
PYTHONPATH=. python scripts/check_operational_run.py
~~~

Para una prueba local se puede reutilizar un forecast horario que ya se haya
descargado, evitando repetir las consultas por lotes al proveedor:

~~~bash
PYTHONPATH=. python scripts/run_daily_inference.py \
  --no-stale \
  --forecast-file data/raw/meteogalicia/forecast_YYYYMMDDTHHMMSSZ_1km.parquet
~~~

`--forecast-file` no desactiva las validaciones: el Parquet debe contener las
72 horas, las variables obligatorias y la asignación a las celdas esperadas.
La opción está pensada para depuración y recuperación de una ejecución local;
la operación diaria debe seguir descargando la emisión más reciente del
proveedor.

Si el estado no tiene cobertura, se corrige la ingesta. Si el paso 2 usa stale,
el health check falla por defecto. La política final de bloqueo o publicación
contingente debe decidirla el responsable operativo.

Dashboard:

~~~bash
streamlit run app.py
~~~

Para pruebas se puede llamar a
run_daily_inference_pipeline(forecast_df=..., history_df=..., grid_df=...). La
inyección explícita de pruebas no cambia la ruta de producción.

## 20. Backtest actual y backtest futuro

El comando actual ejecuta el benchmark ERA5:

~~~bash
PYTHONPATH=. python scripts/run_backtest_verification.py
~~~

Responde a: “¿qué tal ordena y calibra el modelo cuando las features
meteorológicas del día objetivo son conocidas mediante ERA5-Land?”. Todavía no
responde a: “¿qué tal funciona con el forecast que estaba disponible 24, 48 o
72 horas antes?”.

El backtest de vintages debe:

1. seleccionar una fecha histórica de ejecución;
2. cargar el forecast archivado anterior a esa ejecución;
3. recortar la información disponible en esa emisión;
4. generar features con la función de producción;
5. cargar observaciones consolidadas del día válido;
6. calcular target y métricas por horizonte;
7. comparar con ERA5 perfecto.

La diferencia cuantificará degradación por error meteorológico, agregación
espacial y envejecimiento del forecast.

## 21. Corrección estadística futura

No se activa corrección meteorológica porque faltan pares históricos
forecast-observación. Una corrección válida debe aprender del error histórico
por variable, horizonte, estación, celda o región, régimen meteorológico y
versión WRF.

Se entrenará en un periodo y se validará en otro, comparando:

1. modelo con ERA5 real;
2. modelo con forecast bruto;
3. modelo con forecast corregido.

El forecast bruto siempre se conservará para auditoría.

## 22. Mejoras operativas siguientes

### Observaciones de mayor resolución

La ingesta AEMET cubre el bootstrap y la reconciliación diaria mediante
climatología de estaciones, y el colector horario cubre el cierre provisional
de D-1. La siguiente mejora es conectar una fuente observacional con mayor
resolución temporal/espacial —por ejemplo una API de observación de
MeteoGalicia— y conservar el mismo contrato de salida. El histórico AEMET
seguirá siendo el mecanismo de arranque hasta validar esa fuente.

### Forecast multi-modelo

Comparar WRF con una segunda fuente. La dispersión puede convertirse en señal
de incertidumbre solo después de validarla.

### Incertidumbre espacial

Evaluar interpolación, elevación y representatividad de la asignación más
cercana, conservando source_distance_km.

### Detección de drift

Comparar distribuciones diarias de temperatura, humedad, lluvia, VPD y
probabilidades con periodos históricos, alertando ante cambios sostenidos.

### Promoción versionada

Guardar model_release_id, métricas, fecha, fuente, esquema y checksums.
Promover los tres horizontes como un conjunto coherente.

### Alertas de decisión

Añadir ranking, agregación municipal, persistencia y notificaciones, siempre
incluyendo fecha, horizonte, calidad y versión de modelo.

## 23. Glosario

| Término | Definición |
|---|---|
| Reanálisis | Reconstrucción retrospectiva del estado atmosférico |
| Forecast | Predicción futura de un modelo numérico |
| issue_time | Hora de nuestra decisión |
| issued_at | Hora de emisión meteorológica |
| valid_time | Hora futura del valor |
| VPD | Déficit de presión de vapor |
| Ventana crítica | Horas locales 12:00–18:00 |
| Leakage | Uso de información no disponible al decidir |
| Stale | Forecast archivado reutilizado |
| Calibración | Transformación para mejorar interpretación probabilística |
| Percentil | Posición relativa de una celda |
| Manifest | Procedencia, calidad y checksums |
| TreeSHAP | Descomposición local de un modelo de árboles |

## 24. Resumen ejecutivo

La arquitectura separa meteorología, estado reciente, features, modelo de
incendios y presentación. MeteoGalicia proporciona el futuro meteorológico;
ERA5-Land reconstruye el pasado; LightGBM estima riesgo; tres artefactos
serializados representan T+1, T+2 y T+3; el dashboard expone probabilidad,
calidad y procedencia.

La propiedad más importante no es solo producir un mapa, sino poder explicar
qué información utilizó, cuándo estaba disponible, qué forecast recibió, qué
falló, si reutilizó un forecast antiguo y qué versión publicó el resultado.
Esa trazabilidad convierte un experimento de machine learning en una base
operativa verificable.
