# Automatización meteorológica de producción

Este documento describe la automatización que debe quedar activa en el servidor
para que el pipeline no dependa de ejecutar manualmente los comandos de
ingesta. La aplicación mantiene dos tipos de meteorología con funciones
distintas:

1. observaciones recientes, necesarias para construir las memorias de los
   últimos 30 días;
2. previsión MeteoGalicia WRF, necesaria para las variables del día objetivo
   T+1, T+2 y T+3.

La previsión se descarga dentro de `run_daily_inference.py`. La ingesta de
observaciones se ejecuta como servicios systemd independientes y, además, la
inferencia hace una comprobación de seguridad de MeteoGalicia antes de empezar.

## Flujo diario

```text
04:00, 04:30, 05:00
    └─ MeteoGalicia EMA D-1 → interpolación IDW → weather_daily_state.parquet

cada 6 horas
    └─ AEMET actual → parquet horario → cierre de días cuando hay ≥20 horas

05:15 y 10:15
    └─ refresco MeteoGalicia D-1 (guardia) → forecast WRF → features →
       modelos T+1/T+2/T+3 → predicciones + manifest → caché del dashboard

06:00
    └─ climatología diaria AEMET publicada con retraso → reconciliación histórica

cada 30 minutos
    └─ health check de la última salida publicada
```

El proceso MeteoGalicia devuelve rápidamente `already_up_to_date` cuando el
estado ya contiene D-1. Las tres horas son reintentos: si el servicio no estaba
disponible a las 04:00, vuelve a intentarlo antes de la inferencia.

## Fuentes y prioridad

MeteoGalicia EMA es la fuente observada primaria para Galicia. AEMET cumple dos
funciones de contingencia:

- el colector actual acumula medidas horarias cuando el día todavía no está
  consolidado;
- la climatología diaria rellena o reconcilia fechas publicadas con retraso.

Todas las fuentes escriben el mismo estado operativo para que las memorias de
precipitación y sequedad sean continuas. Para evitar que dos servicios
concurrentes publiquen resultados incompatibles, usan:

```text
data/processed/.weather_state_ingest.lock
```

Además del lock, `src/ingestion/weather_state.py` aplica prioridad semántica
al hacer el upsert:

| Fuente | Prioridad | Uso |
|---|---:|---|
| `meteogalicia_ema_idw` | 300 | Observación diaria primaria |
| `aemet_daily_climatology_idw` | 200 | Reconciliación atrasada |
| `aemet_current_observation_idw` | 100 | Continuidad provisional |
| `forecast_proxy` | 0 | Solo escenarios explícitos de contingencia |

Por tanto, una fila AEMET descargada posteriormente no sustituye una fila
MeteoGalicia válida para la misma celda y fecha. El timestamp
`state_as_of` solo decide entre filas de la misma prioridad.

La ingesta también aplica dos protecciones de integridad. Si el Parquet de
estado ya existe pero no se puede leer, la ejecución termina sin escribir un
estado parcial. Si una combinación de filas produjese menos historia que la
retenida actualmente, el upsert se cancela. Estas comprobaciones son
importantes porque el estado es una entrada de seguridad para las memorias de
30 días: ante un error de lectura o de fusión se debe fallar cerradamente, no
publicar cinco días como si fueran un estado completo.

## Unidades versionadas

Las plantillas se encuentran en `deploy/systemd/` y se instalan con:

```bash
sudo bash /srv/fire-risk/app/scripts/install_systemd_units.sh --enable --start
```

El instalador copia y recarga estas unidades:

```text
fire-risk-meteogalicia-observations.service
fire-risk-meteogalicia-observations.timer
fire-risk-aemet-current.service
fire-risk-state.service
fire-risk-state.timer
fire-risk-inference.service
fire-risk-inference.timer
fire-risk-health.service
fire-risk-health.timer
fire-risk-dashboard.service
```

Después de instalar una release nueva:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fire-risk-meteogalicia-observations.timer
sudo systemctl enable --now fire-risk-state.timer
sudo systemctl enable --now fire-risk-aemet-current.service
sudo systemctl enable --now fire-risk-inference.timer
sudo systemctl enable --now fire-risk-health.timer
sudo systemctl restart fire-risk-dashboard.service
```

El script de instalación ejecuta los mismos pasos cuando se le pasan `--enable`
y `--start`; los comandos anteriores son útiles si solo se ha hecho
`daemon-reload` manualmente.

## Variables obligatorias del servidor

En `/srv/fire-risk/config/.env` deben existir, además de la clave MeteoSIX y
las rutas de producción:

```dotenv
PIPELINE_ENVIRONMENT=production
LOCAL_SIMULATION_MODE=false
FORECAST_PROVIDER=auto
FORECAST_AUTO_AEMET_FALLBACK=true
METEOGALICIA_API_KEY=...
GRID_PATH=/srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet
WEATHER_STATE_PATH=/srv/fire-risk/data/processed/state/weather_daily_state.parquet
METEOGALICIA_OBSERVATIONS_PATH=/srv/fire-risk/data/processed/observations/weather_daily_meteogalicia.parquet
METEOGALICIA_OBSERVATIONS_RAW_DIR=/srv/fire-risk/data/raw/meteogalicia/observations
WEATHER_STATE_INGEST_LOCK_PATH=/srv/fire-risk/data/processed/.weather_state_ingest.lock
```

La unidad systemd pasa las rutas críticas explícitamente para que una variable
mal escrita no redirija accidentalmente la escritura a otro directorio. La
variable común del lock se mantiene documentada para las ejecuciones manuales.

## Comprobación tras el despliegue

### 1. Comprobar que las unidades existen y están activas

```bash
systemctl list-unit-files | grep fire-risk
systemctl list-timers --all | grep fire-risk
systemctl status fire-risk-aemet-current.service
systemctl status fire-risk-dashboard.service
```

Se esperan tres timers principales de datos/publicación: MeteoGalicia,
reconciliación AEMET e inferencia; además del timer de health.

### 2. Ejecutar manualmente la ingesta MeteoGalicia

```bash
sudo systemctl start fire-risk-meteogalicia-observations.service
sudo journalctl -u fire-risk-meteogalicia-observations.service -n 100 --no-pager
```

Resultado normal si no hay hueco:

```text
status=already_up_to_date
```

Resultado normal si hay hueco: `status=success`, `closed_days` igual al número
de días recuperados y todas las celdas presentes en el estado.

### 3. Ejecutar la reconciliación AEMET

```bash
sudo systemctl start fire-risk-state.service
sudo journalctl -u fire-risk-state.service -n 100 --no-pager
```

No debe interpretarse como una sustitución de MeteoGalicia. La prioridad de
fuentes mantiene la observación primaria cuando ambas cubren la misma fecha.

### 4. Ejecutar una inferencia controlada

```bash
sudo systemctl reset-failed fire-risk-inference.service
sudo systemctl start fire-risk-inference.service
sudo journalctl -u fire-risk-inference.service -f
```

La primera ejecución puede tardar varios minutos porque consulta lotes de
puntos a MeteoSIX. El servicio está limitado a 30 minutos. Si se interrumpe,
primero hay que comprobar el PID del proceso y eliminar el lock solo cuando el
proceso ya no exista:

```bash
cat /srv/fire-risk/data/processed/.daily_inference.lock
ps -p PID -o pid,etime,cmd
sudo rm /srv/fire-risk/data/processed/.daily_inference.lock
```

No se debe borrar un lock de un proceso activo.

### 5. Comprobar el output y el health check

```bash
sudo systemctl start fire-risk-health.service
sudo journalctl -u fire-risk-health.service -n 100 --no-pager
```

El resultado válido debe indicar, como mínimo:

```text
status=ok
forecast_quality=fresh
degraded=false
minimum_coverage=1.0
cells=29601
```

Si el resultado es `stale`, `degraded` o `unavailable`, no debe presentarse
como una ejecución normal. El dashboard debe mostrar la advertencia asociada.

### 6. Evaluar el forecast frente a observaciones posteriores

Los Parquet horarios conservados en `data/raw/meteogalicia/` permiten construir
una evaluación progresiva sin modificar la inferencia ni el estado meteorológico.
El script `scripts/evaluate_meteogalicia_forecasts.py` agrega cada forecast con
la misma ventana diaria que usa producción y lo empareja, por `cell_id`, con el
estado observado cuando ya se ha cerrado la fecha objetivo.

En el servidor se ejecuta así:

```bash
sudo -u fire-risk bash -lc '
set -Eeuo pipefail
cd /srv/fire-risk/app
PYTHONPATH=/srv/fire-risk/app \
/opt/miniconda3/envs/incendios-forestales/bin/python \
scripts/evaluate_meteogalicia_forecasts.py \
  --forecast-dir /srv/fire-risk/data/raw/meteogalicia \
  --state /srv/fire-risk/data/processed/state/weather_daily_state.parquet \
  --output-dir /srv/fire-risk/data/processed/evaluation/meteogalicia
'
```

El informe se guarda en:

```text
/srv/fire-risk/data/processed/evaluation/meteogalicia/
├── meteogalicia_forecast_metrics.json
└── meteogalicia_forecast_metrics.csv
```

Para inspeccionarlo:

```bash
sudo jq '{forecast_files_found, comparison_cases, observation_date_range,
  issue_date_sources, metrics, skipped}' \
  /srv/fire-risk/data/processed/evaluation/meteogalicia/meteogalicia_forecast_metrics.json
```

El informe calcula MAE, RMSE y sesgo para `tmax_vc`, `rhmin_vc`, `vmax_vc`
y `prec_dia`, además del acierto de lluvia/no lluvia con umbral de 1 mm,
separando T+1, T+2 y T+3. Excluye filas `forecast_proxy` y no usa datos
posteriores a la fecha de emisión para reconstruir el forecast.

`comparison_cases=0` no significa que el proveedor haya fallado: indica que
todavía no existe una observación posterior para cerrar ninguna de las fechas
objetivo. Una o varias comparaciones sirven como caso de estudio preliminar;
no deben presentarse como validación estadística de toda la temporada.

Este comando evalúa la calidad de la entrada meteorológica. No calcula todavía
PR-AUC, Brier score ni calibración del modelo de riesgo. Para esas métricas hay
que conservar cada predicción bajo su `run_id` y esperar las etiquetas EGIF de
las fechas objetivo.

## Fallos y recuperación

### MeteoGalicia no responde

El timer reintenta a las 04:30 y 05:00. La inferencia vuelve a ejecutar el
relleno de observaciones como guardia. Si el forecast sí está disponible pero
el estado de observaciones no alcanza los 30 días, la inferencia debe fallar
cerrada: es preferible no publicar un mapa que inventar acumulaciones.

### AEMET no responde

El colector horario se reinicia automáticamente por systemd. La climatología
diaria se volverá a intentar al día siguiente. Mientras MeteoGalicia siga
aportando D-1, AEMET no es necesaria para que el estado avance.

### Inferencia tarda demasiado

La primera medida es comprobar la resolución de consulta MeteoSIX:

```dotenv
METEOGALICIA_QUERY_RESOLUTION_KM=8.0
```

La malla del forecast sigue siendo WRF 1 km; esta variable controla la densidad
de los puntos consultados. Un valor menor mejora la representación espacial,
pero aumenta las peticiones y puede superar el tiempo de la unidad systemd.
Debe cambiarse tras medir tiempos y respetar los límites de MeteoSIX.

### Reinicio del servidor

Los timers con `Persistent=true` recuperan una ejecución perdida. Después de
un reinicio se debe comprobar:

```bash
systemctl is-active fire-risk-meteogalicia-observations.timer
systemctl is-active fire-risk-state.timer
systemctl is-active fire-risk-aemet-current.service
systemctl is-active fire-risk-inference.timer
systemctl is-active fire-risk-health.timer
```

## Qué queda automatizado y qué no

Queda automatizado:

- captura AEMET cada seis horas;
- actualización de observaciones EMA MeteoGalicia con reintentos;
- reconciliación diaria AEMET;
- descarga del forecast WRF durante inferencia;
- ejecución de los tres horizontes;
- publicación atómica de resultados;
- comprobación periódica de salud;
- arranque del dashboard.

No queda resuelto por esta automatización:

- una serie temporal suficientemente larga de forecasts MeteoGalicia para
  generalizar el error real por horizonte (el evaluador anterior permite
  empezar a archivarla y analizar casos cerrados);
- la corrección estadística forecast-observación;
- alertas externas por correo, Slack o PagerDuty;
- copias de seguridad fuera del servidor;
- autenticación de usuarios, TLS y configuración Nginx.

Estas capacidades deben añadirse como observabilidad y seguridad del servidor,
no mezclarse con la generación de features.
