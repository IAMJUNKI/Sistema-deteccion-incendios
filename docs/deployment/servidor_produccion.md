# Despliegue del pipeline en un servidor

> **Actualización 2026-09-05:** la familia recomendada para la nueva publicación es
> `egif-2d-48-v1` (`forecast_risk_egif_48_t1/t2/t3.joblib`). Las referencias a
> `forecast_risk_egif_t*.joblib` en las secciones de rollback describen la familia histórica de
> 50 variables y no deben mezclarse con la de 48.

## Anexo: publicación de la familia operativa de 48 variables

Entrena los tres artefactos fuera del servidor de inferencia (o en una máquina con suficiente
RAM) y sincroniza el conjunto completo de archivos `.joblib` y `.json` de una sola familia. Para
la familia nueva:

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/processed/tabular/egif \
  --output-dir data/models \
  --feature-contract-version egif-2d-48-v1 \
  --train-years 2016-2020 \
  --calibration-year 2021 \
  --validation-year 2022 \
  --test-years 2023
```

Antes de copiar, deben existir exactamente estos tres artefactos y sus metadatos:

```text
forecast_risk_egif_48_t1.joblib / .json
forecast_risk_egif_48_t2.joblib / .json
forecast_risk_egif_48_t3.joblib / .json
```

En el servidor se puede seleccionar explícitamente la familia con:

```text
FORECAST_MODEL_FAMILY=egif_48
```

Para una transición conservadora, `SHADOW_50_MODEL=true` ejecuta también los artefactos
`forecast_risk_egif_t*.joblib` de 50 variables y registra la diferencia de ranking sin sustituir
el mapa principal. La promoción debe hacerse solo después de comparar el manifest, las métricas
y el comportamiento con forecast fresco, fallback y stale.

El health check reconoce los contratos `egif-2d-48-v1` y `egif-2d-v1`, comprueba respectivamente
48 y 50 columnas y rechaza un manifest cuyos tres modelos no compartan familia y contrato.

## 1. Objetivo

Este documento explica cómo desplegar el sistema de predicción de peligro de
incendio en un servidor Linux para que actualice el estado meteorológico,
descargue el proveedor meteorológico configurado, genere T+1/T+2/T+3,
publique el resultado y sirva el dashboard Streamlit con controles de calidad,
recuperación y seguridad.

El diseño recomendado para el MVP utiliza dos procesos independientes:

~~~text
systemd timer ──> inferencia diaria ──> Parquet, JSON, manifest y caché del dashboard
nginx/TLS ──> Streamlit ──> lectura del último artefacto publicado
~~~

El dashboard no debe ejecutar inferencia ni enriquecer miles de filas al abrirse.
La inferencia es un proceso batch separado y, tras publicar el resultado, ejecuta
`scripts/prepare_dashboard_cache.py`. Streamlit solo lee el último resultado
operativo ya preparado; si el caché falta o no está actualizado, conserva un
fallback seguro que procesa el Parquet original.

El procedimiento reproducible de actualización de código y transferencia de
artefactos pesados está en
docs/deployment/despliegue_codigo_y_datos.md. Esta guía de servidor describe la
arquitectura y la operación de los servicios; el documento enlazado describe
Deploy Keys, releases atómicas y rsync.

La automatización meteorológica versionada se describe en
`docs/deployment/automatizacion_meteorologica.md`. En una release actual se
deben instalar las plantillas de `deploy/systemd/` con
`scripts/install_systemd_units.sh --enable --start`; las secciones posteriores
que muestran unidades creadas a mano se conservan como referencia para
servidores antiguos y no deben duplicarse sobre una instalación nueva.

## 2. Estado actual y alcance

| Capacidad | Estado |
|---|---|
| Cliente MeteoGalicia MeteoSIX | Implementado |
| Cliente AEMET OpenData alternativo | Implementado para pruebas/contingencia |
| Reintentos de descarga | Implementado |
| Archivado JSON y Parquet | Implementado |
| Estado meteorológico reciente | Implementado como contrato CSV/Parquet y backfill AEMET |
| Validación de horas y variables | Implementado |
| Fallback explícito stale | Implementado |
| Lock de ejecución | Implementado |
| Escritura atómica | Implementado |
| Manifiesto y SHA-256 | Implementado |
| Modelos serializados T+1/T+2/T+3 | Implementado |
| Health check de output | Implementado |
| Dashboard con fecha dinámica | Implementado |
| Geometría real de la rejilla en el dashboard | Implementado con fallback aproximado |
| TreeSHAP local | Implementado cuando shap está instalado |
| Backfill diario validado AEMET | Implementado; retraso aproximado de cuatro días |
| Colector AEMET en tiempo casi real | Implementado; ejecuta una captura cada 6 h y acumula la ventana móvil |
| Backtest histórico con vintages de forecast | Pendiente |
| Corrección forecast-observación | Pendiente |
| Autenticación de usuarios del dashboard | Responsabilidad del servidor/proxy |
| Deploy Key GitHub y releases atómicas | Implementado mediante scripts de despliegue |
| Sincronización de datasets/modelos con rsync | Implementado con dry-run y sin borrado por defecto |

La guía incluye una ingesta AEMET para bootstrap, reconciliación y refresco del
histórico validado, además de un colector de observaciones actuales. Como la
climatología diaria llega con retraso y las observaciones actuales tienen una
ventana móvil, ambos procesos cumplen funciones distintas: el backfill corrige
los días publicados y el colector debe permanecer activo para cerrar D-1.

## 3. Arquitectura recomendada

### 3.1 Componentes

| Componente | Función | Ejecución |
|---|---|---|
| Repositorio de aplicación | Código Python y configuración | Release versionada |
| Entorno Conda | Dependencias geoespaciales y ML | Persistente |
| Directorio de datos | Grid, estado, forecasts, outputs y modelos | Disco persistente |
| Colector horario | Acumula observaciones actuales y cierra días con cobertura | systemd service |
| Timer de reconciliación | Incorpora la climatología diaria AEMET ya publicada | systemd timer |
| Timer de inferencia | Ejecuta una inferencia provisional a las 05:00 y puede refrescar a las 10:00 | systemd timer |
| Servicio Streamlit | Visualización | systemd service |
| Nginx | TLS, proxy, límites y autenticación opcional | systemd service |
| Health check | Verificación de frescura y checksum | Timer o monitor |
| Backup | Copia de modelos, estado y metadatos | Timer o proveedor externo |

El proceso de inferencia y el dashboard no deben compartir un proceso Python. Si
Streamlit se reinicia, no debe relanzar una descarga ni duplicar una ejecución.

En el entorno del servidor se debe establecer `PIPELINE_ENVIRONMENT=production`
y mantener `LOCAL_SIMULATION_MODE=false`. Si alguien intenta activar la
simulación local en ese entorno, la inferencia debe fallar antes de publicar un
resultado marcado como operativo.

### 3.2 Servidor mínimo

| Recurso | Mínimo razonable | Recomendado |
|---|---:|---:|
| CPU | 4 vCPU | 8 vCPU |
| RAM | 8 GB | 16–32 GB |
| Disco | 80 GB SSD | 150 GB SSD |
| Sistema | Ubuntu 22.04/24.04 LTS | Ubuntu LTS actualizado |
| Red | salida HTTPS | salida HTTPS y entrada solo por 443 |
| Hora | NTP activo | NTP activo y Europe/Madrid |

La inferencia trabaja con una rejilla de unas 30.000 celdas, dependencias
geoespaciales y archivos Parquet. El tamaño final debe medirse con datos reales.

### 3.3 Zonas de red

El servidor necesita salida HTTPS hacia MeteoGalicia o AEMET según
`FORECAST_PROVIDER`, la fuente de observaciones y el repositorio o registro de
artefactos si se actualiza desde allí.

La entrada pública debería limitarse a:

~~~text
443/tcp  Nginx
22/tcp  SSH restringido por IP, VPN o bastion
~~~

Streamlit debe escuchar únicamente en localhost. No se debe publicar directamente
el puerto 8501 en Internet.

## 4. Preparación inicial

Los comandos siguientes son un ejemplo para Ubuntu y deben adaptarse a la
política de seguridad de la organización.

### 4.1 Usuario de servicio

~~~bash
sudo useradd --system --create-home --home-dir /srv/fire-risk \
  --shell /usr/sbin/nologin fire-risk
sudo mkdir -p /srv/fire-risk/app
sudo mkdir -p /srv/fire-risk/data
sudo chown -R fire-risk:fire-risk /srv/fire-risk
~~~

El usuario de servicio debe escribir en data/raw y data/processed, pero no tener
permisos de administración.

### 4.2 Dependencias del sistema

~~~bash
sudo apt-get update
sudo apt-get install -y git curl nginx ca-certificates build-essential
sudo apt-get install -y libgeos-dev libproj-dev proj-data proj-bin libgdal-dev gdal-bin
~~~

Las librerías geoespaciales pueden variar según Ubuntu. Se deben evitar mezclas
innecesarias entre GDAL del sistema y GDAL de Conda.

### 4.3 Miniconda

Instalar Miniconda o Mambaforge bajo una ruta administrada, por ejemplo:

~~~bash
sudo mkdir -p /opt/miniconda3
sudo chown fire-risk:fire-risk /opt/miniconda3
~~~

Después verificar:

~~~bash
/opt/miniconda3/bin/conda --version
~~~

## 5. Instalar la aplicación

### 5.1 Obtener el código

El código debe desplegarse desde un tag o commit revisado, no desde cambios
locales sin registrar. La opción recomendada es utilizar la Deploy Key de sólo
lectura creada por el bootstrap y el script de releases atómicas. Consultar la
guía completa en docs/deployment/despliegue_codigo_y_datos.md.

~~~bash
sudo -u fire-risk /srv/fire-risk/bin/deploy_code_server.sh \
  --ref TAG_O_COMMIT_VERIFICADO --run-tests
~~~

El script descarga una release nueva, valida la sintaxis y los tests solicitados,
enlaza el almacenamiento persistente y sólo después cambia
/srv/fire-risk/app. No se debe ejecutar git pull automáticamente desde el timer
de inferencia.

### 5.2 Crear el entorno Conda

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk /opt/miniconda3/bin/conda env create \
  --file environment.yml \
  --name incendios-forestales
~~~

Si el entorno ya existe:

~~~bash
sudo -u fire-risk /opt/miniconda3/bin/conda env update \
  --file environment.yml \
  --name incendios-forestales
~~~

Verificar dependencias críticas:

~~~bash
sudo -u fire-risk /opt/miniconda3/envs/incendios-forestales/bin/python \
  -c "import pandas, pyarrow, geopandas, lightgbm, streamlit, shap; print('OK')"
~~~

Las versiones exactas de Python, pandas, pyarrow, geopandas y LightGBM deben
registrarse con el release. Cambiar pyarrow o geopandas puede afectar la
lectura de la rejilla.

### 5.3 Instalar el proyecto

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk /opt/miniconda3/envs/incendios-forestales/bin/pip install --no-deps -e .
~~~

El uso de no-deps evita que pip modifique silenciosamente las dependencias
resueltas por Conda.

## 6. Estructura persistente de datos

El código espera rutas relativas a data cuando se ejecuta desde la raíz del
repositorio. Para que una actualización del código no borre los datos, se puede
mantener el almacenamiento fuera del checkout y crear un enlace:

~~~bash
sudo mkdir -p /srv/fire-risk/data/raw
sudo mkdir -p /srv/fire-risk/data/processed
sudo mkdir -p /srv/fire-risk/data/models
sudo mkdir -p /srv/fire-risk/data/processed/state
sudo mkdir -p /srv/fire-risk/data/processed/grid
sudo chown -R fire-risk:fire-risk /srv/fire-risk/data
cd /srv/fire-risk/app
sudo -u fire-risk ln -s /srv/fire-risk/data data
~~~

La estructura esperada es:

~~~text
/srv/fire-risk/data/
├── raw/meteogalicia/raw/
├── external/egif/
├── processed/
│   ├── grid/
│   ├── state/
│   ├── predicciones_operativas.parquet
│   ├── predicciones_operativas.json
│   └── predicciones_operativas.manifest.json
└── models/
    ├── forecast_risk_egif_48_t1.joblib
    ├── forecast_risk_egif_48_t1.json
    ├── forecast_risk_egif_48_t2.joblib
    ├── forecast_risk_egif_48_t2.json
    ├── forecast_risk_egif_48_t3.joblib
    └── forecast_risk_egif_48_t3.json
~~~

Los archivos de datos y modelos no se deben almacenar en Git.

## 7. Configuración y secretos

### 7.1 Archivo de entorno

Crear el archivo fuera del control de versiones. En el layout con releases se
mantiene en config para que sobreviva a las actualizaciones del código:

~~~bash
sudo install -o fire-risk -g fire-risk -m 600 /dev/null /srv/fire-risk/config/.env
sudo -u fire-risk nano /srv/fire-risk/config/.env
~~~

Contenido mínimo:

~~~text
# auto selecciona MeteoGalicia si tiene clave; en caso contrario AEMET.
FORECAST_PROVIDER=auto
FORECAST_AUTO_AEMET_FALLBACK=true
METEOGALICIA_API_KEY=clave_real
METEOGALICIA_BASE_URL=https://servizos.meteogalicia.gal/apiv5
METEOGALICIA_MODEL=WRF
# Se intenta WRF 1 km y se usa WRF 04 km si la primera malla falla.
METEOGALICIA_GRIDS=1km,04km
METEOGALICIA_QUERY_RESOLUTION_KM=4.0
METEOGALICIA_AUTO_ADJUST_POSITION=true
METEOGALICIA_TIMEOUT_SECONDS=60
METEOGALICIA_MAX_RETRIES=3
FORECAST_MAX_SOURCE_DISTANCE_KM=10
AEMET_API_KEY=clave_aemet_si_se_usa_como_alternativa
AEMET_BASE_URL=https://opendata.aemet.es/opendata/api
AEMET_MUNICIPALITIES_FILE=/srv/fire-risk/data/config/aemet_galicia_municipalities.txt
AEMET_MUNICIPALITIES=15030:43.3623:-8.4115,27028:43.0097:-7.5568,32054:42.3367:-7.8639,36038:42.4310:-8.6440
AEMET_USE_HOURLY_OVERLAY=true
# Mantener vacío en producción; 0 solo para pruebas técnicas si falta mm de lluvia.
AEMET_MISSING_PRECIPITATION_FALLBACK=
AEMET_MAX_SOURCE_DISTANCE_KM=80
INFERENCE_HISTORY_DAYS=30
WEATHER_STATE_PATH=/srv/fire-risk/data/processed/state/weather_daily_state.parquet
WEATHER_OBSERVATIONS_PATH=/srv/fire-risk/data/processed/observations/weather_daily_latest.parquet
AEMET_OBSERVATIONS_RAW_DIR=/srv/fire-risk/data/raw/aemet/observations
AEMET_OBSERVATION_IDW_NEIGHBORS=4
AEMET_HOURLY_OBSERVATIONS_PATH=/srv/fire-risk/data/processed/observations/aemet_hourly_observations.parquet
AEMET_CURRENT_MIN_COVERAGE_HOURS=20
AEMET_HOURLY_RETENTION_HOURS=72
AEMET_INGEST_LOCK_PATH=/srv/fire-risk/data/processed/.aemet_ingest.lock
INFERENCE_LOCK_PATH=/srv/fire-risk/data/processed/.daily_inference.lock
INFERENCE_MANIFEST_PATH=/srv/fire-risk/data/processed/predicciones_operativas.manifest.json
PREDICTIONS_OUTPUT_PATH=/srv/fire-risk/data/processed/predicciones_operativas.parquet
PREDICTIONS_MANIFEST_PATH=/srv/fire-risk/data/processed/predicciones_operativas.manifest.json
PREDICTIONS_DASHBOARD_CACHE_PATH=/srv/fire-risk/data/processed/predicciones_operativas.dashboard.parquet
GRID_PATH=/srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet
MODEL_DIR=/srv/fire-risk/data/models
FORECAST_MODEL_FAMILY=egif_48
EGIF_DATASET_DIR=/srv/fire-risk/data/external/egif
CANONICAL_GRID_CELLS=29601
SHADOW_50_MODEL=true
SHADOW_LEGACY_MODEL=false
~~~

El archivo real no debe aparecer en logs, backups públicos, tickets ni
capturas. En una organización con gestor de secretos, se debe sustituir .env
por una credencial inyectada en tiempo de ejecución.

Para probar actualmente con la clave AEMET, usa `FORECAST_PROVIDER=aemet` y
deja `METEOGALICIA_API_KEY` vacía. Cuando llegue la clave de MeteoGalicia,
puedes volver a `FORECAST_PROVIDER=auto` o fijar `FORECAST_PROVIDER=meteogalicia`.
La modalidad AEMET explícita queda marcada como `fresh_aemet`; si entra por la
cadena automática de contingencia queda marcada como
`fresh_aemet_degraded`. En ambos casos no debe confundirse con el producto WRF
de resolución 1 km.

### 7.2 Permisos

~~~bash
sudo chown fire-risk:fire-risk /srv/fire-risk/config/.env
sudo chmod 600 /srv/fire-risk/config/.env
sudo chmod -R u+rwX /srv/fire-risk/data
~~~

Las claves de la fuente observacional deben mantenerse separadas de la clave de
MeteoGalicia y tener el mínimo permiso necesario.

## 8. Cargar artefactos iniciales

Antes de activar los timers se necesitan rejilla, modelos y estado meteorológico.

### 8.0 Backfill inicial y actualización de observaciones

Con una clave AEMET se puede recuperar el rango histórico diario necesario para
el primer estado sin esperar 30 días:

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/ingest_aemet_weather_state.py \
  --grid /srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet \
  --observations /srv/fire-risk/data/processed/observations/weather_daily_latest.parquet \
  --state /srv/fire-risk/data/processed/state/weather_daily_state.parquet \
  --raw-dir /srv/fire-risk/data/raw/aemet/observations
~~~

El proceso consulta la climatología diaria AEMET para los últimos 30 días
publicados, por defecto hasta `hoy - 4 días`. Como AEMET limita cada consulta a
15 días, el cliente realiza bloques consecutivos, filtra estaciones dentro de
Galicia y realiza una interpolación IDW de cuatro vecinos. En cada ejecución
conserva el JSON bruto de cada bloque y actualiza el estado con escritura
atómica. Si falta alguna fecha, el proceso falla y no publica un estado
incompleto. Si la fuente cambia a MeteoGalicia, el forecast MeteoSIX no aporta
por sí mismo el histórico pasado: debe mantenerse una ingesta de observaciones
independiente o usar un análisis histórico explícito.

### 8.1 Rejilla

Para el modelo EGIF 2D no se debe reutilizar la rejilla legacy de 30.697
celdas. A partir del NetCDF canónico se genera la rejilla con geometría y
variables estáticas:

~~~bash
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/prepare_operational_grid.py \
  --cube /srv/fire-risk/data/processed/datacube/galicia_1km.nc \
  --output /srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet
~~~

Después se cambia `GRID_PATH` a ese fichero y se comprueba que contiene 29.601
filas y todas las columnas estáticas del contrato `egif-2d-v1`.

Como fallback legacy puede conservarse la rejilla antigua en:

~~~text
/srv/fire-risk/data/processed/grid/galicia_grid_1km_2018.parquet
~~~

Verificarla:

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  -c "import geopandas as gpd; p=gpd.read_parquet('data/processed/grid/galicia_grid_1km_egif.parquet'); print(p.shape, p.columns.tolist())"
~~~

Debe contener cell_id, lat_centroid y lon_centroid. La versión geoespacial debe
registrarse junto al release.

### 8.2 Modelos

Los modelos EGIF 2D deben haberse entrenado offline y pasar validación temporal.
El comando recomendado, ejecutado en una máquina con el datacubo, es:

~~~bash
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/train_egif_operational.py \
  --dataset-dir /srv/fire-risk/data/external/egif \
  --output-dir /srv/fire-risk/data/models
~~~

El entrenamiento recorre por lotes y no debe ejecutarse durante la inferencia.
Como alternativa se copian los tres artefactos y sus JSON:

~~~bash
sudo -u fire-risk cp forecast_risk_egif_48_t1.joblib /srv/fire-risk/data/models/
sudo -u fire-risk cp forecast_risk_egif_48_t1.json /srv/fire-risk/data/models/
sudo -u fire-risk cp forecast_risk_egif_48_t2.joblib /srv/fire-risk/data/models/
sudo -u fire-risk cp forecast_risk_egif_48_t2.json /srv/fire-risk/data/models/
sudo -u fire-risk cp forecast_risk_egif_48_t3.joblib /srv/fire-risk/data/models/
sudo -u fire-risk cp forecast_risk_egif_48_t3.json /srv/fire-risk/data/models/
~~~

No se debe mezclar T+1, T+2 y T+3 de releases incompatibles. La inferencia y el
health check validan horizonte, checksums y esquema de features. El antiguo `forecast_risk_t{h}.joblib`
se conserva únicamente como rollback o shadow.

### 8.3 Estado meteorológico

En el despliegue AEMET, el timer de reconciliación ejecuta la ingesta completa
para descargar el histórico diario ya publicado, interpolarlo a la rejilla y
actualizar el estado en una sola operación:

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/ingest_aemet_weather_state.py \
  --grid /srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet \
  --observations /srv/fire-risk/data/processed/observations/weather_daily_latest.parquet \
  --state /srv/fire-risk/data/processed/state/weather_daily_state.parquet \
  --raw-dir /srv/fire-risk/data/raw/aemet/observations
~~~

El estado debe tener 30 días completos por celda antes de activar inferencia.
No se debe crear un estado artificial con ceros si todavía no existe una fuente
observacional operativa. Para cerrar D-1 en tiempo casi real se debe activar
además el colector horario, que mantiene la ventana móvil de AEMET:

~~~bash
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/ingest_aemet_current_observations.py --loop --interval-hours 6
~~~

El comando `update_weather_state.py` sigue disponible para un productor externo
que ya haya generado el Parquet diario, por ejemplo una futura ingesta de
observaciones de MeteoGalicia.

## 9. Pruebas antes de producción

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  -m pytest -q

sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/run_daily_inference.py --help

sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/check_operational_run.py --help
~~~

Antes de llamar a la API real, realizar una prueba sintética de inferencia
inyectando forecast, history, grid y modelos de prueba. Debe comprobar que se
producen tres horizontes y que se crean Parquet, JSON y manifest.

## 10. Ejecución manual controlada

La primera ejecución real debe ser manual:

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/run_daily_inference.py \
  --state /srv/fire-risk/data/processed/state/weather_daily_state.parquet \
  --lock /srv/fire-risk/data/processed/.daily_inference.lock
~~~

Comprobar el resultado:

~~~bash
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/check_operational_run.py
~~~

Revisar forecast_quality, forecast_run_at, forecast_downloaded_at,
forecast_age_hours, cobertura, número de celdas, checksums y ausencia de un
lock abandonado. Si el resultado es stale, no es una ejecución normal aunque el
Parquet exista.
## 11. Programar la actualización del estado

La actualización del estado debe ejecutarse después de que la fuente de
observaciones haya cerrado D-1 y antes de la inferencia. El horario de ejemplo
es 04:20, pero depende del proveedor.

Crear /etc/systemd/system/fire-risk-state.service:

~~~ini
[Unit]
Description=Actualizar estado meteorologico diario de fire-risk
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=fire-risk
Group=fire-risk
WorkingDirectory=/srv/fire-risk/app
EnvironmentFile=/srv/fire-risk/app/.env
ExecStart=/opt/miniconda3/envs/incendios-forestales/bin/python scripts/ingest_aemet_weather_state.py --grid /srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet --observations /srv/fire-risk/data/processed/observations/weather_daily_latest.parquet --state /srv/fire-risk/data/processed/state/weather_daily_state.parquet --raw-dir /srv/fire-risk/data/raw/aemet/observations
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ReadWritePaths=/srv/fire-risk/data
~~~

Crear /etc/systemd/system/fire-risk-state.timer:

~~~ini
[Unit]
Description=Timer de actualizacion de estado meteorologico

[Timer]
OnCalendar=*-*-* 04:20:00 Europe/Madrid
Persistent=true
RandomizedDelaySec=2m
Unit=fire-risk-state.service

[Install]
WantedBy=timers.target
~~~

Activar:

~~~bash
sudo systemctl daemon-reload
sudo systemctl enable --now fire-risk-state.timer
sudo systemctl list-timers fire-risk-state.timer
~~~

### 11.1 Mantener el colector horario AEMET activo

Crear `/etc/systemd/system/fire-risk-aemet-current.service`:

~~~ini
[Unit]
Description=Acumulador de observaciones horarias AEMET
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=fire-risk
Group=fire-risk
WorkingDirectory=/srv/fire-risk/app
EnvironmentFile=/srv/fire-risk/app/.env
ExecStart=/opt/miniconda3/envs/incendios-forestales/bin/python scripts/ingest_aemet_current_observations.py --loop --interval-hours 6
Restart=always
RestartSec=60
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ReadWritePaths=/srv/fire-risk/data
~~

Activar el servicio:

~~~bash
sudo systemctl daemon-reload
sudo systemctl enable --now fire-risk-aemet-current.service
sudo systemctl status fire-risk-aemet-current.service
~~~

El servicio no reemplaza al timer de reconciliación: el primero acumula la
ventana móvil de AEMET para cerrar D-1 y el segundo corrige los días cuando la
climatología validada ya está publicada.

Si se usa el servicio AEMET anterior, la propia ingesta publica primero el
Parquet temporalmente y actualiza el estado de forma atómica. Si se sustituye
por un productor externo, ese productor debe publicar primero un archivo
temporal y moverlo atómicamente a weather_daily_latest.parquet. El servicio no
debe considerar correcta una actualización que no cubra todas las celdas.

## 12. Programar la inferencia diaria

Crear /etc/systemd/system/fire-risk-inference.service:

~~~ini
[Unit]
Description=Inferencia diaria de riesgo de incendios
After=network-online.target fire-risk-state.service
Wants=network-online.target

[Service]
Type=oneshot
User=fire-risk
Group=fire-risk
WorkingDirectory=/srv/fire-risk/app
EnvironmentFile=/srv/fire-risk/app/.env
ExecStart=/opt/miniconda3/envs/incendios-forestales/bin/python scripts/run_daily_inference.py
# Prepara las columnas auxiliares que consume Streamlit después de publicar el
# resultado. Un fallo del enriquecimiento no invalida el output operativo.
ExecStartPost=-/opt/miniconda3/envs/incendios-forestales/bin/python scripts/prepare_dashboard_cache.py
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ReadWritePaths=/srv/fire-risk/data
TimeoutStartSec=45min
~~~

Crear /etc/systemd/system/fire-risk-inference.timer:

~~~ini
[Unit]
Description=Timer de inferencia diaria de riesgo

[Timer]
OnCalendar=*-*-* 05:00:00 Europe/Madrid
Persistent=true
RandomizedDelaySec=5m
Unit=fire-risk-inference.service

[Install]
WantedBy=timers.target
~~~

Activar:

~~~bash
sudo systemctl daemon-reload
sudo systemctl enable --now fire-risk-inference.timer
sudo systemctl list-timers fire-risk-inference.timer
~~~

El timer usa Europe/Madrid para respetar la operación local. Los timestamps
internos siguen normalizándose a UTC. Persistent=true recupera ejecuciones
perdidas tras un reinicio y el lock evita concurrencia.

La ejecución de las 05:00 puede encontrar todavía incompleto el WRF 1 km de las
00:00 UTC. En ese caso el pipeline intenta WRF 04 km y publica el resultado como
`fresh_fallback`. Para que el mapa principal se actualice con la malla de 1 km
cuando ya esté disponible, se recomienda añadir un segundo timer que invoque el
mismo servicio a las 10:00. El lock y las escrituras atómicas hacen segura esta
segunda ejecución; el dashboard mostrará siempre la última publicación válida.

Crear `/etc/systemd/system/fire-risk-inference-refresh.timer`:

~~~ini
[Unit]
Description=Refresco de inferencia con forecast WRF actualizado

[Timer]
OnCalendar=*-*-* 10:00:00 Europe/Madrid
Persistent=true
RandomizedDelaySec=5m
Unit=fire-risk-inference.service

[Install]
WantedBy=timers.target
~~~

Activarlo junto con el timer principal:

~~~bash
sudo systemctl daemon-reload
sudo systemctl enable --now fire-risk-inference-refresh.timer
sudo systemctl list-timers fire-risk-inference.timer fire-risk-inference-refresh.timer
~~~

La segunda ejecución no debe publicar a mitad de escritura: el pipeline utiliza
lock y publicación atómica. Si el WRF 1 km sigue sin estar disponible, puede
conservarse otro `fresh_fallback`; esa condición queda visible en el health check
y en el dashboard.

### 12.1 Logs

~~~bash
sudo journalctl -u fire-risk-state.service -n 100 --no-pager
sudo journalctl -u fire-risk-inference.service -n 200 --no-pager
sudo systemctl status fire-risk-inference.timer
~~~

### 12.2 Política stale

Para bloquear cualquier fallback:

~~~bash
scripts/run_daily_inference.py --no-stale
~~~

Esta opción debe incorporarse al ExecStart si la operación no quiere publicar
mapas antiguos. Si se acepta stale, el health check se debe ejecutar sin
--allow-stale para generar alerta.

## 13. Health check y alertas

Ejecutar después de la inferencia:

~~~bash
cd /srv/fire-risk/app
sudo -u fire-risk env PYTHONPATH=. /opt/miniconda3/envs/incendios-forestales/bin/python \
  scripts/check_operational_run.py
~~~

El check falla si no existe el output o manifest, no coincide el checksum,
faltan horizontes, el forecast es stale no autorizado, la antigüedad supera el
umbral, la cobertura es insuficiente o las probabilidades son inválidas.

Para impedir publicar una ejecución AEMET o un fallback WRF como producto
normal, añade `--fail-on-degraded` al health check. Esta opción también es útil
en el servidor de producción mientras se valida la fuente meteorológica
definitiva.

Se recomienda añadir un servicio de verificación 10–15 minutos después de la
inferencia. El primer nivel de alerta puede ser un log monitorizado por
systemd, Prometheus, Uptime Kuma o el sistema de observabilidad corporativo.

Una alerta debe incluir run_id, hora, forecast_quality, forecast_age_hours,
cobertura, estado meteorológico, mensaje de error y ruta del manifest. Nunca se
debe enviar la clave API.

## 14. Servicio Streamlit

Crear /etc/systemd/system/fire-risk-dashboard.service:

~~~ini
[Unit]
Description=Dashboard Streamlit de fire-risk
After=network.target

[Service]
Type=simple
User=fire-risk
Group=fire-risk
WorkingDirectory=/srv/fire-risk/app
EnvironmentFile=/srv/fire-risk/app/.env
# Si el resultado existe, se prepara antes de aceptar la primera sesión web.
# El prefijo '-' permite arrancar aunque todavía no exista un output inicial.
ExecStartPre=-/opt/miniconda3/envs/incendios-forestales/bin/python scripts/prepare_dashboard_cache.py
ExecStart=/opt/miniconda3/envs/incendios-forestales/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
Restart=on-failure
RestartSec=10
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ReadOnlyPaths=/srv/fire-risk/app
ReadWritePaths=/srv/fire-risk/data

[Install]
WantedBy=multi-user.target
~~~

Activar:

~~~bash
sudo systemctl daemon-reload
sudo systemctl enable --now fire-risk-dashboard.service
sudo systemctl status fire-risk-dashboard.service
curl --fail http://127.0.0.1:8501/_stcore/health
~~~

El dashboard debe leer modelos, rejilla y resultados, pero no escribirlos. Si
necesita cachés, se debe definir una ruta concreta con permisos limitados. El
Parquet `predicciones_operativas.dashboard.parquet` es un artefacto derivado de
lectura, no sustituye al output operativo ni al manifest.

## 15. Nginx y TLS

Nginx debe ser el único servicio accesible desde Internet. Ejemplo:

~~~nginx
server {
    listen 80;
    server_name riesgo.example.org;

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name riesgo.example.org;

    ssl_certificate /etc/letsencrypt/live/riesgo.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/riesgo.example.org/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
~~~

Activar:

~~~bash
sudo nginx -t
sudo systemctl reload nginx
~~~

Antes de publicar, añadir autenticación corporativa, VPN, allowlist de IP u
OIDC delante de Nginx. Streamlit no debe quedar anónimo por defecto.

## 16. Docker como alternativa

Docker puede facilitar reproducibilidad, pero las dependencias geoespaciales
pueden generar imágenes grandes. Si se elige Docker:

1. fijar versiones de Python y dependencias;
2. montar data como volumen persistente;
3. inyectar secretos con Docker secrets o un gestor externo;
4. separar inferencia y dashboard en contenedores;
5. usar scheduler externo o un contenedor dedicado;
6. conservar lock y artefactos en el volumen;
7. no guardar modelos ni datos en capas efímeras.

No se debe ejecutar un cron dentro del contenedor del dashboard. Para este
repositorio, Conda + systemd suele ser más sencillo en el primer servidor.
## 17. Backups

La política mínima debe respaldar:

~~~text
data/processed/grid/
data/processed/state/
data/models/
data/processed/predicciones_operativas.*
data/raw/meteogalicia/raw/
.env o el secreto equivalente mediante el gestor seguro
~~~

Los forecasts brutos pueden tener una retención distinta, pero son necesarios
para construir el backtest forecast-observación.

Cada backup debe probarse con una restauración parcial. Un backup que nunca se
ha restaurado no es una garantía.

## 18. Actualizar una versión

Procedimiento recomendado:

1. comprobar que no hay inferencia activa;
2. guardar el manifiesto actual;
3. desplegar un tag o commit nuevo;
4. ejecutar pruebas;
5. verificar dependencias y modelos;
6. reiniciar el dashboard;
7. ejecutar inferencia manual;
8. validar manifest;
9. reactivar timers.

~~~bash
sudo systemctl stop fire-risk-dashboard.service
sudo systemctl stop fire-risk-inference.timer

cd /srv/fire-risk/app
sudo -u fire-risk git fetch --tags
sudo -u fire-risk git checkout TAG_NUEVO

sudo -u fire-risk /opt/miniconda3/bin/conda env update \
  --file environment.yml --name incendios-forestales

sudo -u fire-risk env PYTHONPATH=. \
  /opt/miniconda3/envs/incendios-forestales/bin/python -m pytest -q

sudo systemctl start fire-risk-dashboard.service
sudo systemctl start fire-risk-inference.timer
~~~

No se debe cambiar el schema de features sin desplegar los tres modelos
compatibles. Si cambia FEATURE_SCHEMA_VERSION, los joblib anteriores deben
retirarse o quedar aislados.

## 19. Rollback y retención de runs

Un rollback debe poder restaurar código, dependencias, tres modelos,
configuración y último output válido.

La escritura atómica protege el output actual, pero antes de un servicio
crítico se recomienda guardar cada ejecución:

~~~text
data/processed/runs/<run_id>/
├── predicciones_operativas.parquet
├── predicciones_operativas.json
└── predicciones_operativas.manifest.json
~~~

La versión actual publica el último resultado y su manifest; la retención por
run es una mejora recomendada para rollback y auditoría.

## 20. Recuperación ante fallos

### No hay respuesta del proveedor

1. revisar conectividad HTTPS;
2. revisar clave y endpoint;
3. revisar límite o cambios de API;
4. revisar logs de reintentos;
5. comprobar forecast archivado completo;
6. decidir si se acepta stale;
7. si no se acepta, mantener el último output visible con alerta.

### Falta el estado meteorológico

1. comprobar que el productor de observaciones terminó;
2. comprobar permisos y ruta;
3. comprobar fechas y celdas;
4. corregir y repetir update_weather_state;
5. no editar el Parquet manualmente sin registrar la intervención.

### Existe un lock

1. comprobar si el servicio sigue activo;
2. leer PID y timestamp del lock;
3. revisar journalctl;
4. borrarlo solo si el proceso ya no existe;
5. volver a ejecutar y validar el manifest.

### Dashboard sin datos

1. revisar el servicio Streamlit;
2. comprobar que existe el output;
3. ejecutar health check;
4. revisar permisos;
5. comprobar Nginx, TLS y websocket.

## 21. Cron como alternativa mínima

Si no se puede usar systemd, cron es menos observable. Siempre se debe usar
ruta absoluta, usuario de servicio, logs y lock:

~~~cron
20 4 * * * cd /srv/fire-risk/app && /usr/bin/flock -n /srv/fire-risk/data/processed/.external-state.lock /opt/miniconda3/envs/incendios-forestales/bin/python scripts/ingest_aemet_weather_state.py --grid /srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet --observations /srv/fire-risk/data/processed/observations/weather_daily_latest.parquet --state /srv/fire-risk/data/processed/state/weather_daily_state.parquet --raw-dir /srv/fire-risk/data/raw/aemet/observations >> /var/log/fire-risk-state.log 2>&1
0 5 * * * cd /srv/fire-risk/app && /usr/bin/flock -n /srv/fire-risk/data/processed/.external-inference.lock /opt/miniconda3/envs/incendios-forestales/bin/python scripts/run_daily_inference.py >> /var/log/fire-risk-inference.log 2>&1
15 5 * * * cd /srv/fire-risk/app && /usr/bin/flock -n /srv/fire-risk/data/processed/.external-check.lock /opt/miniconda3/envs/incendios-forestales/bin/python scripts/check_operational_run.py >> /var/log/fire-risk-health.log 2>&1
~~~

El lock externo y el interno cumplen funciones distintas. Systemd timers son
preferibles por dependencias, estado y logs.

## 22. Checklist de aceptación

### Instalación

- [ ] usuario fire-risk creado;
- [ ] entorno Conda creado;
- [ ] dependencias geoespaciales importan;
- [ ] .env tiene permisos 600;
- [ ] salida HTTPS funciona;
- [ ] NTP está activo;
- [ ] data está en almacenamiento persistente.

### Datos y modelos

- [ ] rejilla leíble;
- [ ] tres modelos presentes;
- [ ] modelo operativo `egif-2d-48-v1` validado y familia histórica de 50 disponible para rollback;
- [ ] rejilla canónica con 29.601 celdas y estáticas completas;
- [ ] estado con 30 días por celda;
- [ ] forecasts raw y processed tienen retención;
- [ ] backups configurados.

### Pipeline

- [ ] tests pasan;
- [ ] inferencia manual produce Parquet;
- [ ] JSON y manifest se crean;
- [ ] checksum valida;
- [ ] stale se identifica;
- [ ] lock se libera;
- [ ] timer está activo;
- [ ] logs son consultables.

### Dashboard

- [ ] Streamlit escucha solo en 127.0.0.1;
- [ ] health endpoint responde;
- [ ] Nginx tiene TLS;
- [ ] autenticación o red privada está activa;
- [ ] websocket funciona;
- [ ] fecha, horizonte, edad y calidad son visibles;
- [ ] TreeSHAP no muestra heurísticas si no está disponible.

## 23. Mejoras que deberían implementarse

### P0: necesarias para operación fiable

1. Automatizar el productor real de observaciones diarias.
2. Guardar cada ejecución en data/processed/runs con run_id.
3. Añadir alertas externas ante fallo o stale.
4. Fijar versiones de dependencias y modelos por release.
5. Añadir autenticación delante del dashboard.
6. Configurar y probar backups.

### P1: necesarias para una herramienta operativa útil

1. Añadir filtros por provincia, municipio y área de gestión.
2. Añadir comparación T+1/T+2/T+3 y evolución entre ejecuciones.
3. Mostrar panel de calidad con cobertura, distancia, emisión, descarga y estado.
4. Mejorar rendimiento mediante GeoJSON simplificado, PyDeck o vector tiles.
5. Añadir ranking descargable de celdas y municipios.
6. Separar visualmente probabilidad calibrada, percentil y nivel táctico.
7. Mostrar versión del modelo y fecha de entrenamiento.

### P2: mejoras científicas

1. Backtest de vintages con forecasts históricos.
2. Error meteorológico por variable y horizonte.
3. Corrección forecast-observación validada.
4. Incertidumbre o ensemble de fuentes.
5. Detección de drift y recalibración periódica.
6. Evaluación por provincia, combustible, altitud y estación.

### P3: escala

1. Object storage versionado.
2. Orquestador como Airflow, Prefect o Dagster.
3. Workers separados del servidor web.
4. Catálogo de datos y registro de modelos.
5. Dashboard stateless y replicable.

## 24. Decisión recomendada

Para el MVP no es necesario introducir Kubernetes, Airflow ni una arquitectura
distribuida. La combinación recomendada es:

~~~text
Ubuntu LTS
- Conda con environment.yml
- usuario fire-risk
- data persistente fuera del código
- systemd timers
- systemd service para Streamlit
- Nginx con TLS y autenticación
- health check y alertas
- backup probado
~~~

La prioridad inmediata no es añadir más complejidad de machine learning, sino
cerrar el circuito operativo: observaciones recientes, monitorización externa,
retención por run, autenticación y visualización fiel de geometría y calidad.

## 25. Valor operativo

Un despliegue correcto permite que bomberos forestales y protección civil
consulten cada mañana un producto reproducible y fechado. La calidad visible
evita interpretar un mapa antiguo como previsión actual; la geometría real evita
errores de localización; el ranking facilita priorizar patrullas; y los
metadatos permiten justificar qué información sustentó cada alerta.

El servidor aporta continuidad, trazabilidad y recuperación. No convierte el
modelo en una verdad operativa: la validez depende de observaciones, forecast y
validación histórica por horizonte.
