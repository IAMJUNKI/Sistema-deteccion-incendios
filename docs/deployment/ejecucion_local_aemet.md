# Ejecución local del pipeline operativo con AEMET

Este documento describe la operación actual en un ordenador personal, sin
servidor y sin `cron`. La clave de AEMET permite ejecutar el forecast municipal
alternativo y capturar observaciones reales. Cuando se incorpore la clave de
MeteoGalicia, el forecast pasará a WRF 1 km/04 km, pero el mantenimiento del
histórico observacional seguirá siendo una tarea independiente.

## Qué resuelve AEMET y qué limitación tiene

AEMET tiene tres productos relevantes para este proyecto:

1. La predicción municipal diaria/horaria, usada como proveedor alternativo
   del forecast T+1/T+2/T+3. No es equivalente a la malla WRF de MeteoGalicia y
   queda marcada como `fresh_aemet`.
2. La climatología diaria validada. Sirve para bootstrap y reconciliación, pero
   se publica aproximadamente cuatro días después de la observación.
3. Las observaciones convencionales actuales, que contienen una ventana móvil
   reciente. El script local las acumula; una llamada aislada no puede
   reconstruir todo el día anterior.

La consecuencia operativa es importante: el estado de 30 días que necesita la
memoria de precipitación no se rellena con ceros ni con una fecha histórica
ficticia. Se construye con la climatología diaria de AEMET para los días ya
publicados y con observaciones horarias acumuladas para cerrar días recientes.

## Configuración inicial

En la raíz del repositorio:

```bash
cp .env.example .env
```

En `.env` se deben establecer al menos:

```dotenv
FORECAST_PROVIDER=aemet
AEMET_API_KEY=la_clave_real_de_aemet
```

Mantén vacía `METEOGALICIA_API_KEY` hasta disponer de ella. No pegues la clave
en comandos, capturas o documentación. Si una clave ha aparecido en un log,
revócala y genera otra.

Usa siempre el intérprete del entorno donde están instaladas las dependencias:

```bash
PYTHON_BIN=/opt/anaconda3/envs/incendios-forestales/bin/python
```

Si el entorno tiene otro nombre, sustituye esa ruta. Se puede comprobar con:

```bash
$PYTHON_BIN -c "import numpy, pandas, requests; print('runtime ok')"
```

## Primera carga: no hay que esperar 30 días

La climatología diaria validada inicializa los últimos 30 días publicados. El
cliente divide automáticamente el rango en bloques de como máximo 15 días, la
restricción que aplica AEMET.

```bash
PYTHONPATH=. $PYTHON_BIN scripts/ingest_aemet_weather_state.py
```

Por defecto el comando pide hasta `hoy - 4 días`. Puede cambiarse de forma
explícita, por ejemplo:

```bash
PYTHONPATH=. $PYTHON_BIN scripts/ingest_aemet_weather_state.py \
  --start-date 2026-07-20 \
  --end-date 2026-08-19
```

La ejecución genera o actualiza:

- `data/raw/aemet/observations/`: respuestas brutas y trazables;
- `data/processed/observations/weather_daily_latest.parquet`: observaciones
  diarias interpoladas a la rejilla;
- `data/processed/state/weather_daily_state.parquet`: estado compacto para
  inferencia.

## Captura de observaciones recientes

La captura horaria debe ejecutarse cada seis horas aproximadamente. El modo
recomendado en un ordenador personal es dejar una terminal abierta:

```bash
PYTHONPATH=. $PYTHON_BIN scripts/ingest_aemet_current_observations.py \
  --loop \
  --interval-hours 6
```

El proceso hace lo siguiente en cada ciclo:

1. Descarga la ventana móvil de AEMET.
2. Filtra estaciones dentro de Galicia.
3. Archiva la respuesta bruta.
4. Hace upsert por `(station_id, valid_time)` en el Parquet horario.
5. Mantiene 72 horas para tolerar un reinicio o una captura perdida.
6. Agrega temperatura máxima, humedad mínima, viento máximo y precipitación
   diaria en hora local `Europe/Madrid`.
7. Solo cierra una estación-día con al menos 20 horas distintas por defecto.
8. Interpola el resultado a la rejilla de 1 km y actualiza el estado de forma
   atómica.

El proceso se detiene con `Ctrl+C`. No hay que dejarlo ejecutándose como root.
Si prefieres no mantenerlo abierto, ejecuta una captura manual:

```bash
PYTHONPATH=. $PYTHON_BIN scripts/ingest_aemet_current_observations.py --once
```

Una ejecución manual aislada puede no cerrar ningún día; eso es correcto y se
indicará en pantalla. Para producción local se recomienda el modo `--loop`,
porque la ventana de observación de AEMET es móvil.

## Rutina diaria manual

La secuencia mínima antes de mostrar el mapa es:

```bash
# Terminal 1: debe estar funcionando durante el periodo de pruebas
PYTHONPATH=. $PYTHON_BIN scripts/ingest_aemet_current_observations.py \
  --loop --interval-hours 6

# Terminal 2, una vez al día, por ejemplo a las 05:00–06:00
PYTHONPATH=. $PYTHON_BIN scripts/ingest_aemet_weather_state.py
PYTHONPATH=. $PYTHON_BIN scripts/run_daily_inference.py --no-stale
PYTHONPATH=. $PYTHON_BIN scripts/check_operational_run.py

# Terminal 3, para visualizar el resultado
PYTHONPATH=. $PYTHON_BIN -m streamlit run app.py
```

La ingesta diaria validada no sustituye al colector horario: sirve para
reconciliar los últimos días cuando AEMET los publica y para corregir valores
provisionales. El colector horario es el que evita perder D-1 durante el
desfase de publicación.

## Simulación local para probar el dashboard

Si el ordenador no ha mantenido el colector horario durante la noche, se puede
hacer una demostración local usando el último día disponible del estado. Esta
opción no falsifica el forecast de MeteoGalicia: mantiene la hora real para
descargar T+1/T+2/T+3 y solo relaja, de forma explícita, la validación del
estado observacional.

En `.env` establece, por ejemplo:

```dotenv
PIPELINE_ENVIRONMENT=local
LOCAL_SIMULATION_MODE=true
LOCAL_SIMULATION_AS_OF_DATE=2026-08-26
```

Esto trata el `26/08` como el último día observado y simula que el siguiente
día habría sido la fecha de corte. El resultado se marca como
`pipeline_run_mode=local_state_simulation`, el manifest conserva la fecha
simulada y el dashboard muestra una advertencia. No se debe usar para evaluar
la calidad del modelo ni para movilizar medios.

El modo está bloqueado si `PIPELINE_ENVIRONMENT=production`. El health check
también rechaza por defecto una ejecución simulada; para una comprobación
local explícita se puede usar `--allow-local-simulation`.

## Qué hacer si se apaga el ordenador

Al reiniciar:

1. Ejecuta de nuevo el backfill diario. Es idempotente y conserva las
   respuestas brutas.
2. Arranca el colector con `--loop`.
3. No ejecutes inferencia hasta que el estado pase la validación de 30 días.

Si el ordenador estuvo apagado más de 12 horas, AEMET puede haber dejado de
ofrecer parte de las horas perdidas en la ventana móvil. En ese caso, espera a
que la climatología diaria validada publique esos días o ejecuta un backfill
con fechas que ya estén disponibles. El pipeline debe fallar explícitamente
si no puede reconstruir la memoria meteorológica.

## Significado de los estados

El dashboard y el manifest distinguen:

- `fresh_aemet`: forecast AEMET reciente, válido como contingencia técnica;
- `fresh`: forecast MeteoGalicia WRF 1 km;
- `fresh_fallback`: MeteoGalicia WRF 04 km porque 1 km no estaba disponible;
- `stale`: se reutilizó el último forecast archivado tras fallar la descarga.

`stale` no debe ocultarse en una decisión operativa. Antes de movilizar medios
se debe comprobar la antigüedad y la fuente mostradas en el dashboard.

## Mac sin cron: opción posterior

Para la fase de pruebas no hace falta automatizar nada: `--loop` es suficiente.
Cuando se necesite que el proceso arranque después de reiniciar macOS, la
opción nativa es un agente `launchd` de usuario que invoque este mismo comando.
Se recomienda configurarlo solo después de validar varios días de capturas
manuales, porque un servicio automático no corrige una clave revocada, una
dependencia rota o una red sin acceso.

## Comprobaciones rápidas

```bash
PYTHONPATH=. $PYTHON_BIN -m pytest -q
PYTHONPATH=. $PYTHON_BIN scripts/ingest_aemet_current_observations.py --help
PYTHONPATH=. $PYTHON_BIN scripts/run_daily_inference.py --help
PYTHONPATH=. $PYTHON_BIN scripts/check_operational_run.py --help
```

La documentación oficial de AEMET describe la actualización continua de las
observaciones horarias y el recurso de climatología diaria con su desfase de
publicación. La API usa además la doble llamada `estado/datos`; el cliente del
proyecto conserva ese comportamiento.
