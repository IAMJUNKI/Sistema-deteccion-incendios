# Simulación local del estado meteorológico AEMET

## 1. Propósito

Este procedimiento permite probar localmente la inferencia y el dashboard
cuando el ordenador no ha mantenido durante la noche el colector horario de
AEMET. En ese caso, el archivo de estado puede terminar varios días antes de
la fecha real y la validación operativa bloquearía la ejecución.

La simulación trata una fecha histórica concreta como el último día observado
disponible. Por ejemplo, si el estado llega hasta `2026-08-26`, se puede
simular que la ejecución operativa habría ocurrido el `2026-08-27` a las
05:00, con `2026-08-26` como D-1.

Esta opción existe únicamente para verificar el circuito técnico en un
ordenador personal. No es un backtest, no corrige la ausencia real de
observaciones recientes y no debe utilizarse para evaluar el modelo ni para
movilizar medios.

## 2. Qué simula y qué no simula

La simulación afecta solo a la validación del estado meteorológico reciente:

1. Lee `data/processed/state/weather_daily_state.parquet`.
2. Usa `LOCAL_SIMULATION_AS_OF_DATE` como último día observado.
3. Valida los 30 días anteriores a esa fecha.
4. Ignora cualquier fila posterior a la fecha simulada.
5. Etiqueta el resultado como `local_state_simulation`.

El reloj del forecast no se modifica. MeteoGalicia continúa consultándose en el
momento real de ejecución y devuelve el forecast actual para T+1/T+2/T+3. Por
eso esta ejecución combina un estado observacional antiguo con un forecast
actual y solo es válida como demostración técnica.

## 3. Requisitos previos

Antes de activar la simulación deben existir:

- el estado meteorológico Parquet;
- la rejilla de Galicia de 1 km;
- los tres modelos serializados en `data/models/`;
- una clave válida de MeteoGalicia en `.env`;
- dependencias instaladas en el entorno Conda del proyecto.

Para comprobar la última fecha del estado sin imprimir datos sensibles:

```bash
PYTHON_BIN=/opt/anaconda3/envs/incendios-forestales/bin/python

PYTHONPATH=. "$PYTHON_BIN" -c 'import pandas as pd; p="data/processed/state/weather_daily_state.parquet"; d=pd.read_parquet(p, columns=["fecha"]); d["fecha"]=pd.to_datetime(d["fecha"]); print(d["fecha"].min().date(), "->", d["fecha"].max().date())'
```

La fecha simulada debe ser igual o anterior a la última fecha realmente
disponible en el estado. Si el estado termina el `2026-08-26`, la configuración
recomendada para esta prueba es `LOCAL_SIMULATION_AS_OF_DATE=2026-08-26`.

## 4. Configuración en `.env`

Añadir o modificar estas variables en el `.env` local:

```dotenv
PIPELINE_ENVIRONMENT=local
LOCAL_SIMULATION_MODE=true
LOCAL_SIMULATION_AS_OF_DATE=2026-08-26
FORECAST_PROVIDER=meteogalicia
```

No se debe incluir la clave de MeteoGalicia en este documento ni en comandos,
capturas o repositorios. La variable `METEOGALICIA_API_KEY` debe permanecer
únicamente en el `.env`, que está excluido por `.gitignore`.

### Significado de las variables

| Variable | Valor | Función |
|---|---|---|
| `PIPELINE_ENVIRONMENT` | `local` | Declara que la ejecución es local |
| `LOCAL_SIMULATION_MODE` | `true` | Activa la validación simulada |
| `LOCAL_SIMULATION_AS_OF_DATE` | `YYYY-MM-DD` | Último día tratado como observado |
| `FORECAST_PROVIDER` | `meteogalicia` | Usa WRF de MeteoGalicia para el forecast |

El modo está permitido solo para `local`, `development`, `dev` o `test`.
Si se intenta activar con `PIPELINE_ENVIRONMENT=production`, el pipeline
falla antes de publicar resultados.

## 5. Ejecución paso a paso

### 5.1 Comprobar MeteoGalicia

Esta prueba no necesita el estado AEMET y consulta un solo lote de puntos:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/check_meteogalicia_forecast.py
```

Debe aparecer `MeteoSIX v5 OK`, con `calidad=fresh` y preferiblemente
`malla=1km`. Si WRF 1 km no supera la validación, puede aparecer
`fresh_fallback` al utilizar WRF 04 km.

### 5.2 Ejecutar la inferencia local

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/run_daily_inference.py --no-stale
```

El comando descarga el forecast completo, asigna los puntos meteorológicos a
la rejilla de 1 km, genera T+1/T+2/T+3 y escribe:

- `data/processed/predicciones_operativas.parquet`;
- `data/processed/predicciones_operativas.json`;
- `data/processed/predicciones_operativas.manifest.json`;
- los forecasts brutos en `data/raw/meteogalicia/`.

Si ya se ha descargado y validado un forecast completo, se puede evitar otra
consulta a MeteoSIX durante la prueba local indicando el Parquet horario:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/run_daily_inference.py \
  --no-stale \
  --forecast-file data/raw/meteogalicia/forecast_YYYYMMDDTHHMMSSZ_1km.parquet
```

El fichero se somete a las mismas comprobaciones de cobertura, timestamps y
distancia a la rejilla que un forecast recién descargado. Esta opción es útil
para reanudar una prueba cuyo proceso se interrumpió después de la descarga;
no sustituye la descarga diaria en producción.

La construcción vectorizada de las memorias meteorológicas tarda normalmente
unos segundos para las 30.697 celdas. Durante la ejecución se muestran las
fases `Estado meteorológico`, `Forecast`, `Features` y los tres modelos para
detectar dónde se encuentra el proceso.

La salida mantiene la fecha real de ejecución del forecast, pero el manifest
incluye:

```json
{
  "pipeline_run_mode": "local_state_simulation",
  "weather_state_validation_mode": "simulated_as_of",
  "weather_state_simulation_as_of_date": "2026-08-26"
}
```

### 5.3 Ejecutar el health check local

El health check rechaza las simulaciones por defecto para proteger el uso en
producción. En local hay que autorizarla explícitamente:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/check_operational_run.py \
  --allow-local-simulation
```

### 5.4 Abrir el dashboard

```bash
PYTHONPATH=. "$PYTHON_BIN" -m streamlit run app.py
```

El dashboard mostrará una advertencia indicando que el estado meteorológico se
ha simulado hasta la fecha configurada. El forecast puede seguir apareciendo
como `fresh`, porque la simulación del estado y la calidad del forecast son
controles independientes.

## 6. Errores frecuentes

### El estado sigue sin tener cobertura suficiente

Comprueba que `LOCAL_SIMULATION_AS_OF_DATE` coincide con una fecha cubierta por
el Parquet y que existen al menos 30 días completos anteriores. Si el estado
termina el `2026-08-26`, no se debe indicar el `2026-08-27` como fecha
simulada.

### El pipeline indica que la simulación solo puede ser local

Se ha dejado `PIPELINE_ENVIRONMENT=production`, o el modo está activado en un
entorno que no es de desarrollo. Para una prueba local usa:

```dotenv
PIPELINE_ENVIRONMENT=local
```

### El health check rechaza la ejecución

Es el comportamiento esperado si se ejecuta sin
`--allow-local-simulation`. Nunca se debe añadir esa opción a los timers o
servicios de producción.

### MeteoGalicia devuelve un error o un forecast stale

La simulación local no evita los controles del forecast. Comprueba la red, la
clave, la disponibilidad de WRF y el resultado de
`scripts/check_meteogalicia_forecast.py`. No se debe desactivar `--no-stale`
para ocultar un fallo durante esta prueba.

## 7. Volver al modo normal

Después de la demostración, editar `.env` y dejar:

```dotenv
PIPELINE_ENVIRONMENT=local
LOCAL_SIMULATION_MODE=false
LOCAL_SIMULATION_AS_OF_DATE=
```

En un servidor se debe establecer siempre:

```dotenv
PIPELINE_ENVIRONMENT=production
LOCAL_SIMULATION_MODE=false
LOCAL_SIMULATION_AS_OF_DATE=
```

Para operación real, el estado debe actualizarse con el backfill diario de
AEMET y el colector de observaciones actuales debe mantenerse activo cada seis
horas. La simulación no sustituye ese proceso.

## 8. Archivos relacionados

- `scripts/run_daily_inference.py`: implementación de la simulación.
- `scripts/check_operational_run.py`: health check y bloqueo de simulaciones.
- `scripts/ingest_aemet_current_observations.py`: colector horario real.
- `scripts/ingest_aemet_weather_state.py`: backfill y reconciliación AEMET.
- `docs/deployment/ejecucion_local_aemet.md`: operación local completa.
- `docs/deployment/servidor_produccion.md`: despliegue con servicios y timers.
