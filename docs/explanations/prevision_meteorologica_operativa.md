# Previsión meteorológica operativa y riesgo de incendio

## 1. Qué predice cada modelo

MeteoGalicia ejecuta un modelo numérico de predicción meteorológica WRF. El
modelo resuelve una aproximación física de la atmósfera a partir de condiciones
iniciales, observaciones y condiciones de contorno. La API MeteoSIX v5 expone
salidas horarias de temperatura, humedad relativa, precipitación y viento para
las mallas WRF de 1 km, 04 km, 12 km y 36 km.

El modelo de este proyecto no predice la meteorología. Aprende la probabilidad
de inicio de incendio condicionada a la meteorología prevista, el terreno, el
combustible y la memoria de sequedad:

```text
forecast meteorológico + terreno + historial -> P(inicio de incendio)
```

ERA5-Land se utiliza para el histórico porque proporciona una serie coherente
de reanálisis. No debe confundirse con una previsión operativa.

El proveedor se selecciona mediante `FORECAST_PROVIDER`:

| Valor | Comportamiento |
|---|---|
| `meteogalicia` | Usa MeteoSIX v5 y exige `METEOGALICIA_API_KEY`. |
| `aemet` | Usa AEMET OpenData y exige `AEMET_API_KEY`. |
| `auto` | Elige MeteoGalicia si hay una clave real; si no, AEMET. |

La modalidad `auto` permite desarrollar y probar el pipeline antes de recibir
la clave de MeteoGalicia. No mezcla fuentes dentro de una misma ejecución: el
manifiesto registra un único proveedor para que el resultado sea reproducible.

## 2. Selección de la malla y degradación controlada

La fuente preferida es WRF `1km`, porque proporciona una representación
espacial más fina de los contrastes de costa, relieve y valles de Galicia.
Como el producto final se calcula sobre una rejilla de 1 km, la malla WRF
`04km` no se presenta como equivalente: es un fallback operativo con menor
detalle meteorológico.

El cliente intenta las mallas en este orden:

```text
WRF 1km -> WRF 04km -> último forecast archivado (stale)
```

La inferencia conserva `forecast_quality`, `forecast_grid`,
`forecast_selection`, `forecast_api_version`, `modelRun` y la antigüedad del
forecast. Los estados son:

| Estado | Significado | Uso operativo |
|---|---|---|
| `fresh` | WRF 1 km validado | Producto principal |
| `fresh_fallback` | WRF 1 km no disponible o incompleto; se usa WRF 04 km | Apoyo preventivo con confirmación |
| `fresh_aemet` | AEMET municipal, con expansión diaria y superposición horaria cuando existe | Pruebas o contingencia; no equivale a WRF |
| `fresh_aemet_proxy` | AEMET municipal con proxy explícito para precipitación ausente | Solo pruebas técnicas; no usar para evaluar peligro |
| `stale` | Se reutiliza el último forecast archivado | No publicar como actualización normal |

La consulta agrupa puntos representativos para respetar el límite de 20
localizaciones por petición. Por defecto se consultan puntos cada 4 km y se
asignan sus valores a la rejilla de 1 km; esta decisión limita el número de
peticiones, pero no cambia la malla meteorológica solicitada al proveedor.
Puede aumentarse la densidad con `METEOGALICIA_QUERY_RESOLUTION_KM=1.0` si el
tiempo de descarga y el límite de peticiones lo permiten.

## 3. Disponibilidad temporal de las ejecuciones WRF

Las horas publicadas por MeteoGalicia son aproximadas y están expresadas en
UTC. La ejecución WRF de 1 km iniciada a las 00:00 UTC termina aproximadamente
a las 07:30 UTC, por lo que una ejecución fija a las 05:00 hora local puede
encontrar todavía incompleto el forecast nuevo. Por eso la lógica de
producción decide por `modelRun` y validación de cobertura, no por la hora del
reloj. En una operación con mayor exigencia de frescura puede generarse un
resultado provisional temprano y refrescarlo después de la publicación de la
malla de 1 km.

## 3.1 Alternativa AEMET mientras no existe la clave MeteoGalicia

AEMET OpenData utiliza una API REST con doble petición: la primera devuelve una
URL temporal en `datos` y la segunda descarga el JSON de la predicción, según la
[especificación oficial de AEMET](https://opendata.aemet.es/AEMET_OpenData_specification.json).
La API ofrece predicción horaria por municipio hasta 48 horas y predicción diaria para
varios días. Por eso el adaptador del proyecto descarga la predicción diaria
para completar T+1/T+2/T+3, y sustituye las horas coincidentes por la
predicción horaria cuando está disponible.

La salida AEMET se asigna a la rejilla de 1 km desde puntos municipales
configurados en `AEMET_MUNICIPALITIES`. Por defecto se incluyen las cuatro
capitales provinciales como configuración de prueba. Esto permite validar el
circuito técnico completo, pero no representa los contrastes meteorológicos
locales de una malla WRF. En consecuencia, el dashboard muestra una advertencia
y el health check puede bloquear la publicación con `--fail-on-degraded`.

Si el endpoint diario no contiene precipitación cuantitativa, la configuración
normal falla de forma explícita porque no se puede convertir una probabilidad de
lluvia en milímetros. Para una prueba técnica aislada puede definirse
`AEMET_MISSING_PRECIPITATION_FALLBACK=0`; el resultado se etiqueta
`fresh_aemet_proxy` y no debe utilizarse para medir ni comparar riesgo.

### 3.2 Observaciones, backfill y precalentamiento

El archivo de observaciones no se crea manualmente. El script
`scripts/ingest_aemet_weather_state.py` consulta la climatología diaria de
AEMET para los últimos 30 días completos, obtiene el inventario de estaciones,
filtra Galicia e interpola cada variable a la rejilla mediante los cuatro
vecinos más cercanos. Publica tanto el input diario como el estado compacto:

```text
data/processed/observations/weather_daily_latest.parquet
data/processed/state/weather_daily_state.parquet
```

La especificación oficial de AEMET documenta el recurso de observación horaria
de las últimas 12 horas y el recurso de climatología diaria para un rango de
fechas y todas las estaciones ([API AEMET](https://opendata.aemet.es/AEMET_OpenData_specification.json)).
El servicio limita cada consulta diaria a 15 días; el script divide una ventana
de 30 días en bloques, reutiliza el inventario de estaciones y conserva un JSON
bruto por bloque. Por eso el primer arranque no necesita esperar 30 ciclos
diarios. Si alguno de los días solicitados todavía no está publicado, la
ingesta falla de forma explícita y no sustituye ese día por cero ni por una
fecha histórica distinta.

Para cerrar D-1 durante el desfase de publicación se ejecuta
`scripts/ingest_aemet_current_observations.py`. Este comando descarga la
ventana móvil, la acumula por estación e instante en
`aemet_hourly_observations.parquet`, y agrega un día solo si tiene al menos 20
horas por estación. En un ordenador personal se deja abierto en una terminal:

```bash
PYTHONPATH=. python scripts/ingest_aemet_current_observations.py \
  --loop --interval-hours 6
```

El backfill y el colector comparten un lock para no publicar dos estados
simultáneamente. Si el ordenador permanece apagado, las horas que salgan de la
ventana móvil no se pueden inventar: se espera a que la climatología validada
las publique o se mantiene la inferencia bloqueada.

MeteoSIX v5 no sustituye este backfill: su operación numérica devuelve forecast
desde el día actual, con un máximo de siete días, y `precipitation_amount` es
la precipitación prevista acumulada durante la hora anterior. No existe en esa
operación un endpoint de observaciones pasadas que permita pedir los últimos 30
días. Por tanto, con MeteoGalicia el forecast seguirá siendo la fuente futura,
pero el estado previo deberá proceder de observaciones de estaciones o de un
análisis histórico separado.

La diferencia de proveedor debe conservarse también en la evaluación: un
benchmark realizado con AEMET no puede compararse directamente con el resultado
de WRF 1 km sin estratificar por proveedor, horizonte, resolución y cobertura.

## 4. Horizontes del producto

Una ejecución a las 05:00 del día `D` genera tres objetivos diarios:

| Horizonte | Pregunta operativa | Fecha válida |
|---|---|---|
| T+1 | ¿Qué riesgo hay durante mañana? | `D + 1` |
| T+2 | ¿Qué riesgo hay durante pasado mañana? | `D + 2` |
| T+3 | ¿Qué riesgo hay durante el tercer día? | `D + 3` |

Cada mapa utiliza la ventana local 12:00–18:00. Para T+2 y T+3, los días
forecast intermedios también pueden alimentar las memorias de precipitación y
temperatura.

## 5. Flujo de producción

1. Seleccionar el proveedor configurado y descargar su forecast.
2. Guardar el JSON original y el forecast normalizado horario.
3. Asignar cada predicción a la rejilla de 1 km.
4. Agregar la ventana crítica y la precipitación diaria.
5. Combinar el forecast con el estado de observaciones disponible hasta la
   emisión, sin utilizar datos posteriores.
6. Ejecutar el modelo serializado específico del horizonte.
7. Guardar probabilidad, percentil, nivel de riesgo, fecha de emisión y estado
   de frescura.

Antes de construir las memorias se carga el estado diario reciente de
`data/processed/state/weather_daily_state.parquet`. Debe contener 30 días
completos anteriores a la emisión para todas las celdas; la fecha de ejecución
no se considera cerrada a las 05:00. El forecast conserva por separado
`forecast_run_at`, `downloaded_at`, `valid_time` y `horizon_hours`.

Si la descarga de 1 km falla, se intenta la malla 04 km y el resultado queda
marcado como `fresh_fallback`. Solo si ambas mallas fallan se reutiliza el
último forecast archivado que cubra los tres días, marcado como `stale`. No se
usa una fila histórica como sustituto silencioso de una previsión.

Cuando `FORECAST_PROVIDER=aemet`, el resultado queda marcado como
`fresh_aemet`, con `provider=aemet`, `grid=municipal` y
`source_resolution=daily_expansion` o `hourly`. No se activa el fallback WRF
porque AEMET es una selección explícita de proveedor, no una malla alternativa
de MeteoGalicia.

## 6. Variables y semántica física

MeteoSIX entrega `precipitation_amount` como precipitación acumulada durante
la hora anterior. Por ello `prec_dia` se obtiene sumando las horas del día, no
promediándolas. La velocidad y la dirección del viento se conservan por
separado; las direcciones no se promedian como números ordinarios. El VPD se
calcula a partir de temperatura y humedad relativa.

## 7. Validación

El benchmark histórico usa meteorología ERA5-Land real y se etiqueta como
`era5_perfect`. Permite comprobar la alineación temporal y la capacidad del
modelo, pero no mide el error real de MeteoGalicia. Para medir esa degradación
se archivan desde producción los forecasts emitidos y, cuando existan
observaciones posteriores, se comparan por variable y horizonte.

La corrección estadística del forecast queda fuera de la primera versión. Solo
se incorporará cuando existan pares históricos forecast-observación; ordenar
los valores de un único día y forzarlos contra la distribución ERA5 no es una
calibración meteorológica válida.

Para el diseño completo de contratos, fórmulas, pruebas, operación, manifiestos
y roadmap, consulta
[`pipeline_operativo_detallado.md`](pipeline_operativo_detallado.md).
