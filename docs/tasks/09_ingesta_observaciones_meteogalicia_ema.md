# 09. Ingesta de Observaciones de Estaciones MeteoGalicia (EMA) y Cierre de Latencia D-4 a D-1

---

## 1. ¿Qué se ha hecho?

- **Módulo cliente y normalizador de observaciones de MeteoGalicia (`src/ingestion/meteogalicia_observations.py`):**
  - Implementación del cliente HTTP para la API Open Data de MeteoGalicia: `https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action`.
  - Normalización del payload JSON multivariable (`listDatosDiarios` -> `listaEstacions` -> `listaMedidas`).
  - Transformación matemática de coordenadas UTM huso 29N (EPSG:25829) a coordenadas geográficas WGS84 (EPSG:4326), con implementación analítica de respaldo y enlace a `pyproj.Transformer`.
  - Mapeo de magnitudes meteorológicas físicas: conversión de viento de m/s a km/h ($\times 3.6$), cálculo físico del déficit de presión de vapor (VPD en kPa) y derivación de variables para la ventana solar crítica (12:00–18:00 hora local).
  - Algoritmo de interpolación espacial ponderada por el inverso de la distancia (IDW con 4 vecinos más cercanos) adaptado a la cuadrícula de 1 km² de Galicia (`galicia_grid_1km_egif.parquet`).
- **Script ejecutable en línea de comandos (`scripts/ingest_meteogalicia_observations.py`):**
  - Incorporación de detección automática del desfase (`--auto-fill-gap`) que inspecciona la fecha máxima disponible en `weather_daily_state.parquet` y descarga exclusivamente el intervalo pendiente hasta la víspera ($D-1$).
  - Gestión atómica de escritura con locks de concurrencia (`run_lock`) y archivado del JSON bruto original para trazabilidad y auditoría.
- **Suite de pruebas unitarias y de integración (`tests/test_meteogalicia_observations.py`):**
  - 5 tests automatizados que validan la exactitud de la transformación geodésica, el parseo de muestras reales de MeteoGalicia (`misc/Datos/meteogalicia_diario_julio2022.json`), la interpolación IDW y la compatibilidad estricta con el contrato de estado `weather_daily_state.parquet`.
  - 40 tests de regresión ejecutados con éxito sin impacto en los pipelines existentes.

## 2. ¿Por qué se ha hecho?

- **Eliminación del cuello de botella de latencia de AEMET:** La climatología diaria de AEMET sufre un desfase estructural de 3 a 5 días ($D-4$) debido a sus protocolos de control de calidad institucional centralizados.
- **Ruptura de las memorias de sequedad en inferencia diaria:** El modelo de Machine Learning requiere 30 días continuos previos a la emisión para estimar la racha seca (`dias_sin_lluvia`), la precipitación acumulada antecedente (`prec_acum_3d`, `prec_acum_7d`) y las medias térmicas (`tmax_media_7d`). Un vacío de 3 a 4 días en el estado forzaba a elegir entre detener la predicción o recurrir a suposiciones no contrastadas.
- **Superioridad de cobertura en el medio rural:** Mientras que AEMET dispone únicamente de ~35 estaciones en Galicia (concentradas en áreas urbanas y costeras), MeteoGalicia cuenta con una red de más de 140 estaciones automáticas (EMA) densamente repartidas por las comarcas forestales del interior gallego.

## 3. ¿Cómo se ha hecho?

- **Flujo de Ejecución y Lógica Algorítmica:**
  1. El script lee el estado meteorológico actual (`data/processed/state/weather_daily_state.parquet`) y obtiene la fecha más reciente registrada (por ejemplo, día 2 a las 23:59).
  2. Determina el intervalo faltante hasta el día de corte operativo ($D-1$, ayer a las 23:59 UTC, por ejemplo días 3, 4 y 5: 72 horas).
  3. Formula una petición HTTP GET con parámetros `datIni=03/09/2026` y `dataFin=05/09/2026` al servicio Open Data de MeteoGalicia.
  4. Extrae los registros de cada estación, proyecta sus coordenadas `utmx`, `utmy` a WGS84, valida que se encuentren dentro del delimitador espacial de Galicia (`GALICIA_BOUNDS`) y unifica las medidas de temperatura máxima/media/mínima, humedad relativa mínima/media, precipitación acumulada y rachas de viento.
  5. En caso de ausencia puntual de un sensor específico en una estación (ej. termómetro sin higrómetro), aplica un mecanismo de proxy con la media diaria para maximizar la representatividad territorial.
  6. Para cada día del lote, proyecta las coordenadas métricas locales y construye un árbol k-d (`scipy.spatial.cKDTree`) para interpolar mediante IDW a las 29.601 celdas de 1 km².
  7. Ejecuta una fusión (`merge_weather_state`) y persistencia atómica temporal con reemplazo seguro en disco.

## 4. ¿Por qué se han elegido estas tecnologías?

- **API REST Open Data de MeteoGalicia (`mgrss/observacion`):** Permite consultas masivas por rango de fechas para todas las estaciones en servicio en un único archivo JSON ligero, sin límites restrictivos por petición y sin requerir autenticación compleja.
- **Transverse Mercator Analítica + `pyproj`:** Garantiza una conversión geodésica de alta precisión entre el sistema oficial UTM Huso 29N (EPSG:25829) y WGS84 (EPSG:4326), asegurando la compatibilidad con los centroides de la cuadrícula de 1 km.
- **Interpolación IDW con `scipy.spatial.cKDTree`:** Ofrece una complejidad algorítmica $O(N \log M)$ para la asignación espacial sobre 30.000 cuadrículas, permitiendo completar la interpolación diaria en menos de un segundo por fecha.
- **Fusión Atómica y Locks (`run_lock`):** Evita condiciones de carrera entre el colector de observaciones y el timer de inferencia diaria en servidores de producción.

## 5. ¿Qué conseguimos con ello?

- **Independencia y robustez operativa:** El sistema puede encenderse en cualquier momento del día y recuperar automáticamente las 72 a 120 horas previas sin necesidad de mantener procesos demonio permanentes de captura horaria continua.
- **Rigor metodológico y Anti-Data-Leakage:** Las características retrospectivas de sequedad y combustible fino se alimentan con observaciones físicas reales de pluviómetros y termómetros, sin inventar ceros ni utilizar datos futuros respecto a la frontera de emisión ($D-1$).
- **Alineación con el Estado del Arte:** Posiciona el TFM en la vanguardia de integración operativa multirred (MeteoGalicia EMA + AEMET Climatología) superando la resolución espacial de los índices continentales como EFFIS/GEFF (10–25 km) y validando la arquitectura de alta densidad promovida por IberFire (Ercibengoa et al., 2025).
