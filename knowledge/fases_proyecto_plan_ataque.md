# Fases del Proyecto
## TFM: Sistema Predictivo para la Anticipación de Incendios Forestales

---

### Fase 1: Configuración de la Infraestructura Espacial y Datos Estáticos
**Objetivo:** Construir el "tablero de juego" geográfico sobre el cual se estructurará toda la información temporal posterior.

#### Pasos Clave:
1. **Definición de la Rejilla Geoespacial (Malla Base):**
   * Descargar los límites geográficos de la Comunidad Autónoma seleccionada desde el CNIG/IGN.
   * Utilizar la librería `GeoPandas` combinado con `shapely` (o el sistema de celdas hexagonales `H3` de Uber / rejilla regular de `E開放`) para generar una malla de polígonos de $1\text{ km} \times 1\text{ km}$.
   * Asignar un identificador único indexable a cada celda (`cell_id`).
2. **Ingesta y Extracción de Variables Topográficas (Copernicus DEM):**
   * Descargar el Modelo Digital de Elevaciones (GLO-30 o GLO-90).
   * Mediante la librería `rasterio` o `rioxarray`, superponer la malla de celdas sobre el raster topográfico.
   * Calcular para cada celda: la altitud media, la pendiente media (crucial para la propagación y accesibilidad) y la orientación del terreno (las laderas solana reciben más radiación y son más secas).
3. **Ingesta y Procesamiento de la Cobertura del Suelo (CORINE Land Cover):**
   * Descargar el dataset vectorial o raster de CORINE.
   * Agrupar las categorías originales en macro-clases de combustible forestal (ej: Bosque denso de frondosas, Bosque de coníferas, Matorral transicional, Pastizal, Zonas agrícolas, Áreas urbanas expuestas).
   * Calcular la clase predominante o el porcentaje de cobertura de cada clase dentro de cada celda de 1x1 km.

**Entregable de la Fase:** Un DataFrame estático o base de datos espacial donde la clave es `cell_id` y las columnas son `[latitud, longitud, altitud, pendiente, orientacion, tipo_combustible_predominante]`. Esto se calcula **una sola vez**.

---

### Fase 2: Ingesta del Histórico Dinámico y Construcción del Target
**Objetivo:** Descargar, limpiar y estructurar los eventos históricos (incendios) y las condiciones meteorológicas del pasado para crear las filas de entrenamiento.

#### Pasos Clave:
1. **Procesamiento de la Variable Objetivo (NASA FIRMS):**
   * Descargar el histórico de anomalías térmicas (MODIS / VIIRS) de los últimos 5 años para la zona geográfica del MVP.
   * **Clusterización Espacio-Temporal:** Escribir un algoritmo en Python (usando `DBSCAN` de `scikit-learn` o reglas vectoriales personalizadas) para agrupar registros de calor contiguos en el espacio (distancia < 5 km) y tiempo (ventana < 4 días). Esto consolida focos dispersos en un único "Incendio Forestal Real".
   * Identificar la fecha de inicio ($T_0$) y la celda origen de cada cluster. Esa celda-día recibirá la etiqueta $Y = 1$ (Inicio de incendio).
2. **Muestreo Inteligente de Negativos Difíciles ($Y = 0$):**
   * Implementar la lógica para evitar que el modelo aprenda sesgos triviales.
   * Generar **Vecinos Espaciales:** Celdas contiguas al incendio que experimentaron idéntica meteorología pero no ardieron por factores de terreno o combustible.
   * Generar **Vecinos Temporales:** La misma celda del incendio, observada 10 días antes o 14 días después.
   * Generar **Muestras Aleatorias Estratificadas:** Celdas distribuidas por todo el territorio y a lo largo de las 4 estaciones del año para dar contexto basal de "no incendio".
   * *Regla de Exclusión:* Eliminar cualquier negativo potencial que se encuentre dentro de la ventana de seguridad de $\pm 5$ días respecto a un incendio real en la misma zona.
3. **Ingesta de la Meteorología Histórica (Copernicus ERA5-Land):**
   * Descargar los archivos horarios en formato NetCDF (.nc) para los rangos de fechas seleccionados en los puntos anteriores (fechas de positivos y fechas de negativos).
   * **Filtrado Operativo por Ventana Crítica:** En lugar de promediar las 24 horas del día, extraer y computar las variables exclusivamente en el rango de máxima vulnerabilidad: **12:00h a 18:00h**.
   * Calcular para cada celda y día: Temperatura Máxima, Humedad Mínima, Velocidad Máxima del Viento, Dirección del Viento y Precipitación del día anterior.

**Entregable de la Fase:** Un dataset integrado de entrenamiento intermedio que une el espacio (`cell_id`) y el tiempo (`fecha`) con su etiqueta objetivo (`target`).

---

### Fase 3: Ingeniería de Características y Prevención de Data Leakage
**Objetivo:** Enriquecer el dataset con variables acumuladas y asegurar que el modelo sea matemáticamente honesto respecto al tiempo real.

#### Pasos Clave:
1. **Generación de Variables Secuenciales y Acumuladas:**
   * Calcular los días consecutivos sin lluvia (*Días Secos Acumulados*).
   * Calcular la precipitación acumulada en ventanas móviles de corto, medio y largo plazo: 3 días, 7 días, 14 días y 30 días. Esto modela el estado de estrés hídrico de la vegetación viva y el combustible muerto.
2. **Inclusión de la Presión Antrópica e Historial:**
   * Calcular la distancia euclídea o por red desde el centro de la celda a la carretera más cercana y al núcleo urbano más cercano (usando OpenStreetMap). El factor humano causa más del 80% de los inicios.
   * Calcular el histórico de incendios acumulado en el vecindario de la celda en los últimos 3 años (frecuencia de recurrencia).
3. **Auditoría Estricta contra el "Data Leakage" (Filtrado Temporal):**
   * Implementar una validación en el código para asegurar que para predecir el riesgo del día $T$, el dataset solo contenga características calculadas con datos disponibles hasta el día $T-1$. Se prohíbe el uso de cualquier métrica observada en el mismo día del incendio.

**Entregable de la Fase:** El **Dataset Maestro de Entrenamiento** en formato optimizado (`.parquet` o `.csv` comprimido), balanceado y estructurado en forma tabular listo para algoritmos de Machine Learning.

---

### Fase 4: Modelado, Entrenamiento y Calibración del Riesgo
**Objetivo:** Desarrollar los algoritmos predictivos, evaluar su capacidad de generalización en el futuro y calibrar las salidas probabilisticas en niveles de riesgo accionables.

#### Pasos Clave:
1. **Estrategia de Validación Temporal Rigurosa (Time-Series Split por Años):**
   * Queda estrictamente prohibido usar un *K-Fold Cross Validation* aleatorio tradicional.
   * Configurar la partición de datos por bloques anuales completos:
     * *Entrenamiento:* 2019, 2020, 2021
     * *Validación / Tuning de Hiperparámetros:* 2022
     * *Test Final (Ciego):* 2023 / 2024
   * Esto simula exactamente las condiciones de producción: predecir el futuro inmediato utilizando únicamente el histórico del pasado.
2. **Entrenamiento y Ajuste de Modelos:**
   * Entrenar una Regresión Logística regularizada como línea base (*Baseline*).
   * Entrenar modelos basados en árboles de decisión optimizados para datos desbalanceados: **XGBoost** y **LightGBM**. Utilizar parámetros de penalización de clase (`scale_pos_weight`) o técnicas de balanceo en el pipeline.
   * Evaluar prioritariamente mediante métricas que penalicen los falsos negativos sin destruir la precisión: AUC-ROC, Precision-Recall Curve (PR-AUC) y F1-Score.
3. **Calibración Isotónica de Probabilidades a Niveles de Riesgo:**
   * Los modelos devuelven una probabilidad matemática cruda (ej: `0.012`). Esta cifra no es intuitiva para un operativo de protección civil.
   * Cruzar el histórico de predicciones del conjunto de validación con los incendios reales ocurridos.
   * Definir los umbrales de riesgo basándose en la tasa de incidencia real:
     * **Riesgo Bajo:** Rangos de probabilidad donde históricamente ocurre < 1% de los incendios.
     * **Riesgo Moderado:** Rango donde se concentra el 10% de los incendios reales.
     * **Riesgo Alto:** Umbral a partir del cual la densidad de incendios reales se dispara exponencialmente.
     * **Riesgo Extremo:** Combinación extrema de viento, calor y sequedad donde el 80% de los días con estas condiciones presentaron focos activos.

**Entregable de la Fase:** El artefacto del modelo entrenado y serializado (`.pkl` o `.json` de XGBoost) junto con la tabla de umbrales calibrados para la conversión a niveles de riesgo.

---

### Fase 5: Desarrollo de la WebApp e Integración Operativa 
**Objetivo:** Construir la interfaz de usuario interactiva y automatizar el flujo diario que alimenta el sistema predictivo con previsiones meteorológicas reales en lugar de reanálisis históricos.

#### Pasos Clave:
1. **Desarrollo del Pipeline de Inferencia Diaria (Script de Producción):**
   * Programar un script en Python que se ejecute de forma automatizada cada madrugada (ej: a las 05:00 AM mediante un Cron Job).
   * Este script realiza una llamada a **MeteoGalicia MeteoSIX v4** para descargar el forecast horario WRF de los próximos tres días.
   * Consultar puntos representativos de la malla WRF en lotes de hasta 20 localizaciones y asignarlos a la rejilla estática de celdas de 1x1 km mediante vecino más cercano.
   * Alimentar los tres modelos de riesgo serializados (T+1, T+2 y T+3) con estas características predictivas.
2. **Construcción de la Interfaz Gráfica con Streamlit:**
   * Crear un cuadro de mando (*Dashboard*) interactivo utilizando la librería `Streamlit`.
   * Integrar componentes de mapeo avanzados como `Folium` o `PyDeck` para pintar las celdas de la rejilla de colores según su nivel de riesgo calibrado (Verde: Bajo -> Rojo/Púrpura: Extremo).
   * Añadir filtros interactivos por provincia, municipio, fecha de predicción (24h/72h) y un panel lateral que muestre la importancia de las variables (usando valores `SHAP` simplificados o la importancia nativa del modelo) para justificar técnicamente por qué una celda específica se encuentra en riesgo extremo (ej: "Humedad relativa críticamente baja combinada con vientos superiores a 40 km/h").
3. **Análisis de Degradación del Modelo (Sección Científica del TFM):**
   * Comparar y documentar en la memoria del TFM la diferencia de rendimiento del modelo cuando se alimenta con datos de reanálisis perfectos (ERA5-Land) frente a forecasts archivados de MeteoGalicia, cuantificando cómo el error meteorológico afecta a la alerta temprana de incendios.

**Entregable de la Fase:** Repositorio de código final, WebApp funcional desplegada en local o en Streamlit Cloud, y la redacción final de la memoria del TFM.
