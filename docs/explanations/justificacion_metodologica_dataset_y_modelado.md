# 📚 Documento Maestro de Justificación Metodológica, Dataset, Variables y Roadmap (TFM)

**Proyecto:** Sistema de Predicción Inteligente de Riesgo de Incendios Forestales en Galicia (TFM)  
**Fecha de Actualización:** Julio 2026  

---

## 1. 📌 Contexto Global, Alcance del Proyecto y Caso de Uso (MVP Galicia)

### 1.1 Definición del Problema
El objetivo central de este Trabajo Fin de Máster (TFM) es el desarrollo de un pipeline integral, operativo y de extremo a extremo para la **predicción diaria de la probabilidad de ignición (Día 0)** de incendios forestales a una resolución espacial de alta definición ($1\text{ km} \times 1\text{ km}$) en la Comunidad Autónoma de Galicia.

A nivel peninsular, el procesamiento diario a 1 km² durante 5 años genera más de 900 millones de registros, lo que comprometería la agilidad en la ingeniería de variables y la iteración de modelos. Galicia se ha seleccionado como la región piloto para el Producto Mínimo Viable (MVP) debido a tres factores críticos:
1. **Extrema recurrencia de incendios:** Galicia concentra históricamente la mayor densidad de fuegos forestales del noroeste peninsular.
2. **Complejidad del ecosistema:** Alta fragmentación de la propiedad de la tierra (minifundio), densa masa forestal y alta interfaz urbano-forestal.
3. **Densidad de red meteorológica:** Disponibilidad de la red observacional y modelos numéricos predictivos de alta resolución de **MeteoGalicia** (WRF a 1–4 km).

### 1.2 Rejilla Geoespacial Estándar
Se adopta la malla regular de celdas de $1000\text{ m} \times 1000\text{ m}$ (coordenadas proyectadas UTM ETRS89 Fuso 29N / EPSG:25829 y ETRS89 LAEA / EPSG:3035). Galicia comprende exactamente **30.697 celdas terrestres**.

---

## 2. 🔬 Justificación Detallada de las Decisiones Metodológicas

---

### 🔹 Decisión A: Ground Truth del Target — EGIF del MITECO (2019–2023) frente a NASA FIRMS
* **Decisión:** Establecer la **Estadística General de Incendios Forestales (EGIF del MITECO)** como el Ground Truth oficial para el entrenamiento supervisado ($Y \in \{0, 1\}$).
* **Justificación:**
  - **Calidad Auditada:** Cada registro del EGIF es verificado sobre el terreno por agentes forestales e incluye el día exacto de ignición, la causa y la superficie quemada ($>0.1\text{ ha}$).
  - **Eliminación del Ruido Térmico:** NASA FIRMS detecta anomalías térmicas por satélite, incluyendo falsos positivos (quemas agrícolas autorizadas, emisiones industriales, reflexiones solares) que sesgan los patrones climáticos aprendidos.
  - **Uso Secundario de NASA FIRMS:** NASA FIRMS (2024+) se preserva para validar la inferencia en tiempo casi-real (Near-Real-Time).

---

### 🔹 Decisión B: Ignición en Día 0 frente a Simulación de Propagación (Fuego Activo)
* **El valor de negocio de la Ignición (Día 0):** El objetivo de un sistema de alerta temprana es la **prevención**. Para desplegar patrullas a las 08:00 AM, el sistema debe predecir la vulnerabilidad del terreno **antes de que salte la chispa**.
* **Prevención de Data Leakage:** Clasificar como positivos ($Y=1$) todos los días que un incendio arde de forma continuada (días 1, 2, 3...) fuerza al modelo a memorizar la regla trivial *"si ayer había fuego en la celda, hoy hay 99% de riesgo"*. Predecir exclusivamente el **Día 0 de inicio** elimina esa distorsión y garantiza un modelo centrado en la susceptibilidad ambiental real.
* **Contagio Espacial y Celdas Vecinas:** 
  1. *Sincronía Atmosférica:* Celdas vecinas comparten la misma ola de calor y sequía (co-vulnerabilidad).
  2. *Contagio Físico:* Se captura en el Datacubo 3D (redes neuronales convolucionales sobre tensores $25 \times 25$ celdas) y en el dataset tabular mediante la variable de distancia a fuegos activos en los últimos 3 días (`dist_fuego_activo_3d`).
  3. *Cicatriz de Incendio (Burn Scar):* Una celda ya quemada reduce su porcentaje de combustible a cero, evitando falsas alarmas posteriores.

---

### 🔹 Decisión C: Selección de 20 Variables y Principio de Parsimonia
* **Prevención de sobreajuste (*Overfitting*):** Incluir decenas de variables redundantes o con varianza cero degrada los modelos y ralentiza la inferencia diaria.
* **Optimización de CORINE Land Cover:** A diferencia de **IberFire** (que incluye 63 columnas para 44 clases de uso de suelo, de las cuales más de 40 resultan nulas en Galicia), condensamos el mapa de combustibles en el **porcentaje efectivo de masa forestal inflamable y matorral (`combustible_pct_forestal`)**, pasando de 63 columnas vacías a 1 columna densa.

---

### 🔹 Decisión D: Abordaje del Sesgo Meteorológico (ERA5 vs MeteoGalicia)
* **Entrenamiento (2019–2023):** Se usa **ERA5-Land** ($9\text{ km}$ reanálisis downscaled a 1 km con DEM) por ser una serie continua y homogénea sin lagunas.
* **Producción Operativa (2025+):** Se usa la previsión de **MeteoGalicia (WRF)**.
* **Mitigación:** Aplicación de **Quantile Mapping** / normalización por celda para eliminar el *Domain Shift*. La cuantificación empírica de esta brecha constituye una contribución central del TFM.

---

### 🔹 Decisión E: Prevención de Fuga de Datos (Desfase Temporal $T-1$)
Todas las variables meteorológicas explicativas ($X_{T-1}$) se desplazan exactamente 24 horas respecto al día de evaluación de la ignición ($Y_T$), garantizando que el modelo prediga el riesgo de hoy basándose estrictamente en la información disponible hasta el cierre de ayer.

---

### 🔹 Decisión F: Tratamiento del Desbalanceo Extremo (0.0026% Positivos)
1. **Hard Negative Mining:** Entrenamiento con submuestreo controlado $1:50$ conservando el 100% de los fuegos reales.
2. **Métricas Válidas:** Rechazo de ROC-AUC como métrica única. Adopción de **PR-AUC (Precision-Recall)**, **Recall a FPR $\le 5\%$** y **Brier Score**.
3. **Calibración Isotónica:** Ajuste de probabilidades continuas de salida mediante *Isotonic Regression*.

---

## 3. 📊 Diccionario Exhaustivo de las 20 Variables del Dataset Maestro

| Nombre de la Variable | Tipo de Dato | Unidades | Fuente de Origen | Descripción y Justificación Física |
| :--- | :--- | :--- | :--- | :--- |
| `cell_id` | Entero (`int64`) | ID Único | Grid Galicia (1 km²) | Identificador unívoco de la celda sobre Galicia. |
| `fecha` | Cadena / Date | `YYYY-MM-DD` | Temporal | Fecha de evaluación de la ignición ($T$). |
| `tmax_vc` | Flotante (`float64`) | $^\circ\text{C}$ | ERA5-Land ($T-1$) | Temperatura máxima del día anterior. |
| `rhmin_vc` | Flotante (`float64`) | $\%$ | ERA5-Land ($T-1$) | Humedad relativa mínima del día anterior. |
| `vmax_vc` | Flotante (`float64`) | $\text{km/h}$ | ERA5-Land ($T-1$) | Velocidad máxima de ráfaga de viento del día anterior. |
| `prec_dia` | Flotante (`float64`) | $\text{mm}$ | ERA5-Land ($T-1$) | Precipitación total diaria del día anterior. |
| `prec_acum_7d` | Flotante (`float64`) | $\text{mm}$ | Calculada ($T-1$) | Precipitación acumulada en los últimos 7 días. |
| `prec_acum_30d` | Flotante (`float64`) | $\text{mm}$ | Calculada ($T-1$) | Precipitación acumulada en los últimos 30 días (estrés hídrico). |
| `tmax_media_7d` | Flotante (`float64`) | $^\circ\text{C}$ | Calculada ($T-1$) | Temperatura máxima media de la última semana. |
| `alerta_30_30` | Binario (`0/1`) | Flag | Regla Experta | Vale $1$ si $T_{max} \ge 30^\circ\text{C}$ y $RH_{min} \le 30\%$. |
| `altitud_media` | Flotante (`float64`) | $\text{m}$ | DEM Copernicus | Elevación media de la celda sobre el nivel del mar. |
| `pendiente_media` | Flotante (`float64`) | Grados ($^\circ$) | DEM Copernicus | Pendiente media. Terrenos inclinados facilitan la convección. |
| `orientacion_media` | Flotante (`float64`) | Grados ($0\text{-}360^\circ$) | DEM Copernicus | Orientación del terreno (solanas vs umbrías). |
| `combustible_pct_forestal` | Flotante (`float64`) | $\%$ | CORINE Land Cover | Porcentaje de la celda cubierto por biomasa inflamable. |
| `dist_carreteras` | Flotante (`float64`) | Metros | CNIG / IGN | Distancia a la red viaria más cercana (accesibilidad humana). |
| `dist_urbanos` | Flotante (`float64`) | Metros | CNIG / IGN | Distancia al núcleo urbano más cercano (interfaz urbano-forestal). |
| `mes` | Entero (`1-12`) | Mes | Temporal | Estacionalidad anual del riesgo. |
| `dia_semana` | Entero (`0-6`) | Día | Temporal | Captura picos de actividad humana semanal. |
| `es_finde` | Binario (`0/1`) | Flag | Temporal | Vale $1$ los sábados y domingos. |
| `dia_año_sin` / `cos` | Flotante (`-1 a 1`) | Cíclica | Calculada | Seno y coseno del día del año ($2\pi \times \text{DOY} / 365.25$). |
| **`target` (Variable Y)** | **Binario (`0/1`)** | **Ignición** | **EGIF MITECO** | **$1$ si se inició un incendio en la celda el día $T$; $0$ en caso contrario.** |

---

## 4. 📈 Benchmark Comparativo con IberFire y Resultados Empíricos

### 4.1 Resultados de IberFire en la Literatura (Ercibengoa et al., 2025)
* **Paper IberFire:** Muestra de entrenamiento balanceada $1:1$ a nivel España. Evaluado sobre 2024 alcanzando un **AUROC de 0.95** y **Accuracy del 86%**.
* **Matiz Metodológico:** IberFire predice presencia de Fuego Activo en datasets balanceados al 50%. Nosotros predecimos **Igniciones en Día 0 sobre datasets con desbalanceo real de 1:50**.

### 4.2 Resultados Empíricos del Experimento Baseline (Evaluación Año 2023 Out-of-Sample)
Evaluación sobre **56.052.722 observaciones** (11.204.405 filas de test ciego en 2023):

| Modelo Baseline | PR-AUC | ROC-AUC | Recall @ $FPR \le 5\%$ | Brier Score | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística (L2)** | $0.000027$ | $0.7784$ | $16.67\%$ | $0.1359$ | Baseline Lineal |
| **Random Forest Classifier** | $0.000031$ | $0.8282$ | $21.57\%$ | $0.0199$ | Ensamble Embolsado |
| **XGBoost Classifier** | $0.000049$ | $0.8297$ | $27.45\%$ | $0.0007$ | Gradiente Potenciado |
| **LightGBM Classifier (Ganador)** | **$0.000071$** | **$0.8377$** | **$32.35\%$** | **$0.0010$** | **Baseline Oficial** |

### 4.3 Análisis de Parsimonia: 20 Variables vs Arquitecturas Complejas
Para comprobar si era necesario incluir índices complejos (VPD, Nesterov, Ratios) o ensambles adicionales (CatBoost), se ejecutó un experimento de sensibilidad. Los resultados mostraron que añadir estas variables y ensambles complejos únicamente mejoró el Recall en un $+0.98\%$ (del $32.35\%$ al $33.33\%$) a costa de triplicar el tiempo de cómputo. Se concluye que **el conjunto de 20 variables principales ya captura la práctica totalidad de la señal física de ignición**, respaldando la elección de la versión parsimoniosa.

### 4.4 Evaluación de la Inflación de ROC-AUC y Filtrado de Extinción Hídrica ($P < 5\text{ mm}$)
Un análisis en profundidad de los 1.645 eventos de ignición históricos (2019-2023) demostró la validez del filtrado físico:
* **🔥 1.582 Fuegos en Días Secos ($P < 5\text{ mm}$):** Ocurrieron con una $T_{max}$ media de $25.5^\circ\text{C}$ y una $RH_{min}$ media del $38.9\%$ (estrés hídrico y desecación del combustible).
* **🌧️ 63 Fuegos en Días Lluviosos ($P \ge 5\text{ mm}$):** Ocurrieron con una $T_{max}$ media de $20.3^\circ\text{C}$ y una $RH_{min}$ media elevada del $58.6\%$.

En física de incendios, cuando la humedad relativa supera el $55\%$, la humedad del combustible fino ($MC_{ff}$) supera el umbral de extinción ($30\%$), anulando la velocidad de propagación. Cualquier ignición accidental en día lluvioso se apaga de forma natural o en fase de conato inicial ($<0.1\text{ ha}$).
- **En el Dataset Completo:** El ROC-AUC alcanza $0.8377$ por el acierto masivo en ceros invernales triviales.
- **En el Dataset Filtrado (Solo Días Secos):** Al eliminar **18.370.328 ceros triviales** conservando el **$96.2\%$ de los incendios reales**, el ROC-AUC se sitúa en $0.7568$, lo que representa la **verdadera capacidad de discriminación del modelo en días de peligro real**.

### 4.5 Matriz de Trazabilidad Técnica: Dónde y Cómo se ha Implementado Cada Decisión en `src/`

Para garantizar la auditoría técnica del repositorio, a continuación se detalla la correspondencia exacta entre las decisiones metodológicas adoptadas y su implementación en los módulos de producción de la carpeta `src/`:

| Decisión Metodológica | Archivo de Código (`src/`) | Función / Lógica Implementada | Detalle Técnico |
| :--- | :--- | :--- | :--- |
| **A. Target EGIF MITECO** | [`src/ingestion/ingest_egif.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/ingestion/ingest_egif.py) | `parse_egif_xml()` y `map_fires_to_grid()` | Extrae 9.549 incendios oficiales verificados por agentes forestales en las provincias 15, 27, 32 y 36 de Galicia ($>0.1\text{ ha}$). |
| **B. Ignición Día 0** | [`src/ingestion/ingest_egif.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/ingestion/ingest_egif.py) | `map_fires_to_grid()` | Agrupa por `(cell_id, fecha)` tomando la fecha de inicio/detección, descartando la propagación de días 2, 3+ para evitar *temporal leakage*. |
| **C. Parsimonia 20 Variables** | [`src/features/build_tabular_dataset.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/features/build_tabular_dataset.py) | `assemble_tabular_dataset()` | Ensambla las 20 columnas principales y condensa las 63 clases de CORINE en la variable única `combustible_pct_forestal`. |
| **D. Quantile Mapping Meteo** | [`src/ingestion/ingest_meteogalicia.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/ingestion/ingest_meteogalicia.py) | `apply_quantile_mapping()` | Ajusta los percentiles de la previsión numérica WRF de MeteoGalicia a la serie de reanálisis ERA5-Land para eliminar el *Domain Shift*. |
| **E. Desfase Temporal $T-1$** | [`src/features/temporal_shift.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/features/temporal_shift.py) | `apply_temporal_shift()` y `compute_climate_memory()` | Aplica `.shift(1)` de 24h a predictores meteorológicos y calcula memorias hídricas a 7d y 30d (`prec_acum_7d`, `prec_acum_30d`). |
| **F. Hard Negative Mining** | [`src/models/train_baseline.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/models/train_baseline.py) | `hard_negative_sampling()` | Mantiene el 100% de los fuegos reales y realiza un submuestreo controlado 1:50 de ceros representativos. |
| **G. Métricas y Calibración** | [`src/models/metrics.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/models/metrics.py) | `evaluate_imbalanced_metrics()` e `isotonic_calibration` | Calcula PR-AUC, Recall @ $FPR \le 5\%$ y ajusta probabilidades con *Isotonic Regression*. |
| **H. Pruning Cero-Físico** | [`scripts/run_baseline_experiment.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/scripts/run_baseline_experiment.py) | Filtrado `df["prec_dia"] < 5.0` | Elimina 18.37M de ceros invernales por extinción hídrica conservando el $96.2\%$ de las igniciones. |
| **I. Red Neuronal 3D Conv3D** | [`src/models/train_nn_3d.py`](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/src/models/train_nn_3d.py) | `FireConv3DNet` en PyTorch | Extrae convoluciones espacio-temporales en parches 3D ($25 \times 25\text{ km}$) multiplicando el PR-AUC ($0.1791$). |

---

## 5. 🗺️ Estructura del Repositorio Actual

```text
Sistema-deteccion-incendios/
├── docs/
│   ├── explanations/
│   │   └── justificacion_metodologica_dataset_y_modelado.md  (Este documento maestro)
│   ├── tasks/
│   │   ├── 01_infraestructura_geoespacial.md
│   │   ├── 02_ingesta_y_dataset_maestro.md
│   │   ├── 03_ingenieria_de_features.md
│   │   ├── 04_entrenamiento_y_modelos_baseline.md
│   │   └── 05_optimizacion_avanzada_tabular_y_blending.md
│   └── tfm_borrador_memoria.md
├── scripts/
│   ├── run_baseline_experiment.py
│   └── run_advanced_ensemble_experiment.py
└── src/
    ├── geospatial/                             (Fase 1 - Rejilla Geoespacial)
    ├── ingestion/                              (Fase 2 - Ingesta EGIF MITECO, ERA5 y MeteoGalicia)
    ├── features/                               (Fase 3 - Shift T-1, Datacubo 3D e Índices Físicos)
    └── models/                                 (Fase 4 - Métricas, Baseline y Sintonización)
```

---

## 6. 🎯 Próximos Pasos Operativos

### 🔹 A. Para la Reunión del Equipo (Sábado)
### 🔹 B. Próximas Tareas para la Continuación del TFM (Fases 5 y 6)

Habiendo validado empíricamente a **LightGBM Standard** como el modelo ganador definitivo (y desestimado las Redes Neuronales 3D por su menor Recall a FPR $\le 5\%$), las tareas inmediatas para completar el TFM son:

1. **Tarea 1: Inferencia Operativa Diaria en Tiempo Real (`src/ingestion/ingest_meteogalicia.py` & `scripts/run_daily_inference.py`)**
   - Automatización de la ingesta matutina de MeteoGalicia (07:00 AM), aplicación de *Quantile Mapping* e inferencia rápida con LightGBM para predecir las 30.697 celdas de Galicia para el día de hoy.

2. **Tarea 2: Dashboard Interactivo en Streamlit (`app.py`)**
   - Desarrollo de la aplicación web interactiva en **Streamlit** con mapas de riesgo en alta definición (PyDeck/Folium), selector de fechas, buscador por municipio/comarca y alertas urgentes.

3. **Tarea 3: Módulo de Explicabilidad Local y Global con Valores SHAP (`src/models/explainability.py`)**
   - Integración de `shap.TreeExplainer` para descomponer en el dashboard el porqué exacto de la alerta en cada celda seleccionada (ej. *"Riesgo alto por $RH_{min} = 18\%$ y 30 días sin lluvia"*).

4. **Tarea 4: Redacción Final y Maquetación de la Memoria del TFM (`docs/tfm_borrador_memoria.md`)**
   - Finalización de los Capítulos 6 (Inferencia y Web App), 7 (Discusión y Conclusiones) y compilación del documento PDF final para la defensa.

---

## 7. 🔗 Origen de los Datos y Arquitectura del Pipeline Final de Producción

### 7.1 Origen Extractivo de Capas de Información (Fase Actual)

| Capa de Información | Fuente de Origen | Formato Raw | Uso en el Proyecto |
| :--- | :--- | :--- | :--- |
| **Ground Truth (Target)** | EGIF del MITECO (2019-2023) | XML (`fire_history.xml`) | Variable objetivo $Y \in \{0, 1\}$ (Día 0 de Ignición). |
| **Meteorología Histórica** | Copernicus ERA5-Land | NetCDF4 (`.nc`) a 9 km | Predictores meteorológicos $T-1$ y memorias 7d/30d. |
| **Elevación y Topografía** | Copernicus DEM (EU-DEM v1.1) | GeoTIFF (`.tif`) a 25 m | Altitud, pendiente y orientación por celda de 1 km. |
| **Uso del Suelo / Biomasa** | CORINE Land Cover (2012/18/24) | GeoTIFF (`.tif`) | Porcentaje de masa forestal inflamable (`combustible_pct_forestal`). |
| **Accesibilidad Humana** | CNIG / IGN (Red Viaria y Urbana) | Vectorial Shapefile (`.shp`) | Distancia euclídea a carreteras y núcleos poblacionales. |
| **Meteorología Operativa** | MeteoGalicia (Modelo WRF) | API JSON / NetCDF a 1-4 km | Previsión diaria a 24h-48h para inferencia en tiempo real. |

---

### 7.2 Diagrama del Pipeline de Producción de Extremo a Extremo

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   1. INGESTA DIARIA OPERATIVA (07:00 AM)                         │
│   Descarga automática de la previsión WRF de MeteoGalicia (API JSON / NetCDF)    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     2. CORRECCIÓN DE SESGO Y PREPROCESADO                        │
│   - Quantile Mapping respecto a la serie climática de entrenamiento ERA5-Land.   │
│   - Generación de desfases T-1 y memorias climáticas acumuladas (7d y 30d).      │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
       ┌─────────────────────────────────┴─────────────────────────────────┐
       ▼                                                                   ▼
┌───────────────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│     3A. VÍA TABULAR 2D (Parquet - 20 Vars)    │ │   3B. VÍA DATACUBO 3D (NetCDF4 - IberFire)    │
│ Matrices de datos con 20 variables principales│ │ Tensores 3D (Tiempo, Y, X) ventana 25x25 km   │
└──────────────────────┬────────────────────────┘ └──────────────────────┬────────────────────────┘
                       │                                                 │
                       ▼                                                 ▼
┌───────────────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│       4A. INFERENCIA MODELO TABULAR GBDT      │ │    4B. INFERENCIA RED NEURONAL 3D CNN-LSTM    │
│  LightGBM Optimizado (Inferencia en Segundos) │ │  Extracción Espacio-Temporal SOTA (PyTorch)   │
└──────────────────────┬────────────────────────┘ └──────────────────────┬────────────────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      5. MAPA DE PROBABILIDAD DE IGNICIÓN                         │
│           Asignación de probabilidad de riesgo a las 30.697 celdas de Galicia     │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   6. DASHBOARD INTERACTIVO Y EXPLICABILIDAD                      │
│   - Aplicación Web en Streamlit con visores PyDeck/Folium.                        │
│   - Descomposición de factores de riesgo con valores SHAP por celda seleccionada.│
└──────────────────────────────────────────────────────────────────────────────────┘
```

