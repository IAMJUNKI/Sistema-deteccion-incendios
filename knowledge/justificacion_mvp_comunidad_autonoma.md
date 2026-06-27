# Justificación del Enfoque por Comunidad Autónoma (MVP)
## TFM: Sistema Predictivo para la Anticipación de Incendios Forestales

### 1. El Problema de la Explosión Combinatoria de los Datos

El principal desafío inicial de este Trabajo Fin de Máster (TFM) no radica en la complejidad de los algoritmos de Machine Learning, sino en la **infraestructura y la ingeniería de datos**. El enfoque propuesto utiliza una discretización espacial en celdas de $1\text{ km} \times 1\text{ km}$ y una resolución temporal diaria. 

Si intentamos abordar el proyecto a nivel nacional (España) desde el primer día, nos enfrentamos a las siguientes magnitudes cuantitativas:

* **Superficie de España:** ~505.990 km². Descontando islas y áreas puramente urbanas, obtenemos aproximadamente **500.000 celdas operativas**.
* **Horizonte Temporal Histórico recomendado:** 5 años para capturar de manera estadísticamente significativa la variabilidad climática y los ciclos de sequía.
* **Cálculo del volumen de datos bruto:**
  $$\text{Total Filas} = 500.000 \text{ celdas} \times 365 \text{ días/año} \times 5 \text{ años} = 912.500.000 \text{ registros}$$

Procesar un dataset tabular de casi **1.000 millones de filas**, donde cada fila debe cruzarse mediante operaciones espaciales pesadas (*Spatial Joins* y extracción de rasters NetCDF de Copernicus ERA5), requiere una infraestructura de computación distribuida (clústeres Spark/Dask, instancias de gran memoria en AWS/GCP) y un presupuesto que excede el alcance típico de un desarrollo inicial de TFM. Un error en el diseño de las *features* implicaría días de reprocesamiento.

### 2. Ventajas del Enfoque MVP (Producto Mínimo Viable)

Acotar el alcance inicial a una única Comunidad Autónoma (por ejemplo, **Galicia** o **Andalucía**) transforma radicalmente la viabilidad del proyecto:

| Métrica de Escala | Escala Nacional (España) | Escala Regional (Ej: Galicia) | Beneficio Operativo |
| :--- | :--- | :--- | :--- |
| **Superficie / Celdas** | ~500.000 celdas | ~30.000 celdas | Reducción del 94% del espacio muestral. |
| **Filas Brutas (5 años)**| ~912,5 Millones | ~54,7 Millones | Manejable en entornos locales de desarrollo (16GB/32GB RAM). |
| **Dataset Filtrado*** | ~2-3 Millones de filas | ~150.000 - 200.000 filas | Velocidad de entrenamiento de modelos en minutos, no en horas. |

*\*Nota: Aplicando el submuestreo de negativos difíciles detallado en la metodología.*

#### Beneficios Clave:
1. **Velocidad de Iteración (Agilidad):** Permite equivocaros rápido y barato. Si descubrís que una variable meteorológica (como la velocidad del viento) necesita un retardo (*lag*) de 3 días en lugar de 2, podéis regenerar el dataset y reentrenar el modelo en minutos.
2. **Foco en la Ingeniería de Características:** El 80% del éxito de este modelo depende de cómo se construyan los "negativos difíciles" y los acumulados de lluvia. Trabajar a escala regional permite validar visualmente en mapas (usando QGIS o GeoPandas) si el cruce espacial de los incendios de la NASA FIRMS y la meteorología se está haciendo correctamente sin colapsar el sistema.
3. **Control de la Heterogeneidad Climática y Ecológica:** España posee regímenes climáticos radicalmente opuestos (el clima oceánico y forestación de eucalipto/pino en Galicia vs. el clima mediterráneo continentalizado y matorral xerófilo en Andalucía). Un único modelo nacional inicial podría verse sesgado por las macro-tendencias, diluyendo patrones críticos locales. Un modelo regional aprende de manera mucho más fina las dinámicas de su propio ecosistema.

### 3. Criterios de Selección para la Comunidad Autónoma Piloto

Se recomienda seleccionar la comunidad autónoma en función de tres criterios ponderados:
1. **Densidad de Histórico de Incendios:** Regiones con alta frecuencia de eventos (Galicia, Asturias, Andalucía) garantizan que el modelo tenga suficientes "casos positivos" (clase minoritaria) para aprender el patrón de inicio.
2. **Disponibilidad de Datos Locales Complementarios:** Presencia de servicios de predicción regionales o inventarios forestales detallados que sirvan para contrastar resultados.
3. **Interés del Caso de Uso:** Galicia representa el reto del minifundio forestal y la alta recurrencia; Andalucía representa el reto del cambio climático extremo, olas de calor severas y topografía abrupta.

### 4. Hoja de Ruta para la Escalabilidad Nacional (Triggers de Transición)

El paso de la escala regional a la nacional no requiere reescribir el proyecto, sino **escalar la infraestructura**. Se determinará que el proyecto está listo para pasar a nivel nacional cuando se cumplan los siguientes hitos de validación en el MVP:

```
[MVP Regional Funcional] 
       │
       ▼
[Métrica Objetivo Alcanzada] ──► (Ej: AUC-ROC > 0.85 y F1-Score balanceado)
       │
       ▼
[Pipeline Automatizado]      ──► Código modularizado en funciones/clases parametrizadas (pasando el código de la C.A. como argumento)
       │
       ▼
[Migración de Infraestructura]──► Uso de herramientas de procesamiento eficiente como Polars o DuckDB en local, o Spark en Cloud.
       │
       ▼
[Despliegue Nacional]
```

Al diseñar el código del MVP de forma modular (por ejemplo, donde la geometría de la región sea un parámetro de entrada), la transición a nivel nacional consistirá simplemente en cambiar el archivo de frontera geométrica (GeoJSON/Shapefile) y ejecutar el proceso en una máquina con mayor capacidad de cómputo.
