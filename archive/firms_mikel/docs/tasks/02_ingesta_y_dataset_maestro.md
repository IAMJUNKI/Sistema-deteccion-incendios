# 02. Ingesta de Datos y Construcción del Dataset Maestro (Tabular y Datacubo)

---

## 1. ¿Qué se ha hecho?
- **Auditoría e Integración de Avances del Equipo:** Evaluación de las aproximaciones de datos en `misc/` (Dataset Tabular 2D de Miquel en `.parquet` y Datacubo 3D de Alfonso en `.nc`).
- **Definición del Target de Ignición Oficial:** Adopción del **EGIF del MITECO (2019–2023)** como fuente oficial de *Ground Truth* para el inicio de incendios forestales ($>0.1$ ha), relegando **NASA FIRMS** a tareas de validación secundaria en tiempo real.
- **Estrategia de Mitigación de Fuga de Datos y Desfase Temporal:** Implementación del *Temporal Shift* de las variables meteorológicas ($T-1$) para predecir el inicio del incendio en el día $T$ mediante las condiciones antecedentes.
- **Tratamiento del Sesgo Meteorológico (ERA5-Land vs MeteoGalicia):** En la primera versión se archiva el forecast bruto y se mide el *domain shift* sin aplicar una corrección no validada. El *Quantile Mapping* queda para una segunda etapa con pares forecast-observación.
- **Estrategia para Desbalanceo Severo:** Selección de **PR-AUC** y **Recall a FPR $\le 5\%$** como métricas principales frente al ROC-AUC tradicional, junto con submuestreo de ceros (*Hard Negative Mining*) y calibración isotónica de probabilidades.

## 2. ¿Por qué se ha hecho?
- **Garantizar el Rigor Metodológico del TFM:** Para comparar de forma académicamente válida la capacidad predictiva de un modelo de Aprendizaje Automático tabular (XGBoost / LightGBM) frente a una arquitectura de Aprendizaje Profundo 3D (CNN/LSTM estilo IberFire), ambos modelos deben nutrirse de la misma definición de variable objetivo y de los mismos periodos temporales.
- **Evitar el Ruido de Anomalías Térmicas:** NASA FIRMS detecta puntos calientes mediante satélites (VIIRS/MODIS), los cuales incluyen naves industriales, reflexiones solares y quemas agrícolas autorizadas. El EGIF incluye únicamente incendios forestales verificados por agentes forestales.
- **Asegurar la Viabilidad Operativa en Galicia:** El modelo debe alimentarse en producción de la previsión de MeteoGalicia. Justificar y mitigar la brecha entre reanálisis (ERA5) y previsión local (MeteoGalicia) constituye una contribución científica clave exigible en un TFM.

## 3. ¿Cómo se ha hecho?
1. **Consolidación del Pipeline de Ingesta:**
   - La meteorología ERA5-Land a 9 km interpolada a la rejilla de 1 km² mediante *downscaling* topográfico (corregida por altitud con el DEM de Copernicus) se empareja espacialmente mediante `cell_id` (EPSG:25829 / EPSG:3035).
   - Los registros de ignición del EGIF se mapean espacialmente asignando a la fecha de inicio ($T$) el valor binario $Y=1$ en la celda correspondiente.
2. **Generación del Dataset Tabular (`.parquet`):**
   - Las filas representan combinaciones `(cell_id, fecha)`.
   - Se aplican ventanas móviles de memoria climática (`prec_acum_7d`, `prec_acum_30d`, `tmax_media_7d`).
   - Se desplaza la matriz de variables meteorológicas un día hacia atrás ($T-1$).
3. **Generación del Datacubo 3D (`.nc`):**
   - Estructuración de tensores en dimensiones `(Tiempo, Y, X, Variables)` conservando la topología espacial contigua para procesadores convolucionales.

## 4. ¿Por qué se han elegido estas tecnologías?
- **Apache Parquet (PyArrow / DuckDB / Pandas):** Formato columnar de alta compresión y velocidad de lectura rápida para algoritmos en árbol (XGBoost / LightGBM / CatBoost).
- **NetCDF4 / Xarray (`.nc`):** Estándar internacional en ciencias de la Tierra para la manipulación de tensores multidimensionales geoespaciales, compatible con PyTorch y TensorFlow.
- **Isotonic Regression (Scikit-Learn):** Calibración de las probabilidades del modelo de incendios por horizonte. La corrección meteorológica entre ERA5 y MeteoGalicia no se activa sin un archivo histórico de pares.

## 5. ¿Qué conseguimos con ello?
- **Unificación y Coordinación del Equipo:** Un criterio único y compartido para Miquel, Alfonso, Raúl y Diego que elimina la duplicidad de esfuerzos antes de la reunión del sábado.
- **Avance al 50% de la Memoria del TFM:** Redacción de la justificación metodológica del Target, el manejo del desbalanceo y el abordaje del *Domain Shift* en los capítulos correspondientes de la memoria.
- **Preparación de la Fase 4 (Modelado):** Infraestructura de datos robusta, reproducible y libre de fuga de datos temporales lista para ejecutar baselines.
