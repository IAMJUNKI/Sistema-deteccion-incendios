# 🧩 Análisis de Arquitectura Dual (Vía Tabular vs Vía Datacubo 3D) y Roadmap Futuro del TFM

**Proyecto:** Sistema de Predicción Inteligente de Riesgo de Incendios Forestales en Galicia (TFM)  
**Fecha:** Julio 2026  

---

## 1. ❓ ¿Tiene sentido conservar las 3 partes del desarrollo o debemos elegir una sola?

**Conservar las tres partes es la decisión metodológica más importante del TFM.**

En un Trabajo Fin de Máster de postgrado, el tribunal no evalúa únicamente que un script funcione, sino el **rigor en la investigación comparativa de alternativas tecnológicas**.

---

### 🔹 Desglose y Rol de Cada Módulo

#### 1️⃣ La Vía Tabular 2D (`src/features/build_tabular_dataset.py`)
* **Qué hace:** Transforma el espacio geoespacial y temporal de Galicia (30.697 celdas $\times$ días) en una matriz bidimensional 2D (`.parquet`), aplicando la memoria climática a 7 y 30 días, el desfase $T-1$ y el Target oficial del EGIF MITECO.
* **Por qué es vital:** Es el cimiento de datos sobre el que se apoya todo el proyecto. Sin esta estructura tabular limpia, no se pueden entrenar algoritmos de árbol rápidos ni extraer baselines.

#### 2️⃣ El Motor de Modelado Clásico (`src/models/train_baseline.py` & `hyperparameter_tuning.py`)
* **Qué hace:** Consume el Parquet y entrena/evalúa la suite de algoritmos tradicionales (Regresión Logística, Random Forest, XGBoost, LightGBM), sintonizando hiperparámetros y optimizando umbrales para maximizar el **Recall a FPR $\le 5\%$**.
* **Por qué es vital:** Establece la **Línea Base (Baseline)** cuantitativa oficial del proyecto. Demuestra de forma empírica cuánto rinde un modelo tabular de gradiente potenciado (alcanzando un **ROC-AUC de 0.8377** con LightGBM).

#### 3️⃣ La Vía Datacubo 3D (`src/features/build_datacube_dataset.py`)
* **Qué hace:** Mantiene la topología espacial real en 3 dimensiones (`Tiempo`, `Y`, `X`) mediante tensores `.nc` (NetCDF4 - formato IberFire), permitiendo que los modelos de Aprendizaje Profundo "vean" la superficie vecina de $25 \times 25$ km alrededor de cada celda.
* **Por qué es vital:** Representa el **Estado del Arte (SOTA)**. Permite entrenar Redes Neuronales Convolucionales 3D (CNN-LSTM) para determinar si la estructura convolucional espacial supera a los modelos tabulares.

---

### 💡 El Argumento de la Comparación Dual para el Tribunal
- Si el TFM presentase únicamente modelos tabulares, el tribunal cuestionaría por qué no se probaron tensores 3D con redes neuronales espacio-temporales como en IberFire.
- Si el TFM presentase únicamente redes neuronales 3D, el tribunal cuestionaría si la enorme complejidad computacional de una Red Neuronal está justificada frente a un LightGBM tabular bien optimizado.

**La combinación de las tres partes genera una Comparación Metodológica Dual (Machine Learning Tabular vs Deep Learning 3D), constituyendo la mayor aportación científica del TFM.**

---

## 2. 🗺️ ¿Qué queda por hacer? Roadmap Futuro del TFM

Con la infraestructura de datos y los baselines finalizados, el proyecto se encuentra al **60% de avance**. Las etapas restantes son:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  FASE 1 - FASE 4A: Infraestructura, Datasets y Baselines (0.8377)       │ ✅ COMPLETADO
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FASE 4B: Entrenamiento de Red Neuronal 3D (IberFire - PyTorch)        │ ⏳ PRÓXIMAS SEMANAS
│  - Construcción de arquitectura Conv3D / Conv2D+LSTM sobre NetCDF.      │
│  - Comparativa formal de métricas (PR-AUC / ROC-AUC) frente a LightGBM.│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FASE 5: Inferencia Operativa Diaria con MeteoGalicia                  │ ⏳ PRÓXIMAS SEMANAS
│  - Ingesta automática de la predicción WRF de MeteoGalicia.            │
│  - Aplicación de Quantile Mapping contra la serie de ERA5-Land.        │
│  - Generación diaria del mapa de riesgo de ignición sobre Galicia.     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FASE 6: Dashboard Interactivo Streamlit y Redacción Final Memoria     │ ⏳ FASE FINAL
│  - Interfaz web operativa Streamlit con visores PyDeck/Folium.         │
│  - Módulo de explicabilidad de decisiones mediante valores SHAP.       │
│  - Ensamblado final del documento PDF de la Memoria del TFM.           │
└────────────────────────────────────────────────────────────────────────┘
```
