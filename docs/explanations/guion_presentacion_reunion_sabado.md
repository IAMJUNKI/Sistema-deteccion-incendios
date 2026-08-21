# 📄 Guion e Informe Técnico para la Reunión del Sábado

**Proyecto:** Sistema de Predicción Inteligente de Riesgo de Incendios Forestales en Galicia (TFM)  
**Objetivo:** Presentación del repositorio unificado, justificación de decisiones integradas y resultados empíricos.  
**Fecha:** Julio 2026  

---

## 1. 🎯 Resumen Ejecutivo

Se han integrado los avances dispersos en un pipeline modular en `src/`, probado empíricamente sobre la serie histórica completa de Galicia (2019-2023, más de **56 millones de registros**).

---

## 2. 💡 Justificación de Decisiones Metodológicas Integradas

### 🔹 A. Ground Truth del Target: EGIF MITECO vs NASA FIRMS
* **Decisión:** Adoptar la **Estadística General de Incendios Forestales (EGIF del MITECO)** como la fuente oficial para construir el target supervisado ($Y \in \{0, 1\}$).
* **Por qué se hizo:** NASA FIRMS detecta anomalías térmicas por satélite que incluyen falsos positivos (quemas agrícolas autorizadas, emisiones industriales). El EGIF está verificado sobre el terreno por agentes forestales e indica el **día exacto de inicio y las coordenadas del incendio ($>0.1\text{ ha}$)**.
* **Dónde afecta:** Elimina el ruido en el entrenamiento. NASA FIRMS (2024+) se preserva únicamente para validar inferencias diarias en tiempo casi-real.

### 🔹 B. Ignición en Día 0 vs Fuego Activo
* **Por qué se hizo:** Para desplegar patrullas a las 08:00 AM, el sistema debe predecir la vulnerabilidad del terreno **antes de que salte la chispa**. Clasificar como positivos ($Y=1$) todos los días que un incendio arde continuadamente introduce una fuga de datos (*temporal leakage*) donde el modelo memoriza la regla trivial de *"si ayer había fuego, hoy sigue habiendo riesgo"*.

### 🔹 C. Parsimonia de 20 Variables vs 60+ Clases de Suelo
* **Por qué se hizo:** Frente a desarrollos como IberFire (que incluyen 63 columnas para 44 clases de uso de suelo, de las cuales más de 40 resultan nulas en Galicia), condensamos el mapa en la variable única de **porcentaje efectivo de masa forestal (`combustible_pct_forestal`)**.
* **Evidencia Empírica:** Al añadir características físicas complejas (VPD, Nesterov, CatBoost), el Recall apenas mejoró un $+0.98\%$ a costa de triplicar el cómputo. **Las 20 variables principales capturan la señal física óptima.**

### 🔹 D. Prevención de Data Leakage (Shift T-1 y Rejillas Multi-Temporales)
* **Shift T-1:** Todas las variables meteorológicas explicativas se desplazan 24 horas respecto al día $T$ de evaluación.
* **Rejillas Multi-Temporales:** Se usa el mapa CORINE 2012 para datos de 2013-2018, CORINE 2018 para 2019-2024 y CORINE 2024 para inferencia operativa (2025+).

### 🔹 E. Sesgo Meteorológico (Quantile Mapping ERA5 vs MeteoGalicia)
* Entrenamiento con reanálisis continuo **ERA5-Land** (2019-2023) e inferencia diaria con la previsión numérico **WRF de MeteoGalicia** ajustada mediante **Quantile Mapping**.

### 🔹 F. Filtrado Físico por Extinción Hídrica ($P < 5\text{ mm}$): Dataset Oficial de Trabajo
* **Decisión Metodológica:** Adoptar el **dataset filtrado por precipitación ($P < 5\text{ mm}$)** como el estándar oficial de entrenamiento y evaluación del proyecto.
* **Por qué se hizo:** En días con lluvia moderada/intensa, la humedad del combustible fino supera el umbral físico de extinción ($30\%$), anulando la propagación del fuego. Eliminar estas observaciones descarta **18.370.328 ceros invernales triviales** (el $32.7\%$ del dataset) **preservando el $96.2\%$ de las igniciones reales** ($1.582$ de $1.645$ fuegos).
* **Beneficio Técnico:** Reduce el volumen de datos de 56M a 37M, acelera el entrenamiento al triple de velocidad y fuerza a todos los modelos (LightGBM, XGBoost, PyTorch Conv3D) a evaluar su precisión sobre los días verdaderamente amenazantes.

### 🔹 G. Superioridad Operativa de Nuestro Modelo frente a AEMET / EFFIS (FWI Tradicional)
* **Resolución Local $1\text{ km} \times 1\text{ km}$ (vs $10\text{--}25\text{ km}$ de EFFIS/AEMET):** Los modelos continentales de AEMET/EFFIS operan a resoluciones de $10\text{--}25\text{ km}$ ($625\text{ km²}$ por celda), imposibilitando la localización de patrullas. Nuestro modelo predice a **$1\text{ km²}$ de definición**.
* **Filtro de Falsas Alarmas ($\text{FPR} \le 5\%$ vs $70\text{--}80\%$ de FWI):** En olas de calor, el FWI tradicional pinta el $70\text{--}80\%$ del territorio en alerta roja, anulando la capacidad de respuesta. Nuestro modelo aísla el **$5\%$ de celdas críticas** manteniendo el $95\%$ del mapa verde y limpio.
* **Machine Learning Supervisado Multimodal vs Fórmulas Pasivas:** AEMET calcula índices empíricos pasivos (canadienses de los años 70). Nuestro sistema aprende de los $9.549$ fuegos reales de Galicia (EGIF) combinando vegetación, topografía, accesibilidad humana y memoria climática.

---

## 3. 📊 Benchmark Cuantitativo Oficial de Modelos

Para garantizar transparencia absoluta ante el equipo y el tribunal del TFM, comparamos los modelos bajo dos regímenes de evaluación sobre el conjunto de test ciego del año **2023 completo**:

---

### 3.1 Tabla A: Evaluación sobre el Dataset Completo (11.204.405 Filas de Test)
*Incluye todos los días del año (días secos + días invernales de lluvia pesada).*

| Enfoque / Modelo | PR-AUC | ROC-AUC | Recall @ $FPR \le 5\%$ | Brier Score | Función Operativa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística (L2)** | $0.000027$ | $0.7784$ | $16.67\%$ | $0.1359$ | Baseline Lineal |
| **Random Forest Classifier** | $0.000031$ | $0.8282$ | $21.57\%$ | $0.0199$ | Ensamble Embolsado |
| **XGBoost Classifier** | $0.000049$ | $0.8297$ | $27.45\%$ | $0.0007$ | Gradiente Potenciado |
| **LightGBM Classifier (Ganador Tabular)** | **$0.000071$** | **$0.8377$** | **$32.35\%$** | **$0.0010$** | **Ganador en Alerta Temprana Global** |
| **Red Neuronal 3D Conv3D (PyTorch)** | **$0.179096$** | $0.8125$ | $12.75\%$ | $0.0940$ | **Mayor PR-AUC Local (SOTA 3D)** |

---

### 3.2 Tabla B: Evaluación sobre el Dataset Filtrado ($P < 5\text{ mm}$ - 7.458.210 Filas de Test)
*Evalúa el rendimiento exclusivamente sobre **días secos de riesgo real** (excluyendo días con lluvia $\ge 5\text{ mm}$).*

| Modelo / Técnica | PR-AUC | ROC-AUC | Recall @ $FPR \le 5\%$ | Brier Score | Función Operativa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Standard (Tabular - GANADOR)** | $0.000067$ | $0.7568$ | **`25.74%`** | **`0.0005`** | **Modelo Oficial de Producción** |
| **XGBoost Optimizado (Tabular)** | $0.000052$ | **`0.7585`** | $20.79\%$ | $0.0022$ | Mayor Separación ROC-AUC |
| **CatBoost Classifier (Tabular)** | $0.000040$ | $0.7311$ | $19.80\%$ | $0.0028$ | Baseline Categórico |
| **LightGBM Cost-Sensitive (20x FN)** | $0.000040$ | $0.7629$ | $16.83\%$ | $0.0210$ | Mayor AUROC Global |
| **LightGBM + Calibración Isotónica** | $0.000065$ | $0.7410$ | $23.76\%$ | $0.0010$ | Probabilidades Suavizadas |
| **Red Neuronal 3D Conv3D (PyTorch)** | **`0.1180`** | $0.7109$ | $9.90\%$ | $0.1173$ | **Mayor PR-AUC Local (SOTA 3D)** |

---

### 🔬 ¿Por qué elegimos el Dataset Filtrado ($P < 5\text{ mm}$) como Estándar de Trabajo?

A primera vista, las cifras aparentes del **Dataset Completo** parecen superiores (ROC-AUC de $0.8377$ y Recall del $32.35\%$). Sin embargo, elegir la evaluación filtrada ($P < 5\text{ mm}$) responde a tres razones metodológicas y prácticas de máxima prioridad:

#### 1. El "Efecto Inflación por Preguntas Fáciles" en el ROC-AUC
* **En el Dataset Completo (11.2M filas):** Hay millones de celdas invernales con lluvias torrenciales donde la humedad del combustible vegetal supera el punto físico de extinción ($>30\%$). El modelo predice un riesgo de $0.0000$ con extrema facilidad. Esos aciertos triviales en días húmedos **inflan artificialmente la métrica ROC-AUC hasta $0.8377$**.
* **En el Dataset Filtrado ($P < 5\text{ mm}$):** Eliminamos los ceros regalo de invierno y evaluamos al modelo exclusivamente en el "examen difícil": **los días secos y calurosos**. El ROC-AUC de **$0.7568$** representa la **verdadera capacidad de discriminación del sistema cuando el peligro acecha de verdad**.

#### 2. La Explicación del Presupuesto de Falsas Alarmas en el Recall
* Al evaluar con la restricción **FPR $\le 5\%$** sobre 11.2 millones de filas del Dataset Completo, la tolerancia del $5\%$ permite un presupuesto amplio de **560.000 falsas alarmas**, elevando numéricamente el Recall al $32.35\%$.
* Sobre los 7.4 millones de filas del Dataset Filtrado ($P < 5\text{ mm}$), el presupuesto del $5\%$ se reduce a **370.000 falsas alarmas**, obteniendo un Recall riguroso del **$25.74\%$** sin "trampas" ni inflación de ceros invernales.

#### 3. Valor Operativo en Producción
En un centro de mando de protección civil a las 08:00 AM en un día de tormenta invernal, nadie consulta el mapa de riesgo porque se sabe que es cero por definición física. El usuario real consulta el sistema en **primavera/verano/días secos**. Por ello, el modelo debe estar optimizado y evaluado sobre la distribución del **diferencial de riesgo en días secos ($P < 5\text{ mm}$)**.

---

### 3.3 Tabla de Casos Emblemáticos de Acierto en Alerta Temprana (Verificación Retrospectiva Agosto 2023)

Prueba empírica de cómo el modelo predijo con 24 horas de antelación ($T-1$) la vulnerabilidad extrema en las celdas donde ocurrieron incendios reales:

| Fecha de Ignición | Celda Afectada | Percentil de Riesgo | Probabilidad Predicha | Nivel de Alerta Asignado |
| :--- | :--- | :--- | :--- | :--- |
| **2023-08-23** | **Celda 6818** | **`99.5%`** | **`21.62%`** | **🔥 Top 0.5% Riesgo Extremo** |
| **2023-08-30** | **Celda 29366** | **`97.9%`** | **`4.82%`** | **🚨 Alerta Urgente (Top 2%)** |
| **2023-08-30** | **Celda 29481** | **`95.8%`** | **`3.34%`** | **🚨 Alerta Urgente (Top 4%)** |
| **2023-08-30** | **Celda 29365** | **`95.4%`** | **`3.15%`** | **🚨 Alerta Urgente (Top 5%)** |
| **2023-08-08** | **Celda 7255** | **`92.8%`** | **`1.73%`** | **🟠 Riesgo Alto (Top 7%)** |
| **2023-08-24** | **Celda 24714** | **`92.0%`** | **`10.02%`** | **🟠 Riesgo Alto (Top 8%)** |
| **2023-08-17** | **Celda 29642** | **`91.1%`** | **`3.36%`** | **🟠 Riesgo Alto (Top 9%)** |

* **Percentil Medio de Riesgo Asignado a Fuegos Reales en Agosto 2023: `84.5%`**

---

## 4. 🚀 Reparto Operativo de Tareas para el Equipo (Fase de Producción)

Habiendo validado que **LightGBM Standard (Tabular 2D)** es el modelo oficial de producción (con un $25.74\%$ de Recall a $FPR \le 5\%$ sobre días secos y un Brier Score de $0.0005$), el reparto de módulos para el equipo se centra en la construcción del sistema operativo:

1. **Pipeline de Inferencia Operativa Diaria (`src/ingestion/ingest_meteogalicia.py` & `scripts/run_daily_inference.py`):**
   - Automatización de la ingesta matutina de la previsión WRF de MeteoGalicia (07:00 AM) y aplicación de *Quantile Mapping* contra la climatología ERA5.
2. **Desarrollo del Dashboard Interactivo en Streamlit (`app.py`):**
   - Construcción de la interfaz web interactiva con mapas de calor de riesgo en alta definición (PyDeck/Folium), selector de fechas y alertas comarcales.
3. **Módulo de Explicabilidad Local y Global con valores SHAP (`src/models/explainability.py`):**
   - Integración de `shap.TreeExplainer` para descomponer en la web el desglose de los factores de riesgo de cada celda seleccionada por el usuario.
4. **Redacción de la Memoria del TFM y Documentación (`docs/tfm_borrador_memoria.md`):**
   - Completar los Capítulos 6 (Inferencia y App Web), 7 (Conclusiones) y maquetación final del documento PDF para la defensa.
