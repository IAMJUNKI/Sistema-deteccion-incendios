# 05. Optimización Avanzada Tabular, Ingeniería Físico-Climática y Ensamble Blending

---

## 1. ¿Qué se ha hecho?
- Implementación de la ingeniería de características físicas avanzadas en `src/features/advanced_physics.py`:
  - **Vapor Pressure Deficit (VPD):** Mapeo de la deficiencia de presión de vapor en agua ($kPa$) combinando $T_{max}$ y $RH_{min}$.
  - **Índice Nesterov:** Acumulador meteorológico de días secos sin precipitación ($P < 5\text{ mm}$).
  - **Ratio Térmico-Eólico:** Indicador sintético de inflamabilidad eólica $\frac{T_{max} \cdot V_{viento}}{\max(RH_{min}, 5)}$.
- Entrenamiento e integración del algoritmo **CatBoost Classifier** especializado en codificación encadenada de variables categóricas.
- Desarrollo del ensamble por promedio ponderado de rangos (**Rank Averaging Blending**) en `src/models/train_ensemble_advanced.py`, combinando **LightGBM + XGBoost + CatBoost**.
- Evaluación empírica sobre el conjunto de test ciego del año **2023 completo** (11.204.405 observaciones out-of-sample).

## 2. ¿Por qué se ha hecho?
- **Maximizar el Rendimiento Tabular sin Complejidad innecesaria:** Antes de incurrir en la elevada complejidad computacional de las Redes Neuronales espacio-temporales 3D, es imprescindible agotar el potencial de los modelos de gradiente potenciado (GBDT) guiados por conocimiento del dominio forestal.
- **Capturar el Estrés Hídrico Atmosférico Real:** La humedad relativa por sí sola ignora la temperatura del aire; el VPD mide la capacidad evaporativa real sobre la vegetación.
- **Reducir la Varianza de los Modelos (Blending):** El promedio de rangos elimina las distorsiones en las escalas de probabilidad entre algoritmos distintos, mejorando la estabilidad del ordenamiento de riesgo.

## 3. ¿Cómo se ha hecho?
1. Se calcula el VPD diario y el índice Nesterov sobre las matrices antecedente $T-1$.
2. Se realiza la división temporal estricta (Train 2019-2022 / Test 2023) y se aplica Hard Negative Mining (1:50) en entrenamiento.
3. Se entrenan de forma independiente LightGBM, XGBoost y CatBoost con regularización adaptativa.
4. Se transforman las probabilidades predichas en test a percentiles de rango ($0.0$ a $1.0$) y se promedian ponderadamente ($40\%$ LightGBM, $30\%$ XGBoost, $30\%$ CatBoost).

## 4. ¿Por qué se han elegido estas tecnologías?
- **CatBoost Classifier:** Algoritmo estado del arte que previene el sobreajuste mediante *Ordered Boosting* y codifica óptimamente clases de orientación y combustible.
- **Rank Averaging (Percentile Blending):** Técnica de ensamble de alto rendimiento en ciencia de datos que supera a la media aritmética directa cuando los modelos tienen calibraciones de probabilidad distintas.

## 5. ¿Qué conseguimos con ello?
- **Resultados Empíricos Oficiales sobre 56.052.722 filas (Evaluación 2023 Out-of-Sample):**

| Modelo / Ensamble | PR-AUC | ROC-AUC | Recall @ $FPR \le 5\%$ | Brier Score | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Avanzado (VPD + Nesterov)** | $0.000047$ | $0.8264$ | **$33.33\%$** | $0.0017$ | **Máximo Recall a FPR $\le 5\%$** |
| **XGBoost Avanzado** | $0.000043$ | **$0.8327$** | $23.53\%$ | $0.0014$ | Mayor ROC-AUC |
| **CatBoost Classifier** | $0.000040$ | $0.8155$ | $20.59\%$ | $0.0028$ | Baseline Categórico |
| **Ensamble Blending (Rank Averaging)** | $0.000041$ | $0.8299$ | $25.49\%$ | $0.3305$ | Ensamble Ponderado |

- **Conclusión Técnica:** La introducción del VPD y del índice Nesterov en **LightGBM Avanzado** eleva la capacidad de detección temprana a su máximo histórico: **$33.33\%$ de Recall a una Tasa de Falsos Positivos $\le 5\%$** (detectando 1 de cada 3 fuegos reales sobre 11.2 millones de filas de test).
- **Resultados Guardados:** `docs/technical/advanced_ensemble_results_2023.csv`.
