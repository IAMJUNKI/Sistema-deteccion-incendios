# 04. Entrenamiento y Evaluación Comparativa de Modelos Baseline

---

## 1. ¿Qué se ha hecho?
- Implementación de la suite de métricas adaptadas a baja prevalencia en `src/models/metrics.py` (**PR-AUC**, **ROC-AUC**, **Recall a FPR $\le 5\%$**, **Brier Score** y **Regresión Isotónica**).
- Creación del script de entrenamiento y evaluación comparativa en `src/models/train_baseline.py`.
- Evaluación de 4 modelos con dividisión temporal estricta (**Train: 2019–2022**, **Test: 2023**):
  1. **Regresión Logística (L2):** Baseline lineal interpretable.
  2. **Random Forest:** Baseline no lineal de ensamble por embolsamiento.
  3. **XGBoost Classifier:** Algoritmo estado del arte en gradiente potenciado.
  4. **LightGBM Classifier:** Algoritmo de gradiente potenciado optimizado en memoria.
- Aplicación de **Hard Negative Mining (1:50)** en el conjunto de entrenamiento para manejar el desbalanceo del $0.0026\%$.

## 2. ¿Por qué se ha hecho?
- **Establecer la Línea Base del TFM:** Antes de entrenar arquitecturas complejas de Aprendizaje Profundo 3D (Datacubo de Alfonso), es imprescindible disponer de métricas de referencia sólidas en modelos tabulares tradicionales.
- **Validación Temporal Real:** No se utiliza validación cruzada aleatoria (K-Fold tradicional) porque provocaría *Spatial/Temporal Leakage*. Evaluar con un año futuro completo (2023) simula exactamente el comportamiento que tendrá el modelo en producción.
- **Rechazo de Métricas Engañosas:** En desbalanceo severo, un modelo trivial que prediga siempre 0 obtendrá un ROC-AUC elevado sin detectar ningún incendio. La métrica reina es el **PR-AUC**.

## 3. ¿Cómo se ha hecho?
1. Se divide la serie temporal: filas con `year <= 2022` para entrenamiento y `year == 2023` para test.
2. En Train, se filtran todos los ceros no informativos y se submuestrea la clase negativa en proporción $1:50$.
3. Se entrenan los 4 modelos de forma aislada sobre la matriz escalada o cruda.
4. Se predicen las probabilidades continuas sobre el año 2023 completo (sin submuestreo).
5. Se calcula la curva Precision-Recall y el Brier Score para medir la calibración.

## 4. ¿Por qué se han elegido estas tecnologías?
- **Scikit-Learn, XGBoost & LightGBM:** Estándares de la industria en Machine Learning supervisado en Python.
- **Isotonic Regression:** Calibrador no paramétrico que ajusta las probabilidades sesgadas por el submuestreo $1:50$ sin alterar el ordenamiento de riesgo.

## 5. ¿Qué conseguimos con ello?
- **Benchmark Comparativo Oficial Obtenido:** Tabla de métricas reales calculada sobre **56.052.722 filas** evaluando sobre el año **2023 completo (out-of-sample)**:

| Modelo | PR-AUC | ROC-AUC | Recall@FPR $\le 5\%$ | Brier Score |
| :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística (L2)** | $0.000027$ | $0.7784$ | $16.67\%$ | $0.1359$ |
| **Random Forest** | $0.000031$ | $0.8282$ | $21.57\%$ | $0.0199$ |
| **XGBoost Classifier** | $0.000049$ | $0.8297$ | $27.45\%$ | $0.0007$ |
| **LightGBM Classifier (Ganador)** | **$0.000071$** | **$0.8377$** | **$32.35\%$** | **$0.0010$** |

- **Conclusión de Modelado:** LightGBM y XGBoost superan ampliamente a la Regresión Logística. LightGBM detecta el **$32.35\%$ de los incendios reales manteniendo una Tasa de Falsos Positivos por debajo del $5\%$**.
- **Resultados Guardados:** CSV oficial en `docs/technical/baseline_results_2023.csv`.
