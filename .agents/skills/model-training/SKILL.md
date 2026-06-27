---
name: model-training
description: >
  Instrucciones para el entrenamiento, evaluación y calibración de modelos de Machine Learning 
  del proyecto: validación temporal por años completos (prohibido K-Fold aleatorio), métricas 
  orientadas a clases desbalanceadas (AUC-ROC, PR-AUC, F1), calibración isotónica de umbrales 
  de riesgo y serialización de modelos. Activar en la Fase 4 del proyecto.
---

# Skill: Modelado y Calibración — Fase 4

## Contexto

El modelo predice la probabilidad de inicio de incendio para cada celda. El problema tiene **clases fuertemente desbalanceadas** (~1-2% de positivos). La validación debe simular exactamente las condiciones reales: predecir el futuro usando solo datos del pasado.

---

## REGLA CRÍTICA: Validación Temporal

```
❌ PROHIBIDO: K-Fold Cross-Validation aleatorio
✅ OBLIGATORIO: Partición por bloques anuales completos

Entrenamiento:  2019 · 2020 · 2021
Validación:     2022  ← tuning de hiperparámetros aquí
Test (ciego):   2023 · 2024  ← NO tocar hasta tener el modelo final
```

**Por qué:** Si se mezclan días aleatoriamente, el modelo puede ver días muy cercanos a un incendio tanto en entrenamiento como en validación → métricas infladas → sistema que falla en producción.

---

## 1. Partición Temporal de Datos

```python
import pandas as pd
from pathlib import Path

def partir_dataset_temporal(df: pd.DataFrame, 
                              col_fecha: str = "fecha") -> tuple:
    """Divide el dataset en train/val/test por años completos.
    
    Returns:
        Tuple de (df_train, df_val, df_test).
    """
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    año = df[col_fecha].dt.year
    
    df_train = df[año.isin([2019, 2020, 2021])].copy()
    df_val   = df[año == 2022].copy()
    df_test  = df[año.isin([2023, 2024])].copy()
    
    print(f"Train:  {len(df_train):>8,} filas | Positivos: {df_train['target'].sum():>5,} ({df_train['target'].mean()*100:.2f}%)")
    print(f"Val:    {len(df_val):>8,} filas | Positivos: {df_val['target'].sum():>5,} ({df_val['target'].mean()*100:.2f}%)")
    print(f"Test:   {len(df_test):>8,} filas | Positivos: {df_test['target'].sum():>5,} ({df_test['target'].mean()*100:.2f}%)")
    
    return df_train, df_val, df_test

FEATURES = [
    "altitud_media", "pendiente_media", "orientacion_media", "combustible_clase",
    "temp_max_12_18h", "humedad_min_12_18h", "viento_max_kmh_12_18h",
    "precip_acum_1d", "precip_acum_3d", "precip_acum_7d", "precip_acum_14d", "precip_acum_30d",
    "dias_sin_lluvia",
    "n_incendios_celda_3años", "n_incendios_vecindario_10km_3años",
    "dist_carretera_m", "dist_nucleo_urbano_m",
    "mes_sin", "mes_cos", "dia_sin", "dia_cos",
]
TARGET = "target"

df_train, df_val, df_test = partir_dataset_temporal(df_maestro)
X_train, y_train = df_train[FEATURES], df_train[TARGET]
X_val, y_val = df_val[FEATURES], df_val[TARGET]
X_test, y_test = df_test[FEATURES], df_test[TARGET]
```

---

## 2. Baseline: Regresión Logística

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",  # Penaliza errores en clase minoritaria
        max_iter=1000,
        random_state=42
    ))
])

baseline.fit(X_train, y_train)
evaluar_modelo(baseline, X_val, y_val, nombre="Baseline Logística")
```

---

## 3. Modelo Principal: XGBoost / LightGBM

```python
import xgboost as xgb
import lightgbm as lgb

# Ratio de desbalanceo para penalizar falsos negativos
ratio_desbalanceo = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Ratio negativos/positivos: {ratio_desbalanceo:.1f}:1")

# XGBoost
modelo_xgb = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio_desbalanceo,  # Clave para desbalanceo
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",                 # Optimizar PR-AUC
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

modelo_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

# LightGBM (alternativa más rápida)
modelo_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    is_unbalance=True,                  # Manejo automático del desbalanceo
    metric="average_precision",
    early_stopping_round=50,
    random_state=42,
    n_jobs=-1,
)

modelo_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.log_evaluation(100)],
)
```

---

## 4. Métricas de Evaluación

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

def evaluar_modelo(modelo, X: pd.DataFrame, y: pd.Series, nombre: str) -> dict:
    """Evalúa un modelo con las métricas principales del proyecto.
    
    Returns:
        Diccionario con todas las métricas calculadas.
    """
    y_proba = modelo.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    auc_roc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    
    print(f"\n{'='*50}")
    print(f"  Modelo: {nombre}")
    print(f"{'='*50}")
    print(f"  AUC-ROC:  {auc_roc:.4f}  (objetivo: > 0.85)")
    print(f"  PR-AUC:   {pr_auc:.4f}  (clave para desbalanceo)")
    print(f"\n{classification_report(y, y_pred, target_names=['No incendio', 'Incendio'])}")
    
    return {"auc_roc": auc_roc, "pr_auc": pr_auc, "modelo": nombre}

# Evaluar en validación (nunca en test hasta el modelo final)
metricas_val = evaluar_modelo(modelo_xgb, X_val, y_val, "XGBoost — Validación 2022")
```

### Interpretación de métricas

| Métrica | Por qué importa en este proyecto |
|---|---|
| **AUC-ROC** | Capacidad discriminativa general. Objetivo: > 0.85 |
| **PR-AUC** | Fundamental con clases desbalanceadas: mide precisión en los incendios reales |
| **Recall** | Prioridad alta: minimizar falsos negativos (incendios no detectados) |
| **Precisión** | Controla las falsas alarmas (alertas innecesarias que saturan al operativo) |

---

## 5. Calibración de Umbrales de Riesgo

Las probabilidades crudas del modelo (ej: `0.023`) no son intuitivas para protección civil. Calibrar umbrales basándose en la **tasa real de incidencia** del conjunto de validación.

```python
import numpy as np
import json

def calibrar_umbrales(modelo, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Calibra los umbrales de riesgo basándose en la tasa de incidencia real.
    
    Metodología:
        - Dividir las predicciones en percentiles.
        - Calcular la tasa real de incendios en cada percentil.
        - Definir umbrales donde la tasa se dispara.
    
    Returns:
        Diccionario con umbrales calibrados para cada nivel de riesgo.
    """
    y_proba = modelo.predict_proba(X_val)[:, 1]
    
    df_cal = pd.DataFrame({"proba": y_proba, "real": y_val.values})
    df_cal = df_cal.sort_values("proba")
    
    # Calcular tasa de incendios en bins de probabilidad
    n_bins = 100
    df_cal["bin"] = pd.qcut(df_cal["proba"], n_bins, labels=False, duplicates="drop")
    
    tasa_por_bin = df_cal.groupby("bin").agg(
        proba_media=("proba", "mean"),
        tasa_incendio=("real", "mean"),
        n_muestras=("real", "count")
    ).reset_index()
    
    print("\n📊 Análisis de calibración:")
    print(tasa_por_bin[tasa_por_bin["tasa_incendio"] > 0.01].to_string())
    
    # Definir umbrales basándose en el análisis visual/cuantitativo
    # ESTOS VALORES SE AJUSTAN SEGÚN LOS RESULTADOS REALES DEL MODELO
    umbrales = {
        "bajo": {"proba_min": 0.0, "proba_max": None},      # < umbral_moderado
        "moderado": {"proba_min": None, "proba_max": None},  # Ajustar con datos
        "alto": {"proba_min": None, "proba_max": None},      # Ajustar con datos
        "extremo": {"proba_min": None, "proba_max": 1.0},   # > umbral_alto
        "metadata": {
            "modelo": type(modelo).__name__,
            "fecha_calibracion": pd.Timestamp.now().isoformat(),
            "n_muestras_val": len(y_val),
            "pr_auc_val": average_precision_score(y_val, y_proba),
        }
    }
    
    return umbrales, tasa_por_bin

umbrales, df_tasas = calibrar_umbrales(modelo_xgb, X_val, y_val)

# Guardar umbrales calibrados
with open("data/models/umbrales_calibrados_v1.json", "w") as f:
    json.dump(umbrales, f, indent=2, default=str)
```

---

## 6. Importancia de Variables (SHAP)

```python
import shap

def calcular_shap(modelo, X: pd.DataFrame, n_muestras: int = 1000) -> None:
    """Calcula y visualiza los valores SHAP del modelo.
    
    Args:
        n_muestras: Número de muestras para el análisis (submuestreo para velocidad).
    """
    X_sample = X.sample(min(n_muestras, len(X)), random_state=42)
    
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_sample)
    
    # Gráfico de importancia global
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Importancia de Variables — SHAP (valores medios)")
    plt.tight_layout()
    plt.savefig("docs/shap_importancia_global.png", dpi=150)
    
    # Gráfico de distribución de impacto
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("docs/shap_distribucion.png", dpi=150)
```

---

## 7. Serialización del Modelo

```python
import joblib
import json

# Guardar modelo XGBoost (formato nativo JSON, preferible a pickle)
modelo_xgb.save_model("data/models/modelo_xgboost_v1.json")

# Guardar pipeline completo con joblib (si incluye preprocesado)
joblib.dump(pipeline_completo, "data/models/pipeline_v1.pkl")

# Verificar que el modelo se carga correctamente
modelo_cargado = xgb.XGBClassifier()
modelo_cargado.load_model("data/models/modelo_xgboost_v1.json")
assert modelo_cargado.predict_proba(X_val[:5]).shape == (5, 2)
print("✅ Modelo serializado y cargado correctamente")
```

---

## Checklist de Calidad — Fase 4

- [ ] La partición temporal es correcta: ningún dato de 2022+ en train, ningún dato de 2023+ en val.
- [ ] AUC-ROC en validación > 0.80 (objetivo > 0.85).
- [ ] PR-AUC en validación documentado.
- [ ] Los hiperparámetros finales se fijaron con el conjunto de validación (2022), NO con test.
- [ ] El test ciego (2023-2024) se evaluó solo una vez al final.
- [ ] Los umbrales de riesgo están calibrados con tasas reales de incidencia.
- [ ] El modelo serializado se carga y produce predicciones correctas.
- [ ] Los valores SHAP están calculados y guardados en `docs/`.
