"""Constructores de los modelos candidatos y de su preprocesado.

Cada familia se envuelve tras la misma interfaz mínima (`fit` / `predict_proba`) para que el
orquestador no tenga que saber con cuál está trabajando. Las diferencias reales se resuelven
aquí: LightGBM y XGBoost gestionan los NaN internamente y no necesitan preprocesado; la
regresión logística y Random Forest van dentro de un `Pipeline` con imputación y escalado.

Respecto de la versión sobre FIRMS ha desaparecido todo el manejo de variables categóricas.
El datacubo EGIF no tiene ninguna: la clase de combustible es ahora nueve fracciones CORINE
continuas y la orientación son ocho fracciones de exposición. Es una mejora —una fracción
distingue una celda mitad matorral mitad bosque, y una etiqueta no—, y de paso elimina el
`OneHotEncoder`, la lista de categorías fijas y la sincronización de dtypes entre train,
validación y test, que era una fuente recurrente de errores silenciosos.

Las importaciones de LightGBM y XGBoost son perezosas: el módulo se importa aunque esas
librerías no estén instaladas y solo falla si se pide explícitamente ese modelo.

## Por qué estos cuatro

- **LightGBM** es el candidato operativo: histogramas, escala a decenas de millones de filas.
- **XGBoost** es un contraste real, no una variación: crece los árboles por niveles en vez de
  por hojas y regulariza distinto. Si ambos coinciden, la conclusión no depende del algoritmo.
- **Regresión logística** es la referencia interpretable obligatoria: justifica —o no— el salto
  a los árboles, y sus coeficientes se leen directamente ante un tribunal.
- **Random Forest** es la referencia de *bagging* frente a *boosting*, y documenta con su propio
  sobreajuste por qué hace falta regularización con parada temprana.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Modelos que digieren NaN sin preprocesado y admiten parada temprana.
NATIVOS = frozenset({"lightgbm", "xgboost"})

MODELOS_SOPORTADOS = ("lightgbm", "xgboost", "logistic_regression", "random_forest")


def _pipeline_sklearn(estimador: Any) -> Pipeline:
    """Envuelve un estimador de scikit-learn con imputación por mediana y escalado.

    El escalado es imprescindible para la regresión logística —sin él, la altitud en metros
    domina numéricamente a una fracción de cobertura entre 0 y 1— e inocuo para Random Forest,
    así que se aplica a ambos por simetría: cualquier diferencia entre ellos será del modelo y
    no del preprocesado.
    """
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("modelo", estimador),
    ])


def construir(nombre: str, params: Optional[dict] = None, semilla: int = 42) -> Any:
    """Instancia un modelo por nombre.

    Raises:
        ValueError: Si el nombre no corresponde a ningún modelo soportado.
        ImportError: Si falta la librería del modelo pedido.
    """
    params = dict(params or {})

    if nombre == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError("LightGBM no está instalado: pip install lightgbm") from e
        return LGBMClassifier(random_state=semilla, **params)

    if nombre == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("XGBoost no está instalado: pip install xgboost") from e
        return XGBClassifier(random_state=semilla, **params)

    if nombre == "logistic_regression":
        return _pipeline_sklearn(LogisticRegression(random_state=semilla, **params))

    if nombre == "random_forest":
        return _pipeline_sklearn(RandomForestClassifier(random_state=semilla, **params))

    raise ValueError(
        f"Modelo desconocido: {nombre!r}. Opciones: {', '.join(MODELOS_SOPORTADOS)}."
    )


def entrenar(
    modelo: Any,
    nombre: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_parada: Optional[pd.DataFrame] = None,
    y_parada: Optional[pd.Series] = None,
    rondas_parada: Optional[int] = None,
) -> tuple[Any, Optional[int]]:
    """Entrena, con parada temprana si el modelo la admite y se aporta conjunto.

    Sin parada temprana los modelos de boosting agotan siempre todos los árboles configurados,
    lo que lleva directo al sobreajuste. El conjunto de parada procede **del año de validación**,
    nunca del de entrenamiento: elegir el número de árboles es una decisión de selección de
    modelo y le corresponde a la validación.

    Returns:
        Tupla (modelo entrenado, mejor iteración o `None` si no hubo parada temprana).
    """
    usa_parada = bool(
        rondas_parada and X_parada is not None and y_parada is not None and nombre in NATIVOS
    )

    if not usa_parada:
        modelo.fit(X_train, y_train)
        return modelo, None

    if nombre == "lightgbm":
        import lightgbm as lgb

        modelo.fit(
            X_train, y_train,
            eval_set=[(X_parada, y_parada)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(rondas_parada, verbose=False), lgb.log_evaluation(0)],
        )
        return modelo, getattr(modelo, "best_iteration_", None)

    # XGBoost >= 2.0 recibe la parada temprana en el constructor, no en `fit`.
    modelo.set_params(early_stopping_rounds=rondas_parada, eval_metric="aucpr")
    modelo.fit(X_train, y_train, eval_set=[(X_parada, y_parada)], verbose=False)
    return modelo, getattr(modelo, "best_iteration", None)


def matriz(marco: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Extrae la matriz de features en el orden esperado.

    Sin categóricas que adaptar, esto es una selección de columnas. Se mantiene como función
    propia porque el orden de las columnas debe ser idéntico entre entrenamiento y predicción:
    LightGBM y XGBoost casan las variables por posición, no por nombre.
    """
    faltan = [v for v in variables if v not in marco.columns]
    if faltan:
        raise KeyError(f"Faltan variables en el lote: {faltan}")
    return marco[variables]


def importancias(modelo: Any, variables: list[str], nombre: str) -> pd.DataFrame:
    """Importancia de variables cuando el modelo la expone, alineada con sus nombres.

    Para la regresión logística se devuelve el valor absoluto del coeficiente, que sobre
    variables escaladas sí es comparable entre sí.
    """
    estimador = modelo.named_steps["modelo"] if isinstance(modelo, Pipeline) else modelo

    valores: Optional[np.ndarray] = None
    if hasattr(estimador, "feature_importances_"):
        valores = np.asarray(estimador.feature_importances_, dtype=float)
    elif hasattr(estimador, "coef_"):
        valores = np.abs(np.asarray(estimador.coef_, dtype=float).ravel())

    if valores is None or len(valores) != len(variables):
        logger.debug("El modelo %s no expone importancias alineables.", nombre)
        return pd.DataFrame(columns=["variable", "importancia"])

    return (
        pd.DataFrame({"variable": variables, "importancia": valores})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
