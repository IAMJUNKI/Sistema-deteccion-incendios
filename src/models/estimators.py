"""Constructores de los modelos candidatos y de sus preprocesados.

Cada familia de modelos se envuelve tras la misma interfaz mínima (`fit` / `predict_proba`)
para que el orquestador no tenga que saber con cuál está trabajando. Las diferencias reales
entre familias se resuelven aquí:

- **LightGBM y XGBoost** consumen las variables categóricas de forma nativa y gestionan los NaN
  internamente. No necesitan preprocesado.
- **Regresión Logística y Random Forest** no admiten ni categóricas ni NaN, así que van dentro
  de un `Pipeline` con imputación y codificación *one-hot*.

Las importaciones de LightGBM y XGBoost son perezosas: el módulo se puede importar aunque esas
librerías no estén instaladas, y solo falla si se pide explícitamente ese modelo.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.dataset import CATEGORICAL_COLS

logger = logging.getLogger(__name__)

#: Modelos que aceptan categóricas y NaN sin preprocesado.
NATIVE_MODELS = {"lightgbm", "xgboost"}


def _one_hot_encoder() -> OneHotEncoder:
    """Crea un OneHotEncoder compatible con distintas versiones de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _sklearn_pipeline(estimator: Any, feature_cols: list[str]) -> Pipeline:
    """Envuelve un estimador de scikit-learn con imputación, escalado y one-hot.

    Args:
        estimator: Clasificador final.
        feature_cols: Columnas que recibirá el pipeline, en orden.

    Returns:
        Pipeline listo para `fit`.
    """
    categoricas = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numericas = [c for c in feature_cols if c not in categoricas]

    preproceso = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                numericas,
            ),
            ("cat", _one_hot_encoder(), categoricas),
        ],
        remainder="drop",
    )
    return Pipeline([("preproceso", preproceso), ("modelo", estimator)])


def build_model(name: str, params: dict, feature_cols: list[str], seed: int = 42) -> Any:
    """Instancia un modelo por nombre.

    Args:
        name: Uno de `lightgbm`, `xgboost`, `logistic_regression`, `random_forest`.
        params: Hiperparámetros del fichero de configuración.
        feature_cols: Columnas de entrada, necesarias para montar el preprocesado.
        seed: Semilla.

    Returns:
        Estimador con interfaz scikit-learn.

    Raises:
        ValueError: Si el nombre no corresponde a ningún modelo soportado.
        ImportError: Si falta la librería del modelo pedido.
    """
    params = dict(params or {})

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError(
                "LightGBM no está instalado. Ejecuta: pip install lightgbm"
            ) from e
        return LGBMClassifier(random_state=seed, **params)

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "XGBoost no está instalado. Ejecuta: pip install xgboost"
            ) from e
        return XGBClassifier(random_state=seed, **params)

    if name == "logistic_regression":
        return _sklearn_pipeline(LogisticRegression(random_state=seed, **params), feature_cols)

    if name == "random_forest":
        return _sklearn_pipeline(RandomForestClassifier(random_state=seed, **params), feature_cols)

    raise ValueError(
        f"Modelo desconocido: {name!r}. Opciones: lightgbm, xgboost, "
        "logistic_regression, random_forest."
    )


def fit_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: Optional[pd.DataFrame] = None,
    y_eval: Optional[pd.Series] = None,
    early_stopping_rounds: Optional[int] = None,
) -> tuple[Any, Optional[int]]:
    """Entrena un modelo, con parada temprana si el modelo la soporta y se aporta conjunto.

    Sin parada temprana, los modelos de boosting agotan siempre todos los árboles configurados,
    lo que con 840 positivos lleva directo al sobreajuste. El conjunto de parada procede del
    **año de validación**, nunca del de entrenamiento: elegir el número de árboles es una
    decisión de selección de modelo y le corresponde a la validación.

    Los baselines de scikit-learn no admiten parada temprana y se entrenan sin más.

    Args:
        model: Estimador sin entrenar.
        model_name: Nombre del modelo.
        X_train: Matriz de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_eval: Matriz del conjunto de parada temprana.
        y_eval: Etiquetas del conjunto de parada temprana.
        early_stopping_rounds: Rondas sin mejora antes de parar. `None` lo desactiva.

    Returns:
        Tupla (modelo entrenado, mejor iteración o `None` si no hubo parada temprana).
    """
    usa_parada = (
        early_stopping_rounds
        and X_eval is not None
        and y_eval is not None
        and model_name in NATIVE_MODELS
    )

    if not usa_parada:
        model.fit(X_train, y_train)
        return model, None

    if model_name == "lightgbm":
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        return model, getattr(model, "best_iteration_", None)

    # XGBoost >= 2.0 recibe la parada temprana en el constructor, no en `fit`.
    model.set_params(early_stopping_rounds=early_stopping_rounds, eval_metric="aucpr")
    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    return model, getattr(model, "best_iteration", None)


def prepare_features(df: pd.DataFrame, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    """Extrae y adapta la matriz de features al modelo que va a consumirla.

    Args:
        df: Bloque preparado por `src.models.dataset`.
        feature_cols: Columnas explicativas.
        model_name: Nombre del modelo destino.

    Returns:
        DataFrame con las columnas en el orden esperado y los tipos adecuados.
    """
    X = df[feature_cols].copy()

    if model_name in NATIVE_MODELS:
        # LightGBM y XGBoost necesitan dtype 'category' explícito para tratarlas como tales.
        for col in CATEGORICAL_COLS:
            if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
    else:
        # El one-hot del pipeline trabaja sobre strings, no sobre Categorical.
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype(str)

    return X


def feature_importances(model: Any, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    """Extrae la importancia de variables cuando el modelo la expone.

    Args:
        model: Modelo entrenado.
        feature_cols: Nombres de las variables de entrada.
        model_name: Nombre del modelo.

    Returns:
        DataFrame ordenado por importancia descendente, vacío si el modelo no la proporciona.
    """
    estimador = model.named_steps["modelo"] if isinstance(model, Pipeline) else model

    if hasattr(estimador, "feature_importances_") and model_name in NATIVE_MODELS:
        valores = np.asarray(estimador.feature_importances_, dtype=float)
        if len(valores) == len(feature_cols):
            return (
                pd.DataFrame({"variable": feature_cols, "importancia": valores})
                .sort_values("importancia", ascending=False)
                .reset_index(drop=True)
            )

    logger.debug("El modelo %s no expone importancias alineables con las variables.", model_name)
    return pd.DataFrame(columns=["variable", "importancia"])
