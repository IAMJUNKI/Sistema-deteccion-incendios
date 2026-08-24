"""Experimentos tabulares reproducibles para selección de variables."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.modeling.evaluation import evaluate_binary_predictions


def fit_baseline_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictors: Iterable[str],
    target: str = "target_ignicion",
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compara cuatro baselines con idéntico corte temporal y variables."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    predictors = list(predictors)
    ratio = float((train[target] == 0).sum() / (train[target] == 1).sum())
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            scale_pos_weight=ratio,
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(train[predictors], train[target])
        probabilities = model.predict_proba(validation[predictors])[:, 1]
        fitted[name] = model
        rows.append(
            {"model": name, **evaluate_binary_predictions(validation[target], probabilities)}
        )
    return fitted, pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def fit_lightgbm_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictors: Iterable[str],
    target: str = "target_ignicion",
    seed: int = 42,
) -> tuple[object, dict[str, float]]:
    """Entrena un baseline LightGBM; nunca usa el test ciego de 2023."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError("Instala lightgbm en el entorno del proyecto.") from error

    predictors = list(predictors)
    ratio = (train[target] == 0).sum() / (train[target] == 1).sum()
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        scale_pos_weight=float(ratio),
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[predictors], train[target])
    probabilities = model.predict_proba(validation[predictors])[:, 1]
    return model, evaluate_binary_predictions(
        validation[target].to_numpy(dtype=np.uint8), probabilities
    )
