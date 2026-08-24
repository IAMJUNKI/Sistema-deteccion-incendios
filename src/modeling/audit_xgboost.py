"""Auditoría persistente de XGBoost para poder revisar resultados fuera de Jupyter."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.modeling.data import (
    TRAIN_YEARS,
    VALIDATION_YEARS,
    load_dataset_contract,
    sample_years_for_training,
)
from src.modeling.features import resolve_feature_set


def run(output_dir: str | Path = "outputs/modeling/xgboost_audit") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_dataset_contract()
    features = resolve_feature_set(contract.predictors, "temporal_compacto")
    train = sample_years_for_training(contract, TRAIN_YEARS, contract.predictors, 100, 0)
    validation = sample_years_for_training(contract, VALIDATION_YEARS, contract.predictors, 100, 0)
    ratio = (train.target_ignicion == 0).sum() / (train.target_ignicion == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    ).fit(train[features], train.target_ignicion)
    p = model.predict_proba(validation[features])[:, 1]
    y = validation.target_ignicion.to_numpy()
    importance = {name: float(value) for name, value in zip(features, model.feature_importances_, strict=True)}
    result = {
        "train_years": list(TRAIN_YEARS),
        "validation_years": list(VALIDATION_YEARS),
        "test_years_not_used": [2023],
        "n_features": len(features),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "top_1pct_recall": float(y[p >= np.quantile(p, 0.99)].sum() / y.sum()),
        "importance_gain": importance,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([x[0] for x in top][::-1], [x[1] for x in top][::-1])
    ax.set(title="XGBoost: importancia por gain")
    fig.tight_layout()
    fig.savefig(output_dir / "importance_gain.png", dpi=160)
    plt.close(fig)
    return result
