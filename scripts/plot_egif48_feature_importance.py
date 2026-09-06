#!/usr/bin/env python3
"""Genera la importancia gain de las variables EGIF 48 por horizonte.

La importancia gain es una medida interna del árbol: indica cuánto reduce una
variable la función de pérdida en los splits del modelo. No es una medida de
causalidad ni sustituye un estudio de permutación o una explicación SHAP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = (1, 2, 3)
COLORS = ("#1769aa", "#2e8b57", "#d97706")


def load_importance(model_dir: Path) -> pd.DataFrame:
    """Carga y normaliza la importancia gain de los tres modelos EGIF 48."""

    frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        path = model_dir / f"forecast_risk_egif_48_t{horizon}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No existe el artefacto EGIF 48: {path}")
        artifact = joblib.load(path)
        gains = artifact.base_model.booster_.feature_importance(importance_type="gain")
        total = float(np.sum(gains))
        if total <= 0:
            raise ValueError(f"El modelo T+{horizon} no tiene importancia gain positiva.")
        frames.append(
            pd.DataFrame(
                {
                    "feature": artifact.feature_columns,
                    "horizon": f"T+{horizon}",
                    "importance": gains / total,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_importance(frame: pd.DataFrame, output: Path, top_n: int = 12) -> None:
    """Guarda un gráfico de barras agrupadas con las variables más importantes."""

    mean_importance = (
        frame.groupby("feature", as_index=False)["importance"].mean()
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    selected = mean_importance["feature"].tolist()[::-1]
    plot_frame = frame[frame["feature"].isin(selected)].copy()
    x = np.arange(len(selected), dtype=float)
    width = 0.25

    fig, axis = plt.subplots(figsize=(12, 8))
    for position, (horizon, color) in enumerate(zip(("T+1", "T+2", "T+3"), COLORS)):
        values = (
            plot_frame[plot_frame["horizon"] == horizon]
            .set_index("feature")
            .reindex(selected)["importance"]
            .to_numpy()
        )
        axis.barh(x + (position - 1) * width, values, height=width * 0.9, color=color, label=horizon)

    axis.set_yticks(x, [feature.replace("_", " ") for feature in selected])
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.set_xlabel("Importancia gain normalizada")
    axis.set_title(
        "Principales variables del modelo EGIF 48 ampliado\n"
        "Importancia interna del LightGBM por horizonte",
        fontsize=15,
        fontweight="bold",
    )
    axis.legend(title="Modelo", ncol=3, loc="lower right")
    axis.grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=9)
    axis.tick_params(axis="x", labelsize=9)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.84, bottom=0.11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/experiments/expanded"))
    parser.add_argument("--output", type=Path, default=Path("docs/technical/egif48_feature_importance.png"))
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    plot_importance(load_importance(args.model_dir), args.output, top_n=args.top_n)
    print(f"Figura guardada en {args.output}")


if __name__ == "__main__":
    main()
