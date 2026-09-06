#!/usr/bin/env python3
"""Genera perfiles por horizonte de la comparación de familias de modelos.

Se utiliza un gráfico de puntos y líneas porque solo hay tres observaciones por
familia: T+1, T+2 y T+3. Una caja de bigotes con tres observaciones sería poco
informativa y podría confundirse con incertidumbre estadística.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = (
    "egif-2d-48-expanded",
    "egif-2d-48-comparable",
    "egif-2d-50-aligned-control",
    "canonical-egif-2d",
    "fwi-cems-baseline",
)
FAMILY_LABELS = {
    "egif-2d-48-expanded": "EGIF 48\nampliado",
    "egif-2d-48-comparable": "EGIF 48\ncomparable",
    "egif-2d-50-aligned-control": "EGIF 50\ncontrol",
    "canonical-egif-2d": "EGIF 50\nactual",
    "fwi-cems-baseline": "FWI\nCEMS",
}
METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall_at_fpr5": "Recall con FPR = 5 %",
    "recall_at_top1%_daily": "Recall en el 1 % superior diario",
}
COLORS = ("#1769aa", "#2e8b57", "#d97706", "#6b7280", "#8b5cf6")


def load_metric_values(path: Path) -> dict[str, dict[str, list[float]]]:
    """Carga las métricas agrupadas por familia y horizonte."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    values: dict[str, dict[str, list[float]]] = {
        family: {metric: [] for metric in METRIC_LABELS} for family in FAMILY_ORDER
    }
    for report in payload["reports"]:
        family = report["family"]
        horizon = report["horizon_days"]
        # El nombre del experimento distingue las dos familias EGIF 48.
        if family == "egif-2d-48":
            experiment = Path(report["artifact"]).parent.name
            family = f"egif-2d-48-{experiment}"
        if family not in values:
            continue
        for metric in METRIC_LABELS:
            value = report["metrics"].get(metric)
            if value is not None and np.isfinite(value):
                values[family][metric].append(float(value))
        values[family].setdefault("horizons", []).append(horizon)

    for family in FAMILY_ORDER:
        for metric in METRIC_LABELS:
            if len(values[family][metric]) != 3:
                raise ValueError(
                    f"Se esperaban tres horizontes para {family}/{metric}; "
                    f"se encontraron {len(values[family][metric])}."
                )
    return values


def plot_horizon_profiles(values: dict[str, dict[str, list[float]]], output: Path) -> None:
    """Genera y guarda la figura con un punto etiquetado por horizonte."""

    def format_metric(metric: str, value: float) -> str:
        if metric == "pr_auc":
            return f"{value:.1e}"
        if metric == "roc_auc":
            return f"{value:.3f}"
        return f"{value:.0%}"

    horizons = (1, 2, 3)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(top=0.78, bottom=0.13, hspace=0.42, wspace=0.18)
    fig.suptitle(
        "Comparación por horizonte en el test ciego 2023\n"
        "Cada punto corresponde al modelo indicado en el eje X",
        fontsize=15,
        fontweight="bold",
    )

    for axis, metric in zip(axes.flat, METRIC_LABELS):
        for family, color in zip(FAMILY_ORDER, COLORS):
            axis.plot(
                horizons,
                values[family][metric],
                marker="o",
                markersize=5,
                linewidth=2,
                color=color,
                label=FAMILY_LABELS[family].replace("\n", " "),
            )

        axis.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        axis.set_xlabel("Horizonte de predicción")
        axis.set_xticks(horizons, [f"T+{horizon}" for horizon in horizons])
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _, metric=metric: format_metric(metric, value))
        )
        axis.tick_params(axis="x", labelsize=10)
        axis.tick_params(axis="y", labelsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=3, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/models/evaluation/model_family_comparison.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/technical/model_family_horizons.png"),
    )
    args = parser.parse_args()
    plot_horizon_profiles(load_metric_values(args.input), args.output)
    print(f"Figura guardada en {args.output}")


if __name__ == "__main__":
    main()
