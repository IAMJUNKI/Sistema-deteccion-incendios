#!/usr/bin/env python3
"""Genera figuras para la evaluación progresiva del forecast MeteoGalicia.

La primera figura muestra qué combinaciones emisión-horizonte ya pueden
compararse con observaciones cerradas. La segunda muestra los errores
agregados por horizonte y variable. El panel de viento se marca como
diagnóstico porque el forecast y la observación no tienen todavía una ventana
temporal completamente homogénea.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


HORIZONS = (1, 2, 3)
VARIABLES = (
    ("tmax_vc", "Temperatura máxima", "°C"),
    ("rhmin_vc", "Humedad relativa mínima", "puntos porcentuales"),
    ("prec_dia", "Precipitación diaria", "mm"),
    ("vmax_vc", "Viento máximo (diagnóstico)", "km/h"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/evaluation/meteogalicia/meteogalicia_forecast_metrics.json"
        ),
        help="Informe JSON generado por evaluate_meteogalicia_forecasts.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/technical"),
        help="Directorio donde se guardarán las figuras PNG.",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("El informe JSON debe contener un objeto en la raíz.")
    return report


def _coverage_rows(report: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Devuelve fechas de emisión y celdas cerradas por horizonte.

    Los informes nuevos incluyen ``forecast_inventory`` y permiten mostrar
    también emisiones todavía pendientes. Los informes anteriores se pueden
    representar usando únicamente sus casos ya cerrados.
    """

    inventory = report.get("forecast_inventory") or []
    if inventory:
        dates = sorted({str(item["issue_date"]) for item in inventory})
        values = np.full((len(dates), len(HORIZONS)), np.nan, dtype=float)
        date_index = {date: index for index, date in enumerate(dates)}
        for item in inventory:
            row = date_index[str(item["issue_date"])]
            closed_cells = item.get("closed_cells") or {}
            for column, horizon in enumerate(HORIZONS):
                if horizon in item.get("closed_horizons", []):
                    values[row, column] = float(closed_cells.get(str(horizon), np.nan))
        return dates, values

    cases = report.get("cases") or []
    dates = sorted({str(case["issue_date"]) for case in cases})
    values = np.full((len(dates), len(HORIZONS)), np.nan, dtype=float)
    date_index = {date: index for index, date in enumerate(dates)}
    for case in cases:
        row = date_index[str(case["issue_date"])]
        values[row, int(case["horizon_days"]) - 1] = float(case["cells"])
    return dates, values


def plot_coverage(report: dict[str, Any], output: Path) -> None:
    """Genera la matriz de emisiones con comparaciones cerradas."""

    dates, values = _coverage_rows(report)
    if not dates:
        raise ValueError("El informe no contiene emisiones o casos comparables.")

    fig_height = max(3.8, 0.62 * len(dates) + 2.1)
    figure, axis = plt.subplots(figsize=(10.4, fig_height))
    cmap = ListedColormap(["#2563a6"])
    cmap.set_bad("#edf1f4")
    masked = np.ma.masked_invalid(values)
    axis.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(HORIZONS)), [f"T+{horizon}" for horizon in HORIZONS])
    axis.set_yticks(range(len(dates)), dates)
    axis.set_xlabel("Horizonte evaluado")
    axis.set_ylabel("Fecha de emisión")
    axis.set_title(
        "Casos forecast-observación cerrados\n"
        f"{int(np.isfinite(values).sum())} comparaciones disponibles de "
        f"{report.get('forecast_files_found', '—')} forecasts archivados",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    axis.tick_params(axis="both", labelsize=10)
    axis.set_xticks(np.arange(-0.5, len(HORIZONS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(dates), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.spines[:].set_visible(False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{int(value):,}".replace(",", ".") if np.isfinite(value) else "pendiente"
            text_color = "white" if np.isfinite(value) else "#536878"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold" if np.isfinite(value) else "normal",
            )

    figure.text(
        0.5,
        0.015,
        "Cada valor cerrado cubre las celdas disponibles de la rejilla; «pendiente» significa que aún no hay observación posterior.",
        ha="center",
        fontsize=9,
        color="#536878",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _metrics_by_key(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["horizon_days"]), str(row["variable"])): row
        for row in report.get("metrics", [])
    }


def plot_errors(report: dict[str, Any], output: Path) -> None:
    """Genera pequeños múltiplos del MAE agregado por horizonte."""

    metrics = _metrics_by_key(report)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    figure.suptitle(
        "Error agregado del forecast MeteoGalicia por horizonte\n"
        "Caso de estudio preliminar; no son intervalos de confianza",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for axis, (variable, title, unit) in zip(axes.flat, VARIABLES):
        values = [metrics.get((horizon, variable), {}).get("mae") for horizon in HORIZONS]
        numeric = np.array(
            [float(value) if value is not None else np.nan for value in values], dtype=float
        )
        axis.plot(
            HORIZONS,
            numeric,
            color="#2563a6",
            marker="o",
            linewidth=2,
            markersize=6,
        )
        if variable == "vmax_vc":
            axis.set_facecolor("#f4f6f7")
            axis.text(
                0.03,
                0.93,
                "Máximo previsto 12–18 h\nvs máximo observado 24 h",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color="#536878",
            )
        for horizon, value in zip(HORIZONS, numeric):
            if np.isfinite(value):
                axis.annotate(
                    f"{value:.2f}",
                    (horizon, value),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("Horizonte")
        axis.set_ylabel(f"MAE ({unit})")
        axis.set_xticks(HORIZONS, [f"T+{horizon}" for horizon in HORIZONS])
        axis.grid(axis="y", color="#d7e0e6", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)

    figure.text(
        0.5,
        0.015,
        "Los valores son medias sobre celdas emparejadas. El viento se conserva como diagnóstico y no como métrica meteorológica definitiva.",
        ha="center",
        fontsize=9,
        color="#536878",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    report = _load_report(args.input)
    coverage_path = args.output_dir / "meteogalicia_case_coverage.png"
    errors_path = args.output_dir / "meteogalicia_forecast_errors.png"
    plot_coverage(report, coverage_path)
    plot_errors(report, errors_path)
    print(f"Figura guardada en {coverage_path}")
    print(f"Figura guardada en {errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
