#!/usr/bin/env python3
"""Genera el informe PDF de la auditoría EGIF-48 sobre 2023."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "technical" / "egif48_test_2023_audit.json"
OUTPUT = ROOT / "output" / "pdf" / "informe_auditoria_egif48_2023.pdf"
TMP = ROOT / "tmp" / "pdfs" / "egif48_audit"

NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#1F7A8C")
ORANGE = colors.HexColor("#E76F51")
GREEN = colors.HexColor("#2A9D8F")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GREY = colors.HexColor("#F3F5F7")
MID_GREY = colors.HexColor("#6B7280")
TEXT = colors.HexColor("#25313C")


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _pct(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def _pp(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.{digits}f} pp"


def _text(value: object) -> str:
    return escape(str(value))


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _table(data: list[list[object]], widths: list[float] | None = None, header: bool = True) -> Table:
    converted = []
    for row_index, row in enumerate(data):
        converted.append(
            [
                item
                if isinstance(item, Paragraph)
                else _p(_text(item), STYLES["table_header"] if header and row_index == 0 else STYLES["table"])
                for item in row
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8DEE4")),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
        start = 1
    else:
        start = 0
    for row_index in range(start, len(data)):
        if (row_index - start) % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GREY))
    table.setStyle(TableStyle(commands))
    return table


def _callout(title: str, body: str, color: colors.Color = TEAL) -> Table:
    content = [
        _p(f"<b>{_text(title)}</b>", STYLES["callout_title"]),
        _p(body, STYLES["callout_body"]),
    ]
    table = Table([[content]], colWidths=[17.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.Color(color.red, color.green, color.blue, alpha=0.08)),
                ("BOX", (0, 0), (-1, -1), 1.0, color),
                ("LINEBEFORE", (0, 0), (0, -1), 5, color),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=MID_GREY,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=TEAL,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=TEXT,
            spaceAfter=7,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MID_GREY,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=TEXT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=10,
            textColor=colors.white,
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12,
            textColor=NAVY,
            spaceAfter=3,
        ),
        "callout_body": ParagraphStyle(
            "CalloutBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=TEXT,
        ),
        "cover_label": ParagraphStyle(
            "CoverLabel",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MID_GREY,
            alignment=TA_CENTER,
        ),
    }


STYLES = _make_styles()


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D8DEE4"))
        canvas.setLineWidth(0.5)
        canvas.line(1.7 * cm, height - 1.25 * cm, width - 1.7 * cm, height - 1.25 * cm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(1.7 * cm, height - 0.9 * cm, "Auditoria EGIF-48 | Test reservado 2023")
    canvas.setStrokeColor(colors.HexColor("#D8DEE4"))
    canvas.setLineWidth(0.5)
    canvas.line(1.7 * cm, 1.25 * cm, width - 1.7 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(1.7 * cm, 0.85 * cm, "Sistema de deteccion de incendios forestales | TFM")
    canvas.drawRightString(width - 1.7 * cm, 0.85 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def _metric_cards(metrics: list[dict[str, object]]) -> Table:
    cards = []
    for item in metrics:
        cards.append(
            [
                _p(_text(f"T+{item['horizon_days']}"), STYLES["cover_label"]),
                _p(_pct(item["recall_at_fpr5"]), STYLES["metric"]),
                _p("Recall @ FPR 5%", STYLES["metric_label"]),
            ]
        )
    table = Table([cards], colWidths=[5.7 * cm] * len(cards))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#C8D8E8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.8, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _make_ranking_chart(report: dict[str, object]) -> Path:
    metrics = report["metrics"]
    fwi = {item["variante"]: item["metrics"] for item in report["fwi"]}
    labels = ["T+1", "T+2", "T+3", "FWI CEMS", "FWI Van Wagner"]
    fpr = [
        *[float(item["recall_at_fpr5"]) * 100 for item in metrics],
        float(fwi["FWI oficial CEMS"]["recall_at_fpr5"]) * 100,
        float(fwi["FWI Van Wagner 1 km"]["recall_at_fpr5"]) * 100,
    ]
    daily = [
        *[float(item["recall_at_top1%_daily"]) * 100 for item in metrics],
        float(fwi["FWI oficial CEMS"]["recall_at_top1%_daily"]) * 100,
        float(fwi["FWI Van Wagner 1 km"]["recall_at_top1%_daily"]) * 100,
    ]
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / "ranking.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=180)
    palette = ["#1F7A8C", "#2A9D8F", "#264653", "#E9C46A", "#E76F51"]
    for axis, values, title in zip(
        axes,
        [fpr, daily],
        ["Recall con FPR objetivo del 5%", "Recall alertando el 1% diario"],
    ):
        bars = axis.bar(labels, values, color=palette, width=0.68)
        axis.set_ylim(0, max(values) * 1.25)
        axis.set_ylabel("Porcentaje de igniciones")
        axis.set_title(title, fontsize=10, fontweight="bold", color="#16324F")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Rendimiento sobre el test ciego 2023", fontsize=12, fontweight="bold", color="#16324F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_reliability_chart(report: dict[str, object]) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / "reliability.png"
    fig, axis = plt.subplots(figsize=(7.8, 4.1), dpi=180)
    colors_by_horizon = ["#1F7A8C", "#2A9D8F", "#E76F51"]
    for horizon, color in zip((1, 2, 3), colors_by_horizon):
        rows = report["reliability"][str(horizon)]
        predicted = [float(row["pred_media"]) * 100 for row in rows]
        observed = [float(row["obs_frecuencia"]) * 100 for row in rows]
        axis.plot(predicted, observed, "o-", label=f"T+{horizon}", color=color, linewidth=1.5)
    bounds = axis.get_xlim()
    axis.plot(bounds, bounds, "--", color="#7A8691", linewidth=1, label="Calibracion perfecta")
    axis.set_xlabel("Probabilidad predicha media (%)")
    axis.set_ylabel("Frecuencia observada (%)")
    axis.set_title("Fiabilidad de la probabilidad calibrada", fontsize=11, fontweight="bold", color="#16324F")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build(report: dict[str, object]) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ranking_chart = _make_ranking_chart(report)
    reliability_chart = _make_reliability_chart(report)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.65 * cm,
        title="Informe de auditoria EGIF-48 sobre el test reservado 2023",
        author="Sistema de deteccion de incendios forestales",
    )
    story = []
    metrics = report["metrics"]
    coverage = report["coverage"]
    fwi = {item["variante"]: item for item in report["fwi"]}

    # Portada
    story += [Spacer(1, 2.0 * cm)]
    story.append(_p("INFORME TECNICO", STYLES["cover_label"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(_p("Auditoria de la familia EGIF-48", STYLES["title"]))
    story.append(_p("Contraste de los modelos forecast_risk_egif_48_t1/t2/t3 sobre el test reservado 2023", STYLES["subtitle"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(_p("Sistema predictivo de anticipacion de incendios forestales", STYLES["subtitle"]))
    story.append(Spacer(1, 1.0 * cm))
    story.append(_metric_cards(metrics))
    story.append(Spacer(1, 1.0 * cm))
    story.append(
        _table(
            [
                ["Elemento", "Valor"],
                ["Año evaluado", "2023"],
                ["Poblacion", f"{coverage['rows_evaluated']:,} celdas-dia"],
                ["Cobertura espacial", f"{coverage['unique_cells']:,} celdas de 1 km"],
                ["Igniciones reales", f"{coverage['rows_evaluated'] and metrics[0]['n_positives']:,}"],
                ["Contrato", report["feature_contract_version"]],
                ["Fecha de elaboracion", datetime.now().strftime("%d/%m/%Y")],
            ],
            widths=[5.2 * cm, 11.9 * cm],
        )
    )
    story.append(Spacer(1, 1.0 * cm))
    story.append(_p("Documento preparado para la memoria y la defensa del TFM.", STYLES["small"]))
    story.append(PageBreak())

    # Resumen ejecutivo
    story.append(_p("1. Resumen ejecutivo", STYLES["h1"]))
    story.append(
        _p(
            "Se ha auditado la familia operativa EGIF-48 ya entrenada, sin modificar ni reentrenar sus artefactos. "
            "La evaluación cubre toda la población disponible de 2023: 29.601 celdas de 1 km durante 365 días, "
            "con 530 igniciones reales. El objetivo ha sido reproducir, con los modelos serializados, las "
            "comprobaciones metodológicas del pipeline definitivo.",
            STYLES["body"],
        )
    )
    story.append(
        _p(
            "Los tres modelos presentan una capacidad de ordenación consistente. El mejor recall al mantener un "
            "5% de falsos positivos corresponde a T+1, con 44,15%; T+3 alcanza 42,08% y T+2 41,13%. "
            "En la política operativa de alertar el 1% del territorio cada día, T+2 obtiene el mayor resultado, "
            "con 11,32% de las igniciones capturadas.",
            STYLES["body"],
        )
    )
    story.append(
        _callout(
            "Conclusión principal",
            "Los modelos EGIF-48 superan claramente al FWI en las dos métricas operativas evaluadas. "
            "La ventaja debe interpretarse como resultado de un benchmark retrospectivo ERA5: las memorias "
            "meteorologicas respetan T-1, pero las variables meteorologicas del dia objetivo proceden del reanalisis "
            "y no de un forecast historico archivado.",
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(ranking_chart), width=17.2 * cm, height=6.6 * cm))
    story.append(_p("Figura 1. Comparacion de las metricas de ranking en el test reservado 2023.", STYLES["small"]))

    # Alcance y metodologia
    story.append(_p("2. Alcance y protocolo de evaluacion", STYLES["h1"]))
    story.append(
        _p(
            "El procedimiento carga los tres objetos ForecastRiskModel desde data/models y valida su contrato "
            "antes de realizar predicciones. Para cada lote se comprueba la presencia de las 48 variables, la "
            "ausencia de valores no finitos y la cobertura de la particion anual. Las metricas de ordenacion se "
            "calculan sobre la puntuacion cruda; la probabilidad calibrada se reserva para Brier, fiabilidad y "
            "niveles de riesgo.",
            STYLES["body"],
        )
    )
    story.append(
        _table(
            [
                ["Control", "Resultado"],
                ["Año reservado", "2023"],
                ["Fechas cubiertas", f"{coverage['first_date']} a {coverage['last_date']}"],
                ["Filas leidas", f"{coverage['rows_seen']:,}"],
                ["Filas evaluadas", f"{coverage['rows_evaluated']:,}"],
                ["Filas descartadas", f"{coverage['rows_dropped_incomplete_or_invalid']:,}"],
                ["Dias unicos", str(coverage["unique_days"])],
                ["Celdas por dia", f"{coverage['cells_per_day_min']:,} - {coverage['cells_per_day_max']:,}"],
                ["Cobertura completa", "Si" if coverage["full_population"] else "No"],
            ],
            widths=[6.0 * cm, 11.1 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(_p("El dataset declara la alineacion temporal egif-operational-tminus1-v1 y el recalculo de las memorias meteorologicas hasta el dia anterior.", STYLES["body"]))

    # Contrato
    story.append(_p("3. Integridad de los artefactos", STYLES["h1"]))
    story.append(
        _p(
            "Los tres artefactos superan los controles de horizonte, contrato EGIF-48, separacion entre entrenamiento, "
            "calibracion, validacion y test, calibracion Platt, ausencia del FWI entre los predictores y coherencia "
            "con sus sidecars JSON. La auditoria tambien ha reproducido exactamente las metricas deterministas "
            "guardadas en los metadatos de cada modelo.",
            STYLES["body"],
        )
    )
    artifact_data = [["Horizonte", "Familia", "Features", "Split temporal", "Metadatos test"]]
    for item in report["artifact_checks"]:
        artifact_data.append(
            [
                f"T+{item['horizon_days']}",
                item["model_family"],
                "48 / EGIF-48",
                "Correcto",
                item.get("metadata_test_comparison", {}).get("status", "-").capitalize(),
            ]
        )
    story.append(_table(artifact_data, widths=[2.0 * cm, 4.0 * cm, 3.0 * cm, 4.0 * cm, 4.1 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        _callout(
            "Lectura temporal correcta",
            "El contrato operativo evita que las memorias acumuladas usen informacion posterior al dia objetivo. "
            "No obstante, temperatura, humedad, viento y VPD del dia objetivo siguen siendo ERA5 observado/reanalizado. "
            "Por ello, este informe no debe presentarse como una validacion de prediccion meteorologica real.",
            ORANGE,
        )
    )

    # Metricas
    story.append(PageBreak())
    story.append(_p("4. Resultados de los modelos", STYLES["h1"]))
    story.append(_p("La tabla siguiente reproduce el cuadro de metricas del proyecto para cada horizonte.", STYLES["body"]))
    metric_data = [["Modelo", "PR-AUC", "ROC-AUC", "Recall @ FPR 5%", "IC 90%", "Recall top 1% diario", "Brier"]]
    for item in metrics:
        metric_data.append(
            [
                f"T+{item['horizon_days']}",
                _fmt(item["pr_auc"], 6),
                _fmt(item["roc_auc"], 4),
                _pct(item["recall_at_fpr5"]),
                f"{_pct(item['recall_ci90_low'])} - {_pct(item['recall_ci90_high'])}",
                _pct(item["recall_at_top1%_daily"]),
                _fmt(item["brier_score"], 8),
            ]
        )
    story.append(_table(metric_data, widths=[1.5 * cm, 2.0 * cm, 2.0 * cm, 3.0 * cm, 3.3 * cm, 3.0 * cm, 2.0 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        _p(
            f"El ROC-AUC restringido a la temporada junio-septiembre se sitúa entre {_fmt(min(item['roc_auc_temporada'] for item in metrics), 4)} y {_fmt(max(item['roc_auc_temporada'] for item in metrics), 4)}. "
            f"El ROC-AUC medio dentro del dia, que aisla la señal espacial, se sitúa entre {_fmt(min(item['roc_auc_dentro_del_dia'] for item in metrics), 4)} y {_fmt(max(item['roc_auc_dentro_del_dia'] for item in metrics), 4)}.",
            STYLES["body"],
        )
    )

    story.append(_p("4.1. Comparacion pareada entre horizontes", STYLES["h2"]))
    paired_data = [["Metrica", "Comparacion", "Diferencia", "IC 90%", "McNemar", "Veredicto"]]
    for item in report["paired_comparisons"]:
        if item["metrica"] == "recall@fpr5":
            paired_data.append(
                [
                    "Recall @ FPR 5%",
                    f"{item['modelo_a']} vs {item['modelo_b']}",
                    _pp(item["diferencia"]),
                    f"{_pp(item['dif_ci90_low'])} - {_pp(item['dif_ci90_high'])}",
                    _fmt(item["mcnemar_p"], 4),
                    item["veredicto"],
                ]
            )
    story.append(_table(paired_data, widths=[3.0 * cm, 3.4 * cm, 2.2 * cm, 4.0 * cm, 2.2 * cm, 2.5 * cm]))
    story.append(_p("T+1 presenta una ventaja pareada frente a T+2 en el recall @ FPR 5%, con una diferencia del 3,02 pp y un IC del 90% que no contiene el cero. La diferencia frente a T+3 no es concluyente.", STYLES["body"]))

    # FWI
    story.append(_p("5. Contraste con el baseline FWI", STYLES["h1"]))
    story.append(
        _p(
            "Se han evaluado las dos referencias disponibles en el datacubo: el FWI oficial de CEMS y el FWI de Van Wagner calculado a 1 km. El primero representa el producto publicado, aunque nace a una resolucion espacial mas gruesa. El segundo tiene resolucion nativa, pero utiliza extremos de la ventana 12-18 h en lugar de la lectura de mediodia solar.",
            STYLES["body"],
        )
    )
    fwi_data = [["Sistema", "PR-AUC", "ROC-AUC", "Recall @ FPR 5%", "Recall top 1% diario"]]
    for name, item in fwi.items():
        m = item["metrics"]
        fwi_data.append([name, _fmt(m["pr_auc"], 6), _fmt(m["roc_auc"], 4), _pct(m["recall_at_fpr5"]), _pct(m["recall_at_top1%_daily"])])
    for item in metrics:
        fwi_data.append([f"EGIF-48 T+{item['horizon_days']}", _fmt(item["pr_auc"], 6), _fmt(item["roc_auc"], 4), _pct(item["recall_at_fpr5"]), _pct(item["recall_at_top1%_daily"])])
    story.append(_table(fwi_data, widths=[5.5 * cm, 2.3 * cm, 2.3 * cm, 3.4 * cm, 3.7 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    base = fwi["FWI oficial CEMS"]["metrics"]
    t1 = metrics[0]
    story.append(
        _callout(
            "Ventaja frente a CEMS",
            f"En recall @ FPR 5%, T+1 alcanza {_pct(t1['recall_at_fpr5'])} frente a {_pct(base['recall_at_fpr5'])} del FWI CEMS: una ventaja de {_pp(t1['recall_at_fpr5'] - base['recall_at_fpr5'])}. "
            f"En la politica diaria del 1%, la ventaja es de {_pp(t1['recall_at_top1%_daily'] - base['recall_at_top1%_daily'])}.",
        )
    )
    story.append(_p("Las comparaciones pareadas frente a CEMS confirman que las diferencias son concluyentes para los tres horizontes tanto en recall @ FPR 5% como en recall diario al 1%.", STYLES["body"]))

    # Errores y target
    story.append(PageBreak())
    story.append(_p("6. Analisis de errores y sensibilidad del target", STYLES["h1"]))
    story.append(_p("Al umbral que respeta un 5% de falsos positivos, se detectan 234 de 530 igniciones con T+1, 218 con T+2 y 223 con T+3. El detalle por mes permite identificar la estacionalidad de los aciertos y errores.", STYLES["body"]))
    months = sorted({row["mes"] for values in report["errors"].values() for row in values["por_mes"]})
    month_data = [["Mes", "Igniciones", "T+1", "T+2", "T+3"]]
    by_horizon = {h: {row["mes"]: row for row in rows["por_mes"]} for h, rows in report["errors"].items()}
    for month in months:
        total = by_horizon["1"][month]["igniciones"]
        month_data.append([str(month), str(total), _pct(by_horizon["1"][month]["recall"]), _pct(by_horizon["2"][month]["recall"]), _pct(by_horizon["3"][month]["recall"])])
    story.append(_table(month_data, widths=[2.0 * cm, 3.0 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(_p("La definicion del target tambien se ha tensionado sin reentrenar. Para incendios de al menos 1 ha hay 93 casos y el recall @ FPR 5% se mantiene entre 34,41% y 36,56%. Para al menos 10 ha solo hay 22 casos; el resultado oscila entre 45,45% y 50,00%, por lo que debe interpretarse con mucha cautela.", STYLES["body"]))
    target_data = [["Definicion", "Casos", "T+1", "T+2", "T+3"]]
    target_rows = {row["definicion"]: {} for row in report["target_sensitivity"]["1"]}
    for horizon, rows in report["target_sensitivity"].items():
        for row in rows:
            target_rows.setdefault(row["definicion"], {})[horizon] = row
    for label, values in target_rows.items():
        target_data.append([label, str(values.get("1", {}).get("igniciones", "-")), _pct(values.get("1", {}).get("recall_at_fpr5")), _pct(values.get("2", {}).get("recall_at_fpr5")), _pct(values.get("3", {}).get("recall_at_fpr5"))])
    story.append(_table(target_data, widths=[6.4 * cm, 2.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]))

    # Fiabilidad y riesgo
    story.append(_p("7. Fiabilidad y niveles de riesgo", STYLES["h1"]))
    story.append(_p("La calibracion Platt transforma las puntuaciones corregidas por el muestreo en probabilidades interpretables. La siguiente figura resume la relacion entre la probabilidad predicha y la frecuencia observada por cuantiles.", STYLES["body"]))
    story.append(Image(str(reliability_chart), width=15.5 * cm, height=8.0 * cm))
    story.append(_p("Figura 2. Fiabilidad de la probabilidad calibrada por horizonte.", STYLES["small"]))
    risk_data = [["Horizonte", "Nivel", "Celdas-dia", "% territorio", "Igniciones", "Incidencia"]]
    for horizon in ("1", "2", "3"):
        for row in report["risk_levels"][horizon]["levels"]:
            risk_data.append([f"T+{horizon}", row["nivel"], f"{row['celdas_dia']:,}", _pct(row["porcentaje_territorio"]), str(row["igniciones"]), _pct(row["incidencia"])])
    story.append(_table(risk_data, widths=[2.0 * cm, 3.0 * cm, 3.3 * cm, 3.2 * cm, 2.3 * cm, 3.0 * cm]))

    # Espacial e importancia
    story.append(PageBreak())
    story.append(_p("8. Componente espacial e importancia de variables", STYLES["h1"]))
    story.append(_p("Con los artefactos congelados se ha calculado un diagnostico descriptivo por cinco bandas diagonales del territorio. Esta lectura muestra heterogeneidad espacial, pero no debe confundirse con la validacion leave-one-band-out del pipeline definitivo, que exigiria reentrenar un modelo dejando cada banda fuera.", STYLES["body"]))
    spatial_data = [["Horizonte", "Banda", "Igniciones", "Recall @ FPR 5%", "ROC-AUC dentro del dia"]]
    for row in report["spatial_slices"]:
        spatial_data.append([f"T+{row['horizon_days']}", str(row["banda_diagonal"]), str(row["n_positives"]), _pct(row["recall_at_fpr5"]), _fmt(row["roc_auc_dentro_del_dia"], 4)])
    story.append(_table(spatial_data, widths=[2.5 * cm, 2.0 * cm, 3.0 * cm, 4.0 * cm, 5.7 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(_p("Las importancias de LightGBM se han inspeccionado directamente desde cada artefacto. En el informe JSON se conserva el top 15 por horizonte para trazabilidad y para evitar reducir una conclusion multivariable a una sola variable dominante.", STYLES["body"]))
    importance_data = [["Horizonte", "Variables con mayor importancia"]]
    for horizon in ("1", "2", "3"):
        top = report["feature_importance_top15"][horizon][:8]
        names = ", ".join(f"{row['feature']} ({_fmt(row['importance'], 0)})" for row in top)
        importance_data.append([f"T+{horizon}", names])
    story.append(_table(importance_data, widths=[2.2 * cm, 14.9 * cm]))

    # Limitaciones y conclusiones
    story.append(_p("9. Limitaciones y conclusiones", STYLES["h1"]))
    story.append(
        _p(
            "La auditoria confirma la consistencia interna de la familia EGIF-48 y su ventaja de ranking frente al FWI en el año reservado. Sin embargo, no convierte el benchmark ERA5 en una simulacion de produccion: para cerrar esa brecha es necesario disponer de historico de forecasts meteorologicos y evaluar las mismas fechas de emision con sus errores reales.",
            STYLES["body"],
        )
    )
    conclusions = [
        "La cobertura y el contrato EGIF-48 son completos y reproducibles.",
        "Las metricas de los artefactos se reproducen sin discrepancias numericas.",
        "Los tres horizontes superan al FWI CEMS y al FWI Van Wagner en ranking operativo.",
        "T+1 es el mejor en recall al 5% de FPR; T+2 es el mejor al alertar el 1% diario.",
        "La prueba de semillas y la validacion leave-one-band-out quedan fuera porque requieren reentrenamiento.",
    ]
    bullet_text = "<br/>".join(f"- {_text(item)}" for item in conclusions)
    story.append(_callout("Conclusiones para la memoria", bullet_text, GREEN))

    # Reproducibilidad
    story.append(_p("10. Reproducibilidad", STYLES["h1"]))
    story.append(_p("La auditoria se ejecuta con el siguiente comando, sin reentrenar los modelos:", STYLES["body"]))
    command = "PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python scripts/audit_egif48_test.py --test-years 2023 --bootstrap-samples 200 --batch-size 250000 --include-van-wagner"
    story.append(_callout("Comando de auditoria", f"<font name='Courier'>{_text(command)}</font>", NAVY))
    story.append(Spacer(1, 0.25 * cm))
    story.append(_p("El detalle completo se conserva en docs/technical/egif48_test_2023_audit.json. Este PDF es la version redactada para lectura y presentacion.", STYLES["small"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return OUTPUT


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    output = build(report)
    print(output)


if __name__ == "__main__":
    main()
