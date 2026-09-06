#!/usr/bin/env python3
"""Genera el PDF ilustrado con las figuras y resultados del modelo EGIF."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "informe_graficos_resultados.pdf"
HORIZONS_IMAGE = ROOT / "docs" / "technical" / "model_family_horizons.png"
IMPORTANCE_IMAGE = ROOT / "docs" / "technical" / "egif48_feature_importance.png"


def styles() -> dict[str, ParagraphStyle]:
    """Devuelve estilos legibles para el informe A4."""

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#15324b"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#536878"),
            spaceAfter=18,
        ),
        "heading": ParagraphStyle(
            "ReportHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#15324b"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#253746"),
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#536878"),
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#536878"),
            spaceBefore=5,
            spaceAfter=12,
        ),
        "quote": ParagraphStyle(
            "ReportQuote",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=15,
            leftIndent=14,
            rightIndent=14,
            borderColor=colors.HexColor("#d2e4ee"),
            borderWidth=1,
            borderPadding=9,
            backColor=colors.HexColor("#f3f8fb"),
            spaceBefore=8,
            spaceAfter=10,
        ),
    }


def footer(canvas, doc) -> None:
    """Dibuja el pie de página y el número de página."""

    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d7e0e6"))
    canvas.line(doc.leftMargin, 1.35 * cm, width - doc.rightMargin, 1.35 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#667985"))
    canvas.drawString(doc.leftMargin, 0.9 * cm, "Informe de figuras - modelo EGIF")
    canvas.drawRightString(width - doc.rightMargin, 0.9 * cm, f"Página {doc.page}")
    canvas.restoreState()


def make_table() -> Table:
    """Construye la tabla de resultados del EGIF 48 ampliado."""

    data = [
        ["Horizonte", "PR-AUC", "ROC-AUC", "Recall FPR 5 %", "Recall top 1 %"],
        ["T+1", "0,000707", "0,8713", "44,15 %", "9,62 %"],
        ["T+2", "0,000555", "0,8695", "41,13 %", "11,32 %"],
        ["T+3", "0,000664", "0,8720", "42,08 %", "9,81 %"],
    ]
    table = Table(data, colWidths=[2.2 * cm, 2.3 * cm, 2.3 * cm, 3.1 * cm, 3.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#15324b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd7de")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7fa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_pdf() -> Path:
    """Genera y devuelve el PDF final."""

    missing = [path for path in (HORIZONS_IMAGE, IMPORTANCE_IMAGE) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan figuras requeridas: {missing}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    style = styles()
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.75 * cm,
        title="Informe de figuras y resultados del modelo EGIF",
        author="Sistema de detección de incendios",
    )
    width = A4[0] - document.leftMargin - document.rightMargin
    story = [
        Spacer(1, 0.4 * cm),
        Paragraph("Informe de figuras y resultados", style["title"]),
        Paragraph("Modelo EGIF 48 - evaluación temporal y lectura operativa", style["subtitle"]),
        Paragraph(
            f"Fecha de generación: {date.today().strftime('%d/%m/%Y')}<br/>"
            "Test ciego 2023 - 10.804.365 filas y 530 igniciones positivas",
            style["small"],
        ),
        Spacer(1, 0.35 * cm),
        Paragraph("1. Objetivo", style["heading"]),
        Paragraph(
            "Este informe reúne las figuras que explican el resultado del modelo EGIF 48. "
            "La finalidad no es presentar una predicción exacta de cada incendio, sino "
            "mostrar si el modelo ordena mejor las celdas para priorizar vigilancia y "
            "movilización de medios.",
            style["body"],
        ),
        Paragraph(
            "El problema está muy desbalanceado. Por ello se utilizan PR-AUC, ROC-AUC, "
            "recall con una FPR del 5 % y recall dentro del 1 % de celdas prioritarias, "
            "en lugar de accuracy como métrica principal.",
            style["body"],
        ),
        Paragraph("2. Figura 1 - rendimiento por horizonte", style["heading"]),
        Image(str(HORIZONS_IMAGE), width=width, height=width * 0.625),
        Paragraph(
            "Figura 1. Cada panel representa una métrica. El eje horizontal contiene T+1, "
            "T+2 y T+3; cada punto es el resultado de una familia en un horizonte y la "
            "línea conecta esos puntos. No es un intervalo de confianza.",
            style["caption"],
        ),
        Paragraph("Cómo leer la figura", style["heading"]),
        Paragraph(
            "El gráfico de puntos conectados sustituye al boxplot porque cada familia solo "
            "tiene tres observaciones. Una caja de bigotes habría parecido una distribución "
            "estadística sin tener suficientes observaciones para estimarla.",
            style["body"],
        ),
        Paragraph(
            "El EGIF 48 ampliado se mantiene por encima del FWI CEMS en las cuatro métricas "
            "mostradas. La comparación debe entenderse como capacidad de ranking en este "
            "test retrospectivo; no demuestra todavía calibración frente a un forecast real "
            "de MeteoGalicia.",
            style["body"],
        ),
        make_table(),
        Spacer(1, 0.2 * cm),
        Paragraph("3. Figura 2 - importancia interna de variables", style["heading"]),
        Image(str(IMPORTANCE_IMAGE), width=width, height=width * 0.625),
        Paragraph(
            "Figura 2. Variables con mayor importancia media según el gain interno de "
            "LightGBM. Las barras están normalizadas por horizonte.",
            style["caption"],
        ),
        Paragraph("Interpretación", style["heading"]),
        Paragraph(
            "Destacan vpd_mean, las memorias de precipitación de 3 y 30 días, la humedad "
            "relativa, la elevación y la longitud de carreteras locales. Esta combinación es "
            "coherente con la sequedad atmosférica, la humedad acumulada y el contexto "
            "territorial de la ignición.",
            style["body"],
        ),
        Paragraph(
            "La importancia gain no es causalidad. Variables correlacionadas pueden repartirse "
            "la importancia. La memoria científica final debería complementar esta figura con "
            "permutación, SHAP y ablaciones por grupos.",
            style["body"],
        ),
        PageBreak(),
        Paragraph("4. Figuras que deben añadirse antes de la versión final", style["heading"]),
        Paragraph(
            "La figura más útil para la toma de decisiones será la curva de recall frente al "
            "presupuesto espacial: 1 %, 5 % y 10 % de celdas priorizadas. Permite convertir "
            "las métricas en una capacidad concreta de vigilancia. También se recomienda un "
            "mapa de un día de test con predicción e igniciones observadas y un diagrama de "
            "fiabilidad por horizonte.",
            style["body"],
        ),
        Paragraph(
            "El diagrama de fiabilidad debe esperar a disponer de varios pares "
            "forecast-observación de MeteoGalicia. ERA5 es un benchmark retrospectivo con "
            "meteorología conocida y no permite presentar esa calibración como operacional.",
            style["body"],
        ),
        Paragraph("5. Conclusión para la defensa", style["heading"]),
        Paragraph(
            "En el test retrospectivo 2023, el EGIF 48 ampliado separa mejor las celdas con "
            "ignición que el FWI CEMS y los controles EGIF 50 evaluados. La utilidad debe "
            "explicarse como priorización espacial bajo un presupuesto de vigilancia, no como "
            "certeza de incendio.",
            style["quote"],
        ),
        Paragraph(
            "Procedencia: evaluación temporal completa guardada en "
            "data/models/evaluation/model_family_comparison.json; importancia interna de "
            "los artefactos EGIF 48 ampliados en data/models/experiments/expanded. Las figuras "
            "no llevan una fuente superpuesta para mantener la lectura limpia; la trazabilidad "
            "queda en este texto y en el informe técnico del repositorio.",
            style["small"],
        ),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
