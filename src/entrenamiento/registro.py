"""Registro de resultados: un fichero que crece por filas, no una carpeta que crece por ficheros.

Cada ejecución escribía antes su propio CSV con marca de tiempo. La intención era buena —evitar
que dos ejecuciones se pisaran, que es un fallo que ya nos costó resultados— pero el efecto fue
que diez ejecuciones dejaron treinta y cinco ficheros sin trackear, y el historial del proyecto
dejó de caber en la cabeza de nadie.

Aquí se separan dos cosas que no son lo mismo:

- **La bitácora**: `docs/technical/resultados.csv`, una tabla acumulativa con una fila por
  experimento y modelo. Tiene nombre estable, va a Git y se lee como un diff: cada commit
  enseña exactamente qué números cambiaron. Es lo que sostiene el capítulo de resultados.
- **El detalle de una ejecución**: los CSV con marca de tiempo, que siguen escribiéndose para
  poder auditar una corrida concreta, pero quedan fuera de Git. Son reproducibles volviendo a
  lanzar el experimento; no hay razón para versionarlos.

Reejecutar un experimento con la misma etiqueta **sustituye** sus filas en vez de añadir otras.
Así la bitácora refleja el estado actual del conocimiento y no una pila de intentos, y sigue
siendo auditable porque el diff de Git conserva lo que había antes.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
DIR_TECNICA = RAIZ / "docs" / "technical"
BITACORA = DIR_TECNICA / "resultados.csv"

#: Columnas de contexto que preceden a las métricas, para que una fila se entienda sola.
CONTEXTO = ["experimento", "fecha", "commit", "dataset", "n_predictores_dataset",
            "n_variables", "derivadas", "conjunto"]


def commit_actual() -> str:
    """Hash corto del commit, para poder reproducir una fila de la bitácora.

    Devuelve `sin-git` si no se puede determinar: una bitácora sin hash sigue siendo útil, y
    fallar aquí abortaría un entrenamiento de veinte minutos por un detalle de trazabilidad.
    """
    try:
        salida = subprocess.run(
            ["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        estado = subprocess.run(
            ["git", "-C", str(RAIZ), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if salida.returncode != 0:
            return "sin-git"
        hash_corto = salida.stdout.strip()
        return f"{hash_corto}+sucio" if estado.stdout.strip() else hash_corto
    except (OSError, subprocess.SubprocessError):
        return "sin-git"


def registrar(
    tabla: pd.DataFrame,
    experimento: str,
    contexto: Optional[dict] = None,
    bitacora: Path = BITACORA,
) -> Path:
    """Añade o sustituye las filas de un experimento en la bitácora acumulativa.

    Args:
        tabla: Resultados de la ejecución, una fila por modelo.
        experimento: Nombre estable del experimento. Reejecutarlo sustituye sus filas.
        contexto: Metadatos que se copian a todas las filas (versión del dataset, conjunto de
            variables usado, etc.).
        bitacora: Ruta del CSV acumulativo.

    Returns:
        La ruta de la bitácora.
    """
    nuevas = tabla.copy()
    nuevas.insert(0, "experimento", experimento)
    nuevas.insert(1, "fecha", datetime.now().strftime("%Y-%m-%d %H:%M"))
    nuevas.insert(2, "commit", commit_actual())
    for clave, valor in (contexto or {}).items():
        nuevas[clave] = valor

    if bitacora.exists():
        previas = pd.read_csv(bitacora)
        sustituidas = int((previas["experimento"] == experimento).sum())
        if sustituidas:
            logger.info("Bitácora: se sustituyen %s filas previas de %r",
                        sustituidas, experimento)
        previas = previas[previas["experimento"] != experimento]
        combinada = pd.concat([previas, nuevas], ignore_index=True)
    else:
        combinada = nuevas

    # Orden de columnas estable: el contexto primero, y luego las métricas por orden
    # alfabético. Sin esto, dos ejecuciones con métricas distintas producen diffs ilegibles.
    presentes = [c for c in CONTEXTO if c in combinada.columns]
    resto = sorted(c for c in combinada.columns if c not in presentes)
    combinada = combinada[presentes + resto]

    bitacora.parent.mkdir(parents=True, exist_ok=True)
    combinada.sort_values(["experimento", "modelo"] if "modelo" in combinada.columns
                          else ["experimento"]).to_csv(bitacora, index=False)
    logger.info("Bitácora actualizada: %s (%s filas, %s experimentos)",
                bitacora, len(combinada), combinada["experimento"].nunique())
    return bitacora


def leer(bitacora: Path = BITACORA) -> pd.DataFrame:
    """Lee la bitácora completa, vacía si aún no existe."""
    return pd.read_csv(bitacora) if bitacora.exists() else pd.DataFrame()
