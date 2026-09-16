#!/usr/bin/env python3
"""Prepara el Parquet enriquecido que consume el dashboard Streamlit.

La inferencia diaria publica primero el resultado operativo y, a continuación,
este proceso materializa las columnas auxiliares que necesita la interfaz.
Así, el primer usuario no tiene que ejecutar el enriquecimiento de 88.803 filas
durante la apertura de la página.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.operational.artifacts import atomic_write_parquet
from src.webapp.utils.data_loader import (
    DASHBOARD_CACHE_VERSION,
    enrich_dataset_metadata,
)

DEFAULT_INPUT_PATH = Path("data/processed/predicciones_operativas.parquet")


def default_input_path() -> Path:
    """Obtiene la ruta del resultado operativo desde el entorno."""
    return Path(os.getenv("PREDICTIONS_OUTPUT_PATH", str(DEFAULT_INPUT_PATH)))


def default_output_path(input_path: Path) -> Path:
    """Obtiene la ruta del Parquet preparado desde el entorno o por convención."""
    configured = os.getenv("PREDICTIONS_DASHBOARD_CACHE_PATH", "").strip()
    if configured:
        return Path(configured)
    return input_path.with_name(f"{input_path.stem}.dashboard{input_path.suffix}")


def prepare_dashboard_cache(input_path: Path, output_path: Path, *, force: bool = False) -> bool:
    """Prepara el artefacto del dashboard si la fuente es más reciente.

    Args:
        input_path: Parquet operativo publicado por la inferencia.
        output_path: Parquet enriquecido que leerá Streamlit.
        force: Recalcula aunque el artefacto existente parezca actualizado.

    Returns:
        ``True`` si se ha generado o actualizado el artefacto.

    Raises:
        FileNotFoundError: Si no existe el Parquet de entrada.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el output operativo: {input_path}")

    if not force and output_path.exists():
        try:
            if output_path.stat().st_mtime_ns >= input_path.stat().st_mtime_ns:
                cached = pd.read_parquet(output_path, columns=["dashboard_cache_version"])
                if (
                    not cached.empty
                    and cached["dashboard_cache_version"].eq(DASHBOARD_CACHE_VERSION).all()
                ):
                    return False
        except (OSError, KeyError, ValueError, TypeError):
            pass

    frame = pd.read_parquet(input_path)
    enriched = enrich_dataset_metadata(frame)
    enriched["dashboard_cache_version"] = DASHBOARD_CACHE_VERSION
    atomic_write_parquet(enriched, output_path)
    return True


def parse_args() -> argparse.Namespace:
    """Construye los argumentos de línea de comandos."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcula el caché aunque esté actualizado.",
    )
    return parser.parse_args()


def main() -> int:
    """Ejecuta la preparación del artefacto."""
    args = parse_args()
    output_path = args.output or default_output_path(args.input)
    updated = prepare_dashboard_cache(args.input, output_path, force=args.force)
    status = "actualizado" if updated else "ya estaba actualizado"
    print(f"Caché del dashboard {status}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
