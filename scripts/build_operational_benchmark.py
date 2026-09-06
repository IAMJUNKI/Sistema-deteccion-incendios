#!/usr/bin/env python3
"""Construye el dataset EGIF histórico con memoria meteorológica hasta T-1.

El Parquet canónico no se sobrescribe. La salida se publica en un directorio
separado para que el entrenamiento operativo nunca confunda el benchmark
retrospectivo con las entradas disponibles en producción.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features.operational_benchmark import (
    DEFAULT_BATCH_SIZE,
    build_operational_dataset,
)


def _years(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        return ()
    if "-" in value:
        start, end = (int(item.strip()) for item in value.split("-", 1))
        if start > end:
            raise ValueError("El intervalo de años debe ir de menor a mayor.")
        return tuple(range(start, end + 1))
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("Debe indicarse al menos un año.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/tabular/egif"),
        help="Dataset canónico con metadata.json y particiones anuales.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/tabular/egif_operational"),
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Años, por ejemplo 2016-2023 o 2016,2017. Por defecto, todos los disponibles.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--static-layer-quality",
        default="provisional_import",
        choices=("provisional_import", "validated", "unknown"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reemplaza una salida previa; úsalo solo después de revisar su contenido.",
    )
    args = parser.parse_args()
    result = build_operational_dataset(
        args.source_dir,
        args.output_dir,
        years=_years(args.years) if args.years else None,
        batch_size=args.batch_size,
        static_layer_quality=args.static_layer_quality,
        force=args.force,
    )
    print(f"Dataset operativo publicado en {result}")


if __name__ == "__main__":
    main()

