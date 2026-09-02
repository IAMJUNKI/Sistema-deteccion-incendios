#!/usr/bin/env python3
"""Prepara la rejilla vectorial operativa a partir del NetCDF EGIF.

El NetCDF conserva las capas estáticas y la máscara 1 km, mientras que la
inferencia necesita además geometrías para el mapa y columnas estáticas en la
misma tabla. Este adaptador genera exactamente las 29.601 celdas activas que
usa el Parquet EGIF y evita mezclarla con la rejilla legacy de 30.697 celdas.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

from src.features.canonical_contract import CANONICAL_FEATURES, CANONICAL_WEATHER_FEATURES
from src.geospatial.grid import crear_rejilla_vectorial


def prepare_operational_grid(
    cube_path: str | Path,
    output_path: str | Path,
    *,
    cell_size_meters: float = 1000.0,
) -> Path:
    """Exporta la máscara activa, geometría y estáticas del datacubo EGIF."""

    with xr.open_dataset(cube_path) as cube:
        if "is_galicia" not in cube:
            raise ValueError("El NetCDF no contiene la máscara is_galicia.")
        if not {"x", "y"}.issubset(cube.dims):
            raise ValueError("El NetCDF no contiene dimensiones espaciales x/y.")
        grid = crear_rejilla_vectorial(cube, cell_size_meters)
        active = grid["is_galicia"].eq(1).to_numpy()
        static_columns = sorted(
            set(CANONICAL_FEATURES) - set(CANONICAL_WEATHER_FEATURES)
        )
        missing = [column for column in static_columns if column not in cube.data_vars]
        if missing:
            raise ValueError(
                "El datacubo no contiene todas las variables estáticas canónicas: "
                f"{missing}"
            )
        for column in static_columns:
            values = np.asarray(cube[column].values)
            if values.ndim != 2 or values.shape != (cube.sizes["y"], cube.sizes["x"]):
                raise ValueError(f"La variable estática {column} no tiene dimensión y/x.")
            grid[column] = values.ravel().astype("float32")

    result = grid.loc[active].copy()
    result = result.to_crs("EPSG:4326")
    result["lat_centroid"] = result.geometry.centroid.y.astype("float64")
    result["lon_centroid"] = result.geometry.centroid.x.astype("float64")
    result = result.drop(columns=["x", "y"], errors="ignore")
    expected = int(os.getenv("CANONICAL_GRID_CELLS", "29601"))
    if len(result) != expected:
        raise ValueError(
            f"La máscara del datacubo contiene {len(result):,} celdas activas; "
            f"se esperaban {expected:,}."
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/grid/galicia_grid_1km_egif.parquet"),
    )
    args = parser.parse_args()
    destination = prepare_operational_grid(args.cube, args.output)
    print(f"Rejilla EGIF operativa guardada en {destination}")


if __name__ == "__main__":
    main()
