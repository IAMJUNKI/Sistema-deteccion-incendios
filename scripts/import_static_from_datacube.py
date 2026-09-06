#!/usr/bin/env python3
"""Importa capas estáticas de un datacube EGIF ya construido.

Esta utilidad permite arrancar el workflow en un equipo sin sesión AWS válida
cuando se dispone de un ``galicia_1km.nc`` compartido por el equipo. Solo toma
``is_galicia`` y las capas estáticas; no reutiliza meteorología, FWI ni targets.
La procedencia se registra en ``static_import_manifest.json`` y la fuente
original no se modifica.

No debe usarse para ocultar una actualización del DEM: cuando se disponga de
un DEM local validado, la reconstrucción normal con ``--rebuild-static`` sigue
siendo la ruta científica de referencia.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from src.config import (
    GRID_DIR,
    GRID_PATH,
    HUMAN_ACTIVITY_CUBE_PATH,
    LANDCOVER_CUBE_PATH,
    SPATIAL_CUBE_PATH,
    TOPOGRAPHY_CUBE_PATH,
)
from src.features.canonical_contract import CANONICAL_FEATURES, CANONICAL_WEATHER_FEATURES
from src.geospatial.grid import DEFAULT_CRS, crear_rejilla_vectorial, guardar_rejilla_datacube
from src.geospatial.human_activity import CANONICAL_DATACUBE_VARIABLE_FLAGS as HUMAN_FLAGS
from src.geospatial.topography import CANONICAL_DATACUBE_VARIABLE_FLAGS as TOPOGRAPHY_FLAGS
from src.geospatial.vegetation import CANONICAL_DATACUBE_VARIABLE_FLAGS as LANDCOVER_FLAGS


def _selected(flags: dict[str, bool]) -> list[str]:
    return [name for name, enabled in flags.items() if enabled]


def _static_layer(
    source: xr.Dataset,
    variables: list[str],
    destination: Path,
    *,
    source_path: Path,
) -> None:
    missing = [name for name in variables if name not in source.data_vars]
    if missing:
        raise ValueError(
            f"El datacube compartido no contiene las variables estáticas {missing}; "
            "no se crea una capa incompleta."
        )
    layer = xr.Dataset({"is_galicia": source["is_galicia"]})
    for name in variables:
        variable = source[name]
        if variable.dims != ("y", "x"):
            raise ValueError(
                f"La variable estática {name} tiene dimensiones {variable.dims}; "
                "se esperaba únicamente ('y', 'x')."
            )
        layer[name] = variable
    layer.attrs = {
        "source_datacube": str(source_path),
        "source_kind": "shared_static_layer_import",
        "crs": source.attrs["crs"],
        "spatial_resolution": source.attrs.get("spatial_resolution", "1000 m"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    layer.to_netcdf(destination)


def _write_operational_grid(
    source: xr.Dataset,
    destination: Path,
    *,
    cell_size_meters: float,
) -> int:
    grid = crear_rejilla_vectorial(source, cell_size_meters)
    weather = set(CANONICAL_WEATHER_FEATURES)
    static_columns = [column for column in CANONICAL_FEATURES if column not in weather]
    missing = [column for column in static_columns if column not in source.data_vars]
    if missing:
        raise ValueError(f"Faltan variables para la rejilla operativa: {missing}")
    for column in static_columns:
        grid[column] = np.asarray(source[column].values).ravel().astype("float32")

    active = grid.loc[grid["is_galicia"].eq(1)].copy()
    expected = int(os.getenv("CANONICAL_GRID_CELLS", "29601"))
    if len(active) != expected:
        raise ValueError(
            f"El datacube compartido contiene {len(active):,} celdas activas; "
            f"se esperaban {expected:,}."
        )
    centroids_wgs84 = active.geometry.centroid.to_crs("EPSG:4326")
    active = active.to_crs("EPSG:4326")
    active["lat_centroid"] = centroids_wgs84.y.astype("float64")
    active["lon_centroid"] = centroids_wgs84.x.astype("float64")
    active = active.drop(columns=["x", "y"], errors="ignore")
    destination.parent.mkdir(parents=True, exist_ok=True)
    active.to_parquet(destination, index=False)
    return len(active)


def import_static_datacube(
    source_path: str | Path,
    *,
    spatial_cube_path: str | Path = SPATIAL_CUBE_PATH,
    grid_path: str | Path = GRID_PATH,
    topography_path: str | Path = TOPOGRAPHY_CUBE_PATH,
    landcover_path: str | Path = LANDCOVER_CUBE_PATH,
    human_activity_path: str | Path = HUMAN_ACTIVITY_CUBE_PATH,
    operational_grid_path: str | Path = GRID_DIR / "galicia_grid_1km_egif.parquet",
    cell_size_meters: float = 1000.0,
) -> Path:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"No existe el datacube fuente: {source_path}")

    with xr.open_dataset(source_path) as source:
        if not {"x", "y"}.issubset(source.dims) or "is_galicia" not in source.data_vars:
            raise ValueError("El datacube fuente debe contener dimensiones x/y e is_galicia.")
        if source["is_galicia"].dims != ("y", "x"):
            raise ValueError("is_galicia debe tener dimensiones ('y', 'x').")
        source_crs = str(source.attrs.get("crs") or DEFAULT_CRS)
        source.attrs["crs"] = source_crs

        spatial = xr.Dataset({"is_galicia": source["is_galicia"]})
        spatial.attrs = {
            "title": "Galicia spatial grid imported from shared datacube",
            "description": "Spatial mask reused only to bootstrap the historical workflow.",
            "module": "grid",
            "crs": source_crs,
            "spatial_resolution": source.attrs.get("spatial_resolution", "1000 m"),
        }
        spatial["is_galicia"].attrs = dict(source["is_galicia"].attrs)
        full_grid = crear_rejilla_vectorial(spatial, cell_size_meters)
        guardar_rejilla_datacube(spatial, full_grid, spatial_cube_path, grid_path)

        _static_layer(source, _selected(TOPOGRAPHY_FLAGS), Path(topography_path), source_path=source_path)
        _static_layer(source, _selected(LANDCOVER_FLAGS), Path(landcover_path), source_path=source_path)
        _static_layer(source, _selected(HUMAN_FLAGS), Path(human_activity_path), source_path=source_path)
        cells = _write_operational_grid(
            source, Path(operational_grid_path), cell_size_meters=cell_size_meters
        )

    manifest_path = Path(spatial_cube_path).parent / "static_import_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_datacube": str(source_path),
                "source_kind": "shared_static_layer_import",
                "crs": source_crs,
                "active_cells": cells,
                "spatial_cube": str(spatial_cube_path),
                "grid": str(grid_path),
                "operational_grid": str(operational_grid_path),
                "layers": {
                    "topography": _selected(TOPOGRAPHY_FLAGS),
                    "landcover": _selected(LANDCOVER_FLAGS),
                    "human_activity": _selected(HUMAN_FLAGS),
                },
                "dynamic_variables_reused": False,
                "warning": (
                    "Static layers were imported from an existing datacube. "
                    "Rebuild with a validated local Copernicus DEM for a final scientific run."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--operational-grid",
        type=Path,
        default=GRID_DIR / "galicia_grid_1km_egif.parquet",
    )
    parser.add_argument("--cell-size", type=float, default=1000.0)
    args = parser.parse_args()
    manifest = import_static_datacube(
        args.source,
        operational_grid_path=args.operational_grid,
        cell_size_meters=args.cell_size,
    )
    print(f"Capas estáticas importadas. Manifiesto: {manifest}")


if __name__ == "__main__":
    main()
