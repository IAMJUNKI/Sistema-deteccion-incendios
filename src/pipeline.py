"""Pipeline principal para ensamblar el datacubo histórico completo."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import (
    DATACUBE_END,
    DATACUBE_PATH,
    DATACUBE_START,
    EGIF_TARGET_PATH,
    LANDCOVER_CUBE_PATH,
    METEOROLOGY_CUBE_PATH,
    SPATIAL_CUBE_PATH,
    TIME_CUBE_PATH,
    TOPOGRAPHY_CUBE_PATH,
)

DEFAULT_SPATIAL_CUBE = SPATIAL_CUBE_PATH
DEFAULT_TOPOGRAPHY_CUBE = TOPOGRAPHY_CUBE_PATH
DEFAULT_LANDCOVER_CUBE = LANDCOVER_CUBE_PATH
DEFAULT_TIME_CUBE = TIME_CUBE_PATH
DEFAULT_METEOROLOGY_CUBE = METEOROLOGY_CUBE_PATH
DEFAULT_EGIF_TARGET = EGIF_TARGET_PATH
DEFAULT_OUTPUT = DATACUBE_PATH
DATACUBE_START_DATE = DATACUBE_START
DATACUBE_END_DATE = DATACUBE_END


def construir_datacubo_completo(
    spatial_cube_path: str | Path,
    topography_cube_path: str | Path,
    landcover_cube_path: str | Path,
    time_cube_path: str | Path,
    meteorology_cube_path: str | Path,
    egif_target_path: str | Path,
    output_path: str | Path,
    start_date: str = DATACUBE_START_DATE,
    end_date: str = DATACUBE_END_DATE,
) -> None:
    """Une las capas estáticas, temporales, meteorológicas y el target EGIF.

    El intervalo termina en la última fecha del XML EGIF, por lo que el target
    vale 0 en toda celda activa sin ignición y no necesita un indicador de cobertura.
    """
    target = pd.read_parquet(egif_target_path)

    with (
        xr.open_dataset(spatial_cube_path) as spatial,
        xr.open_dataset(topography_cube_path) as topography,
        xr.open_dataset(landcover_cube_path) as landcover,
        xr.open_dataset(time_cube_path) as temporal,
        xr.open_dataset(meteorology_cube_path) as meteorology,
    ):
        temporal = temporal.sel(time=slice(start_date, end_date))
        meteorology = meteorology.sel(time=slice(start_date, end_date))
        target_values = np.full(
            (meteorology.sizes["time"], meteorology.sizes["y"], meteorology.sizes["x"]),
            np.nan,
            dtype=np.float32,
        )
        active = spatial["is_galicia"].values.astype(bool)
        target_values[:, :, :] = np.where(active, 0.0, np.nan)
        burned_area_values = target_values.copy()
        large_fire_values = target_values.copy()

        date_positions = {
            pd.Timestamp(value): position for position, value in enumerate(meteorology.time.values)
        }
        rows = (target["cell_id"].to_numpy() // meteorology.sizes["x"]).astype(int)
        cols = (target["cell_id"].to_numpy() % meteorology.sizes["x"]).astype(int)
        if "superficie_ha" in target:
            areas = target["superficie_ha"].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            areas = np.zeros(len(target), dtype=np.float32)
        for row, col, date, area in zip(
            rows, cols, pd.to_datetime(target["fecha"]), areas, strict=True
        ):
            if date in date_positions:
                target_values[date_positions[date], row, col] = 1.0
                burned_area_values[date_positions[date], row, col] = area
                large_fire_values[date_positions[date], row, col] = float(area >= 500.0)

        target_cube = xr.Dataset(
            {
                "target_ignicion": (("time", "y", "x"), target_values),
                "burned_area_ha": (("time", "y", "x"), burned_area_values),
                "large_fire_500ha": (("time", "y", "x"), large_fire_values),
            },
            coords={"time": meteorology.time, "y": meteorology.y, "x": meteorology.x},
        )
        target_cube["target_ignicion"].attrs = {
            "long_name": "Official EGIF ignition target",
            "description": "1=one or more ignitions in the cell/day; 0=no ignition.",
            "source": "EGIF-MITECO",
        }
        target_cube["burned_area_ha"].attrs = {
            "long_name": "Total burned area associated with ignitions",
            "units": "ha",
            "role": "Outcome only; never use as a predictor.",
        }
        target_cube["large_fire_500ha"].attrs = {
            "long_name": "Large-fire outcome indicator",
            "description": "1 when total burned area in the cell/day is at least 500 ha.",
            "role": "Outcome only; never use as a predictor.",
        }
        complete = xr.merge(
            [spatial, topography, landcover, temporal, meteorology, target_cube],
            compat="equals",
            combine_attrs="drop_conflicts",
        )
        complete.attrs = {
            "title": "Galicia wildfire historical datacube",
            "period": f"{start_date} to {end_date}",
            "target_contract": "All dates are covered by the supplied EGIF XML; 0 means no ignition.",
            "time_contract": "No temporal shift; meteorological accumulations include date T.",
        }
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name, variable in complete.data_vars.items()
            if variable.ndim > 0
        }
        complete.to_netcdf(destination, encoding=encoding)


def main() -> None:
    """Construye el datacubo completo desde las salidas de cada fase."""
    parser = argparse.ArgumentParser(description="Ensambla el datacubo histórico completo.")
    parser.add_argument("--spatial-cube", default=str(DEFAULT_SPATIAL_CUBE))
    parser.add_argument("--topography-cube", default=str(DEFAULT_TOPOGRAPHY_CUBE))
    parser.add_argument("--landcover-cube", default=str(DEFAULT_LANDCOVER_CUBE))
    parser.add_argument("--time-cube", default=str(DEFAULT_TIME_CUBE))
    parser.add_argument("--meteorology-cube", default=str(DEFAULT_METEOROLOGY_CUBE))
    parser.add_argument("--egif-target", default=str(DEFAULT_EGIF_TARGET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--tabular-output",
        default="data/processed/tabular_egif",
        help="Directorio del dataset Parquet particionado para ML clásico.",
    )
    parser.add_argument(
        "--skip-tabular",
        action="store_true",
        help="Construye solo el NetCDF; omite la exportación tabular.",
    )
    parser.add_argument("--start-date", default=DATACUBE_START_DATE)
    parser.add_argument("--end-date", default=DATACUBE_END_DATE)
    args = parser.parse_args()
    construir_datacubo_completo(
        args.spatial_cube,
        args.topography_cube,
        args.landcover_cube,
        args.time_cube,
        args.meteorology_cube,
        args.egif_target,
        args.output,
        args.start_date,
        args.end_date,
    )
    if not args.skip_tabular:
        from src.features.tabular import exportar_datacubo_tabular

        exportar_datacubo_tabular(args.output, args.tabular_output)


if __name__ == "__main__":
    main()
