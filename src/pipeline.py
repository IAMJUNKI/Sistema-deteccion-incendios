"""Pipeline principal para ensamblar el datacubo histórico completo."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_SPATIAL_CUBE = Path("data/processed/cube_topography_test.nc")
DEFAULT_TOPOGRAPHY_CUBE = Path("data/processed/topography_test.nc")
DEFAULT_LANDCOVER_CUBE = Path("data/processed/landcover_test.nc")
DEFAULT_TIME_CUBE = Path("data/processed/time_2018_2023.nc")
DEFAULT_METEOROLOGY_CUBE = Path("data/processed/meteorology_2018_2023.nc")
DEFAULT_EGIF_TARGET = Path("data/processed/target/egif_target_2018_2023.parquet")
DEFAULT_EGIF_METADATA = Path("data/processed/target/egif_target_2018_2023_metadata.json")
DEFAULT_OUTPUT = Path("data/processed/datacube_2018_2023.nc")


def construir_datacubo_completo(
    spatial_cube_path: str | Path,
    topography_cube_path: str | Path,
    landcover_cube_path: str | Path,
    time_cube_path: str | Path,
    meteorology_cube_path: str | Path,
    egif_target_path: str | Path,
    egif_metadata_path: str | Path,
    output_path: str | Path,
) -> None:
    """Une las capas estáticas, temporales, meteorológicas y el target EGIF.

    El target vale 0 únicamente en fechas cubiertas por el XML EGIF. Las fechas
    posteriores a la última observación se almacenan como valores ausentes para
    impedir que se conviertan en negativos artificiales.
    """
    target = pd.read_parquet(egif_target_path)
    metadata = json.loads(Path(egif_metadata_path).read_text(encoding="utf-8"))
    observed_end = pd.Timestamp(metadata["available_event_period"]["end"])

    with (
        xr.open_dataset(spatial_cube_path) as spatial,
        xr.open_dataset(topography_cube_path) as topography,
        xr.open_dataset(landcover_cube_path) as landcover,
        xr.open_dataset(time_cube_path) as temporal,
        xr.open_dataset(meteorology_cube_path) as meteorology,
    ):
        target_values = np.full(
            (meteorology.sizes["time"], meteorology.sizes["y"], meteorology.sizes["x"]),
            np.nan,
            dtype=np.float32,
        )
        active = spatial["is_galicia"].values.astype(bool)
        observed_dates = pd.DatetimeIndex(meteorology.time.values) <= observed_end
        target_values[observed_dates, :, :] = np.where(active, 0.0, np.nan)

        date_positions = {
            pd.Timestamp(value): position for position, value in enumerate(meteorology.time.values)
        }
        rows = (target["cell_id"].to_numpy() // meteorology.sizes["x"]).astype(int)
        cols = (target["cell_id"].to_numpy() % meteorology.sizes["x"]).astype(int)
        for row, col, date in zip(rows, cols, pd.to_datetime(target["fecha"]), strict=True):
            if date in date_positions:
                target_values[date_positions[date], row, col] = 1.0

        target_cube = xr.Dataset(
            {
                "target_ignicion": (("time", "y", "x"), target_values),
                "egif_observed": (
                    "time",
                    (pd.DatetimeIndex(meteorology.time.values) <= observed_end).astype(np.uint8),
                ),
            },
            coords={"time": meteorology.time, "y": meteorology.y, "x": meteorology.x},
        )
        target_cube["target_ignicion"].attrs = {
            "long_name": "Official EGIF ignition target",
            "description": "1=one or more ignitions in the cell/day; 0=no ignition; NaN=uncovered.",
            "source": "EGIF-MITECO",
        }
        target_cube["egif_observed"].attrs = {
            "long_name": "EGIF source coverage indicator",
            "description": "1 when the date is covered by the supplied EGIF XML.",
        }

        complete = xr.merge(
            [spatial, topography, landcover, temporal, meteorology, target_cube],
            compat="equals",
            combine_attrs="drop_conflicts",
        )
        complete.attrs = {
            "title": "Galicia wildfire historical datacube",
            "period": (
                f"{pd.Timestamp(complete.time.min().values):%Y-%m-%d} to "
                f"{pd.Timestamp(complete.time.max().values):%Y-%m-%d}"
            ),
            "egif_observed_until": observed_end.date().isoformat(),
            "target_contract": "Dates without EGIF coverage are missing, never negative.",
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
    parser.add_argument("--egif-metadata", default=str(DEFAULT_EGIF_METADATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    construir_datacubo_completo(
        args.spatial_cube,
        args.topography_cube,
        args.landcover_cube,
        args.time_cube,
        args.meteorology_cube,
        args.egif_target,
        args.egif_metadata,
        args.output,
    )


if __name__ == "__main__":
    main()

