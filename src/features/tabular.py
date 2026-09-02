"""Exportación por bloques del datacubo a un dataset tabular para ML clásico."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from src.config import DATACUBE_PATH, TABULAR_DATASET_DIR

logger = logging.getLogger(__name__)

OUTCOME_COLUMNS = {
    "target_ignicion",
    "burned_area_ha",
    "large_fire_500ha",
}
SAMPLING_AUXILIARY_COLUMNS = {
    "is_near_ignition_25x25_10d",
}
BASELINE_COLUMNS = {
    # Se conserva para comparar el modelo frente al índice físico, pero no se
    # entrega a los algoritmos de ML como predictor.
    "fire_weather_index",
}
NON_PREDICTOR_COLUMNS = (
    OUTCOME_COLUMNS
    | {
        "is_galicia",
        "cell_id",
        "year",
        # Se mantienen para trazabilidad, pero no añaden señal al modelo actual.
        "aspect_no_data_fraction",  # Constante a cero en la malla de Galicia.
        "precipitation_sum_1d",  # Duplicado exacto de precipitation_sum.
    }
    | SAMPLING_AUXILIARY_COLUMNS
    | BASELINE_COLUMNS
)


def obtener_columnas_predictoras(data_variables: set[str]) -> list[str]:
    """Devuelve las columnas del cubo permitidas como entrada de un modelo.

    ``year`` se conserva en los Parquet para particionar y realizar los cortes
    temporales, pero no puede ser un predictor: codificaría una tendencia de la
    serie histórica que no está definida para un año futuro de producción. Las
    columnas constantes o duplicadas se conservan en los datos para auditoría,
    pero se excluyen del contrato de entrenamiento.
    """
    return sorted(data_variables - NON_PREDICTOR_COLUMNS)


def exportar_datacubo_tabular(
    cube_path: str | Path = DATACUBE_PATH,
    output_dir: str | Path = TABULAR_DATASET_DIR,
    chunk_days: int = 7,
) -> None:
    """Escribe bloques semanales y los consolida en un Parquet por año.

    Cada fila equivale a una pareja ``fecha`` y ``cell_id`` activa de Galicia.
    Se conserva el target EGIF y los resultados auxiliares, pero estos últimos
    quedan explícitamente separados de las variables predictoras en el esquema.
    """
    if chunk_days < 1:
        raise ValueError("chunk_days debe ser mayor que cero.")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"El directorio de salida ya contiene datos: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(cube_path) as cube:
        required = {"is_galicia", "target_ignicion"}
        missing = required.difference(cube.data_vars)
        if missing:
            raise ValueError(f"El datacubo no contiene variables requeridas: {sorted(missing)}")

        active_positions = np.flatnonzero(cube["is_galicia"].values.ravel())
        cell_ids = xr.DataArray(
            np.arange(cube.sizes["y"] * cube.sizes["x"], dtype=np.int32).reshape(
                cube.sizes["y"], cube.sizes["x"]
            ),
            dims=("y", "x"),
            coords={"y": cube.y, "x": cube.x},
            name="cell_id",
        )
        data = cube.assign(cell_id=cell_ids)
        dates = pd.DatetimeIndex(data.time.values)
        predictors = obtener_columnas_predictoras(set(data.data_vars))
        part = 0
        dropped_incomplete = 0
        total_parts = (len(dates) + chunk_days - 1) // chunk_days
        announced_years: set[int] = set()

        for start in range(0, len(dates), chunk_days):
            end = min(start + chunk_days, len(dates))
            block = (
                data.isel(time=slice(start, end))
                .stack(cell=("y", "x"))
                .isel(cell=active_positions)
            )
            frame = block.to_dataframe().reset_index()
            # Coordenada escalar residual de ERA5: identifica un único miembro
            # de ensemble y no describe la celda ni la fecha.
            frame.drop(columns="number", errors="ignore", inplace=True)
            frame = frame.loc[frame["target_ignicion"].notna()].copy()
            frame.rename(columns={"time": "fecha"}, inplace=True)
            frame["target_ignicion"] = frame["target_ignicion"].astype(np.uint8)
            if "large_fire_500ha" in frame:
                frame["large_fire_500ha"] = frame["large_fire_500ha"].astype(np.uint8)
            frame["year"] = pd.to_datetime(frame["fecha"]).dt.year.astype(np.int16)
            complete_rows = frame.dropna(subset=predictors)
            dropped_incomplete += len(frame) - len(complete_rows)

            for year, year_frame in complete_rows.groupby("year", sort=True):
                if year not in announced_years:
                    logger.info(
                        "      Exportando año %s (bloque %d/%d).", year, part + 1, total_parts
                    )
                    announced_years.add(year)
                destination = output_dir / f"year={year}" / f"part-{part:05d}.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                year_frame.to_parquet(destination, index=False)
            part += 1

        finalizar_exportacion_tabular(cube_path, output_dir, dropped_incomplete)


def finalizar_exportacion_tabular(
    cube_path: str | Path, output_dir: str | Path, dropped_incomplete_predictors: int = 0
) -> None:
    """Consolida un dataset tabular ya exportado y escribe sus metadatos."""
    output_dir = Path(output_dir)
    with xr.open_dataset(cube_path) as cube:
        predictors = obtener_columnas_predictoras(set(cube.data_vars))
        time_contract = cube.attrs.get("time_contract")
    annual_files = _consolidar_partes_anuales(output_dir)
    metadata = {
        "row_definition": "One active 1 km Galicia cell on one EGIF-covered date with complete predictors.",
        "target": "target_ignicion (EGIF-MITECO)",
        "outcome_columns_not_predictors": sorted(OUTCOME_COLUMNS),
        "sampling_auxiliary_columns_not_predictors": sorted(SAMPLING_AUXILIARY_COLUMNS),
        "baseline_columns_not_predictors": sorted(BASELINE_COLUMNS),
        "predictor_columns": predictors,
        "partitioning": "year=YYYY/dataset_YYYY.parquet",
        "annual_files": annual_files,
        "dropped_rows_incomplete_predictors": dropped_incomplete_predictors,
        "source_datacube": str(cube_path),
        "time_contract": time_contract,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _consolidar_partes_anuales(output_dir: Path) -> dict[str, int]:
    """Consolida bloques temporales en un único Parquet anual verificable."""
    annual_rows: dict[str, int] = {}
    for year_dir in sorted(output_dir.glob("year=*")):
        year = year_dir.name.removeprefix("year=")
        logger.info("      Consolidando Parquet anual %s.", year)
        parts = sorted(year_dir.glob("part-*.parquet"))
        destination = year_dir / f"dataset_{year}.parquet"
        if destination.exists():
            with pq.ParquetFile(destination) as annual:
                annual_rows[year] = annual.metadata.num_rows
            for part in parts:
                part.unlink()
            continue
        if not parts:
            continue
        temporary = destination.with_suffix(".partial")
        writer = None
        rows = 0
        try:
            for part in parts:
                parquet = pq.ParquetFile(part)
                try:
                    for batch in parquet.iter_batches(batch_size=250_000):
                        if writer is None:
                            writer = pq.ParquetWriter(
                                temporary, batch.schema, compression="snappy"
                            )
                        writer.write_batch(batch)
                        rows += batch.num_rows
                finally:
                    parquet.close()
        finally:
            if writer is not None:
                writer.close()
        with pq.ParquetFile(temporary) as written:
            verified_rows = written.metadata.num_rows
        if writer is None or verified_rows != rows:
            raise RuntimeError(f"No se pudo verificar la consolidación anual: {year_dir.name}")
        temporary.replace(destination)
        for part in parts:
            part.unlink()
        annual_rows[year] = rows
    return annual_rows


def main() -> None:
    """Exporta el cubo final a Parquet particionado."""
    parser = argparse.ArgumentParser(description="Exporta el datacubo EGIF a Parquet tabular.")
    parser.add_argument("--cube", default=str(DATACUBE_PATH))
    parser.add_argument("--output-dir", default=str(TABULAR_DATASET_DIR))
    parser.add_argument("--chunk-days", type=int, default=7)
    args = parser.parse_args()
    exportar_datacubo_tabular(args.cube, args.output_dir, args.chunk_days)


if __name__ == "__main__":
    main()
