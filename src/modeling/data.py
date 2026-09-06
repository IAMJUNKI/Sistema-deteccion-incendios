"""Carga segura de los Parquet EGIF y particiones temporales de modelado."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from src.config import TABULAR_DATASET_DIR
from src.modeling.features import FORBIDDEN_COLUMNS

TARGET_COLUMN = "target_ignicion"
TRAIN_YEARS = (2019, 2020, 2021)
VALIDATION_YEARS = (2022,)
TEST_YEARS = (2023,)


def _day_number(values: pd.Series) -> np.ndarray:
    """Convierte fechas a días desde el epoch sin asumir una unidad temporal de Pandas."""
    dates = pd.to_datetime(values, errors="raise")
    return (
        (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)
    ).to_numpy(dtype=np.int64)


@dataclass(frozen=True)
class DatasetContract:
    """Metadatos mínimos necesarios para entrenar sin fuga de información."""

    dataset_dir: Path
    predictors: list[str]
    annual_files: dict[str, int]


def load_dataset_contract(dataset_dir: str | Path = TABULAR_DATASET_DIR) -> DatasetContract:
    """Lee y valida el contrato publicado por la exportación tabular."""
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró el metadato del dataset: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictors = list(metadata["predictor_columns"])
    forbidden = sorted(set(predictors) & FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"El metadato contiene predictores prohibidos: {forbidden}")
    if not predictors:
        raise ValueError("El metadato no declara predictores.")

    return DatasetContract(dataset_dir, predictors, dict(metadata["annual_files"]))


def annual_parquet_paths(contract: DatasetContract, years: Iterable[int]) -> list[Path]:
    """Devuelve las particiones anuales requeridas, comprobando su existencia."""
    paths = [contract.dataset_dir / f"year={year}" / f"dataset_{year}.parquet" for year in years]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan particiones anuales: {missing}")
    return paths


def load_years(
    contract: DatasetContract,
    years: Iterable[int],
    columns: Iterable[str],
) -> pd.DataFrame:
    """Carga solo años y columnas solicitados; usar muestreo antes de cargar todo."""
    paths = annual_parquet_paths(contract, years)
    return (
        pads.dataset([str(path) for path in paths], format="parquet")
        .to_table(columns=list(columns))
        .to_pandas()
    )


def sample_years_for_training(
    contract: DatasetContract,
    years: Iterable[int],
    predictors: Iterable[str],
    negative_cell_modulus: int = 25,
    negative_cell_remainder: int = 0,
) -> pd.DataFrame:
    """Carga positivos y negativos mediante un hash reproducible celda–día."""
    if negative_cell_modulus < 1:
        raise ValueError("negative_cell_modulus debe ser mayor que cero.")
    if not 0 <= negative_cell_remainder < negative_cell_modulus:
        raise ValueError("negative_cell_remainder debe estar entre 0 y el módulo menos uno.")
    columns = ["fecha", "cell_id", TARGET_COLUMN, *predictors]
    frames: list[pd.DataFrame] = []
    dataset = pads.dataset(
        [str(path) for path in annual_parquet_paths(contract, years)], format="parquet"
    )
    for batch in dataset.scanner(columns=columns, batch_size=100_000).to_batches():
        frame = batch.to_pandas()
        day_number = _day_number(frame["fecha"])
        row_hash = (
            frame["cell_id"].to_numpy(dtype=np.int64) * 1_000_003 + day_number
        ) % negative_cell_modulus
        keep = (frame[TARGET_COLUMN].to_numpy() == 1) | (row_hash == negative_cell_remainder)
        if keep.any():
            frames.append(frame.loc[keep])
    if not frames:
        raise ValueError("El muestreo no produjo filas.")
    return pd.concat(frames, ignore_index=True)


def split_name_for_year(year: int) -> str:
    """Clasifica un año según el protocolo temporal fijo del TFM."""
    if year in TRAIN_YEARS:
        return "train"
    if year in VALIDATION_YEARS:
        return "validation"
    if year in TEST_YEARS:
        return "test"
    raise ValueError(f"Año fuera del protocolo temporal: {year}")
