"""Construcción streaming del dataset histórico alineado con producción.

El datacubo canónico se conserva como benchmark retrospectivo. Sus acumulados
incluyen el día de la fila porque fueron calculados antes de exportarse a la
rejilla EGIF. La inferencia, en cambio, necesita memorias disponibles antes del
día objetivo. Este módulo recalcula únicamente esas columnas y publica una
segunda salida auditable sin cargar los 86 millones de filas en memoria.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.canonical_contract import (
    CANONICAL_FEATURE_SCHEMA_VERSION,
    CANONICAL_FEATURES,
)

OPERATIONAL_ALIGNMENT_VERSION = "egif-operational-tminus1-v1"
OPERATIONAL_TEMPORAL_SEMANTICS = (
    "target-day weather + historical/forecast memory through target_date - 1 day"
)
DEFAULT_BATCH_SIZE = 250_000
RAIN_THRESHOLD_MM = 1.0

MEMORY_COLUMNS = (
    "precipitation_sum_3d",
    "precipitation_sum_7d",
    "precipitation_sum_14d",
    "precipitation_sum_30d",
    "temperature_mean_7d",
    "relative_humidity_mean_7d",
    "relative_humidity_mean_14d",
    "wind_speed_mean_7d",
    "consecutive_dry_days",
)

ROLLING_DEFINITIONS = (
    ("precipitation_sum_3d", "precipitation_sum", 3, "sum"),
    ("precipitation_sum_7d", "precipitation_sum", 7, "sum"),
    ("precipitation_sum_14d", "precipitation_sum", 14, "sum"),
    ("precipitation_sum_30d", "precipitation_sum", 30, "sum"),
    ("temperature_mean_7d", "temperature_mean", 7, "mean"),
    ("relative_humidity_mean_7d", "relative_humidity_mean", 7, "mean"),
    ("relative_humidity_mean_14d", "relative_humidity_mean", 14, "mean"),
    ("wind_speed_mean_7d", "wind_speed_mean", 7, "mean"),
)

CONTEXT_COLUMNS = (
    "cell_id",
    "fecha",
    "precipitation_sum",
    "temperature_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "wind_speed_max",
)
REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        "cell_id",
        "fecha",
        "precipitation_sum",
        "temperature_mean",
        "relative_humidity_mean",
        "wind_speed_max",
        "wind_speed_mean",
        "target_ignicion",
    }
)


def _year_path(dataset_dir: Path, year: int) -> Path:
    """Resuelve una partición anual del datacubo recibido."""

    candidates = (
        dataset_dir / f"year={year}" / f"dataset_{year}.parquet",
        dataset_dir / f"dataset_{year}.parquet",
        dataset_dir / f"dataset_maestro_{year}.parquet",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró la partición {year}; se esperaba una de: "
        f"{', '.join(str(candidate) for candidate in candidates)}"
    )


def _available_years(dataset_dir: Path, requested: tuple[int, ...] | None) -> tuple[int, ...]:
    """Devuelve años existentes, o valida los solicitados explícitamente."""

    if requested:
        years = tuple(sorted(set(int(year) for year in requested)))
    else:
        partitioned = [
            int(path.parent.name.split("=", 1)[1])
            for path in dataset_dir.glob("year=*/dataset_*.parquet")
            if "=" in path.parent.name
        ]
        flat = [
            int(path.stem.rsplit("_", 1)[1])
            for path in dataset_dir.glob("dataset_*.parquet")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        ]
        years = tuple(sorted(set(partitioned or flat)))
    if not years:
        raise FileNotFoundError(f"No hay particiones anuales en {dataset_dir}.")
    for year in years:
        _year_path(dataset_dir, year)
    return years


def _normalise_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("Europe/Madrid").dt.normalize().dt.tz_localize(None)
    # Las fechas diarias del datacubo son fechas civiles sin zona horaria. No
    # deben interpretarse primero como UTC porque eso las desplazaría al día
    # anterior en Europe/Madrid.
    return parsed.dt.normalize()


def _hash_files(metadata_path: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [metadata_path, *paths]:
        digest.update(str(path.name).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_source_metadata(source_dir: Path) -> dict[str, object]:
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Falta el metadata del datacubo: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared = metadata.get("predictor_columns")
    if list(declared or []) != sorted(CANONICAL_FEATURES):
        raise ValueError(
            "El datacubo de entrada no publica exactamente los 50 predictores canónicos."
        )
    return metadata


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas necesarias en el Parquet: {missing}")
    frame = frame.copy()
    frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="coerce")
    frame["fecha"] = _normalise_dates(frame["fecha"])
    if frame[["cell_id", "fecha"]].isna().any().any():
        raise ValueError("El Parquet contiene cell_id o fecha inválidos.")
    return frame.sort_values(["cell_id", "fecha"], kind="stable").reset_index(drop=True)


def _previous_features(context: pd.DataFrame) -> pd.DataFrame:
    """Calcula memorias de cada fila usando estrictamente el pasado."""

    context = context.sort_values(["cell_id", "fecha"], kind="stable").reset_index(drop=True)
    grouped = context.groupby("cell_id", sort=False)
    for name, source, window, aggregation in ROLLING_DEFINITIONS:
        context[name] = grouped[source].transform(
            lambda values, w=window, agg=aggregation: values.shift(1)
            .rolling(w, min_periods=1)
            .agg(agg)
        )

    dry = context["precipitation_sum"].notna() & context["precipitation_sum"].lt(
        RAIN_THRESHOLD_MM
    )
    segments = (~dry).groupby(context["cell_id"], sort=False).cumsum()
    streak = dry.astype("int64").groupby(
        [context["cell_id"], segments], sort=False
    ).cumsum()
    context["consecutive_dry_days"] = (
        streak.groupby(context["cell_id"], sort=False).shift(1).fillna(0).astype("float64")
    )
    return context.loc[context["_current"], ["cell_id", "fecha", *MEMORY_COLUMNS]].copy()


def _align_batch(frame: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    """Alinea un lote con la cola de hasta 30 días del lote anterior."""

    frame = _validate_frame(frame)
    original_columns = list(frame.columns)
    current = frame[list(CONTEXT_COLUMNS)].copy()
    current["_current"] = True
    previous = tail.copy()
    if not previous.empty:
        previous["fecha"] = pd.to_datetime(previous["fecha"], errors="coerce")
    else:
        previous = previous.astype({"fecha": "datetime64[ns]"})
    previous["_current"] = False
    context = pd.concat([previous, current], ignore_index=True)
    context = context.sort_values(["cell_id", "fecha", "_current"], kind="stable")
    context = context.drop_duplicates(["cell_id", "fecha"], keep="last").reset_index(drop=True)
    memory = _previous_features(context)

    aligned = frame.drop(columns=[column for column in MEMORY_COLUMNS if column in frame], errors="ignore")
    aligned = aligned.merge(memory, on=["cell_id", "fecha"], how="left", validate="one_to_one")
    for column in MEMORY_COLUMNS:
        if column not in original_columns:
            original_columns.append(column)
    aligned = aligned[original_columns]
    aligned["cell_id"] = aligned["cell_id"].astype("int64")
    return aligned


def _tail_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[list(CONTEXT_COLUMNS)]
        .sort_values(["cell_id", "fecha"], kind="stable")
        .groupby("cell_id", sort=False, group_keys=False)
        .tail(30)
        .reset_index(drop=True)
    )


def _write_aligned_year(
    source_path: Path,
    destination_path: Path,
    tail: pd.DataFrame,
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    rows = 0
    cells_seen: set[int] = set()
    first_date = last_date = None
    final_tail = tail
    try:
        parquet = pq.ParquetFile(source_path)
        expected_rows = parquet.metadata.num_rows
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            aligned = _align_batch(record_batch.to_pandas(), final_tail)
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(destination_path, schema)
            elif table.schema != schema:
                table = table.cast(schema, safe=False)
            writer.write_table(table)
            # La cola debe acumularse entre lotes. Mantener solo el lote actual
            # perdería el contexto de las primeras celdas cuando un Parquet
            # divide sus filas en fragmentos pequeños.
            final_tail = _tail_from_frame(pd.concat([final_tail, aligned], ignore_index=True))
            rows += len(aligned)
            cells_seen.update(aligned["cell_id"].astype("int64").unique().tolist())
            date_values = pd.to_datetime(aligned["fecha"])
            first_date = min(first_date, date_values.min()) if first_date is not None else date_values.min()
            last_date = max(last_date, date_values.max()) if last_date is not None else date_values.max()
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise ValueError(f"La partición de entrada está vacía: {source_path}")
    if rows != expected_rows:
        raise ValueError(
            f"La alineación cambió el número de filas de {source_path}: "
            f"{rows} frente a {expected_rows}."
        )
    return final_tail, {
        "rows": int(rows),
        "active_cells": len(cells_seen),
        "first_date": str(first_date.date()),
        "last_date": str(last_date.date()),
    }


def build_operational_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    years: tuple[int, ...] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    static_layer_quality: str = "provisional_import",
    force: bool = False,
) -> Path:
    """Construye la exportación histórica alineada sin tocar el datacubo.

    Args:
        source_dir: Directorio del datacubo canónico con metadata y particiones.
        output_dir: Destino versionado del dataset operativo.
        years: Años a transformar; por defecto, todos los disponibles.
        batch_size: Filas por lote de lectura de Parquet.
        static_layer_quality: Calidad declarada de las capas estáticas.
        force: Permite reemplazar una salida previa completa.

    Returns:
        Ruta del dataset operativo publicado.
    """

    source = Path(source_dir)
    destination = Path(output_dir)
    if batch_size < 1:
        raise ValueError("batch_size debe ser positivo.")
    source_metadata = _validate_source_metadata(source)
    selected_years = _available_years(source, years)
    source_paths = [_year_path(source, year) for year in selected_years]
    if destination.exists() and not force:
        raise FileExistsError(
            f"Ya existe {destination}. Usa --force solo después de revisar la salida."
        )

    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    stats: dict[str, dict[str, object]] = {}
    tail = pd.DataFrame(columns=list(CONTEXT_COLUMNS))
    try:
        for year, source_path in zip(selected_years, source_paths):
            destination_path = temporary / f"year={year}" / f"dataset_{year}.parquet"
            tail, year_stats = _write_aligned_year(
                source_path,
                destination_path,
                tail,
                batch_size=batch_size,
            )
            stats[str(year)] = year_stats
        parent_hash = _hash_files(source / "metadata.json", source_paths)
        metadata = {
            "dataset_version": "egif_operational_tminus1_v1",
            "feature_contract_version": CANONICAL_FEATURE_SCHEMA_VERSION,
            "operational_feature_contract_version": "egif-2d-48-v1",
            "predictor_columns": list(sorted(CANONICAL_FEATURES)),
            "alignment_version": OPERATIONAL_ALIGNMENT_VERSION,
            "training_temporal_semantics": OPERATIONAL_TEMPORAL_SEMANTICS,
            "source_datacube": str(source),
            "parent_dataset_hash": parent_hash,
            "years": list(selected_years),
            "row_counts": {year: value["rows"] for year, value in stats.items()},
            "active_cells": max(value["active_cells"] for value in stats.values()),
            "dropped_rows": 0,
            "dropped_rows_incomplete_predictors": 0,
            "static_layer_quality": static_layer_quality,
            "source_time_contract": source_metadata.get("time_contract"),
            "weather_benchmark": "era5_perfect_benchmark",
            "memory_columns_recomputed": list(MEMORY_COLUMNS),
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "annual_files": stats,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        if destination.exists():
            if not force:
                raise FileExistsError(f"La salida apareció durante la construcción: {destination}")
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def validate_operational_dataset(
    dataset_dir: str | Path,
    *,
    require_contract: str | None = None,
) -> dict[str, object]:
    """Valida que un dataset sea apto para entrenamiento sin fuga temporal."""

    root = Path(dataset_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Falta metadata.json en el dataset operativo: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("alignment_version") != OPERATIONAL_ALIGNMENT_VERSION:
        raise ValueError(
            "El dataset no tiene la alineación temporal operativa requerida "
            f"({OPERATIONAL_ALIGNMENT_VERSION})."
        )
    if metadata.get("training_temporal_semantics") != OPERATIONAL_TEMPORAL_SEMANTICS:
        raise ValueError("El dataset no declara la semántica temporal operativa esperada.")
    # La fuente puede ser inclusiva; esta capa solo es válida si deja constancia
    # explícita de que todas las memorias fueron recalculadas con frontera T-1.
    if set(metadata.get("memory_columns_recomputed", [])) != set(MEMORY_COLUMNS):
        raise ValueError("El dataset conserva memorias inclusivas sin recalcular.")
    if require_contract == "egif-2d-48-v1" and metadata.get(
        "operational_feature_contract_version"
    ) != "egif-2d-48-v1":
        raise ValueError("El dataset no declara compatibilidad con el contrato EGIF de 48 variables.")
    return metadata
