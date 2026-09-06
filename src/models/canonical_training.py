"""Entrenamiento acotado en memoria para el modelo 2D EGIF.

El datacubo tiene decenas de millones de filas. Este módulo evita concatenarlo
entero: conserva positivos y una muestra determinista de negativos para
entrenar, pero recorre la población completa por lotes para calibrar y evaluar.
La familia histórica de 50 variables puede desplazar la etiqueta a
``issue_date + h``; la familia operativa de 48 representa el día objetivo y
reconstruye ``issue_date`` hacia atrás. El benchmark ERA5 sigue siendo un techo
metodológico, no una evaluación de forecasts archivados.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from src.entrenamiento.calibracion import PlattProbabilityCalibrator, ProbabilityCalibrator
from src.entrenamiento.metricas import evaluate
from src.features.canonical_contract import (
    CANONICAL_FEATURE_SCHEMA_VERSION,
    EGIF_48_FEATURE_CONTRACT_VERSION,
    TARGET_COLUMN,
    ensure_feature_matrix_for_contract,
    load_feature_columns_for_contract,
)
from src.features.operational_benchmark import (
    OPERATIONAL_TEMPORAL_SEMANTICS,
    validate_operational_dataset,
)
from src.models.forecast_risk_model import ForecastRiskModel

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - depende del entorno offline
    LGBMClassifier = None

LOGGER = logging.getLogger(__name__)
HASH_PRIME = 1_000_003
DEFAULT_BATCH_SIZE = 250_000


def _year_path(dataset_dir: Path, year: int) -> Path:
    """Resuelve las dos distribuciones recibidas del datacubo EGIF."""

    candidates = (
        dataset_dir / f"year={year}" / f"dataset_{year}.parquet",
        dataset_dir / f"dataset_{year}.parquet",
        dataset_dir / f"dataset_maestro_{year}.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No se encontró la partición {year}. Se esperaba una de: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def _dataset_for_year(dataset_dir: Path, year: int) -> pads.Dataset:
    return pads.dataset(str(_year_path(dataset_dir, year)), format="parquet")


def _day_number(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values, errors="coerce").to_numpy(dtype="datetime64[D]").astype(np.int64)


def _columns(feature_columns: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(["cell_id", "fecha", TARGET_COLUMN, *feature_columns]))


def _has_complete_feature_row(
    frame: pd.DataFrame,
    feature_columns: list[str],
    contract_version: str,
) -> pd.Series:
    """Indica qué filas tienen todos los predictores finitos y disponibles.

    El primer día del histórico puede carecer de la memoria de 30 días porque
    no existe contexto anterior en el archivo recibido. Esas filas se conservan
    en el dataset para auditoría, pero no deben alimentar un modelo operativo.
    """

    matrix = ensure_feature_matrix_for_contract(
        frame, feature_columns, contract_version
    )
    return matrix.notna().all(axis=1) & np.isfinite(matrix.to_numpy(dtype=float)).all(axis=1)


def _normalise_day(values: pd.Series) -> pd.Series:
    """Normaliza fechas EGIF sin convertir una etiqueta a otro día local."""

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert("Europe/Madrid").dt.normalize().dt.tz_localize(None)


def _load_target_lookup(dataset_dir: Path, years: Iterable[int]) -> pd.DataFrame:
    """Lee solo las etiquetas de ``years`` para construir un target desplazado.

    Se carga por año de features (y su siguiente año) y se libera al terminar
    cada bloque. Así el desplazamiento T+h no obliga a cargar el datacubo
    completo en memoria.
    """

    frames: list[pd.DataFrame] = []
    for year in sorted(set(int(value) for value in years)):
        try:
            dataset = _dataset_for_year(dataset_dir, year)
        except FileNotFoundError:
            continue
        for batch in dataset.scanner(
            columns=["cell_id", "fecha", TARGET_COLUMN],
            batch_size=DEFAULT_BATCH_SIZE,
        ).to_batches():
            frame = batch.to_pandas()
            frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="coerce")
            frame["fecha"] = _normalise_day(frame["fecha"])
            frame[TARGET_COLUMN] = pd.to_numeric(
                frame[TARGET_COLUMN], errors="coerce"
            )
            frames.append(frame.dropna(subset=["cell_id", "fecha", TARGET_COLUMN]))
    if not frames:
        raise FileNotFoundError("No hay etiquetas EGIF para construir el target desplazado.")
    lookup = pd.concat(frames, ignore_index=True)
    lookup["cell_id"] = lookup["cell_id"].astype("int64")
    lookup[TARGET_COLUMN] = lookup[TARGET_COLUMN].astype("int8")
    return lookup.drop_duplicates(["cell_id", "fecha"], keep="last")


def _attach_horizon_target(
    frame: pd.DataFrame,
    target_lookup: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Añade a las features del día de emisión la etiqueta del día T+h."""

    frame = frame.copy()
    frame["fecha"] = _normalise_day(frame["fecha"])
    frame["target_date"] = frame["fecha"] + pd.Timedelta(days=horizon_days)
    labels = target_lookup.rename(
        columns={"fecha": "target_date", TARGET_COLUMN: "_target_horizon"}
    )
    frame = frame.merge(
        labels[["cell_id", "target_date", "_target_horizon"]],
        on=["cell_id", "target_date"],
        how="left",
        validate="many_to_one",
    )
    frame[TARGET_COLUMN] = frame["_target_horizon"]
    frame = frame.drop(columns=["_target_horizon"])
    # Los últimos h días de una partición no tienen etiqueta si el siguiente
    # año no está archivado. No se inventa un cero: el consumidor decide si los
    # excluye del entrenamiento o de la evaluación y conserva la cobertura.
    return frame.reset_index(drop=True)


def _iter_year(
    dataset_dir: Path,
    year: int,
    feature_columns: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    target_lookup: pd.DataFrame | None = None,
    horizon_days: int | None = None,
    target_day_features: bool = False,
) -> Iterator[pd.DataFrame]:
    dataset = _dataset_for_year(dataset_dir, year)
    columns = _columns(feature_columns)
    for batch in dataset.scanner(columns=columns, batch_size=batch_size).to_batches():
        frame = batch.to_pandas()
        frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="coerce").astype("int64")
        if target_day_features:
            if horizon_days not in {1, 2, 3}:
                raise ValueError("horizon_days debe ser 1, 2 o 3 al construir el benchmark.")
            frame["fecha"] = _normalise_day(frame["fecha"])
            frame["target_date"] = frame["fecha"]
            frame["issue_date"] = frame["target_date"] - pd.Timedelta(days=int(horizon_days))
            frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
            frame["target"] = frame[TARGET_COLUMN]
        elif target_lookup is not None:
            if horizon_days not in {1, 2, 3}:
                raise ValueError("horizon_days debe ser 1, 2 o 3 al desplazar etiquetas.")
            frame = _attach_horizon_target(frame, target_lookup, int(horizon_days))
        else:
            frame[TARGET_COLUMN] = (
                pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
                .fillna(0)
                .astype("int8")
            )
        yield frame


def _sample_training_year(
    dataset_dir: Path,
    year: int,
    feature_columns: list[str],
    *,
    horizon_days: int,
    negative_ratio: int,
    remainder: int,
    batch_size: int,
    target_day_features: bool = False,
    feature_contract_version: str = CANONICAL_FEATURE_SCHEMA_VERSION,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if negative_ratio < 1:
        raise ValueError("negative_ratio debe ser un entero positivo.")
    kept: list[pd.DataFrame] = []
    total = positives = negative_rows_total = sampled_negatives = 0
    rows_without_valid_features = 0
    target_lookup = None if target_day_features else _load_target_lookup(dataset_dir, (year, year + 1))
    labelled_rows = 0
    for frame in _iter_year(
        dataset_dir,
        year,
        feature_columns,
        batch_size=batch_size,
        target_lookup=target_lookup,
        horizon_days=horizon_days,
        target_day_features=target_day_features,
    ):
        total += len(frame)
        frame = frame.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        labelled_rows += len(frame)
        if frame.empty:
            continue
        complete = _has_complete_feature_row(
            frame, feature_columns, feature_contract_version
        )
        rows_without_valid_features += int((~complete).sum())
        frame = frame.loc[complete].reset_index(drop=True)
        if frame.empty:
            continue
        target = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        positives += int(target.sum())
        negative_rows_total += int((target == 0).sum())
        hash_row = (
            frame["cell_id"].to_numpy(dtype=np.int64) * HASH_PRIME
            + _day_number(frame["fecha"])
        ) % negative_ratio
        keep = (target == 1) | (hash_row == remainder)
        if keep.any():
            selected = frame.loc[keep].copy()
            sampled_negatives += int((selected[TARGET_COLUMN] == 0).sum())
            kept.append(selected)
    if not kept:
        raise ValueError(f"El muestreo de {year} no produjo filas.")
    result = pd.concat(kept, ignore_index=True)
    return result, {
        "year": int(year),
        "rows_total": total,
        "rows_labelled": labelled_rows,
        "rows_without_future_label": total - labelled_rows,
        "rows_without_valid_features": rows_without_valid_features,
        "positives_total": positives,
        "negatives_total": negative_rows_total,
        "rows_sampled": len(result),
        "negatives_sampled": sampled_negatives,
    }


def _sample_training(
    dataset_dir: Path,
    years: Iterable[int],
    feature_columns: list[str],
    *,
    horizon_days: int,
    negative_ratio: int,
    batch_size: int,
    target_day_features: bool = False,
    feature_contract_version: str = CANONICAL_FEATURE_SCHEMA_VERSION,
) -> tuple[pd.DataFrame, dict[str, object]]:
    years = tuple(int(year) for year in years)
    frames: list[pd.DataFrame] = []
    yearly: list[dict[str, int]] = []
    for year in years:
        frame, meta = _sample_training_year(
            dataset_dir,
            int(year),
            feature_columns,
            horizon_days=horizon_days,
            negative_ratio=negative_ratio,
            remainder=0,
            batch_size=batch_size,
            target_day_features=target_day_features,
            feature_contract_version=feature_contract_version,
        )
        frames.append(frame)
        yearly.append(meta)
        LOGGER.info(
            "EGIF %s: %s filas seleccionadas (%s positivos)",
            year,
            f"{len(frame):,}",
            f"{meta['positives_total']:,}",
        )
    train = pd.concat(frames, ignore_index=True)
    train = train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    total_negatives = sum(item["negatives_total"] for item in yearly)
    sampled_negatives = sum(item["negatives_sampled"] for item in yearly)
    meta = {
        "years": [int(year) for year in years],
        "negative_ratio": int(negative_ratio),
        "yearly": yearly,
        "rows_total": sum(item["rows_total"] for item in yearly),
        "rows_without_valid_features": sum(
            item["rows_without_valid_features"] for item in yearly
        ),
        "positives_total": sum(item["positives_total"] for item in yearly),
        "negatives_total": total_negatives,
        "rows_sampled": len(train),
        "negatives_sampled": sampled_negatives,
        "negative_sampling_rate": sampled_negatives / total_negatives
        if total_negatives
        else 1.0,
    }
    return train, meta


def _predict_years(
    model: object,
    dataset_dir: Path,
    years: Iterable[int],
    feature_columns: list[str],
    *,
    horizon_days: int,
    batch_size: int,
    target_day_features: bool = False,
    feature_contract_version: str = CANONICAL_FEATURE_SCHEMA_VERSION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    days: list[np.ndarray] = []
    for year in years:
        seen = 0
        target_lookup = None if target_day_features else _load_target_lookup(dataset_dir, (int(year), int(year) + 1))
        for frame in _iter_year(
            dataset_dir,
            int(year),
            feature_columns,
            batch_size=batch_size,
            target_lookup=target_lookup,
            horizon_days=horizon_days,
            target_day_features=target_day_features,
        ):
            frame = frame.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
            if frame.empty:
                continue
            complete = _has_complete_feature_row(
                frame, feature_columns, feature_contract_version
            )
            frame = frame.loc[complete].reset_index(drop=True)
            if frame.empty:
                continue
            scores.append(
                model.predict_proba(
                    ensure_feature_matrix_for_contract(
                        frame, feature_columns, feature_contract_version
                    )
                )[:, 1]
            )
            labels.append(frame[TARGET_COLUMN].to_numpy(dtype=np.int8))
            days.append(_day_number(frame["fecha"]))
            seen += len(frame)
        LOGGER.info("Evaluado EGIF %s: %s filas completas", year, f"{seen:,}")
    if not labels:
        raise ValueError("No hay filas para evaluar el modelo canónico.")
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(days)


def _dataset_hash(dataset_dir: Path, years: Iterable[int]) -> str:
    digest = hashlib.sha256()
    metadata_path = dataset_dir / "metadata.json"
    paths = [metadata_path, *[_year_path(dataset_dir, int(year)) for year in years]]
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def train_canonical_horizon_model(
    dataset_dir: str | Path,
    *,
    horizon_days: int,
    output_dir: str | Path = "data/models",
    feature_columns: Iterable[str] | None = None,
    train_years: Iterable[int] = (2019, 2020, 2021),
    validation_year: int = 2022,
    test_years: Iterable[int] = (2023,),
    negative_ratio: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_estimators: int = 400,
    random_state: int = 42,
    bootstrap_samples: int = 200,
    dataset_hash: str | None = None,
    feature_contract_version: str = CANONICAL_FEATURE_SCHEMA_VERSION,
    calibration_year: int | None = None,
    target_day_features: bool = False,
    artifact_prefix: str | None = None,
    model_family: str | None = None,
    experiment_name: str | None = None,
) -> tuple[ForecastRiskModel, dict[str, object]]:
    """Entrena, calibra y serializa un horizonte del modelo EGIF 2D.

    ``target_day_features=True`` exige la exportación histórica alineada: cada
    fila contiene las variables del día objetivo (ERA5 perfecto en el
    benchmark) y sus memorias se calculan hasta el día anterior. El modo por
    defecto mantiene la semántica del artefacto histórico de 50 variables para
    que siga siendo un rollback reproducible.
    """

    if LGBMClassifier is None:
        raise RuntimeError("LightGBM no está instalado en el entorno actual.")
    if horizon_days not in {1, 2, 3}:
        raise ValueError("horizon_days debe ser 1, 2 o 3.")

    root = Path(dataset_dir)
    aligned_metadata: dict[str, object] | None = None
    if target_day_features:
        aligned_metadata = validate_operational_dataset(
            root,
            require_contract=(
                EGIF_48_FEATURE_CONTRACT_VERSION
                if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION
                else None
            ),
        )
    columns = list(
        feature_columns
        or load_feature_columns_for_contract(feature_contract_version, root)
    )
    train_years = tuple(int(year) for year in train_years)
    test_years = tuple(int(year) for year in test_years)
    train, sampling = _sample_training(
        root,
        train_years,
        columns,
        horizon_days=horizon_days,
        negative_ratio=negative_ratio,
        batch_size=batch_size,
        target_day_features=target_day_features,
        feature_contract_version=feature_contract_version,
    )
    x_train = ensure_feature_matrix_for_contract(train, columns, feature_contract_version)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=random_state + horizon_days,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    y_val, p_val, day_val = _predict_years(
        model,
        root,
        (validation_year,),
        columns,
        horizon_days=horizon_days,
        batch_size=batch_size,
        target_day_features=target_day_features,
        feature_contract_version=feature_contract_version,
    )
    calibration_year = int(calibration_year) if calibration_year is not None else None
    calibration_metrics: dict[str, float] | None = None
    if calibration_year is None:
        y_cal, p_cal, day_cal = y_val, p_val, day_val
    else:
        y_cal, p_cal, day_cal = _predict_years(
            model,
            root,
            (calibration_year,),
            columns,
            horizon_days=horizon_days,
            batch_size=batch_size,
            target_day_features=target_day_features,
            feature_contract_version=feature_contract_version,
        )
    calibrator_class = (
        PlattProbabilityCalibrator
        if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION
        else ProbabilityCalibrator
    )
    calibrator = calibrator_class(
        sampling["negative_sampling_rate"], max_fit_points=2_000_000
    ).fit(y_cal, p_cal, seed=random_state + horizon_days)
    p_val_calibrated = np.clip(calibrator.transform(p_val), 0.0, 1.0)
    if calibration_year is not None:
        calibration_metrics = evaluate(
            y_cal,
            p_cal,
            np.clip(calibrator.transform(p_cal), 0.0, 1.0),
            day_cal,
            top_k=0.01,
            n_boot=bootstrap_samples,
            seed=random_state + horizon_days,
        )

    y_test, p_test, day_test = _predict_years(
        model,
        root,
        test_years,
        columns,
        horizon_days=horizon_days,
        batch_size=batch_size,
        target_day_features=target_day_features,
        feature_contract_version=feature_contract_version,
    )
    p_test_calibrated = np.clip(calibrator.transform(p_test), 0.0, 1.0)
    resolved_dataset_hash = dataset_hash or _dataset_hash(
        root,
        (
            *train_years,
            *([calibration_year] if calibration_year is not None else []),
            validation_year,
            *test_years,
        ),
    )
    resolved_prefix = artifact_prefix or "forecast_risk_egif"
    resolved_family = model_family or (
        "egif-2d-48" if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION else "canonical-egif-2d"
    )
    metrics: dict[str, object] = {
        "horizon_days": horizon_days,
        "feature_schema_version": feature_contract_version,
        "feature_contract_version": feature_contract_version,
        "model_family": resolved_family,
        "experiment_name": experiment_name or "unspecified",
        "weather_source": "era5_perfect_benchmark",
        "provider_training_context": {
            "source": "ERA5-Land",
            "mode": "era5_perfect_benchmark",
            "operational_forecast_history_available": False,
        },
        "dataset_version": root.name,
        "alignment_version": (
            aligned_metadata.get("alignment_version") if aligned_metadata else None
        ),
        "static_layer_quality": (
            aligned_metadata.get("static_layer_quality") if aligned_metadata else None
        ),
        "model_version": f"egif-lightgbm-{feature_contract_version}-t{horizon_days}-v1",
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "label_definition": f"target_ignicion en issue_date + {horizon_days} días",
        "train_years": list(train_years),
        "calibration_year": calibration_year,
        "validation_year": int(validation_year),
        "test_years": list(test_years),
        "feature_columns": columns,
        "sampling": sampling,
        "dataset_hash": resolved_dataset_hash,
        "hash_dataset": resolved_dataset_hash,
        "seed": random_state + horizon_days,
        "calibration_method": (
            "prior_correction+platt"
            if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION
            else "prior_correction+isotonic"
        ),
        "training_temporal_semantics": (
            OPERATIONAL_TEMPORAL_SEMANTICS
            if target_day_features
            else "issue_day_features_target_label_shifted"
        ),
        "validation": evaluate(
            y_val,
            p_val,
            p_val_calibrated,
            day_val,
            top_k=0.01,
            n_boot=bootstrap_samples,
            seed=random_state + horizon_days,
        ),
        "test": evaluate(
            y_test,
            p_test,
            p_test_calibrated,
            day_test,
            top_k=0.01,
            n_boot=bootstrap_samples,
            seed=random_state + horizon_days,
        ),
    }
    if calibration_metrics is not None:
        metrics["calibration"] = calibration_metrics
    artifact = ForecastRiskModel(
        base_model=model,
        calibrator=calibrator,
        feature_columns=columns,
        horizon_days=horizon_days,
        metadata=metrics,
        feature_schema_version=feature_contract_version,
        model_family=resolved_family,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / f"{resolved_prefix}_t{horizon_days}.joblib")
    (output / f"{resolved_prefix}_t{horizon_days}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return artifact, metrics


def train_all_canonical_horizons(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path = "data/models",
    include_challenger: bool = False,
    **kwargs: object,
) -> dict[int, dict[str, object]]:
    """Entrena T+1, T+2 y T+3 liberando cada modelo al terminar."""

    reports: dict[int, dict[str, object]] = {}
    for horizon in (1, 2, 3):
        _, report = train_canonical_horizon_model(
            dataset_dir,
            horizon_days=horizon,
            output_dir=output_dir,
            **kwargs,
        )
        reports[horizon] = report
        if include_challenger:
            train_canonical_xgboost_challenger(
                dataset_dir,
                horizon_days=horizon,
                output_dir=output_dir,
                **kwargs,
            )
    return reports


def train_all_egif48_horizons(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path = "data/models",
    train_years: Iterable[int] = (2019, 2020),
    calibration_year: int = 2021,
    validation_year: int = 2022,
    test_years: Iterable[int] = (2023,),
    include_challenger: bool = False,
    artifact_prefix: str = "forecast_risk_egif_48",
    experiment_name: str = "unspecified",
    **kwargs: object,
) -> dict[int, dict[str, object]]:
    """Entrena la familia operativa revisada de 48 variables.

    Los años se interpretan como años del día objetivo. Por eso las filas de
    entrenamiento se construyen con meteorología ERA5 del día objetivo y la
    fecha de emisión queda a ``horizon_days`` días antes, reproduciendo la
    alineación de producción sin fingir que ERA5 sea un forecast archivado.
    """

    return train_all_canonical_horizons(
        dataset_dir,
        output_dir=output_dir,
        include_challenger=include_challenger,
        train_years=train_years,
        calibration_year=calibration_year,
        validation_year=validation_year,
        test_years=test_years,
        feature_contract_version=EGIF_48_FEATURE_CONTRACT_VERSION,
        target_day_features=True,
        artifact_prefix=artifact_prefix,
        model_family="egif-2d-48",
        experiment_name=experiment_name,
        **kwargs,
    )


def train_all_aligned50_horizons(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path = "data/models",
    train_years: Iterable[int] = (2019, 2020),
    calibration_year: int = 2021,
    validation_year: int = 2022,
    test_years: Iterable[int] = (2023,),
    include_challenger: bool = False,
    artifact_prefix: str = "forecast_risk_egif_50_aligned",
    experiment_name: str = "aligned50",
    **kwargs: object,
) -> dict[int, dict[str, object]]:
    """Entrena un control de 50 variables con la semántica temporal operativa.

    Esta familia no se publica automáticamente. Sirve para comparar el
    contrato de 48 variables y el canónico de 50 sobre las mismas memorias,
    manteniendo aparte los artefactos históricos usados para rollback.
    """

    return train_all_canonical_horizons(
        dataset_dir,
        output_dir=output_dir,
        include_challenger=include_challenger,
        train_years=train_years,
        calibration_year=calibration_year,
        validation_year=validation_year,
        test_years=test_years,
        feature_contract_version=CANONICAL_FEATURE_SCHEMA_VERSION,
        target_day_features=True,
        artifact_prefix=artifact_prefix,
        model_family="egif-2d-50-aligned-control",
        experiment_name=experiment_name,
        **kwargs,
    )


def train_canonical_xgboost_challenger(
    dataset_dir: str | Path,
    *,
    horizon_days: int,
    output_dir: str | Path = "data/models",
    feature_columns: Iterable[str] | None = None,
    train_years: Iterable[int] = (2019, 2020, 2021),
    validation_year: int = 2022,
    test_years: Iterable[int] = (2023,),
    negative_ratio: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_estimators: int = 400,
    random_state: int = 42,
    bootstrap_samples: int = 200,
    dataset_hash: str | None = None,
    feature_contract_version: str = CANONICAL_FEATURE_SCHEMA_VERSION,
    calibration_year: int | None = None,
    target_day_features: bool = False,
    artifact_prefix: str | None = None,
    model_family: str | None = None,
    experiment_name: str | None = None,
) -> tuple[ForecastRiskModel, dict[str, object]]:
    """Entrena XGBoost como challenger, sin ponerlo en producción automáticamente."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError("XGBoost no está instalado para entrenar el challenger.") from exc
    if horizon_days not in {1, 2, 3}:
        raise ValueError("horizon_days debe ser 1, 2 o 3.")

    root = Path(dataset_dir)
    aligned_metadata: dict[str, object] | None = None
    if target_day_features:
        aligned_metadata = validate_operational_dataset(
            root,
            require_contract=(
                EGIF_48_FEATURE_CONTRACT_VERSION
                if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION
                else None
            ),
        )
    columns = list(
        feature_columns
        or load_feature_columns_for_contract(feature_contract_version, root)
    )
    train_years = tuple(int(year) for year in train_years)
    test_years = tuple(int(year) for year in test_years)
    train, sampling = _sample_training(
        root,
        train_years,
        columns,
        horizon_days=horizon_days,
        negative_ratio=negative_ratio,
        batch_size=batch_size,
        target_day_features=target_day_features,
        feature_contract_version=feature_contract_version,
    )
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state + horizon_days,
    )
    model.fit(
        ensure_feature_matrix_for_contract(train, columns, feature_contract_version),
        train[TARGET_COLUMN],
    )
    y_val, p_val, day_val = _predict_years(
        model,
        root,
        (validation_year,),
        columns,
        horizon_days=horizon_days,
        batch_size=batch_size,
        target_day_features=target_day_features,
        feature_contract_version=feature_contract_version,
    )
    calibration_year = int(calibration_year) if calibration_year is not None else None
    if calibration_year is None:
        y_cal, p_cal, day_cal = y_val, p_val, day_val
    else:
        y_cal, p_cal, day_cal = _predict_years(
            model,
            root,
            (calibration_year,),
            columns,
            horizon_days=horizon_days,
            batch_size=batch_size,
            target_day_features=target_day_features,
            feature_contract_version=feature_contract_version,
        )
    calibrator_class = (
        PlattProbabilityCalibrator
        if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION
        else ProbabilityCalibrator
    )
    calibrator = calibrator_class(
        sampling["negative_sampling_rate"], max_fit_points=2_000_000
    ).fit(y_cal, p_cal, seed=random_state + horizon_days)
    y_test, p_test, day_test = _predict_years(
        model,
        root,
        test_years,
        columns,
        horizon_days=horizon_days,
        batch_size=batch_size,
        target_day_features=target_day_features,
        feature_contract_version=feature_contract_version,
    )
    p_val_calibrated = np.clip(calibrator.transform(p_val), 0.0, 1.0)
    p_test_calibrated = np.clip(calibrator.transform(p_test), 0.0, 1.0)
    resolved_dataset_hash = dataset_hash or _dataset_hash(
        root,
        (*train_years, *([calibration_year] if calibration_year is not None else []), validation_year, *test_years),
    )
    resolved_prefix = artifact_prefix or "forecast_risk_egif"
    resolved_family = model_family or (
        "egif-2d-48-xgboost-challenger"
        if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION
        else "canonical-egif-2d-xgboost-challenger"
    )
    metrics: dict[str, object] = {
        "horizon_days": horizon_days,
        "feature_schema_version": feature_contract_version,
        "feature_contract_version": feature_contract_version,
        "model_family": resolved_family,
        "experiment_name": experiment_name or "unspecified",
        "weather_source": "era5_perfect_benchmark",
        "provider_training_context": {
            "source": "ERA5-Land",
            "mode": "era5_perfect_benchmark",
            "operational_forecast_history_available": False,
        },
        "dataset_version": root.name,
        "alignment_version": (
            aligned_metadata.get("alignment_version") if aligned_metadata else None
        ),
        "static_layer_quality": (
            aligned_metadata.get("static_layer_quality") if aligned_metadata else None
        ),
        "model_version": f"egif-xgboost-challenger-{feature_contract_version}-t{horizon_days}-v1",
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "label_definition": f"target_ignicion en issue_date + {horizon_days} días",
        "train_years": list(train_years),
        "calibration_year": calibration_year,
        "validation_year": int(validation_year),
        "test_years": list(test_years),
        "feature_columns": columns,
        "sampling": sampling,
        "dataset_hash": resolved_dataset_hash,
        "hash_dataset": resolved_dataset_hash,
        "seed": random_state + horizon_days,
        "calibration_method": (
            "prior_correction+platt"
            if feature_contract_version == EGIF_48_FEATURE_CONTRACT_VERSION
            else "prior_correction+isotonic"
        ),
        "training_temporal_semantics": (
            OPERATIONAL_TEMPORAL_SEMANTICS
            if target_day_features
            else "issue_day_features_target_label_shifted"
        ),
        "validation": evaluate(
            y_val, p_val, p_val_calibrated, day_val,
            top_k=0.01, n_boot=bootstrap_samples, seed=random_state + horizon_days,
        ),
        "test": evaluate(
            y_test, p_test, p_test_calibrated, day_test,
            top_k=0.01, n_boot=bootstrap_samples, seed=random_state + horizon_days,
        ),
    }
    artifact = ForecastRiskModel(
        base_model=model,
        calibrator=calibrator,
        feature_columns=columns,
        horizon_days=horizon_days,
        metadata=metrics,
        feature_schema_version=feature_contract_version,
        model_family=resolved_family,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        artifact,
        output / f"{resolved_prefix}_t{horizon_days}_xgboost_challenger.joblib",
    )
    (output / f"{resolved_prefix}_t{horizon_days}_xgboost_challenger.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return artifact, metrics
