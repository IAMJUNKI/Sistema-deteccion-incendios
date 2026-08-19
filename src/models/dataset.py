"""Carga, reparación temporal y muestreo del dataset maestro de incendios.

Este módulo es la única puerta de entrada a los `dataset_maestro_YYYY.parquet`. Se encarga de:

1. **Resolver la ruta de los datos** desde `DATA_ROOT` (variable de entorno o `.env`), de forma
   que ninguna ruta quede escrita a fuego en el código.
2. **Corregir el desfase temporal** documentado en `MODEL_DOCUMENTATION.md` §2.6: la columna
   `fecha` de los ficheros está adelantada un día respecto al evento real, y la meteorología
   corresponde al mismo día del incendio (no al anterior).
3. **Generar las dos variantes del dataset** que compara el TFM:
   - ``nowcast``: la meteorología del día del incendio (diagnóstico).
   - ``t1``: la meteorología del día anterior (previsión a 24 h, objetivo del proyecto).
4. **Construir el conjunto de entrenamiento** conservando el 100 % de los positivos y
   submuestreando negativos con *hard negative mining* real.

Los ficheros se procesan **por bloques de celdas** porque un año completo son 11,2 millones de
filas: cargarlos todos a la vez agota la memoria de un portátil.
"""

import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Esquema del dataset ──────────────────────────────────────────────────────

CELL_COL = "cell_id"
DATE_COL = "fecha"
DATE_REAL_COL = "fecha_real"
TARGET_COL = "target"

#: Variables meteorológicas. Son las que se desplazan un día en la variante ``t1``.
#: `alerta_30_30` se incluye porque es una regla derivada de `tmax_vc` y `rhmin_vc` del mismo
#: día: si no se desplazase con ellas, filtraría información del día D en el modelo de previsión.
METEO_COLS = [
    "tmax_vc",
    "rhmin_vc",
    "vmax_vc",
    "prec_dia",
    "prec_acum_7d",
    "prec_acum_30d",
    "tmax_media_7d",
    "alerta_30_30",
]

#: Variables estáticas del terreno: constantes por celda, nunca se desplazan.
STATIC_COLS = [
    "altitud_media",
    "pendiente_media",
    "orientacion_media",
    "combustible_pct_forestal",
]

#: Variables categóricas. Los experimentos anteriores del repositorio las descartaban por ser
#: strings; aquí se recuperan como categóricas nativas de LightGBM/XGBoost.
CATEGORICAL_COLS = ["combustible_clase", "orientacion_clase"]

#: Categorías fijadas explícitamente para que train, validación y test compartan la misma
#: codificación aunque a un bloque concreto le falte alguna clase.
CATEGORIES = {
    "combustible_clase": [
        "urbano",
        "agricola",
        "bosque_frondosas",
        "bosque_coniferas",
        "bosque_mixto",
        "pastizal",
        "matorral",
        "agua_humedal",
        "otros",
        "sin_dato",
    ],
    "orientacion_clase": ["norte", "este", "sur", "oeste", "plana", "sin_dato"],
}

#: Variables de calendario. Ya vienen calculadas sobre la fecha real del evento (verificado:
#: `dia_semana == (fecha - 1 día).dayofweek` al 100 %), así que **no** hay que recalcularlas.
CALENDAR_COLS = ["mes", "dia_semana", "es_finde"]

#: Estacionalidad cíclica derivada. Se añade porque `mes` como entero rompe la continuidad entre
#: diciembre y enero, que meteorológicamente son días contiguos.
CYCLIC_COLS = ["dia_anio_sin", "dia_anio_cos"]

MODES = ("nowcast", "t1")

#: Columnas que existen físicamente en el parquet. Todo lo demás se calcula al vuelo.
PARQUET_COLS = (
    [CELL_COL, DATE_COL, TARGET_COL]
    + METEO_COLS
    + STATIC_COLS
    + CATEGORICAL_COLS
    + CALENDAR_COLS
)

#: Días del año anterior que se cargan como margen cuando se piden variables derivadas. Los
#: índices secuenciales (días sin lluvia, Nesterov) necesitan historia previa: con un solo día
#: de margen, cada 1 de enero reiniciaría el contador y falsearía los eneros.
MARGIN_DAYS_DERIVED = 40


def feature_columns(
    cyclic: bool = True,
    derived: bool = False,
    explicit: Optional[list[str]] = None,
) -> list[str]:
    """Devuelve la lista ordenada de variables explicativas del modelo.

    Args:
        cyclic: Si se incluyen `dia_anio_sin` y `dia_anio_cos`.
        derived: Si se incluyen las variables de `src.features.derived`.
        explicit: Lista explícita del fichero de configuración. Si se aporta, manda sobre todo
            lo demás y se devuelve tal cual.

    Returns:
        Lista de nombres de columna que consume el modelo.
    """
    if explicit:
        return list(explicit)

    cols = METEO_COLS + STATIC_COLS + CATEGORICAL_COLS + CALENDAR_COLS
    if cyclic:
        cols = cols + list(CYCLIC_COLS)
    if derived:
        from src.features.derived import derived_columns

        cols = cols + derived_columns()
    return cols


# ─── Resolución de rutas ──────────────────────────────────────────────────────


def _read_dotenv(ruta: Path) -> dict[str, str]:
    """Lee un `.env` sencillo sin depender de python-dotenv."""
    valores: dict[str, str] = {}
    if not ruta.exists():
        return valores
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        valores[clave.strip()] = valor.split("#")[0].strip().strip('"').strip("'")
    return valores


def resolve_data_root(explicit: Optional[Union[str, Path]] = None) -> Path:
    """Localiza el directorio que contiene los `dataset_maestro_YYYY.parquet`.

    Orden de precedencia: argumento explícito, variable de entorno `DATA_ROOT`, clave
    `DATA_ROOT` del `.env` del repositorio, y ubicación por defecto dentro de `data/raw/`.

    Args:
        explicit: Ruta indicada por el usuario, si la hay.

    Returns:
        Directorio donde viven los parquets.

    Raises:
        FileNotFoundError: Si no se encuentra ningún dataset en las rutas candidatas.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidatas: list[Path] = []

    if explicit:
        candidatas.append(Path(explicit))
    if os.environ.get("DATA_ROOT"):
        candidatas.append(Path(os.environ["DATA_ROOT"]))

    env = _read_dotenv(repo_root / ".env")
    if env.get("DATA_ROOT"):
        candidatas.append(Path(env["DATA_ROOT"]))

    candidatas.append(repo_root / "data" / "raw" / "dataset_maestro")
    # Último recurso: la carpeta de descargas, donde suelen quedar tras compartirlos.
    candidatas.append(Path(os.path.expanduser("~")) / "Downloads")

    for c in candidatas:
        if c.is_dir() and any(c.glob("dataset_maestro_*.parquet")):
            if "Downloads" in str(c):
                logger.warning(
                    "Usando los datasets desde %s. Conviene moverlos a data/raw/dataset_maestro/ "
                    "y fijar DATA_ROOT en el .env: una carpeta de descargas se vacía sin querer.",
                    c,
                )
            return c

    raise FileNotFoundError(
        "No se han encontrado los dataset_maestro_*.parquet. Rutas probadas: "
        + ", ".join(str(c) for c in candidatas)
        + ". Define DATA_ROOT en el .env del repositorio."
    )


def year_path(year: int, data_root: Optional[Path] = None) -> Path:
    """Devuelve la ruta del parquet de un año concreto."""
    root = data_root or resolve_data_root()
    return root / f"dataset_maestro_{year}.parquet"


def expected_rows(year: int, data_root: Optional[Path] = None) -> int:
    """Número de filas que debe tener un año, leído de los metadatos del parquet.

    Sirve como referencia independiente para comprobar que una evaluación ha recorrido el año
    entero. Es una lectura de metadatos, no de datos: cuesta milisegundos.

    Args:
        year: Año consultado.
        data_root: Directorio de los parquets.

    Returns:
        Total de filas del fichero de ese año.
    """
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(year_path(year, data_root)).metadata.num_rows)


def available_years(data_root: Optional[Path] = None) -> list[int]:
    """Lista los años para los que existe un parquet."""
    root = data_root or resolve_data_root()
    años = []
    for p in sorted(root.glob("dataset_maestro_*.parquet")):
        try:
            años.append(int(p.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return años


# ─── Preparación de bloques ───────────────────────────────────────────────────


def _prepare_block(df: pd.DataFrame, mode: str, cyclic: bool = True) -> pd.DataFrame:
    """Aplica la corrección temporal a un bloque de celdas ya cargado.

    Convierte `fecha` (adelantada un día) en `fecha_real`, y en modo ``t1`` desplaza las
    variables meteorológicas una posición dentro de cada celda para que el modelo solo vea
    información anterior al día que predice.

    Args:
        df: Bloque con todas las filas de un rango de celdas, incluido el margen del año previo.
        mode: ``"nowcast"`` o ``"t1"``.
        cyclic: Si se generan las variables de estacionalidad cíclica.

    Returns:
        El bloque preparado, ordenado por celda y fecha.
    """
    if mode not in MODES:
        raise ValueError(f"mode debe ser uno de {MODES}, recibido: {mode!r}")

    df = df.sort_values([CELL_COL, DATE_COL]).reset_index(drop=True)

    if mode == "t1":
        # La fila de fecha F contiene la meteo del día F-1. Tomando la meteo de la fila
        # anterior (fecha F-1) obtenemos la del día F-2, es decir el día previo al evento.
        presentes = [c for c in METEO_COLS if c in df.columns]
        df[presentes] = df.groupby(CELL_COL, sort=False)[presentes].shift(1)

    # La fecha del fichero está adelantada un día respecto al evento real (ver §2.6).
    df[DATE_REAL_COL] = df[DATE_COL] - pd.Timedelta(days=1)

    if cyclic:
        doy = df[DATE_REAL_COL].dt.dayofyear.to_numpy()
        df["dia_anio_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
        df["dia_anio_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = pd.Categorical(
                df[col].fillna("sin_dato"), categories=CATEGORIES[col]
            )

    return df


def iter_blocks(
    year: int,
    columns: list[str],
    mode: str,
    block_size: int = 4000,
    data_root: Optional[Path] = None,
    cyclic: bool = True,
    derived: Optional[dict] = None,
) -> Iterator[pd.DataFrame]:
    """Recorre un año por bloques de celdas, ya preparados y recortados a ese año.

    Para poder desplazar la meteorología sin perder el 1 de enero, cada bloque carga también el
    último día del fichero del año anterior. Ese margen se descarta antes de devolver el bloque,
    de modo que la población evaluada es idéntica en las dos variantes del dataset.

    Args:
        year: Año a recorrer.
        columns: Columnas del parquet que se necesitan (se añaden las obligatorias).
        mode: ``"nowcast"`` o ``"t1"``.
        block_size: Número de celdas por bloque.
        data_root: Directorio de los parquets.
        cyclic: Si se generan las variables de estacionalidad cíclica.

    Yields:
        Bloques con `fecha_real` dentro del año pedido.
    """
    root = data_root or resolve_data_root()
    ruta = year_path(year, root)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el dataset del año {year}: {ruta}")

    # Se leen siempre todas las columnas del parquet: la diferencia de coste frente a leer un
    # subconjunto es marginal, y evita toda una clase de errores en los que una variable
    # derivada se queda sin la columna base con la que se calcula.
    cols = list(PARQUET_COLS)

    ruta_previa = year_path(year - 1, root)
    # El fichero del año Y-1 termina el 1 de enero de Y. Sin derivadas basta ese día para poder
    # desplazar la meteorología; con derivadas hacen falta varias semanas de historia.
    dias_margen = MARGIN_DAYS_DERIVED if derived is not None else 1
    fecha_margen = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=dias_margen - 1)

    total_celdas = 30_697
    for lo in range(0, total_celdas, block_size):
        hi = lo + block_size
        filtro = [(CELL_COL, ">=", lo), (CELL_COL, "<", hi)]

        piezas = []
        if ruta_previa.exists():
            margen = pd.read_parquet(
                ruta_previa, columns=cols, filters=filtro + [(DATE_COL, ">=", fecha_margen)]
            )
            if len(margen):
                piezas.append(margen)
        piezas.append(pd.read_parquet(ruta, columns=cols, filters=filtro))

        bloque = pd.concat(piezas, ignore_index=True)
        if bloque.empty:
            continue

        bloque = _prepare_block(bloque, mode=mode, cyclic=cyclic)

        # Las derivadas se calculan ANTES de recortar el año, para que los índices secuenciales
        # lleguen a enero ya calentados con la historia del año anterior.
        if derived is not None:
            from src.features.derived import add_derived_features

            bloque = add_derived_features(
                bloque, derived.get("climatology"), derived.get("daily_stats")
            )

        bloque = bloque[bloque[DATE_REAL_COL].dt.year == year]
        if len(bloque):
            yield bloque.reset_index(drop=True)


# ─── Muestreo del conjunto de entrenamiento ───────────────────────────────────


def _hard_negative_mask(
    df: pd.DataFrame,
    prec_max: float = 5.0,
    tmax_min: float = 20.0,
    rhmin_max: float = 60.0,
) -> np.ndarray:
    """Marca los negativos *difíciles*: días con condiciones de incendio en los que no ardió.

    Un muestreo aleatorio de negativos llena el entrenamiento de noches de enero lluviosas, que
    el modelo separa trivialmente. El gradiente se gasta aprendiendo que en invierno no arde.
    Forzando negativos con condiciones de riesgo, el modelo tiene que aprender la frontera real.

    Los umbrales por defecto salen del perfil de los 1.761 incendios del histórico: temperatura
    máxima media 25,0 °C, humedad mínima media 40,1 % y 96 % de ellos con menos de 5 mm de lluvia.

    Args:
        df: Bloque preparado.
        prec_max: Precipitación por debajo de la cual se considera día seco (mm).
        tmax_min: Temperatura máxima mínima para considerar el día cálido (°C).
        rhmin_max: Humedad relativa mínima por debajo de la cual el aire está seco (%).

    Returns:
        Máscara booleana de negativos difíciles.
    """
    es_negativo = df[TARGET_COL].to_numpy() == 0
    # Las comparaciones con NaN devuelven False, así que las filas sin meteo (el primer día de
    # cada celda en modo t1) quedan clasificadas como negativos fáciles, que es lo correcto.
    duro = (
        (df["prec_dia"].to_numpy() < prec_max)
        & (df["tmax_vc"].to_numpy() >= tmax_min)
        & (df["rhmin_vc"].to_numpy() <= rhmin_max)
    )
    return es_negativo & duro


def build_training_set(
    years: list[int],
    mode: str,
    neg_ratio: int = 50,
    hard_fraction: float = 0.8,
    seed: int = 42,
    block_size: int = 4000,
    data_root: Optional[Path] = None,
    cyclic: bool = True,
    hard_kwargs: Optional[dict] = None,
    derived: Optional[dict] = None,
    columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Construye el conjunto de entrenamiento con submuestreo dirigido de negativos.

    Conserva el 100 % de los positivos y muestrea negativos hasta la proporción `neg_ratio`,
    tomando la mayor parte de ellos del subconjunto de negativos difíciles. Se recorre el dataset
    dos veces: la primera para contar y fijar las tasas de muestreo, la segunda para muestrear.

    Args:
        years: Años de entrenamiento.
        mode: ``"nowcast"`` o ``"t1"``.
        neg_ratio: Negativos por cada positivo.
        hard_fraction: Fracción del presupuesto de negativos que sale de los difíciles.
        seed: Semilla del generador aleatorio.
        block_size: Celdas por bloque.
        data_root: Directorio de los parquets.
        cyclic: Si se generan las variables cíclicas.
        hard_kwargs: Umbrales alternativos para `_hard_negative_mask`.

    Returns:
        Tupla con el DataFrame de entrenamiento y un diccionario de metadatos del muestreo
        (incluye `neg_sampling_rate`, necesario después para corregir el sesgo de prior).
    """
    root = data_root or resolve_data_root()
    cols = columns or feature_columns(cyclic=cyclic, derived=derived is not None)
    hard_kwargs = hard_kwargs or {}

    # ── Pasada 1: contar positivos y negativos (duros y fáciles) ──
    # El conteo solo necesita el objetivo y las tres variables del criterio de dificultad, así
    # que se recorre sin calcular derivadas: son caras y aquí no se usan.
    logger.info("Pasada 1/2 — contando clases en %s (modo %s)...", years, mode)
    cols_conteo = [TARGET_COL, "prec_dia", "tmax_vc", "rhmin_vc"]
    n_pos = n_hard = n_easy = 0
    for year in years:
        for bloque in iter_blocks(
            year, cols_conteo, mode, block_size, root, cyclic=False
        ):
            duro = _hard_negative_mask(bloque, **hard_kwargs)
            pos = int((bloque[TARGET_COL] == 1).sum())
            n_pos += pos
            n_hard += int(duro.sum())
            n_easy += len(bloque) - pos - int(duro.sum())

    if n_pos == 0:
        raise ValueError(f"No hay positivos en los años {years}.")

    presupuesto = neg_ratio * n_pos
    cupo_duro = min(int(presupuesto * hard_fraction), n_hard)
    cupo_facil = min(presupuesto - cupo_duro, n_easy)
    tasa_dura = cupo_duro / max(n_hard, 1)
    tasa_facil = cupo_facil / max(n_easy, 1)
    # Tasa global de conservación de negativos: es el factor que usa la corrección de prior.
    tasa_global = (cupo_duro + cupo_facil) / max(n_hard + n_easy, 1)

    logger.info(
        "  positivos=%s | negativos duros=%s (tasa %.4f%%) | fáciles=%s (tasa %.4f%%)",
        f"{n_pos:,}",
        f"{n_hard:,}",
        tasa_dura * 100,
        f"{n_easy:,}",
        tasa_facil * 100,
    )

    # ── Pasada 2: muestrear ──
    logger.info("Pasada 2/2 — muestreando...")
    rng = np.random.default_rng(seed)
    piezas = []
    for year in years:
        for bloque in iter_blocks(
            year, cols, mode, block_size, root, cyclic=cyclic, derived=derived
        ):
            duro = _hard_negative_mask(bloque, **hard_kwargs)
            es_pos = bloque[TARGET_COL].to_numpy() == 1
            sorteo = rng.random(len(bloque))
            conservar = es_pos | (duro & (sorteo < tasa_dura))
            facil = (~es_pos) & (~duro)
            conservar |= facil & (sorteo < tasa_facil)
            if conservar.any():
                piezas.append(bloque.loc[conservar])

    train = pd.concat(piezas, ignore_index=True)
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    meta = {
        "mode": mode,
        "years": list(years),
        "n_rows": int(len(train)),
        "n_positives": int(train[TARGET_COL].sum()),
        "n_positives_universe": int(n_pos),
        "n_negatives_universe": int(n_hard + n_easy),
        "neg_ratio_target": neg_ratio,
        "hard_fraction": hard_fraction,
        "neg_sampling_rate": float(tasa_global),
        "true_prevalence": float(n_pos / (n_pos + n_hard + n_easy)),
        "seed": seed,
    }
    logger.info(
        "  entrenamiento: %s filas | %s positivos | prevalencia real %.2e",
        f"{meta['n_rows']:,}",
        f"{meta['n_positives']:,}",
        meta["true_prevalence"],
    )
    return train, meta


def load_eval_year(
    year: int,
    mode: str,
    block_size: int = 4000,
    data_root: Optional[Path] = None,
    cyclic: bool = True,
    derived: Optional[dict] = None,
    columns: Optional[list[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Recorre un año completo, sin submuestrear, para evaluación o calibración.

    La evaluación se hace siempre sobre el año íntegro: submuestrear el test cambiaría la
    prevalencia y haría que las métricas dejasen de ser comparables entre experimentos.
    """
    cols = columns or feature_columns(cyclic=cyclic, derived=derived is not None)
    yield from iter_blocks(
        year, cols, mode, block_size, data_root, cyclic, derived=derived
    )
