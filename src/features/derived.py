"""Variables derivadas para el modelado de riesgo de ignición.

El dataset maestro entrega magnitudes meteorológicas **absolutas** (temperatura, humedad,
viento, lluvia). El diagnóstico del primer entrenamiento mostró que con ellas el modelo aprende
sobre todo estacionalidad: distingue agosto de enero con ROC-AUC 0,94, pero al restringir la
evaluación a un solo día —alertar el 1 % de celdas más peligrosas de hoy— el recall se desploma
al 4 %. Operativamente eso es lo contrario de lo que hace falta: una brigada no necesita que le
digan que en verano hay más riesgo.

Este módulo construye tres familias de variables pensadas para atacar ese problema:

1. **Anomalías frente a la climatología de la celda.** ¿Hace más calor del que suele hacer
   *aquí* en *este mes*? Elimina el sesgo de altitud y latitud: 28 °C en la sierra de Ancares
   es una anomalía extrema, y en Ourense es un martes de julio.
2. **Posición relativa dentro del día.** ¿Cómo de seca está esta celda comparada con el resto de
   Galicia *hoy*? Es señal puramente espacial: al construirse día a día, la estacionalidad
   desaparece por completo.
3. **Índices físicos de sequedad del combustible.** VPD, días sin lluvia encadenados, índice de
   Nesterov y ratio térmico-eólico. Combinan varias magnitudes en la forma en que la física del
   fuego dice que interactúan, en lugar de dejar que el modelo lo descubra desde cero con solo
   840 incendios.

Las estadísticas de referencia (climatología y estadísticos diarios) se aprenden **solo con los
años de entrenamiento** y se aplican después a validación y test, igual que cualquier otro
parámetro ajustado.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CELL_COL = "cell_id"
DATE_COL = "fecha_real"

#: Variables sobre las que se calculan anomalías y posiciones relativas.
BASE_VARS = ["tmax_vc", "rhmin_vc", "vmax_vc", "prec_acum_7d"]

UMBRAL_LLUVIA_MM = 1.0
UMBRAL_RESET_NESTEROV_MM = 3.0


# ─── Estadísticos de referencia ───────────────────────────────────────────────


def fit_cell_climatology(bloques, vars_base: Optional[list[str]] = None) -> pd.DataFrame:
    """Aprende la climatología media por celda y mes a partir de los años de entrenamiento.

    Se agrega por mes y no por día del año porque con tres o cuatro años de histórico un
    promedio por día concreto tendría tres o cuatro muestras: puro ruido. Por mes son del orden
    de un centenar, suficientes para una media estable.

    Args:
        bloques: Iterable de DataFrames preparados (los que produce `src.models.dataset`).
        vars_base: Variables a promediar. Por defecto, `BASE_VARS`.

    Returns:
        DataFrame con índice (cell_id, mes) y una columna `<var>_clim` por variable.
    """
    vars_base = vars_base or BASE_VARS
    sumas = []
    for bloque in bloques:
        cols = [v for v in vars_base if v in bloque.columns]
        g = bloque.groupby([CELL_COL, "mes"])[cols].agg(["sum", "count"])
        sumas.append(g)

    total = pd.concat(sumas).groupby(level=[0, 1]).sum()
    clim = pd.DataFrame(index=total.index)
    for v in vars_base:
        if (v, "sum") in total.columns:
            clim[f"{v}_clim"] = total[(v, "sum")] / total[(v, "count")].replace(0, np.nan)
    logger.info("Climatología por celda y mes: %s filas", f"{len(clim):,}")
    return clim


def fit_daily_stats(bloques, vars_base: Optional[list[str]] = None) -> pd.DataFrame:
    """Aprende la media y la desviación típica espacial de cada día.

    Permite después expresar cada celda como su posición relativa dentro del mapa de ese día,
    que es la señal que sobrevive cuando se elimina la estacionalidad.

    Args:
        bloques: Iterable de DataFrames preparados.
        vars_base: Variables a resumir. Por defecto, `BASE_VARS`.

    Returns:
        DataFrame indexado por fecha con columnas `<var>_media_dia` y `<var>_std_dia`.
    """
    vars_base = vars_base or BASE_VARS
    partes = []
    for bloque in bloques:
        cols = [v for v in vars_base if v in bloque.columns]
        g = bloque.groupby(DATE_COL)[cols].agg(["sum", "count", lambda x: (x**2).sum()])
        partes.append(g)

    total = pd.concat(partes).groupby(level=0).sum()
    stats = pd.DataFrame(index=total.index)
    for v in vars_base:
        if (v, "sum") not in total.columns:
            continue
        n = total[(v, "count")].replace(0, np.nan)
        media = total[(v, "sum")] / n
        # Varianza a partir de sumas acumuladas, para poder agregarla por bloques.
        var = (total[(v, "<lambda_0>")] / n) - media**2
        stats[f"{v}_media_dia"] = media
        stats[f"{v}_std_dia"] = np.sqrt(var.clip(lower=0))
    logger.info("Estadísticos diarios: %s días", f"{len(stats):,}")
    return stats


# ─── Índices físicos ──────────────────────────────────────────────────────────


def vapor_pressure_deficit(tmax_c: np.ndarray, rhmin_pct: np.ndarray) -> np.ndarray:
    """Déficit de presión de vapor (kPa): la "sed" que tiene la atmósfera.

    Diferencia entre el vapor que el aire podría contener a esa temperatura y el que realmente
    contiene. Predice la desecación del combustible fino mejor que la humedad relativa sola,
    porque un 30 % de humedad a 15 °C no seca igual que un 30 % a 35 °C.
    """
    t = np.asarray(tmax_c, dtype=np.float64)
    rh = np.clip(np.asarray(rhmin_pct, dtype=np.float64), 0.0, 100.0)
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return np.clip(es * (1.0 - rh / 100.0), 0.0, None)


def _dias_sin_lluvia(df: pd.DataFrame, umbral: float = UMBRAL_LLUVIA_MM) -> np.ndarray:
    """Días secos encadenados hasta la fecha, contados dentro de cada celda.

    Se reinicia a cero el día que llueve por encima del umbral. Vectorizado agrupando por
    "tramo seco": cada episodio de lluvia abre un grupo nuevo y el contador es la posición
    dentro del grupo.
    """
    seco = (df["prec_dia"].to_numpy() < umbral).astype(np.int8)
    tramo = (1 - seco).cumsum()
    return (
        df.assign(_seco=seco, _tramo=tramo)
        .groupby([CELL_COL, "_tramo"])
        .cumcount()
        .to_numpy()
        * seco
    )


def _nesterov(df: pd.DataFrame, reset_mm: float = UMBRAL_RESET_NESTEROV_MM) -> np.ndarray:
    """Índice de Nesterov: acumulación de (Tmax − Trocío) · Tmax en días sin lluvia.

    Es un índice clásico de peligrosidad que crece cuanto más se prolonga un episodio seco y
    caluroso, y se reinicia con la lluvia. Aporta memoria del episodio, no solo del día.
    """
    tmax = df["tmax_vc"].to_numpy(dtype=np.float64)
    rh = np.clip(df["rhmin_vc"].to_numpy(dtype=np.float64), 1.0, 100.0)
    a, b = 17.27, 237.3
    alpha = (a * tmax) / (b + tmax) + np.log(rh / 100.0)
    trocio = (b * alpha) / (a - alpha)
    aporte = np.clip(tmax - trocio, 0.0, None) * np.clip(tmax, 0.0, None)

    reset = (df["prec_dia"].to_numpy() >= reset_mm).astype(np.int8)
    tramo = reset.cumsum()
    serie = pd.Series(np.where(reset == 1, 0.0, aporte), index=df.index)
    return serie.groupby([df[CELL_COL].to_numpy(), tramo]).cumsum().to_numpy()


# ─── Ensamblado ───────────────────────────────────────────────────────────────


def add_derived_features(
    df: pd.DataFrame,
    climatology: Optional[pd.DataFrame] = None,
    daily_stats: Optional[pd.DataFrame] = None,
    vars_base: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Añade a un bloque preparado todas las variables derivadas.

    El bloque debe contener la serie temporal completa de cada celda y venir ordenado por
    (celda, fecha), que es como lo entrega `src.models.dataset.iter_blocks`: los índices
    secuenciales (días sin lluvia, Nesterov) dependen del orden.

    Args:
        df: Bloque preparado.
        climatology: Salida de `fit_cell_climatology`. Si es `None` se omiten las anomalías.
        daily_stats: Salida de `fit_daily_stats`. Si es `None` se omiten las relativas diarias.
        vars_base: Variables sobre las que derivar. Por defecto, `BASE_VARS`.

    Returns:
        El bloque con las columnas derivadas añadidas.
    """
    vars_base = vars_base or BASE_VARS
    df = df.sort_values([CELL_COL, DATE_COL]).reset_index(drop=True)

    # ── 1. Índices físicos de sequedad ──
    df["vpd"] = vapor_pressure_deficit(df["tmax_vc"], df["rhmin_vc"]).astype("float32")
    df["dias_sin_lluvia"] = _dias_sin_lluvia(df).astype("float32")
    df["nesterov"] = _nesterov(df).astype("float32")
    df["ratio_termico_eolico"] = (
        df["tmax_vc"].to_numpy() * df["vmax_vc"].to_numpy()
        / np.clip(df["rhmin_vc"].to_numpy(), 5.0, None)
    ).astype("float32")

    # ── 2. Anomalías frente a la climatología de la propia celda ──
    if climatology is not None:
        df = df.merge(climatology, left_on=[CELL_COL, "mes"], right_index=True, how="left")
        for v in vars_base:
            col = f"{v}_clim"
            if col in df.columns:
                df[f"{v}_anom"] = (df[v] - df[col]).astype("float32")
        df = df.drop(columns=[c for c in df.columns if c.endswith("_clim")])

    # ── 3. Posición relativa dentro del día ──
    if daily_stats is not None:
        df = df.merge(daily_stats, left_on=DATE_COL, right_index=True, how="left")
        for v in vars_base:
            media, std = f"{v}_media_dia", f"{v}_std_dia"
            if media in df.columns:
                df[f"{v}_z_dia"] = (
                    (df[v] - df[media]) / df[std].replace(0, np.nan)
                ).astype("float32")
        df = df.drop(
            columns=[c for c in df.columns if c.endswith(("_media_dia", "_std_dia"))]
        )

    # ── 4. Interacciones combustible × sequedad ──
    # La sequedad solo importa donde hay algo que arda: multiplicar por la carga de combustible
    # expresa esa condición de forma explícita, en lugar de esperar a que el árbol la infiera.
    forestal = df["combustible_pct_forestal"].to_numpy() / 100.0
    df["vpd_x_forestal"] = (df["vpd"].to_numpy() * forestal).astype("float32")
    df["seq_x_forestal"] = (df["dias_sin_lluvia"].to_numpy() * forestal).astype("float32")

    return df


def fit_context(
    iter_blocks_fn,
    train_years: list[int],
    all_years: list[int],
    mode: str,
    cache_dir: Optional["Path"] = None,
    vars_base: Optional[list[str]] = None,
) -> dict:
    """Ajusta (o recupera de caché) los estadísticos de referencia de las derivadas.

    Separa deliberadamente el origen de cada uno:

    - La **climatología por celda** es un parámetro aprendido y se ajusta **solo con los años de
      entrenamiento**. Usar validación o test aquí sería filtrar información.
    - Los **estadísticos espaciales diarios** se calculan para todos los años evaluados, porque
      son propios de cada fecha: «el 14 de agosto de 2022» no existe en el histórico de
      entrenamiento. No usan la variable objetivo y, sobre todo, **están disponibles en
      producción**: el día que se predice se dispone de la previsión de MeteoGalicia para toda
      Galicia, así que su media y desviación espaciales son calculables antes de predecir.

    Args:
        iter_blocks_fn: Función `(year) -> iterador de bloques` ya configurada.
        train_years: Años con los que aprender la climatología.
        all_years: Años para los que hacen falta estadísticos diarios.
        mode: Variante del dataset, solo para nombrar la caché.
        cache_dir: Directorio donde guardar y buscar los ficheros precalculados.
        vars_base: Variables sobre las que derivar.

    Returns:
        Diccionario con las claves `climatology` y `daily_stats`.
    """
    from pathlib import Path

    vars_base = vars_base or BASE_VARS
    etiqueta = f"{mode}_tr{min(train_years)}-{max(train_years)}_a{min(all_years)}-{max(all_years)}"

    ruta_clim = ruta_stats = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ruta_clim = cache_dir / f"climatologia_{etiqueta}.parquet"
        ruta_stats = cache_dir / f"stats_diarios_{etiqueta}.parquet"
        if ruta_clim.exists() and ruta_stats.exists():
            logger.info("Contexto de derivadas recuperado de caché: %s", cache_dir)
            return {
                "climatology": pd.read_parquet(ruta_clim),
                "daily_stats": pd.read_parquet(ruta_stats),
            }

    def bloques(años):
        for a in años:
            yield from iter_blocks_fn(a)

    logger.info("Ajustando climatología por celda y mes con %s...", train_years)
    clim = fit_cell_climatology(bloques(train_years), vars_base)

    logger.info("Ajustando estadísticos espaciales diarios con %s...", all_years)
    stats = fit_daily_stats(bloques(all_years), vars_base)

    if ruta_clim is not None:
        clim.to_parquet(ruta_clim)
        stats.to_parquet(ruta_stats)
        logger.info("Contexto de derivadas guardado en caché.")

    return {"climatology": clim, "daily_stats": stats}


def derived_columns(vars_base: Optional[list[str]] = None) -> list[str]:
    """Lista los nombres de todas las columnas que genera `add_derived_features`."""
    vars_base = vars_base or BASE_VARS
    cols = ["vpd", "dias_sin_lluvia", "nesterov", "ratio_termico_eolico",
            "vpd_x_forestal", "seq_x_forestal"]
    cols += [f"{v}_anom" for v in vars_base]
    cols += [f"{v}_z_dia" for v in vars_base]
    return cols
