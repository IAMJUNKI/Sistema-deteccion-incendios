"""
CONSTRUCCION DEL DATASET 2D ENRIQUECIDO  (Enrique)
==================================================

Que hace
--------
Construye el dataset tabular (una fila = celda x dia) para modelar el riesgo de
ignicion, ampliando el dataset 2D actual del equipo con las variables que se han
detectado ausentes al compararlo con el pipeline 3D de Alfonso.

Novedades frente al dataset actual (Miquel/Diego):
  * dias_sin_lluvia        -> dias consecutivos sin precipitacion (>1 mm). Indicador
                              clasico de sequedad del combustible fino. NO estaba.
  * rhmin_media_7d         -> memoria de humedad (antes solo habia memoria de temperatura)
  * vmax_media_7d          -> memoria de viento
  * prec_acum_3d / 14d     -> ventanas cortas y medias (antes solo 7d y 30d)
  * vpd                    -> Vapor Pressure Deficit: demanda evaporativa real de la
                              atmosfera sobre la vegetacion (combina T y HR)
  * combustible one-hot    -> cada clase de combustible como variable propia
                              (antes todo se resumia en % forestal, perdiendo el matiz
                              de que matorral y coniferas son las que mas arden)
  * orientacion one-hot    -> sur/norte/este/oeste como variables propias
  * dia_anio_sin / cos     -> estacionalidad como circulo continuo (31-dic y 1-ene juntos)

Se mantienen las convenciones del equipo:
  * Ventana critica 12-18h local en las variables meteorologicas (tmax_vc, rhmin_vc, vmax_vc)
  * Shift T-1: para predecir el dia T solo se usan datos hasta T-1 (anti data leakage)
  * Rejilla de 1 km (30.697 celdas) de la Fase 1

Target
------
El target se enchufa al final y es INTERCAMBIABLE (constante TARGET_SOURCE):
  * "firms" -> positivos de los parquets de Miquel (lo que realmente usa hoy el benchmark)
  * "egif"  -> registros oficiales EGIF (pendiente: el XML disponible solo cubre 2014-2017)
Asi, decida lo que decida el equipo, el dataset no hay que rehacerlo.

Entradas
--------
  Datos/meteo_celdas/meteo_celdas_1km_YYYY.parquet          (meteo propia, metodo C validado)
  Datos/grids/galicia_grid_1km_2018.parquet                 (rejilla Fase 1, Diego)
  Sistema-deteccion-incendios/misc/Dataset/Miquel/dataset_maestro_YYYY.parquet  (target FIRMS)

Salida
------
  Datos/dataset_2d_enriquecido/dataset_2d_YYYY.parquet      (un archivo por anio)

Uso
---
  conda activate incendios-forestales
  python construye_dataset_2d_enriquecido.py           # todos los anios
  python construye_dataset_2d_enriquecido.py 2022      # un anio suelto
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuracion
TARGET_SOURCE = "firms"          # "firms" | "egif"
ANIOS = [2019, 2020, 2021, 2022, 2023]
UMBRAL_LLUVIA_MM = 1.0           # a partir de aqui se considera que "ha llovido"

BASE = Path(__file__).resolve().parent
DIR_METEO = BASE / "Datos" / "meteo_celdas"
DIR_GRID = BASE / "Datos" / "grids" / "galicia_grid_1km_2018.parquet"
DIR_TARGET = BASE / "Sistema-deteccion-incendios" / "misc" / "Dataset" / "Miquel"
DIR_SALIDA = BASE / "Datos" / "dataset_2d_enriquecido"


# ---------------------------------------------------------------- utilidades
def calcula_vpd(tmax_c: np.ndarray, rhmin_pct: np.ndarray) -> np.ndarray:
    """
    Vapor Pressure Deficit (kPa): cuanta "sed" tiene la atmosfera.

    Es la diferencia entre el vapor de agua que el aire podria contener a esa
    temperatura y el que realmente contiene. A mayor VPD, mas rapido se seca la
    vegetacion. Es mejor predictor que la humedad relativa sola, porque un 30%
    de humedad a 15 grados no seca igual que un 30% a 35 grados.
    """
    es = 0.6108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))   # presion de saturacion
    return es * (1.0 - np.clip(rhmin_pct, 0, 100) / 100.0)


def _a_matriz(df: pd.DataFrame, columna: str, n_celdas: int, n_dias: int) -> np.ndarray:
    """
    Reorganiza una columna larga (celda x dia apilados) en una matriz (celdas x dias).

    Trabajar en matriz permite calcular todas las ventanas moviles de todas las celdas
    a la vez con operaciones vectorizadas, en vez de recorrer 30.697 grupos uno a uno
    (que tardaba minutos por variable).
    """
    return df[columna].to_numpy(dtype="float64").reshape(n_celdas, n_dias)


def _ventana_previa(mat: np.ndarray, k: int, modo: str = "suma") -> np.ndarray:
    """
    Ventana movil de los k dias ANTERIORES (excluyendo el dia actual).

    Excluir el dia actual es la regla anti data-leakage del proyecto: para predecir
    el dia T solo se puede mirar hasta T-1. Se calcula con sumas acumuladas, que
    resuelven la ventana en una sola pasada.
    """
    n, d = mat.shape
    prev = np.empty_like(mat)
    prev[:, 0] = np.nan
    prev[:, 1:] = mat[:, :-1]

    valido = ~np.isnan(prev)
    x = np.where(valido, prev, 0.0)
    acum = np.cumsum(x, axis=1)
    cuenta = np.cumsum(valido, axis=1).astype("float64")

    idx = np.arange(d)
    lo = idx - k                                   # limite izquierdo de la ventana
    tomar = lambda c: c - np.where(lo >= 0, c[:, np.clip(lo, 0, d - 1)], 0.0)

    suma = tomar(acum)
    n_validos = tomar(cuenta)
    if modo == "suma":
        return np.where(n_validos > 0, suma, np.nan)
    return np.where(n_validos > 0, suma / np.maximum(n_validos, 1), np.nan)


def _dias_sin_lluvia(mat_precipitacion: np.ndarray) -> np.ndarray:
    """
    Dias consecutivos sin lluvia significativa, contados hasta el dia anterior.
    Se reinicia a 0 cuando llueve y suma 1 por cada dia seco encadenado.
    """
    n, d = mat_precipitacion.shape
    prev = np.empty_like(mat_precipitacion)
    prev[:, 0] = np.nan
    prev[:, 1:] = mat_precipitacion[:, :-1]

    seco = (np.nan_to_num(prev, nan=0.0) < UMBRAL_LLUVIA_MM)
    salida = np.zeros((n, d), dtype="float32")
    contador = np.zeros(n, dtype="float32")
    for j in range(d):
        contador = np.where(seco[:, j], contador + 1, 0.0)
        salida[:, j] = contador
    return salida


def carga_meteo_con_margen(anio: int, celda_min: int, celda_max: int) -> pd.DataFrame:
    """
    Carga la meteo de un BLOQUE de celdas para el anio pedido, mas los ultimos
    35 dias del anio anterior.

    Dos motivos para trabajar por bloques:
      * el margen del anio anterior es necesario porque las ventanas moviles de
        30 dias necesitan historia previa (si no, los eneros saldrian incompletos);
      * un anio entero son 11,2 millones de filas y no cabe en memoria de una vez,
        asi que se procesa por grupos de celdas y se va escribiendo a disco.
    """
    filtro_celdas = [("cell_id", ">=", celda_min), ("cell_id", "<", celda_max)]
    piezas = [pd.read_parquet(DIR_METEO / f"meteo_celdas_1km_{anio}.parquet",
                              filters=filtro_celdas)]

    previo = DIR_METEO / f"meteo_celdas_1km_{anio - 1}.parquet"
    if previo.exists():
        df_prev = pd.read_parquet(previo, filters=filtro_celdas)
        corte = pd.Timestamp(f"{anio - 1}-11-26")
        piezas.insert(0, df_prev[pd.to_datetime(df_prev.fecha) >= corte])

    df = pd.concat(piezas, ignore_index=True)
    df["fecha"] = pd.to_datetime(df.fecha)
    return df.sort_values(["cell_id", "fecha"]).reset_index(drop=True)


def carga_target(anio: int) -> pd.DataFrame:
    """Devuelve las combinaciones (cell_id, fecha) con ignicion segun la fuente elegida."""
    if TARGET_SOURCE == "firms":
        ruta = DIR_TARGET / f"dataset_maestro_{anio}.parquet"
        tgt = pd.read_parquet(ruta, columns=["cell_id", "fecha", "target"])
        tgt = tgt[tgt.target == 1].copy()
        tgt["fecha"] = pd.to_datetime(tgt.fecha)
        return tgt[["cell_id", "fecha", "target"]]

    raise NotImplementedError(
        "El target EGIF esta pendiente: el XML disponible solo cubre 2014-2017, "
        "sin solape con el periodo de entrenamiento 2019-2023."
    )


# ---------------------------------------------------------------- construccion
def construye_bloque(anio: int, celda_min: int, celda_max: int,
                     grid: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """Construye las filas del anio para un bloque de celdas."""

    # 1. Meteo propia (ya interpolada a 1 km con el metodo validado) + margen
    df = carga_meteo_con_margen(anio, celda_min, celda_max)

    # 2. Comprobacion: todas las celdas deben tener los mismos dias (rejilla completa).
    celdas = np.sort(df.cell_id.unique())
    dias = np.sort(df.fecha.unique())
    n_celdas, n_dias = len(celdas), len(dias)
    if len(df) != n_celdas * n_dias:
        raise ValueError(f"Faltan filas: {len(df):,} != {n_celdas:,} celdas x {n_dias} dias")

    # 3. Memoria climatica y shift T-1, en forma matricial (celdas x dias)
    m_prec = _a_matriz(df, "prec_dia", n_celdas, n_dias)
    m_tmax = _a_matriz(df, "tmax_vc", n_celdas, n_dias)
    m_rhmin = _a_matriz(df, "rhmin_vc", n_celdas, n_dias)
    m_vmax = _a_matriz(df, "vmax_vc", n_celdas, n_dias)

    nuevas = {
        "prec_acum_3d": _ventana_previa(m_prec, 3, "suma"),
        "prec_acum_7d": _ventana_previa(m_prec, 7, "suma"),
        "prec_acum_14d": _ventana_previa(m_prec, 14, "suma"),
        "prec_acum_30d": _ventana_previa(m_prec, 30, "suma"),
        "tmax_media_7d": _ventana_previa(m_tmax, 7, "media"),
        "rhmin_media_7d": _ventana_previa(m_rhmin, 7, "media"),
        "vmax_media_7d": _ventana_previa(m_vmax, 7, "media"),
        "dias_sin_lluvia": _dias_sin_lluvia(m_prec),
    }
    # Shift T-1 de las variables del propio dia: en produccion, al predecir el dia T
    # todavia no se conoce el tiempo de hoy, solo el de ayer.
    for nombre, mat in [("tmax_vc", m_tmax), ("rhmin_vc", m_rhmin),
                        ("vmax_vc", m_vmax), ("prec_dia", m_prec)]:
        desplazada = np.empty_like(mat)
        desplazada[:, 0] = np.nan
        desplazada[:, 1:] = mat[:, :-1]
        nuevas[f"{nombre}_t1"] = desplazada

    for nombre, mat in nuevas.items():
        df[nombre] = mat.reshape(-1).astype("float32")
    print("  memoria climatica calculada (3/7/14/30d, medias 7d, dias sin lluvia) + shift T-1")

    # 4. VPD sobre las variables ya desplazadas (T-1)
    df["vpd_t1"] = calcula_vpd(df.tmax_vc_t1.values, df.rhmin_vc_t1.values)

    # 5. Recortar el margen: nos quedamos solo con el anio pedido
    df = df[df.fecha.dt.year == anio].copy()

    # 6. Terreno (constante por celda) + one-hot de combustible y orientacion
    df = df.merge(grid, on="cell_id", how="left")

    for clase in ["matorral", "bosque_coniferas", "bosque_frondosas", "bosque_mixto",
                  "pastizal", "agricola", "urbano", "agua_humedal"]:
        df[f"comb_{clase}"] = (df.combustible_clase == clase).astype("int8")
    for orient in ["norte", "sur", "este", "oeste"]:
        df[f"orient_{orient}"] = (df.orientacion_clase == orient).astype("int8")

    # 7. Calendario. La estacionalidad se codifica como seno/coseno para que el
    #    31 de diciembre y el 1 de enero queden juntos (son dias parecidos).
    fechas = df.fecha
    df["mes"] = fechas.dt.month.astype("int8")
    df["dia_semana"] = fechas.dt.dayofweek.astype("int8")
    df["es_finde"] = df.dia_semana.isin([5, 6]).astype("int8")
    df["dia_anio_sin"] = np.sin(2 * np.pi * fechas.dt.dayofyear / 365.25).astype("float32")
    df["dia_anio_cos"] = np.cos(2 * np.pi * fechas.dt.dayofyear / 365.25).astype("float32")

    # 8. Target (intercambiable)
    df = df.merge(tgt, on=["cell_id", "fecha"], how="left")
    df["target"] = df.target.fillna(0).astype("int8")

    # 9. Aligerar tipos
    df = df.drop(columns=["orientacion_clase", "combustible_clase"])
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def construye_anio(anio: int, celdas_por_bloque: int = 2000) -> None:
    """
    Construye el anio completo procesando la rejilla por bloques de celdas.

    Cada bloque se guarda en su propio archivo dentro de Datos/dataset_2d_enriquecido/{anio}/.
    Trabajar asi tiene dos ventajas: la memoria no se dispara (un anio entero son
    11,2 millones de filas) y el proceso es REANUDABLE: si se corta, al relanzarlo
    salta los bloques que ya existen.
    """
    print(f"\n=== {anio} ===", flush=True)

    # Datos que se reutilizan en todos los bloques: terreno y target del anio
    grid = pd.DataFrame(gpd.read_parquet(DIR_GRID))[
        ["cell_id", "altitud_media", "pendiente_media", "orientacion_media",
         "orientacion_clase", "combustible_clase", "combustible_pct_forestal"]]
    tgt = carga_target(anio)

    dir_anio = DIR_SALIDA / str(anio)
    dir_anio.mkdir(parents=True, exist_ok=True)

    max_celda = int(grid.cell_id.max()) + 1
    for lo in range(0, max_celda, celdas_por_bloque):
        hi = min(lo + celdas_por_bloque, max_celda)
        destino = dir_anio / f"bloque_{lo:05d}.parquet"
        if destino.exists():
            print(f"  [SKIP] celdas {lo}-{hi} ya construidas", flush=True)
            continue
        bloque = construye_bloque(anio, lo, hi, grid, tgt)
        if bloque.empty:
            continue
        bloque.to_parquet(destino, index=False)
        print(f"  celdas {lo:>6}-{hi:<6} -> {len(bloque):>8,} filas | "
              f"positivos {int(bloque.target.sum()):>3}", flush=True)
        del bloque

    print(f"  {anio} completo en {dir_anio}", flush=True)


def carga_anio(anio: int) -> pd.DataFrame:
    """Lee de vuelta un anio completo (todos sus bloques) como un unico DataFrame."""
    bloques = sorted((DIR_SALIDA / str(anio)).glob("bloque_*.parquet"))
    return pd.concat([pd.read_parquet(b) for b in bloques], ignore_index=True)


if __name__ == "__main__":
    anios = [int(a) for a in sys.argv[1:]] or ANIOS
    print(f"Construyendo dataset 2D enriquecido | target = {TARGET_SOURCE.upper()}")
    for anio in anios:
        construye_anio(anio)
    print("\nListo.")
