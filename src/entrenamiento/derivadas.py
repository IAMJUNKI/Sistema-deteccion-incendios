"""Variables derivadas que el datacubo no trae.

El cubo EGIF ya incorpora buena parte de lo que en el pipeline anterior había que calcular a
mano: acumulados de precipitación a 3/7/14/30 días, medias móviles a 7 días y
`consecutive_dry_days`. Todo eso se usa tal cual y **no se recalcula**.

Lo que sí falta, y es lo que aporta este módulo, son variables que no transforman la
meteorología sino que la **contextualizan**:

- **Anomalías** (`*_anom`): la desviación respecto de la climatología de *esa celda en ese mes*.
  Treinta grados no significan lo mismo en A Coruña en marzo que en Ourense en agosto. Un
  modelo con variables absolutas aprende el mapa climático de Galicia; con anomalías aprende
  cuándo un día se sale de lo normal, que es lo que precede a un incendio.
- **Z-scores espaciales diarios** (`*_z_dia`): la posición de la celda dentro de la distribución
  de *toda Galicia ese día*. Responde a «¿es esta la celda más seca hoy?», que es exactamente
  la discriminación intradía que las métricas operativas exigen y que las variables absolutas
  no dan, porque el día entero se mueve en bloque.

Además se generan un par de interacciones baratas y, **solo si el dataset no las trae ya**, el
déficit de presión de vapor. Esa condición es intencionada: cuando Alfonso publique el Parquet
con `vpd_mean` y `vpd_max_12_18h`, este módulo dejará de calcular su propia versión y usará la
del cubo, que además incorpora la ventana crítica de tarde.

## Una simplificación deliberada

El pipeline anterior calculaba también el índice de Nesterov, que es una acumulación secuencial
por celda y obliga a cargar la serie temporal completa de cada celda en memoria. Aquí se ha
dejado fuera a propósito: obligaría a recorrer el dataset por bloques de celdas en vez de por
lotes de filas, multiplicando las pasadas de lectura, y su papel —memoria de sequía— ya lo
cubren `consecutive_dry_days` y los cuatro acumulados de precipitación que trae el cubo.

El resultado es que todas las derivadas de aquí son **sin estado**: se calculan fila a fila o
mediante un `merge` con un contexto precalculado. Eso permite evaluar el año completo por lotes
acotados sin cargar nunca 10 millones de filas de golpe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from src.entrenamiento.contrato import (
    BASE_DERIVADAS, COL_CELDA, COL_FECHA, RAIZ, Contrato,
)

logger = logging.getLogger(__name__)

DIR_CACHE = RAIZ / "data" / "processed" / "contexto_derivadas"

#: Fracciones CORINE que suman la cobertura forestal. `forest_cover_fraction` es su suma exacta
#: y está marcada para excluirse del perfil final, así que se reconstruye desde las tres.
FRACCIONES_BOSQUE = ("broadleaf_forest", "coniferous_forest", "mixed_forest")

#: Mínimo de observaciones para que una celda-mes tenga climatología fiable.
MIN_OBS_CLIMATOLOGIA = 20

#: Calendario mínimo, que se genera solo si el cubo no lo trae.
#:
#: El perfil final del datacubo retira las nueve variables de calendario, con el argumento de
#: que «se derivan de `time` cuando se necesita». Es correcto, pero entonces alguien tiene que
#: derivarlas, y ese alguien es este módulo. Se generan tres y no nueve porque ya está medido
#: que las demás no aportan: `month_sin` y `month_cos` salieron con importancia por permutación
#: de -0,00009 y -0,00004, y `month`, `iso_week` y `day_of_year` son la misma variable escrita
#: de tres formas (Spearman 0,97 entre ellas).
#:
#: - `dia_anio_sin` / `dia_anio_cos`: estacionalidad continua y sin discontinuidad de fin de
#:   año, que es lo que un `month` entero no da.
#: - `es_finde`: el 96 % de los incendios en Galicia son de causa humana, y la actividad humana
#:   en el monte tiene ciclo semanal. Es la única variable de calendario con una hipótesis
#:   causal detrás y no meramente estacional.
CALENDARIO = ("dia_anio_sin", "dia_anio_cos", "es_finde")


@dataclass(frozen=True)
class Contexto:
    """Estadísticos precalculados que convierten valores absolutos en relativos."""

    climatologia: pd.DataFrame   # cell_id, mes, {var}_media, {var}_desv
    diarios: pd.DataFrame        # fecha, {var}_media_dia, {var}_desv_dia
    variables: list[str]
    anios_climatologia: list[int]

    def resumen(self) -> str:
        return (
            f"Contexto: climatología de {len(self.climatologia):,} celda-mes "
            f"(años {self.anios_climatologia}), {len(self.diarios):,} días, "
            f"sobre {len(self.variables)} variables base"
        )


def deficit_presion_vapor(temperatura_c: pd.Series, humedad_relativa: pd.Series) -> pd.Series:
    """Déficit de presión de vapor en kPa (Tetens).

    Mide cuánta agua puede arrancarle el aire a la vegetación, que es lo que de verdad seca el
    combustible. No es la temperatura ni la humedad por separado: es la física que las une.
    """
    saturacion = 0.6108 * np.exp((17.27 * temperatura_c) / (temperatura_c + 237.3))
    return (saturacion * (1.0 - np.clip(humedad_relativa, 0.0, 100.0) / 100.0)).astype(np.float32)


def variables_base(contrato: Contrato) -> list[str]:
    """Variables meteorológicas base presentes en el dataset, para anomalías y z-scores."""
    return [v for v in BASE_DERIVADAS if contrato.tiene(v)]


def _ruta_cache(nombre: str, anios: Sequence[int]) -> Path:
    return DIR_CACHE / f"{nombre}_{min(anios)}-{max(anios)}.parquet"


def ajustar_contexto(
    contrato: Contrato,
    anios_climatologia: Sequence[int],
    anios_diarios: Sequence[int],
    usar_cache: bool = True,
) -> Contexto:
    """Calcula climatología por celda-mes y estadísticos espaciales por día.

    La climatología se ajusta **solo con los años de entrenamiento**: usar el año de validación
    para definir «lo normal» sería dejar que el modelo conociera el clima del periodo sobre el
    que se le evalúa.

    Los estadísticos diarios, en cambio, se calculan sobre todos los años que se vayan a
    procesar. No es fuga: describen el estado de Galicia en un día concreto y en producción se
    obtienen del propio mapa de previsión del día, sin conocer ningún incendio.

    Args:
        contrato: Contrato del dataset.
        anios_climatologia: Años de entrenamiento.
        anios_diarios: Todos los años que se van a puntuar.
        usar_cache: Reutiliza los Parquet cacheados si existen. Ambas pasadas son caras.
    """
    variables = variables_base(contrato)
    if not variables:
        raise ValueError(
            f"Ninguna de las variables base {BASE_DERIVADAS} está en el dataset. "
            "Revisa contrato.BASE_DERIVADAS."
        )

    ruta_clima = _ruta_cache("climatologia", anios_climatologia)
    ruta_dias = _ruta_cache("diarios", anios_diarios)

    if usar_cache and ruta_clima.exists() and ruta_dias.exists():
        logger.info("Contexto reutilizado desde caché: %s", DIR_CACHE)
        return Contexto(
            climatologia=pd.read_parquet(ruta_clima),
            diarios=pd.read_parquet(ruta_dias),
            variables=variables,
            anios_climatologia=list(anios_climatologia),
        )

    logger.info("Ajustando climatología celda-mes con %s...", list(anios_climatologia))
    climatologia = _agregar_climatologia(contrato, anios_climatologia, variables)

    logger.info("Ajustando estadísticos espaciales diarios con %s...", list(anios_diarios))
    diarios = _agregar_diarios(contrato, anios_diarios, variables)

    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    climatologia.to_parquet(ruta_clima, index=False)
    diarios.to_parquet(ruta_dias, index=False)

    contexto = Contexto(climatologia, diarios, variables, list(anios_climatologia))
    logger.info(contexto.resumen())
    return contexto


def _leer_base(contrato: Contrato, anios: Sequence[int], variables: Sequence[str]):
    """Recorre solo las columnas necesarias para los estadísticos, en lotes."""
    columnas = [COL_CELDA, COL_FECHA, *variables]
    dataset = pads.dataset([str(p) for p in contrato.rutas(anios)], format="parquet")
    for lote in dataset.scanner(columns=columnas, batch_size=1_000_000).to_batches():
        yield lote.to_pandas()


def _acumular(marcos, clave: list[str], variables: Sequence[str]) -> pd.DataFrame:
    """Suma, suma de cuadrados y conteo por clave, acumulando lote a lote.

    Se acumulan momentos en vez de concatenar los datos porque el conjunto completo no cabe
    cómodamente en memoria: 32 millones de filas por cuatro variables.
    """
    acumulado: Optional[pd.DataFrame] = None
    for marco in marcos:
        parcial = marco[clave + list(variables)].copy()
        for variable in variables:
            parcial[f"{variable}__sq"] = parcial[variable].astype(np.float64) ** 2
        agregado = parcial.groupby(clave, sort=False).agg(
            **{f"{v}__suma": (v, "sum") for v in variables},
            **{f"{v}__sumasq": (f"{v}__sq", "sum") for v in variables},
            __n=(variables[0], "size"),
        ).reset_index()
        acumulado = agregado if acumulado is None else (
            pd.concat([acumulado, agregado], ignore_index=True)
            .groupby(clave, sort=False).sum().reset_index()
        )
    if acumulado is None:
        raise ValueError("No se leyó ninguna fila para calcular estadísticos.")
    return acumulado


def _media_y_desviacion(acumulado: pd.DataFrame, variables: Sequence[str],
                        sufijo: str) -> pd.DataFrame:
    """Convierte momentos acumulados en media y desviación típica."""
    n = acumulado["__n"].to_numpy(dtype=np.float64)
    salida = acumulado.drop(columns=[c for c in acumulado.columns if c.startswith("__")
                                     or c.endswith(("__suma", "__sumasq"))]).copy()
    salida["n"] = n.astype(np.int32)
    for variable in variables:
        media = acumulado[f"{variable}__suma"].to_numpy(dtype=np.float64) / n
        varianza = acumulado[f"{variable}__sumasq"].to_numpy(dtype=np.float64) / n - media ** 2
        salida[f"{variable}_media{sufijo}"] = media.astype(np.float32)
        salida[f"{variable}_desv{sufijo}"] = np.sqrt(np.maximum(varianza, 0.0)).astype(np.float32)
    return salida


def _agregar_climatologia(contrato: Contrato, anios: Sequence[int],
                          variables: Sequence[str]) -> pd.DataFrame:
    def con_mes():
        for marco in _leer_base(contrato, anios, variables):
            marco["mes"] = pd.to_datetime(marco[COL_FECHA]).dt.month.astype(np.int8)
            yield marco

    acumulado = _acumular(con_mes(), [COL_CELDA, "mes"], variables)
    climatologia = _media_y_desviacion(acumulado, variables, sufijo="")
    escasas = int((climatologia["n"] < MIN_OBS_CLIMATOLOGIA).sum())
    if escasas:
        logger.warning("%s celda-mes con menos de %s observaciones.", f"{escasas:,}",
                       MIN_OBS_CLIMATOLOGIA)
    return climatologia


def _agregar_diarios(contrato: Contrato, anios: Sequence[int],
                     variables: Sequence[str]) -> pd.DataFrame:
    acumulado = _acumular(_leer_base(contrato, anios, variables), [COL_FECHA], variables)
    return _media_y_desviacion(acumulado, variables, sufijo="_dia")


def columna_vpd(contrato: Contrato) -> Optional[str]:
    """Nombre de la columna de VPD que se debe usar, o `None` si no hay forma de tenerla.

    Si el cubo trae `vpd_mean`, se usa esa y no se calcula nada: la del cubo se deriva de la
    temperatura y la humedad horarias, mientras que la nuestra parte de los agregados diarios.
    """
    if contrato.tiene("vpd_mean"):
        return "vpd_mean"
    if contrato.tiene("temperature_max", "relative_humidity_min"):
        return "vpd"
    return None


def columna_forestal(contrato: Contrato) -> Optional[str]:
    """Nombre de la columna de cobertura forestal que se debe usar.

    `forest_cover_fraction` es, por definición del cubo, la suma exacta de las tres fracciones
    de bosque. Calcular además una `forestal` propia crearía dos columnas idénticas: el árbol
    repartiría los cortes entre ambas y la importancia de la cobertura forestal aparecería
    dividida por dos, que es exactamente el modo en que la redundancia vuelve ilegible una
    tabla de importancias.

    Por eso se prefiere la del cubo cuando está, y solo se reconstruye desde las tres
    fracciones cuando no está —que es lo que ocurrirá cuando se aplique el perfil final de
    variables, donde `forest_cover_fraction` figura como excluida—.
    """
    if contrato.tiene("forest_cover_fraction"):
        return "forest_cover_fraction"
    if all(contrato.tiene(f) for f in FRACCIONES_BOSQUE):
        return "forestal"
    return None


def calendario_ausente(contrato: Contrato) -> list[str]:
    """Variables de calendario que hay que generar porque el cubo ya no las trae.

    Se comprueba contra los nombres del cubo, no contra los nuestros: si `day_of_year_sin`
    vuelve a estar en el dataset, no se duplica con un `dia_anio_sin` propio.
    """
    equivalentes = {
        "dia_anio_sin": ("day_of_year_sin",),
        "dia_anio_cos": ("day_of_year_cos",),
        "es_finde": ("is_weekend",),
    }
    return [
        nuestra for nuestra in CALENDARIO
        if not any(contrato.tiene(suya) for suya in equivalentes[nuestra])
    ]


def anadir_calendario(marco: pd.DataFrame, cuales: Sequence[str]) -> pd.DataFrame:
    """Deriva el calendario mínimo a partir de la fecha."""
    if not cuales:
        return marco
    fechas = pd.to_datetime(marco[COL_FECHA])
    if "dia_anio_sin" in cuales or "dia_anio_cos" in cuales:
        # 365,25 y no 365: sin el cuarto de día, en un bisiesto el ciclo se desfasa y el
        # 31 de diciembre no cae junto al 1 de enero.
        angulo = 2 * np.pi * fechas.dt.dayofyear.to_numpy() / 365.25
        if "dia_anio_sin" in cuales:
            marco["dia_anio_sin"] = np.sin(angulo).astype(np.float32)
        if "dia_anio_cos" in cuales:
            marco["dia_anio_cos"] = np.cos(angulo).astype(np.float32)
    if "es_finde" in cuales:
        marco["es_finde"] = (fechas.dt.dayofweek >= 5).to_numpy().astype(np.int8)
    return marco


def columnas_derivadas(contrato: Contrato) -> list[str]:
    """Nombres de las columnas que `anadir_derivadas` va a generar, sin calcular nada.

    Solo enumera lo que este módulo **crea**: si una variable ya viene en el cubo, no aparece
    aquí porque no hay nada que generar.
    """
    variables = variables_base(contrato)
    vpd = columna_vpd(contrato)
    forestal = columna_forestal(contrato)
    nombres: list[str] = list(calendario_ausente(contrato))

    if vpd == "vpd":
        nombres.append("vpd")
    if contrato.tiene("temperature_max", "wind_speed_max"):
        nombres.append("ratio_termico_eolico")
    if forestal == "forestal":
        nombres.append("forestal")
    if forestal is not None:
        if contrato.tiene("consecutive_dry_days"):
            nombres.append("sequia_x_forestal")
        if vpd is not None:
            nombres.append("vpd_x_forestal")

    nombres += [f"{v}_anom" for v in variables]
    nombres += [f"{v}_z_dia" for v in variables]
    return nombres


def anadir_derivadas(marco: pd.DataFrame, contexto: Contexto,
                     contrato: Contrato) -> pd.DataFrame:
    """Añade las variables derivadas a un lote ya cargado.

    Es una operación sin estado: el mismo lote produce siempre el mismo resultado, y no depende
    de qué otros lotes se hayan procesado antes. Eso es lo que permite evaluar por lotes.
    """
    marco = marco.copy()

    # ── Calendario mínimo, solo si el cubo ya no lo trae ──────────────────────────────────
    marco = anadir_calendario(marco, calendario_ausente(contrato))

    # ── Déficit de presión de vapor, solo si el cubo no lo trae ya ────────────────────────
    vpd_col = columna_vpd(contrato)
    if vpd_col == "vpd":
        marco["vpd"] = deficit_presion_vapor(
            marco["temperature_max"], marco["relative_humidity_min"]
        )

    # ── Interacción térmico-eólica: calor y viento juntos, no por separado ────────────────
    if contrato.tiene("temperature_max", "wind_speed_max"):
        marco["ratio_termico_eolico"] = (
            marco["temperature_max"] * np.log1p(marco["wind_speed_max"].clip(lower=0))
        ).astype(np.float32)

    # ── Interacciones con el combustible disponible ───────────────────────────────────────
    # Se reutiliza la cobertura forestal del cubo si existe, en vez de duplicarla.
    forestal_col = columna_forestal(contrato)
    if forestal_col == "forestal":
        marco["forestal"] = sum(marco[f] for f in FRACCIONES_BOSQUE).astype(np.float32)
    if forestal_col is not None:
        if contrato.tiene("consecutive_dry_days"):
            marco["sequia_x_forestal"] = (
                marco["consecutive_dry_days"] * marco[forestal_col]
            ).astype(np.float32)
        if vpd_col is not None:
            marco["vpd_x_forestal"] = (marco[vpd_col] * marco[forestal_col]).astype(np.float32)

    # ── Anomalías respecto de la climatología de la celda en ese mes ──────────────────────
    marco["__mes"] = pd.to_datetime(marco[COL_FECHA]).dt.month.astype(np.int8)
    marco = marco.merge(
        contexto.climatologia.rename(columns={"mes": "__mes"}).drop(columns=["n"]),
        on=[COL_CELDA, "__mes"], how="left",
    )
    for variable in contexto.variables:
        marco[f"{variable}_anom"] = (
            marco[variable] - marco[f"{variable}_media"]
        ).astype(np.float32)

    # ── Posición de la celda dentro de Galicia ese día ────────────────────────────────────
    marco = marco.merge(contexto.diarios.drop(columns=["n"]), on=COL_FECHA, how="left")
    for variable in contexto.variables:
        desviacion = marco[f"{variable}_desv_dia"].to_numpy(dtype=np.float64)
        # Un día uniforme en toda Galicia tiene desviación cero: el z-score no está definido y
        # se fija a cero, que es la lectura correcta —ninguna celda destaca sobre las demás—.
        marco[f"{variable}_z_dia"] = np.where(
            desviacion > 1e-9,
            (marco[variable].to_numpy(dtype=np.float64) - marco[f"{variable}_media_dia"]) / np.where(desviacion > 1e-9, desviacion, 1.0),
            0.0,
        ).astype(np.float32)

    auxiliares = ["__mes"] + [c for c in marco.columns
                              if c.endswith(("_media", "_desv", "_media_dia", "_desv_dia"))]
    return marco.drop(columns=[c for c in auxiliares if c in marco.columns])
