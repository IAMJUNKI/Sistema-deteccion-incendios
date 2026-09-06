"""Contexto espacial: lo que cada celda no sabe de sus vecinas.

Al aplanar el datacubo en una tabla, cada fila queda aislada: el modelo no sabe que la celda
contigua existe, ni que ayer ardió algo a cinco kilómetros. Es la debilidad estructural medida
del sistema —la discriminación dentro del día se queda en 0,74 de AUC frente a 0,87 global— y
la razón es que acertar el día es un problema meteorológico y acertar la celda es un problema
espacial.

Este módulo devuelve esa información a la fila sin abandonar la representación tabular.

## Qué NO se construye, y por qué

La tentación es promediar las variables meteorológicas de las ocho celdas contiguas. Medido
sobre un día de agosto de 2022, la correlación entre el valor de una celda y la media de sus
vecinas es:

    temperature_max          r = 0,9996
    relative_humidity_min    r = 0,9998
    elevation_mean           r = 0,9894

Son la misma variable. Tiene explicación: la meteorología procede de ERA5-Land, cuya resolución
nativa ronda los 9 km, interpolada después a 1 km. Ocho celdas contiguas comparten el mismo
píxel de origen. Promediarlas produce una copia y reparte la importancia entre dos columnas
idénticas.

Lo que sí varía a escala de kilómetro es el **combustible** (r = 0,72 para frondosas, 0,79 para
matorral) y, sobre todo, **dónde ha ardido recientemente**.

## Lo que sí se construye

**Historial de igniciones en el entorno.** Es la señal ausente. En 2022, el 71,7 % de las
igniciones tuvo otra a menos de 12,5 km durante los diez días previos, frente al 30,2 % de los
negativos tomados al azar: una razón de 2,37. Ese agrupamiento espacio-temporal responde a
causas reales —una ola de calor afecta a una comarca entera, y las quemas agrícolas y los
incendios intencionados se concentran— y el modelo actual no lo ve.

**Continuidad del combustible.** La fracción forestal media del entorno, que mide si el
combustible forma una masa continua o está fragmentado.

## La trampa que hay que evitar

El datacubo ya trae `is_near_ignition_25x25_10d`, que marca el entorno de una ignición durante
**el día del evento y los diez anteriores**. Incluir el propio día la convierte en una respuesta
disfrazada de pregunta, y por eso el contrato la prohíbe como predictor.

Aquí la ventana temporal termina **estrictamente en D-1**. No es un detalle de implementación
sino la condición que hace legítimas estas variables, así que se comprueba con una prueba
dedicada: una ignición aislada no puede aparecer en su propio historial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.entrenamiento.contrato import COL_CELDA, COL_FECHA, COL_TARGET, Contrato

logger = logging.getLogger(__name__)

#: Lado de la celda en metros. El datacubo es una rejilla regular de 1 km en EPSG:3035.
LADO_CELDA = 1000.0

#: Radios de vecindad en celdas (equivalen a kilómetros). Uno corto para el entorno inmediato
#: y otro comarcal, que es la escala a la que se agrupan las igniciones.
RADIOS = (5, 12)

#: Ventanas temporales en días. Siempre terminan en D-1.
VENTANAS = (7, 30)

#: Tope para los días transcurridos desde la ignición cercana más reciente. Sin tope, las
#: celdas donde nunca ha ardido tendrían un valor arbitrario que el árbol trataría como
#: información; con tope, todas comparten el mismo valor de «hace mucho o nunca».
DIAS_MAXIMO = 90

FRACCIONES_BOSQUE = ("broadleaf_forest", "coniferous_forest", "mixed_forest")


def _day_number(values: pd.Series) -> np.ndarray:
    """Convierte fechas a días desde el epoch sin depender de la unidad de Pandas.

    Pandas 3 puede representar ``datetime64`` con precisión de microsegundos, mientras que
    versiones anteriores solían usar nanosegundos. Convertir directamente con ``astype`` y
    dividir por una constante fija hace que varios días distintos colapsen en el mismo día.
    """
    dates = pd.to_datetime(values, errors="raise")
    return (
        (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)
    ).to_numpy(dtype=np.int64)


@dataclass(frozen=True)
class Rejilla:
    """Traducción entre coordenadas proyectadas y posiciones de una matriz.

    Se construye a partir de las coordenadas del propio dataset en vez de suponer la geometría,
    porque `cell_id` no sigue una fórmula fila-columna deducible: la rejilla original incluye
    celdas fuera de Galicia que no se exportan.
    """

    x_min: float
    y_max: float
    n_filas: int
    n_columnas: int

    def indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fila y columna de cada coordenada."""
        columna = np.rint((np.asarray(x) - self.x_min) / LADO_CELDA).astype(np.int32)
        fila = np.rint((self.y_max - np.asarray(y)) / LADO_CELDA).astype(np.int32)
        return fila, columna

    @property
    def forma(self) -> tuple[int, int]:
        return self.n_filas, self.n_columnas


def construir_rejilla(x: np.ndarray, y: np.ndarray) -> Rejilla:
    """Deduce la geometría de la rejilla a partir de las coordenadas presentes."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x_min, y_max = float(x.min()), float(y.max())
    n_columnas = int(np.rint((x.max() - x_min) / LADO_CELDA)) + 1
    n_filas = int(np.rint((y_max - y.min()) / LADO_CELDA)) + 1
    return Rejilla(x_min, y_max, n_filas, n_columnas)


def _suma_ventana(matriz: np.ndarray, radio: int) -> np.ndarray:
    """Suma de cada ventana cuadrada de lado ``2*radio+1``, mediante imagen integral.

    Se implementa con sumas acumuladas en vez de con una convolución porque el coste no depende
    del radio: con radio 12 la ventana tiene 625 celdas y una convolución directa las recorrería
    todas para cada píxel.
    """
    acumulada = np.pad(matriz, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    n_filas, n_columnas = matriz.shape
    filas = np.arange(n_filas)
    columnas = np.arange(n_columnas)

    f0 = np.clip(filas - radio, 0, n_filas)
    f1 = np.clip(filas + radio + 1, 0, n_filas)
    c0 = np.clip(columnas - radio, 0, n_columnas)
    c1 = np.clip(columnas + radio + 1, 0, n_columnas)

    return (acumulada[np.ix_(f1, c1)] - acumulada[np.ix_(f0, c1)]
            - acumulada[np.ix_(f1, c0)] + acumulada[np.ix_(f0, c0)])


@dataclass
class ContextoEspacial:
    """Historial de igniciones ya agregado, listo para consultarse fila a fila.

    Attributes:
        rejilla: Geometría de la malla.
        dia_minimo: Día entero (desde época) del primer día cubierto.
        acumuladas: Para cada radio, matriz `(día, fila, columna)` con el número de igniciones
            ocurridas en la ventana espacial durante los días anteriores. Ya excluye el día
            propio: la posición `d` contiene el recuento de `[d-ventana, d-1]`.
        cobertura: Fracción forestal media del entorno, por radio. Es estática.
    """

    rejilla: Rejilla
    dia_minimo: int
    acumuladas: dict[tuple[int, int], np.ndarray]
    ultima_ignicion: dict[int, np.ndarray]
    cobertura: dict[int, np.ndarray]
    radios: tuple[int, ...]
    ventanas: tuple[int, ...]

    def resumen(self) -> str:
        forma = next(iter(self.acumuladas.values())).shape
        return (f"Contexto espacial: rejilla {self.rejilla.forma}, {forma[0]} días, "
                f"radios {self.radios} km, ventanas {self.ventanas} días")


def columnas_vecindad(radios: Sequence[int] = RADIOS,
                      ventanas: Sequence[int] = VENTANAS) -> list[str]:
    """Nombres de las columnas que `anadir_vecindad` va a generar."""
    nombres = []
    for radio in radios:
        for ventana in ventanas:
            nombres.append(f"igniciones_{radio}km_{ventana}d")
        nombres.append(f"dias_desde_ignicion_{radio}km")
        nombres.append(f"forestal_vecindad_{radio}km")
    return nombres


def ajustar_contexto_espacial(
    marco: pd.DataFrame,
    radios: Sequence[int] = RADIOS,
    ventanas: Sequence[int] = VENTANAS,
    dias_maximo: int = DIAS_MAXIMO,
) -> ContextoEspacial:
    """Precalcula el historial espacial de igniciones a partir de la población completa.

    Debe recibir **todas** las filas de los años implicados, no una muestra: el historial cuenta
    igniciones ocurridas, y con negativos submuestreados el recuento seguiría siendo correcto
    —los positivos se conservan siempre— pero las coordenadas de las celdas sin ignición harían
    falta igualmente para construir la rejilla.

    Args:
        marco: Filas con `x`, `y`, `fecha` y `target_ignicion`.
        radios: Radios de vecindad en celdas.
        ventanas: Ventanas temporales en días, todas terminadas en D-1.
        dias_maximo: Tope para los días desde la última ignición cercana.
    """
    faltan = [c for c in ("x", "y", COL_FECHA, COL_TARGET) if c not in marco.columns]
    if faltan:
        raise KeyError(f"El contexto espacial necesita columnas ausentes: {faltan}")

    rejilla = construir_rejilla(marco["x"].to_numpy(), marco["y"].to_numpy())
    dias = _day_number(marco[COL_FECHA])
    dia_minimo, dia_maximo = int(dias.min()), int(dias.max())
    n_dias = dia_maximo - dia_minimo + 1
    fila, columna = rejilla.indices(marco["x"].to_numpy(), marco["y"].to_numpy())

    logger.info("Contexto espacial: rejilla %s, %s días, %s igniciones",
                rejilla.forma, n_dias, int(marco[COL_TARGET].sum()))

    # Malla densa de igniciones por día. Con 222x201 celdas y ~1.500 días son unos 67 MB
    # en enteros de 16 bits, perfectamente manejable.
    positivos = marco[COL_TARGET].to_numpy() == 1

    # Una misma ignición repetida en el marco se sumaría dos veces, y el historial dejaría de
    # ser un recuento de igniciones para pasar a ser un recuento de filas. Ya pasó al reunir
    # el marco a partir de dos consultas solapadas, así que se comprueba. Solo se miran los
    # positivos: son los únicos que entran en la malla y son pocos, de modo que el guardián
    # no encarece el caso de recibir la población completa.
    clave = np.stack([dias[positivos] - dia_minimo, fila[positivos], columna[positivos]])
    if np.unique(clave, axis=1).shape[1] != int(positivos.sum()):
        raise ValueError(
            "Hay igniciones repetidas (misma celda y mismo día) en el marco del contexto. "
            "El historial las contaría dos veces; deduplica antes de ajustar el contexto."
        )

    malla = np.zeros((n_dias, *rejilla.forma), dtype=np.int16)
    np.add.at(malla, (dias[positivos] - dia_minimo, fila[positivos], columna[positivos]), 1)

    acumuladas: dict[tuple[int, int], np.ndarray] = {}
    for radio in radios:
        # Suma espacial primero: es lineal, así que el orden con la suma temporal es
        # indiferente, y hacerlo así evita repetir la ventana espacial por cada ventana
        # temporal.
        espacial = np.stack([_suma_ventana(malla[d], radio) for d in range(n_dias)])
        acumulada_temporal = np.cumsum(espacial, axis=0)

        for ventana in ventanas:
            resultado = np.zeros_like(espacial)
            # Posición d = suma de [d-ventana, d-1]. El limite superior es d-1, nunca d:
            # es lo que impide que una ignición aparezca en su propio historial.
            for d in range(1, n_dias):
                inicio = max(d - ventana, 0)
                previo = acumulada_temporal[inicio - 1] if inicio > 0 else 0
                resultado[d] = acumulada_temporal[d - 1] - previo
            acumuladas[(radio, ventana)] = resultado.astype(np.int32)

        del espacial, acumulada_temporal

    # Días transcurridos desde la ignición más reciente del entorno, también hasta D-1.
    ultima: dict[int, np.ndarray] = {}
    for radio in radios:
        espacial = np.stack([_suma_ventana(malla[d], radio) for d in range(n_dias)]) > 0
        transcurridos = np.full((n_dias, *rejilla.forma), dias_maximo, dtype=np.int16)
        ultimo_visto = np.full(rejilla.forma, -10_000, dtype=np.int32)
        for d in range(n_dias):
            # Se consulta antes de actualizar, de modo que el día propio nunca cuenta.
            transcurridos[d] = np.minimum(d - ultimo_visto, dias_maximo)
            ultimo_visto = np.where(espacial[d], d, ultimo_visto)
        ultima[radio] = transcurridos
        del espacial

    # Continuidad del combustible: estática, se calcula una vez sobre un día cualquiera.
    cobertura: dict[int, np.ndarray] = {}
    disponibles = [c for c in FRACCIONES_BOSQUE if c in marco.columns]
    if disponibles:
        primer_dia = marco[dias == dia_minimo]
        f0, c0 = rejilla.indices(primer_dia["x"].to_numpy(), primer_dia["y"].to_numpy())
        forestal = np.zeros(rejilla.forma, dtype=np.float32)
        presente = np.zeros(rejilla.forma, dtype=np.float32)
        forestal[f0, c0] = primer_dia[disponibles].sum(axis=1).to_numpy()
        presente[f0, c0] = 1.0
        for radio in radios:
            suma = _suma_ventana(forestal, radio)
            cuenta = _suma_ventana(presente, radio)
            cobertura[radio] = np.where(cuenta > 0, suma / np.maximum(cuenta, 1e-9), np.nan)

    return ContextoEspacial(rejilla, dia_minimo, acumuladas, ultima, cobertura,
                            tuple(radios), tuple(ventanas))


def ajustar_desde_contrato(
    contrato: Contrato,
    anios: Sequence[int],
    radios: Sequence[int] = RADIOS,
    ventanas: Sequence[int] = VENTANAS,
    dias_maximo: int = DIAS_MAXIMO,
) -> ContextoEspacial:
    """Construye el contexto leyendo del dataset, como hace `derivadas.ajustar_contexto`.

    No hace falta recorrer los cincuenta millones de filas: el historial solo cuenta
    igniciones, y la geometría de la rejilla sale de un único día completo. Se leen los
    positivos de todos los años más el primer y el último día, que delimitan el rango temporal.
    """
    import pyarrow.dataset as pads

    columnas = ["x", "y", COL_FECHA, COL_TARGET, *FRACCIONES_BOSQUE]
    conjunto = pads.dataset([str(p) for p in contrato.rutas(anios)], format="parquet")
    positivos = conjunto.to_table(
        columns=columnas, filter=(pads.field(COL_TARGET) == 1)).to_pandas()

    bordes = []
    for anio, extremo in ((min(anios), "min"), (max(anios), "max")):
        uno = pads.dataset([str(contrato.ruta(anio))], format="parquet")
        fechas = uno.to_table(columns=[COL_FECHA]).to_pandas()[COL_FECHA]
        limite = fechas.min() if extremo == "min" else fechas.max()
        bordes.append(uno.to_table(
            columns=columnas, filter=(pads.field(COL_FECHA) == limite)).to_pandas())

    # Las igniciones que caen en un día de borde llegan por las dos consultas. Sin deduplicar,
    # el historial las contaría como dos incendios distintos.
    marco = pd.concat([positivos, *bordes], ignore_index=True).drop_duplicates(
        subset=["x", "y", COL_FECHA], keep="first", ignore_index=True)

    return ajustar_contexto_espacial(marco, radios, ventanas, dias_maximo)


def anadir_vecindad(marco: pd.DataFrame, contexto: ContextoEspacial) -> pd.DataFrame:
    """Añade las variables de vecindad a un lote ya cargado.

    Es una consulta a tablas precalculadas, así que no depende de qué otros lotes se hayan
    procesado antes y permite evaluar el año completo por lotes acotados.
    """
    marco = marco.copy()
    dias = _day_number(marco[COL_FECHA]) - contexto.dia_minimo
    fila, columna = contexto.rejilla.indices(marco["x"].to_numpy(), marco["y"].to_numpy())

    n_dias = next(iter(contexto.acumuladas.values())).shape[0]
    dentro = (dias >= 0) & (dias < n_dias)
    if not dentro.all():
        raise ValueError(
            f"{int((~dentro).sum()):,} filas caen fuera del rango temporal del contexto. "
            "El contexto debe ajustarse sobre todos los años que se vayan a puntuar."
        )

    for radio in contexto.radios:
        for ventana in contexto.ventanas:
            marco[f"igniciones_{radio}km_{ventana}d"] = (
                contexto.acumuladas[(radio, ventana)][dias, fila, columna].astype(np.float32)
            )
        marco[f"dias_desde_ignicion_{radio}km"] = (
            contexto.ultima_ignicion[radio][dias, fila, columna].astype(np.float32)
        )
        if radio in contexto.cobertura:
            marco[f"forestal_vecindad_{radio}km"] = (
                contexto.cobertura[radio][fila, columna].astype(np.float32)
            )

    return marco
