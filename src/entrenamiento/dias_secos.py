"""Reparación de `consecutive_dry_days` para Parquet construidos con versiones antiguas.

Por qué existe este módulo
--------------------------
Las versiones antiguas del datacubo exportaban `consecutive_dry_days`, pero el valor publicado
no era reproducible a partir del resto de la fila. Se calcula sobre la rejilla original de ERA5 (unos 9 km) y
después se interpola espacialmente a la rejilla de 1 km, y esa interpolación destruye el
significado de la variable.

La razón es que interpolar conmuta con unas operaciones y no con otras. Una media móvil
sobrevive: el promedio de varias medias sigue siendo una media, y por eso
`relative_humidity_mean_14d` y `wind_speed_mean_7d` sí coinciden hasta el último decimal al
recalcularlas. Una racha de días secos, en cambio, es una operación no lineal y con memoria:
su valor de hoy depende de toda la historia previa de la celda. Al interpolarla se está
promediando la racha de una celda que lleva un mes seca con la de su vecina, a la que llovió
ayer, y el resultado no es la racha de ninguna de las dos.

Evidencia medida sobre el datacubo antiguo, año 2022 (29.601 celdas x 365 días)
-------------------------------------------------------------------------------
- El 11,25 % de los valores publicados no son enteros, siendo una cuenta de días.
- 42.550 celda-día en los que la celda no registra lluvia en su propia `precipitation_sum`
  y aun así el contador desciende.
- 123.935 celda-día en los que llueven más de 1 mm y el contador no se reinicia.

Estado actual
-------------
El pipeline de ingesta vigente deriva la racha al final del proceso, desde la
``precipitation_sum`` ya interpolada al grid de 1 km. Esa versión conserva la continuidad entre
años y publica un contador entero, de modo que sobre el datacubo actual **este módulo no llega
a aplicarse**.

No se retira porque la decisión no debe depender de que nadie recuerde qué versión del dataset
tiene delante. `pipeline_definitivo.py` llama a `diagnosticar()` al arrancar y solo repara si
encuentra el patrón del fallo; sobre el datacubo vigente el diagnóstico da cero en ambos
síntomas y la columna se usa tal cual viene.

Qué hace este módulo
--------------------
Recalcula la variable directamente desde la columna `precipitation_sum` que el Parquet ya
publica, de modo que el valor sea coherente con la lluvia de su propia fila y reproducible
por cualquiera que tenga el dataset. No hace falta regenerar el datacubo.

Convenio temporal
-----------------
La racha **incluye la fecha T**, igual que el resto de acumulados del dataset. Vale 0 el día
que llueve y suma uno por cada jornada seca encadenada.

Limitación conocida
-------------------
El recálculo se hace año a año, así que el contador se reinicia el 1 de enero y las rachas
que cruzan el fin de año quedan truncadas. Afecta a los primeros días de enero, fuera de la
temporada de incendios, y en la práctica no toca ninguna ignición de campaña. Se documenta
porque es una diferencia real frente a la versión del datacubo, que sí arrastra diciembre.

Uso
---
    from src.entrenamiento import contrato, dias_secos

    c = contrato.cargar()
    dias_secos.construir_cache(c, c.anios)        # una sola vez
    marco = dias_secos.corregir(marco, [2019, 2020])
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

RAIZ = Path(__file__).resolve().parents[2]
DIR_CACHE = RAIZ / "data" / "processed" / "dias_secos"

COL_CELDA = "cell_id"
COL_FECHA = "fecha"

#: Nombre de la variable, idéntico al del datacubo: se sustituye en el sitio.
VARIABLE = "consecutive_dry_days"

#: Columna de la que se deriva.
ORIGEN = "precipitation_sum"

#: Umbral de lluvia significativa, en milímetros. Por debajo, el día cuenta como seco.
UMBRAL_LLUVIA_MM = 1.0


def rachas(matriz_precipitacion: np.ndarray) -> np.ndarray:
    """Días consecutivos sin lluvia significativa, contando el día en curso.

    Recibe una matriz (celdas x días) ordenada por fecha dentro de cada celda. Se recorre el
    eje temporal —365 iteraciones— resolviendo todas las celdas a la vez en cada paso, en
    lugar de agrupar por celda, que exigiría 29.601 pasadas.
    """
    n_celdas, n_dias = matriz_precipitacion.shape
    seco = matriz_precipitacion < UMBRAL_LLUVIA_MM
    salida = np.zeros((n_celdas, n_dias), dtype="float32")
    contador = np.zeros(n_celdas, dtype="float32")
    for j in range(n_dias):
        contador = np.where(seco[:, j], contador + 1.0, 0.0)
        salida[:, j] = contador
    return salida


def _ruta_cache(año: int) -> Path:
    return DIR_CACHE / f"year={año}" / f"dias_secos_{año}.parquet"


def construir_cache(contrato, años: Iterable[int], recalcular: bool = False) -> list[Path]:
    """Recalcula la racha para cada año y la deja en disco.

    Solo se leen del Parquet las tres columnas necesarias, así que un año ocupa unos 130 MB
    en memoria en lugar de los 1,5 GB del fichero completo.

    Args:
        contrato: Contrato del dataset, tal como lo devuelve `contrato.cargar()`.
        años: Años a procesar.
        recalcular: Si es cierto, rehace los años que ya estuvieran en caché.

    Returns:
        Rutas de los ficheros generados o reutilizados.
    """
    rutas = []
    for año in años:
        destino = _ruta_cache(año)
        if destino.exists() and not recalcular:
            print(f"  {año}: ya estaba en caché")
            rutas.append(destino)
            continue

        columnas = [COL_CELDA, COL_FECHA, ORIGEN]
        tabla = pads.dataset(str(contrato.ruta(año)), format="parquet").to_table(columns=columnas)
        marco = tabla.to_pandas(split_blocks=True, self_destruct=True)
        del tabla

        # El fichero no viene agrupado por celda. El orden se calcula aparte y se aplica solo
        # a los vectores necesarios, sin reordenar el marco completo.
        orden = np.lexsort((marco[COL_FECHA].to_numpy(), marco[COL_CELDA].to_numpy()))
        n_celdas = marco[COL_CELDA].nunique()
        n_dias = len(marco) // n_celdas
        if n_celdas * n_dias != len(marco):
            raise ValueError(
                f"El año {año} no tiene una rejilla completa: {len(marco):,} filas no son "
                f"{n_celdas:,} celdas x {n_dias} días. Revisar antes de calcular rachas."
            )

        lluvia = marco[ORIGEN].to_numpy()[orden].reshape(n_celdas, n_dias)
        matriz = rachas(lluvia)

        vector = np.empty(len(marco), dtype="float32")
        vector[orden] = matriz.ravel()

        salida = marco[[COL_CELDA, COL_FECHA]].copy()
        salida[VARIABLE] = vector

        destino.parent.mkdir(parents=True, exist_ok=True)
        salida.to_parquet(destino, index=False)
        print(f"  {año}: {len(salida):,} filas · racha media {vector.mean():.2f} días · "
              f"máxima {vector.max():.0f}")
        rutas.append(destino)
        del marco, salida, matriz, lluvia

    return rutas


def diagnosticar(contrato, año: int, filas: int = 2_000_000) -> dict:
    """Comprueba si la columna publicada es ya un contador de días coherente.

    Existe para que nadie tenga que acordarse de qué versión del datacubo tiene delante. El
    pipeline de ingesta vigente calcula la racha correctamente y esta reparación sobra; sobre
    un Parquet heredado, en cambio, es imprescindible. En lugar de decidirlo por convención
    —que envejece mal y falla en silencio— se mira el dato.

    Dos síntomas bastan para distinguirlos, y ambos son consecuencia directa de haber
    interpolado espacialmente un contador:

    1. **Valores no enteros.** Una cuenta de días no puede valer 8,87. Aparecen al promediar
       las rachas de celdas vecinas que van por días distintos.
    2. **Lluvia sin reinicio.** Si en la celda llueve por encima del umbral, la racha tiene que
       ser cero ese día. Si no lo es, el valor no procede de la lluvia de esa misma fila.

    Returns:
        Diccionario con la proporción de cada síntoma y la conclusión en `necesita_correccion`.
    """
    tabla = pads.dataset(str(contrato.ruta(año)), format="parquet").head(
        filas, columns=[VARIABLE, ORIGEN]
    )
    marco = tabla.to_pandas()
    del tabla

    racha = marco[VARIABLE].to_numpy(dtype="float64")
    lluvia = marco[ORIGEN].to_numpy(dtype="float64")
    validos = np.isfinite(racha) & np.isfinite(lluvia)
    racha, lluvia = racha[validos], lluvia[validos]

    no_enteros = float(np.mean(np.abs(racha - np.round(racha)) > 1e-4)) if len(racha) else 0.0
    llueve = lluvia >= UMBRAL_LLUVIA_MM
    sin_reinicio = float(np.mean(racha[llueve] > 0.5)) if llueve.any() else 0.0

    return {
        "año": año,
        "filas_examinadas": int(len(racha)),
        "proporcion_no_enteros": no_enteros,
        "proporcion_lluvia_sin_reinicio": sin_reinicio,
        # Umbrales holgados: el fallo de interpolación producía un 11 % y un 1,2 %
        # respectivamente, mientras que un contador correcto da exactamente cero en ambos.
        "necesita_correccion": bool(no_enteros > 0.001 or sin_reinicio > 0.001),
    }


def _clave(celdas: np.ndarray, fechas) -> np.ndarray:
    """Comprime el par (celda, fecha) en un único entero.

    Un `merge` de pandas sobre dos columnas construye una tabla hash del marco entero, y la
    caché de un año son once millones de filas. Reducir la pareja a un solo `int64` permite
    resolver el cruce con una búsqueda binaria sobre un vector ordenado, que no copia nada.
    """
    dias = pd.to_datetime(fechas).to_numpy(dtype="datetime64[D]").astype(np.int64)
    return celdas.astype(np.int64) * 100_000 + dias


#: Años ya leídos de disco, como (claves ordenadas, valores). `corregir` se llama una vez por
#: lote de evaluación —unas veinte por año—, y releer el Parquet cada vez dominaría el tiempo.
#: Se guardan vectores de numpy y no un DataFrame: unos 130 MB por año en lugar de 400.
_EN_MEMORIA: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _leer(año: int) -> tuple[np.ndarray, np.ndarray]:
    if año not in _EN_MEMORIA:
        ruta = _ruta_cache(año)
        if not ruta.exists():
            raise FileNotFoundError(
                f"Falta la caché de {año} en {ruta}.\n"
                "Ejecutar primero dias_secos.construir_cache(contrato, años)."
            )
        tabla = pd.read_parquet(ruta)
        claves = _clave(tabla[COL_CELDA].to_numpy(), tabla[COL_FECHA])
        orden = np.argsort(claves, kind="stable")
        _EN_MEMORIA[año] = (claves[orden], tabla[VARIABLE].to_numpy(dtype="float32")[orden])
        del tabla, claves, orden
    return _EN_MEMORIA[año]


def olvidar() -> None:
    """Vacía la caché en memoria. Útil entre años para no acumular vectores grandes."""
    _EN_MEMORIA.clear()


def corregir(marco: pd.DataFrame, años: Sequence[int]) -> pd.DataFrame:
    """Sustituye la columna publicada por la recalculada, emparejando por `(cell_id, fecha)`.

    La columna conserva su nombre y su posición, de modo que el resto del pipeline —la lista de
    predictores del contrato, la ablación, el modelo— no necesita saber nada de esto. El orden
    de las filas del marco de entrada se respeta.

    Raises:
        FileNotFoundError: Si falta la caché de algún año. Ejecutar antes `construir_cache`.
        ValueError: Si alguna fila se queda sin valor, señal de que las claves no casan.
    """
    pedidas = _clave(marco[COL_CELDA].to_numpy(), marco[COL_FECHA])
    valores = np.full(len(marco), np.nan, dtype="float32")

    for año in años:
        claves, rachas_año = _leer(año)
        posicion = np.searchsorted(claves, pedidas)
        # `searchsorted` devuelve dónde *encajaría* cada clave, exista o no. Hay que comprobar
        # que la posición está dentro del vector y que la clave coincide de verdad.
        dentro = posicion < len(claves)
        encontrada = np.zeros(len(marco), dtype=bool)
        encontrada[dentro] = claves[posicion[dentro]] == pedidas[dentro]
        valores[encontrada] = rachas_año[posicion[encontrada]]

    sin_valor = int(np.isnan(valores).sum())
    if sin_valor:
        raise ValueError(
            f"{sin_valor:,} filas se quedaron sin racha. Suele indicar que el marco trae "
            f"fechas de años ausentes de {list(años)}, o que `fecha` no es una fecha."
        )

    resultado = marco.copy()
    resultado[VARIABLE] = valores
    return resultado
