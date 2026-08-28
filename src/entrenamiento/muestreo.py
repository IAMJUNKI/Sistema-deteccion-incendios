"""Estrategias de submuestreo de la clase negativa.

Con una prevalencia de 1,5·10⁻⁴ hay que descartar negativos para que el gradiente aprenda algo.
La pregunta no es *si* submuestrear sino **cuántos** conservar y **cuáles**, y ninguna de las dos
tiene respuesta escrita en este proyecto: el módulo 25 se heredó del pipeline común sin
justificación, y el borrador de la memoria describe una estrategia distinta —minado de negativos
duros a 1:50— que además está medida y empeora el modelo.

## La restricción que condiciona todo el diseño

La corrección de prior de King y Zeng multiplica las *odds* del modelo por la fracción de
negativos conservada. Esa fórmula solo es válida si **todos los negativos tuvieron la misma
probabilidad de sobrevivir al muestreo**. En cuanto se estratifica o se agrupa por clusters, cada
negativo tiene su propia tasa y aplicar una tasa global deja la calibración mal sin que nada
falle de forma visible: las probabilidades salen plausibles y son incorrectas.

Por eso toda estrategia de este módulo cumple dos condiciones:

1. Dentro de cada estrato el muestreo es **uniforme y con tasa conocida**.
2. Se devuelve una **tasa por fila**, y el estrato se puede recalcular a partir de las variables
   de cualquier fila, también en producción, donde no hay etiqueta.

Con eso la corrección de prior sigue siendo exacta, solo que aplicada fila a fila.

## Por qué el minado de negativos duros está aquí como control negativo

Entrenar solo con los negativos que el modelo confunde parece buena idea y es la estrategia que
la memoria describe. Medida sobre este proyecto, hundió el recall de LightGBM de 0,694 a 0,555.
El motivo es que introduce un desplazamiento de covariables: la distribución de negativos con la
que se entrena deja de parecerse a la que habrá en producción, y la corrección de prior no puede
arreglar eso, porque no es un cambio de prevalencia sino un cambio de forma. Se conserva
implementada para poder reproducir el resultado negativo, que es lo que justifica haberla
descartado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRIMO_HASH = 1_000_003

#: ## Por qué muestrear dentro de una muestra exige azar y no otro hash
#:
#: El estudio de muestreo carga un embalse por hash y después reparte dentro de él. La tentación
#: es reutilizar el hash con otro módulo, pero **ningún hash determinista sobre las mismas claves
#: es independiente del primero**, y el fallo es peor que perder algo de aleatoriedad.
#:
#: Con el embalse en `3c + d ≡ 0 (mod 5)`, la condición del embalse fija `d ≡ 2c (mod 5)`. Al
#: aplicar un segundo hash `c + 3d (mod 5)` sobre esas filas, sustituyendo queda `7c ≡ 2c`, de
#: modo que la selección se reduce a `c ≡ 0 (mod 5)`: **elige celdas enteras en vez de
#: celdas-día**. El recuento de filas sale correcto y el muestreo parece funcionar, pero el
#: modelo solo ve un quinto del territorio con todos sus días. En el barrido eso hundió el recall
#: de 0,39 a 0,24 sin que nada fallara.
#:
#: Cambiar de primos no lo arregla: cualquier par de hashes lineales sobre `(celda, día)` acaba
#: teniendo esta clase de degeneración para algún módulo. La solución es muestrear al azar dentro
#: del embalse, que además es exactamente lo correcto: el embalse ya es insesgado, así que una
#: selección aleatoria uniforme dentro de él compone las tasas por construcción.

#: Estratos por defecto para la estrategia estratificada. Mes y combustible son las dos
#: dimensiones con las que varía de verdad la dificultad de un negativo: un día de agosto sobre
#: matorral no se parece en nada a uno de febrero sobre cultivo.
ESTRATOS_POR_DEFECTO = ("mes", "combustible")


@dataclass(frozen=True)
class Muestra:
    """Resultado de una estrategia de muestreo.

    Attributes:
        mascara: Booleano por fila del marco original: qué se conserva.
        tasas: Tasa de muestreo aplicada a cada fila conservada. Los positivos llevan 1,0
            porque se conservan todos. Es el vector que consume la corrección de prior.
        estrato: Identificador del estrato de cada fila conservada, para poder auditar el
            reparto y recalcularlo en producción.
        meta: Diccionario descriptivo de la estrategia y su resultado.
    """

    mascara: np.ndarray
    tasas: np.ndarray
    estrato: np.ndarray
    meta: dict

    def aplicar(self, marco: pd.DataFrame) -> pd.DataFrame:
        """Devuelve el marco muestreado con la tasa y el estrato como columnas."""
        salida = marco.loc[self.mascara].copy()
        salida["tasa_muestreo"] = self.tasas
        salida["estrato"] = self.estrato
        return salida


def _hash_fila(cell_id: np.ndarray, dia: np.ndarray, modulo: int) -> np.ndarray:
    """Hash reproducible celda-día, idéntico al de `src/modeling/data.py`."""
    return (cell_id.astype(np.int64) * PRIMO_HASH + dia) % modulo


def _seleccionar_negativos(objetivo: np.ndarray, cell_id: np.ndarray, dia: np.ndarray,
                           modulo: int, resto: int, independiente: bool,
                           semilla: int) -> np.ndarray:
    """Máscara de filas conservadas: todos los positivos más 1 de cada `modulo` negativos.

    Con `independiente`, la selección se hace al azar en vez de por hash, que es lo único
    correcto cuando se muestrea dentro de un embalse ya obtenido por hash (ver la nota sobre
    degeneración al principio del módulo).
    """
    mascara = objetivo == 1
    if not independiente:
        return mascara | (_hash_fila(cell_id, dia, modulo) == resto)

    negativos = np.flatnonzero(objetivo == 0)
    cupo = min(int(round(len(negativos) / modulo)), len(negativos))
    if cupo:
        elegidos = np.random.default_rng(semilla + resto).choice(
            negativos, size=cupo, replace=False
        )
        mascara[elegidos] = True
    return mascara


def uniforme(
    objetivo: np.ndarray, cell_id: np.ndarray, dia: np.ndarray,
    modulo: int = 25, resto: int = 0, independiente: bool = False, semilla: int = 42,
) -> Muestra:
    """Conserva todos los positivos y un negativo de cada `modulo`.

    Es la estrategia actual y la referencia contra la que se comparan las demás. Su virtud es
    que la tasa es la misma para todos los negativos, así que la corrección de prior es un
    único número y no hay nada que pueda salir mal.

    Args:
        independiente: Selecciona al azar en vez de por hash. Obligatorio cuando el marco ya es
            una muestra obtenida con este mismo hash.
    """
    mascara = _seleccionar_negativos(objetivo, cell_id, dia, modulo, resto,
                                     independiente, semilla)
    # La tasa se mide sobre lo realmente conservado, no se supone: ni el hash ni el redondeo
    # del cupo reparten exactamente 1/modulo.
    total_neg = int((objetivo == 0).sum())
    conservados_neg = int((objetivo[mascara] == 0).sum())
    tasa = conservados_neg / total_neg if total_neg else 1.0
    tasas = np.where(objetivo[mascara] == 1, 1.0, tasa)
    return Muestra(
        mascara=mascara,
        tasas=tasas,
        estrato=np.zeros(int(mascara.sum()), dtype=np.int32),
        meta={"estrategia": "uniforme", "modulo": modulo, "resto": resto,
              "n_estratos": 1, "tasa_global": tasa, "independiente": independiente},
    )


def estratificado(
    objetivo: np.ndarray, cell_id: np.ndarray, dia: np.ndarray,
    claves: pd.DataFrame, modulo: int = 25, resto: int = 0,
    minimo_por_estrato: int = 200, independiente: bool = False, semilla: int = 42,
) -> Muestra:
    """Muestrea al mismo ritmo dentro de cada estrato definido por `claves`.

    Frente al uniforme, garantiza que ninguna combinación poco frecuente —agosto sobre
    humedal, por ejemplo— desaparezca por azar. Un estrato con menos de `minimo_por_estrato`
    negativos se conserva **entero**, porque submuestrear lo que ya es escaso solo destruye
    información; su tasa pasa a ser 1,0 y la corrección de prior lo tiene en cuenta.

    Args:
        claves: Columnas categóricas que definen el estrato, alineadas con `objetivo`.
    """
    llave = claves.astype(str).agg("|".join, axis=1).to_numpy()
    codigos, indices = np.unique(llave, return_inverse=True)

    conservar = _seleccionar_negativos(objetivo, cell_id, dia, modulo, resto,
                                       independiente, semilla)

    # Estratos escasos: se rescatan enteros.
    es_negativo = objetivo == 0
    rescatados = 0
    for codigo in range(len(codigos)):
        del_estrato = es_negativo & (indices == codigo)
        if del_estrato.sum() < minimo_por_estrato:
            conservar |= del_estrato
            rescatados += 1

    tasas = np.empty(int(conservar.sum()), dtype=np.float64)
    obj_muestra = objetivo[conservar]
    ind_muestra = indices[conservar]
    tasas[obj_muestra == 1] = 1.0

    # La tasa real de cada estrato se mide, no se supone: el hash no reparte exactamente
    # 1/modulo en estratos pequeños, y usar el valor nominal descalibraría esas filas.
    for codigo in range(len(codigos)):
        en_estrato_total = int((es_negativo & (indices == codigo)).sum())
        if en_estrato_total == 0:
            continue
        sel = (obj_muestra == 0) & (ind_muestra == codigo)
        conservados = int(sel.sum())
        tasas[sel] = conservados / en_estrato_total if conservados else 1.0

    logger.info("Estratificado: %s estratos, %s rescatados por escasez", len(codigos), rescatados)
    return Muestra(
        mascara=conservar,
        tasas=tasas,
        estrato=ind_muestra.astype(np.int32),
        meta={"estrategia": "estratificado", "modulo": modulo, "resto": resto,
              "n_estratos": int(len(codigos)), "estratos_rescatados": rescatados,
              "claves": list(claves.columns)},
    )


def por_clusters(
    objetivo: np.ndarray, variables: pd.DataFrame,
    modulo: int = 25, n_clusters: int = 60, semilla: int = 42,
    submuestra_ajuste: int = 200_000, asignacion: str = "proporcional",
) -> Muestra:
    """Agrupa los negativos en el espacio de variables y muestrea proporcionalmente en cada grupo.

    La idea, de muestreo por diversidad: el muestreo uniforme reparte el presupuesto según la
    frecuencia de cada región del espacio, así que las condiciones raras —que suelen ser las
    difíciles— se llevan pocas filas. Agrupar primero y muestrear dentro de cada grupo garantiza
    que todas las regiones estén representadas.

    Se usa `MiniBatchKMeans` porque hay decenas de millones de negativos, y se ajusta sobre una
    submuestra: los centroides no necesitan ver todos los puntos para quedar bien situados.

    ## Las dos formas de repartir el presupuesto, y en qué se diferencian de verdad

    - **`proporcional`**: cada grupo cede la misma *fracción* de sus filas. La composición
      esperada de la muestra es **idéntica** a la del muestreo uniforme; lo único que cambia es
      que la cobertura queda garantizada en vez de dejada al azar. Es reducción de varianza, como
      el muestreo estratificado en encuestas, no un cambio de sesgo. Con más de un millón de
      negativos esa varianza ya es despreciable, así que **no se espera ninguna mejora**: está
      como referencia intermedia entre el uniforme y el equilibrado.

    - **`equilibrado`**: cada grupo cede el mismo *número* de filas. Esto sí cambia la
      composición: sobrerrepresenta a propósito las regiones poco pobladas del espacio de
      variables, que suelen ser las condiciones raras y difíciles. Cada grupo acaba con su propia
      tasa de muestreo —muy distintas entre sí— y sin corrección por fila la calibración quedaría
      mal. Es la variante que puede aportar algo, y también la única que necesita de verdad la
      maquinaria de tasas por fila.

    Args:
        asignacion: `"proporcional"` o `"equilibrado"`.

    Raises:
        ValueError: Si la asignación no es una de las dos.
    """
    if asignacion not in ("proporcional", "equilibrado"):
        raise ValueError(
            f"asignacion debe ser 'proporcional' o 'equilibrado', recibido: {asignacion!r}"
        )
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    es_negativo = objetivo == 0
    indices_neg = np.flatnonzero(es_negativo)

    rng = np.random.default_rng(semilla)
    ajuste = rng.choice(indices_neg, size=min(submuestra_ajuste, len(indices_neg)),
                        replace=False)

    escalador = StandardScaler()
    X_ajuste = escalador.fit_transform(
        variables.iloc[ajuste].to_numpy(dtype=np.float32, na_value=0.0)
    )
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=semilla,
                             batch_size=4096, n_init=3).fit(X_ajuste)

    etiquetas = np.full(len(objetivo), -1, dtype=np.int32)
    paso = 500_000
    for inicio in range(0, len(indices_neg), paso):
        bloque = indices_neg[inicio:inicio + paso]
        X = escalador.transform(
            variables.iloc[bloque].to_numpy(dtype=np.float32, na_value=0.0)
        )
        etiquetas[bloque] = kmeans.predict(X)

    objetivo_total = int(round(len(indices_neg) / modulo))
    poblados = [c for c in range(n_clusters) if (etiquetas == c).any()]
    conservar = objetivo == 1
    tasas_por_cluster = {}

    # En el reparto equilibrado el cupo por grupo es fijo, así que los grupos pequeños se
    # agotan y ceden su sobrante. Se recorren de menor a mayor para que ese sobrante se
    # redistribuya entre los que aún tienen filas, en vez de perderse.
    orden = (sorted(poblados, key=lambda c: int((etiquetas == c).sum()))
             if asignacion == "equilibrado" else poblados)

    restante = objetivo_total
    for posicion, c in enumerate(orden):
        del_cluster = np.flatnonzero(etiquetas == c)
        if asignacion == "proporcional":
            cupo = max(1, int(round(len(del_cluster) * objetivo_total / len(indices_neg))))
        else:
            cupo = int(round(restante / (len(orden) - posicion)))
        cupo = max(1, min(cupo, len(del_cluster)))
        elegidos = rng.choice(del_cluster, size=cupo, replace=False)
        conservar[elegidos] = True
        tasas_por_cluster[c] = cupo / len(del_cluster)
        restante = max(restante - cupo, 0)

    obj_muestra = objetivo[conservar]
    etiq_muestra = etiquetas[conservar]
    tasas = np.where(
        obj_muestra == 1, 1.0,
        np.array([tasas_por_cluster.get(int(e), 1.0) for e in etiq_muestra]),
    )

    valores = np.array(list(tasas_por_cluster.values()))
    logger.info(
        "Clusters (%s): %s grupos, %s negativos de %s, tasas de %.5f a %.5f",
        asignacion, len(tasas_por_cluster), int((obj_muestra == 0).sum()),
        len(indices_neg), valores.min(), valores.max(),
    )
    return Muestra(
        mascara=conservar,
        tasas=tasas,
        estrato=etiq_muestra.astype(np.int32),
        meta={"estrategia": f"clusters_{asignacion}", "modulo": modulo,
              "n_clusters": n_clusters, "n_estratos": len(tasas_por_cluster),
              "asignacion": asignacion, "semilla": semilla,
              "tasa_minima": float(valores.min()), "tasa_maxima": float(valores.max())},
    )


def duros(
    objetivo: np.ndarray, cell_id: np.ndarray, dia: np.ndarray,
    dificultad: np.ndarray, modulo: int = 25, fraccion_dura: float = 0.8,
    resto: int = 0,
) -> Muestra:
    """Control negativo: prioriza los negativos que más se parecen a un incendio.

    **Esta estrategia empeora el modelo y está aquí para documentarlo**, porque es la que
    describe el borrador de la memoria. Medida sobre este proyecto hundió el recall de LightGBM
    de 0,694 a 0,555.

    La razón es que rompe la premisa de la corrección de prior de una forma que ninguna fórmula
    arregla: no cambia la *proporción* de negativos sino su *composición*. El modelo aprende a
    separar incendios de días calurosos secos, que es un problema distinto —y más difícil— del
    que tendrá delante en producción, donde la inmensa mayoría de los días no se parecen nada a
    un incendio.

    Args:
        dificultad: Puntuación por fila; a mayor valor, negativo más parecido a un positivo.
        fraccion_dura: Qué parte del presupuesto se gasta en los negativos más difíciles.
    """
    es_negativo = objetivo == 0
    indices_neg = np.flatnonzero(es_negativo)
    presupuesto = int(round(len(indices_neg) / modulo))
    n_duros = int(round(presupuesto * fraccion_dura))

    orden = indices_neg[np.argsort(-dificultad[indices_neg], kind="stable")]
    elegidos_duros = orden[:n_duros]

    faciles = np.setdiff1d(indices_neg, elegidos_duros, assume_unique=False)
    hash_fila = _hash_fila(cell_id[faciles], dia[faciles], max(modulo, 2))
    elegidos_faciles = faciles[hash_fila == resto][: presupuesto - n_duros]

    conservar = objetivo == 1
    conservar[elegidos_duros] = True
    conservar[elegidos_faciles] = True

    obj_muestra = objetivo[conservar]
    # La tasa deja de tener un significado limpio: el muestreo ya no es aleatorio dentro de
    # ningún estrato. Se devuelve la global como aproximación, y esa es precisamente la razón
    # por la que la calibración de esta estrategia no es de fiar.
    tasas = np.where(obj_muestra == 1, 1.0, presupuesto / len(indices_neg))
    return Muestra(
        mascara=conservar,
        tasas=tasas,
        estrato=np.zeros(int(conservar.sum()), dtype=np.int32),
        meta={"estrategia": "duros", "modulo": modulo, "fraccion_dura": fraccion_dura,
              "n_estratos": 1, "aviso": "calibracion no fiable: muestreo no aleatorio"},
    )


def dificultad_climatica(marco: pd.DataFrame) -> np.ndarray:
    """Puntuación de «parecido a un día de incendio» a partir de calor y sequedad.

    Solo se usa para la estrategia `duros`, como control negativo.
    """
    partes = []
    for columna, signo in (("temperature_max", 1.0), ("vpd_mean", 1.0),
                           ("relative_humidity_min", -1.0), ("precipitation_sum_7d", -1.0)):
        if columna in marco.columns:
            valores = marco[columna].to_numpy(dtype=np.float64)
            desviacion = valores.std() or 1.0
            partes.append(signo * (valores - valores.mean()) / desviacion)
    if not partes:
        raise ValueError("No hay variables meteorológicas para medir la dificultad.")
    return np.sum(partes, axis=0)


ESTRATEGIAS: dict[str, Callable[..., Muestra]] = {
    "uniforme": uniforme,
    "estratificado": estratificado,
    "clusters": por_clusters,
    "duros": duros,
}
