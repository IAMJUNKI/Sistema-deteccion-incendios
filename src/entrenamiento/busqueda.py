"""Búsqueda de hiperparámetros con validación cruzada temporal.

La validación cruzada tal y como se enseña —`KFold`, particiones al azar— es incorrecta en este
problema por dos motivos independientes:

**Rompe el tiempo.** Una partición aleatoria entrena con días de 2022 y valida con días de 2020,
es decir, usa el futuro para predecir el pasado. En un sistema que debe funcionar sobre datos
que aún no existen, eso infla los resultados sin que se note.

**Rompe el espacio.** Dos celdas contiguas el mismo día son casi la misma observación. Repartir
las dos al azar entre entrenamiento y validación es enseñarle al modelo la respuesta.

La alternativa válida es la **ventana expansiva** (*forward chaining*): cada pliegue entrena con
todos los años anteriores y valida con el siguiente, siempre hacia adelante.

    pliegue 1   entrena 2019                validación 2020
    pliegue 2   entrena 2019-2020           validación 2021
    pliegue 3   entrena 2019-2020-2021      validación 2022

Tres pliegues en vez de uno, y una configuración solo gana si funciona en tres años distintos.
Eso es mucho más difícil de conseguir por azar, que es justamente el riesgo de toda búsqueda de
hiperparámetros: con 1.659 incendios el ruido de la métrica ronda los dos puntos, así que probar
doscientas configuraciones y quedarse con la mejor encuentra algo dos puntos por encima del
óptimo real **sin que el modelo sea mejor**.

Por eso este módulo hace tres cosas que la receta estándar no hace:

1. Puntúa con la métrica del proyecto —recall a coste operativo fijo— y no con la que trae
   scikit-learn por defecto, que aquí sería inútil: prediciendo siempre «no incendio» se acierta
   el 99,99 % de las veces.
2. **Desactiva la parada temprana durante la búsqueda.** Detener el entrenamiento mirando el
   mismo pliegue con el que después se puntúa es una fuga sutil pero real: el número de árboles
   se elige con la respuesta delante. Durante la búsqueda el número de árboles es un parámetro
   más; el ganador se reajusta luego con parada temprana en el pipeline normal.
3. No reajusta el ganador internamente (`refit=False`). La confirmación se hace fuera, sobre la
   población completa y con el test pareado, porque una mejora que no supera ese test es ruido
   con buena suerte.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from src.entrenamiento import metricas

logger = logging.getLogger(__name__)


class VentanaExpansiva:
    """Particionador temporal compatible con scikit-learn.

    Cada pliegue entrena con todos los años disponibles anteriores al de validación. El primer
    año nunca se valida, porque no tiene pasado con el que entrenar.
    """

    def __init__(self, anios: np.ndarray, minimo_positivos: int = 100,
                 target: Optional[np.ndarray] = None):
        """
        Args:
            anios: Año de cada fila, alineado con `X`.
            minimo_positivos: Pliegues con menos positivos que esto se descartan; su métrica
                sería ruido y contaminaría la media entre pliegues.
            target: Etiquetas, solo para poder comprobar `minimo_positivos`.
        """
        self.anios = np.asarray(anios)
        self.minimo_positivos = minimo_positivos
        self.target = None if target is None else np.asarray(target)
        self._pliegues = self._construir()

    def _construir(self) -> list[tuple[np.ndarray, np.ndarray]]:
        disponibles = sorted(np.unique(self.anios))
        pliegues: list[tuple[np.ndarray, np.ndarray]] = []
        for i, anio in enumerate(disponibles[1:], start=1):
            entrena = np.flatnonzero(np.isin(self.anios, disponibles[:i]))
            valida = np.flatnonzero(self.anios == anio)
            if self.target is not None:
                positivos = int(self.target[valida].sum())
                if positivos < self.minimo_positivos:
                    logger.warning(
                        "Pliegue con validación %s descartado: solo %s positivos.",
                        anio, positivos,
                    )
                    continue
            pliegues.append((entrena, valida))
        if not pliegues:
            raise ValueError("No se pudo construir ningún pliegue temporal.")
        return pliegues

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self._pliegues

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._pliegues)

    def describir(self) -> str:
        lineas = []
        for entrena, valida in self._pliegues:
            anios_tr = [int(a) for a in sorted(np.unique(self.anios[entrena]))]
            anio_val = int(np.unique(self.anios[valida])[0])
            positivos = "" if self.target is None else \
                f", {int(self.target[valida].sum()):,} positivos"
            lineas.append(
                f"  entrena {anios_tr} ({len(entrena):,} filas) "
                f"-> valida {anio_val} ({len(valida):,} filas{positivos})"
            )
        return "\n".join(lineas)


def scorer_recall_at_fpr(fpr_max: float = 0.05):
    """Devuelve un scorer de scikit-learn que puntúa con el recall a coste operativo fijo.

    Se escribe como callable `(estimator, X, y)` en vez de con `make_scorer` a propósito: la
    firma de `make_scorer` para métricas que necesitan probabilidades ha cambiado entre versiones
    de scikit-learn (`needs_proba` frente a `response_method`), y un callable simple funciona en
    todas sin condicionales.
    """
    def puntuar(estimador: Any, X, y) -> float:
        probabilidades = estimador.predict_proba(X)[:, 1]
        recall, _, _ = metricas.recall_at_fpr(np.asarray(y), probabilidades, fpr_max)
        return recall

    puntuar.__name__ = f"recall_at_fpr{int(fpr_max * 100)}"
    return puntuar


#: Espacios de búsqueda. Distribuciones y no rejillas: con seis parámetros, una rejilla
#: razonable son miles de combinaciones y el muestreo aleatorio cubre mejor el espacio con el
#: mismo presupuesto de cómputo.
#:
#: `n_estimators` entra como parámetro porque la parada temprana está desactivada durante la
#: búsqueda (ver el docstring del módulo). `scale_pos_weight` cubre desde ignorar el desbalanceo
#: hasta compensarlo por completo: en la muestra hay unos 324 negativos por positivo.
ESPACIOS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "n_estimators": randint(200, 1500),
        "learning_rate": loguniform(0.01, 0.12),
        "num_leaves": randint(15, 128),
        "max_depth": randint(4, 12),
        "min_child_samples": randint(50, 1000),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1.0, 100.0),
        "scale_pos_weight": loguniform(1.0, 300.0),
    },
    "xgboost": {
        "n_estimators": randint(200, 1500),
        "learning_rate": loguniform(0.01, 0.12),
        "max_depth": randint(3, 10),
        "min_child_weight": loguniform(1.0, 200.0),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1.0, 100.0),
        "reg_alpha": loguniform(0.01, 10.0),
        "scale_pos_weight": loguniform(1.0, 300.0),
    },
    "random_forest": {
        "n_estimators": randint(200, 600),
        "max_depth": randint(8, 30),
        "min_samples_leaf": randint(5, 200),
        "max_features": uniform(0.2, 0.6),
    },
    "logistic_regression": {
        "C": loguniform(1e-3, 1e2),
    },
}

#: Parámetros que se fijan durante la búsqueda y no se exploran. `subsample_freq` es necesario
#: para que `subsample` surta efecto en LightGBM; sin él se ignora en silencio.
FIJOS: dict[str, dict[str, Any]] = {
    "lightgbm": {"n_jobs": -1, "verbosity": -1, "subsample_freq": 1,
                 "importance_type": "gain"},
    "xgboost": {"n_jobs": -1, "tree_method": "hist", "verbosity": 0},
    "random_forest": {"n_jobs": -1, "class_weight": "balanced_subsample"},
    "logistic_regression": {"max_iter": 2000, "class_weight": "balanced"},
}


def buscar(
    modelo: str,
    X: pd.DataFrame,
    y: pd.Series,
    particionador: VentanaExpansiva,
    n_configuraciones: int = 40,
    fpr_max: float = 0.05,
    semilla: int = 42,
    espacio: Optional[dict] = None,
    ruta_checkpoint: Optional[Path] = None,
    reanudar: bool = True,
    tam_bloque: int = 5,
) -> pd.DataFrame:
    """Explora `n_configuraciones` al azar y devuelve la tabla completa de resultados.

    Se devuelven **todas** las configuraciones, no solo la ganadora: la dispersión entre ellas
    es la que dice si la búsqueda ha encontrado algo o si el modelo es insensible a sus
    hiperparámetros, y esa es una conclusión publicable por sí misma.

    ## Por qué no se usa `RandomizedSearchCV` directamente

    `RandomizedSearchCV.fit()` no devuelve nada hasta haber terminado las `n_iter`
    configuraciones. En una búsqueda de más de una hora cualquier interrupción —una batería que
    se agota, un cierre de sesión— tira a la basura todo el cómputo, aunque fuera por la
    configuración 38 de 40. Ya nos pasó.

    La solución **no** es escribir un bucle de validación cruzada propio: sustituir código
    estándar y probado por código nuestro es precisamente como se cuelan errores sutiles, y en
    una defensa «usamos scikit-learn» vale más que «nos hicimos un bucle».

    Lo que se hace es partir la búsqueda en bloques y dejar que scikit-learn siga haciendo todo
    el trabajo:

    1. `ParameterSampler` genera las combinaciones. Es exactamente la clase que
       `RandomizedSearchCV` usa por dentro, así que con la misma semilla salen las mismas
       configuraciones y en el mismo orden.
    2. Cada bloque se evalúa con `GridSearchCV`, que admite una lista explícita de
       combinaciones. El ajuste, la partición y la puntuación son los de la librería.
    3. Entre bloques se guarda la tabla, y `reanudar` continúa donde se quedó.

    El coste es despreciable y el resultado es idéntico al de `RandomizedSearchCV`, solo que
    una interrupción cuesta como mucho un bloque en vez de la búsqueda entera.

    Args:
        modelo: Nombre del estimador.
        X: Matriz de variables, con todos los años de la ventana.
        y: Etiquetas.
        particionador: Ventana expansiva ya construida.
        n_configuraciones: Cuántas muestrear. Pocas y bien elegidas: cada configuración extra
            aumenta la probabilidad de que la mejor lo sea por azar.
        fpr_max: Coste operativo al que se puntúa.
        semilla: Semilla del muestreo de configuraciones.
        espacio: Espacio propio. Por defecto, el de `ESPACIOS`.
        ruta_checkpoint: CSV donde guardar el avance tras cada bloque.
        reanudar: Si el checkpoint existe y es compatible, retoma en vez de empezar de cero.
        tam_bloque: Configuraciones por llamada a `GridSearchCV`. Más pequeño guarda más a
            menudo y pierde menos ante una interrupción; más grande reduce el número de
            llamadas. Con cinco se pierde como mucho un octavo de una búsqueda de cuarenta.

    Returns:
        Tabla ordenada de mejor a peor, con la media y la desviación entre pliegues.
    """
    from sklearn.model_selection import GridSearchCV, ParameterSampler

    from src.entrenamiento import modelos

    espacio = espacio if espacio is not None else ESPACIOS[modelo]
    base = modelos.construir(modelo, FIJOS.get(modelo, {}), semilla)
    n_pliegues = particionador.get_n_splits()

    # Misma generación que `RandomizedSearchCV`, que instancia esta misma clase con la misma
    # semilla. Fijar la lista por adelantado es lo que permite reanudar por índice.
    candidatas = list(ParameterSampler(espacio, n_configuraciones, random_state=semilla))

    logger.info("Búsqueda en %s: %s configuraciones x %s pliegues (bloques de %s)",
                modelo, len(candidatas), n_pliegues, tam_bloque)
    logger.info("Ventana temporal:\n%s", particionador.describir())

    filas, hechas = _cargar_checkpoint(ruta_checkpoint, reanudar, len(candidatas))
    if hechas:
        logger.info("Reanudando: %s de %s configuraciones ya evaluadas en %s",
                    len(hechas), len(candidatas), ruta_checkpoint)

    pendientes = [(i, p) for i, p in enumerate(candidatas) if i not in hechas]

    for inicio_bloque in range(0, len(pendientes), tam_bloque):
        bloque = pendientes[inicio_bloque:inicio_bloque + tam_bloque]
        reloj = time.perf_counter()

        # GridSearchCV admite una lista de rejillas; cada dict con un único valor por parámetro
        # equivale a una configuración concreta. Así el ajuste, la partición y la puntuación
        # siguen siendo los de scikit-learn.
        rejilla = [{clave: [valor] for clave, valor in parametros.items()}
                   for _, parametros in bloque]
        gs = GridSearchCV(
            estimator=base,
            param_grid=rejilla,
            scoring=scorer_recall_at_fpr(fpr_max),
            cv=particionador,
            # n_jobs=1 a propósito: los estimadores ya paralelizan internamente, y anidar
            # paralelismo multiplica la memoria y vuelve los tiempos impredecibles.
            n_jobs=1,
            refit=False,
            # Una configuración inviable no debe tirar una búsqueda de una hora: se registra
            # con puntuación nula y queda al final de la tabla.
            error_score=np.nan,
        )
        try:
            gs.fit(X, y)
            resultados = pd.DataFrame(gs.cv_results_)
        except Exception:
            # `error_score=nan` absorbe los fallos sueltos, pero scikit-learn vuelve a lanzar
            # si TODAS las configuraciones de la llamada fallaron. Sin esta red, un bloque
            # entero de configuraciones inviables mataría la búsqueda —exactamente la
            # fragilidad que el troceado viene a eliminar—. Se registra el bloque como fallido
            # y se continúa; el log deja constancia para poder revisarlo.
            logger.exception(
                "Falló el bloque completo de configuraciones %s; se registran como fallidas.",
                [i for i, _ in bloque],
            )
            resultados = None

        columnas_pliegue = ([c for c in resultados.columns
                             if c.startswith("split") and c.endswith("_test_score")]
                            if resultados is not None else [])
        for posicion, (indice, parametros) in enumerate(bloque):
            if resultados is None:
                fila = {"configuracion": indice, "recall_medio": float("nan"),
                        "recall_desv": float("nan"), "recall_minimo": float("nan"),
                        "segundos": 0.0}
                fila.update({f"pliegue_{i + 1}": float("nan") for i in range(n_pliegues)})
            else:
                fila = {
                    "configuracion": indice,
                    "recall_medio": float(resultados.at[posicion, "mean_test_score"]),
                    "recall_desv": float(resultados.at[posicion, "std_test_score"]),
                    "recall_minimo": float(resultados.loc[posicion, columnas_pliegue].min()),
                    "segundos": round(
                        float(resultados.at[posicion, "mean_fit_time"]) * n_pliegues, 1
                    ),
                }
                fila.update({f"pliegue_{i + 1}": float(resultados.at[posicion, c])
                             for i, c in enumerate(columnas_pliegue)})
            fila.update({k: (v.item() if hasattr(v, "item") else v)
                         for k, v in parametros.items()})
            filas.append(fila)

        _guardar_checkpoint(ruta_checkpoint, filas)
        mejor = max((f["recall_medio"] for f in filas
                     if not np.isnan(f["recall_medio"])), default=float("nan"))
        logger.info("  [%s/%s] bloque en %.0f min · mejor hasta ahora %.4f",
                    len(filas), len(candidatas), (time.perf_counter() - reloj) / 60, mejor)

    tabla = pd.DataFrame(filas)
    # Las configuraciones fallidas quedan al final y no pueden ganar por accidente.
    return tabla.sort_values("recall_medio", ascending=False,
                             na_position="last").reset_index(drop=True)


def _cargar_checkpoint(
    ruta: Optional[Path], reanudar: bool, n_esperadas: int
) -> tuple[list[dict], set[int]]:
    """Lee el avance previo, descartándolo si no es compatible con esta búsqueda."""
    if ruta is None or not reanudar or not Path(ruta).exists():
        return [], set()
    try:
        previo = pd.read_csv(ruta)
    except (OSError, pd.errors.ParserError):
        logger.warning("El checkpoint %s no se pudo leer; se empieza de cero.", ruta)
        return [], set()

    if "configuracion" not in previo.columns:
        logger.warning("El checkpoint %s no tiene columna 'configuracion'; se descarta.", ruta)
        return [], set()
    if previo["configuracion"].max() >= n_esperadas:
        # Viene de una búsqueda con más configuraciones: reanudar mezclaría dos experimentos.
        logger.warning(
            "El checkpoint %s tiene configuraciones fuera de rango (max %s >= %s); se descarta.",
            ruta, int(previo["configuracion"].max()), n_esperadas,
        )
        return [], set()

    filas = previo.to_dict("records")
    return filas, set(previo["configuracion"].astype(int))


def _guardar_checkpoint(ruta: Optional[Path], filas: list[dict]) -> None:
    """Escribe el avance de forma atómica, para que una interrupción no deje un CSV a medias."""
    if ruta is None:
        return
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".parcial")
    pd.DataFrame(filas).to_csv(temporal, index=False)
    temporal.replace(ruta)


def elegir(tabla: pd.DataFrame, espacio: Sequence[str],
           penalizar_inestabilidad: bool = True) -> dict:
    """Selecciona una configuración de la tabla, con un criterio explícito.

    Por defecto no se coge la de mayor media sino la mejor por **media menos desviación** entre
    pliegues. El motivo es el sesgo del ganador: la configuración con la media más alta suele
    serlo en parte por haber tenido suerte en un pliegue, y penalizar la dispersión favorece a
    la que funciona en los tres años en vez de a la que brilla en uno.

    Args:
        tabla: Salida de `buscar`.
        espacio: Nombres de los hiperparámetros a extraer.
        penalizar_inestabilidad: Si es falso, se coge la de mayor media a secas.
    """
    if tabla.empty:
        raise ValueError("La tabla de búsqueda está vacía.")

    if penalizar_inestabilidad:
        puntuacion = tabla["recall_medio"] - tabla["recall_desv"]
    else:
        puntuacion = tabla["recall_medio"]

    indice = puntuacion.idxmax()
    # Se accede columna a columna con `.at` y no con `.loc[fila]`: extraer una fila entera de un
    # DataFrame de tipos mixtos la promociona a float64, y `num_leaves` acabaría valiendo 31.0.
    # Un entero convertido en decimal ensucia el YAML de hiperparámetros y algunos estimadores
    # lo rechazan directamente.
    parametros = {p: tabla.at[indice, p] for p in espacio if p in tabla.columns}
    # RandomizedSearchCV devuelve numpy scalars; los estimadores y el YAML quieren tipos nativos.
    return {clave: (valor.item() if hasattr(valor, "item") else valor)
            for clave, valor in parametros.items()}
