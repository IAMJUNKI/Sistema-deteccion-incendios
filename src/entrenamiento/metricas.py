"""Métricas de evaluación para clasificación con desbalanceo extremo.

Sustituye a `src/models/metrics.py`, que se conserva intacto para poder reproducir los
resultados anteriores. Las diferencias respecto a aquel módulo están documentadas en
`MODEL_DOCUMENTATION.md` §3.3, y son tres:

1. **PR-AUC con `average_precision_score`** en lugar de `auc(recall, precision)`. El segundo
   interpola linealmente la curva Precision-Recall, lo que introduce un sesgo optimista.
2. **Lift sobre la prevalencia**, porque un PR-AUC de 7·10⁻⁵ es ilegible sin normalizar: parece
   un fracaso y en realidad puede ser ocho veces mejor que el azar.
3. **Intervalos de confianza por bootstrap**. Con ~100-200 incendios en el conjunto de test, un
   punto porcentual de recall son dos incendios: sin intervalos no se pueden ordenar modelos.

Se añade además `recall_at_top_k_daily`, que responde a la pregunta operativa real: si cada día
se patrulla el 1 % del territorio, ¿cuántos incendios se cazan?
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def recall_at_fpr(
    y_true: np.ndarray, y_prob: np.ndarray, fpr_max: float = 0.05
) -> tuple[float, float, float]:
    """Recall alcanzable manteniendo la tasa de falsos positivos en `fpr_max`.

    El umbral se fija en el cuantil `1 - fpr_max` de las puntuaciones de la clase negativa, lo
    que evita ordenar los 11 millones de filas que exigiría `roc_curve`.

    **Tratamiento de empates.** A esta escala hay millones de filas con la misma puntuación: un
    modelo de árboles asigna el mismo valor a todas las que caen en la misma hoja, y la
    regresión isotónica lo agrava porque es una función escalonada. Si el umbral cae sobre una
    meseta, la comparación `>=` se lleva la meseta entera y el FPR real se dispara muy por
    encima del objetivo, inflando el recall de forma espectacular y falsa. Cuando se detecta ese
    caso se pasa a comparación estricta, que es la opción conservadora. En ambos casos se
    devuelve el FPR **realmente alcanzado**, para que el punto de operación sea auditable.

    Args:
        y_true: Etiquetas reales (0/1).
        y_prob: Puntuación o probabilidad predicha.
        fpr_max: Tasa de falsos positivos objetivo.

    Returns:
        Tupla (recall, umbral, fpr_alcanzado).
    """
    neg = y_prob[y_true == 0]
    pos = y_prob[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return 0.0, float("inf"), 0.0

    umbral = umbral_at_fpr(neg, fpr_max)
    return float((pos >= umbral).mean()), umbral, float((neg >= umbral).mean())


def umbral_at_fpr(negativos: np.ndarray, fpr_max: float = 0.05) -> float:
    """Umbral más bajo que respeta el presupuesto de falsos positivos.

    Devuelve el menor valor `t` tal que `(negativos >= t).mean() <= fpr_max`. Es decir: la
    alerta más generosa que cabe dentro del presupuesto, nunca una que se lo salte.

    **Por qué no vale un cuantil.** `np.quantile` interpola entre observaciones y no garantiza
    nada sobre el recuento resultante: puede devolver un punto para el que la tasa real quede
    marginalmente por encima del objetivo. Y sobre todo, no resuelve los empates. A esta escala
    hay millones de filas con la misma puntuación —un árbol asigna el mismo valor a toda una
    hoja, y la calibración isotónica lo agrava porque es escalonada—, así que un umbral que
    caiga sobre una meseta se lleva la meseta entera y dispara la tasa muy por encima de lo
    pedido, inflando el recall de forma espectacular y falsa.

    Aquí se recorren los valores distintos de mayor a menor acumulando cuántos negativos quedan
    por encima, y se elige el primero que cabe en el presupuesto. El resultado cumple la
    restricción por construcción, con empates o sin ellos, y siempre con comparación `>=`.
    """
    if len(negativos) == 0:
        return float("inf")

    permitidos = int(np.floor(len(negativos) * fpr_max))
    if permitidos == 0:
        # Ni un solo falso positivo cabe en el presupuesto: no se puede alertar nada.
        return float(np.nextafter(negativos.max(), np.inf))

    valores, cuentas = np.unique(negativos, return_counts=True)   # ascendente
    # Negativos con valor mayor o igual que cada valor distinto.
    acumulado = np.cumsum(cuentas[::-1])[::-1]
    cabe = np.flatnonzero(acumulado <= permitidos)
    if len(cabe) == 0:
        # Ni el valor más alto cabe: todo está empatado en el máximo.
        return float(np.nextafter(valores[-1], np.inf))
    return float(valores[cabe[0]])


def recall_at_top_k_daily(
    y_true: np.ndarray, y_prob: np.ndarray, day_index: np.ndarray, k: float = 0.01
) -> float:
    """Recall alertando cada día únicamente el `k` por uno de celdas con más riesgo.

    Es la métrica más honesta operativamente. Un FPR del 5 % sobre Galicia son 1.535 km², que
    ninguna brigada patrulla; el 1 % son 307 celdas, que sí. Al aplicarse día a día, refleja que
    la decisión de despliegue se toma cada mañana sobre el mapa de ese día.

    Args:
        y_true: Etiquetas reales (0/1).
        y_prob: Puntuación predicha.
        day_index: Identificador entero del día de cada fila.
        k: Fracción de celdas que se alertan cada día.

    Returns:
        Fracción de incendios que caen dentro de la selección diaria.
    """
    return float(capturas_top_k_daily(y_true, y_prob, day_index, k).mean() or 0.0)


def capturas_top_k_daily(
    y_true: np.ndarray, y_prob: np.ndarray, day_index: np.ndarray, k: float = 0.01
) -> np.ndarray:
    """Indicador por incendio de si cayó dentro del `k` por uno diario de mayor riesgo.

    Devolver el vector y no solo su media permite construirle un intervalo de confianza por
    bootstrap, igual que se hace con el recall a FPR fijo. Sin intervalo, la métrica operativa
    del proyecto no se podía usar para decidir entre dos modelos: una diferencia de dos puntos
    sobre 1.659 incendios puede ser ruido y no había forma de saberlo.
    """
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return np.zeros(0, dtype=bool)

    orden = np.argsort(day_index, kind="stable")
    dias = day_index[orden]
    probs = y_prob[orden]
    reales = y_true[orden]

    cortes = np.flatnonzero(np.diff(dias)) + 1
    capturas: list[np.ndarray] = []
    for ini, fin in zip(
        np.concatenate([[0], cortes]), np.concatenate([cortes, [len(dias)]])
    ):
        p_dia = probs[ini:fin]
        y_dia = reales[ini:fin]
        if y_dia.sum() == 0:
            continue
        n_alerta = max(1, int(round(len(p_dia) * k)))
        umbral = np.partition(p_dia, -n_alerta)[-n_alerta]
        capturas.append((p_dia[y_dia == 1] >= umbral))

    return np.concatenate(capturas) if capturas else np.zeros(0, dtype=bool)


def bootstrap_top_k_daily_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    day_index: np.ndarray,
    k: float = 0.01,
    n_boot: int = 1000,
    alpha: float = 0.10,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalo de confianza del recall diario, remuestreando los incendios capturados."""
    capturas = capturas_top_k_daily(y_true, y_prob, day_index, k).astype(np.float64)
    if len(capturas) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    muestras = rng.choice(capturas, size=(n_boot, len(capturas)), replace=True).mean(axis=1)
    return (
        float(np.percentile(muestras, 100 * alpha / 2)),
        float(np.percentile(muestras, 100 * (1 - alpha / 2))),
    )


def bootstrap_recall_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_max: float = 0.05,
    n_boot: int = 1000,
    alpha: float = 0.10,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalo de confianza del recall remuestreando los incendios.

    Se remuestrean solo los positivos porque son los escasos: con ~100 fuegos son ellos, y no
    los millones de negativos, los que determinan la incertidumbre de la métrica. El umbral se
    mantiene fijo (calculado sobre los negativos, que no se remuestrean).

    Args:
        y_true: Etiquetas reales.
        y_prob: Puntuación predicha.
        fpr_max: Tasa de falsos positivos que fija el umbral.
        n_boot: Número de réplicas bootstrap.
        alpha: Nivel de significación (0.10 → intervalo del 90 %).
        seed: Semilla.

    Returns:
        Tupla (límite inferior, límite superior).
    """
    idx_pos = np.flatnonzero(y_true == 1)
    if len(idx_pos) == 0:
        return 0.0, 0.0

    # El umbral se mantiene fijo: procede de los negativos, que son abundantes y no se
    # remuestrean. La incertidumbre que interesa es la de los ~1.600 incendios.
    aciertos = aciertos_at_fpr(y_true, y_prob, fpr_max).astype(np.float64)

    rng = np.random.default_rng(seed)
    muestras = rng.choice(aciertos, size=(n_boot, len(aciertos)), replace=True).mean(axis=1)
    return (
        float(np.percentile(muestras, 100 * alpha / 2)),
        float(np.percentile(muestras, 100 * (1 - alpha / 2))),
    )


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    day_index: Optional[np.ndarray] = None,
    fpr_max: float = 0.05,
    top_k: float = 0.01,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Calcula el cuadro completo de métricas del proyecto.

    Separa deliberadamente dos familias de métricas que se calculan sobre entradas distintas:

    - **Métricas de ordenación** (PR-AUC, ROC-AUC, recalls): sobre la puntuación **cruda** del
      modelo. La calibración es una transformación monótona, así que en teoría no altera el
      orden; en la práctica la isotónica es escalonada y crea millones de empates que sí
      distorsionan cualquier métrica basada en umbral.
    - **Métricas de calibración** (Brier): sobre la probabilidad **calibrada**, que es la única
      que tiene significado como probabilidad. Antes de recalibrar, el Brier mide el sesgo del
      submuestreo y no la calidad del modelo, así que se omite si no se aporta.

    Args:
        y_true: Etiquetas reales (0/1).
        y_score: Puntuación cruda del modelo, tal como sale de `predict_proba`.
        y_prob_calibrated: Probabilidad recalibrada. Opcional.
        day_index: Identificador entero del día, para el recall diario. Opcional.
        fpr_max: Tasa de falsos positivos objetivo.
        top_k: Fracción de celdas alertadas cada día.
        n_boot: Réplicas bootstrap.
        seed: Semilla.

    Returns:
        Diccionario de métricas.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int(y_true.sum())
    prevalencia = float(n_pos / len(y_true)) if len(y_true) else 0.0

    ap = float(average_precision_score(y_true, y_score)) if n_pos else float("nan")
    rec_fpr, umbral, fpr_real = recall_at_fpr(y_true, y_score, fpr_max)
    lo, hi = bootstrap_recall_ci(y_true, y_score, fpr_max, n_boot, seed=seed)

    resultado = {
        "n_rows": int(len(y_true)),
        "n_positives": n_pos,
        "prevalence": prevalencia,
        "pr_auc": ap,
        "lift_vs_azar": float(ap / prevalencia) if prevalencia > 0 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if n_pos else float("nan"),
        f"recall_at_fpr{int(fpr_max * 100)}": rec_fpr,
        "recall_ci90_low": lo,
        "recall_ci90_high": hi,
        "threshold_at_fpr": umbral,
        "fpr_achieved": fpr_real,
    }

    if day_index is not None:
        clave = f"recall_at_top{top_k:.0%}_daily"
        resultado[clave] = recall_at_top_k_daily(y_true, y_score, day_index, top_k)
        bajo, alto = bootstrap_top_k_daily_ci(
            y_true, y_score, day_index, top_k, n_boot, seed=seed
        )
        resultado[f"{clave}_ci90_low"] = bajo
        resultado[f"{clave}_ci90_high"] = alto

    resultado["brier_score"] = (
        float(brier_score_loss(y_true, np.asarray(y_prob_calibrated, dtype=np.float64)))
        if y_prob_calibrated is not None
        else float("nan")
    )
    return resultado


def aciertos_at_fpr(
    y_true: np.ndarray, y_prob: np.ndarray, fpr_max: float = 0.05
) -> np.ndarray:
    """Indicador por incendio de si el modelo lo detecta al umbral de `fpr_max`.

    Replica el mismo tratamiento de empates que `recall_at_fpr`, de modo que la media de este
    vector coincide exactamente con el recall que se reporta.
    """
    neg = y_prob[y_true == 0]
    pos = y_prob[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return np.zeros(len(pos), dtype=bool)
    return pos >= umbral_at_fpr(neg, fpr_max)


def comparar_pareado(
    aciertos_a: np.ndarray,
    aciertos_b: np.ndarray,
    nombre_a: str = "A",
    nombre_b: str = "B",
    n_boot: int = 2000,
    alpha: float = 0.10,
    seed: int = 42,
) -> dict:
    """Compara dos modelos sobre **los mismos incendios**, que es la comparación correcta.

    ## Por qué no basta con mirar si los intervalos se solapan

    Los intervalos de confianza que acompañan a cada modelo son *marginales*: describen cuánto
    variaría su recall si repitiéramos el experimento con otra muestra de incendios. Cuando dos
    modelos se evalúan sobre **el mismo** conjunto de incendios, buena parte de esa variabilidad
    es común a los dos —un año con incendios difíciles lo es para ambos— y al comparar los
    intervalos por separado esa varianza compartida se cuenta dos veces.

    El resultado es un test **conservador de más**: declara «no concluyente» diferencias que sí
    lo son. Es el motivo por el que este proyecto llevaba meses sin poder afirmar que un modelo
    fuera mejor que otro.

    Aquí se remuestrean los incendios **una sola vez por réplica** y se evalúan los dos modelos
    sobre esa misma remuestra. Lo que se acumula es la *diferencia*, cuya varianza ya no incluye
    la parte común. Si el intervalo de la diferencia no contiene el cero, la diferencia es real.

    Se acompaña de los conteos discordantes de McNemar, que son la lectura más directa que
    existe: cuántos incendios caza uno y se le escapan al otro.

    Args:
        aciertos_a: Vector booleano por incendio para el modelo A.
        aciertos_b: Ídem para el modelo B, **sobre los mismos incendios y en el mismo orden**.
        nombre_a: Etiqueta del modelo A.
        nombre_b: Etiqueta del modelo B.
        n_boot: Réplicas bootstrap.
        alpha: Nivel de significación (0,10 -> intervalo del 90 %).
        seed: Semilla.

    Returns:
        Diccionario con la diferencia, su intervalo, los conteos discordantes y el veredicto.

    Raises:
        ValueError: Si los vectores no tienen la misma longitud, que significaría que no se
            están comparando los mismos incendios.
    """
    a = np.asarray(aciertos_a, dtype=bool)
    b = np.asarray(aciertos_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(
            f"Los vectores de aciertos deben describir los mismos incendios: "
            f"{a.shape} frente a {b.shape}."
        )
    if len(a) == 0:
        raise ValueError("No hay incendios sobre los que comparar.")

    solo_a = int((a & ~b).sum())
    solo_b = int((b & ~a).sum())
    discordantes = solo_a + solo_b

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(n_boot, len(a)))
    diferencias = a[indices].mean(axis=1) - b[indices].mean(axis=1)

    bajo = float(np.percentile(diferencias, 100 * alpha / 2))
    alto = float(np.percentile(diferencias, 100 * (1 - alpha / 2)))
    concluyente = bool(bajo > 0 or alto < 0)

    # McNemar exacto sobre los pares discordantes: bajo la hipótesis nula, cada discordancia
    # es una moneda justa. Es un test independiente del bootstrap, y que ambos coincidan da
    # confianza en el veredicto.
    if discordantes:
        try:
            from scipy.stats import binomtest

            p_valor = float(binomtest(solo_a, discordantes, 0.5).pvalue)
        except ImportError:  # pragma: no cover
            p_valor = float("nan")
    else:
        p_valor = 1.0

    return {
        "modelo_a": nombre_a,
        "modelo_b": nombre_b,
        "n_incendios": int(len(a)),
        "recall_a": float(a.mean()),
        "recall_b": float(b.mean()),
        "diferencia": float(a.mean() - b.mean()),
        "dif_ci90_low": bajo,
        "dif_ci90_high": alto,
        "concluyente": concluyente,
        "ambos": int((a & b).sum()),
        f"solo_{nombre_a}": solo_a,
        f"solo_{nombre_b}": solo_b,
        "ninguno": int((~a & ~b).sum()),
        "mcnemar_p": p_valor,
        "veredicto": (
            f"{nombre_a} mejor" if concluyente and bajo > 0
            else f"{nombre_b} mejor" if concluyente
            else "no concluyente"
        ),
    }


def comparar_modelos(
    y_true: np.ndarray,
    puntuaciones: dict[str, np.ndarray],
    day_index: Optional[np.ndarray] = None,
    fpr_max: float = 0.05,
    top_k: float = 0.01,
    n_boot: int = 2000,
    alpha: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara todos los pares de modelos con el test pareado, en las dos métricas.

    Args:
        y_true: Etiquetas reales, comunes a todos los modelos.
        puntuaciones: Diccionario nombre -> puntuación cruda sobre las mismas filas.
        day_index: Índice de día. Si se aporta se compara también el recall diario.

    Returns:
        Tabla con una fila por par de modelos y métrica, ordenada por métrica y diferencia.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    nombres = list(puntuaciones)

    aciertos: dict[str, dict[str, np.ndarray]] = {
        f"recall@fpr{int(fpr_max * 100)}": {
            n: aciertos_at_fpr(y_true, np.asarray(p), fpr_max) for n, p in puntuaciones.items()
        }
    }
    if day_index is not None:
        aciertos[f"recall@top{top_k:.0%}/día"] = {
            n: capturas_top_k_daily(y_true, np.asarray(p), day_index, top_k)
            for n, p in puntuaciones.items()
        }

    filas = []
    for metrica, por_modelo in aciertos.items():
        for i, a in enumerate(nombres):
            for b in nombres[i + 1:]:
                resultado = comparar_pareado(
                    por_modelo[a], por_modelo[b], a, b, n_boot, alpha, seed
                )
                filas.append({
                    "metrica": metrica,
                    **{k: v for k, v in resultado.items() if not k.startswith("solo_")},
                    "solo_a": resultado[f"solo_{a}"],
                    "solo_b": resultado[f"solo_{b}"],
                })

    return pd.DataFrame(filas).sort_values(
        ["metrica", "diferencia"], ascending=[True, False]
    ).reset_index(drop=True)


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 12
) -> pd.DataFrame:
    """Tabla de fiabilidad: probabilidad predicha frente a frecuencia observada.

    Los cortes se hacen por cuantiles y no en tramos iguales porque a esta prevalencia casi
    todas las predicciones se acumulan cerca de cero y los tramos uniformes quedarían vacíos.

    Args:
        y_true: Etiquetas reales.
        y_prob: Probabilidad predicha (recalibrada).
        n_bins: Número de tramos.

    Returns:
        DataFrame con la predicción media, la frecuencia observada y el tamaño de cada tramo.
    """
    bordes = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(bordes) < 3:
        return pd.DataFrame(columns=["bin", "pred_media", "obs_frecuencia", "n"])
    tramo = np.clip(np.digitize(y_prob, bordes[1:-1]), 0, len(bordes) - 2)
    df = pd.DataFrame({"bin": tramo, "p": y_prob, "y": y_true})
    return (
        df.groupby("bin")
        .agg(pred_media=("p", "mean"), obs_frecuencia=("y", "mean"), n=("y", "size"))
        .reset_index()
    )


def summarize(resultados: list[dict]) -> pd.DataFrame:
    """Ordena una lista de resultados en una tabla comparable, de mayor a menor recall."""
    df = pd.DataFrame(resultados)
    col_recall = next((c for c in df.columns if c.startswith("recall_at_fpr")), None)
    return df.sort_values(col_recall, ascending=False) if col_recall else df
