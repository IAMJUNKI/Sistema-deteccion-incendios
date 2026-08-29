"""Selección de variables con cuatro criterios independientes.

En el pipeline sobre FIRMS no había selección: se le tiraban al modelo las 22 variables
disponibles y se confiaba en que los árboles ignorasen las inútiles. Con 50 predictores publicados
en el contrato actual, eso deja de ser razonable, por tres
motivos concretos: variables redundantes reparten la importancia entre sí y la vuelven
ilegible; cada variable de más es una oportunidad de sobreajuste con solo 6.189 positivos; y un
tribunal preguntará por qué está cada una.

Los cuatro criterios responden preguntas distintas y **ninguno decide por sí solo**:

| Criterio | Pregunta | Coste |
|---|---|---|
| `senal_univariante` | ¿Discrimina algo esta variable por sí sola? | segundos |
| `redundancia` | ¿Hay pares que dicen lo mismo? | segundos |
| `importancia_permutacion` | ¿Cuánto se pierde si la estropeo? | minutos |
| `ablacion_grupos` | ¿Aporta este bloque temático entero? | un entrenamiento por grupo |

La permutación es el criterio con más autoridad porque se mide **sobre validación y con el
modelo ya entrenado**: captura la contribución real de la variable, incluidas las interacciones,
y no su correlación marginal con el target. Pero es también el más caro, así que la señal
univariante y la redundancia sirven para llegar a la permutación con una lista más corta.

Nota sobre la métrica de permutación: se usa ROC-AUC y no PR-AUC. La permutación exige
submuestrear para ser asequible, y PR-AUC depende de la prevalencia, así que sobre una
submuestra daría un número que no describe ninguna población real. ROC-AUC es invariante a la
prevalencia y sí se puede comparar entre submuestra y año completo.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def submuestra_estratificada(
    marco: pd.DataFrame, target: str, n_maximo: int = 1_000_000, semilla: int = 42
) -> pd.DataFrame:
    """Reduce un marco conservando **todos** los positivos y muestreando negativos.

    Los positivos son el recurso escaso: tirarlos al azar destruiría la precisión de cualquier
    métrica. Los negativos sobran.
    """
    if len(marco) <= n_maximo:
        return marco
    positivos = marco.index[marco[target] == 1]
    negativos = marco.index[marco[target] == 0]
    cupo = max(n_maximo - len(positivos), 1)
    rng = np.random.default_rng(semilla)
    elegidos = rng.choice(negativos, size=min(cupo, len(negativos)), replace=False)
    return marco.loc[np.concatenate([positivos.to_numpy(), elegidos])]


def senal_univariante(
    marco: pd.DataFrame, variables: Sequence[str], target: str
) -> pd.DataFrame:
    """ROC-AUC de cada variable usada en solitario como puntuación.

    Un valor de 0,5 significa que la variable, por sí sola, no distingue nada. Se devuelve
    `|AUC - 0,5|` como fuerza de la señal, porque una variable que ordena al revés discrimina
    igual de bien: el modelo solo tiene que invertirle el signo.
    """
    y = marco[target].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Se necesitan ambas clases para medir señal univariante.")

    filas = []
    for variable in variables:
        columna = marco[variable].to_numpy(dtype=np.float64)
        validos = np.isfinite(columna)
        if validos.sum() < 2 or len(np.unique(y[validos])) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y[validos], columna[validos])
        filas.append({
            "variable": variable,
            "auc": auc,
            "fuerza": abs(auc - 0.5) if np.isfinite(auc) else np.nan,
            "nulos_pct": float((~validos).mean() * 100),
        })
    return (
        pd.DataFrame(filas)
        .sort_values("fuerza", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def redundancia(
    marco: pd.DataFrame, variables: Sequence[str], umbral: float = 0.95
) -> pd.DataFrame:
    """Pares de variables cuya correlación absoluta supera el umbral.

    Se usa Spearman y no Pearson: capta relaciones monótonas aunque no sean lineales, que es lo
    que importa para modelos de árboles, cuyos cortes solo dependen del orden.
    """
    matriz = marco[list(variables)].corr(method="spearman").abs()
    superior = np.triu(np.ones(matriz.shape, dtype=bool), k=1)
    filas, columnas = np.where(superior & (matriz.to_numpy() >= umbral))
    return (
        pd.DataFrame({
            "variable_a": matriz.index[filas],
            "variable_b": matriz.columns[columnas],
            "correlacion": matriz.to_numpy()[filas, columnas],
        })
        .sort_values("correlacion", ascending=False)
        .reset_index(drop=True)
    )


def podar_redundantes(
    variables: Sequence[str], pares: pd.DataFrame, senal: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    """De cada par redundante elimina el miembro con menos señal univariante.

    Se procesan los pares de mayor a menor correlación y se salta el par si alguno de los dos ya
    se ha eliminado, para no encadenar borrados y quedarse sin ninguna variable de una familia.

    Returns:
        Tupla con la lista superviviente y el registro de qué se quitó y por qué.
    """
    fuerza = senal.set_index("variable")["fuerza"].to_dict()
    supervivientes = list(variables)
    descartadas: list[dict] = []

    for _, par in pares.iterrows():
        a, b = par["variable_a"], par["variable_b"]
        if a not in supervivientes or b not in supervivientes:
            continue
        perdedora = a if fuerza.get(a, 0.0) < fuerza.get(b, 0.0) else b
        ganadora = b if perdedora == a else a
        supervivientes.remove(perdedora)
        descartadas.append({
            "descartada": perdedora,
            "a_favor_de": ganadora,
            "correlacion": par["correlacion"],
            "fuerza_descartada": fuerza.get(perdedora, np.nan),
            "fuerza_conservada": fuerza.get(ganadora, np.nan),
        })

    logger.info("Poda por redundancia: %s variables -> %s", len(variables), len(supervivientes))
    return supervivientes, pd.DataFrame(descartadas)


def importancia_permutacion(
    modelo: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    variables: Sequence[str],
    repeticiones: int = 3,
    semilla: int = 42,
) -> pd.DataFrame:
    """Caída de ROC-AUC al barajar cada variable, medida sobre validación.

    Barajar una columna destruye su relación con el target conservando su distribución. Si el
    ROC-AUC no baja, la variable no estaba aportando nada que el modelo usara.

    Es la única medida de importancia que no se deja engañar por la redundancia de forma
    silenciosa: dos variables que dicen lo mismo se cubren mutuamente y ambas salen con
    importancia baja, lo cual es informativo —conviene quitar una— en vez de engañoso.
    """
    rng = np.random.default_rng(semilla)
    base = roc_auc_score(y, modelo.predict_proba(X)[:, 1])
    logger.info("ROC-AUC de referencia para la permutación: %.4f", base)

    filas = []
    for variable in variables:
        original = X[variable].to_numpy().copy()
        caidas = []
        for _ in range(repeticiones):
            X[variable] = rng.permutation(original)
            caidas.append(base - roc_auc_score(y, modelo.predict_proba(X)[:, 1]))
        X[variable] = original
        filas.append({
            "variable": variable,
            "caida_media": float(np.mean(caidas)),
            "caida_desv": float(np.std(caidas)),
        })

    return (
        pd.DataFrame(filas)
        .assign(auc_base=base)
        .sort_values("caida_media", ascending=False)
        .reset_index(drop=True)
    )


def ablacion_grupos(
    grupos: dict[str, list[str]],
    variables: Sequence[str],
    entrenar_y_evaluar: Callable[[list[str]], dict],
    metrica: str = "roc_auc",
) -> pd.DataFrame:
    """Entrena una vez con todo y una vez sin cada grupo temático completo.

    Responde a la pregunta que se hace en una defensa: «¿y la topografía, para qué la metéis?».
    La respuesta deja de ser una intuición y pasa a ser un número.

    Args:
        grupos: Diccionario grupo -> variables, tal como lo expone el contrato.
        variables: Conjunto de partida.
        entrenar_y_evaluar: Función que recibe una lista de variables y devuelve un diccionario
            de métricas. La inyecta el orquestador para que este módulo no dependa de él.
        metrica: Clave del diccionario que se compara.
    """
    variables = list(variables)
    referencia = entrenar_y_evaluar(variables)
    filas = [{
        "grupo_retirado": "(ninguno)",
        "n_variables": len(variables),
        metrica: referencia[metrica],
        "delta": 0.0,
    }]

    for grupo, del_grupo in sorted(grupos.items()):
        restantes = [v for v in variables if v not in set(del_grupo)]
        if not restantes or len(restantes) == len(variables):
            continue
        resultado = entrenar_y_evaluar(restantes)
        filas.append({
            "grupo_retirado": grupo,
            "n_variables": len(restantes),
            metrica: resultado[metrica],
            "delta": resultado[metrica] - referencia[metrica],
        })
        logger.info("Sin %s (%s variables): %s = %.4f (delta %+.4f)",
                    grupo, len(restantes), metrica, resultado[metrica], filas[-1]["delta"])

    return pd.DataFrame(filas).sort_values("delta").reset_index(drop=True)


#: Umbral por encima del cual dos variables se consideran la misma cosa y una sobra sin
#: discusión posible. A 0,99 de Spearman no hay ningún corte de árbol que las distinga.
UMBRAL_DUPLICADO = 0.99

#: Zona gris: fuertemente correlacionadas pero no idénticas. Aquí NO se decide solo; se ofrece.
UMBRAL_DISCUTIBLE = 0.90


def poda_automatica(
    marco: pd.DataFrame,
    variables: Sequence[str],
    target: str,
    umbral_duplicado: float = UMBRAL_DUPLICADO,
    umbral_nulos: float = 50.0,
    minimo_valores_distintos: int = 2,
    marco_control: Optional[pd.DataFrame] = None,
) -> tuple[list[str], pd.DataFrame]:
    """Elimina únicamente lo que es demostrablemente prescindible, dejando constancia.

    La distinción con el resto del módulo es deliberada: aquí solo entra lo que se puede
    justificar con una medida y no admite opinión —una constante no informa, y dos variables
    con Spearman 0,999 no las distingue ningún corte de árbol—. Todo lo demás, aunque parezca
    obvio, se deja para que lo decida quien ejecute, con la curva de compromiso delante.

    Cada eliminación se devuelve con su motivo y la evidencia numérica que la sostiene, porque
    en la memoria hay que poder responder a «¿por qué quitasteis esta variable?» con un dato.

    Returns:
        Tupla (supervivientes, informe de descartes con motivo y evidencia).
    """
    variables = list(variables)
    descartes: list[dict] = []
    vivas = list(variables)

    # ── Constantes: no pueden separar nada ────────────────────────────────────────────────
    # Se exige que la variable sea constante TAMBIÉN en el marco de control cuando se aporta.
    # Una variable constante en una muestra y no en otra no es constante: es un síntoma de que
    # algo ha ido mal al construirla, y descartarla en silencio ocultaría el problema en vez de
    # resolverlo. Ha pasado una vez con `vpd_x_forestal`, sin poder reproducirlo después.
    for variable in list(vivas):
        distintos = marco[variable].nunique(dropna=True)
        if distintos >= minimo_valores_distintos:
            continue
        if marco_control is not None and variable in marco_control.columns:
            control = marco_control[variable].nunique(dropna=True)
            if control >= minimo_valores_distintos:
                logger.error(
                    "%s es constante en el conjunto principal (%s valores) pero NO en el de "
                    "control (%s valores). No se descarta: revisa cómo se construye.",
                    variable, distintos, control,
                )
                continue
        vivas.remove(variable)
        descartes.append({
            "variable": variable, "motivo": "constante",
            "evidencia": f"{distintos} valor(es) distinto(s)", "valor": float(distintos),
        })

    # ── Nulos por encima del umbral: imputarlos sería inventar la mayoría de la columna ───
    for variable in list(vivas):
        pct = float(marco[variable].isna().mean() * 100)
        if pct > umbral_nulos:
            vivas.remove(variable)
            descartes.append({
                "variable": variable, "motivo": "exceso de nulos",
                "evidencia": f"{pct:.1f} % nulos", "valor": pct,
            })

    # ── Duplicados: de cada par idéntico sobrevive el de más señal univariante ────────────
    if vivas:
        senal = senal_univariante(marco, vivas, target)
        fuerza = senal.set_index("variable")["fuerza"].to_dict()
        pares = redundancia(marco, vivas, umbral=umbral_duplicado)
        for _, par in pares.iterrows():
            a, b = par["variable_a"], par["variable_b"]
            if a not in vivas or b not in vivas:
                continue
            perdedora = a if fuerza.get(a, 0.0) < fuerza.get(b, 0.0) else b
            ganadora = b if perdedora == a else a
            vivas.remove(perdedora)
            descartes.append({
                "variable": perdedora, "motivo": "duplicada",
                "evidencia": f"|Spearman| = {par['correlacion']:.4f} con {ganadora}",
                "valor": float(par["correlacion"]),
            })

    logger.info("Poda automática: %s variables -> %s (%s descartes)",
                len(variables), len(vivas), len(descartes))
    return vivas, pd.DataFrame(descartes)


def clusters_correlacion(
    marco: pd.DataFrame, variables: Sequence[str], umbral: float = UMBRAL_DISCUTIBLE
) -> pd.DataFrame:
    """Agrupa variables en familias que se solapan por encima del umbral.

    Se construyen componentes conexas del grafo «están correlacionadas»: si A se parece a B y B
    a C, las tres van al mismo grupo aunque A y C no se parezcan directamente. Es la forma
    honesta de presentar la redundancia, porque enseña la familia completa en vez de pares
    sueltos y deja ver que quitar una de cada familia es una decisión, no una obviedad.
    """
    variables = list(variables)
    matriz = marco[variables].corr(method="spearman").abs().to_numpy()
    padre = list(range(len(variables)))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            if matriz[i, j] >= umbral:
                padre[raiz(i)] = raiz(j)

    familias: dict[int, list[str]] = {}
    for i, variable in enumerate(variables):
        familias.setdefault(raiz(i), []).append(variable)

    filas = []
    for indice, miembros in enumerate(sorted(familias.values(), key=len, reverse=True)):
        if len(miembros) < 2:
            continue
        posiciones = [variables.index(m) for m in miembros]
        interna = matriz[np.ix_(posiciones, posiciones)]
        filas.append({
            "familia": indice,
            "n": len(miembros),
            "correlacion_min": float(interna[np.triu_indices(len(miembros), k=1)].min()),
            "variables": ", ".join(sorted(miembros)),
        })
    return pd.DataFrame(filas)


def _resolver_ic(resultado: dict, metrica: str) -> tuple[str, str]:
    """Localiza las claves del intervalo de confianza que corresponden a esta métrica.

    Raises:
        KeyError: Si la métrica no tiene intervalo. Es preferible fallar a dibujar el de otra.
    """
    propio = (f"{metrica}_ci90_low", f"{metrica}_ci90_high")
    if all(clave in resultado for clave in propio):
        return propio
    if metrica.startswith("recall_at_fpr") and "recall_ci90_low" in resultado:
        return ("recall_ci90_low", "recall_ci90_high")
    disponibles = sorted(k for k in resultado if k.endswith(("_ci90_low", "ci90_low")))
    raise KeyError(
        f"La métrica {metrica!r} no tiene intervalo de confianza. "
        f"Claves de intervalo disponibles: {disponibles}"
    )


def curva_compromiso(
    orden: Sequence[str],
    entrenar_y_evaluar: Callable[[list[str]], dict],
    escalones: Sequence[int],
    metrica: str = "recall_at_fpr5",
    ic: Optional[tuple[str, str]] = None,
) -> pd.DataFrame:
    """Entrena con las `k` mejores variables para cada `k` y mide qué se gana y qué se pierde.

    Es el corazón del análisis: convierte «¿cuántas variables uso?» de una cuestión de gusto en
    una curva con intervalos de confianza. Con ella se puede afirmar que 20 variables rinden
    igual que 54 —o que no— en vez de suponerlo.

    Args:
        orden: Variables ordenadas de más a menos importante.
        entrenar_y_evaluar: Función que recibe una lista y devuelve un diccionario de métricas.
        escalones: Valores de `k` a probar.
        metrica: Métrica principal de la curva.
        ic: Claves de los extremos del intervalo. Si es `None` se resuelven a partir del nombre
            de la métrica, que es lo correcto: fijarlas antes hacía que al cambiar de métrica se
            dibujara el intervalo de otra, y una curva con el punto en 0,06 y el intervalo en
            0,36 no avisa de que algo va mal, simplemente miente.
    """
    orden = list(orden)
    escalones = sorted({min(k, len(orden)) for k in escalones if k > 0})
    filas = []
    for k in escalones:
        subconjunto = orden[:k]
        resultado = entrenar_y_evaluar(subconjunto)
        if ic is None:
            ic = _resolver_ic(resultado, metrica)
        # Se guardan TODAS las métricas, no solo la que gobierna la selección: dos métricas
        # pueden ordenar los conjuntos de forma distinta —el recall global y el recall diario
        # lo hacen— y sin registrarlas todas habría que repetir la escalera entera para
        # descubrirlo.
        filas.append({
            "k": k,
            **{f"metrica_{clave}": valor for clave, valor in resultado.items()},
            metrica: resultado[metrica],
            "ic_bajo": resultado.get(ic[0], np.nan),
            "ic_alto": resultado.get(ic[1], np.nan),
            "roc_auc": resultado.get("roc_auc", np.nan),
            "pr_auc": resultado.get("pr_auc", np.nan),
            "variables": ", ".join(subconjunto),
        })
        logger.info("k=%s: %s = %.4f (IC90 %.4f – %.4f)", k, metrica, filas[-1][metrica],
                    filas[-1]["ic_bajo"], filas[-1]["ic_alto"])
    return pd.DataFrame(filas)


def recomendar(curva: pd.DataFrame, metrica: str = "recall_at_fpr5") -> dict:
    """Propone el conjunto más pequeño que no es peor que el mejor de forma concluyente.

    La regla es explícita y auditable: se localiza el `k` con mejor métrica y se acepta el `k`
    más pequeño cuyo límite superior del intervalo alcance el límite inferior del mejor. Si los
    intervalos se solapan, la diferencia no es concluyente, y entre dos modelos indistinguibles
    se prefiere el más simple —que es el criterio que el propio proyecto viene aplicando—.

    No sustituye a la decisión: la deja tomada por defecto y documentada, y quien ejecuta puede
    elegir otro punto de la curva con la tabla delante.
    """
    if curva.empty:
        raise ValueError("La curva de compromiso está vacía.")

    mejor = curva.loc[curva[metrica].idxmax()]
    candidatas = curva[curva["ic_alto"] >= mejor["ic_bajo"]].sort_values("k")
    elegida = candidatas.iloc[0] if not candidatas.empty else mejor

    return {
        "k_recomendado": int(elegida["k"]),
        "k_mejor": int(mejor["k"]),
        "metrica_recomendada": float(elegida[metrica]),
        "metrica_mejor": float(mejor[metrica]),
        "coste_pp": float((mejor[metrica] - elegida[metrica]) * 100),
        "variables_ahorradas": int(mejor["k"] - elegida["k"]),
        "concluyente": bool(elegida["k"] == mejor["k"]),
        "variables": elegida["variables"].split(", "),
    }


def resumen(
    senal: pd.DataFrame,
    permutacion: Optional[pd.DataFrame] = None,
    descartadas: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Une los criterios en una sola tabla, que es la que se lleva a la memoria.

    `descartadas` admite los dos informes de descarte que produce el módulo, que tienen esquemas
    distintos: el de `poda_automatica` (`variable`, `motivo`, `evidencia`) y el de
    `podar_redundantes` (`descartada`, `a_favor_de`, `correlacion`).
    """
    tabla = senal[["variable", "auc", "fuerza", "nulos_pct"]].copy()
    if permutacion is not None:
        tabla = tabla.merge(
            permutacion[["variable", "caida_media", "caida_desv"]], on="variable", how="left"
        )

    if descartadas is not None and not descartadas.empty:
        columnas = set(descartadas.columns)
        if {"descartada", "a_favor_de"} <= columnas:
            clave, motivo = "descartada", descartadas["a_favor_de"]
        elif {"variable", "motivo"} <= columnas:
            clave = "variable"
            motivo = descartadas["motivo"]
            if "evidencia" in columnas:
                motivo = motivo + " (" + descartadas["evidencia"].astype(str) + ")"
        else:
            logger.warning("Informe de descartes con esquema no reconocido: %s",
                           sorted(columnas))
            clave, motivo = None, None
        if clave is not None:
            tabla["descartada_porque"] = tabla["variable"].map(
                dict(zip(descartadas[clave], motivo))
            )
    orden = "caida_media" if "caida_media" in tabla.columns else "fuerza"
    return tabla.sort_values(orden, ascending=False, na_position="last").reset_index(drop=True)
