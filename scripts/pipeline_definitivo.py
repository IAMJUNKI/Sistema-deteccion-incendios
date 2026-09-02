"""Pipeline definitivo de modelado — consolidación de los tres análisis independientes.

Qué es esto
-----------
El punto de entrada único y final de la fase de modelado del TFM. Consolida los tres estudios
que se hicieron por separado sobre el dataset EGIF y produce, en una sola ejecución, el modelo
que se entrega y toda la evidencia que lo sostiene en la memoria.

Cada etapa está aquí porque contesta a una pregunta concreta que se va a plantear en la
defensa. Ninguna está por completitud decorativa:

    ¿por qué ese algoritmo?            → etapa `modelos` y `pareado`
    ¿por qué esas variables?           → etapa `importancia`
    ¿no estará mirando al futuro?      → etapa `circularidad`
    ¿no se habrá aprendido el mapa?    → etapa `espacial`
    ¿esto se repite si lo repito?      → etapa `semilla`
    ¿mejora lo que ya se publica?      → etapa `fwi`
    ¿qué se le escapa?                 → etapa `errores`
    ¿me puedo creer la probabilidad?   → etapa `fiabilidad`
    ¿depende de cómo definís el fuego? → etapa `target`
    ¿cómo se usa esto un martes?       → etapa `riesgo`
    ¿y el número de verdad, sin trampa? → etapa `ciego`, bajo --abrir-test-ciego

De dónde sale cada decisión
---------------------------
1. **Modelo: LightGBM.** Ya no se afirma: la etapa `modelos` entrena los cuatro y la etapa
   `pareado` dice con un test estadístico si la diferencia es real o es ruido. Medido sobre
   2022, LightGBM gana a los otros tres de forma concluyente —6,3 puntos de recall sobre
   XGBoost, con McNemar en p≈5e-8—, muy por encima del suelo de ruido de 1,45 puntos que fija
   la etapa `semilla`. Los análisis previos que daban empate entre LightGBM y XGBoost se
   hicieron sobre versiones anteriores del dataset; con el actual no lo hay.

2. **Las tres variables acordadas** vienen del datacubo desde el 30 de agosto. Las dos medias
   móviles se verificaron recalculándolas y coinciden hasta el último decimal.
   `consecutive_dry_days` no, y se corrige aquí: ver `src/entrenamiento/dias_secos.py`.

Modelo entregado
----------------
**48 de los 50 predictores.** Quedan fuera `precipitation_sum` y `consecutive_dry_days`.

El motivo NO es que rinda más. Rinde 38,03 % frente a 36,95 % del conjunto completo, y esa
diferencia de 1,09 puntos es menor que el suelo de ruido de 1,45 que fija la etapa `semilla`:
estadísticamente es un empate, y presentarlo como una mejora sería leer ruido. Además, esa
comparación se hizo sobre el año de validación, así que tampoco serviría para decidir.

El motivo es que, **a igualdad de rendimiento, se elige la versión que no admite la objeción**.
Esas dos variables contienen la lluvia del propio día T, posterior a la ignición en muchos
casos. Renunciar a ellas no cuesta nada medible y elimina la crítica más seria que se le puede
hacer al contrato temporal del dataset. Es una decisión metodológica, no un ajuste a los datos.

La etapa `circularidad` deja constancia de las dos cosas: cuánto se habría ganado incluyéndolas
—nada— y qué pasa si se es todavía más estricto.

3. **Calibración de Platt, no isotónica.** La isotónica es escalonada y genera millones de
   empates que degradan cualquier métrica basada en umbral; se midió un coste de más de seis
   puntos de recall al aplicarla sobre la probabilidad de decisión. Platt es estrictamente
   monótona: preserva el orden exacto, de modo que ranking y probabilidad calibrada coinciden
   y el sistema puede operar sobre una única magnitud.

4. **El suelo de ruido se mide, no se hereda.** La etapa `semilla` reentrena cambiando solo la
   semilla aleatoria. Toda diferencia menor que esa dispersión es ruido, y así se interpretan
   el resto de etapas.

Protocolo
---------
Entrenamiento 2019-2020 · calibración 2021 · validación 2022 · **2023 no se toca**.

El calibrador se ajusta sobre un año que el modelo no ha visto y con su prevalencia real
intacta: ajustarlo dentro de muestra reproduciría el sobreajuste en lugar de corregirlo.

2023 no interviene en ninguna decisión. Las cifras de 2022 describen el proceso de selección
—se eligió el algoritmo y se discutieron las variables mirándolas—, así que están levemente
sesgadas al alza. La estimación insesgada del rendimiento sale de la etapa `ciego`, que abre
2023 una sola vez, con todo congelado, y solo cuando se pide con `--abrir-test-ciego`.

Uso
---
    conda activate incendios-forestales
    python scripts/pipeline_definitivo.py                      # todo
    python scripts/pipeline_definitivo.py --etapas definitivo,fwi
    python scripts/pipeline_definitivo.py --rehacer            # ignora resultados previos
    python scripts/pipeline_definitivo.py --abrir-test-ciego   # solo al cerrar, una vez

Cada etapa guarda su tabla en `docs/technical/` y se salta si ya existe, de modo que una
ejecución interrumpida se retoma donde estaba.

El baseline FWI necesita `fwi_van_wagner.py` en `src/baselines/`. Si no está, la etapa `fwi`
detiene la
ejecución en lugar de omitirse en silencio: un informe al que le falta el baseline parece
completo sin serlo.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.entrenamiento import (  # noqa: E402
    calibracion,
    contrato as mod_contrato,
    datos,
    dias_secos,
    metricas,
    modelos,
    seleccion,
)

DIR_MODELOS = RAIZ / "data" / "models"
DIR_TECNICA = RAIZ / "docs" / "technical"

#: Carpeta donde se espera `fwi_van_wagner.py`, que calcula el índice a 1 km desde nuestra
#: propia meteorología. No confundir con `src/ingestion/fwi.py`, que descarga el FWI oficial
#: de CEMS a 27,5 km: son dos fuentes distintas del mismo índice y ambas sirven de baseline.
#: Se puede apuntar a otra ubicación con `--fwi-ruta`.
RUTA_FWI = RAIZ / "src" / "baselines"

ETAPAS = ("contrato", "modelos", "definitivo", "pareado", "circularidad", "importancia",
          "espacial", "semilla", "fwi", "errores", "fiabilidad", "target", "riesgo")


@dataclass
class Configuracion:
    """Parámetros del pipeline. Todo lo que decide un resultado vive aquí y se guarda con él."""

    años_entrenamiento: tuple[int, ...] = (2019, 2020)
    año_calibracion: int = 2021
    año_validacion: int = 2022
    año_reservado: int = 2023

    modelo: str = "lightgbm"
    # Sin ajuste de hiperparámetros: dos de los tres análisis midieron que afinar aporta
    # menos que el ruido de implementación, así que se usan valores razonables y estables.
    hiperparametros: dict = field(default_factory=lambda: {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
    })
    rondas_parada: int = 50
    candidatos: tuple[str, ...] = ("lightgbm", "xgboost", "random_forest", "logistic_regression")

    # Se conserva un negativo de cada N. Un estudio previo del proyecto comparó dieciséis
    # combinaciones y midió que N=100 da un recall indistinguible de N=25 entrenando cuatro
    # veces más rápido; se adopta esa medición.
    modulo_muestreo: int = 100
    semilla: int = 42
    semillas_estabilidad: tuple[int, ...] = (42, 7, 123, 2024, 31)
    fpr_objetivo: float = 0.05         # presupuesto: 5 % del territorio en alerta
    top_k_diario: float = 0.01         # presupuesto de patrulla: 1 % de celdas al día
    n_bootstrap: int = 1000
    meses_temporada: tuple[int, ...] = (6, 7, 8, 9)

    #: Lluvia del propio día T. Es la lectura estricta del problema: si el fuego empieza a las
    #: tres y llueve a las diez de la noche, esa lluvia está en la fila.
    circulares_estrictas: tuple[str, ...] = ("precipitation_sum", "consecutive_dry_days")
    #: Variables que el modelo entregado NO usa. Coinciden con las circulares estrictas por la
    #: decisión del equipo del 31/08: ver el apartado «Modelo entregado» de la cabecera.
    excluir_del_modelo: tuple[str, ...] = ("precipitation_sum", "consecutive_dry_days")
    #: Lectura amplia: **todos** los acumulados incluyen la fecha T por diseño del datacubo, así
    #: que arrastran el mismo problema aunque estén dominados por los días anteriores.
    circulares_amplias: tuple[str, ...] = (
        "precipitation_sum", "consecutive_dry_days",
        "precipitation_sum_3d", "precipitation_sum_7d", "precipitation_sum_14d",
        "precipitation_sum_30d", "temperature_mean_7d", "relative_humidity_mean_7d",
        "relative_humidity_mean_14d", "wind_speed_mean_7d",
    )
    #: Bandas diagonales de la validación cruzada espacial.
    bloques_espaciales: int = 5


# ---------------------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------------------
class CalibradorPlatt:
    """Corrección de prior seguida de un ajuste logístico (Platt).

    Dos etapas, cada una con su cometido:

    1. **Corrección de prior**: el submuestreo de negativos infla las probabilidades en un
       factor conocido. La corrección lo deshace de forma exacta, sin aprender nada.
    2. **Ajuste de Platt**: una regresión logística de un solo parámetro sobre el logaritmo de
       las probabilidades corregidas, ajustada en un año no visto.

    Frente a la alternativa isotónica, Platt es estrictamente monótona. Eso significa que el
    orden de las celdas por riesgo es idéntico antes y después de calibrar, y que los umbrales
    operativos pueden fijarse directamente sobre la probabilidad calibrada. La isotónica, al ser
    escalonada, asigna el mismo valor a millones de celdas y desplaza cualquier corte basado en
    percentiles. Es la discrepancia que separaba dos de los tres análisis y así queda resuelta.
    """

    def __init__(self, tasa_negativos: float):
        self.tasa_negativos = float(tasa_negativos)
        self.logistica: LogisticRegression | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def ajustar(self, prob_cruda: np.ndarray, y: np.ndarray) -> "CalibradorPlatt":
        corregida = calibracion.prior_correction(prob_cruda, self.tasa_negativos)
        self.logistica = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.logistica.fit(self._logit(corregida).reshape(-1, 1), y)
        return self

    def aplicar(self, prob_cruda: np.ndarray) -> np.ndarray:
        corregida = calibracion.prior_correction(prob_cruda, self.tasa_negativos)
        if self.logistica is None:
            return corregida
        return self.logistica.predict_proba(self._logit(corregida).reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------------------
# Contexto compartido entre etapas
# ---------------------------------------------------------------------------------------
class Contexto:
    """Estado que comparten las etapas: datos de entrenamiento, modelo y puntuaciones.

    Existe para que ninguna etapa vuelva a leer del disco algo que otra ya leyó. Las
    puntuaciones sobre el año de validación se guardan por nombre, de modo que la comparación
    pareada pueda enfrentar modelos que se entrenaron en etapas distintas.
    """

    def __init__(self, cfg: Configuracion):
        self.cfg = cfg
        self.contrato = mod_contrato.cargar()
        self.todas = list(self.contrato.predictores)
        self.variables = [v for v in self.todas if v not in set(cfg.excluir_del_modelo)]
        self.train: pd.DataFrame | None = None
        self.parada: pd.DataFrame | None = None
        self.tasa_negativos: float = 0.0
        self.modelo = None
        self.calibrador: CalibradorPlatt | None = None
        self.validacion: pd.DataFrame | None = None      # fecha, cell_id, y, score
        self.prob_calibrada: np.ndarray | None = None
        self.puntuaciones: dict[str, np.ndarray] = {}    # nombre -> score sobre validación
        self.resultados: dict[str, dict] = {}
        self._coordenadas: pd.DataFrame | None = None

    @property
    def clave_recall(self) -> str:
        """Nombre de la métrica principal, que `metricas.evaluate` compone con el FPR pedido."""
        return f"recall_at_fpr{int(self.cfg.fpr_objetivo * 100)}"

    @property
    def clave_diario(self) -> str:
        return f"recall_at_top{self.cfg.top_k_diario:.0%}_daily"

    def preparar_datos(self) -> None:
        """Muestrea entrenamiento y parada una sola vez para toda la ejecución."""
        if self.train is not None:
            return
        cfg = self.cfg
        años = [*cfg.años_entrenamiento, cfg.año_calibracion, cfg.año_validacion]
        dias_secos.construir_cache(self.contrato, años)

        self.train, info = self._muestrear(cfg.años_entrenamiento)
        self.parada, _ = self._muestrear([cfg.año_calibracion])
        self.tasa_negativos = float(info["tasa_negativos"])
        print(f"  entrenamiento: {len(self.train):,} filas · "
              f"{int(self.train.target_ignicion.sum())} igniciones")
        print(f"  parada:        {len(self.parada):,} filas · "
              f"{int(self.parada.target_ignicion.sum())} igniciones")
        print(f"  tasa de negativos conservada: {self.tasa_negativos:.4f}")

    def _muestrear(self, años) -> tuple[pd.DataFrame, dict]:
        marco, info = datos.muestrear_entrenamiento(
            self.contrato, list(años), modulo=self.cfg.modulo_muestreo, columnas_extra=("x", "y")
        )
        return dias_secos.corregir(marco, list(años)), info

    def coordenadas(self) -> pd.DataFrame:
        """Mapa `cell_id -> (x, y)`. Es estático, así que se lee una vez de un año cualquiera."""
        if self._coordenadas is None:
            marco = pd.read_parquet(
                self.contrato.ruta(self.cfg.año_validacion), columns=["cell_id", "x", "y"]
            )
            self._coordenadas = marco.drop_duplicates("cell_id").reset_index(drop=True)
            del marco
        return self._coordenadas

    def entrenar(self, variables: list[str], nombre: str | None = None, semilla: int | None = None):
        """Entrena un modelo con parada temprana sobre el año de calibración."""
        nombre = nombre or self.cfg.modelo
        params = self.cfg.hiperparametros if nombre == "lightgbm" else None
        modelo = modelos.construir(nombre, params, semilla=semilla or self.cfg.semilla)
        modelo, mejor = modelos.entrenar(
            modelo, nombre,
            self.train[variables], self.train.target_ignicion,
            self.parada[variables], self.parada.target_ignicion,
            rondas_parada=self.cfg.rondas_parada,
        )
        return modelo, mejor

    def puntuar(self, modelo, año: int, variables: list[str]) -> pd.DataFrame:
        """Recorre la población completa de un año y devuelve verdad, puntuación y fecha.

        Se evalúa sin submuestrear: la prevalencia real es la que da sentido a las métricas. El
        recorrido va por lotes para que la memoria no dependa del tamaño del año, y al terminar
        se comprueba que se cubrió el año entero. Una ejecución que pierda celdas en silencio
        produce métricas *mejores* que las reales, que es el peor fallo posible en un TFM.
        """
        trozos, filas = [], 0
        for lote in datos.iter_evaluacion(self.contrato, [año], predictores=None):
            filas += len(lote)
            lote = dias_secos.corregir(lote, [año])
            trozos.append(pd.DataFrame({
                "fecha": lote["fecha"].to_numpy(),
                "cell_id": lote["cell_id"].to_numpy(),
                "y": lote["target_ignicion"].to_numpy(),
                "score": modelo.predict_proba(lote[variables])[:, 1].astype("float32"),
            }))
            del lote
        datos.verificar_cobertura(self.contrato, [año], filas)
        return pd.concat(trozos, ignore_index=True)

    def metricas_de(self, marco: pd.DataFrame, prob=None) -> dict:
        """Cuadro completo sobre un año ya puntuado, con los añadidos del proyecto."""
        cfg = self.cfg
        resultado = metricas.evaluate(
            y_true=marco.y.to_numpy(),
            y_score=marco.score.to_numpy(),
            y_prob_calibrated=prob,
            day_index=datos.indice_dia(marco.fecha),
            fpr_max=cfg.fpr_objetivo,
            top_k=cfg.top_k_diario,
            n_boot=cfg.n_bootstrap,
            seed=cfg.semilla,
        )
        resultado["roc_auc_temporada"] = roc_en_temporada(marco, cfg.meses_temporada)
        resultado["roc_auc_dentro_del_dia"] = roc_dentro_del_dia(marco)
        return resultado

    def entrenar_y_evaluar(self, variables: list[str]) -> dict:
        """Ciclo completo con un subconjunto de variables. Lo consumen ablación y curva."""
        modelo, _ = self.entrenar(variables)
        marco = self.puntuar(modelo, self.cfg.año_validacion, variables)
        resultado = self.metricas_de(marco)
        del modelo, marco
        return resultado


# ---------------------------------------------------------------------------------------
# Métricas propias del proyecto
# ---------------------------------------------------------------------------------------
def roc_en_temporada(marco: pd.DataFrame, meses: tuple[int, ...]) -> float:
    """ROC-AUC restringido a los meses de campaña.

    El ROC del año completo premia sobre todo separar agosto de enero, que es lo que ya dice el
    calendario. Restringir a la temporada mide lo que de verdad importa: discriminar entre días
    y celdas cuando todos ellos son potencialmente peligrosos.
    """
    sub = marco[pd.to_datetime(marco.fecha).dt.month.isin(meses)]
    if sub.y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(sub.y, sub.score))


def roc_dentro_del_dia(marco: pd.DataFrame) -> float:
    """ROC-AUC calculado día a día y promediado: aísla la señal espacial.

    Responde a la pregunta operativa —dado que hoy es 12 de agosto, ¿acierta *qué celdas*?—
    eliminando por completo la contribución del calendario.
    """
    valores = [
        roc_auc_score(g.y, g.score)
        for _, g in marco.groupby("fecha", sort=False)
        if g.y.nunique() == 2
    ]
    return float(np.mean(valores)) if valores else float("nan")


def bandas_diagonales(coordenadas: pd.DataFrame, n_bloques: int) -> pd.Series:
    """Asigna cada celda a una banda diagonal del mapa.

    Las bandas van en diagonal y no en cuadrantes porque Galicia tiene un gradiente muy marcado
    entre la costa atlántica y el interior de Ourense. Unos cuadrantes norte-sur dejarían un
    pliegue entero sin costa, y el modelo fallaría por no haber visto ese clima, no por
    incapacidad de generalizar. La diagonal reparte costa e interior en todos los pliegues.
    """
    escala = (coordenadas.x + coordenadas.y).to_numpy()
    cortes = np.quantile(escala, np.linspace(0, 1, n_bloques + 1)[1:-1])
    return pd.Series(np.searchsorted(cortes, escala), index=coordenadas.cell_id)


# ---------------------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------------------
#: Columnas que jamás pueden ser predictoras porque el modelo se compara **contra** ellas.
#: El datacubo publica el FWI oficial de CEMS dentro del Parquet para poder usarlo de
#: referencia, correctamente declarado fuera de los predictores. Si alguna regeneración futura
#: lo colara entre ellos, el modelo llevaría el índice dentro y la afirmación «superamos al
#: FWI» pasaría a ser circular: estaríamos comparando el índice contra sí mismo. El contrato no
#: vigila este caso, así que se vigila aquí.
COLUMNAS_BASELINE = frozenset({"fire_weather_index", "ffmc", "dmc", "dc", "isi", "bui", "fwi"})


def etapa_contrato(ctx: Contexto) -> pd.DataFrame:
    """Deja constancia de sobre qué datos exactos se ejecutó todo lo demás."""
    c = ctx.contrato
    print(c.resumen())
    ciclicas = [v for v in ctx.variables if v.endswith(("_sin", "_cos"))]
    print(f"\n  codificación cíclica del calendario: {ciclicas or 'ninguna, como se acordó'}")
    print(f"  contrato temporal: {c.contrato_temporal}")
    print(f"  filas descartadas por predictores incompletos: {c.filas_descartadas:,}")

    coladas = sorted(COLUMNAS_BASELINE & set(c.predictores))
    if coladas:
        raise SystemExit(
            f"\nEl metadato declara como predictoras columnas que son el baseline: {coladas}.\n"
            "El modelo no puede usar de entrada el índice contra el que se compara: la\n"
            "comparación quedaría vacía de sentido. Deben moverse a\n"
            "`baseline_columns_not_predictors` en metadata.json antes de continuar."
        )
    print(f"  columnas de baseline fuera de los predictores: correcto "
          f"({len(COLUMNAS_BASELINE)} nombres vigilados)")

    filas = [{"clave": "predictores", "valor": len(c.predictores)},
             {"clave": "años", "valor": str(c.anios)},
             {"clave": "filas_totales", "valor": c.filas(c.anios)},
             {"clave": "contrato_temporal", "valor": c.contrato_temporal},
             {"clave": "filas_descartadas", "valor": c.filas_descartadas},
             {"clave": "variables_ciclicas", "valor": str(ciclicas)},
             {"clave": "año_reservado", "valor": ctx.cfg.año_reservado}]
    for grupo in sorted(c.grupos):
        filas.append({"clave": f"grupo_{grupo}", "valor": len(c.grupos[grupo])})
    return pd.DataFrame(filas)


def etapa_modelos(ctx: Contexto) -> pd.DataFrame:
    """Entrena los cuatro candidatos y los mide con el mismo protocolo.

    La elección del algoritmo deja de ser una afirmación del texto y pasa a ser una tabla. Se
    incluye la regresión logística a propósito: si el boosting no le sacara una ventaja clara,
    habría que entregar el modelo lineal, que es interpretable y se explica en dos frases.
    """
    ctx.preparar_datos()
    filas = []
    for nombre in ctx.cfg.candidatos:
        inicio = time.time()
        try:
            modelo, mejor = ctx.entrenar(ctx.variables, nombre=nombre)
        except Exception as error:                       # noqa: BLE001
            print(f"  {nombre}: no disponible ({type(error).__name__}: {error})")
            continue
        marco = ctx.puntuar(modelo, ctx.cfg.año_validacion, ctx.variables)
        resultado = ctx.metricas_de(marco)
        ctx.puntuaciones[nombre] = marco.score.to_numpy()
        if ctx.validacion is None:
            ctx.validacion = marco[["fecha", "cell_id", "y"]].copy()
        filas.append({"modelo": nombre, "arboles": mejor, "minutos": (time.time() - inicio) / 60,
                      **resultado})
        print(f"  {nombre:22s} recall {resultado[ctx.clave_recall]:.4f} · "
              f"diario {resultado[ctx.clave_diario]:.4f} · "
              f"ROC-día {resultado['roc_auc_dentro_del_dia']:.4f} · "
              f"{(time.time() - inicio) / 60:.1f} min")
        del modelo, marco
    return metricas.summarize(filas)


def etapa_definitivo(ctx: Contexto) -> pd.DataFrame:
    """Entrena el modelo que se entrega, lo calibra y lo valida sobre el año completo."""
    ctx.preparar_datos()
    cfg = ctx.cfg

    print("  entrenando LightGBM con parada temprana sobre el año de calibración")
    ctx.modelo, mejor = ctx.entrenar(ctx.variables)
    print(f"  árboles utilizados: {mejor or cfg.hiperparametros['n_estimators']}")

    print(f"  calibrando sobre {cfg.año_calibracion}, que el modelo no ha visto")
    marco_calib = ctx.puntuar(ctx.modelo, cfg.año_calibracion, ctx.variables)
    ctx.calibrador = CalibradorPlatt(ctx.tasa_negativos).ajustar(
        marco_calib.score.to_numpy(), marco_calib.y.to_numpy()
    )
    print(f"  {len(marco_calib):,} filas · {int(marco_calib.y.sum())} igniciones")
    del marco_calib

    print(f"  validando sobre {cfg.año_validacion} completo")
    ctx.validacion = ctx.puntuar(ctx.modelo, cfg.año_validacion, ctx.variables)
    ctx.prob_calibrada = ctx.calibrador.aplicar(ctx.validacion.score.to_numpy())
    ctx.puntuaciones["definitivo"] = ctx.validacion.score.to_numpy()

    resultado = ctx.metricas_de(ctx.validacion, prob=ctx.prob_calibrada)
    ctx.resultados["definitivo"] = resultado
    for clave, valor in resultado.items():
        print(f"    {clave:34s} {valor:.6f}" if isinstance(valor, float)
              else f"    {clave:34s} {valor:,}")

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    with open(DIR_MODELOS / "modelo_definitivo.pkl", "wb") as f:
        pickle.dump({"modelo": ctx.modelo, "calibrador": ctx.calibrador,
                     "variables": ctx.variables, "configuracion": asdict(cfg),
                     "tasa_negativos": ctx.tasa_negativos}, f)
    print(f"\n  modelo guardado en {DIR_MODELOS / 'modelo_definitivo.pkl'}")
    return pd.DataFrame([{"variante": "definitivo", "n_variables": len(ctx.variables),
                          **resultado}])


def etapa_pareado(ctx: Contexto) -> pd.DataFrame:
    """Compara los modelos sobre **los mismos incendios**, con bootstrap y McNemar.

    Es la comparación correcta porque los intervalos de
    confianza de cada modelo por separado son marginales: dos modelos pueden tener intervalos
    solapados y aun así uno ganar sistemáticamente al otro en los mismos fuegos. McNemar mira
    exactamente eso, los casos en que discrepan.
    """
    if len(ctx.puntuaciones) < 2:
        print("  hacen falta al menos dos modelos puntuados; ejecutar antes la etapa `modelos`")
        return pd.DataFrame()
    print(f"  enfrentando: {', '.join(ctx.puntuaciones)}")
    tabla = metricas.comparar_modelos(
        y_true=ctx.validacion.y.to_numpy(),
        puntuaciones=ctx.puntuaciones,
        day_index=datos.indice_dia(ctx.validacion.fecha),
        fpr_max=ctx.cfg.fpr_objetivo,
        top_k=ctx.cfg.top_k_diario,
        n_boot=ctx.cfg.n_bootstrap,
        seed=ctx.cfg.semilla,
    )
    print(tabla.to_string(index=False))
    concluyentes = tabla[tabla.get("concluyente", False)] if "concluyente" in tabla else tabla
    print(f"\n  comparaciones concluyentes: {len(concluyentes)} de {len(tabla)}")
    print("  Las no concluyentes significan empate técnico: elegir entre esos modelos es una")
    print("  decisión de conveniencia (velocidad, interpretabilidad), no de rendimiento.")
    return tabla


def etapa_circularidad(ctx: Contexto) -> pd.DataFrame:
    """¿Se apoya el modelo en información posterior a la ignición? Tres lecturas del problema.

    El datacubo no aplica desfase temporal. `precipitation_sum` es la lluvia de **todo** el día
    T: si el fuego empieza a las tres y llueve a las diez de la noche, esa lluvia está en la
    fila. Y no es la única —**todos** los acumulados incluyen la fecha T por diseño—, así que
    quitar solo esas dos variables sería una respuesta incompleta: `precipitation_sum_3d`, que
    es la variable más importante del modelo, también arrastra la lluvia del día T.

    Por eso se miden tres escenarios, de menos a más exigente:

    1. **estricta**: fuera la lluvia del día y la racha de días secos.
    2. **amplia**: fuera además todos los acumulados y medias móviles, que también incluyen T.
    3. **sin desfase (pronóstico)**: se reconstruyen los acumulados excluyendo el día T por
       aritmética exacta sobre las columnas publicadas, y se retira toda observación del propio
       día. El resultado solo usa información disponible al acabar T-1, así que ya no es un
       índice de diagnóstico sino un pronóstico de verdad. Es el escenario que cierra la
       discusión: si aquí el modelo sigue batiendo al FWI, la objeción desaparece del todo.
    """
    ctx.preparar_datos()
    cfg = ctx.cfg
    base = ctx.resultados.get("definitivo")
    filas = []

    # El modelo entregado ya excluye las circulares estrictas, así que el escenario que hay que
    # medir aquí es el contrario: qué se gana **añadiéndolas**. Es lo que cuantifica el precio
    # de la decisión, y sale del cuadro de la etapa `definitivo` como referencia.
    for etiqueta, usar in [("completa (incluye el día T)", ctx.todas),
                           ("amplia", [v for v in ctx.todas if v not in cfg.circulares_amplias])]:
        restringidas = list(usar)
        modelo, _ = ctx.entrenar(restringidas)
        marco = ctx.puntuar(modelo, cfg.año_validacion, restringidas)
        resultado = ctx.metricas_de(marco)
        ctx.puntuaciones[f"sin_circulares_{etiqueta}"] = marco.score.to_numpy()
        fila = {"escenario": etiqueta, "n_variables": len(restringidas), **resultado}
        if base:
            fila["caida_recall"] = base[ctx.clave_recall] - resultado[ctx.clave_recall]
        filas.append(fila)
        print(f"  {etiqueta:10s} {len(restringidas):>2} variables · "
              f"recall {resultado[ctx.clave_recall]:.4f} · "
              f"ROC-día {resultado['roc_auc_dentro_del_dia']:.4f}"
              + (f" · caída {fila['caida_recall']:+.4f}" if base else ""))
        del modelo, marco

    print("\n  escenario de pronóstico: se reconstruyen los acumulados sin el día T")
    resultado, n_vars = _variante_pronostico(ctx)
    fila = {"escenario": "pronóstico (solo T-1)", "n_variables": n_vars, **resultado}
    if base:
        fila["caida_recall"] = base[ctx.clave_recall] - resultado[ctx.clave_recall]
        print(f"  pronóstico {n_vars:>2} variables · recall {resultado[ctx.clave_recall]:.4f} · "
              f"ROC-día {resultado['roc_auc_dentro_del_dia']:.4f} · "
              f"caída {fila['caida_recall']:+.4f}")
    filas.append(fila)

    if base:
        print(f"\n  referencia (modelo definitivo, {len(ctx.variables)} variables): "
              f"recall {base[ctx.clave_recall]:.4f}")
        print("  Toda caída menor que el suelo de ruido de la etapa `semilla` es un empate.")
    return pd.DataFrame(filas)


#: Acumulados que pueden despojarse del día T con aritmética exacta sobre lo publicado.
#: sumas: total de la ventana menos el valor de hoy. medias: se deshace el promedio.
DESFASABLES_SUMA = {
    "precipitation_sum_3d": ("precipitation_sum", 3),
    "precipitation_sum_7d": ("precipitation_sum", 7),
    "precipitation_sum_14d": ("precipitation_sum", 14),
    "precipitation_sum_30d": ("precipitation_sum", 30),
}
DESFASABLES_MEDIA = {
    "temperature_mean_7d": ("temperature_mean", 7),
    "relative_humidity_mean_7d": ("relative_humidity_mean", 7),
    "relative_humidity_mean_14d": ("relative_humidity_mean", 14),
    "wind_speed_mean_7d": ("wind_speed_mean", 7),
}


def _desfasar(marco: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reconstruye los acumulados excluyendo el día T y retira las observaciones de hoy.

    La aritmética es exacta, no una aproximación. Una suma de siete días que incluye hoy menos
    el valor de hoy es la suma de los seis anteriores. Una media de siete días multiplicada por
    siete, menos el valor de hoy, dividida entre seis, es la media de los seis anteriores.

    Lo que queda no contiene ni un solo dato del día que se quiere predecir: es el conjunto de
    variables con el que se podría emitir un aviso la tarde anterior.
    """
    salida = marco.copy()
    conservadas = []
    for nombre, (origen, ventana) in DESFASABLES_SUMA.items():
        if nombre in salida and origen in salida:
            salida[nombre] = salida[nombre] - salida[origen]
            conservadas.append(nombre)
    for nombre, (origen, ventana) in DESFASABLES_MEDIA.items():
        if nombre in salida and origen in salida:
            salida[nombre] = (salida[nombre] * ventana - salida[origen]) / (ventana - 1)
            conservadas.append(nombre)
    return salida, conservadas


def _variante_pronostico(ctx: Contexto) -> tuple[dict, int]:
    """Entrena y evalúa el escenario de pronóstico puro."""
    cfg = ctx.cfg
    train, _ = _desfasar(ctx.train)
    parada, _ = _desfasar(ctx.parada)

    # Fuera toda observación del propio día T: la meteorología diaria y la ventana 12-18 h.
    del_dia = [v for v in ctx.variables
               if v.startswith(("temperature_", "relative_humidity_", "wind_speed_", "vpd_"))
               and not v.endswith(("_7d", "_14d"))]
    del_dia += ["precipitation_sum", "consecutive_dry_days"]
    variables = [v for v in ctx.variables if v not in set(del_dia)]

    modelo = modelos.construir(cfg.modelo, cfg.hiperparametros, cfg.semilla)
    modelo, _ = modelos.entrenar(
        modelo, cfg.modelo, train[variables], train.target_ignicion,
        parada[variables], parada.target_ignicion, rondas_parada=cfg.rondas_parada,
    )

    trozos, filas = [], 0
    for lote in datos.iter_evaluacion(ctx.contrato, [cfg.año_validacion], predictores=None):
        filas += len(lote)
        lote = dias_secos.corregir(lote, [cfg.año_validacion])
        lote, _ = _desfasar(lote)
        trozos.append(pd.DataFrame({
            "fecha": lote["fecha"].to_numpy(),
            "cell_id": lote["cell_id"].to_numpy(),
            "y": lote["target_ignicion"].to_numpy(),
            "score": modelo.predict_proba(lote[variables])[:, 1].astype("float32"),
        }))
        del lote
    datos.verificar_cobertura(ctx.contrato, [cfg.año_validacion], filas)
    marco = pd.concat(trozos, ignore_index=True)
    ctx.puntuaciones["pronostico"] = marco.score.to_numpy()
    resultado = ctx.metricas_de(marco)
    del modelo, marco, train, parada
    return resultado, len(variables)


def etapa_importancia(ctx: Contexto) -> pd.DataFrame:
    """Importancia por permutación y ablación por grupos temáticos.

    Son dos preguntas distintas. La permutación dice qué usa el modelo entrenado; la ablación
    dice qué pasaría si esa familia de variables no existiera. Difieren cuando hay redundancia:
    una variable puede salir con importancia baja porque otra la cubre, y aun así su grupo
    entero ser imprescindible.
    """
    ctx.preparar_datos()
    if ctx.modelo is None:
        ctx.modelo, _ = ctx.entrenar(ctx.variables)

    print("  importancia por permutación sobre una submuestra de validación")
    muestra = ctx.parada.sample(n=min(200_000, len(ctx.parada)), random_state=ctx.cfg.semilla)
    # La función baraja las columnas in situ, así que hay que darle un marco propio y no una
    # vista del original: sobre una vista, pandas avisa y la escritura podría no propagarse.
    importancia = seleccion.importancia_permutacion(
        ctx.modelo, muestra[ctx.variables].copy(), muestra.target_ignicion.to_numpy(),
        ctx.variables, repeticiones=3, semilla=ctx.cfg.semilla,
    )
    print(importancia.head(12).to_string(index=False))
    importancia.to_csv(DIR_TECNICA / "importancia_permutacion.csv", index=False)

    print("\n  ablación por grupos temáticos (un entrenamiento completo por grupo)")
    grupos = {g: [v for v in vs if v in ctx.variables]
              for g, vs in ctx.contrato.grupos.items()}
    grupos = {g: vs for g, vs in grupos.items() if vs}
    tabla = seleccion.ablacion_grupos(
        grupos, ctx.variables, ctx.entrenar_y_evaluar, metrica=ctx.clave_recall
    )
    print(tabla.to_string(index=False))
    return tabla


def etapa_espacial(ctx: Contexto) -> pd.DataFrame:
    """Validación cruzada por bandas diagonales: ¿generaliza a territorio no visto?

    Es la prueba más dura del conjunto y la que más veces ha desmentido conclusiones. Un modelo
    puede tener un ROC excelente y estar simplemente recordando qué celdas arden —los montes de
    Ourense arden todos los años—, lo que no sirve para desplegarlo en un sitio nuevo. Aquí se
    entrena sin una banda entera del mapa y se evalúa solo en ella.

    La caída frente al resultado normal es la medida honesta de cuánto del rendimiento es
    señal meteorológica transferible y cuánto es memoria del mapa.
    """
    ctx.preparar_datos()
    banda_de = bandas_diagonales(ctx.coordenadas(), ctx.cfg.bloques_espaciales)
    banda_train = ctx.train.cell_id.map(banda_de)
    banda_parada = ctx.parada.cell_id.map(banda_de)

    filas = []
    for bloque in range(ctx.cfg.bloques_espaciales):
        train = ctx.train[banda_train != bloque]
        parada = ctx.parada[banda_parada != bloque]
        if train.target_ignicion.sum() < 20:
            print(f"  banda {bloque}: muy pocas igniciones fuera de ella, se omite")
            continue

        modelo = modelos.construir(ctx.cfg.modelo, ctx.cfg.hiperparametros, ctx.cfg.semilla)
        modelo, _ = modelos.entrenar(
            modelo, ctx.cfg.modelo, train[ctx.variables], train.target_ignicion,
            parada[ctx.variables], parada.target_ignicion, rondas_parada=ctx.cfg.rondas_parada,
        )
        marco = ctx.puntuar(modelo, ctx.cfg.año_validacion, ctx.variables)
        marco = marco[marco.cell_id.map(banda_de) == bloque]
        resultado = ctx.metricas_de(marco)
        filas.append({"banda": bloque, "celdas": int(marco.cell_id.nunique()),
                      "igniciones": int(marco.y.sum()), **resultado})
        print(f"  banda {bloque}: recall {resultado[ctx.clave_recall]:.4f} · "
              f"ROC-día {resultado['roc_auc_dentro_del_dia']:.4f} · "
              f"{int(marco.y.sum())} igniciones")
        del modelo, marco, train, parada

    tabla = pd.DataFrame(filas)
    base = ctx.resultados.get("definitivo")
    if base is not None and len(tabla):
        media = tabla["roc_auc_dentro_del_dia"].mean()
        print(f"\n  ROC-día con el reparto habitual: {base['roc_auc_dentro_del_dia']:.4f}")
        print(f"  ROC-día en territorio no visto:  {media:.4f}")
        print(f"  caída: {base['roc_auc_dentro_del_dia'] - media:+.4f}")
        print("  Toda caída aquí es rendimiento que NO se transfiere a una zona nueva y")
        print("  debe declararse como tal en la memoria.")
    return tabla


def etapa_semilla(ctx: Contexto) -> pd.DataFrame:
    """Reentrena cambiando solo la semilla: fija el suelo de ruido de todo el estudio.

    Sin esta etapa no se puede interpretar ninguna otra. Una diferencia de medio punto entre
    dos variantes solo significa algo si el mismo modelo, repetido, varía menos que eso. Aquí
    se mide esa dispersión, y pasa a ser el listón que cualquier conclusión debe superar.
    """
    ctx.preparar_datos()
    filas = []
    for semilla in ctx.cfg.semillas_estabilidad:
        modelo, _ = ctx.entrenar(ctx.variables, semilla=semilla)
        marco = ctx.puntuar(modelo, ctx.cfg.año_validacion, ctx.variables)
        resultado = ctx.metricas_de(marco)
        filas.append({"semilla": semilla, **resultado})
        print(f"  semilla {semilla:>5}: recall {resultado[ctx.clave_recall]:.4f} · "
              f"diario {resultado[ctx.clave_diario]:.4f}")
        del modelo, marco

    tabla = pd.DataFrame(filas)
    recall = tabla[ctx.clave_recall]
    dispersion = float(recall.max() - recall.min())
    igniciones = dispersion * tabla["n_positives"].iloc[0]
    print(f"\n  dispersión del recall solo por la semilla: {dispersion:.4f} "
          f"({igniciones:.0f} igniciones)")
    print("  >> Ninguna diferencia menor que esto puede presentarse como un hallazgo.")
    tabla.attrs["suelo_ruido"] = dispersion
    return tabla


def etapa_fwi(ctx: Contexto, ruta_fwi: Path) -> pd.DataFrame:
    """Baseline físico: el índice canadiense de peligro de incendio calculado a 1 km.

    Batir a una regresión logística solo demuestra que el boosting gana a un modelo lineal.
    Para sostener que el sistema aporta algo sobre lo que ya se publica operativamente hay que
    batir al FWI, que es el índice que difunden AEMET y EFFIS.

    A diferencia del script original, aquí un fallo al importar **detiene la ejecución**: un
    baseline que desaparece en silencio deja un informe que parece completo y no lo está.
    """
    sys.path.insert(0, str(ruta_fwi))
    try:
        from fwi_van_wagner import calcular_fwi
    except ImportError as error:
        raise SystemExit(
            f"\nNo se encuentra `fwi_van_wagner.py` en {ruta_fwi}.\n"
            "Es el módulo con las ecuaciones de Van Wagner que sostiene el baseline físico.\n"
            "Indicar su ubicación con\n"
            f"--fwi-ruta, o excluir la etapa `fwi` de --etapas.\n\nDetalle: {error}"
        ) from error

    necesarias = ["temperature_max_12_18h", "relative_humidity_min_12_18h",
                  "wind_speed_max_12_18h", "precipitation_sum"]
    faltan = [c for c in necesarias if c not in ctx.contrato.predictores]
    if faltan:
        raise SystemExit(f"El dataset no trae las columnas que necesita el FWI: {faltan}")

    año = ctx.cfg.año_validacion
    marco = pd.read_parquet(ctx.contrato.ruta(año),
                            columns=["cell_id", "fecha", *necesarias, "target_ignicion"])
    print(f"  calculando el índice sobre {len(marco):,} filas (es acumulado: no se trocea)")
    indices = calcular_fwi(marco, col_t=necesarias[0], col_h=necesarias[1],
                           col_w=necesarias[2], col_p=necesarias[3], verbose=False)
    puntuado = pd.DataFrame({
        "fecha": marco["fecha"].to_numpy(),
        "cell_id": marco["cell_id"].to_numpy(),
        "y": marco["target_ignicion"].to_numpy(),
        "score": np.nan_to_num(indices["fwi"].to_numpy(), nan=0.0).astype("float32"),
    })
    del marco, indices

    resultado = ctx.metricas_de(puntuado)
    ctx.puntuaciones["FWI"] = puntuado.score.to_numpy()
    base = ctx.resultados.get("definitivo")
    if base:
        ventaja = base[ctx.clave_recall] - resultado[ctx.clave_recall]
        print(f"  recall del modelo   {base[ctx.clave_recall]:.4f}")
        print(f"  recall del FWI      {resultado[ctx.clave_recall]:.4f}")
        print(f"  ventaja             {ventaja:+.4f}  ≈ "
              f"{ventaja * base['n_positives']:+.0f} igniciones al año")
    print("\n  Nota obligatoria para la memoria: el FWI se calcula aquí con los extremos de la")
    print("  ventana 12-18 h y no con valores de mediodía solar, lo que lo sobreestima de forma")
    print("  sistemática. Vale como ordenación de riesgo, no como categoría oficial de peligro.")
    return pd.DataFrame([{"variante": "baseline FWI", "n_variables": 4, **resultado}])


def etapa_errores(ctx: Contexto) -> pd.DataFrame:
    """Qué igniciones se escapan y en qué se diferencian de las que se detectan.

    Un recall del 40 % significa que seis de cada diez fuegos no se ven. Decir *cuáles* es lo
    que separa un modelo de un sistema: si lo que se escapa son los incendios pequeños de
    invierno, el sistema sirve; si son los grandes de agosto, no sirve para nada.
    """
    if ctx.validacion is None:
        print("  hace falta la etapa `definitivo`")
        return pd.DataFrame()

    y = ctx.validacion.y.to_numpy()
    aciertos = metricas.aciertos_at_fpr(y, ctx.validacion.score.to_numpy(), ctx.cfg.fpr_objetivo)
    positivos = ctx.validacion[y == 1].copy()
    positivos["detectada"] = aciertos

    columnas = ["temperature_max_12_18h", "relative_humidity_min_12_18h", "wind_speed_max_12_18h",
                "precipitation_sum", "consecutive_dry_days", "elevation_mean", "scrub",
                "road_length_km", "burned_area_ha"]
    columnas = [c for c in columnas if c in ctx.contrato.predictores or c == "burned_area_ha"]
    contexto = pd.read_parquet(ctx.contrato.ruta(ctx.cfg.año_validacion),
                               columns=["cell_id", "fecha", *columnas])
    positivos = positivos.merge(contexto, on=["cell_id", "fecha"], how="left")
    del contexto

    positivos["mes"] = pd.to_datetime(positivos.fecha).dt.month
    filas = []
    for columna in [*columnas, "mes"]:
        detectadas = positivos.loc[positivos.detectada, columna]
        perdidas = positivos.loc[~positivos.detectada, columna]
        filas.append({
            "variable": columna,
            "media_detectadas": float(detectadas.mean()),
            "media_perdidas": float(perdidas.mean()),
            "diferencia": float(detectadas.mean() - perdidas.mean()),
        })
    tabla = pd.DataFrame(filas)
    print(f"  {int(aciertos.sum())} detectadas · {int((~aciertos).sum())} perdidas "
          f"de {len(aciertos)} igniciones")
    print(tabla.to_string(index=False))

    por_mes = (positivos.groupby("mes")
               .agg(igniciones=("detectada", "size"), detectadas=("detectada", "sum"))
               .assign(recall=lambda d: d.detectadas / d.igniciones).reset_index())
    print("\n  recall por mes:")
    print(por_mes.to_string(index=False))
    por_mes.to_csv(DIR_TECNICA / "errores_por_mes.csv", index=False)
    return tabla


def etapa_fiabilidad(ctx: Contexto) -> pd.DataFrame:
    """¿Se puede leer la probabilidad como una probabilidad?

    Si el modelo dice 3 % y de esas celdas arde el 3 %, la salida es interpretable y se pueden
    fijar umbrales con sentido. Si dice 3 % y arde el 0,1 %, el número solo sirve para ordenar.
    Es la diferencia entre entregar un ranking y entregar un sistema de alerta.
    """
    if ctx.prob_calibrada is None:
        print("  hace falta la etapa `definitivo`")
        return pd.DataFrame()
    tabla = metricas.reliability_table(ctx.validacion.y.to_numpy(), ctx.prob_calibrada, n_bins=12)
    print(tabla.to_string(index=False))
    return tabla


def etapa_target(ctx: Contexto) -> pd.DataFrame:
    """¿Cambia la conclusión si se define el fuego de otra manera?

    El target actual cuenta cualquier ignición registrada, incluidos los conatos de pocos
    metros. Un tribunal preguntará si el modelo detecta *los incendios que importan*. Aquí se
    reevalúa el mismo modelo restringiendo los positivos por superficie quemada, sin reentrenar:
    lo que cambia es la pregunta, no el sistema.
    """
    if ctx.validacion is None:
        print("  hace falta la etapa `definitivo`")
        return pd.DataFrame()

    extra = pd.read_parquet(ctx.contrato.ruta(ctx.cfg.año_validacion),
                            columns=["cell_id", "fecha", "burned_area_ha", "large_fire_500ha"])
    marco = ctx.validacion.merge(extra, on=["cell_id", "fecha"], how="left")
    del extra

    filas = []
    for etiqueta, mascara in [
        ("todas las igniciones", marco.y == 1),
        ("superficie ≥ 1 ha", (marco.y == 1) & (marco.burned_area_ha >= 1)),
        ("superficie ≥ 10 ha", (marco.y == 1) & (marco.burned_area_ha >= 10)),
        ("superficie ≥ 100 ha", (marco.y == 1) & (marco.burned_area_ha >= 100)),
        ("grandes incendios (≥500 ha)", (marco.y == 1) & (marco.large_fire_500ha == 1)),
    ]:
        n = int(mascara.sum())
        if n < 10:
            print(f"  {etiqueta}: solo {n} casos, no se evalúa")
            continue
        sub = marco[(marco.y == 0) | mascara].copy()
        sub["y"] = mascara[sub.index].astype(int).to_numpy()
        resultado = ctx.metricas_de(sub)
        filas.append({"definicion": etiqueta, "igniciones": n, **resultado})
        print(f"  {etiqueta:30s} {n:>5} igniciones · "
              f"recall {resultado[ctx.clave_recall]:.4f} · "
              f"ROC-día {resultado['roc_auc_dentro_del_dia']:.4f}")
        del sub
    return pd.DataFrame(filas)


def etapa_ciego(ctx: Contexto) -> pd.DataFrame:
    """Evaluación única sobre el año reservado. **Se ejecuta una sola vez, al final de todo.**

    Por qué hace falta
    ------------------
    Todas las cifras de 2022 están, en rigor, contaminadas. No porque el modelo haya visto ese
    año —no lo ha visto—, sino porque *nosotros* sí: elegimos LightGBM mirando su resultado en
    2022, y discutimos qué variables retirar mirando su resultado en 2022. Cada decisión tomada
    a la vista de un conjunto lo convierte, un poco, en conjunto de entrenamiento. El sesgo es
    pequeño pero real, y es exactamente el reproche que un tribunal sabe hacer.

    2023 no ha intervenido en ninguna decisión. Por eso su número es el único que puede
    presentarse como estimación insesgada del rendimiento en producción, y por eso solo sirve
    si se mira **una vez**: en cuanto se usa para elegir algo, deja de ser ciego y se acabaron
    los años limpios del dataset.

    Qué se congela antes de abrirlo
    -------------------------------
    Algoritmo, hiperparámetros, conjunto de variables, protocolo de muestreo y calibrador. Nada
    de eso puede tocarse después de leer este resultado. Si el número decepciona, se reporta el
    número; no se vuelve atrás a retocar el modelo.
    """
    if ctx.modelo is None or ctx.calibrador is None:
        print("  hace falta la etapa `definitivo` en la misma ejecución")
        return pd.DataFrame()

    año = ctx.cfg.año_reservado
    print(f"  ABRIENDO {año}, el año reservado. Esto solo debe hacerse una vez.")
    print(f"  Configuración congelada: {ctx.cfg.modelo} · {len(ctx.variables)} variables · "
          f"entrenamiento {list(ctx.cfg.años_entrenamiento)}")

    # El año reservado queda fuera de `preparar_datos` a propósito, así que su caché de rachas
    # no existe todavía. Se construye aquí, que es el único punto donde se le da uso.
    dias_secos.construir_cache(ctx.contrato, [año])
    marco = ctx.puntuar(ctx.modelo, año, ctx.variables)
    prob = ctx.calibrador.aplicar(marco.score.to_numpy())
    resultado = ctx.metricas_de(marco, prob=prob)

    for clave, valor in resultado.items():
        print(f"    {clave:34s} {valor:.6f}" if isinstance(valor, float)
              else f"    {clave:34s} {valor:,}")

    base = ctx.resultados.get("definitivo")
    if base:
        print(f"\n  recall en {ctx.cfg.año_validacion} (donde se tomaron las decisiones): "
              f"{base[ctx.clave_recall]:.4f}")
        print(f"  recall en {año} (ciego, insesgado):                     "
              f"{resultado[ctx.clave_recall]:.4f}")
        print(f"  diferencia: {resultado[ctx.clave_recall] - base[ctx.clave_recall]:+.4f}")
        print("\n  La cifra del año ciego es la que va en la memoria como rendimiento esperado.")
        print("  La de validación describe el proceso de selección, no el rendimiento.")
    return pd.DataFrame([{"año": año, "condicion": "test ciego", **resultado}])


def etapa_riesgo(ctx: Contexto) -> pd.DataFrame:
    """Traduce la probabilidad a los cuatro niveles y mide su incidencia real.

    La utilidad de los niveles no está en el corte elegido sino en que la incidencia observada
    crezca de forma marcada entre ellos: es lo que permite a quien recibe la alerta confiar en
    que «extremo» significa algo distinto de «alto».
    """
    if ctx.prob_calibrada is None:
        print("  hace falta la etapa `definitivo`")
        return pd.DataFrame()

    y = ctx.validacion.y.to_numpy()
    niveles, umbrales = calibracion.risk_levels(ctx.prob_calibrada)
    nombres = {0: "Bajo", 1: "Moderado", 2: "Alto", 3: "Extremo"}
    filas = []
    for nivel in sorted(np.unique(niveles)):
        mascara = niveles == nivel
        n_celdas = int(mascara.sum())
        n_igniciones = int(y[mascara].sum())
        filas.append({"nivel": nombres[int(nivel)], "celdas_dia": n_celdas,
                      "porcentaje_territorio": n_celdas / len(y), "igniciones": n_igniciones,
                      "incidencia": n_igniciones / n_celdas if n_celdas else 0.0})
    tabla = pd.DataFrame(filas)
    base = tabla.loc[tabla.nivel == "Bajo", "incidencia"]
    if len(base) and base.iloc[0] > 0:
        tabla["veces_sobre_nivel_bajo"] = (tabla.incidencia / base.iloc[0]).round(1)
    print(tabla.to_string(index=False))

    (DIR_TECNICA / "umbrales_riesgo.json").write_text(
        json.dumps({k: float(v) for k, v in umbrales.items()}, indent=2), encoding="utf-8"
    )
    return tabla


# ---------------------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------------------
FUNCIONES = {
    "contrato": etapa_contrato,
    "modelos": etapa_modelos,
    "definitivo": etapa_definitivo,
    "pareado": etapa_pareado,
    "circularidad": etapa_circularidad,
    "importancia": etapa_importancia,
    "espacial": etapa_espacial,
    "semilla": etapa_semilla,
    "fwi": etapa_fwi,
    "errores": etapa_errores,
    "fiabilidad": etapa_fiabilidad,
    "target": etapa_target,
    "riesgo": etapa_riesgo,
    "ciego": etapa_ciego,
}


def ejecutar(cfg: Configuracion, etapas: list[str], rehacer: bool, ruta_fwi: Path) -> None:
    print("=" * 88)
    print("PIPELINE DEFINITIVO — modelado consolidado sobre el dataset EGIF")
    print(f"Entrenamiento {list(cfg.años_entrenamiento)} · calibración {cfg.año_calibracion} · "
          f"validación {cfg.año_validacion} · {cfg.año_reservado} sin tocar")
    print(f"Etapas: {', '.join(etapas)}")
    print("=" * 88)

    DIR_TECNICA.mkdir(parents=True, exist_ok=True)
    ctx = Contexto(cfg)
    arranque = time.time()

    for i, nombre in enumerate(etapas, start=1):
        destino = DIR_TECNICA / f"pipeline_{nombre}.csv"
        print(f"\n{'─' * 88}\n[{i}/{len(etapas)}] {nombre.upper()}\n{'─' * 88}")
        if destino.exists() and not rehacer and nombre not in ("definitivo",):
            print(f"  ya existe {destino.name}; se reutiliza (--rehacer para recalcular)")
            continue

        inicio = time.time()
        tabla = (FUNCIONES[nombre](ctx, ruta_fwi) if nombre == "fwi"
                 else FUNCIONES[nombre](ctx))
        if tabla is not None and len(tabla):
            tabla.to_csv(destino, index=False)
            print(f"\n  → {destino.name}  ({(time.time() - inicio) / 60:.1f} min)")

    (DIR_TECNICA / "pipeline_definitivo_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{'=' * 88}")
    print(f"Terminado en {(time.time() - arranque) / 60:.1f} minutos.")
    print(f"Tablas en {DIR_TECNICA}")
    print(f"El año {cfg.año_reservado} no se ha abierto en ningún momento.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--etapas", default="all",
                   help=f"lista separada por comas. Disponibles: {', '.join(ETAPAS)}")
    p.add_argument("--rehacer", action="store_true",
                   help="recalcula las etapas cuyo resultado ya esté en disco")
    p.add_argument("--fwi-ruta", type=Path, default=RUTA_FWI,
                   help=f"carpeta donde vive fwi.py (por defecto {RUTA_FWI})")
    p.add_argument("--abrir-test-ciego", action="store_true",
                   help="evalúa sobre el año reservado. Solo al final, con todo congelado, "
                        "y una única vez en todo el proyecto")
    args = p.parse_args()

    etapas = list(ETAPAS) if args.etapas == "all" else [e.strip() for e in args.etapas.split(",")]
    desconocidas = [e for e in etapas if e not in FUNCIONES]
    if desconocidas:
        raise SystemExit(
            f"Etapas desconocidas: {desconocidas}. Disponibles: {list(ETAPAS)} y `ciego`."
        )

    if args.abrir_test_ciego:
        if "definitivo" not in etapas:
            etapas = ["definitivo", *etapas]
        etapas.append("ciego")
    elif "ciego" in etapas:
        raise SystemExit(
            "La etapa `ciego` abre el año reservado y exige pedirlo a propósito:\n"
            "  python scripts/pipeline_definitivo.py --abrir-test-ciego"
        )

    ejecutar(Configuracion(), etapas, args.rehacer, args.fwi_ruta)


if __name__ == "__main__":
    main()
