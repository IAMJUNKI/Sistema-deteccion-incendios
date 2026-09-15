"""Calibración de Platt sobre probabilidades con submuestreo de negativos.

Por qué vive aquí y no en el script que lo usa
----------------------------------------------
El modelo entregado se serializa junto con su calibrador. Si la clase estuviera definida en el
script de entrenamiento, el `pickle` guardaría una referencia a `__main__.CalibradorPlatt` y el
fichero solo podría abrirse ejecutando ese mismo script; cualquier otro programa —el servicio
operativo, un cuaderno de análisis, el tribunal reproduciendo el trabajo— fallaría al cargarlo
con un `AttributeError` desconcertante.

Al residir en un módulo importable, la referencia serializada es estable y el modelo se abre
desde cualquier sitio con `pickle.load`.

Por qué Platt y no isotónica
----------------------------
La regresión isotónica es una función escalonada: asigna valores idénticos a bloques enteros de
observaciones. Sobre diez millones de filas eso genera millones de empates, y cualquier umbral
definido por percentiles —como el del 5 % del territorio en alerta— cae en mitad de un escalón y
se desplaza de forma arbitraria. Fue el origen de una discrepancia de más de seis puntos de
recall entre dos de los análisis del proyecto.

Platt es estrictamente monótona: no introduce empates, el orden de las celdas es idéntico antes
y después de calibrar, y ranking y probabilidad calibrada pasan a ser la misma magnitud.
"""

from __future__ import annotations

import numpy as np

from src.entrenamiento import calibracion


class CalibradorPlatt:
    """Corrección de prior seguida de un ajuste logístico.

    Dos etapas, cada una con su cometido:

    1. **Corrección de prior.** El submuestreo de negativos infla las probabilidades en un
       factor conocido. La corrección de King y Zeng lo deshace de forma exacta, sin aprender
       nada del dato.
    2. **Ajuste de Platt.** Una regresión logística de un solo parámetro sobre el logaritmo de
       las probabilidades corregidas, ajustada en un año que el modelo no ha visto y con su
       prevalencia real intacta.

    Args:
        tasa_negativos: Fracción de negativos conservada al muestrear (por ejemplo 0,01 si se
            guardó uno de cada cien).
    """

    def __init__(self, tasa_negativos: float, max_puntos_ajuste: int = 1_000_000):
        self.tasa_negativos = float(tasa_negativos)
        self.max_puntos_ajuste = max_puntos_ajuste
        self.intercepto: float | None = None
        self.pendiente: float | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def ajustar(self, prob_cruda: np.ndarray, y: np.ndarray, semilla: int = 42) -> CalibradorPlatt:
        """Ajusta Platt sin materializar innecesariamente todo el año de calibración.

        Conserva todas las igniciones y submuestrea únicamente negativos, compensando su
        selección con pesos. La logística de un predictor se resuelve con Newton-Raphson para
        evitar el fallo nativo observado con L-BFGS en este entorno de Windows.
        """
        prob_cruda = np.asarray(prob_cruda)
        y = np.asarray(y)
        idx = np.arange(len(y))
        pesos = None
        if len(y) > self.max_puntos_ajuste:
            positivos = np.flatnonzero(y == 1)
            negativos = np.flatnonzero(y == 0)
            conservar_negativos = max(1, self.max_puntos_ajuste - len(positivos))
            conservar_negativos = min(conservar_negativos, len(negativos))
            seleccion = np.random.default_rng(semilla).choice(
                negativos, size=conservar_negativos, replace=False
            )
            idx = np.concatenate([positivos, seleccion])
            pesos = np.ones(len(idx), dtype="float64")
            pesos[len(positivos):] = len(negativos) / conservar_negativos

        x = self._logit(calibracion.prior_correction(prob_cruda[idx], self.tasa_negativos))
        y_ajuste = y[idx].astype("float64")
        if pesos is None:
            pesos = np.ones(len(idx), dtype="float64")

        media = float(np.average(y_ajuste, weights=pesos))
        media = float(np.clip(media, 1e-9, 1 - 1e-9))
        intercepto = float(np.log(media / (1 - media)))
        pendiente = 1.0
        for _ in range(100):
            z = np.clip(intercepto + pendiente * x, -30.0, 30.0)
            probabilidad = 1.0 / (1.0 + np.exp(-z))
            curvatura = pesos * probabilidad * (1.0 - probabilidad)
            gradiente = np.array([
                np.sum(pesos * (y_ajuste - probabilidad)),
                np.sum(pesos * (y_ajuste - probabilidad) * x),
            ])
            h00 = float(np.sum(curvatura) + 1e-10)
            h01 = float(np.sum(curvatura * x))
            h11 = float(np.sum(curvatura * x * x) + 1e-10)
            determinante = h00 * h11 - h01 * h01
            if determinante <= 1e-20:
                break
            paso_intercepto = (h11 * gradiente[0] - h01 * gradiente[1]) / determinante
            paso_pendiente = (h00 * gradiente[1] - h01 * gradiente[0]) / determinante
            intercepto += float(paso_intercepto)
            pendiente += float(paso_pendiente)
            if max(abs(paso_intercepto), abs(paso_pendiente)) < 1e-8:
                break

        self.intercepto = intercepto
        self.pendiente = pendiente
        return self

    def aplicar(self, prob_cruda: np.ndarray) -> np.ndarray:
        """Convierte la puntuación cruda del modelo en probabilidad calibrada."""
        corregida = calibracion.prior_correction(prob_cruda, self.tasa_negativos)
        if self.intercepto is None or self.pendiente is None:
            return corregida
        z = np.clip(self.intercepto + self.pendiente * self._logit(corregida), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))
