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
from sklearn.linear_model import LogisticRegression

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

    def __init__(self, tasa_negativos: float):
        self.tasa_negativos = float(tasa_negativos)
        self.logistica: LogisticRegression | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def ajustar(self, prob_cruda: np.ndarray, y: np.ndarray) -> "CalibradorPlatt":
        """Ajusta el calibrador sobre un año no visto."""
        corregida = calibracion.prior_correction(prob_cruda, self.tasa_negativos)
        self.logistica = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.logistica.fit(self._logit(corregida).reshape(-1, 1), y)
        return self

    def aplicar(self, prob_cruda: np.ndarray) -> np.ndarray:
        """Convierte la puntuación cruda del modelo en probabilidad calibrada."""
        corregida = calibracion.prior_correction(prob_cruda, self.tasa_negativos)
        if self.logistica is None:
            return corregida
        return self.logistica.predict_proba(self._logit(corregida).reshape(-1, 1))[:, 1]
