"""Recalibración de probabilidades tras el submuestreo de negativos.

Entrenar con negativos submuestreados 1:50 infla las probabilidades predichas unas cincuenta
veces. Un modelo así puede anunciar «12 % de probabilidad de incendio» donde la probabilidad
real es del orden de 10⁻³. Sin corregirlo, los umbrales de riesgo Bajo / Moderado / Alto /
Extremo que promete el proyecto no se pueden construir, porque no habría ninguna probabilidad
con significado físico que umbralizar.

La corrección se aplica en dos etapas:

1. **Corrección de prior** (King & Zeng, 2001). Analítica y sin datos: deshace exactamente el
   sesgo introducido por el muestreo, multiplicando las *odds* por la tasa de conservación de
   negativos. Si se conservó una fracción ``r`` de los negativos:

   ``odds_reales = odds_modelo · r``

2. **Regresión isotónica** ajustada sobre el **año de validación completo**, con su prevalencia
   real intacta. Absorbe la miscalibración residual que la corrección analítica no cubre, como
   el exceso de confianza propio de los modelos de boosting.

El orden importa: la isotónica se ajusta *después* de la corrección de prior, sobre
probabilidades que ya están en la escala correcta.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


def prior_correction(y_prob: np.ndarray, neg_sampling_rate: float) -> np.ndarray:
    """Deshace el sesgo de prevalencia introducido por el submuestreo de negativos.

    Args:
        y_prob: Probabilidades tal como las devuelve el modelo entrenado con submuestreo.
        neg_sampling_rate: Fracción de negativos conservada al construir el entrenamiento
            (`neg_sampling_rate` del diccionario que devuelve `build_training_set`).

    Returns:
        Probabilidades trasladadas a la prevalencia real.

    Raises:
        ValueError: Si la tasa no está en (0, 1].
    """
    tasa = np.asarray(neg_sampling_rate, dtype=np.float64)

    if tasa.ndim == 0:
        if not 0.0 < float(tasa) <= 1.0:
            raise ValueError(
                f"neg_sampling_rate debe estar en (0, 1], recibido: {neg_sampling_rate}"
            )
        if float(tasa) == 1.0:
            return np.asarray(y_prob, dtype=np.float64)
    else:
        # Vector de tasas por fila: lo produce el muestreo estratificado o por clusters, donde
        # cada negativo tuvo su propia probabilidad de sobrevivir. Aplicar una tasa global en ese
        # caso descalibra en silencio, que es el fallo que esta rama evita.
        if tasa.shape != np.shape(y_prob):
            raise ValueError(
                f"El vector de tasas debe tener la forma de y_prob: "
                f"{tasa.shape} frente a {np.shape(y_prob)}."
            )
        if not np.all((tasa > 0.0) & (tasa <= 1.0)):
            raise ValueError("Todas las tasas deben estar en (0, 1].")

    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1 - 1e-12)
    odds = (p / (1.0 - p)) * tasa
    return odds / (1.0 + odds)


class ProbabilityCalibrator:
    """Calibrador en dos etapas: corrección de prior más regresión isotónica.

    Se ajusta sobre el año de validación, nunca sobre el de entrenamiento: la isotónica
    aprendida sobre predicciones dentro de muestra reproduce el sobreajuste del modelo y empeora
    la calibración en lugar de mejorarla.

    Attributes:
        neg_sampling_rate: Tasa de conservación de negativos usada en el entrenamiento.
        isotonic: Regresor isotónico ajustado, o `None` si solo se aplicó la corrección de prior.
    """

    def __init__(self, neg_sampling_rate: float, max_fit_points: int = 2_000_000):
        """
        Args:
            neg_sampling_rate: Fracción de negativos conservada al entrenar.
            max_fit_points: Tope de filas para ajustar la isotónica. Por encima se submuestrean
                los negativos y se compensa con pesos, para no ordenar 11 millones de puntos.
        """
        self.neg_sampling_rate = neg_sampling_rate
        self.max_fit_points = max_fit_points
        self.isotonic: Optional[IsotonicRegression] = None

    def fit(self, y_val: np.ndarray, p_val: np.ndarray, seed: int = 42) -> "ProbabilityCalibrator":
        """Ajusta la isotónica sobre el conjunto de validación.

        Args:
            y_val: Etiquetas reales del año de validación (sin submuestrear).
            p_val: Probabilidades crudas del modelo sobre ese año.
            seed: Semilla del submuestreo interno.

        Returns:
            El propio calibrador, ya ajustado.
        """
        y_val = np.asarray(y_val).astype(np.int8)
        p_corr = prior_correction(p_val, self.neg_sampling_rate)

        idx_pos = np.flatnonzero(y_val == 1)
        idx_neg = np.flatnonzero(y_val == 0)

        if len(y_val) > self.max_fit_points:
            # Se conservan todos los positivos y se muestrean negativos; el peso devuelve a la
            # muestra la prevalencia original, de modo que la isotónica se ajusta a la
            # distribución real y no a la de la submuestra.
            rng = np.random.default_rng(seed)
            cupo = max(self.max_fit_points - len(idx_pos), 1)
            elegidos = rng.choice(idx_neg, size=min(cupo, len(idx_neg)), replace=False)
            idx = np.concatenate([idx_pos, elegidos])
            pesos = np.concatenate(
                [np.ones(len(idx_pos)), np.full(len(elegidos), len(idx_neg) / len(elegidos))]
            )
            logger.info(
                "  isotónica ajustada sobre %s filas (todos los positivos + %s negativos ponderados)",
                f"{len(idx):,}",
                f"{len(elegidos):,}",
            )
        else:
            idx = np.arange(len(y_val))
            pesos = np.ones(len(y_val))

        self.isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.isotonic.fit(p_corr[idx], y_val[idx], sample_weight=pesos)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Aplica la calibración a probabilidades crudas del modelo.

        Args:
            y_prob: Probabilidades tal como salen de `predict_proba`.

        Returns:
            Probabilidades calibradas a la prevalencia real.
        """
        p = prior_correction(y_prob, self.neg_sampling_rate)
        return self.isotonic.transform(p) if self.isotonic is not None else p

    def __repr__(self) -> str:
        estado = "ajustado" if self.isotonic is not None else "solo corrección de prior"
        return f"ProbabilityCalibrator(neg_sampling_rate={self.neg_sampling_rate:.6f}, {estado})"


class PlattProbabilityCalibrator:
    """Corrección de prior seguida de calibración logística (Platt scaling).

    Se ajusta sobre un año separado y conserva la prevalencia real mediante
    pesos cuando el año de calibración se reduce para controlar memoria. El
    predictor de la regresión es el logit de la probabilidad corregida, que
    permite que Platt aprenda simultáneamente escala y desplazamiento sin
    perder la corrección de prevalencia introducida por el muestreo.
    """

    def __init__(self, neg_sampling_rate: float, max_fit_points: int = 2_000_000):
        self.neg_sampling_rate = float(neg_sampling_rate)
        self.max_fit_points = int(max_fit_points)
        self.platt: LogisticRegression | None = None

    @staticmethod
    def _logit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
        return np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-values))

    def fit(
        self,
        y_calibration: np.ndarray,
        p_calibration: np.ndarray,
        seed: int = 42,
    ) -> "PlattProbabilityCalibrator":
        y = np.asarray(y_calibration).astype(np.int8)
        p = prior_correction(p_calibration, self.neg_sampling_rate)
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        if len(positives) == 0 or len(negatives) == 0:
            self.platt = None
            return self

        if len(y) > self.max_fit_points:
            rng = np.random.default_rng(seed)
            negative_count = min(self.max_fit_points - len(positives), len(negatives))
            negative_count = max(negative_count, 1)
            selected_negatives = rng.choice(negatives, size=negative_count, replace=False)
            indices = np.concatenate([positives, selected_negatives])
            weights = np.concatenate(
                [np.ones(len(positives)), np.full(len(selected_negatives), len(negatives) / negative_count)]
            )
        else:
            indices = np.arange(len(y))
            weights = np.ones(len(y))

        self.platt = LogisticRegression(
            solver="lbfgs",
            random_state=seed,
            max_iter=200,
        )
        self.platt.fit(self._logit(p[indices]).reshape(-1, 1), y[indices], sample_weight=weights)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        p = prior_correction(y_prob, self.neg_sampling_rate)
        if self.platt is None:
            return p
        return self.platt.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]

    def __repr__(self) -> str:
        state = "ajustado" if self.platt is not None else "solo corrección de prior"
        return f"PlattProbabilityCalibrator(neg_sampling_rate={self.neg_sampling_rate:.6f}, {state})"


def risk_levels(
    p_calibrated: np.ndarray, quantiles: tuple[float, ...] = (0.90, 0.98, 0.995)
) -> tuple[np.ndarray, dict[str, float]]:
    """Traduce probabilidades calibradas a los cuatro niveles de riesgo del proyecto.

    Los cortes se fijan por cuantiles de la distribución de riesgo, no por valores absolutos de
    probabilidad: a esta prevalencia un umbral fijo como 0,5 no se alcanzaría jamás. Definirlos
    por cuantiles hace además que el volumen de alertas sea predecible para quien las recibe.

    Args:
        p_calibrated: Probabilidades ya calibradas.
        quantiles: Cuantiles que separan Bajo|Moderado, Moderado|Alto y Alto|Extremo.

    Returns:
        Tupla con el array de niveles (0=Bajo, 1=Moderado, 2=Alto, 3=Extremo) y el diccionario
        de umbrales aplicados.
    """
    cortes = np.quantile(p_calibrated, quantiles)
    niveles = np.digitize(p_calibrated, cortes)
    umbrales = {
        "moderado": float(cortes[0]),
        "alto": float(cortes[1]),
        "extremo": float(cortes[2]),
    }
    return niveles.astype(np.int8), umbrales
