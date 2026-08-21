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
    if not 0.0 < neg_sampling_rate <= 1.0:
        raise ValueError(
            f"neg_sampling_rate debe estar en (0, 1], recibido: {neg_sampling_rate}"
        )
    if neg_sampling_rate == 1.0:
        return np.asarray(y_prob, dtype=np.float64)

    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1 - 1e-12)
    odds = (p / (1.0 - p)) * neg_sampling_rate
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
