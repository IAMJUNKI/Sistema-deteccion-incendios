# Estado del Arte y Posicionamiento del TFM

## ¿Tiene sentido nuestro proyecto? ¿Ya se ha hecho?

---

## La respuesta corta

**Sí tiene sentido. No, no está hecho exactamente así. Pero debes conocer lo que existe para posicionarte correctamente.**

El campo de la predicción de incendios con ML está activo y hay trabajo serio publicado — ignorarlo sería un error académico. Conocerlo y diferenciarse de él es lo que convierte este TFM en un trabajo defendible y con valor real.

---

## Lo que ya existe

### 1. IberFire — El trabajo más relevante que debes conocer

> *"IberFire — a detailed creation of a spatio-temporal dataset for wildfire risk assessment in Spain"*
> Ercibengoa et al., arXiv:2505.00837 (2025)

**Qué es:** Un datacube espacio-temporal de acceso abierto que cubre toda España peninsular con resolución de **1km × 1km × 1 día** desde 2007 hasta 2024. Disponible en Zenodo con el código completo de construcción en GitHub.

**Qué incluye:** ~120-260 features en 8 categorías: topografía, meteorología (ERA5), vegetación (NDVI), cobertura del suelo (CORINE), actividad humana, historial de incendios, geografía.

**Lo que significa para tu TFM:**

| Aspecto | Implicación |
|---|---|
| Misma resolución (1km × 1día) | Nuestra metodología espacial está validada por trabajo publicado |
| Mismas fuentes (ERA5, FIRMS, CORINE) | Nuestras elecciones de datos son correctas |
| Código abierto en GitHub | Podemos inspeccionar su metodología para comparar |
| Solo dataset, no sistema operativo | **No compite con nuestro objetivo final** |

**La diferencia clave:** IberFire es un *dataset*, no un *sistema de predicción operativo*. Construyen los datos y los publican. Nosotros construimos el pipeline de inferencia diaria que produce alertas reales cada mañana usando MeteoGalicia.

---

### 2. IPIF — El sistema oficial de AEMET (2026)

AEMET ha lanzado en 2026 el **IPIF** (Índice de Peligro de Incendios Forestales), que sustituye al Canadian Fire Weather Index (FWI) tradicional. Incorpora humedad del suelo, estado de la vegetación e imágenes satelitales, con resolución de 1km.

**Lo que significa:** El sistema institucional se está modernizando en la misma dirección que nuestro TFM. Esto **valida la relevancia del problema** — si hasta AEMET lo está haciendo, el problema existe y es importante.

**La diferencia:** El IPIF es un índice meteorológico operado por una institución pública, no un modelo de ML entrenado con datos históricos de incendios reales. No usa XGBoost ni calibración isotónica de umbrales basada en tasas reales de incidencia.

---

### 3. GEFF/EFFIS — El sistema europeo

El **Global ECMWF Fire Forecast (GEFF)** alimenta el **European Forest Fire Information System (EFFIS)** de Copernicus. Opera con resolución de ~10-25km y usa ERA5 como forzamiento.

**La diferencia:** Resolución mucho más gruesa (10-25km vs 1km nuestro), basado en índices físicos (FWI), no en ML entrenado con datos de ignición históricos.

---

### 4. Investigación académica con ML

Existe literatura científica (2020-2025) que aplica XGBoost, Random Forest y CNNs a la predicción de incendios en España. Los trabajos más comunes:

- **Susceptibilidad de incendio** (fire susceptibility): predice qué zonas son estructuralmente propensas. Es estático — no predice cuándo.
- **Predicción de peligro diario**: algunos trabajos en Australia, California y Portugal. En España, escasos y de resolución baja.
- **Detección de incendios activos con CV**: no predicción, sino detección de lo que ya arde.

**Lo que no existe publicado aún para España/Galicia:**
- Un pipeline de extremo a extremo (datos → modelo → dashboard → alerta diaria) completamente operativo con MeteoGalicia para Galicia
- La cuantificación de la degradación de rendimiento al pasar de ERA5 (entrenamiento) a previsión meteorológica regional (producción)
- Calibración de umbrales de riesgo basada en tasas reales de incidencia por zona

---

## Mapa de posicionamiento del TFM

```
                        ACADÉMICO
                            │
    IberFire ────────────── │ ─────── Investigación FWI clásica
    (dataset, sin           │         (GEFF/EFFIS, FWI índices)
     pipeline)              │
                            │
    ────────────────────────┼──────────────────────────────────
                            │
                NUESTRO TFM │               IPIF (AEMET)
                ★           │               (sistema institucional,
                            │                no ML supervisado)
                            │
                        OPERATIVO
```

Estamos en el cuadrante que **más falta hace y menos existe**: sistema operativo completo con ML supervisado, calibrado con datos reales y adaptado a Galicia con MeteoGalicia.

---

## Qué hace que nuestro TFM sea defendible y con valor propio

### 1. El análisis de degradación ERA5 → previsión real

Nadie ha publicado de forma sistemática **cuánto se degrada un modelo de ignición al pasar de reanálisis histórico a previsión operativa**, descompuesto por horizonte temporal (24h vs 48h vs 72h). Esto es una contribución científica genuina y cuantificable.

### 2. MeteoGalicia como fuente de producción

Usar la red de 170 estaciones de MeteoGalicia en lugar de AEMET (35 en Galicia) para el pipeline de inferencia es una decisión de ingeniería que mejora la calidad de las interpolaciones y que no aparece en los trabajos publicados sobre España.

### 3. Calibración de umbrales con tasas reales de incidencia

La mayoría de trabajos publicados reportan AUC-ROC y F1 pero no traducen las probabilidades a niveles de alerta calibrados con datos reales. El proceso de calibración isotónica con tasas de incidencia históricas es una contribución metodológica diferenciadora.

### 4. Pipeline de extremo a extremo funcional

IberFire es un dataset. EFFIS es una infraestructura europea. Nuestro TFM es un sistema completo que puedes encender mañana en un portátil y que produce un mapa de riesgo nuevo cada día. Eso tiene valor real, más allá de lo académico.

### 5. Arquitectura diseñada para escalar

El código paramétrico (la región como argumento de entrada) y el diseño modular hacen que el paso de Galicia a España completa sea una cuestión de infraestructura, no de rediseño. Eso es lo que convierte un TFM en un producto.

---

## Qué deberías citar y cómo posicionarte en la memoria

### Citas obligatorias

- **IberFire** (arXiv:2505.00837): "Nuestro trabajo extiende el enfoque de IberFire hacia un sistema operativo end-to-end, incorporando MeteoGalicia para la inferencia diaria y cuantificando la degradación de rendimiento al sustituir el reanálisis ERA5 por previsiones meteorológicas en tiempo real."
- **EFFIS/GEFF**: "Frente a los sistemas operativos europeos existentes (EFFIS, resolución ~25km), nuestro modelo trabaja a resolución de 1km y está entrenado específicamente con datos de ignición históricos de Galicia, capturando las dinámicas locales del minifundio forestal."
- **IPIF (AEMET, 2026)**: "La reciente transición de AEMET del FWI al IPIF confirma la relevancia y oportunidad del enfoque basado en ML e integración multimodal de datos."

### Frase de posicionamiento para la memoria

> *"A diferencia de los trabajos precedentes que o bien construyen datasets de alta resolución (IberFire) o bien operan sistemas de índices meteorológicos a escala continental (EFFIS/GEFF), este TFM presenta un pipeline completo de extremo a extremo: desde la ingesta de datos históricos y la construcción de negativos difíciles hasta la inferencia diaria con previsiones meteorológicas de MeteoGalicia, la calibración estadística de niveles de riesgo y el despliegue de un dashboard operativo. La cuantificación de la brecha de rendimiento entre el modo histórico (ERA5) y el modo predictivo (MeteoGalicia) constituye la contribución científica central del trabajo."*

---

## Conclusión

| | ¿Ya existe? | Nuestra diferencia |
|---|---|---|
| Dataset 1km España | ✅ IberFire | Nosotros lo construimos para Galicia con MeteoGalicia |
| Sistema operativo ML incendios | ❌ No publicado para España | Nuestra contribución principal |
| Análisis degradación ERA5→previsión | ❌ No publicado | Contribución científica genuina |
| Calibración de umbrales con tasas reales | ⚠️ Raro en la literatura | Diferenciador metodológico |
| Dashboard operativo con SHAP | ❌ No existe para Galicia | Producto final del TFM |

**El TFM tiene sentido. Tiene trabajo previo sobre el que apoyarse, y tiene hueco real donde aportar.**
