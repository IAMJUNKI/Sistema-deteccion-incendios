---
library_name: lightgbm
tags:
- wildfire-risk
- forest-fire
- galicia
- geospatial
- lightgbm
- tabular-classification
- operational-forecast
---

# Sistema de alerta temprana de incendios forestales en Galicia

Este repositorio de Hugging Face distribuye los modelos y los artefactos
operativos del [Sistema de detección de incendios forestales](https://github.com/IAMJUNKI/Sistema-deteccion-incendios).
El sistema estima, para cada celda de aproximadamente 1 km × 1 km de Galicia,
la probabilidad de que se produzca una ignición en los siguientes horizontes:

- **T+1:** próximas 24 horas;
- **T+2:** próximas 48 horas;
- **T+3:** próximas 72 horas.

El resultado está pensado como apoyo a la priorización preventiva y a la
visualización operativa. No es un sistema certificado de emergencia ni
sustituye las comunicaciones oficiales de protección civil.

## Qué contiene

| Ruta | Contenido |
|---|---|
| `data/models/forecast_risk_egif_48_t1.joblib` | Modelo LightGBM calibrado para T+1 |
| `data/models/forecast_risk_egif_48_t2.joblib` | Modelo LightGBM calibrado para T+2 |
| `data/models/forecast_risk_egif_48_t3.joblib` | Modelo LightGBM calibrado para T+3 |
| `data/models/*.json` | Contrato de variables, versión y metadatos de cada modelo |
| `data/models/active_model_manifest.json` | Manifiesto de la familia de modelos activa |
| `data/processed/grid/galicia_grid_1km_egif.parquet` | Rejilla espacial canónica de Galicia |
| `data/processed/predicciones_operativas.parquet` | Predicciones operativas más recientes |
| `data/processed/predicciones_operativas.manifest.json` | Metadatos y hashes de la predicción publicada |
| `data/processed/state/weather_daily_state.parquet` | Estado meteorológico móvil de los últimos 30 días |

Los ficheros `.joblib` son artefactos tabulares del proyecto y no se cargan
mediante `transformers`. Para usarlos correctamente hay que respetar el
contrato de 48 variables y el preprocesado implementado en el repositorio de
GitHub.

## Contexto de entrenamiento e inferencia

La familia activa es `egif-2d-48-v1` y contiene 48 variables meteorológicas,
topográficas, de vegetación y de exposición humana. Los modelos se entrenaron
con históricos EGIF de 2016–2020, se calibraron con 2021, se validaron con 2022
y se reservaron los incendios de 2023 como test temporal.

El contexto meteorológico del entrenamiento histórico usa ERA5-Land como
benchmark. La inferencia operativa del servidor usa previsiones WRF de
MeteoGalicia, con el proveedor alternativo configurado en producción cuando es
necesario. Por tanto, los resultados publicados diariamente representan el
estado operativo más reciente y no una evaluación histórica perfecta.

## Cómo descargarlo

El repositorio es público, por lo que no hace falta token para descargarlo.
Desde una copia local del proyecto:

```bash
# Solo lo necesario para abrir el Dashboard (descarga ligera)
python scripts/download_artifacts.py \
  --repo-id Junkii/galicia_wildfire_risk \
  --only-dashboard

# Modelos, rejilla, predicciones y estado meteorológico de 30 días
python scripts/download_artifacts.py \
  --repo-id Junkii/galicia_wildfire_risk
```

Después se puede iniciar el Dashboard con:

```bash
python -m streamlit run app.py
```

La guía completa de instalación y el código de inferencia están en el
[repositorio de GitHub](https://github.com/IAMJUNKI/Sistema-deteccion-incendios).

## Actualización de los datos

El servidor de producción mantiene la copia local como fuente operativa.
Después de cada inferencia diaria correcta publica en Hugging Face un snapshot
coherente con las predicciones y el estado meteorológico acumulado. Los
modelos y la rejilla son artefactos estáticos: solo cambian cuando se promociona
una nueva versión de modelos o se ejecuta una publicación completa.

La aplicación local puede sincronizar este repositorio una vez al arrancar el
proceso de Streamlit mediante `HF_AUTO_DOWNLOAD=true`. El Dashboard no hace una
descarga a Hugging Face por cada usuario o por cada visita.

## Limitaciones y uso responsable

- La salida es una estimación probabilística por celda, no una detección directa
  de llamas ni una predicción determinista de un incendio concreto.
- La calidad depende de la previsión meteorológica disponible, la cobertura de
  datos y la fecha del snapshot publicado.
- Las probabilidades no deben interpretarse fuera del contrato de variables y
  del horizonte para el que se entrenó cada modelo.
- Este repositorio no constituye una alerta oficial, recomendación de
  evacuación ni garantía de ausencia de incendios.
