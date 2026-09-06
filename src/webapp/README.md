# Módulo: webapp — Dashboard e Integración Operativa

**Fase 5 del proyecto.** Dashboard interactivo en Streamlit y pipeline de inferencia diaria alimentado con un proveedor meteorológico configurable: MeteoGalicia (WRF) o AEMET durante las pruebas.

## Responsabilidad

- Construir el dashboard Streamlit con mapas de riesgo interactivos (Folium/PyDeck).
- Implementar filtros por provincia, municipio y horizonte temporal (24h/48h/72h).
- Mostrar panel de interpretabilidad con contribuciones TreeSHAP calculadas
  sobre el mismo artefacto serializado que produce la predicción; si SHAP no
  está disponible, mostrar una advertencia explícita.
- Programar el script de inferencia diaria que descarga previsiones y genera los mapas T+1/T+2/T+3. La ejecución puede ser provisional a las 05:00 y refrescarse tras la publicación del WRF 1 km.
- Documentar y cuantificar la degradación de rendimiento (ERA5 real vs. WRF MeteoGalicia vs. AEMET municipal).

## Entregable

- WebApp desplegada en local o Streamlit Cloud.
- Script de producción para el pipeline de inferencia diaria.
- Análisis comparativo de rendimiento (datos perfectos vs. previsión).

## Dependencias externas

- `FORECAST_PROVIDER` en `.env`: `meteogalicia`, `aemet` o `auto`.
- `METEOGALICIA_API_KEY` o `AEMET_API_KEY`, según el proveedor seleccionado.
- Tres modelos serializados por horizonte en `data/models/`. La familia
  preferente es `forecast_risk_egif_48_t1/t2/t3.joblib`; los artefactos de 50
  variables y legacy permanecen disponibles para rollback/shadow.
- Estado meteorológico reciente con 30 días completos por celda en
  `data/processed/state/weather_daily_state.parquet`.

## Ejecución

Antes de entrenar el modelo operativo, generar la exportación histórica alineada:

```bash
PYTHONPATH=. python scripts/build_operational_benchmark.py \
  --source-dir data/processed/tabular/egif \
  --output-dir data/processed/tabular/egif_operational \
  --years 2016-2023
```

Entrenar los modelos offline (después de construir la salida alineada):

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/processed/tabular/egif_operational \
  --output-dir data/models \
  --feature-contract-version egif-2d-48-v1 \
  --train-years 2019-2020 --calibration-year 2021 \
  --validation-year 2022 --test-years 2023 \
  --experiment-name comparable
```

Generar el forecast operativo y los tres mapas:

```bash
PYTHONPATH=. python scripts/run_daily_inference.py
```

La inferencia intenta WRF 1 km y, si no está disponible o está incompleto,
utiliza WRF 04 km como fallback visible en el dashboard. Si ninguna malla es
válida, reutiliza un archivo anterior como `stale`.

La inferencia debe ir precedida por la actualización del estado reciente:

```bash
PYTHONPATH=. python scripts/update_weather_state.py \
  --input data/processed/observations/weather_daily_latest.parquet
```

El pipeline archiva el forecast normalizado en `data/raw/meteogalicia/` y el
resultado conjunto en `data/processed/predicciones_operativas.parquet`. Si la
descarga falla, reutiliza únicamente un forecast archivado que cubra los tres
días y marca el resultado como `stale`. El resultado incluye un manifiesto con
checksums y puede verificarse con:

Cuando se usa AEMET, el resultado aparece como `fresh_aemet` y el dashboard
advierte que la fuente es municipal y que la expansión diaria a 72 horas no
equivale a la resolución de WRF 1 km. Si se activa el proxy explícito de
precipitación para una prueba, el estado será `fresh_aemet_proxy`.

```bash
PYTHONPATH=. python scripts/check_operational_run.py
```

La documentación detallada está en
`docs/explanations/pipeline_operativo_detallado.md`.

Para desplegar la inferencia y el dashboard como servicios persistentes en un
servidor Linux, consulta `docs/deployment/servidor_produccion.md`.

Para operar desde un ordenador personal sin servidor ni `cron`, consulta
`docs/deployment/ejecucion_local_aemet.md`.

## Solución de errores de dependencias

El dashboard debe ejecutarse con el intérprete del entorno Conda del proyecto.
Si se lanza con el `streamlit` global de Anaconda, puede aparecer
`ModuleNotFoundError: No module named 'folium'` aunque Folium sí esté instalado
en el entorno correcto.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate incendios-forestales
python -c "import folium, streamlit_folium, geopandas; print('Dependencias del dashboard OK')"
python -m streamlit run app.py
```

La forma `python -m streamlit` es intencionada: garantiza que Streamlit y
Folium se cargan desde el mismo Python. Como alternativa, puede utilizarse la
ruta absoluta del entorno:

```bash
/opt/anaconda3/envs/incendios-forestales/bin/python -m streamlit run app.py
```

La comprobación debe mostrar rutas bajo
`/opt/anaconda3/envs/incendios-forestales/`. No es recomendable instalar
Folium en el entorno base para resolver este error, porque el proyecto usa
Python 3.11 y dependencias geoespaciales fijadas en `environment.yml`.
