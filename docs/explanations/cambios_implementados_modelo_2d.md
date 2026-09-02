# Cambios implementados: modelo 2D EGIF y pipeline operativo

## 1. Objetivo de la integración

El proyecto pasa a distinguir dos problemas diferentes:

1. La previsión meteorológica la proporcionan modelos numéricos externos, principalmente MeteoGalicia WRF.
2. El modelo del proyecto estima la probabilidad de inicio de incendio condicionada a meteorología, terreno, vegetación, actividad humana e historial.

Por tanto, no se entrena un modelo propio que intente extrapolar la temperatura actual. El pipeline consume el forecast disponible, comprueba su calidad, lo transforma al contrato de variables del modelo y ejecuta tres predictores independientes: T+1, T+2 y T+3.

## 2. Materiales incorporados al repositorio

Los artefactos recibidos se han organizado así:

```text
data/external/egif/
├── galicia_1km.nc
├── dataset_2019.parquet
├── dataset_2020.parquet
├── dataset_2021.parquet
├── dataset_2022.parquet
├── dataset_2023.parquet
├── metadata.json
└── variables.md

docs/references/
├── API_MeteoSIX_v5_es.pdf
├── IberFire_2505.00837v2.pdf
├── Datacube_memoria.ipynb
├── modelado_2d_egif.html
└── README.md
```

Los ficheros grandes se han copiado al repositorio de trabajo para que los scripts puedan utilizarlos con rutas coherentes, pero permanecen ignorados por Git. En particular, los cinco Parquet y el NetCDF ocupan varios gigabytes y no deben añadirse mediante `git add`. La documentación y los contratos sí son versionables. Los originales de `Downloads` se conservan como copia de seguridad.

## 3. Contrato canónico de features

Se ha creado el contrato `egif-2d-v1` en `src/features/canonical_contract.py`. Este contrato fija:

- exactamente 50 predictores;
- el nombre y orden de cada predictor;
- `target_ignicion` como etiqueta EGIF;
- la rejilla canónica de 29.601 celdas;
- `Europe/Madrid` para ventanas horarias locales;
- la versión del contrato que debe coincidir entre entrenamiento e inferencia;
- el rechazo explícito de columnas duplicadas o ambiguas, como `precipitation_sum_1d`;
- la conservación del modelo antiguo de 23 variables únicamente como rollback o shadow model.

La validación falla de forma explícita si un dataset o un artefacto de modelo utiliza 47, 50 con nombres distintos, columnas duplicadas o un contrato diferente. Esto evita que entrenamiento e inferencia parezcan funcionar usando variables incompatibles.

## 4. Preparación de la rejilla

El script `scripts/prepare_operational_grid.py` prepara la rejilla operativa a partir de `galicia_1km.nc`. La salida es una tabla espacial de referencia que permite:

- trabajar únicamente con las celdas canónicas de Galicia;
- comprobar que la rejilla tiene las 29.601 celdas esperadas;
- asociar las predicciones meteorológicas a cada celda;
- separar la geometría y las variables estáticas de la lógica de inferencia.

La malla WRF 1 km es la opción preferente. WRF 4 km es un fallback espacial válido, pero la salida queda etiquetada con su resolución y no se presenta como equivalente a 1 km.

## 5. Proveedores meteorológicos

Se normalizó la obtención de forecast mediante una interfaz común `ForecastProvider`. Cada resultado conserva:

```text
provider
model
model_version
grid
issued_at
downloaded_at
valid_start
valid_end
coverage_percentage
missing_variables
missing_hours
quality
source_resolution
```

La prioridad operativa es:

```text
MeteoGalicia WRF 1 km
        ↓ si falla validación
MeteoGalicia WRF 4 km
        ↓ si está habilitado
AEMET municipal degradado
        ↓ como último recurso
último forecast válido marcado como stale
```

El forecast bruto se guarda antes de agregar variables. Las unidades normalizadas son temperatura en °C, humedad relativa en %, precipitación en mm, velocidad del viento en km/h y dirección en grados. `probPrecipitacion` de AEMET nunca se interpreta como milímetros.

La ausencia de horas o variables críticas no se rellena silenciosamente con ceros. El horizonte pasa a `incomplete`, `degraded` o `unavailable` según la fuente y la cobertura. Si se utiliza el último forecast archivado, se conserva su antigüedad y se marca `stale`.

## 6. Construcción de features meteorológicas

`src/features/operational_features.py` construye las features diarias a partir de observaciones disponibles hasta `issue_time` y del forecast correspondiente al día objetivo. Se aplican estas reglas:

- los timestamps se almacenan internamente en UTC;
- la ventana crítica se evalúa entre las 12:00 y las 18:00 de `Europe/Madrid`;
- se calculan máximos, mínimos, medias, precipitación acumulada y VPD;
- se calculan memorias de precipitación y temperatura de 3, 7, 14 y 30 días;
- `consecutive_dry_days` se obtiene de precipitación observada;
- no se utilizan observaciones posteriores a la hora de emisión;
- la ausencia de una variable crítica se propaga como degradación de calidad;
- se conservan las variables estáticas del datacubo y el historial de incendios.

El VPD se deriva de temperatura y humedad relativa. La precipitación cuantitativa debe proceder de una variable de precipitación real; una probabilidad de lluvia no es una cantidad de lluvia.

## 7. Targets por horizonte

El entrenamiento ya no reutiliza una etiqueta genérica para los tres mapas. Para cada fecha base se generan targets independientes:

```text
target_t1 = incendio en issue_date + 1 día
target_t2 = incendio en issue_date + 2 días
target_t3 = incendio en issue_date + 3 días
```

El benchmark con ERA5 se identifica como `era5_perfect`: sirve para medir el techo aproximado con meteorología histórica observada/reanalizada, pero no debe presentarse como rendimiento operativo de un forecast real.

## 8. Entrenamiento y artefactos

`src/models/canonical_training.py` y `scripts/train_egif_operational.py` implementan el entrenamiento operativo:

- lectura por lotes con PyArrow para evitar que el proceso sea terminado por falta de memoria;
- submuestreo determinista de negativos únicamente durante entrenamiento;
- evaluación sobre la población completa disponible;
- separación temporal por años completos;
- entrenamiento independiente de T+1, T+2 y T+3;
- LightGBM como modelo principal;
- XGBoost opcional como challenger;
- calibración independiente por horizonte;
- PR-AUC, Brier score, calibración y métricas con presupuesto espacial.

Los artefactos esperados son:

```text
data/models/forecast_risk_egif_t1.joblib
data/models/forecast_risk_egif_t2.joblib
data/models/forecast_risk_egif_t3.joblib
```

Cada artefacto contiene el modelo, calibrador, horizonte, versión del contrato, versión del dataset, contexto meteorológico, fecha de entrenamiento, semilla, hash del dataset y métricas. El modelo no se vuelve a entrenar durante la ejecución diaria.

## 9. Ejecución diaria

`scripts/run_daily_inference.py` realiza el flujo operativo:

1. carga y valida el estado meteorológico reciente;
2. obtiene o reutiliza el forecast horario;
3. prueba WRF 1 km y después WRF 4 km;
4. usa AEMET sólo si la contingencia está habilitada y configurada;
5. valida horas, variables, unidades y valores ausentes;
6. asigna el forecast a la rejilla canónica;
7. construye features de T+1, T+2 y T+3;
8. carga los tres artefactos serializados;
9. calcula probabilidades calibradas, nivel y percentil de riesgo;
10. guarda resultados, forecast bruto, features, manifest y checksums;
11. deja disponible un health check reproducible.

La inferencia no selecciona una fila histórica como sustituto silencioso del forecast. En modo local se puede fijar una fecha de simulación mediante las variables documentadas en `docs/deployment/simulacion_local_estado_aemet.md`; en producción esa simulación debe estar desactivada.

## 10. Calidad, trazabilidad y fallback

Cada publicación debe poder responder a estas preguntas: qué forecast se utilizó, cuándo se emitió, qué modelo lo generó, qué resolución tenía, qué horas faltaban, qué features se calcularon y con qué versión del modelo se produjo el mapa.

El manifest incluye `issue_time`, `issued_at`, proveedor, modelo, malla, calidad, antigüedad, cobertura, ausencias, `feature_contract_version`, versión de modelo y checksums. El comando `scripts/check_operational_run.py` valida la coherencia entre estos artefactos.

Los estados relevantes son `fresh`, `fresh_fallback`, `fresh_aemet_degraded`, `stale`, `incomplete`, `invalid` y `unavailable`. Un mapa degradado debe ser visible como tal y nunca confundirse con una predicción fresca de WRF 1 km.

## 11. Dashboard

La aplicación Streamlit se ha ajustado para:

- centrar el mapa y el encuadre en Galicia;
- atenuar u ocultar el resto de España;
- mostrar mapas independientes T+1, T+2 y T+3;
- mostrar proveedor, modelo, malla, antigüedad y cobertura;
- advertir ante `degraded`, `stale`, `incomplete` o `unavailable`;
- permitir presupuestos espaciales del 1 %, 5 % y 10 %;
- mostrar modelo y versión del artefacto;
- conservar el modelo antiguo como comparación shadow;
- avisar cuando los rankings de modelos difieren de forma importante;
- evitar que AEMET municipal parezca espacialmente equivalente a WRF 1 km.

## 12. Pruebas implementadas

Se han añadido pruebas para parser y fallback meteorológico, conversión UTC–hora local, ventana 12:00–18:00, VPD, acumulaciones, ausencia de horas o variables, valores `-9999`, ausencia de observaciones futuras, alineación emisión–validez–horizonte, reproducibilidad de la inferencia, separación de targets y validación del contrato.

La suite focalizada del pipeline canónico pasa. Existe además una limitación del entorno Conda actual: una prueba que importa NetCDF falla si `netCDF4` busca `libjpeg.8.dylib` y el entorno sólo tiene `libjpeg.9.dylib`. `environment.yml` declara `libjpeg-turbo`; hay que actualizar o recrear el entorno para resolver esa dependencia.

## 13. Pendientes antes de declarar producción plena

- Archivar forecasts operativos de MeteoGalicia junto con observaciones verificadas.
- Construir un histórico emparejado forecast–observación.
- Evaluar la degradación real por horizonte y por WRF 1 km, WRF 4 km y AEMET.
- Añadir corrección estadística por variable, estación y horizonte sólo cuando existan pares suficientes.
- Promover el modelo 2D EGIF después de una validación temporal completa.
- Configurar en servidor el volumen persistente, timer, bloqueo, rotación de logs, backups y alertas.
- Instalar dependencias del entorno, incluyendo la biblioteca NetCDF/LibJPEG compatible.

## 14. Comandos de referencia

Preparar la rejilla:

```bash
PYTHONPATH=. python scripts/prepare_operational_grid.py \
  --cube data/external/egif/galicia_1km.nc \
  --output data/processed/grid/galicia_grid_1km_egif.parquet
```

Entrenar los tres modelos:

```bash
PYTHONPATH=. python scripts/train_egif_operational.py \
  --dataset-dir data/external/egif \
  --output-dir data/models
```

Ejecutar la inferencia:

```bash
PYTHONPATH=. python scripts/run_daily_inference.py --no-stale
```

Validar la corrida:

```bash
PYTHONPATH=. python scripts/check_operational_run.py
```

La configuración local se toma de `.env` y los secretos no deben escribirse en este documento ni versionarse.
