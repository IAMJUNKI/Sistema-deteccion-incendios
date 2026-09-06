# Informe de cambios, resultados y despliegue — release v0.4.0

**Fecha del informe:** 6 de septiembre de 2026

**Release de código:** `v0.4.0` (`0d06fac`)

**Rama de referencia:** `main`

**Test ciego documentado:** año 2023, población completa

## 1. Resumen ejecutivo

El proyecto ha pasado de un flujo principalmente retrospectivo a una
arquitectura que separa tres responsabilidades:

1. construir un dataset histórico reproducible;
2. entrenar modelos de riesgo compatibles con la semántica de producción;
3. ejecutar cada día una inferencia a partir del forecast meteorológico
   disponible.

La familia seleccionada para la prueba operativa es `egif-2d-48`, entrenada
con 48 variables. Esta familia está formada por **tres artefactos independientes**:

```text
forecast_risk_egif_48_t1.joblib  → incendio al día siguiente
forecast_risk_egif_48_t2.joblib  → incendio dentro de dos días
forecast_risk_egif_48_t3.joblib  → incendio dentro de tres días
```

Por tanto, la descripción correcta no es que un único modelo produzca tres
mapas, sino que existe un modelo y un calibrador específico para cada
horizonte. Esto permite que T+1, T+2 y T+3 tengan contratos, etiquetas,
métricas y comportamiento propios.

El FWI de Copernicus/CEMS no es otro modelo de aprendizaje automático. Se usa
como **baseline físico**, es decir, como referencia independiente para saber si
el ranking aprendido aporta valor adicional.

La evaluación disponible es prometedora para el ranking preventivo, pero no
autoriza todavía a presentar el resultado como rendimiento final en producción:

- el test utiliza ERA5-Land como meteorología histórica conocida
  (`era5_perfect_benchmark`);
- todavía no existe un histórico suficiente de forecasts MeteoGalicia
  emparejados con observaciones;
- las capas estáticas están marcadas como `provisional_import`;
- la probabilidad absoluta necesita volver a calibrarse cuando exista
  histórico real de forecasts.

La versión actual debe publicarse, por tanto, como **piloto operativo
provisional**, manteniendo el modelo anterior en modo shadow.

## 2. Estado de Git y alcance de este informe

La release `v0.4.0` contiene los cambios funcionales de alineación del forecast,
dataset operativo, modelos de 48 variables, evaluación y despliegue. El commit
más reciente es:

```text
0d06fac fix(inference): align archived forecasts with canonical grid
```

En el momento de redactar este documento existen modificaciones locales en
varios componentes del dashboard (`src/webapp/components/` y sus pruebas) que
no forman parte de `v0.4.0` hasta que se revisen y se haga commit. Este informe
describe exclusivamente el código y los artefactos versionados en la release,
además de los resultados guardados en el entorno local.

## 3. Cambios funcionales acumulados

La siguiente tabla agrupa los commits con cambios funcionales relevantes. Las
subidas de ficheros binarios y documentos auxiliares se incluyen en la parte
de referencias, pero no se confunden con una nueva lógica de modelado.

| Commit | Cambio principal | Motivo |
|---|---|---|
| `7089f4a` | Mejora del workflow de construcción del datacubo, ERA5 y FWI | Hacer reproducible la generación de datos históricos y más robusta la ingesta meteorológica |
| `a4549c8` | Ampliación del histórico EGIF y corrección de rachas secas | Incorporar más años y evitar una definición inconsistente de sequía |
| `e42aaa9` | Uso de las rachas secas del datacubo final durante el entrenamiento | Evitar que entrenamiento y dataset utilicen dos implementaciones distintas |
| `d4eb8fb` | Escritura de meteorología por bloques temporales y nuevas pruebas | Reducir memoria y mejorar la duración de la ingesta |
| `26ccdb6` | Cliente MeteoGalicia/AEMET, pipeline operativo, contratos, dashboard y despliegue | Convertir el prototipo en un sistema ejecutable en servidor |
| `d883aec` | Componentes de días secos y calibración de Platt | Preparar una calibración más coherente con el modelo documentado |
| `49d6e71` | Evolución del pipeline definitivo | Consolidar la generación de variables y la evaluación del modelo anterior |
| `c26c34c` | Revisión de la documentación del modelo final | Separar hipótesis, resultados y limitaciones metodológicas |
| `1106ec3` | Actualización de evidencias técnicas CSV/JSON | Conservar métricas, fiabilidad, importancia y errores como artefactos auditables |
| `f523f48` | Dataset temporalmente alineado, contrato de 48 variables, entrenamiento de familias, comparación y promoción | Resolver la diferencia entre el modelo documentado y el modelo operativo |
| `0d06fac` | Reasignación de forecasts archivados a la rejilla canónica y deduplicación | Permitir utilizar forecasts antiguos con 30.697 celdas junto a la rejilla EGIF de 29.601 |

También se incorporaron al repositorio las referencias de MeteoSIX v5, IberFire,
el notebook del datacubo y el HTML de modelado 2D. Los datos grandes siguen
siendo artefactos locales o de servidor y no deben subirse a GitHub.

## 4. Datos y separación de capas

### 4.1 Datacubo canónico

El datacubo histórico situado en:

```text
data/processed/tabular/egif/
```

se conserva sin sobrescribirlo. Es la referencia retrospectiva para auditoría,
comparación con resultados anteriores y reconstrucción de la procedencia de
los datos.

### 4.2 Dataset operativo alineado

Se creó una segunda salida:

```text
data/processed/tabular/egif_operational/
```

Esta salida conserva una fila por celda y día y recalcula las memorias
meteorológicas con una frontera temporal estricta:

```text
meteorología del día objetivo
+
memoria meteorológica hasta target_date - 1 día
```

El resultado exportado cubre los años 2016–2023, aproximadamente 86,5 millones
de filas y 29.601 celdas activas. El 1 de enero de 2016 no dispone de contexto
histórico anterior dentro de los datos entregados; esas filas se conservan para
auditoría, pero el entrenador las excluye cuando no tienen predictores válidos.

Las memorias recalculadas son:

```text
precipitation_sum_3d
precipitation_sum_7d
precipitation_sum_14d
precipitation_sum_30d
temperature_mean_7d
relative_humidity_mean_7d
relative_humidity_mean_14d
wind_speed_mean_7d
consecutive_dry_days
```

El proceso se ejecuta por particiones y mantiene una cola de hasta 30 días por
celda. Así puede cruzar diciembre y enero sin cargar todo el datacubo en
memoria.

El `metadata.json` del dataset registra, entre otros campos:

```text
feature_contract_version
alignment_version
training_temporal_semantics
source_datacube
parent_dataset_hash
years
row_counts
active_cells
dropped_rows
static_layer_quality
created_at
```

La metadata es una barrera contra el error de volver a entrenar por accidente
con acumulaciones inclusivas del datacubo canónico.

## 5. Contrato de 48 variables

La familia seleccionada usa:

```text
feature_contract_version = egif-2d-48-v1
```

El contrato contiene exactamente 48 predictores. Se eliminaron del contrato
canónico de 50 variables:

```text
precipitation_sum
consecutive_dry_days
```

La razón es que estas variables se conservan para auditoría y baseline, pero
no forman parte de la familia experimental de 48 variables.

El FWI tampoco es predictor del modelo 48. Se conserva como baseline físico.
Tampoco entran en la matriz de predictores `cell_id`, `fecha`, coordenadas,
geometrías, etiquetas, identificadores ni variables de calendario.

Las 48 columnas, en el orden congelado, son:

```text
agriculture
artificial
aspect_000_045_fraction
aspect_045_090_fraction
aspect_090_135_fraction
aspect_135_180_fraction
aspect_180_225_fraction
aspect_225_270_fraction
aspect_270_315_fraction
aspect_315_360_fraction
broadleaf_forest
building_area_fraction
coniferous_forest
elevation_mean
elevation_std
mixed_forest
open_spaces
precipitation_sum_14d
precipitation_sum_30d
precipitation_sum_3d
precipitation_sum_7d
relative_humidity_mean
relative_humidity_mean_14d
relative_humidity_mean_7d
relative_humidity_min
relative_humidity_min_12_18h
residential_area_fraction
road_length_km
road_length_local_km
road_length_main_km
road_length_other_km
road_length_track_km
scrub
slope_mean
slope_std
temperature_max
temperature_max_12_18h
temperature_mean
temperature_mean_7d
temperature_min
vpd_max_12_18h
vpd_mean
water
wetlands
wind_speed_max
wind_speed_max_12_18h
wind_speed_mean
wind_speed_mean_7d
```

La carga del modelo comprueba el nombre, orden, versión y número de columnas.
Un modelo de 48 variables no puede mezclarse con el contrato de 50 variables.

## 6. Semántica temporal corregida

Para cada fecha de emisión y horizonte se define:

```text
issue_time
  → información observada disponible hasta issue_time
  → forecast del día objetivo
  → features del horizonte
  → target_ignicion del día objetivo
```

Las etiquetas son independientes:

```text
T+1: target_ignicion en issue_date + 1 día
T+2: target_ignicion en issue_date + 2 días
T+3: target_ignicion en issue_date + 3 días
```

Las memorias no pueden utilizar lluvia, temperatura, humedad o viento del propio
día objetivo cuando esas observaciones todavía no estaban disponibles. En
inferencia, el día objetivo procede del forecast y los días anteriores se
construyen con la combinación de estado observado y días previstos intermedios.

La evaluación histórica utiliza ERA5-Land porque no existe todavía un archivo
histórico de forecasts MeteoGalicia. Esto se etiqueta como:

```text
era5_perfect_benchmark
```

Es un benchmark retrospectivo con meteorología futura conocida. No es una
medición del error real que tendrá WRF cuando el sistema funcione cada día.

## 7. Familias y artefactos de modelos

Se generaron varias familias para que la comparación sea justa:

| Familia | Contrato | Semántica | Uso |
|---|---|---|---|
| EGIF 48 comparable | `egif-2d-48-v1` | alineada | Entrenamiento 2019–2020 |
| EGIF 48 ampliado | `egif-2d-48-v1` | alineada | Entrenamiento 2016–2020; familia candidata |
| EGIF 50 control | `egif-2d-v1` | alineada | Control no publicable automáticamente |
| EGIF 50 actual | canónico | semántica anterior | Rollback/shadow |
| FWI CEMS | baseline | índice físico | Referencia externa |

Las particiones de los experimentos de modelos aprendidos son:

```text
Entrenamiento: 2019–2020 o 2016–2020
Calibración:   2021
Validación:    2022
Test ciego:    2023
```

Se utiliza LightGBM como modelo principal, submuestreo determinista de
negativos únicamente durante el entrenamiento y evaluación sobre la población
completa. El ratio inicial de negativos fue 1:100.

Cada horizonte se entrena y calibra por separado con:

```text
corrección de prior por submuestreo + Platt scaling
```

Los metadatos de cada `.joblib` incluyen horizonte, contrato, alineación,
dataset, hash, años de entrenamiento, semilla, fuente meteorológica,
calibración, versión del modelo y métricas.

## 8. Resultados del test ciego 2023

El informe se encuentra en:

```text
data/models/evaluation/model_family_comparison.json
```

La evaluación utiliza:

```text
10.804.365 filas
530 positivos
prevalencia = 0,004905 %
```

La prevalencia es extremadamente baja. Por ello, un PR-AUC de `0,0007` no se
debe leer como un porcentaje de aciertos; debe compararse con la prevalencia y
con un presupuesto espacial de vigilancia.

### 8.1 Familia EGIF 48 ampliada frente a FWI

| Horizonte | EGIF 48 PR-AUC | FWI PR-AUC | EGIF 48 ROC-AUC | FWI ROC-AUC | EGIF 48 recall@FPR5 | FWI recall@FPR5 | EGIF 48 recall@top 1 % | FWI recall@top 1 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T+1 | 0,000707 | 0,000154 | 0,8713 | 0,7720 | 44,15 % | 19,62 % | 9,62 % | 2,83 % |
| T+2 | 0,000555 | 0,000154 | 0,8695 | 0,7720 | 41,13 % | 19,62 % | 11,32 % | 2,83 % |
| T+3 | 0,000664 | 0,000154 | 0,8720 | 0,7720 | 42,08 % | 19,62 % | 9,81 % | 2,83 % |

En términos relativos, EGIF 48 ampliado obtiene:

- entre 3,59 y 4,58 veces el PR-AUC del FWI;
- entre 2,10 y 2,25 veces el recall del FWI con FPR fija del 5 %;
- entre 3,40 y 4,00 veces el recall del FWI al inspeccionar el 1 % superior
  de celdas cada día;
- una mejora de aproximadamente 0,097–0,100 puntos de ROC-AUC.

Este resultado indica que la familia aprendida ordena mejor las celdas para una
estrategia preventiva. No significa que la probabilidad mostrada sea cuatro
veces una probabilidad real ni que el modelo explique causalmente los
incendios.

El FWI no tiene Brier score en este informe porque se trata de un índice, no de
una probabilidad calibrada. El Brier de EGIF 48 se sitúa alrededor de
`4,92 × 10⁻⁵`, muy cerca de la escala de la prevalencia climatológica del test;
por ello, el principal resultado actual está en el ranking, no en la
interpretación literal de la probabilidad absoluta.

### 8.2 Comparación con el control alineado de 50 variables

El control de 50 variables permite comparar familias con la misma semántica
temporal y la misma población:

| Horizonte | PR-AUC EGIF 48 ampliado | PR-AUC control 50 | Diferencia relativa | Recall top 1 %: EGIF 48 | Recall top 1 %: control 50 |
|---|---:|---:|---:|---:|---:|
| T+1 | 0,000707 | 0,000480 | +47,3 % | 9,62 % | 7,74 % |
| T+2 | 0,000555 | 0,000507 | +9,4 % | 11,32 % | 6,79 % |
| T+3 | 0,000664 | 0,000539 | +23,0 % | 9,81 % | 7,74 % |

Con FPR fija del 5 %, EGIF 48 mejora al control 50 en T+1 y T+3, pero queda
aproximadamente 0,94 puntos porcentuales por debajo del control en T+2.
Esto es una razón para conservar el control y no afirmar que las 48 variables
son superiores en cualquier criterio operativo.

### 8.3 Comparación con el modelo 50 actual

El modelo 50 actual aparece como `canonical-egif-2d` y fue evaluado con otra
semántica temporal. Esa comparación es útil para conocer la transición, pero no
es tan limpia como la comparación contra el control 50 alineado.

Su PR-AUC baja de `0,000509` en T+1 a `0,000378` en T+3, mientras que EGIF 48
ampliado se mantiene entre `0,000555` y `0,000707`. Esto sugiere que la nueva
semántica temporal y el entrenamiento ampliado son más importantes que añadir
simplemente dos columnas.

## 9. Comparación gráfica por horizonte

La caja de bigotes no era la mejor representación para estos datos. Cada
familia solo tiene tres valores —T+1, T+2 y T+3—, por lo que una caja podía
parecer una distribución estadística cuando realmente solo resumía tres
horizontes. Se sustituyó por un gráfico de puntos conectados:

![Comparación por horizonte](../technical/model_family_horizons.png)

La lectura es directa:

- el eje horizontal indica el horizonte: `T+1`, `T+2` o `T+3`;
- cada línea de color representa una familia de modelos;
- cada punto es el resultado de esa familia en el horizonte indicado;
- la línea permite ver si la métrica mejora o empeora al alejarse en el tiempo;
- el FWI aparece prácticamente horizontal porque es el mismo índice diario usado
  como baseline.

Esta figura no representa intervalos de confianza. El JSON conserva intervalos
bootstrap del 90 % para recall, pero no las réplicas completas necesarias para
dibujar una distribución de incertidumbre.

La figura permite ver tres hechos:

1. EGIF 48 ampliado queda por encima del FWI y de las familias 50 en las cuatro
   métricas mostradas.
2. El FWI es prácticamente plano entre horizontes porque es el mismo índice
   diario usado como referencia.
3. La familia EGIF 50 actual pierde rendimiento con más claridad en T+3 que el
   modelo 48 ampliado, aunque la comparación tiene semánticas distintas.

La figura se puede regenerar desde el informe, sin acceder a los Parquet:

```bash
MPLCONFIGDIR=/tmp/fire-risk-mpl \
/opt/anaconda3/envs/incendios-forestales/bin/python \
scripts/plot_model_family_horizons.py
```

La explicación ilustrada de estas figuras, con una tabla de resultados y las
limitaciones para la memoria del TFM, está separada en
[informe_graficos_resultados.md](informe_graficos_resultados.md). También se
puede generar el PDF de presentación con `scripts/build_figure_report_pdf.py`.

## 10. Interpretabilidad del modelo

Para la memoria conviene acompañar el resultado global con una figura que
explique qué variables utiliza el modelo. Se ha generado la importancia
`gain` interna del LightGBM para las 12 variables con mayor importancia media
entre los tres horizontes:

![Importancia de variables EGIF 48](../technical/egif48_feature_importance.png)

La lectura principal es:

- `vpd_mean` es la variable con mayor importancia interna en los tres modelos;
- las memorias de precipitación, especialmente `precipitation_sum_3d` y
  `precipitation_sum_30d`, aparecen de forma consistente;
- la humedad relativa, la elevación y la longitud de carreteras locales
  también contribuyen al ranking;
- el patrón es parecido entre T+1, T+2 y T+3, aunque cambian las magnitudes.

Esta figura sirve para explicar el modelo, pero no demuestra causalidad. La
importancia gain mide cómo se utilizan las variables dentro de los árboles y
puede verse afectada por correlaciones entre predictores. Para una versión
científica final conviene complementarla con importancia por permutación,
SHAP y ablaciones por grupos de variables.

Se regenera con:

```bash
MPLCONFIGDIR=/tmp/fire-risk-mpl \
/opt/anaconda3/envs/incendios-forestales/bin/python \
scripts/plot_egif48_feature_importance.py
```

### 10.1. Figuras recomendadas para la memoria

Estas dos figuras son suficientes para explicar el resultado actual, pero la
memoria quedará más completa si se añaden figuras que respondan a preguntas
operativas concretas. La selección recomendada es:

| Figura | Pregunta que responde | Estado | Precaución |
|---|---|---|---|
| Perfil por horizonte | ¿Cómo cambia el rendimiento entre T+1, T+2 y T+3? | Disponible | Son tres puntos por familia, no una distribución estadística. |
| Importancia de variables | ¿Qué predictores utiliza internamente el modelo? | Disponible | No debe interpretarse como causalidad. |
| Recall frente a presupuesto espacial | ¿Cuántas igniciones cubrimos vigilando el 1 %, 5 % o 10 % de las celdas? | Siguiente figura | Hay que calcular y guardar explícitamente esos umbrales para cada día. |
| Diagrama de fiabilidad | ¿Las probabilidades 0.01, 0.05 o 0.20 tienen una frecuencia observada compatible? | Pendiente | Debe separarse por horizonte y validarse con forecasts archivados. |
| Mapa de un día de test | ¿Dónde se concentra el riesgo predicho y cómo se compara con las igniciones observadas? | Pendiente | Debe mostrar la misma rejilla, fecha y leyenda para evitar comparaciones engañosas. |
| Arquitectura del pipeline | ¿Cómo pasan los datos de la fuente meteorológica al mapa? | Parcialmente disponible | Puede usarse el diagrama Mermaid de la sección de despliegue. |

La figura de presupuesto espacial será especialmente relevante para la
movilización de medios: PR-AUC resume el ranking global, pero no dice cuántas
igniciones se cubren con una capacidad limitada de vigilancia. El gráfico debe
mostrar, para cada horizonte, la curva `presupuesto de celdas (%)` frente a
`recall`, incluyendo al menos 1 %, 5 % y 10 %, y comparando EGIF 48, el control
EGIF 50 y FWI.

El diagrama de fiabilidad debe esperar a que se disponga de pares
`forecast-observación` de MeteoGalicia. Las probabilidades evaluadas con
ERA5-perfect sirven como benchmark retrospectivo, pero no permiten afirmar que
la probabilidad esté calibrada para un forecast real. Hasta entonces, la
memoria debe presentar la calibración como una validación pendiente y no como
una propiedad ya demostrada en producción.

Por tanto, el conjunto mínimo que se puede enseñar ahora es el perfil por
horizonte y la importancia de variables. El conjunto final recomendado para la
defensa es ese par más la curva de presupuesto, un mapa espacial de un día de
test y el diagrama de fiabilidad cuando exista histórico meteorológico
emparejado.

## 11. Qué significa que existan tres modelos

El horizonte cambia la decisión operativa:

```text
T+1 → decidir vigilancia o movilización para mañana
T+2 → planificar recursos con más anticipación
T+3 → detectar zonas persistentes de vulnerabilidad
```

Un solo modelo reutilizado para los tres días obligaría a suponer que el mismo
patrón de relación entre meteorología, terreno e ignición es válido para todas
las distancias temporales. Los tres artefactos permiten:

- etiquetas diferentes;
- meteorología prevista diferente;
- memoria de días intermedios diferente;
- calibradores distintos;
- umbrales o presupuestos operativos distintos;
- medir la degradación específica de cada horizonte.

En el benchmark ERA5-perfect la meteorología futura se conoce exactamente, por
lo que es normal que los resultados de T+1, T+2 y T+3 sean parecidos. La
degradación real solo aparecerá al evaluar forecasts MeteoGalicia archivados
contra las observaciones que finalmente ocurrieron.

## 12. Inferencia operativa implementada

El flujo diario es:

```text
estado meteorológico reciente
        ↓
MeteoGalicia WRF 1 km
        ↓ si falla validación
MeteoGalicia WRF 04 km
        ↓ si está habilitado
AEMET degradado
        ↓ si no hay forecast fresco y se permite
último forecast válido marcado como stale
        ↓
asignación a 29.601 celdas EGIF
        ↓
features T+1/T+2/T+3
        ↓
tres modelos EGIF 48 + modelo 50 shadow
        ↓
predicciones, niveles, percentiles y manifest
```

El código ya no sustituye silenciosamente el forecast por una fila histórica.
Además, si un forecast archivado contiene una rejilla antigua de 30.697 celdas
y conserva sus coordenadas, `run_daily_inference.py` lo reasigna a la rejilla
canónica de 29.601 celdas y elimina duplicados por coordenada y hora. Si el
forecast no tiene coordenadas suficientes para hacerlo, la inferencia falla de
forma explícita.

Una ejecución local validada produjo:

```text
calidad del forecast: fresh
malla: 1km
cobertura: 100 %
celdas: 29.601
filas de salida: 88.803 = 29.601 × 3 horizontes
```

Los artefactos de salida son:

```text
data/raw/meteogalicia/                    forecast horario bruto
data/processed/predicciones_operativas.features.parquet
data/processed/predicciones_operativas.parquet
data/processed/predicciones_operativas.manifest.json
data/models/active_model_manifest.json
```

El manifest conserva proveedor, emisión, antigüedad, resolución, cobertura,
calidad, contrato, versión del modelo y hashes. El dashboard puede advertir si
el resultado es `fresh`, `fresh_fallback`, `fresh_aemet_degraded`, `stale`,
`incomplete` o `unavailable`.

## 13. Despliegue en servidor

El despliegue separa código, artefactos pesados y secretos:

```mermaid
flowchart LR
    A[Ordenador de operación] -->|git push + tag| B[GitHub]
    B -->|Deploy Key read-only| C[release en servidor]
    A -->|rsync incremental| D[/srv/fire-risk/data persistente]
    D --> E[modelos y rejilla]
    C --> F[systemd]
    E --> F
    F --> G[inferencia y health check]
    G --> H[predicciones + manifest]
    H --> I[Streamlit localhost:8501]
    I --> J[Nginx HTTPS público]
```

### 13.1 Qué va a GitHub

```text
código Python
tests
documentación
plantillas systemd
environment.yml
.env.example
scripts de deploy y rollback
```

### 13.2 Qué se transfiere mediante rsync

```text
data/models/
data/processed/grid/galicia_grid_1km_egif.parquet
data/external/egif/          solo si se necesita para entrenamiento/auditoría
```

No se deben transferir desde el portátil al servidor los estados meteorológicos
de producción, forecasts generados por el servidor ni las predicciones
operativas. Esos artefactos pertenecen al almacenamiento persistente del
servidor.

### 13.3 Credenciales

Se mantienen tres credenciales separadas:

| Credencial | Dirección | Ubicación |
|---|---|---|
| Deploy Key GitHub | servidor → GitHub | `/srv/fire-risk/.ssh/` |
| Clave de transferencia | ordenador → servidor | solo en el ordenador de operación |
| Claves API | servidor → proveedores | `/srv/fire-risk/config/.env` |

La clave de MeteoGalicia nunca se introduce en GitHub ni en los comandos que se
compartan. El `.env` del servidor debe tener permisos `600`.

### 13.4 Secuencia de publicación

Después de revisar y hacer commit de los cambios de código:

```bash
git push origin main
git tag -a v0.4.0 -m "EGIF 48 operational pilot"
git push origin v0.4.0
```

El `deploy/deploy.env` local debe apuntar al tag revisado:

```text
DEPLOY_HOST=62.171.135.241
DEPLOY_PORT=22
DEPLOY_USER=root
DEPLOY_PATH=/srv/fire-risk
DEPLOY_SERVICE_USER=fire-risk
DEPLOY_REMOTE_BASE=/srv/fire-risk/data
DEPLOY_REF=v0.4.0
DEPLOY_SSH_KEY=/Users/junki/.ssh/fire-risk-server
DEPLOY_RUN_TESTS=true
DEPLOY_RESTART_DASHBOARD=true
DEPLOY_RUN_CHECK=true
DEPLOY_REQUIRE_OUTPUT=false
DEPLOY_USE_SUDO=true
```

Primero se valida sin modificar el servidor:

```bash
make deploy-dry-run
```

Después se suben solo los artefactos necesarios para inferencia:

```bash
make deploy-runtime
```

Este comando usa `rsync` incremental, conserva parciales y no borra por defecto
artefactos remotos. Una transferencia interrumpida puede repetirse y continuará
desde los ficheros parciales.

Finalmente se publica el código:

```bash
make deploy
```

El script remoto descarga el tag con la Deploy Key, ejecuta los tests, crea una
release inmutable, cambia atómicamente el enlace `/srv/fire-risk/app` y puede
reiniciar el dashboard y ejecutar el smoke test.

### 13.5 Configuración de producción

En `/srv/fire-risk/config/.env` se debe establecer:

```text
PIPELINE_ENVIRONMENT=production
LOCAL_SIMULATION_MODE=false
FORECAST_PROVIDER=auto
FORECAST_AUTO_AEMET_FALLBACK=true
FORECAST_MODEL_FAMILY=egif_48
SHADOW_50_MODEL=true
GRID_PATH=/srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet
MODEL_DIR=/srv/fire-risk/data/models
WEATHER_STATE_PATH=/srv/fire-risk/data/processed/state/weather_daily_state.parquet
CANONICAL_GRID_CELLS=29601
METEOGALICIA_GRIDS=1km,04km
```

`auto` permite la cadena MeteoGalicia 1 km → MeteoGalicia 04 km → AEMET
degradado → stale permitido por configuración. Si se quiere probar únicamente
MeteoGalicia, se puede usar `FORECAST_PROVIDER=meteogalicia`.

### 13.6 Estado meteorológico inicial

Antes de la primera inferencia, el servidor debe construir su propio estado
meteorológico de 30 días usando la rejilla canónica. No se debe copiar el
Parquet de estado generado en el portátil:

```bash
sudo -u fire-risk \
  /opt/miniconda3/envs/incendios-forestales/bin/python \
  /srv/fire-risk/app/scripts/ingest_aemet_weather_state.py \
  --grid /srv/fire-risk/data/processed/grid/galicia_grid_1km_egif.parquet \
  --days 30
```

La orden debe ejecutarse con el `.env` de producción cargado o a través de una
unidad systemd que use `EnvironmentFile=/srv/fire-risk/config/.env`.

### 13.7 Servicios y Nginx

Las unidades actuales incluyen:

```text
fire-risk-dashboard.service
fire-risk-inference.service
fire-risk-inference.timer
fire-risk-health.service
fire-risk-health.timer
```

Se instalan con:

```bash
sudo /srv/fire-risk/app/scripts/install_systemd_units.sh --enable --start
```

La inferencia se ejecuta a las 05:15 y 10:15. Streamlit escucha únicamente en
`127.0.0.1:8501`; Nginx debe ser el único punto de entrada público y publicar
el dashboard mediante HTTPS.

Una vez arrancado el servicio se comprueba:

```bash
sudo systemctl start fire-risk-inference.service
sudo -u fire-risk /srv/fire-risk/app/scripts/check_server_deployment.sh \
  --require-output
sudo journalctl -u fire-risk-inference.service -n 100 --no-pager
```

### 13.8 Limitación actual del automatismo

Las unidades versionadas ejecutan inferencia y health check, pero todavía no
incluyen una unidad específica para mantener las observaciones AEMET cada seis
horas. Para que el servidor sea completamente autónomo hay que instalar un
servicio o timer para:

```bash
scripts/ingest_aemet_current_observations.py --loop --interval-hours 6
```

Hasta que se añada esa unidad, el estado meteorológico debe actualizarse
manualmente o mediante un mecanismo externo supervisado. Si no se hace, la
inferencia acabará usando un estado antiguo y el health check debe marcarlo.

## 14. Rollback y shadow model

El modelo 50 actual no se elimina. Se conserva en `data/models/` y puede
ejecutarse en shadow:

```text
FORECAST_MODEL_FAMILY=egif_48
SHADOW_50_MODEL=true
```

El mapa publicado procede del EGIF 48, mientras el 50 se ejecuta para comparar
rankings y detectar cambios importantes. Para volver a una release anterior se
utiliza el script de rollback del servidor y se reinicia Streamlit. Los datos,
modelos y outputs están en un volumen persistente independiente del checkout de
código.

## 15. Qué falta para declarar producción científica definitiva

La implementación técnica permite ejecutar un piloto en servidor, pero quedan
estas condiciones científicas:

1. validar definitivamente DEM, usos del suelo, carreteras y demás capas
   estáticas;
2. repetir el dataset alineado con `static_layer_quality` validada;
3. acumular forecasts MeteoGalicia con `issued_at`, `valid_time` y observación
   posterior real;
4. evaluar por separado WRF 1 km, WRF 04 km y AEMET;
5. medir la degradación verdadera de T+1 a T+3;
6. recalibrar probabilidades con pares forecast–observación;
7. validar la utilidad con presupuestos reales del operativo de vigilancia;
8. añadir la unidad systemd de observaciones AEMET y alertas externas;
9. probar restauración de backups y rollback de modelos.

Hasta completar estos puntos, la etiqueta correcta es:

```text
EGIF 48 — piloto operativo provisional
```

## 16. Comandos científicos reproducibles

Construir el dataset alineado:

```bash
make PYTHON=/opt/anaconda3/envs/incendios-forestales/bin/python \
  build-operational-dataset
```

Entrenar la familia ampliada de 48 variables:

```bash
make PYTHON=/opt/anaconda3/envs/incendios-forestales/bin/python \
  train-egif-expanded
```

Entrenar el control alineado de 50 variables:

```bash
make PYTHON=/opt/anaconda3/envs/incendios-forestales/bin/python \
  train-egif-50-control
```

Evaluar todas las familias:

```bash
make PYTHON=/opt/anaconda3/envs/incendios-forestales/bin/python \
  evaluate-model-families
```

Promover para una prueba local cuando las capas sean provisionales:

```bash
PYTHONPATH=. /opt/anaconda3/envs/incendios-forestales/bin/python \
  scripts/promote_model_family.py \
  --source-dir data/models/experiments/expanded \
  --destination-dir data/models \
  --dataset-dir data/processed/tabular/egif_operational \
  --prefix forecast_risk_egif_48 \
  --allow-provisional
```

`--allow-provisional` es una excepción para pruebas locales. No debe utilizarse
para declarar que el modelo está científicamente validado.

## 17. Conclusión

La principal mejora no consiste únicamente en pasar de 50 a 48 variables. El
cambio importante es que ahora existe una cadena completa y auditable:

```text
datos históricos
  → memoria meteorológica sin fuga temporal
  → contrato de features versionado
  → tres modelos independientes por horizonte
  → calibración separada
  → forecast MeteoGalicia con fallback etiquetado
  → ranking preventivo y manifest
  → dashboard y despliegue reproducible
```

Los resultados de 2023 muestran que EGIF 48 ampliado supera al FWI como sistema
de ordenación espacial y mejora generalmente al control alineado de 50. La
conclusión operativa razonable es utilizarlo para priorizar vigilancia y
movilización, no para afirmar que una celda causará un incendio ni para
interpretar la salida como una probabilidad perfecta. El siguiente salto de
calidad será medirlo con forecasts reales de MeteoGalicia y observaciones
posteriores archivadas durante la operación.
