# Modelado final: consolidación de los tres análisis independientes

**Sistema de predicción de igniciones forestales en Galicia**
Fase de modelado · Documento científico y puente operativo · 5 de septiembre de 2026

---

## Nota de versionado y alcance operativo

Este documento conserva los resultados retrospectivos obtenidos con el datacubo anterior. No
deben confundirse con una evaluación de un forecast real. Desde esta versión se separan cuatro
productos:

| Producto | Uso | Meteorología | Horizonte |
|---|---|---|---|
| Modelo retrospectivo de 48 variables | Resultado científico | ERA5 perfecta del día observado | Día histórico |
| Familia operativa `egif-2d-48-v1` | Producción | Forecast MeteoGalicia WRF | T+1/T+2/T+3 |
| Control alineado de 50 variables | Comparación científica | ERA5-perfect con memoria T-1 | T+1/T+2/T+3 |
| Familia `egif-2d-v1` de 50 variables | Rollback/shadow | Forecast compatible | T+1/T+2/T+3 |
| FWI CEMS | Baseline físico | Copernicus | Diario |

La familia operativa de 48 y el control alineado de 50 se entrenan sobre
`data/processed/tabular/egif_operational/`, no sobre el Parquet canónico. El
datacubo canónico conserva acumulaciones inclusivas y se mantiene como fuente
retrospectiva auditable. La construcción de la capa alineada y los comandos
exactos están documentados en
`docs/explanations/implementacion_alineacion_operativa.md`.

El modelo retrospectivo de 48 variables y el nuevo modelo operativo comparten el conjunto de
predictores, pero no son el mismo artefacto: el primero conoce retrospectivamente la meteorología
del día objetivo; el segundo debe recibirla de MeteoGalicia. Hasta archivar suficientes parejas
forecast-observación, el benchmark ERA5 se etiqueta como `era5_perfect_benchmark` y no se presenta
como rendimiento operativo.

La familia operativa nueva se publicará únicamente cuando existan los tres artefactos y hayan
pasado el contrato, el backtest temporal y la revisión de calidad. Mientras tanto, el modelo de
50 variables se conserva como rollback; nunca se mezclan columnas de ambas familias.

## Resumen retrospectivo

Este documento describe el pipeline definitivo de modelado, que consolida en un único punto de
entrada reproducible los tres análisis que el equipo había realizado por separado sobre el
dataset EGIF. Sustituye a los estudios previos como referencia para el capítulo de modelado de
la memoria.

El resultado científico documentado es un **LightGBM retrospectivo sobre 48 predictores**,
calibrado con regresión de Platt y validado sobre el año 2022 completo sin submuestrear. Sus
cifras no incluyen el error de un forecast meteorológico real.

| Métrica | Valor |
|---|---|
| Recall con el 5 % del territorio en alerta | **39,66 %** (IC 90 %: 37,67 – 41,71) |
| Recall en el 1 % de celdas más críticas del día | 7,66 % |
| ROC-AUC global | 0,8659 |
| ROC-AUC en temporada (junio–septiembre) | 0,8052 |
| ROC-AUC dentro del día | 0,7431 |
| PR-AUC | 0,00141 (lift 9,18× sobre azar) |
| Filas evaluadas | 10.804.365 |
| Igniciones | 1.659 (prevalencia 0,0154 %) |

**El resultado principal para la memoria:** el modelo supera a las **dos** versiones del índice
FWI disponibles. Frente al oficial de Copernicus CEMS, +17,2 puntos, equivalentes a **286
igniciones adicionales al año** con el mismo presupuesto de vigilancia; frente al calculado a
1 km con las ecuaciones de Van Wagner, +15,7 puntos y 261 igniciones.

**El resultado metodológicamente más relevante:** una variante que solo emplea información
disponible al terminar el día anterior alcanza el 39,54 % de recall, **estadísticamente
indistinguible del modelo completo** (0,12 puntos de diferencia frente a un suelo de ruido de
1,93). El sistema no es únicamente un índice de diagnóstico: funciona como aviso anticipado sin
coste medible.

---

## 1. Objetivo y alcance

El equipo había producido tres análisis independientes sobre el mismo dataset, con protocolos y
conclusiones parcialmente divergentes. Este trabajo tenía tres objetivos:

1. **Unificar** los tres en un pipeline único, reproducible con un solo comando.
2. **Resolver las discrepancias** entre ellos, en particular el conflicto sobre el método de
   calibración, que producía diferencias de más de seis puntos de recall.
3. **Cerrar las objeciones metodológicas** pendientes con evidencia medida, no con argumentos.

El resultado científico es `scripts/pipeline_definitivo.py`, con las etapas que producen el
modelo retrospectivo y la evidencia que lo respalda. La publicación operativa se realiza con
`scripts/train_egif_operational.py` y `scripts/run_daily_inference.py`, que mantienen su propia
familia de artefactos, proveedor meteorológico y manifest.

---

## 2. Datos de partida

El dataset procede del datacubo canónico, reconstruido el 5 de septiembre de 2026 con la serie
histórica completa, en formato Parquet particionado por año.

| Concepto | Valor |
|---|---|
| Predictores declarados | 50 |
| Años disponibles | 2016–2023 |
| Filas totales | 86.494.122 |
| Igniciones registradas | 12.699 |
| Filas descartadas por predictores incompletos | 0 |
| Contrato temporal | Sin desfase; los acumulados incluyen la fecha T |
| Codificación cíclica del calendario | Ninguna |

Los 50 predictores se agrupan en cuatro familias temáticas: meteorología (22), topografía (12),
cobertura del suelo (9) y actividad humana (7).

Este apartado describe el snapshot de 2019–2023 utilizado para los resultados retrospectivos.
La reconstrucción operativa ampliada incorpora 2016–2018 y vuelve a generar el metadata bajo el
mismo contrato de 50 columnas del datacubo; la familia operativa selecciona 48 de ellas.

La ausencia de variables de calendario es deliberada y se comprueba en la etapa `contrato`. Un
modelo que dispusiera de la fecha podría alcanzar buenas métricas globales limitándose a
aprender la estacionalidad, sin extraer información meteorológica alguna.

### 2.1 Auditoría de integridad previa

Antes de entrenar se verificó la coherencia interna del dataset:

- **Sin fugas del target.** Ningún predictor deriva de la variable objetivo. Las columnas de
  resultado (`burned_area_ha`, `large_fire_500ha`, `target_ignicion`) y la auxiliar de muestreo
  (`is_near_ignition_25x25_10d`) están declaradas fuera del conjunto de predictores y el
  contrato lo verifica al cargar.
- **Coherencia física del déficit de presión de vapor.** `vpd_mean` y `vpd_max_12_18h` cuadran
  con la formulación de Magnus aplicada a la temperatura y la humedad de su propia fila
  (correlaciones de 0,9944 y 0,9999).
- **Relaciones de orden.** Mínimo ≤ media ≤ máximo en temperatura, humedad y viento; los
  extremos de la ventana 12–18 h contenidos en el rango del día completo. Se cumplen en los
  10,8 millones de filas de 2022 **sin una sola excepción**.

### 2.2 Incidencia detectada y corregida: `consecutive_dry_days`

Sobre la versión anterior del datacubo, la auditoría detectó que una de las tres variables
acordadas en la reunión del 29 de agosto no era reproducible a partir del resto de la fila. Los días consecutivos sin lluvia significativa
presentaban tres anomalías incompatibles con su definición:

| Anomalía | Alcance (año 2022) |
|---|---|
| Valores no enteros, siendo una cuenta de días | 11,25 % de las filas |
| Celda sin lluvia en su propia columna y contador que desciende | 42.550 celda-día |
| Lluvia superior a 1 mm y contador que no se reinicia | 123.935 celda-día |

**Causa.** La variable se calcula sobre la rejilla original de ERA5-Land (≈9 km) y después se
interpola espacialmente a la rejilla de 1 km. La interpolación conmuta con las operaciones
lineales pero no con las que dependen de la historia: promediar la racha de una celda que lleva
un mes seca con la de su vecina, a la que llovió el día anterior, no produce la racha de
ninguna de las dos. Las otras dos variables acordadas —medias móviles de humedad y viento— sí
sobreviven a la interpolación, y de hecho coinciden con el recálculo independiente hasta el
límite de la precisión de coma flotante simple.

**Solución.** El pipeline de ingesta se corrigió para derivar la racha al final del proceso, a
partir de la precipitación ya interpolada a cada celda de 1 km. La versión vigente del datacubo
publica un contador entero, coherente con la lluvia de su propia fila y con continuidad entre
años. La incidencia queda resuelta en origen.

**Salvaguarda permanente.** `src/entrenamiento/dias_secos.py` conserva la reparación para
Parquet heredados, y el pipeline **diagnostica la columna al arrancar**: mide la proporción de
valores no enteros y de días con lluvia que no reinician el contador, y solo repara si detecta
el patrón. Sobre el datacubo actual el diagnóstico da 0,00 % en ambos síntomas y la columna no
se toca. Así la decisión no depende de que nadie recuerde qué versión del dataset tiene delante.

---

## 3. Protocolo experimental

### 3.1 Separación temporal de los años

| Año | Papel | Igniciones |
|---|---|---|
| 2016–2020 | Entrenamiento | 9.584 |
| 2021 | Parada temprana y ajuste del calibrador | 926 |
| 2022 | Validación: conjunto sobre el que se tomaron las decisiones | 1.659 |
| 2023 | **Test ciego: no interviene en ninguna decisión** | 530 |

El periodo de entrenamiento se amplió a cinco años al reconstruir el datacubo. Antes de
adoptarlo se comprobó que el registro EGIF no estuviera infrarregistrado en los años nuevos:
2016 y 2017 son, de hecho, los peores de la serie (2.246 y 2.980 igniciones frente a las 1.659
de 2022), de modo que aportan comportamiento de fuego extremo que el modelo no había visto.

El año reservado, 2023, es el más flojo de la serie con 530 igniciones. Sus intervalos de
confianza serán por tanto más anchos que los de validación, y así debe advertirse al reportarlo.

La separación es temporal y no aleatoria. Un reparto aleatorio de filas colocaría días
consecutivos de la misma ola de calor a ambos lados de la partición, y el modelo obtendría
métricas excelentes sin haber aprendido nada transferible a un año futuro.

La ejecución operativa conservará dos configuraciones comparables: entrenamiento 2019–2020 y
entrenamiento ampliado 2016–2020. En ambas, 2021 se reserva para calibración, 2022 para
validación y 2023 para el test ciego. Las métricas de cada configuración y de cada horizonte se
guardarán en los JSON de los artefactos, no se sustituirán entre sí.

### 3.2 Submuestreo de negativos y corrección de prior

Con una prevalencia del 0,0154 %, los días-celda sin ignición son masivamente redundantes. Se
conservan **todas las igniciones y uno de cada cien negativos**, seleccionados mediante un hash
determinista de `(cell_id, fecha)` para que la muestra sea reproducible y estable entre
ejecuciones.

El factor de submuestreo se adoptó a partir de la medición del análisis independiente que
comparó dieciséis combinaciones y estableció que 1/100 produce un recall indistinguible de 1/25
con un cuarto del coste de entrenamiento.

El submuestreo infla las probabilidades predichas en un factor conocido, que se deshace con la
corrección de prior de King y Zeng antes de calibrar. **La evaluación nunca se submuestrea**: se
realiza sobre la población completa del año, y una comprobación de cobertura aborta la ejecución
si el recorrido no cubre exactamente las filas que declara el contrato.

### 3.3 Calibración: por qué Platt y no isotónica

Dos de los tres análisis previos discrepaban en este punto: uno no observaba efecto al calibrar
y el otro perdía más de seis puntos de recall.

La causa es que la regresión isotónica es una función **escalonada**: asigna valores idénticos a
bloques enteros de observaciones. Sobre diez millones de filas eso genera millones de empates, y
cualquier umbral definido por percentiles —como el del 5 % del territorio— cae en mitad de un
escalón y desplaza el corte de forma arbitraria. El primer análisis no lo sufría porque calculaba
las métricas de ordenación sobre la puntuación cruda, antes de calibrar; el segundo aplicaba la
isotónica a la probabilidad de decisión y por eso se degradaba.

**La regresión de Platt resuelve la discrepancia** por ser estrictamente monótona: no introduce
empates, el orden de las celdas es idéntico antes y después de calibrar, y ranking y probabilidad
calibrada pasan a ser la misma magnitud. El sistema puede así operar sobre un único número.

En la familia operativa de 48 variables el calibrador se ajusta sobre 2021, un año que el modelo
no ha visto y con su prevalencia real intacta. El orden es corrección de prior por el submuestreo
y después Platt. El contrato histórico de 50 variables mantiene su calibrador serializado
existente (corrección de prior más isotónica) para que el rollback siga siendo reproducible; no
se debe describir ese artefacto como si utilizara Platt.

### 3.4 Métricas y por qué estas

La exactitud (*accuracy*) no aparece en ningún punto del trabajo: un modelo que predijera
ausencia de fuego todos los días acertaría el 99,985 % de las veces. Las métricas empleadas
responden a la pregunta operativa real.

- **Recall a coste operativo fijo (FPR ≤ 5 %).** Fracción de igniciones capturadas si se vigila
  de forma reforzada el 5 % del territorio. Presupuesto fijo, realista e interpretable sin
  formación estadística.
- **Recall en el 1 % diario.** La misma idea aplicada día a día: las 296 celdas más críticas de
  cada jornada. Más exigente y más próximo al despliegue real de medios.
- **ROC-AUC en tres versiones.** Global, restringido a temporada (junio–septiembre) y calculado
  dentro de cada día y promediado. La progresión 0,8659 → 0,8052 → 0,7431 es informativa: mide
  cuánto del rendimiento procede de distinguir agosto de enero —algo que el calendario ya
  resuelve— y cuánto de discriminar celdas dentro de un mismo día, que es la aportación real.
- **Intervalos de confianza al 90 % por bootstrap sobre las igniciones**, no sobre todas las
  filas: la incertidumbre procede casi por completo de los 1.659 positivos.

---

## 4. El modelo retrospectivo documentado

**LightGBM sobre 48 de los 50 predictores.** Quedan excluidas `precipitation_sum` y
`consecutive_dry_days`.

La familia operativa que implementa esta misma decisión se identifica como
`egif-2d-48-v1` y se serializa separadamente como
`forecast_risk_egif_48_t1.joblib`, `forecast_risk_egif_48_t2.joblib` y
`forecast_risk_egif_48_t3.joblib`. Los artefactos `forecast_risk_egif_t*.joblib` pertenecen al
contrato histórico de 50 variables y solo pueden actuar como rollback/shadow.

### 4.1 Justificación de la exclusión

La versión de 48 variables obtiene 39,66 % de recall frente al 39,12 % del conjunto completo.
**Esa diferencia de 0,54 puntos no justifica la decisión**: es muy inferior al suelo de ruido de
1,93 puntos que establece la etapa `semilla`, por lo que ambas versiones son estadísticamente
indistinguibles. Presentar la diferencia como una mejora sería interpretar ruido, y además la
comparación se realizó sobre el conjunto de validación, lo que la invalidaría como criterio de
selección.

**El criterio aplicado es otro: a igualdad de rendimiento se elige la versión que no admite la
objeción.** Ambas variables contienen la precipitación del propio día T, que en muchos casos es
posterior a la ignición. Prescindir de ellas no tiene coste medible y elimina la crítica más
seria que puede formularse contra el contrato temporal del dataset. Es una decisión
metodológica, no un ajuste a los datos.

La configuración queda registrada en el parámetro `excluir_del_modelo` y se serializa junto con
el modelo, de modo que cualquier consumidor futuro puede verificar qué variables se usaron y por
qué se descartaron las restantes.

### 4.2 Hiperparámetros

No se realizó búsqueda de hiperparámetros. Dos de los tres análisis previos midieron que el
ajuste fino aporta menos que el ruido de implementación, por lo que se emplean valores estables
y se documenta la decisión: 800 árboles como techo para el pipeline científico, tasa de
aprendizaje 0,05, 63 hojas, mínimo 30 observaciones por hoja, submuestreo de filas y columnas
al 80 % y parada temprana a 50 rondas sobre el año de calibración. La familia operativa parte
de 400 árboles como configuración inicial, no aplica parada temprana sobre el test y nunca
reentrena durante la inferencia.

Se fija `deterministic` y `force_row_wise` para que la ejecución sea reproducible.

---

## 5. Resultados por etapa

### 5.1 Selección de algoritmo (`modelos`, `pareado`)

Se entrenaron cuatro candidatos con protocolo idéntico. La regresión logística se incluye
deliberadamente: si el *boosting* no le sacara ventaja sustancial, lo correcto sería entregar el
modelo lineal, interpretable y explicable en dos frases.

| Modelo | Recall @5 % | Recall 1 % diario | ROC dentro del día |
|---|---|---|---|
| **LightGBM** | **39,66 %** | **7,66 %** | **0,7431** |
| XGBoost | 35,74 % | 7,11 % | 0,7321 |
| Random Forest | 34,00 % | 7,11 % | 0,6903 |
| Regresión logística | 30,98 % | 3,98 % | 0,7011 |

La comparación se completó con el test pareado sobre las mismas igniciones y el estadístico de
McNemar. LightGBM supera a los otros tres de forma concluyente:

| Comparación | Diferencia | p (McNemar) |
|---|---|---|
| LightGBM vs. regresión logística | +8,68 pp | 6,4 · 10⁻¹⁶ |
| LightGBM vs. Random Forest | +5,67 pp | 1,1 · 10⁻⁶ |
| LightGBM vs. XGBoost | +3,92 pp | 1,8 · 10⁻⁴ |

Las diferencias superan el suelo de ruido de 1,93 puntos. **Conviene señalar que
esto corrige una conclusión anterior**: análisis previos sobre versiones distintas del dataset
habían encontrado empate entre LightGBM y XGBoost. Con el dataset de 50 predictores no lo hay.

### 5.2 Estabilidad frente a la semilla (`semilla`)

Cinco reentrenamientos idénticos salvo por la semilla aleatoria:

| Semilla | Recall @5 % | Recall 1 % diario |
|---|---|---|
| 42 | 39,66 % | 7,66 % |
| 7 | 39,60 % | 8,38 % |
| 123 | 39,42 % | 8,44 % |
| 2024 | 39,96 % | 8,50 % |
| 31 | 41,35 % | 8,86 % |

**Dispersión: 1,93 puntos de recall**, equivalentes a 32 igniciones. Este valor es el suelo de
ruido del estudio y condiciona la lectura de todas las demás etapas: ninguna diferencia inferior
a 1,93 puntos puede presentarse como un hallazgo. Es un listón exigente y conviene que lo sea.

Sin esta medición, cualquier comparación del trabajo sería ininterpretable.

### 5.3 Circularidad temporal (`circularidad`)

El dataset no aplica desfase temporal: la meteorología corresponde al propio día de la ignición,
y la precipitación diaria agrega las 24 horas completas. Una ignición declarada a las 15:00
convive en su fila con la lluvia caída a las 22:00. Es la objeción de mayor calado que admite el
diseño del dataset.

Una respuesta limitada a retirar las dos variables más evidentes sería incompleta, porque **todos
los acumulados incluyen la fecha T por construcción** —incluida `precipitation_sum_3d`, la
variable de mayor importancia del modelo—. Se midieron por tanto tres escenarios de exigencia
creciente:

| Escenario | Variables | Recall @5 % | ROC dentro del día |
|---|---|---|---|
| **Modelo entregado** | 48 | **39,66 %** | 0,7431 |
| **Pronóstico (solo hasta T-1)** | 36 | **39,54 %** | 0,7344 |
| Completo (incluye el día T) | 50 | 39,12 % | 0,7400 |
| Amplia (sin ningún acumulado) | 40 | 36,23 % | 0,7413 |

El escenario de pronóstico merece detalle. Los acumulados pueden despojarse del día T mediante
aritmética exacta sobre las columnas publicadas: una suma de siete días que incluye hoy, menos
el valor de hoy, **es** la suma de los seis días anteriores; una media de siete días multiplicada
por siete, menos el valor de hoy, dividida entre seis, **es** la media de los seis anteriores. La
transformación se verificó contra ejemplos calculados a mano. Retirando además toda observación
del propio día quedan 36 variables que solo contienen información disponible al terminar el día
T-1.

**Ese modelo alcanza el 39,54 % de recall, a 0,12 puntos del modelo completo.** La diferencia es
un orden de magnitud menor que el suelo de ruido de 1,93: son estadísticamente indistinguibles.

La implicación es sustantiva. Renunciar por completo a los datos del día que se predice **no
tiene coste medible**, de modo que el sistema no se limita a describir el día en curso: puede
emitir el aviso la tarde anterior, que es el horizonte que un servicio de extinción necesita
para movilizar medios, y sigue superando a las dos versiones del FWI por más de quince puntos.

Y responde a la objeción del contrato temporal de forma definitiva: aun rechazando de plano el
uso de información del propio día, las conclusiones del trabajo se mantienen intactas. Merece
la pena señalar que con el periodo de entrenamiento anterior esta variante perdía 2,1 puntos;
al disponer de cinco años de entrenamiento la pérdida desaparece.

### 5.4 Comparación con el estándar operativo (`fwi`)

Superar a una regresión logística demuestra únicamente que el *boosting* supera a un modelo
lineal. Para sostener que el sistema aporta valor sobre lo que ya se publica es necesario
compararlo con el **Fire Weather Index** del sistema canadiense (CFFDRS), que es el índice que
difunden AEMET y EFFIS y el que se emplea operativamente.

El FWI se calculó a resolución de 1 km sobre nuestro propio dataset, con las ecuaciones de Van
Wagner, en lugar de descargarlo de CEMS a 27,5 km e interpolarlo. El resultado es un baseline
más exigente que el de la referencia bibliográfica.

Se evalúan las dos fuentes del índice de que dispone el proyecto, porque cada una tiene una
limitación distinta y ninguna por separado cierra la discusión. El **oficial de CEMS** está
calculado como manda la definición, pero nace a 27,5 km: al interpolarlo a 1 km, todas las
celdas bajo un mismo píxel comparten valor y no puede discriminar dentro de él. El de **Van
Wagner a 1 km** sí distingue celda a celda, pero sustituye la lectura de mediodía por los
extremos de la ventana 12–18 h.

| | Recall @5 % | ROC dentro del día | Ventaja del modelo |
|---|---|---|---|
| **Modelo** | **39,66 %** | **0,7431** | — |
| FWI Van Wagner 1 km | 23,93 % | 0,6148 | +15,7 pp · **261 igniciones/año** |
| FWI oficial CEMS | 22,42 % | 0,6015 | +17,2 pp · **286 igniciones/año** |

Superar a las dos cierra a la vez las dos réplicas posibles: que la ventaja proceda únicamente
de disponer de mayor resolución, y que el índice con el que se compara no sea el que realmente
se publica.

**Salvedad que debe constar en la memoria:** el FWI oficial se define sobre observaciones de
mediodía solar, mientras que aquí se calcula con los extremos de la ventana 12–18 h. Esa
sustitución sobreestima el índice de forma sistemática. Es aceptable para su uso como baseline
comparativo —interesa su capacidad de ordenar celdas por riesgo, no su valor absoluto—, pero
invalida el uso de los umbrales oficiales de la tabla de peligro.

### 5.5 Contribución de las variables (`importancia`)

**Importancia por permutación** (caída de ROC-AUC al barajar cada variable, tres repeticiones):

| Variable | Caída de ROC-AUC |
|---|---|
| `precipitation_sum_3d` | 0,0265 |
| `relative_humidity_mean_7d` | 0,0216 |
| `elevation_mean` | 0,0120 |
| `vpd_mean` | 0,0108 |
| `road_length_local_km` | 0,0106 |
| `precipitation_sum_7d` | 0,0102 |

La presencia de `road_length_local_km` entre las primeras es coherente con la etiología conocida
de los incendios en Galicia, de origen humano en su gran mayoría.

**Ablación por grupos temáticos** (un entrenamiento completo por grupo retirado):

| Grupo retirado | Variables restantes | Recall @5 % | Δ |
|---|---|---|---|
| Meteorología | 28 | 25,92 % | **−13,74 pp** |
| Topografía | 36 | 37,49 % | −2,17 pp |
| Actividad humana | 41 | 38,76 % | −0,90 pp |
| Cobertura del suelo | 39 | 39,18 % | −0,48 pp |
| Ninguno | 48 | 39,66 % | — |

La meteorología es el núcleo del modelo: sin ella el recall cae más de trece puntos. La
topografía aporta 2,17 puntos, por encima del suelo de ruido de 1,93 y por tanto demostrable.

**Actividad humana y cobertura del suelo quedan por debajo del suelo de ruido** (0,90 y 0,48
puntos): su contribución no es distinguible de la variación aleatoria. Son dieciséis variables
cuya aportación el estudio no puede demostrar. Conviene enunciarlo así —«no demostrable»— y no
como «no aportan nada», que es una afirmación más fuerte de lo que el dato sostiene.

### 5.6 Generalización espacial (`espacial`)

Es la prueba más exigente del conjunto. Un modelo puede obtener métricas excelentes limitándose
a memorizar qué celdas arden con regularidad —los montes de Ourense arden todos los años— sin
haber aprendido relaciones meteorológicas transferibles. Ese modelo funcionaría en Galicia y en
ningún otro sitio.

El territorio se divide en cinco **bandas diagonales**, se entrena excluyendo una banda completa
y se evalúa exclusivamente sobre ella. Las bandas son diagonales y no cuadrantes norte-sur
porque Galicia presenta un gradiente climático acusado entre la costa atlántica y el interior
ourensano: unos cuadrantes dejarían algún pliegue sin litoral y el modelo fallaría por no haber
visto ese régimen climático, no por incapacidad de generalizar.

| Banda | Igniciones | Recall @5 % | ROC dentro del día |
|---|---|---|---|
| 0 | 646 | 26,01 % | 0,5920 |
| 1 | 442 | 28,96 % | 0,6705 |
| 2 | 253 | 24,90 % | 0,6557 |
| 3 | 166 | 27,11 % | 0,6361 |
| 4 | 152 | 29,61 % | 0,5869 |

**ROC dentro del día: 0,7431 con el reparto habitual, 0,6282 en territorio no visto.** La caída
de 0,115 debe declararse explícitamente en la memoria: cuantifica la fracción del rendimiento
que **no se transfiere** a una zona no representada en el entrenamiento. El sistema está
validado para Galicia; su despliegue en otra comunidad exigiría reentrenamiento.

### 5.7 Análisis de errores (`errores`)

Un recall del 38 % implica que seis de cada diez igniciones no se detectan. Identificar
**cuáles** es lo que distingue un modelo de un sistema operativo.

| Mes | Igniciones | Recall |
|---|---|---|
| Enero | 115 | 42,61 % |
| Febrero | 139 | 40,29 % |
| Marzo | 72 | **0,00 %** |
| Abril | 94 | 18,09 % |
| Mayo | 116 | 21,55 % |
| Junio | 50 | 2,00 % |
| Julio | 428 | **55,37 %** |
| Agosto | 491 | 48,07 % |
| Septiembre | 122 | 25,41 % |
| Octubre | 29 | 20,69 % |

El rendimiento se concentra donde importa: julio y agosto acumulan 919 de las 1.659 igniciones y
el modelo detecta más de la mitad en julio. El fracaso de marzo —**cero de setenta y dos**— es
coherente con la etiología:
las igniciones de marzo proceden mayoritariamente de quemas agrícolas descontroladas, un
fenómeno cultural y de calendario laboral que el modelo no observa, ya que solo dispone de
meteorología, terreno y vegetación.

Las igniciones no detectadas se producen sistemáticamente en días **más húmedos, más frescos y
con menos días secos acumulados** que las detectadas, lo que confirma que el modelo ordena por
severidad meteorológica y que lo que se le escapa son los fuegos cuyo origen no es meteorológico.

### 5.8 Calibración y niveles de riesgo (`fiabilidad`, `riesgo`)

La tabla de fiabilidad, sobre doce tramos de igual tamaño, muestra correspondencia monótona
entre probabilidad predicha y frecuencia observada, con **infraestimación sistemática en torno a
un factor 1,4** (tramo superior: 6,89 · 10⁻⁴ predicho frente a 9,47 · 10⁻⁴ observado). La salida es apta para ordenar y para fijar umbrales; no
debe interpretarse como probabilidad absoluta sin advertirlo.

Traducción a niveles operativos, por cuantiles de la probabilidad calibrada:

| Nivel | % del territorio | Celdas-día | Igniciones | Incidencia | Veces sobre «Bajo» |
|---|---|---|---|---|---|
| Bajo | 90,0 % | 9.723.928 | 735 | 7,6 · 10⁻⁵ | 1,0 |
| Moderado | 8,0 % | 864.349 | 526 | 6,1 · 10⁻⁴ | 8,1 |
| Alto | 1,5 % | 162.066 | 238 | 1,5 · 10⁻³ | 19,4 |
| Extremo | 0,5 % | 54.022 | 160 | 3,0 · 10⁻³ | **39,2** |

Lo relevante no es la posición de los cortes sino que la incidencia observada crece de forma
marcada y monótona entre niveles. Es lo que permite a quien recibe el aviso confiar en que
«extremo» designa una situación cualitativamente distinta de «alto».

### 5.9 Sensibilidad a la definición del objetivo (`target`)

El registro EGIF contabiliza toda ignición, incluidos conatos de escasa superficie. Reevaluando
el mismo modelo sobre subconjuntos definidos por superficie quemada:

| Definición | Igniciones | Recall @5 % | ROC dentro del día |
|---|---|---|---|
| Todas | 1.659 | 39,66 % | 0,7431 |
| ≥ 1 ha | 351 | 38,18 % | 0,6919 |
| ≥ 10 ha | 106 | 35,85 % | 0,6306 |
| ≥ 100 ha | 29 | 27,59 % | 0,6077 |
| ≥ 500 ha | 12 | 25,00 % | 0,4529 |

El recall se mantiene razonablemente estable hasta las 10 ha, lo que indica que el modelo no
debe su rendimiento a acertar conatos irrelevantes, aunque desciende en los estratos superiores.
**En el de grandes incendios el ROC dentro del día cae a 0,4529, por debajo del azar**; debe reportarse, si bien con la cautela que impone un tamaño de muestra de
doce casos, insuficiente para sostener conclusión alguna.

---

## 6. Test ciego sobre 2023: pendiente y deliberado

El año 2023 no ha intervenido en ninguna etapa de este trabajo.

La razón es que las cifras de 2022, siendo el modelo ciego a ese año, **no lo son respecto al
proceso de decisión**: la elección del algoritmo entre cuatro candidatos, la exclusión de dos
variables y la fijación del umbral operativo se resolvieron examinando resultados sobre 2022.
Toda decisión adoptada a la vista de un conjunto lo incorpora parcialmente al entrenamiento, y
sesga sus métricas al alza.

2023 proporciona la única estimación insesgada del rendimiento en producción, y solo la
proporciona si se emplea **una única vez**, con la configuración enteramente congelada
—algoritmo, hiperparámetros, conjunto de variables, protocolo de muestreo y calibrador—. Por eso
la etapa correspondiente no se ejecuta por defecto y exige el argumento explícito
`--abrir-test-ciego`.

**Compromiso metodológico:** el resultado que arroje esa ejecución se reportará sin
modificaciones. No se revisará el modelo a la vista de él; hacerlo convertiría 2023 en un
segundo conjunto de validación y agotaría el último año no contaminado del dataset.

En la memoria, la cifra de 2023 constituye el rendimiento esperado del sistema; la de 2022
documenta el proceso de selección.

---

## 7. Limitaciones declaradas

Se enumeran de forma explícita por integridad metodológica:

1. **Generalización espacial limitada.** El ROC dentro del día desciende de 0,7431 a 0,6282 en
   territorio no representado en el entrenamiento. El sistema está validado para Galicia.
2. **Igniciones de origen no meteorológico.** El recall es nulo en marzo (0 de 72) y del 2 % en
   junio, cuando predominan las quemas agrícolas. El modelo no dispone de variables que
   capturen ese fenómeno.
3. **Grandes incendios.** Sin capacidad demostrada de discriminación en el estrato de más de 500
   hectáreas, aunque con una muestra de doce casos que no permite concluir.
4. **Calibración conservadora.** Infraestima el riesgo en torno a un factor 1,4; apta para
   ordenación y umbrales, no para lectura como probabilidad absoluta.
5. **Aportación no demostrable de dos familias de variables.** Actividad humana y cobertura del
   suelo quedan por debajo del suelo de ruido en la ablación. No se afirma que sobren, sino que
   este estudio no puede demostrar su contribución.
6. **FWI sobreestimado.** El baseline se calcula con extremos de la ventana 12–18 h en lugar de
   observaciones de mediodía solar. Válido como ordenación, no para las categorías oficiales.
7. **Un único año de validación.** Las conclusiones se apoyan en 2022; una validación cruzada
   temporal sobre varios años reduciría la incertidumbre, a costa de consumir el año reservado.
8. **Año ciego pequeño.** 2023 es el año más flojo de la serie (530 igniciones), de modo que la
   estimación insesgada que produzca tendrá intervalos más anchos que los de validación.
9. **Error de forecast aún no medido.** La primera familia operativa se entrena con el benchmark
   ERA5-perfect porque no existe un archivo histórico de predicciones de MeteoGalicia. Hay que
   archivar varios meses de pares forecast-observación antes de estimar y corregir el sesgo por
   variable, estación y horizonte.

---

## 8. Reproducibilidad

### Ejecución

```bash
conda activate incendios-forestales
python scripts/pipeline_definitivo.py                    # las trece etapas ordinarias
python scripts/pipeline_definitivo.py --etapas modelos,fwi
python scripts/pipeline_definitivo.py --rehacer          # ignora resultados previos
python scripts/pipeline_definitivo.py --abrir-test-ciego # solo al cerrar, una vez
```

Cada etapa guarda su tabla y se omite si el fichero ya existe, de modo que una ejecución
interrumpida se retoma donde estaba. Tiempo total aproximado: 30 minutos sobre el periodo
2016-2023.

### Artefactos

| Ruta | Contenido |
|---|---|
| `scripts/pipeline_definitivo.py` | Pipeline completo: trece etapas más el test ciego |
| `src/entrenamiento/dias_secos.py` | Diagnóstico y reparación de `consecutive_dry_days` |
| `src/entrenamiento/platt.py` | Calibrador, en módulo propio para que el modelo sea portable |
| `data/models/modelo_definitivo.pkl` | Modelo, calibrador, variables y configuración |
| `data/models/forecast_risk_egif_48_t*.joblib` | Familia operativa EGIF 48 por horizonte |
| `docs/technical/pipeline_*.csv` | Una tabla por etapa |
| `docs/technical/importancia_permutacion.csv` | Importancia de las 48 variables |
| `docs/technical/errores_por_mes.csv` | Recall mensual |
| `docs/technical/umbrales_riesgo.json` | Cortes de los cuatro niveles |
| `docs/technical/pipeline_definitivo_config.json` | Configuración completa de la ejecución |

La configuración se serializa junto al modelo, de modo que cualquier resultado puede rastrearse
hasta los parámetros exactos que lo produjeron.

### Verificación del propio pipeline

Antes de ejecutarlo sobre los datos reales, todas las etapas se validaron sobre un dataset
sintético en miniatura con esquema idéntico al real. La prueba no se limitó a comprobar que el
código no falla: los datos sintéticos se construyeron con señal inyectada en una única variable,
y las etapas de importancia y ablación la identificaron correctamente, verificando que miden lo
que declaran medir. El procedimiento detectó dos defectos de integración que fueron corregidos.

---

## 9. Componentes integrados

El pipeline no se escribió de cero: reutiliza los desarrollos que el proyecto ya tenía y añade
lo necesario para cerrar las objeciones pendientes. Se detalla la procedencia de cada bloque
para que quede trazable qué es reutilización y qué es aportación nueva de esta fase.

| Componente | Procedencia |
|---|---|
| Librería de entrenamiento, métricas a coste operativo fijo, comparación pareada con McNemar, selección de variables, tabla de fiabilidad | Línea de modelado supervisado del proyecto |
| Baseline FWI, comparación de cuatro algoritmos, análisis de errores, sensibilidad a la definición del objetivo, planteamiento de la circularidad, factor de submuestreo 1/100 | Línea de análisis del datacubo EGIF |
| Datacubo canónico, variables meteorológicas acumuladas y contrato de datos | Línea de ingesta y construcción del dataset |
| Validación cruzada espacial por bandas, suelo de ruido por semilla, corrección de `consecutive_dry_days`, escenario de pronóstico, protocolo de test ciego e integración de todo lo anterior | Fase de consolidación (este documento) |
