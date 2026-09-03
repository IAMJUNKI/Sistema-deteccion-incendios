# Modelado final: consolidación de los tres análisis independientes

**Sistema de predicción de igniciones forestales en Galicia**
Fase de modelado · Documento de cierre · 2 de septiembre de 2026

---

## Resumen

Este documento describe el pipeline definitivo de modelado, que consolida en un único punto de
entrada reproducible los tres análisis que el equipo había realizado por separado sobre el
dataset EGIF. Sustituye a los estudios previos como referencia para el capítulo de modelado de
la memoria.

El modelo entregado es un **LightGBM sobre 48 predictores**, calibrado con regresión de Platt y
validado sobre el año 2022 completo sin submuestrear.

| Métrica | Valor |
|---|---|
| Recall con el 5 % del territorio en alerta | **38,03 %** (IC 90 %: 36,17 – 40,08) |
| Recall en el 1 % de celdas más críticas del día | 6,69 % (IC 90 %: 5,67 – 7,72) |
| ROC-AUC global | 0,8634 |
| ROC-AUC en temporada (junio–septiembre) | 0,7964 |
| ROC-AUC dentro del día | 0,7351 |
| PR-AUC | 0,00141 (lift 9,15× sobre azar) |
| Filas evaluadas | 10.804.365 |
| Igniciones | 1.659 (prevalencia 0,0154 %) |

**El resultado principal para la memoria:** el modelo detecta **14,1 puntos más** de igniciones
que el índice FWI —el estándar operativo que publican AEMET y EFFIS— con el mismo presupuesto de
vigilancia. Sobre las 1.659 igniciones de 2022 equivale a **234 incendios adicionales al año**.

**El resultado metodológicamente más relevante:** una variante que solo usa información
disponible al terminar el día anterior alcanza el 35,93 % de recall y **sigue superando al FWI
en doce puntos**. El sistema no es únicamente un índice de diagnóstico: funciona como aviso
anticipado.

---

## 1. Objetivo y alcance

El equipo había producido tres análisis independientes sobre el mismo dataset, con protocolos y
conclusiones parcialmente divergentes. Este trabajo tenía tres objetivos:

1. **Unificar** los tres en un pipeline único, reproducible con un solo comando.
2. **Resolver las discrepancias** entre ellos, en particular el conflicto sobre el método de
   calibración, que producía diferencias de más de seis puntos de recall.
3. **Cerrar las objeciones metodológicas** pendientes con evidencia medida, no con argumentos.

El resultado es `scripts/pipeline_definitivo.py`, con catorce etapas que producen tanto el
modelo de producción como toda la evidencia que lo respalda.

---

## 2. Datos de partida

El dataset procede del datacubo canónico, publicado el 30 de agosto de 2026 en formato Parquet
particionado por año.

| Concepto | Valor |
|---|---|
| Predictores declarados | 50 |
| Años disponibles | 2019–2023 |
| Filas totales | 53.015.391 |
| Filas descartadas por predictores incompletos | 0 |
| Contrato temporal | Sin desfase; los acumulados incluyen la fecha T |
| Codificación cíclica del calendario | Ninguna |

Los 50 predictores se agrupan en cuatro familias temáticas: meteorología (22), topografía (12),
cobertura del suelo (9) y actividad humana (7).

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

### 2.2 Corrección aplicada: `consecutive_dry_days`

La auditoría detectó que una de las tres variables acordadas en la reunión del 29 de agosto no
era reproducible a partir del resto de la fila. Los días consecutivos sin lluvia significativa
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

**Solución.** La variable se reconstruye desde la columna `precipitation_sum` que el propio
Parquet publica, con lo que queda coherente con la lluvia de su misma fila y es reproducible por
cualquiera que disponga del dataset. La implementación está en `src/entrenamiento/dias_secos.py`
y no requiere regenerar el datacubo.

**Limitación conocida.** El recálculo se realiza año a año, de modo que el contador se reinicia
el 1 de enero y las rachas que cruzan el fin de año quedan truncadas. Afecta a los primeros días
de enero, fuera de la temporada de incendios. Se documenta por ser una diferencia real respecto
a la versión del datacubo, que sí arrastra el mes de diciembre anterior.

---

## 3. Protocolo experimental

### 3.1 Separación temporal de los años

| Año | Papel |
|---|---|
| 2019–2020 | Entrenamiento |
| 2021 | Parada temprana y ajuste del calibrador |
| 2022 | Validación: conjunto sobre el que se tomaron las decisiones |
| 2023 | **Test ciego: no interviene en ninguna decisión** |

La separación es temporal y no aleatoria. Un reparto aleatorio de filas colocaría días
consecutivos de la misma ola de calor a ambos lados de la partición, y el modelo obtendría
métricas excelentes sin haber aprendido nada transferible a un año futuro.

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

El calibrador se ajusta sobre 2021, un año que el modelo no ha visto y con su prevalencia real
intacta. Ajustarlo dentro de la muestra de entrenamiento reproduciría el sobreajuste en lugar de
corregirlo.

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
  dentro de cada día y promediado. La progresión 0,8634 → 0,7964 → 0,7351 es informativa: mide
  cuánto del rendimiento procede de distinguir agosto de enero —algo que el calendario ya
  resuelve— y cuánto de discriminar celdas dentro de un mismo día, que es la aportación real.
- **Intervalos de confianza al 90 % por bootstrap sobre las igniciones**, no sobre todas las
  filas: la incertidumbre procede casi por completo de los 1.659 positivos.

---

## 4. El modelo entregado

**LightGBM sobre 48 de los 50 predictores.** Quedan excluidas `precipitation_sum` y
`consecutive_dry_days`.

### 4.1 Justificación de la exclusión

La versión de 48 variables obtiene 38,03 % de recall frente al 36,95 % del conjunto completo.
**Esa diferencia de 1,09 puntos no justifica la decisión**: es inferior al suelo de ruido de
1,45 puntos que establece la etapa `semilla`, por lo que ambas versiones son estadísticamente
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
y se documenta la decisión: 800 árboles como techo, tasa de aprendizaje 0,05, 63 hojas, mínimo
30 observaciones por hoja, submuestreo de filas y columnas al 80 %, y parada temprana a 50
rondas sobre el año de calibración. El modelo final utilizó 304 árboles.

Se fija `deterministic` y `force_row_wise` para que la ejecución sea reproducible.

---

## 5. Resultados por etapa

### 5.1 Selección de algoritmo (`modelos`, `pareado`)

Se entrenaron cuatro candidatos con protocolo idéntico. La regresión logística se incluye
deliberadamente: si el *boosting* no le sacara ventaja sustancial, lo correcto sería entregar el
modelo lineal, interpretable y explicable en dos frases.

| Modelo | Recall @5 % | Recall 1 % diario | ROC dentro del día |
|---|---|---|---|
| **LightGBM** | **38,03 %** | **6,69 %** | **0,7351** |
| XGBoost | 31,71 % | 5,30 % | 0,7008 |
| Random Forest | 31,34 % | 6,33 % | 0,6783 |
| Regresión logística | 29,54 % | 4,46 % | 0,6970 |

La comparación se completó con el test pareado sobre las mismas igniciones y el estadístico de
McNemar. LightGBM supera a los otros tres de forma concluyente:

| Comparación | Diferencia | p (McNemar) |
|---|---|---|
| LightGBM vs. regresión logística | +8,50 pp | 5,0 · 10⁻¹² |
| LightGBM vs. Random Forest | +6,69 pp | 3,5 · 10⁻⁸ |
| LightGBM vs. XGBoost | +6,33 pp | 4,6 · 10⁻⁸ |

Las diferencias superan ampliamente el suelo de ruido de 1,45 puntos. **Conviene señalar que
esto corrige una conclusión anterior**: análisis previos sobre versiones distintas del dataset
habían encontrado empate entre LightGBM y XGBoost. Con el dataset de 50 predictores no lo hay.

### 5.2 Estabilidad frente a la semilla (`semilla`)

Cinco reentrenamientos idénticos salvo por la semilla aleatoria:

| Semilla | Recall @5 % | Recall 1 % diario |
|---|---|---|
| 42 | 38,03 % | 6,69 % |
| 7 | 38,82 % | 6,93 % |
| 123 | 38,52 % | 6,87 % |
| 2024 | 37,37 % | 5,67 % |
| 31 | 37,97 % | 6,21 % |

**Dispersión: 1,45 puntos de recall**, equivalentes a 24 igniciones. Este valor es el suelo de
ruido del estudio y condiciona la lectura de todas las demás etapas: ninguna diferencia inferior
a 1,45 puntos puede presentarse como un hallazgo.

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
| **Modelo entregado** | 48 | **38,03 %** | 0,7351 |
| Completo (incluye el día T) | 50 | 36,95 % | 0,7395 |
| **Pronóstico (solo hasta T-1)** | 36 | **35,93 %** | 0,7362 |
| Amplia (sin ningún acumulado) | 40 | 34,12 % | 0,7277 |

El escenario de pronóstico merece detalle. Los acumulados pueden despojarse del día T mediante
aritmética exacta sobre las columnas publicadas: una suma de siete días que incluye hoy, menos
el valor de hoy, **es** la suma de los seis días anteriores; una media de siete días multiplicada
por siete, menos el valor de hoy, dividida entre seis, **es** la media de los seis anteriores. La
transformación se verificó contra ejemplos calculados a mano. Retirando además toda observación
del propio día quedan 36 variables que solo contienen información disponible al terminar el día
T-1.

**Ese modelo alcanza el 35,93 % de recall y sigue superando al FWI en doce puntos.** La
implicación es sustantiva: el sistema no se limita a describir el día en curso, sino que puede
emitir un aviso la tarde anterior, que es el horizonte que un servicio de extinción necesita para
movilizar medios. Y responde a la objeción de forma definitiva: aun rechazando por completo el
uso de datos del propio día, las conclusiones del trabajo se mantienen.

### 5.4 Comparación con el estándar operativo (`fwi`)

Superar a una regresión logística demuestra únicamente que el *boosting* supera a un modelo
lineal. Para sostener que el sistema aporta valor sobre lo que ya se publica es necesario
compararlo con el **Fire Weather Index** del sistema canadiense (CFFDRS), que es el índice que
difunden AEMET y EFFIS y el que se emplea operativamente.

El FWI se calculó a resolución de 1 km sobre nuestro propio dataset, con las ecuaciones de Van
Wagner, en lugar de descargarlo de CEMS a 27,5 km e interpolarlo. El resultado es un baseline
más exigente que el de la referencia bibliográfica.

| | Recall @5 % | ROC global | ROC dentro del día |
|---|---|---|---|
| Modelo | **38,03 %** | 0,8634 | **0,7351** |
| FWI | 23,93 % | 0,8095 | 0,6148 |

**Ventaja: 14,1 puntos, equivalentes a 234 igniciones adicionales al año** con idéntico
presupuesto de vigilancia. La distancia se amplía en la métrica dentro del día, que es la
operativamente significativa.

**Salvedad que debe constar en la memoria:** el FWI oficial se define sobre observaciones de
mediodía solar, mientras que aquí se calcula con los extremos de la ventana 12–18 h. Esa
sustitución sobreestima el índice de forma sistemática. Es aceptable para su uso como baseline
comparativo —interesa su capacidad de ordenar celdas por riesgo, no su valor absoluto—, pero
invalida el uso de los umbrales oficiales de la tabla de peligro.

### 5.5 Contribución de las variables (`importancia`)

**Importancia por permutación** (caída de ROC-AUC al barajar cada variable, tres repeticiones):

| Variable | Caída de ROC-AUC |
|---|---|
| `precipitation_sum_3d` | 0,0256 |
| `relative_humidity_mean_7d` | 0,0172 |
| `vpd_mean` | 0,0156 |
| `wind_speed_max_12_18h` | 0,0119 |
| `road_length_local_km` | 0,0104 |
| `elevation_mean` | 0,0103 |

La presencia de `road_length_local_km` entre las primeras es coherente con la etiología conocida
de los incendios en Galicia, de origen humano en su gran mayoría.

**Ablación por grupos temáticos** (un entrenamiento completo por grupo retirado):

| Grupo retirado | Variables restantes | Recall @5 % | Δ |
|---|---|---|---|
| Meteorología | 28 | 21,64 % | **−16,40 pp** |
| Topografía | 36 | 35,14 % | −2,89 pp |
| Actividad humana | 41 | 36,35 % | −1,69 pp |
| Cobertura del suelo | 39 | 36,95 % | −1,08 pp |
| Ninguno | 48 | 38,03 % | — |

La meteorología es el núcleo del modelo. La **cobertura del suelo se queda en −1,08 puntos, por
debajo del suelo de ruido de 1,45**: su aportación no es distinguible de la variación aleatoria.
Nueve variables sobre composición de la vegetación cuya contribución no puede demostrarse.

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

| Banda | Celdas | Igniciones | Recall @5 % | ROC dentro del día |
|---|---|---|---|---|
| 0 | 5.961 | 646 | 24,30 % | 0,6091 |
| 1 | 5.943 | 442 | 28,51 % | 0,6758 |
| 2 | 5.877 | 253 | 26,48 % | 0,6430 |
| 3 | 5.901 | 166 | 24,70 % | 0,5808 |
| 4 | 5.919 | 152 | 17,11 % | 0,5500 |

**ROC dentro del día: 0,7351 con el reparto habitual, 0,6117 en territorio no visto.** La caída
de 0,123 debe declararse explícitamente en la memoria: cuantifica la fracción del rendimiento
que **no se transfiere** a una zona no representada en el entrenamiento. El sistema está
validado para Galicia; su despliegue en otra comunidad exigiría reentrenamiento.

### 5.7 Análisis de errores (`errores`)

Un recall del 38 % implica que seis de cada diez igniciones no se detectan. Identificar
**cuáles** es lo que distingue un modelo de un sistema operativo.

| Mes | Igniciones | Recall |
|---|---|---|
| Enero | 115 | 49,57 % |
| Febrero | 139 | 46,76 % |
| Marzo | 72 | **2,78 %** |
| Abril | 94 | 20,21 % |
| Mayo | 116 | 25,00 % |
| Junio | 50 | 6,00 % |
| Julio | 428 | **50,70 %** |
| Agosto | 491 | 41,55 % |
| Septiembre | 122 | 23,77 % |
| Octubre | 29 | 20,69 % |

El rendimiento se concentra donde importa: julio y agosto acumulan 919 de las 1.659 igniciones y
el modelo detecta cerca de la mitad. El fracaso de marzo (2,78 %) es coherente con la etiología:
las igniciones de marzo proceden mayoritariamente de quemas agrícolas descontroladas, un
fenómeno cultural y de calendario laboral que el modelo no observa, ya que solo dispone de
meteorología, terreno y vegetación.

Las igniciones no detectadas se producen sistemáticamente en días **más húmedos, más frescos y
con menos días secos acumulados** que las detectadas, lo que confirma que el modelo ordena por
severidad meteorológica y que lo que se le escapa son los fuegos cuyo origen no es meteorológico.

### 5.8 Calibración y niveles de riesgo (`fiabilidad`, `riesgo`)

La tabla de fiabilidad, sobre doce tramos de igual tamaño, muestra correspondencia monótona
entre probabilidad predicha y frecuencia observada, con **infraestimación sistemática en torno a
un factor dos** (tramo superior: 8,90 · 10⁻⁴ predicho frente a 9,22 · 10⁻⁴ observado; tramos
intermedios con desviaciones mayores). La salida es apta para ordenar y para fijar umbrales; no
debe interpretarse como probabilidad absoluta sin advertirlo.

Traducción a niveles operativos, por cuantiles de la probabilidad calibrada:

| Nivel | % del territorio | Celdas-día | Igniciones | Incidencia | Veces sobre «Bajo» |
|---|---|---|---|---|---|
| Bajo | 90,0 % | 9.723.928 | 742 | 7,6 · 10⁻⁵ | 1,0 |
| Moderado | 8,0 % | 864.349 | 571 | 6,6 · 10⁻⁴ | 8,7 |
| Alto | 1,5 % | 162.066 | 208 | 1,3 · 10⁻³ | 16,8 |
| Extremo | 0,5 % | 54.022 | 138 | 2,6 · 10⁻³ | **33,5** |

Lo relevante no es la posición de los cortes sino que la incidencia observada crece de forma
marcada y monótona entre niveles. Es lo que permite a quien recibe el aviso confiar en que
«extremo» designa una situación cualitativamente distinta de «alto».

### 5.9 Sensibilidad a la definición del objetivo (`target`)

El registro EGIF contabiliza toda ignición, incluidos conatos de escasa superficie. Reevaluando
el mismo modelo sobre subconjuntos definidos por superficie quemada:

| Definición | Igniciones | Recall @5 % | ROC dentro del día |
|---|---|---|---|
| Todas | 1.659 | 38,03 % | 0,7351 |
| ≥ 1 ha | 351 | 37,89 % | 0,6847 |
| ≥ 10 ha | 106 | 38,68 % | 0,6407 |
| ≥ 100 ha | 29 | 37,93 % | 0,6374 |
| ≥ 500 ha | 12 | 25,00 % | 0,4320 |

El recall se mantiene estable, lo que indica que el modelo no debe su rendimiento a acertar
conatos irrelevantes. **En el estrato de grandes incendios el ROC dentro del día cae a 0,4320,
por debajo del azar**; debe reportarse, si bien con la cautela que impone un tamaño de muestra de
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

1. **Generalización espacial limitada.** El ROC dentro del día desciende de 0,7351 a 0,6117 en
   territorio no representado en el entrenamiento. El sistema está validado para Galicia.
2. **Igniciones de origen no meteorológico.** El recall se desploma en marzo (2,78 %), cuando
   predominan las quemas agrícolas. El modelo no dispone de variables que capturen ese
   fenómeno.
3. **Grandes incendios.** Sin capacidad demostrada de discriminación en el estrato de más de 500
   hectáreas, aunque con una muestra de doce casos que no permite concluir.
4. **Calibración conservadora.** Infraestima el riesgo en torno a un factor dos; apta para
   ordenación y umbrales, no para lectura como probabilidad absoluta.
5. **FWI sobreestimado.** El baseline se calcula con extremos de la ventana 12–18 h en lugar de
   observaciones de mediodía solar. Válido como ordenación, no para las categorías oficiales.
6. **Rachas truncadas en el cambio de año.** El recálculo de días secos se realiza por año, con
   efecto limitado a los primeros días de enero, fuera de temporada.
7. **Un único año de validación.** Las conclusiones se apoyan en 2022; una validación cruzada
   temporal sobre varios años reduciría la incertidumbre, a costa de consumir el año reservado.

---

## 8. Reproducibilidad

### Ejecución

```bash
conda activate incendios-forestales
python scripts/pipeline_definitivo.py                    # las trece etapas
python scripts/pipeline_definitivo.py --etapas modelos,fwi
python scripts/pipeline_definitivo.py --rehacer          # ignora resultados previos
python scripts/pipeline_definitivo.py --abrir-test-ciego # solo al cerrar, una vez
```

Cada etapa guarda su tabla y se omite si el fichero ya existe, de modo que una ejecución
interrumpida se retoma donde estaba. Tiempo total aproximado: 25 minutos.

### Artefactos

| Ruta | Contenido |
|---|---|
| `scripts/pipeline_definitivo.py` | Pipeline completo, catorce etapas |
| `src/entrenamiento/dias_secos.py` | Corrección de `consecutive_dry_days` |
| `data/models/modelo_definitivo.pkl` | Modelo, calibrador, variables y configuración |
| `docs/technical/pipeline_*.csv` | Una tabla por etapa |
| `docs/technical/importancia_permutacion.csv` | Importancia de las 48 variables |
| `docs/technical/errores_por_mes.csv` | Recall mensual |
| `docs/technical/umbrales_riesgo.json` | Cortes de los cuatro niveles |
| `docs/technical/pipeline_definitivo_config.json` | Configuración completa de la ejecución |

La configuración se serializa junto al modelo, de modo que cualquier resultado puede rastrearse
hasta los parámetros exactos que lo produjeron.

### Verificación del propio pipeline

Antes de ejecutarlo sobre los datos reales, las catorce etapas se validaron sobre un dataset
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
