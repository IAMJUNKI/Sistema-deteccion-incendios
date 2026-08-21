# Documentación del Modelo Predictivo

> **Propósito.** Registro vivo de todas las decisiones de modelado del TFM: qué se entrena,
> con qué datos, y **por qué** se ha elegido cada cosa. Este documento se actualiza en el mismo
> commit que el código al que se refiere. Si una decisión cambia, se edita aquí y se anota
> en el historial de revisiones del final.

| | |
|---|---|
| **Proyecto** | Sistema Predictivo de Anticipación de Incendios Forestales — Galicia |
| **Estado** | 🟡 En desarrollo — código de entrenamiento escrito y validado, pendiente de ejecución completa |
| **Última actualización** | 2026-08-18 |
| **Rama** | `main` |

---

## 1. Objetivo

### 1.1 Qué predice el sistema

Para cada **celda de 1 km² de Galicia** y cada **día**, estimar la probabilidad de que se
produzca el **inicio** de un incendio forestal (`target ∈ {0,1}`), usando exclusivamente
información disponible **antes** de ese día.

Es un problema de **clasificación binaria supervisada sobre datos tabulares con desbalanceo
extremo** (prevalencia ≈ 2,6·10⁻⁵, es decir 1 positivo por cada ~38.000 filas).

Se predice la **ignición**, no la propagación. Una celda que arde tres días seguidos genera un
único positivo, el del día de inicio. Etiquetar los días sucesivos convertiría el problema en la
regla trivial «si ayer ardía, hoy también», que es fuga temporal y no tiene valor preventivo.

### 1.2 Qué modelos se entrenan

Dos familias de árboles como candidatos principales, más dos baselines de referencia:

| Rol | Modelo | Motivo |
|---|---|---|
| Principal | **LightGBM** | Candidato a modelo operativo |
| Contraste | **XGBoost** | Segunda opinión con algoritmo de split distinto |
| Baseline lineal | **Regresión Logística** | Referencia interpretable obligatoria |
| Baseline ensemble | **Random Forest** | Referencia de bagging (solo sobre el set submuestreado) |

### 1.3 La pregunta experimental

El experimento central **no** es «qué modelo gana», sino **cuánto cuesta anticipar 24 horas**.
Cada modelo se entrena dos veces sobre exactamente el mismo split y las mismas métricas:

| | Variante **Nowcast** | Variante **Previsión T-1** |
|---|---|---|
| Features | Meteorología del **día D** | Meteorología del **día D-1** |
| Target | Incendio del día D | Incendio del día D |
| Responde a | *¿Hoy hay condiciones de incendio?* | *¿Mañana habrá condiciones de incendio?* |
| Utilidad | Diagnóstico | **Alerta temprana (objetivo del TFM)** |

La diferencia de rendimiento entre ambas columnas es una medida limpia y controlada de la
**degradación por anticipación temporal**, que la memoria (§2.1 y §3.4) declara como
contribución científica central y que hasta ahora no tenía ningún experimento que la sustentara.

---

## 2. Datos

### 2.1 Dataset utilizado

`dataset_maestro_{2019..2024}.parquet` — seis ficheros, uno por año.

| Propiedad | Valor |
|---|---|
| Granularidad | 1 fila = (celda de 1 km², día) |
| Celdas | 30.697 (rejilla Fase 1, CORINE 2018, EPSG:25829) |
| Cobertura temporal | 2019 – 2024 (6 años completos) |
| Filas totales | 67.287.821 |
| Columnas | 20 |
| Positivos totales | 1.761 |
| Prevalencia global | 2,6·10⁻⁵ |
| Tamaño en disco | 2,4 GB |

**Ubicación.** Los ficheros **no están en el repositorio y no deben estarlo nunca**. Cada
persona los coloca en su ruta local y la declara en `.env`:

```
DATA_ROOT=<ruta local>/data/raw/dataset_maestro/
```

Ruta recomendada dentro del repo (ya cubierta por `.gitignore`):
`Sistema-deteccion-incendios/data/raw/dataset_maestro/`

> ⚠️ **Regla estricta del equipo.** Los datasets nunca se hacen `git add`, `commit` ni `push`.
> El `.gitignore` bloquea `data/raw/dataset_maestro/`, `**/dataset_maestro_*.parquet`,
> `Datos/`, `misc/Dataset/` y `data/interim/`. Verificado con `git check-ignore -v`.

### 2.2 Distribución de la variable objetivo

| Año | Filas | Días | Positivos | Prevalencia |
|---|---:|---:|---:|---:|
| 2019 | 11.204.405 | 365 | 296 | 2,6·10⁻⁵ |
| 2020 | 11.235.102 | 366 | 372 | 3,3·10⁻⁵ |
| 2021 | 11.204.405 | 365 | 172 | 1,5·10⁻⁵ |
| 2022 | 11.204.405 | 365 | **703** | 6,3·10⁻⁵ |
| 2023 | 11.204.405 | 365 | **102** | 9,1·10⁻⁶ |
| 2024 | 11.235.102 | 366 | 116 | 1,0·10⁻⁵ |

**La variabilidad interanual es enorme**: 2022 tiene 6,9 veces más incendios que 2023. Esto
condiciona todo el diseño de la evaluación (§2.5) y es la razón por la que un test de un solo
año no permite ordenar modelos.

### 2.3 Variables

| Bloque | Columnas | Notas |
|---|---|---|
| Identificación | `cell_id`, `fecha` | Clave primaria compuesta |
| Meteorología directa | `tmax_vc`, `rhmin_vc`, `vmax_vc`, `prec_dia` | Ventana crítica 12–18 h hora local |
| Memoria climática | `prec_acum_7d`, `prec_acum_30d`, `tmax_media_7d` | Ventanas móviles |
| Índice de riesgo | `alerta_30_30` | Regla 30 °C / 30 % HR |
| Topografía | `altitud_media`, `pendiente_media`, `orientacion_media`, `orientacion_clase` | Copernicus DEM GLO-30 |
| Combustible | `combustible_clase`, `combustible_pct_forestal` | CORINE Land Cover 2018 |
| Calendario | `mes`, `dia_semana`, `es_finde` | Derivadas de la fecha |
| Objetivo | `target` | NASA FIRMS (ver §2.4) |

**Nulos.** Únicamente en `combustible_clase` y `combustible_pct_forestal`: 134 celdas (0,44 %)
sin cobertura CORINE, constante en los seis años. Se tratan como categoría propia
(`sin_dato`) en lugar de imputarse — la ausencia de cobertura es informativa, no aleatoria.

**Categóricas recuperadas.** `combustible_clase` y `orientacion_clase` son strings y los
experimentos anteriores del repositorio las **descartaban**. Se recuperan como categóricas
nativas de LightGBM/XGBoost: la clase de combustible (matorral vs. coníferas vs. agrícola) es
uno de los predictores físicamente más relevantes y hasta ahora solo entraba de forma
degradada a través de `combustible_pct_forestal`.

### 2.4 Origen de la variable objetivo

El target procede de **NASA FIRMS** (focos de calor satelitales agregados a celda·día), **no**
del EGIF del MITECO. Se evaluó EGIF como fuente primaria por su mayor fiabilidad (registros
auditados en campo), pero el XML disponible cubre únicamente **2014–2017**, sin solape con el
periodo de entrenamiento 2019–2024. EGIF queda como validación cruzada independiente para el
periodo que sí cubre.

Limitación reconocida: FIRMS incluye ruido térmico antropogénico (quemas agrícolas
autorizadas, actividad industrial) y sufre omisiones por nubosidad. Ambos efectos se documentan
como fuente de error irreducible del ground truth.

### 2.5 Split temporal

Se **prohíbe la validación cruzada aleatoria**: mezclar días de distintos años en train y test
permite al modelo aprender la meteorología concreta de un episodio y reconocerla en el otro
lado del split. La validación es estrictamente temporal.

**Protocolo de desarrollo** (para entrenar, tunear y calibrar):

```
Train        2019 · 2020 · 2021     840 incendios     33,6 M filas
Validación   2022                   703 incendios     11,2 M filas
Test ciego   2023 · 2024            218 incendios     22,4 M filas
```

**Protocolo de reporte** (para el capítulo de resultados) — backtesting con ventana expansiva,
entrenando siempre solo con el pasado:

```
train ≤2021 → test 2022      train ≤2022 → test 2023      train ≤2023 → test 2024
```

Total evaluado en el reporte: **991 incendios** repartidos en tres años de test independientes,
frente a los 102 de un único test de 2023.

**Regla de higiene.** El test ciego (2023+2024) no se mira hasta que el modelo está congelado.
Todo el ajuste de hiperparámetros, del umbral operativo y de la calibración se decide con 2022.

### 2.6 ⚠️ Reparación del desfase temporal

Al auditar el dataset se detectó que **el shift T-1 no está aplicado**, pese a que se intentó:

```
prec_acum_7d  == suma(t-6 .. t)     [INCLUYE el día actual]     err_max = 0.000000
tmax_media_7d == media(t-6 .. t)    [INCLUYE el día actual]     err_max = 0.000000
alerta_30_30  == (tmax_vc>=30 & rhmin_vc<=30) de la misma fila:  100.00 %

dia_semana == (fecha - 1 día).dayofweek :  100.00 %   ← offset constante
mes        == (fecha - 1 día).month     :  100.00 %
```

`mes`, `dia_semana` y `es_finde` se calcularon sobre `fecha − 1 día` con coincidencia perfecta,
y cada fichero anual cubre del 2 de enero al 1 de enero siguiente. La interpretación es que se
construyó la tabla `(meteo de D, incendio de D)` y después se hizo `fecha = D + 1` para
materializar el desfase — pero al desplazar la fecha se desplazó **también** el target, de modo
que el shift nunca llegó a producirse.

El día de la semana lo corrobora. Reconstruyendo la fecha real de los 1.761 incendios:

```
fecha tal cual    : lun 17,5 %  mar 20,6 %  mié 11,1 %  jue 11,6 %  vie 13,6 %  sáb 12,7 %  dom 12,8 %
fecha menos 1 día : lun 20,6 %  mar 11,1 %  mié 11,6 %  jue 13,6 %  vie 12,7 %  sáb 12,8 %  dom 17,5 %
```

Con `fecha − 1` el pico cae en **domingo y lunes** (38,1 % entre ambos), el patrón conocido de
incendios de causa humana. Con `fecha` tal cual el pico caería en martes y el fin de semana
sería el mínimo, lo que carece de sentido físico.

**Implicación.** Los resultados previos del repositorio (ROC-AUC 0,8377) son de **nowcasting**,
no de previsión a 24 h, y no son reproducibles en producción, donde el día D no se conoce.

**Tratamiento adoptado.** Se entrenan las dos variantes (§1.3) en lugar de descartar una. La
reparación se aplica con `groupby("cell_id").shift(1)` sobre las 7 variables meteorológicas,
concatenando los seis años para no perder los eneros; la rejilla está completa (30.697 celdas ×
365/366 días sin huecos), así que la operación es exacta. Las variables de calendario se
recalculan sobre la fecha corregida.

> **Pendiente de confirmación:** que Miquel, autor de los parquets, verifique si el join del
> target se hizo antes o después del relabel de fecha. La evidencia es estructural y
> estadística, no un testimonio directo.

---

## 3. Decisiones técnicas (el porqué)

### 3.1 Por qué modelos de árboles con boosting

**Por qué árboles y no redes neuronales.** El dataset es tabular, heterogéneo (continuas,
categóricas, binarias) y con solo 1.761 positivos. Los modelos de gradient boosting sobre
árboles son el estado del arte reconocido en datos tabulares y, sobre todo, **no necesitan
escalado ni codificación previa**, capturan interacciones no lineales (temperatura alta *y*
humedad baja *y* pendiente sur) sin declararlas, y no colapsan cuando el número de positivos es
de tres cifras. Una red neuronal con 1.761 ejemplos positivos sobreajustaría sin aportar nada
que los árboles no capturen ya.

**Por qué LightGBM como principal.** Cuatro razones concretas para este problema:
- **Escala.** Es *histogram-based*: agrupa los valores continuos en 255 bins antes de buscar
  splits, lo que hace que 33 M × 17 se entrene en minutos en un portátil, no en horas.
- **NaN nativos.** Aprende la dirección óptima para los valores faltantes en cada split. Las
  134 celdas sin CORINE no requieren imputación arbitraria.
- **Categóricas nativas.** Consume `combustible_clase` y `orientacion_clase` sin one-hot,
  buscando la partición óptima del conjunto de categorías. Con one-hot, un árbol necesitaría
  varios niveles para aislar «matorral o coníferas».
- **Continuidad.** Es el modelo que el equipo ya declaró operativo, así que los resultados
  nuevos son directamente contrastables con los anteriores.

**Por qué XGBoost como contraste.** Aporta una segunda opinión *real*, no una variación del
mismo modelo: crece los árboles por niveles (*level-wise*) frente al crecimiento por hojas
(*leaf-wise*) de LightGBM, y aplica regularización L1/L2 explícita sobre los pesos de las hojas.
Si ambos coinciden, la conclusión es robusta al algoritmo; si divergen, sabemos que el resultado
depende del sesgo inductivo y hay que decirlo.

**Por qué Random Forest solo como baseline.** Con 33 M de filas el consumo de memoria es
prohibitivo (guarda las muestras en cada nodo), no admite NaN ni categóricas, y sus
probabilidades salen mal calibradas a prevalencia extrema porque promedia votos de árboles
completamente crecidos. Se entrena únicamente sobre el conjunto submuestreado, como referencia
de bagging frente a boosting.

**Por qué se mantiene la Regresión Logística.** Un TFM necesita demostrar que la complejidad
está justificada. Si un modelo lineal con 17 variables alcanzase un rendimiento similar, la
elección correcta sería el modelo lineal. Es la referencia que hace defendible el salto a
árboles, y además sus coeficientes son directamente interpretables ante un tribunal.

### 3.2 Por qué esta estrategia de desbalanceo

La prevalencia es **2,6·10⁻⁵**: un positivo por cada ~38.000 filas. Entrenar sin tratar el
desbalanceo produce un modelo que predice cero en todo el territorio y acierta el 99,997 %.

**Estrategia adoptada: submuestreo aleatorio de negativos + recalibración.**

**(a) Submuestreo aleatorio 1:50.** Se conserva el **100 % de los positivos** y se muestrean
negativos **uniformemente** hasta una proporción 1:50.

> ⚠️ **Decisión revertida tras medirla.** El diseño inicial estratificaba el muestreo hacia
> *negativos difíciles* (días secos y cálidos sin ignición, `hard_fraction = 0.8`), con el
> argumento clásico de que así el modelo aprende la frontera real en lugar de separar
> trivialmente el invierno. **Medido, hace justo lo contrario.**
>
> Al llenar el entrenamiento de días secos y cálidos, positivos y negativos quedan
> meteorológicamente casi idénticos: la meteorología deja de discriminar y lo único que separa
> unos de otros es la **ubicación de la celda**. El modelo memoriza qué celdas ardieron, y eso
> no transfiere a otro año. Es un desplazamiento de covariables introducido por el propio
> muestreo: la distribución de entrenamiento deja de parecerse a la de evaluación, que es el
> año completo.
>
> El síntoma que lo delató fue que la regresión logística puntuaba **mejor en validación que en
> entrenamiento** (ROC-AUC 0,8455 frente a 0,7981). Un hueco negativo solo puede significar que
> el conjunto de entrenamiento es artificialmente más difícil que la población real.
>
> Barrido sobre validación 2022, variante `t1`
> (`docs/technical/model_v1_fixed_hf*_resultados.csv`):
>
> | `hard_fraction` | LogReg ROC-AUC | LogReg Recall@5 % | LightGBM ROC-AUC | LightGBM Recall@5 % |
> |---|---|---|---|---|
> | **0,0** (aleatorio puro) | **0,9434** | **78,5 %** | **0,9088** | **42,0 %** |
> | 0,18 (proporción natural) | 0,9454 | 76,2 % | 0,8924 | 13,4 % |
> | 0,5 | 0,9245 | 68,4 % | 0,8440 | 3,1 % |
> | 0,8 (diseño inicial) | 0,8455 | 39,8 % | 0,5549 | 5,3 % |
>
> El concepto de negativo difícil sigue siendo válido, pero pertenece a la **evaluación**
> —medir el rendimiento restringido a los días de riesgo real, como ya argumenta §5.4 de la
> memoria del TFM— y no al muestreo de entrenamiento. El parámetro se conserva en el fichero de
> configuración (`hard_fraction`, por defecto `0.0`) para poder reproducir el barrido.

> Nota de honestidad: los experimentos anteriores del repositorio llamaban «Hard Negative
> Mining» a un `negatives.sample(n=...)` uniforme. Tras este hallazgo, el muestreo uniforme
> resulta ser la opción correcta — pero por razones que aquellos experimentos no documentaban,
> y el nombre sigue sin corresponder a lo que se hace.

**(b) Ponderación de clase moderada.** `scale_pos_weight` entre 3 y 10 *combinada* con (a), no
como sustituto. Aplicada sola, sobre 33 M de filas es lenta y el tamaño efectivo de la clase
positiva sigue siendo 840: pesar más los mismos 840 ejemplos no crea información nueva.

**(c) Recalibración obligatoria a la prevalencia real.** Submuestrear negativos 1:50 infla las
probabilidades predichas aproximadamente 50 veces. Un modelo entrenado así puede decir «12 % de
probabilidad» donde la probabilidad real es del orden de 10⁻³.

Se corrige en dos pasos: desplazamiento del logit por el ratio de priors,
`log(π_train / π_real)`, y después **regresión isotónica ajustada sobre 2022** (año de
validación, con prevalencia real intacta). Sin esto, los umbrales de riesgo
Bajo / Moderado / Alto / Extremo que promete el proyecto no pueden construirse, porque no
existiría ninguna probabilidad con significado físico que umbralizar.

**Por qué se descarta SMOTE explícitamente.** Es la técnica que un tribunal preguntará, así que
conviene tener la respuesta escrita:
- **Genera casos físicamente imposibles.** Interpola en el espacio de features y produce celdas
  con, por ejemplo, altitud 340 m, combustible matorral y una combinación meteorológica que no
  ocurre en la naturaleza. El modelo aprende de territorio inventado.
- **Es inviable a esta escala.** Equilibrar 1:1 exigiría sintetizar decenas de millones de
  positivos.
- **Destruye la calibración**, que es precisamente lo que este sistema necesita para los
  umbrales de riesgo.
- **Introduce fuga espacio-temporal.** Interpola entre incendios de años y comarcas distintos,
  creando positivos que mezclan información de ambos lados del split temporal.

### 3.3 Por qué estas métricas

**Métricas descartadas y por qué** (importante dejarlo por escrito):

| Métrica | Por qué no |
|---|---|
| **Accuracy** | Predecir siempre «no incendio» da 99,997 %. No informa de nada. |
| **F1 con umbral 0,5** | El 0,5 carece de significado tras submuestrear y recalibrar. Solo se reporta en el umbral operativo elegido, y como secundaria. |
| **ROC-AUC como métrica única** | Está inflada por los negativos invernales triviales: acertar millones de ceros fáciles sube el AUC sin aportar valor operativo. Se conserva por comparabilidad con la literatura, declarada como secundaria. |

**Métricas principales:**

- **PR-AUC** vía `average_precision_score`. Es la métrica adecuada a baja prevalencia porque
  ignora los verdaderos negativos, que aquí son el 99,997 % de los datos.
  *No* se usa `auc(recall, precision)` —lo que hacía `metrics.py`— porque interpola linealmente
  la curva Precision-Recall e introduce un sesgo optimista.

- **Lift sobre la prevalencia** (`PR-AUC / prevalencia`). Un PR-AUC de 7·10⁻⁵ suena a fracaso
  absoluto y en realidad es ~8 veces mejor que el azar. Sin esta normalización la cifra es
  ilegible para un lector no especialista.

- **Recall @ FPR ≤ 5 %.** La métrica operativa: qué fracción de incendios se detecta alertando
  como máximo el 5 % del territorio. Responde a la pregunta que hace un jefe de brigada.

- **Recall @ top-1 % de celdas.** Más honesta operativamente: el 5 % de Galicia son 1.535 km²,
  que ninguna brigada patrulla. El 1 % son 307 celdas, que sí.

**Métricas de calibración:** **Brier score** y **curva de fiabilidad**, pero únicamente
*después* de recalibrar. Antes de la recalibración no miden calidad del modelo, sino el sesgo
introducido por el submuestreo.

**Tratamiento de empates y separación de entradas.** Detectado al ejecutar la primera versión
del pipeline: las métricas basadas en umbral se disparaban de forma imposible (un modelo con
ROC-AUC 0,71 aparentaba un 72 % de recall al 5 % de falsos positivos, cuando una curva ROC que
pasara por ese punto exigiría un AUC de al menos 0,82).

La causa son los **empates masivos**. A esta escala millones de filas comparten la misma
puntuación: un modelo de árboles asigna el mismo valor a todas las que caen en la misma hoja, y
la regresión isotónica lo agrava porque es una función escalonada. Cuando el umbral cae sobre
una meseta, la comparación `>=` se lleva la meseta entera y el FPR real supera con mucho el
objetivo.

Se corrige por dos vías, y ambas quedan como norma del proyecto:

1. **Cada familia de métricas se calcula sobre la entrada que le corresponde.** Las de
   ordenación (PR-AUC, ROC-AUC, recalls) sobre la **puntuación cruda** del modelo; las de
   calibración (Brier) sobre la **probabilidad calibrada**. La calibración es monótona y en
   teoría no altera el orden, pero los empates que introduce sí distorsionan cualquier métrica
   basada en umbral.
2. **El FPR realmente alcanzado se reporta siempre** (`fpr_achieved`), junto al recall. Si hay
   empates en el umbral se pasa a comparación estricta, que es la opción conservadora. Un
   recall a FPR fijo sin el FPR alcanzado al lado no es auditable.

**Cuantificación de la incertidumbre — no opcional:** **bootstrap sobre los positivos con IC al
90 %**. Con 218 incendios en el test ciego, un punto porcentual de recall equivale a dos
incendios. Sin intervalos de confianza no es posible ordenar modelos, y hacerlo igualmente es
el error metodológico que se está corrigiendo respecto a la versión anterior de los resultados.

> **Criterio de decisión.** Si los intervalos de confianza de dos configuraciones se solapan, la
> diferencia se declara **no concluyente** y se prefiere el modelo más simple. No se elige
> ganador por décimas.

### 3.4 Ingeniería de variables

El primer entrenamiento con las 19 variables originales dejó un diagnóstico incómodo: ROC-AUC
global de 0,90–0,94 pero **Recall @ top-1 % diario** de apenas 2–4 %. Las dos métricas miden
cosas distintas:

- El **ROC-AUC global** mezcla todos los días del año, así que se lleva casi gratis la
  discriminación estacional: agosto arde, enero no.
- El **Recall @ top-1 % diario** obliga a elegir las 307 celdas más peligrosas *de hoy*. La
  estacionalidad no ayuda, porque todas las celdas comparten el mismo día.

Operativamente solo importa la segunda: una brigada no necesita que le digan que en verano hay
más riesgo, sino a qué monte ir esta mañana.

#### Diagnóstico (`notebooks/02_feature_engineering.ipynb`)

Se entrenó una regresión logística con **un solo grupo de variables cada vez**, evaluando sobre
validación 2022:

| grupo | n | ROC-AUC |
|---|---|---|
| memoria (`prec_acum_*`, `tmax_media_7d`) | 4 | 0,9264 |
| físicas (VPD, Nesterov…) | 4 | 0,9263 |
| meteo cruda | 4 | 0,9256 |
| anomalías | 4 | 0,8678 |
| **calendario solo** | 5 | **0,8568** |
| terreno | 4 | 0,6851 |
| relativas (`_z_dia`) | 4 | 0,6849 |
| **todas juntas** | **29** | **0,9270** |

Cinco variables de calendario dan 0,857 de los 0,927 finales, y 29 variables juntas apenas
mejoran a 4 (0,9270 frente a 0,9264). La conclusión es que el modelo vivía de la estacionalidad.

#### La métrica que separa lo decorativo de lo operativo

Se introdujo el **ROC-AUC calculado dentro de cada día**. Una variable estacional da 0,500 aquí
por construcción, porque es idéntica en toda Galicia esa jornada:

| variable | AUC global | AUC intra-día |
|---|---|---|
| `rhmin_vc` / `rhmin_vc_z_dia` | 0,911 / 0,749 | **0,720** |
| `nesterov` | 0,906 | **0,702** |
| `vpd` | 0,916 | 0,659 |
| `pendiente_media` | 0,646 | 0,649 |
| `altitud_media` | 0,636 | 0,644 |
| `tmax_vc` | 0,897 | 0,583 |
| `tmax_vc_anom` | 0,855 | 0,522 |
| `mes`, `dia_anio_sin/cos`, `es_finde` | 0,51–0,84 | **0,500** exacto |

Que el calendario salga en 0,500 clavado valida que la métrica mide lo que dice medir.

**Hallazgo principal: manda la humedad, no la temperatura.** `rhmin_vc` discrimina espacialmente
a 0,720, mientras `tmax_vc` se queda en 0,583 y su anomalía en 0,522. La temperatura indica en
qué *estación* estamos; la humedad mínima indica en qué *ladera* está hoy el peligro.

#### Las tres familias construidas (`src/features/derived.py`)

1. **Anomalías frente a la climatología de la celda** (`*_anom`): ¿hace más calor del que suele
   hacer *aquí* en *este mes*? Elimina el sesgo de altitud y latitud — 28 °C en Ancares es una
   anomalía extrema y en Ourense es un martes de julio. Se agregan por celda y **mes**, no por
   día del año: con tres años de histórico, un promedio por día concreto tendría tres muestras.
2. **Posición relativa dentro del día** (`*_z_dia`): ¿cómo de seca está esta celda comparada con
   el resto de Galicia *hoy*? Al construirse día a día, la estacionalidad desaparece por
   completo. Nota: dentro de un día es una transformación monótona, así que **no añade señal
   espacial nueva** — lo que aporta es la misma señal despojada del ruido estacional, que es lo
   que permite a un modelo global aprovecharla.
3. **Índices físicos de sequedad**: VPD, días sin lluvia encadenados, índice de Nesterov y ratio
   térmico-eólico. Combinan magnitudes como dice la física del fuego, en lugar de esperar a que
   el árbol lo descubra desde cero con 840 incendios.

Más dos **interacciones** combustible × sequedad, porque la sequedad solo importa donde hay algo
que arda.

#### Anti-leakage de los estadísticos de referencia

| estadístico | ajustado con | por qué |
|---|---|---|
| Climatología por celda y mes | **solo años de entrenamiento** | Es un parámetro aprendido; usar validación o test sería filtrar información |
| Estadísticos espaciales diarios | todos los años evaluados | Son propios de cada fecha («el 14 de agosto de 2022» no existe en el histórico). No usan la variable objetivo y **están disponibles en producción**: el día que se predice se dispone de la previsión de MeteoGalicia para toda Galicia, así que su media y desviación espaciales son calculables antes de predecir |

Ambos se cachean en `data/processed/derived_context/` porque ajustarlos exige dos recorridos
completos del histórico.

#### Selección final

| conjunto | n | AUC global | AUC intra-día |
|---|---|---|---|
| ACTUAL (absolutas) | 17 | 0,8753 | 0,6505 |
| **PROPUESTO (derivadas)** | 20 | 0,8840 | **0,7161** |
| AMBOS | 31 | 0,8726 | 0,6987 |

Añadir las absolutas de vuelta (`AMBOS`) **empeora**: no faltaban variables, distraían las que
había. Se eliminó además la redundancia detectada (|ρ| > 0,8): `tmax_vc`↔`tmax_media_7d` (0,90),
`vpd`↔`rhmin_vc` (0,88), `dias_sin_lluvia`↔`prec_dia` (0,84).

Se **conservan las categóricas**, que los experimentos anteriores del repositorio descartaban
por ser cadenas de texto. Tienen un factor 6 entre clases:

| `combustible_clase` | lift |
|---|---|
| matorral | 1,76 |
| pastizal | 1,47 |
| bosque_coniferas | 1,13 |
| bosque_frondosas | 0,51 |
| agrícola | 0,30 |

Es exactamente el matiz que se perdía al resumirlo todo en `combustible_pct_forestal`.

### 3.5 Reproducibilidad

| Decisión | Justificación |
|---|---|
| Semilla fija (`random_state=42`) en muestreo, split y modelos | Que dos personas del equipo obtengan el mismo número |
| Hiperparámetros en `configs/*.yaml`, no en el código | Auditables y versionables sin tocar la lógica |
| Modelo serializado en `data/models/*.pkl` | Separar **entrenar** de **predecir**: hoy el pipeline de inferencia reentrena en cada ejecución, así que sirve un modelo distinto cada vez |
| Rutas vía `DATA_ROOT` en `.env` | Elimina las tres convenciones de rutas incompatibles que conviven hoy en el repositorio |
| Nombre del artefacto con años y variante (`v1_lightgbm_t1_tr2019-2021_val2022.joblib`) | Un mismo modelo se reentrena con distintos cortes temporales. Sin los años en el nombre, los pliegues se sobrescribían entre sí y el fichero en disco no correspondía al experimento reportado |
| Contexto de derivadas cacheado en `data/processed/derived_context/` | Ajustar climatología y estadísticos diarios exige dos recorridos completos del histórico (~2 min); se reutilizan entre ejecuciones |

#### Salvaguardas contra el fallo silencioso

Tres controles añadidos tras encontrarse el problema correspondiente en una ejecución real:

| Control | Qué evita |
|---|---|
| **Comprobación de cobertura** en `predict_years` contra los metadatos del parquet | Se observó la pérdida transitoria de un bloque entero (4.000 celdas, 46 incendios) sin que se lanzara ninguna excepción: dos modelos evaluaron sobre 9.744.405 filas en vez de 11.204.405 y las métricas salieron con aspecto perfectamente normal. Solo se detectó al comparar recuentos entre modelos. Ahora, si falta una fila, la ejecución aborta |
| **`--protocol expanding` exige `--evaluate-test`** | La ventana expansiva evalúa sobre años que en el protocolo fijo son test ciego. La primera versión los tocaba sin pedirlo, quemando el test sin avisar |
| **Parada temprana con conjunto de validación** | Sin ella, los modelos de boosting agotaban siempre los árboles configurados (400 × 63 hojas = 25.200 hojas para 42.808 filas con 840 positivos) |

---

## 4. Implementación

### 4.1 Estructura del código

| Fichero | Responsabilidad |
|---|---|
| `src/models/dataset.py` | Resolución de `DATA_ROOT`, corrección del desfase temporal, recorrido por bloques de celdas, *hard negative mining* |
| `src/models/estimators.py` | Construcción de los cuatro modelos y su preprocesado; importancia de variables |
| `src/models/calibration.py` | Corrección de prior + isotónica; traducción a niveles de riesgo |
| `src/models/evaluate.py` | Métricas, recall diario a top-k y bootstrap |
| `scripts/train_model.py` | Orquestador: entrena, calibra, evalúa y serializa |
| `configs/model_v1.yaml` | Hiperparámetros, split, umbrales y salidas |

Los módulos previos (`train_baseline.py`, `metrics.py`, `train_ensemble_advanced.py`) **no se
han modificado**, para que los resultados anteriores sigan siendo reproducibles y contrastables.

### 4.2 Procesamiento por bloques

Un año son 11,2 millones de filas y no cabe en memoria junto con el resto del pipeline. Los
ficheros se recorren por **bloques de celdas** (4.000 celdas ≈ 1,5 M de filas). Cada bloque
carga además el último día del fichero del año anterior como margen, de modo que el
desplazamiento temporal no pierde el 1 de enero.

Consecuencia importante para la validez del experimento: **la población evaluada es idéntica en
las dos variantes**. Verificado en la prueba de humo — mismas filas, mismas claves
`(cell_id, fecha_real)` y mismo vector de target en `nowcast` y en `t1`. Sin esa garantía, la
diferencia entre ambas mediría también un cambio de población, no solo el efecto del desfase.

### 4.3 Separación entre entrenar y predecir

`scripts/train_model.py` produce un artefacto serializado en `data/models/` que contiene el
modelo, el calibrador ajustado, la lista de variables, los umbrales de riesgo y una copia de la
configuración usada. El pipeline de inferencia carga ese artefacto en lugar de reentrenar.

Esto corrige un problema del pipeline anterior, donde `run_daily_inference.py` reentrenaba el
modelo en cada ejecución: el dashboard servía un modelo distinto cada vez que arrancaba, y no
había forma de reproducir un número enseñado en una captura.

### 4.4 Coste computacional medido

Prueba de humo sobre un año de entrenamiento y uno de validación:

| Fase | Tiempo |
|---|---|
| Pasada 1 — contar clases (4 columnas × 11,2 M filas) | 4 s |
| Pasada 2 — muestrear (19 columnas × 11,2 M filas) | 13 s |
| **Inferencia sobre el año de validación completo** | **24 min** |
| Calibración isotónica + métricas + bootstrap | 9 s |

Leer los parquets es barato; **el coste está en la inferencia sobre el año completo**, y depende
mucho del modelo. Los baselines de scikit-learn pasan por `ColumnTransformer` y son órdenes de
magnitud más lentos que LightGBM y XGBoost, que predicen de forma nativa sobre las categóricas.

Por eso los baselines van dimensionados a la baja en la configuración: su papel es justificar el
salto a boosting, no competir por ser el modelo operativo. Para iterar rápido:

```bash
python -m scripts.train_model --models lightgbm xgboost
```

### 4.5 Protocolo de ejecución

```bash
# Desarrollo: entrena y calibra. NO toca el test ciego.
python -m scripts.train_model --config configs/model_v1.yaml

# Solo una variante o un modelo concreto
python -m scripts.train_model --models lightgbm --modes t1

# Reporte multi-año con ventana expansiva
python -m scripts.train_model --protocol expanding

# Test ciego: solo una vez, con el modelo ya congelado
python -m scripts.train_model --evaluate-test
```

El test ciego (2023 + 2024) exige el flag explícito `--evaluate-test`. No es un detalle
cosmético: es lo que impide que se acabe ajustando el modelo contra el test sin darse cuenta,
que es exactamente lo que ocurrió en la versión anterior del pipeline.

---

## 5. Resultados

> ⚠️ **Resultados de validación, no definitivos.** Medidos sobre **2022**, el año de validación.
> El **test ciego (2023-2024) sigue sin abrirse** salvo por la ejecución accidental documentada
> en §7. No citar en la memoria hasta cerrar el ajuste de hiperparámetros.

Configuración: train 2019-2021 (840 incendios) · validación 2022 (703 incendios sobre
11.204.405 filas) · submuestreo aleatorio 1:50 · variables derivadas · parada temprana.
Fichero: `docs/technical/model_v1_fixed_derivadas_resultados.csv`.

### 5.1 Comparativa de modelos

| modelo | variante | ROC-AUC | Recall @ FPR 5 % | IC 90 % | Recall @ top 1 % diario | hueco train-val |
|---|---|---:|---:|---:|---:|---:|
| **LightGBM** | **t1** | 0,9332 | **78,24 %** | 75,68 – 80,80 % | 1,00 % | +0,052 |
| Reg. logística | t1 | **0,9417** | 75,82 % | 73,12 – 78,52 % | **6,12 %** | −0,007 |
| Reg. logística | nowcast | 0,9380 | 70,27 % | 67,43 – 73,26 % | 3,27 % | +0,006 |
| XGBoost | t1 | 0,9289 | 69,99 % | 67,13 – 72,97 % | 2,84 % | +0,035 |
| LightGBM | nowcast | 0,9358 | 67,99 % | 65,01 – 70,84 % | 2,70 % | +0,050 |
| XGBoost | nowcast | 0,9309 | 67,57 % | 64,86 – 70,55 % | 2,70 % | +0,040 |

**No hay ganador estadístico.** LightGBM `t1` (75,68 – 80,80 %) y la regresión logística `t1`
(73,12 – 78,52 %) tienen intervalos que se solapan. Por el criterio del proyecto —si los
intervalos se solapan la diferencia no es concluyente y se prefiere el modelo más simple— **el
modelo de referencia sigue siendo la regresión logística**, que además tiene hueco de
sobreajuste negativo mientras el boosting mantiene +0,04 a +0,05.

### 5.2 Efecto de las variables derivadas

Comparación directa contra el mismo protocolo con las variables absolutas originales
(`docs/technical/model_v1_fixed_ANTES_derivadas.csv`), en puntos porcentuales de Recall @ FPR 5 %:

| modelo | variante | absolutas | derivadas | Δ |
|---|---|---:|---:|---:|
| LightGBM | t1 | 27,17 % | 78,24 % | **+51,1** |
| XGBoost | t1 | 26,60 % | 69,99 % | **+43,4** |
| LightGBM | nowcast | 22,48 % | 67,99 % | **+45,5** |
| XGBoost | nowcast | 20,48 % | 67,57 % | **+47,1** |
| Reg. logística | t1 | 78,95 % | 75,82 % | −3,1 |
| Reg. logística | nowcast | 72,40 % | 70,27 % | −2,1 |

El efecto es **asimétrico y muy grande**: los modelos de árboles multiplican por tres su recall
y pasan de ir muy por detrás de la regresión logística a igualarla. La logística pierde entre
dos y tres puntos. La interpretación es que las derivadas codifican explícitamente lo que un
modelo lineal ya podía capturar por sí solo (relaciones suaves y monótonas), mientras que a los
árboles les evitan tener que descubrirlo con 840 incendios.

**El hueco de sobreajuste también se reduce** en los árboles: de +0,086 a +0,050 en LightGBM
`nowcast` y de +0,089 a +0,040 en XGBoost. Coherente con la lectura anterior.

### 5.3 El objetivo declarado NO se ha cumplido

La ingeniería de variables se hizo para mejorar la discriminación **dentro del día**, que es la
que decide a qué monte va la brigada. Ese objetivo **no se ha conseguido**:

| modelo | variante | top 1 % diario, absolutas | top 1 % diario, derivadas | Δ |
|---|---|---:|---:|---:|
| Reg. logística | t1 | 4,13 % | 6,12 % | +2,0 |
| LightGBM | nowcast | 2,13 % | 2,70 % | +0,6 |
| XGBoost | nowcast | 1,99 % | 2,70 % | +0,7 |
| XGBoost | t1 | 3,41 % | 2,84 % | −0,6 |
| LightGBM | t1 | 2,42 % | 1,00 % | −1,4 |
| Reg. logística | nowcast | 31,29 % | 3,27 % | **−28,0** |

Sigue en el rango del 1-6 %: alertando cada día el 1 % de las celdas de Galicia se capturan
menos de uno de cada quince incendios. Toda la ganancia se ha producido en el ranking **global**,
que es el que se beneficia de acertar la estacionalidad.

El cuaderno predijo una mejora del AUC intra-día de 0,6505 a 0,7161 que no se ha traducido en
esta métrica. Tres motivos posibles, aún sin discriminar: son métricas distintas (AUC medio
dentro del día frente a recall en la cola extrema del 1 %), la muestra del cuaderno era 1:150 y
no el año completo, y la lista final del `config` no es idéntica a la `PROPUESTO` del cuaderno
(se le añadieron las categóricas y la estacionalidad cíclica).

**Sobre la caída de 31,29 % a 3,27 %** en la regresión logística `nowcast`: ese 31 % lo sostenía
`prec_dia` **del mismo día**, que discrimina muy fuerte dentro de la jornada — si hoy llovió en
una celda y no en la vecina, se sabe cuál no arde. Al pasar a las derivadas se retiró `prec_dia`
cruda de la lista. Ese 31 % era en buena parte **circular**: depende de observar la lluvia del
propio día del incendio, que en producción no se tiene, sino una previsión. En la variante `t1`,
que es la honesta, la misma configuración ya solo daba 4,13 %.

### 5.4 Nowcast frente a previsión T-1

En los tres modelos y en los dos juegos de variables, **`t1` iguala o supera a `nowcast`**:

| modelo | nowcast | t1 | Δ |
|---|---:|---:|---:|
| LightGBM | 67,99 % | 78,24 % | +10,3 |
| Reg. logística | 70,27 % | 75,82 % | +5,6 |
| XGBoost | 67,57 % | 69,99 % | +2,4 |

Anticipar 24 horas **no degrada** el modelo, y aparentemente lo mejora. Contradice la hipótesis
inicial de §1.3 y la lectura preliminar de la primera ejecución.

La explicación más probable es que **FIRMS registra la fecha de detección satelital, no la de
ignición**. Si el satélite detecta el incendio en la pasada del día siguiente, la meteorología
etiquetada como `t1` es en realidad la del día en que se prendió. Encaja con el desfase de +1 día
detectado en la columna `fecha` (§2.6). **Pendiente de aclarar con el autor de los parquets antes
de escribir nada en la memoria**, porque decide cuál de las dos variantes es la correcta.

### 5.5 Importancia de variables

Las derivadas dominan el ranking de los tres modelos. Top 8 de LightGBM `t1`:

```
vpd · altitud_media · nesterov · dia_anio_sin · vmax_vc_z_dia · dia_anio_cos ·
pendiente_media · tmax_vc_anom
```

Y de XGBoost `t1`:

```
vpd · nesterov · dias_sin_lluvia · tmax_vc_anom · ratio_termico_eolico ·
combustible_clase · rhmin_vc_anom · dia_anio_sin
```

El **VPD encabeza ambos**, y los índices físicos de sequedad (Nesterov, días sin lluvia, ratio
térmico-eólico) ocupan la mitad de las posiciones. `combustible_clase`, que los experimentos
anteriores del repositorio descartaban por ser texto, entra en el top 6 de XGBoost.

La presencia de `dia_anio_sin` y `dia_anio_cos` en posiciones altas es coherente con §5.3: parte
importante de lo que el modelo usa sigue siendo estacionalidad.

### 5.6 Calibración y umbrales de riesgo

_(pendiente: falta la curva de fiabilidad y la tabla de umbrales Bajo/Moderado/Alto/Extremo)_

---

## 6. Limitaciones reconocidas

1. **Ground truth imperfecto.** FIRMS incluye ruido térmico antropogénico y omite detecciones
   por nubosidad. No es corregible con los datos disponibles.
2. **Potencia estadística limitada.** 1.761 positivos en seis años. Las diferencias pequeñas
   entre modelos no son detectables y no se reportarán como si lo fueran.
3. **Cobertura del suelo semi-estática.** CORINE 2018 se asume constante para 2019–2024.
4. **Sin variables de proximidad humana.** El 96 % de los incendios son de causa humana, pero
   el dataset no incluye distancia a carreteras ni a núcleos urbanos (OSM/CNIG). Es la
   ampliación con mayor recorrido esperable.
5. **Brecha de producción sin cuantificar.** El modelo aprende de reanálisis ERA5-Land y en
   producción debería inferir con previsión WRF de MeteoGalicia. Esa degradación adicional es
   distinta de la que mide el experimento Nowcast vs. T-1 y sigue pendiente.

---

## 7. Historial de revisiones

| Fecha | Cambio | Autor |
|---|---|---|
| 2026-08-18 | Creación del documento. Auditoría del dataset, detección del desfase temporal, plan de modelado aprobado (matriz Nowcast × T-1, split 2019-21/2022/2023-24, submuestreo dirigido + recalibración, métricas con IC bootstrap). | Equipo TFM |
| 2026-08-18 | Implementación del pipeline: `dataset.py`, `estimators.py`, `calibration.py`, `evaluate.py`, `train_model.py` y `configs/model_v1.yaml`. Prueba de humo superada de extremo a extremo; shift T-1 verificado columna a columna. | Equipo TFM |
| 2026-08-18 | Instalados LightGBM 4.7.0 y XGBoost 3.4.1. Datasets movidos a `data/raw/dataset_maestro/`. Primera ejecucion real: detectado y corregido un sesgo por empates en las metricas de umbral (metricas de orden sobre puntuacion cruda, de calibracion sobre probabilidad calibrada, y `fpr_achieved` siempre reportado). Resultados preliminares de LightGBM en 5.1. | Equipo TFM |
