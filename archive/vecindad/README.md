# Variables de contexto espacial (vecindad)

**Estado:** medido, verificado y **no adoptado**. El código funciona y está integrado tras una
bandera apagada por defecto; nadie lo usa todavía. Esta carpeta guarda la evidencia y el
razonamiento para que la decisión de adoptarlo o descartarlo se tome con datos y no de memoria.

**Fecha:** septiembre de 2026 · **Autor:** Miquel Jiménez · **Rama:** trabajo de modelado 2D

---

## 1 · Qué problema se intentaba resolver

El sistema aplana un datacubo en una tabla de celda-día. Al hacerlo, cada fila queda aislada: el
modelo no sabe que la celda contigua existe, ni que ayer ardió algo a cinco kilómetros.

Eso se traduce en una asimetría que estaba medida y documentada en la sección 9.3 del notebook
de descubrimiento: el modelo distingue muy bien **qué días** son peligrosos y bastante peor
**qué celdas** lo son.

```
ROC-AUC global      0,872     ← acierta el día
ROC-AUC intradía    0,751     ← acierta la celda
```

Acertar el día es un problema meteorológico y el modelo tiene la meteorología. Acertar la celda
es un problema espacial y el modelo no tiene nada espacial. La hipótesis era que devolver a cada
fila algo de contexto de su entorno cerraría parte de esa brecha.

## 2 · Qué NO se construyó, y por qué

La idea inmediata —promediar las variables meteorológicas de las ocho celdas contiguas— se
descartó **antes** de escribir código, tras medir sobre un día de agosto de 2022 la correlación
entre el valor de una celda y la media de sus vecinas:

| Variable | r con la media de sus vecinas | |
|---|---|---|
| `relative_humidity_min` | 0,9998 | redundante |
| `temperature_max` | 0,9996 | redundante |
| `elevation_mean` | 0,9894 | redundante |
| `scrub` | 0,7946 | aporta algo |
| `broadleaf_forest` | 0,7191 | aporta algo |

Son la misma variable. La explicación es que la meteorología procede de ERA5-Land, cuya
resolución nativa ronda los 9 km, interpolada después a 1 km: ocho celdas contiguas comparten
el mismo píxel de origen. Promediarlas produce una copia y reparte la importancia entre dos
columnas idénticas.

Lo que sí varía a escala de kilómetro es el **combustible** y, sobre todo, **dónde ha ardido
recientemente**.

## 3 · Qué se construyó

Ocho columnas, para dos radios (5 y 12 km) y dos ventanas (7 y 30 días):

- `igniciones_{radio}km_{ventana}d` — cuántas igniciones hubo en el entorno
- `dias_desde_ignicion_{radio}km` — cuánto hace de la más reciente, con tope en 90 días
- `forestal_vecindad_{radio}km` — fracción forestal media del entorno, estática

La justificación empírica del agrupamiento espacio-temporal, medida sobre 2022:

```
igniciones con otra a ≤12,5 km en los 10 días previos:   71,7 %
lo mismo para negativos tomados al azar:                 30,2 %
                                                 razón:   2,37x
```

El agrupamiento responde a causas reales: una ola de calor afecta a una comarca entera, y las
quemas agrícolas y los incendios intencionados se concentran en el espacio y en el tiempo.

### La trampa que había que evitar

El datacubo ya trae `is_near_ignition_25x25_10d`, que marca el entorno de una ignición durante
**el día del evento y los diez anteriores**. Incluir el propio día la convierte en la respuesta
disfrazada de pregunta, y por eso el contrato la prohíbe como predictor.

Aquí la ventana termina **estrictamente en D-1**. No es un detalle de implementación: es la
condición que hace legítimas estas variables. Se comprueba de dos maneras independientes,
descritas en la sección 6.

## 4 · Resultado

Año de validación 2022 · 1.659 igniciones sobre 10.804.365 filas · LightGBM idéntico en ambas
ramas, mismas filas de entrenamiento, misma semilla.

| | sin vecindad | con vecindad |
|---|---|---|
| recall @ 5 % FPR | 39,06 % (648) | **44,73 % (742)** |
| recall @ top 1 %/día | 7,47 % (124) | **9,28 % (154)** |
| ROC-AUC global | 0,8717 | 0,8872 |
| **ROC-AUC intradía** | **0,7507** | **0,7856** |

Comparación pareada sobre las mismas igniciones —no comparando intervalos marginales, que para
evaluaciones pareadas es demasiado conservador—:

| Métrica | Diferencia | IC 90 % | McNemar | Veredicto |
|---|---|---|---|---|
| recall @ 5 % FPR | **+5,67 pp** | [+3,98, +7,41] | p = 3,2·10⁻⁸ | concluyente |
| recall @ top 1 %/día | **+1,81 pp** | [+0,72, +2,83] | p = 0,0059 | concluyente |

Son **94 igniciones más detectadas al año** con el mismo presupuesto de vigilancia.

Lo que más sostiene la hipótesis no es el recall, sino dónde se produjo la mejora: el AUC
intradía sube de 0,751 a 0,786. La brecha que motivó todo el ejercicio se cerró por donde se
había predicho. Y `dias_desde_ignicion_5km` y `dias_desde_ignicion_12km` entran en los puestos
**2 y 3** de importancia sobre 69 variables, por delante de `vpd_mean` y de toda la topografía.

## 5 · Por qué no se ha adoptado

**El motivo es operativo, no estadístico.** Estas variables usan el registro de igniciones hasta
D-1. En la evaluación es legítimo y está verificado. Pero EGIF es un registro oficial que se
consolida con semanas de retraso, así que en operación real «lo que ardió ayer» puede no estar
disponible ayer.

Si la latencia de EGIF es de días, las variables sirven tal cual. Si es de semanas, hay que
reconstruirlas sobre una fuente con latencia menor (detecciones satelitales, por ejemplo) y
volver a medir, porque ya no serían las mismas variables.

**Esa pregunta —¿con qué latencia se consolida EGIF?— es la que bloquea la decisión, y es para
Alberto, que es quien maneja la fuente.**

Motivos secundarios, todos subsanables:

- El pipeline entregable es `scripts/pipeline_definitivo.py`, de Enrique. Portar estas variables
  allí es decisión del equipo, no unilateral.
- La selección de variables se corrió sobre 61 columnas; ahora habría 69 y dos de las nuevas
  entran en el top-3, así que el conjunto recomendado ya no sería el mismo.
- Los hiperparámetros se buscaron sin estas variables.

## 6 · Cómo se verificó que no hay fuga

Un resultado de +5,67 pp en un problema con prevalencia de 1,5·10⁻⁴ es exactamente el tipo de
cifra que suele venir de una fuga. Se comprobó por dos caminos independientes.

**Pruebas unitarias** (`tests/test_entrenamiento_vecindad.py`, grupo `TestSinFugaTemporal`),
sobre mallas sintéticas donde la respuesta correcta se conoce de antemano. La que manda:
una ignición aislada tiene que ver **0** en su propio historial; si viera 1, se estaría leyendo
a sí misma.

**Recuento por fuerza bruta sobre los datos reales** (`verificar_sin_fuga.py`), contando
igniciones una a una desde la tabla cruda, sin imágenes integrales ni sumas acumuladas —es
decir, por un algoritmo completamente distinto al que se quiere verificar—. Sobre 800 filas
del año de validación, la mitad positivas:

```
igniciones_5km_7d      OK      igniciones_12km_7d       OK
igniciones_5km_30d     OK      igniciones_12km_30d      OK
dias_desde_ignicion_5km  OK    dias_desde_ignicion_12km OK

SIN FUGA: todos los valores coinciden con el recuento manual hasta D-1

igniciones de la muestra sin ninguna otra cerca en los 30 días previos: 40
todas ellas ven 0 en su propio historial
```

## 7 · Un fallo que apareció por el camino

Al reunir el marco del contexto se hacían dos consultas solapadas —todos los positivos, más dos
días completos para fijar el rango temporal—. Las igniciones que caían en un día de borde
entraban **dos veces**, y el historial las contaba como dos incendios distintos.

Eran 2 de 5.659: lo bastante pocas para no moverse en ninguna métrica, y lo bastante graves para
que las variables dejaran de significar lo que dicen. Se detectó porque el número de igniciones
que anunciaba el registro no cuadraba con el que decía la malla.

Ahora `ajustar_contexto_espacial` rechaza el marco si detecta igniciones repetidas, y hay una
prueba de regresión (`test_avisa_si_hay_igniciones_repetidas`).

## 8 · Dónde está el código

**Sigue en su sitio y funcionando.** No se archivó porque está integrado y probado; lo que se
archiva aquí es la evidencia y el razonamiento.

| Ruta | Qué es |
|---|---|
| `src/entrenamiento/vecindad.py` | El módulo |
| `tests/test_entrenamiento_vecindad.py` | 26 pruebas |
| `src/entrenamiento/experimento.py` | Bandera `usar_vecindad`, **apagada por defecto** |
| `scripts/entrenar_egif.py` | Opción `--con-vecindad` |

La bandera está apagada a propósito: encenderla sola cambiaría en silencio las cifras que ya
están publicadas en el notebook de descubrimiento y en el entregable técnico.

Para reproducir el resultado de la sección 4:

```bash
python archive/vecindad/evaluar_vecindad.py        # la comparación con/sin
python scripts/entrenar_egif.py --modelos lightgbm --con-vecindad   # la ruta integrada
```

Las dos rutas dan cifras idénticas hasta el último dígito, incluida la iteración de parada
temprana (669). Es la comprobación de que la integración hace lo mismo que el experimento.

## 9 · Contenido de esta carpeta

| Fichero | Qué es |
|---|---|
| `README.md` | Este documento |
| `evaluar_vecindad.py` | El experimento comparativo con test pareado |
| `verificar_sin_fuga.py` | El recuento por fuerza bruta de la sección 6 |
| `resultados.csv` | Métricas completas de las dos ramas |
| `pareado.csv` | Salida del test pareado |

## 10 · Cambios colaterales, que sí se quedan

Dos cambios salieron de este trabajo pero **no dependen de él** y se mantienen aplicados:

**`iter_evaluacion` acepta `columnas_extra`** (`src/entrenamiento/datos.py`). No había forma de
arrastrar columnas no predictoras hasta los lotes de evaluación, aunque su hermana
`muestrear_entrenamiento` sí lo permitía desde el principio. Es una asimetría que faltaba.

**La suite de pruebas volvió a ejecutarse entera** (`tests/conftest.py`). Es el hallazgo ya
descrito en la sección 4 de `docs/propuesta_reorganizacion.md`: `conftest.py` eximía 10 de los
13 módulos que importan `geopandas`/`cdsapi`, y los 3 restantes abortaban la recolección
completa. La suite pasó de **0 pruebas recogidas a 263 pasando**. No tiene nada que ver con la
vecindad; apareció al intentar ejecutar las pruebas nuevas.
