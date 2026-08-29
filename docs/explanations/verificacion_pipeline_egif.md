# Verificación del pipeline EGIF

**Enrique Bravo** · Agosto de 2026

> Verificación independiente del pipeline EGIF antes de adoptarlo como base del proyecto. Cubre tres planos: la integridad de los datos publicados, el rendimiento alcanzable con los modelos habituales, y dos supuestos que condicionan cualquier cifra que se publique.
>
> Cuaderno reproducible: `notebooks/16_verificacion_egif_enrique.ipynb`

**Protocolo común a todos los resultados.** Entrenamiento 2019-2021 · validación 2022 completo (10.804.365 filas, 1.659 igniciones) · **2023 permanece cerrado**. El entrenamiento conserva el 100 % de las igniciones y submuestrea negativos 1:50; la evaluación se hace sobre el año completo con su prevalencia real intacta, porque submuestrear la validación la multiplicaría artificialmente y el PR-AUC depende directamente de ella. Los intervalos de confianza son al 90 % por bootstrap sobre los positivos.

---

## 1. Integridad del dataset

### 1.1 Coherencia entre documentación y datos

Los 47 predictores declarados en `metadata.json` están todos presentes en los Parquet, y las cuatro columnas de resultado —`target_ignicion`, `burned_area_ha`, `large_fire_500ha` y `is_near_ignition_25x25_10d`— quedan correctamente excluidas de la matriz de predictores.

### 1.2 Coherencia interna

Reejecutar el pipeline completo no es viable: requiere el cubo NetCDF, los rásters de CORINE, el modelo de elevaciones y varias horas de proceso. Se obtiene una garantía equivalente comprobando que los valores publicados son coherentes con las fórmulas que declara la documentación. Si una variable derivada no cuadra con aquellas de las que dice derivarse, hay un fallo en el pipeline aunque no lo ejecutemos.

| Comprobación | Resultado |
|---|---|
| Las nueve fracciones CORINE suman 1 | Desvío máximo 0,0000 |
| VPD reconstruible desde temperatura y humedad (Magnus) | Correlación 0,9956 |
| Viario desglosado suma el total | Error máximo 0,00000 km |
| Temperatura mín ≤ media ≤ máx | 0 filas incoherentes de 400.000 |
| Ventana 12-18 h dentro del rango del día | 0 filas incoherentes |
| Variables estáticas constantes por celda | Un único valor por celda |
| Unidades del viento | Máximo 46,7 → **km/h**, no m/s |

**7 de 7 superadas.** El pipeline está bien construido. La última comprobación resuelve además una ambigüedad: la documentación no declaraba las unidades del viento y un máximo de 46,7 admitía las dos lecturas; en m/s implicaría rachas de 168 km/h, incompatibles con el registro climático gallego.

### 1.3 Cobertura y target

| Año | Filas | Celdas | Días | Igniciones EGIF | Igniciones FIRMS | Multiplicador |
|---|---|---|---|---|---|---|
| 2019 | 10.804.365 | 29.601 | 365 | 1.595 | 296 | ×5,4 |
| 2020 | 10.833.966 | 29.601 | 366 | 1.479 | 372 | ×4,0 |
| 2021 | 10.804.365 | 29.601 | 365 | 926 | 172 | ×5,4 |
| 2022 | 10.804.365 | 29.601 | 365 | 1.659 | 703 | ×2,4 |
| 2023 (hasta 26-nov) | 9.768.330 | 29.601 | 330 | 530 | 102 | ×5,2 |
| **Total** | **53.015.391** | | | **6.189** | **1.645** | **×3,8** |

El registro oficial aporta casi cuatro veces más igniciones que la fuente satelital anterior, y la prevalencia pasa de 1 por cada 75.000 filas a **1 por cada 8.566**. Es la consecuencia más relevante del cambio de fuente: con esta densidad de positivos los intervalos de confianza se estrechan lo suficiente para que las comparaciones entre modelos dejen de caer dentro del ruido.

**Dos cuestiones a documentar:** el criterio de la máscara que deja 29.601 celdas activas frente a las 30.697 de la rejilla original, y el hecho de que la cobertura del EGIF termine el 26 de noviembre de 2023, lo que deja 2023 incompleto y sin 2024 disponible.

---

## 2. Rendimiento de referencia

Los cuatro modelos sobre las 47 variables, validados en 2022 completo:

| Modelo | ROC-AUC | ROC dentro del día | Recall@FPR5 % | IC 90 % | Igniciones detectadas | Top 1 %/día | Hueco train-val |
|---|---|---|---|---|---|---|---|
| **LightGBM** | 0,8674 | 0,7394 | **40,02 %** | 37,85 – 42,14 | **664** de 1.659 | 8,86 % | +0,132 |
| XGBoost | 0,8682 | 0,7467 | 39,06 % | 36,95 – 41,23 | 648 | 8,38 % | +0,122 |
| Random Forest | 0,8551 | 0,7215 | 36,05 % | 34,24 – 38,03 | 598 | 6,27 % | +0,145 |
| Regresión logística | 0,8409 | 0,6983 | 29,72 % | 27,91 – 31,53 | 493 | 4,22 % | +0,011 |

**Vigilando el 5 % del territorio gallego se detectan 664 de las 1.659 igniciones de 2022.** Con una regresión logística sobre las mismas variables serían 493.

Tres lecturas:

- **El salto a los modelos de gradiente potenciado está justificado sin discusión**: 171 igniciones más que el modelo lineal, con intervalos disjuntos.
- **LightGBM y XGBoost son indistinguibles** (40,02 % frente a 39,06 %, intervalos ampliamente solapados). Elegir uno u otro es una decisión de conveniencia y así conviene presentarlo, no como una victoria.
- **La brecha entre el ROC global (≈0,87) y el ROC dentro del día (≈0,74) es de 13 puntos.** El ROC global mezcla todos los días del año y premia sobre todo distinguir agosto de enero, que es lo que ya dice el calendario. El ROC calculado día a día aísla la señal espacial, que es la que responde a la pregunta operativa: a qué monte ir esta mañana. Es la cifra honesta para la memoria.

Los modelos de árboles sobreajustan de forma apreciable —hueco entre entrenamiento y validación de +0,12 a +0,15, frente a +0,011 del lineal—, lo que deja margen de mejora por regularización.

### 2.1 Qué variables aportan de verdad

La importancia nativa de los árboles cuenta cuántas veces se usa cada variable para partir, y favorece a las continuas con muchos valores distintos. La importancia por permutación mide lo relevante: cuánto empeora el modelo al destruir la información de esa variable barajándola.

| Variable | Caída de recall al permutarla |
|---|---|
| `relative_humidity_mean_7d` | +5,73 % |
| `elevation_mean` | +4,46 % |
| **`road_length_local_km`** | **+4,22 %** |
| `precipitation_sum_3d` | +3,13 % |
| `temperature_mean_7d` | +3,07 % |
| `precipitation_sum` | +2,59 % |
| `wind_speed_mean` | +2,53 % |
| `agriculture` | +2,35 % |

La memoria de humedad a siete días encabeza el ranking, por delante de cualquier variable del día en curso. Y destaca `road_length_local_km` en tercera posición, por delante de casi toda la meteorología: las carreteras locales son las pistas forestales y los caminos de acceso al monte, por donde entra la actividad humana que origina la mayoría de las igniciones. El desglose del viario por tipo estaba justificado.

### 2.2 Cuánta diferencia entre modelos es real

Cualquier cifra de este estudio depende de decisiones aleatorias: qué negativos entran en el submuestreo y cómo se inicializan los árboles. Repitiendo el mismo modelo con cinco semillas distintas:

| Semilla | Recall@FPR5 % | Top 1 %/día | ROC día | Igniciones detectadas |
|---|---|---|---|---|
| 7 | 40,57 % | 7,66 % | 0,7374 | 673 |
| 21 | 40,27 % | 7,78 % | 0,7368 | 668 |
| 42 | 39,18 % | 7,66 % | 0,7408 | 650 |
| 101 | 39,48 % | 8,20 % | 0,7350 | 655 |
| 2024 | 39,18 % | 7,59 % | 0,7401 | 650 |

**La dispersión es de 1,39 puntos de recall, equivalente a 23 igniciones.** Este número acota el ruido de implementación, que es distinto del intervalo de confianza: el intervalo mide la incertidumbre por tener pocos incendios, esta dispersión la que introduce el propio procedimiento.

De aquí sale un criterio práctico: **cualquier diferencia entre configuraciones menor que 1,39 puntos no debe interpretarse como una mejora**. Aplicado a la tabla anterior, la distancia entre LightGBM y XGBoost (0,96 puntos) queda por debajo de ese umbral.

### 2.3 Qué aporta cada bloque de variables

La importancia por permutación mide variables sueltas, y con predictores correlacionados eso engaña: si dos variables dicen lo mismo, barajar una no empeora el modelo porque la otra la sustituye. La ablación por grupos evita ese problema retirando bloques temáticos completos.

Se entrena el mismo modelo quitando un grupo cada vez, con todo lo demás idéntico:

| Configuración | Variables | Recall@FPR5 % | Diferencia | IC 90 % |
|---|---|---|---|---|
| **Completo** | 47 | **35,73 %** | — | 33,85 – 37,55 |
| Sin meteorología del día | 36 | 36,03 % | **+0,30** | 34,22 – 38,27 |
| Sin VPD | 45 | 35,61 % | −0,12 | 33,62 – 37,37 |
| Sin cobertura del suelo | 38 | 34,82 % | −0,91 | 33,07 – 36,94 |
| Sin memoria climática | 41 | 34,52 % | −1,21 | 32,89 – 36,46 |
| Sin actividad humana | 40 | 33,19 % | −2,54 | 31,62 – 35,01 |
| Sin topografía | 35 | 33,07 % | −2,66 | 31,32 – 34,70 |

*(Ejecutado con `ablacion_y_espacial_egif.py`, LightGBM, mismo protocolo.)*

**Ningún grupo resulta imprescindible por sí solo.** Todos los intervalos se solapan con el del conjunto completo, lo que indica redundancia: cuando falta un bloque, los demás compensan parte de su información. Los que más se acercan a ser determinantes son **topografía** (−2,66) y **actividad humana** (−2,54), ambos al límite de la significación y coherentes con lo que ya señalaba la importancia por permutación.

El resultado más llamativo es que **retirar las once variables de meteorología del día no empeora el modelo** —de hecho lo mejora ligeramente—, mientras que quitar la memoria climática sí cuesta 1,21 puntos. Leído junto a la sección 2.1, donde la humedad media de siete días encabeza el ranking, apunta en una dirección clara: **lo que informa del riesgo no es tanto el tiempo que hace hoy como el que ha hecho durante la semana anterior**. Tiene sentido físico: el combustible fino tarda días en secarse, y ese estado acumulado es el que determina si una ignición prospera.

Conviene ser prudente con la lectura: al no haber diferencias concluyentes, estos números orientan sobre dónde buscar simplificaciones, pero no autorizan a retirar bloques sin más comprobación.

---

## 3. Curva de presupuesto operativo

El 5 % de falsas alarmas es una convención estadística, no una restricción real. Quien dirige un operativo no razona en tasas sino en medios disponibles: cuántas brigadas hay, cuántas zonas se pueden cubrir esta mañana. Esta sección traduce el modelo a esa decisión.

| Territorio vigilado | Celdas al día | Igniciones detectadas | Recall diario | Eficiencia (igniciones por 100 celdas) |
|---|---|---|---|---|
| 0,5 % | 148 | 95 de 1.659 | 5,73 % | **64,2** |
| 1 % | 296 | 147 | 8,86 % | 49,7 |
| 2 % | 592 | 214 | 12,90 % | 36,2 |
| 5 % | 1.480 | 378 | 22,78 % | 25,5 |
| 10 % | 2.960 | 569 | 34,30 % | 19,2 |
| 20 % | 5.920 | 838 | 50,51 % | 14,2 |

Con 148 celdas al día —una vigilancia mínima, del orden de una zona por comarca— se detectan 95 igniciones al año. Multiplicar por cuarenta ese despliegue, hasta 5.920 celdas diarias, solo multiplica por nueve la detección.

**La eficiencia cae de 64 a 14 igniciones por cada 100 celdas vigiladas**, y la curva se aplana de forma visible a partir del 5 % del territorio. Ese es el punto donde ampliar la vigilancia deja de compensar, y es la respuesta que necesita quien tiene que justificar un presupuesto.

---

## 4. El contrato temporal

El metadato del dataset declara explícitamente:

> `time_contract: "No temporal shift; meteorological accumulations include date T"`

Para predecir la ignición del día T, el modelo utiliza la meteorología **del propio día T**, incluida la temperatura máxima de la tarde. Verificado numéricamente sobre una celda con ignición el 2 de julio de 2022: `precipitation_sum_7d` de ese día equivale exactamente a la suma de T-6 hasta T incluido (2,0531 mm), y no a la de T-7 hasta T-1 (4,2090 mm).

Se midió la alternativa —desplazar todos los predictores un día, de modo que la fila del día T lleve lo observado en la víspera— entrenando el mismo modelo con ambas configuraciones:

| Variante | Igniciones | ROC-AUC | ROC día | Recall@FPR5 % | IC 90 % | Top 1 %/día |
|---|---|---|---|---|---|---|
| **Día T** | 1.659 | 0,8680 | 0,7408 | **39,18 %** | 37,07 – 41,41 | **7,66 %** |
| Víspera (T-1) | 1.654 | 0,8555 | 0,7356 | 36,88 % | 35,13 – 38,94 | 7,38 % |

**Se mantiene el contrato del día T.** Rinde 2,3 puntos por encima en la métrica operativa; la diferencia no llega a ser concluyente porque los intervalos se solapan, pero su dirección es consistente en las cuatro métricas.

**La consecuencia debe quedar escrita en la memoria.** Las variables del día T no están disponibles en el momento en que se toma la decisión: a las ocho de la mañana, la temperatura máxima de la tarde todavía no ha ocurrido. Por tanto, en producción esas variables tendrán que alimentarse con la **previsión meteorológica de MeteoGalicia**, no con observaciones.

Es el mismo diseño de los sistemas operativos de referencia —el FWI canadiense y el EFFIS europeo predicen el día T con la previsión del día T— y es perfectamente defendible, pero obliga a dos cosas:

1. **Declararlo con claridad**: el rendimiento aquí medido, obtenido con observaciones, es el techo teórico del sistema, no su rendimiento en operación.
2. **Cuantificar la degradación** al sustituir observaciones por pronósticos, desglosada por horizonte (24, 48 y 72 horas). Sin esa medición, las cifras publicadas describen un escenario que no se da en producción.

El segundo punto convierte el análisis de degradación en un requisito del proyecto, no en una mejora opcional.

---

## 5. Generalización espacial

La validación temporal responde a «¿funcionará el año que viene?». No responde a «¿funcionaría en otra zona?». Si el modelo estuviera memorizando qué celdas concretas arden, en lugar de aprender qué condiciones provocan fuego, acertaría en el futuro de las mismas comarcas y fallaría al cambiar de territorio.

Se divide Galicia en cinco bloques geográficos por bandas diagonales, se entrena con cuatro y se evalúa exclusivamente en el quinto, que el modelo no ha visto nunca:

| Partición | Celdas | Igniciones | Recall@FPR5 % | IC 90 % | ROC dentro del día |
|---|---|---|---|---|---|
| **Referencia temporal (toda Galicia)** | 29.601 | 1.659 | **40,02 %** | 37,85 – 42,14 | **0,7394** |
| Bloque 0 | 5.961 | 646 | 23,99 % | 21,21 – 26,78 | 0,5921 |
| Bloque 1 | 5.943 | 442 | 26,92 % | 23,30 – 30,32 | 0,6724 |
| Bloque 2 | 5.877 | 253 | 28,85 % | 24,11 – 33,62 | 0,6714 |
| Bloque 3 | 5.901 | 166 | 31,33 % | 25,90 – 37,38 | 0,6229 |
| Bloque 4 | 5.919 | 152 | 24,34 % | 19,08 – 28,98 | 0,5459 |
| **Media espacial** | | | **27,09 %** | | **0,6209** |

**En territorio desconocido la discriminación espacial cae de 0,7394 a 0,6209 —casi 12 puntos— y el recall del 40,0 % al 27,1 %.** En el peor bloque el ROC dentro del día queda en 0,5459, apenas por encima del azar.

Parte de la caída es esperable: cada bloque entrena con un 20 % menos de datos y los regímenes de incendio difieren entre comarcas. Pero la magnitud indica que una porción sustancial del rendimiento proviene de conocer qué celdas concretas han ardido históricamente, y no solo de las condiciones ambientales.

**Consecuencia para la memoria:** la afirmación de que el sistema escala a nivel nacional «simplemente cambiando el archivo de frontera geométrica» no está respaldada por la evidencia. Trasladarlo a otra comunidad exigiría reentrenar con su propio histórico. Conviene reformular ese pasaje como hipótesis pendiente de validación y presentar esta medición como limitación cuantificada, que es una posición más sólida ante un tribunal que una afirmación optimista sin respaldo.

