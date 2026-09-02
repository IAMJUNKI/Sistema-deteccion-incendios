# Propuesta para la memoria: construcción del datacubo

## Construcción del datacubo espacio-temporal

La unidad de análisis del trabajo es una celda de 1 km × 1 km y un día. Esta
elección permite representar conjuntamente la heterogeneidad territorial de
Galicia y la variación diaria de las condiciones que favorecen una ignición. El
producto resultante es un **datacubo**: una estructura multidimensional en la
que todas las fuentes comparten la misma rejilla espacial y, cuando procede, el
mismo eje temporal. Esta organización sigue la idea metodológica de IberFire
(Erzibengoa Calvo *et al.*, 2025), adaptada aquí a una escala regional, al
registro oficial EGIF y a un contrato de variables más compacto y explicable.

### Rejilla, máscara y periodo

La rejilla se genera en ETRS89-LAEA (EPSG:3035), una proyección de igual área
adecuada para Europa. Se construye primero un rectángulo regular de celdas de
1.000 m y se añade después la máscara `is_galicia`, que identifica las celdas
cuyo centro pertenece al límite administrativo de Galicia. El rectángulo se
conserva completo en el NetCDF porque permite almacenar capas rasterizadas con
las mismas coordenadas `x` e `y`; para el modelado tabular solo se exportan las
29.601 celdas marcadas como Galicia.

El cubo final comprende 1.791 fechas, desde el 1 de enero de 2019 hasta el 26
de noviembre de 2023, última fecha cubierta por el XML de EGIF empleado. Se
recupera meteorología desde el 1 de diciembre de 2018 para poder calcular
ventanas de precipitación de hasta 30 días desde el primer día modelable.

### Integración de las fuentes

Las variables estáticas se calculan una sola vez por celda. La topografía se
obtiene del Copernicus DEM GLO-30 y se resume mediante elevación media y
desviación típica, pendiente media y desviación típica y ocho fracciones de
orientación. La cobertura del suelo procede de CORINE Land Cover 2018 y se
agrega en nueve fracciones interpretables: superficies artificiales,
agricultura, bosque de frondosas, bosque de coníferas, bosque mixto, matorral,
espacios abiertos, humedales y agua. Finalmente, una instantánea versionada de
OpenStreetMap de 2022 aporta la longitud total de vías, sus longitudes por tipo
(principal, local, pista y otras) y las fracciones de área residencial y de
edificios.

Las variables dinámicas proceden del reanálisis ERA5-Land horario. Se agregan a
escala diaria y se interpolan linealmente desde su rejilla original a los
centroides de las celdas de Galicia. En las celdas costeras donde dicha
interpolación no es válida se utiliza el píxel terrestre ERA5-Land válido más
cercano, evitando que queden filas incompletas. El bloque meteorológico incluye
temperatura, humedad relativa, velocidad del viento, precipitación, déficit de
presión de vapor y sus resúmenes de la franja crítica de 12:00 a 18:00
Europe/Madrid, además de acumulados de precipitación a 3, 7, 14 y 30 días,
medias móviles y días secos consecutivos.

Como referencia física independiente se añade el Fire Weather Index (FWI)
histórico de CEMS/EFFIS. Es un índice diario calculado por el servicio a partir
de forzamiento ERA5, con resolución original de 0,25°. Para mantener una misma
rejilla, se asigna a cada celda de 1 km mediante vecino más cercano, siguiendo
la incorporación descrita en IberFire. FWI no expresa igniciones observadas:
se conserva como baseline comparable; nunca se entrega como predictor a los
modelos de ML.

La variable objetivo se construye a partir de la Estadística General de
Incendios Forestales (EGIF-MITECO). Cada evento se asigna a una pareja
celda-fecha; `target_ignicion` vale 1 cuando existe al menos una ignición EGIF
en dicha celda y día, y 0 en caso contrario. Se mantienen también la superficie
quemada y el indicador de gran incendio como resultados auxiliares, nunca como
entradas del modelo. La máscara `is_near_ignition_25x25_10d` marca las celdas
incluidas en una ventana espacial de 25 × 25 celdas alrededor de una ignición
en el día del evento y en los diez días anteriores. Su único uso admisible es
facilitar experimentos de muestreo de negativos; no es una variable predictora
ni estaría disponible en producción.

### Contrato temporal y producto final

El producto histórico actual es un análisis diario: las variables
meteorológicas y sus acumulados describen el propio día `T`, por lo que los
acumulados incluyen `T`. Esta decisión permite comparar condiciones diarias e
igniciones observadas, pero debe distinguirse de una predicción operativa. En
un despliegue futuro, ERA5-Land tendría que sustituirse por datos observados y
previsiones de MeteoGalicia, definiendo explícitamente qué información está
disponible antes de empezar el día `T`.

El datacubo NetCDF contiene 56 variables de datos: 50 candidatas a predictor,
un baseline físico (`fire_weather_index`) y cinco variables de resultado o
control (`is_galicia`, `target_ignicion`, `burned_area_ha`,
`large_fire_500ha` e `is_near_ignition_25x25_10d`). Las 50 variables
predictoras se distribuyen como sigue:

| Bloque | Variables | Número |
|---|---|---:|
| Topografía | elevación, pendiente y orientación | 12 |
| Cobertura del suelo | fracciones CORINE | 9 |
| Actividad humana | red viaria y ocupación residencial | 7 |
| Meteorología | condiciones diarias, ventana crítica y memoria meteorológica | 22 |
| **Total predictores** |  | **50** |

`fire_weather_index` se conserva adicionalmente en el Parquet como baseline,
fuera de la matriz de predictores, para compararlo con los modelos entrenados.

Para los modelos tabulares, el cubo se transforma en cinco ficheros Parquet
anuales. Cada fila representa una celda activa y una fecha, e incorpora
`fecha`, `x`, `y`, `cell_id` y `year` para trazabilidad y partición temporal.
Estos identificadores no se incluyen como predictores. La exportación final
contiene 53.015.391 filas completas y no descarta ninguna por valores faltantes
en las variables predictoras.

### Reproducibilidad y control de calidad

La construcción se centraliza en un único workflow reproducible. Este valida
las capas estáticas locales, determina la última fecha disponible de EGIF,
incorpora las capas dinámicas, ensambla el NetCDF y exporta los Parquet. El
perfil canónico de variables evita incluir capas redundantes o poco
interpretables: se excluyen las coordenadas y el calendario como señal del
modelo, la rugosidad por su fuerte redundancia con la pendiente, la fracción
forestal total por ser suma exacta de las tres fracciones de bosque, las dos
distancias OSM por su saturación en cero a 1 km y las variables técnicas o
duplicadas. La validación del producto comprueba las dimensiones, periodo,
variables esperadas, cobertura de EGIF, filas Parquet, ausencia de nulos y
separación estricta entre identificadores, resultados y predictores.
