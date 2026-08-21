# Project Plan - Predicción incendios forestales

Created: June 27, 2026 1:09 PM

## Idea general y objetivo

Desarrollar un sistema capaz de estimar el **riesgo de incendio forestal por zona geográfica** a partir de datos meteorológicos, satelitales, geoespaciales y ambientales.

El objetivo es anticipar qué zonas presentan mayor probabilidad de incendio en las próximas **24/72 horas**, generando una herramienta de alerta temprana que permita apoyar la prevención, la planificación de recursos y la toma de decisiones operativas.

## Inputs

- **Datos meteorológicos**
    - Temperatura.
    - Humedad.
    - Viento.
    - Precipitación.
    - Radiación.
    - Días sin lluvia.
    - Precipitación acumulada en ventanas de 3, 7, 14 y 30 días.
- **Datos satelitales de incendios**
    - Focos activos históricos.
    - Áreas quemadas.
    - Actividad térmica detectada por satélite.
    - Coordenadas y fecha de detección del incendio.
- **Datos geoespaciales y ambientales**
    - Tipo de vegetación.
    - Uso y cobertura del suelo.
    - Altitud.
    - Pendiente.
    - Orientación del terreno.
    - Distancia a carreteras, zonas urbanas, masas forestales o áreas protegidas.
- **Variables temporales e históricas**
    - Mes.
    - Estación.
    - Día del año.
    - Histórico de incendios en la misma zona.
    - Incendios recientes en zonas cercanas.

## fuentes de datos

**TABLA 1 — Datasets prioritarios**

1. NASA FIRMS — [https://firms.modaps.eosdis.nasa.gov/download/](https://firms.modaps.eosdis.nasa.gov/download/)
Uso: Variable objetivo (focos de incendio por fecha/localización)
Por qué: Permite construir el target de incendio en próximas 24/48/72h
2. ERA5-Land (Copernicus CDS) — [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
Uso: Variables meteorológicas (temperatura, viento, precipitación, humedad suelo)
Por qué: Meteorología homogénea en todo el territorio
3. CORINE Land Cover (Copernicus) — [https://land.copernicus.eu/en/products/corine-land-cover](https://land.copernicus.eu/en/products/corine-land-cover)
Uso: Tipo de suelo y vegetación
Por qué: Aproximación al combustible vegetal, factor clave del riesgo
4. Copernicus DEM GLO-30/GLO-90 — [https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)
Uso: Altitud, pendiente, orientación, rugosidad
Por qué: La topografía afecta propagación y comportamiento del fuego
5. CNIG/IGN límites administrativos — [https://centrodedescargas.cnig.es/CentroDescargas/index.jsp](https://centrodedescargas.cnig.es/CentroDescargas/index.jsp)
Uso: Agregación espacial, mapas por municipio/provincia/CCAA
Por qué: Necesario para presentar resultados interpretables

---

**TABLA 2 — Datasets complementarios**

EGIF-MITECO — [https://www.miteco.gob.es/es/biodiversidad/temas/incendios-forestales/estadisticas-incendios.html](https://www.miteco.gob.es/es/biodiversidad/temas/incendios-forestales/estadisticas-incendios.html)

Cuándo: Para contrastar con estadística oficial española

Recomendación: Usar como validación y contexto, no como target principal

AEMET OpenData — [https://opendata.aemet.es/centrodedescargas/inicio](https://opendata.aemet.es/centrodedescargas/inicio)

Cuándo: Para pasar de metodología histórica a sistema operativo 24-72h

Recomendación: Incluir en industrialización o como extensión, no dependencia básica

OpenStreetMap/Geofabrik — [https://download.geofabrik.de/europe/spain.html](https://download.geofabrik.de/europe/spain.html)

Cuándo: Variables de proximidad (carreteras, núcleos urbanos, infraestructuras)

Recomendación: Útil para factor humano, se puede dejar fuera del MVP

## Algoritmos y modelos que se utilizarían

- **Regresión logística** como modelo base e interpretable, para tener una primera referencia de rendimiento.
- **Random Forest** para capturar relaciones no lineales entre variables climáticas, vegetación, topografía e histórico de incendios.
- **XGBoost / LightGBM** como modelos principales, por su alto rendimiento en datos tabulares, variables heterogéneas y problemas con eventos poco frecuentes.
- **Red neuronal temporal — LSTM / GRU** para explotar la evolución histórica de las variables por zona, por ejemplo temperatura, humedad, viento o lluvia acumulada en los días previos al incendio.
- **Temporal Fusion Transformer** como modelo avanzado opcional, orientado a series temporales multivariantes, especialmente útil si se quiere predecir el riesgo combinando variables meteorológicas históricas, variables estáticas de cada zona y patrones temporales.
- **Red neuronal —** para modelar la relación entre zonas vecinas, considerando que el riesgo de incendio en una celda puede estar influido por condiciones similares o incendios recientes en áreas próximas.

## Outputs

- Probabilidad estimada de incendio por zona geográfica.
- Nivel de riesgo: bajo, medio, alto o extremo.
- Mapa interactivo de riesgo diario.
- Alertas por zona.
- Ranking de áreas más vulnerables.
- Explicación de las variables que justifican cada predicción.
- Evolución temporal del riesgo por zona.
- Dashboard final para consulta operativa.

## Conclusión ejecutiva

El resultado sería un sistema actualizable de forma **diaria**, capaz de integrar nuevos datos meteorológicos y satelitales para generar mapas de riesgo y alertas de incendio a 24/72 horas.

La aplicación práctica es clara: apoyar la prevención, priorizar recursos, anticipar zonas críticas y facilitar la toma de decisiones en gestión forestal, protección civil, administraciones públicas o entidades ambientales. Además, el enfoque es industrializable porque puede estructurarse como un pipeline recurrente de datos, modelo predictivo, explicación de resultados y dashboard operativo. El tema mantiene el enfoque original de predicción de incendios forestales con Big Data e IA, pero con una formulación más sólida y orientada a producto.

# Metodología del TFM: Predicción de Riesgo de Incendio Forestal

## La idea central

Queremos construir un sistema que diga, para cada zona de España y cada día, **qué probabilidad hay de que empiece un incendio en las próximas 24-72 horas**. No predecimos incendios que ya están ardiendo, predecimos el **inicio**, porque es lo único que tiene valor real: avisar antes de que pase.

---

## 1. ¿Cómo dividimos el mapa y el tiempo?

Dividimos España en una rejilla de **celdas de 1km x 1km**, y cada celda se observa **día a día**. Así, cada fila de nuestro dataset es: *"esta celda, este día"*.

**Por qué así:** es la forma estándar de convertir un mapa continuo en algo que un modelo pueda aprender, y nos permite luego pintar mapas de riesgo por municipio o provincia, que es justo lo que querríamos enseñar al final.

---

## 2. ¿De dónde sacamos los incendios reales (los positivos)?

Usamos un satélite de la NASA (FIRMS) que detecta focos de calor en el suelo. Cada foco trae su ubicación exacta y el día en que se detectó.

**El matiz importante:** un incendio real puede generar varios focos detectados en días distintos según se va propagando. Si los contáramos todos por separado, estaríamos contando un mismo incendio muchas veces. Por eso agrupamos los focos que están cerca en espacio (menos de 5km) y en tiempo (menos de 4 días) y los tratamos como **un único evento**. Solo nos quedamos con el primer día de ese evento, porque eso es lo que queremos predecir: el inicio.

**Por qué así:** es el criterio que usan organismos internacionales de teledetección de incendios para no inflar artificialmente el número de eventos.

---

## 3. ¿De dónde sacamos los "no incendios" (los negativos)?

Aquí está el truco del problema: el satélite solo nos dice **dónde SÍ hubo fuego**. Nunca nos dice explícitamente "aquí no hubo fuego". Eso lo tenemos que construir nosotros.

No vale coger días y zonas al azar, porque sería demasiado fácil para el modelo (un día de enero en Asturias, evidentemente, no va a arder). Necesitamos negativos "difíciles", parecidos a los positivos, para que el modelo aprenda a afinar de verdad. Usamos tres tipos:

- **Vecinos espaciales:** zonas muy cerca de un incendio real, el mismo día, que no ardieron.
- **Vecinos temporales:** la misma zona donde sí hubo incendio, pero en días cercanos donde no lo hubo.
- **Aleatorios pero representativos:** muestreamos al azar por toda España, cuidando que haya ejemplos de todas las estaciones del año y todas las regiones, no solo de verano o solo del sur.

**Cuidado importante:** si una zona va a arder dentro de pocos días, no la podemos usar como "no incendio" justo antes, porque en realidad ya estaba en proceso de incendiarse. Por eso excluimos como negativos los 5 días alrededor de cualquier incendio real.

**Por qué así:** sin negativos "parecidos" a los positivos, el modelo aprendería trivialidades (verano = riesgo) en vez de patrones reales de meteorología y terreno.

---

## 4. ¿Qué variables usamos para predecir?

Combinamos cuatro tipos de información por cada celda y cada día:

- **Meteorología** (temperatura, humedad, viento, lluvia acumulada): de Copernicus (ERA5).
- **Vegetación y tipo de suelo** (bosque, matorral, urbano...): de CORINE Land Cover.
- **Topografía** (altitud, pendiente, orientación del terreno): de un modelo digital de elevación.
- **Historial de incendios** en esa zona y en zonas cercanas.

**El matiz de las horas:** la temperatura máxima del día puede darse a las 3 de la tarde, y el viento máximo a las 6 de la mañana. Si los tratamos como dos números sueltos, perdemos si realmente coincidieron a la misma hora (que es cuando el riesgo se dispara). Por eso, en vez de mirar las 24 horas del día, calculamos los valores extremos solo dentro de la franja de mayor riesgo (12h a 18h), que es cuando suele darse la combinación peligrosa de calor + viento + sequedad.

**Otro cuidado clave (que evita "trampas"):** para predecir el riesgo de mañana, solo podemos usar datos hasta hoy. Nunca usamos información del mismo día del incendio para predecirlo, porque en la vida real ese dato no estaría disponible todavía a esa hora.

---

## 5. ¿Cómo entrenamos y evaluamos el modelo?

Probamos varios modelos, de más simple a más complejo (regresión logística, Random Forest, XGBoost, y redes neuronales). Pero el modelo no es lo más delicado, lo delicado es **cómo dividimos los datos para entrenar y validar**:

Dividimos por **años completos** (por ejemplo: entrenar con 2015-2021, validar con 2022, probar con 2023-2024), nunca mezclando días al azar.

**Por qué así:** si mezcláramos días al azar, el modelo podría "hacer trampa" viendo días muy cercanos al incendio tanto en entrenamiento como en validación, y parecería mucho mejor de lo que realmente es. Dividir por años simula exactamente lo que pasaría en la vida real: predecir el futuro con datos del pasado.

---

## 6. ¿Cómo convertimos una probabilidad en "riesgo bajo / medio / alto"?

El modelo no da un sí o un no, da una probabilidad (por ejemplo, 7%). Para convertir eso en niveles de riesgo que un gestor pueda entender, no ponemos umbrales a ojo. Los calibramos con los propios datos:

Miramos, dentro de cada rango de probabilidad que da el modelo, **cuántos incendios reales hubo de verdad**. Si vemos que a partir de un 5% de probabilidad estimada el número real de incendios se dispara, ese es nuestro corte de "riesgo alto". Así, cada nivel de riesgo (bajo, moderado, alto) está justificado con datos reales, no es una opinión.

---

## 7. ¿Cómo lo usamos de verdad para predecir el día de mañana?

Todo lo anterior sirve para entrenar y validar el modelo con datos históricos, es decir, con lo que **ya sabemos que pasó**. Pero el objetivo final es que el sistema diga, cada día, "mañana hay riesgo alto en esta zona" — y mañana, por definición, todavía no ha pasado.

**Aquí está el matiz importante:** para entrenar usamos datos meteorológicos reales y confirmados (de Copernicus). Pero para predecir el día siguiente en producción, no tenemos ese dato real todavía — tenemos la **previsión meteorológica** (por ejemplo, de AEMET), que es una estimación, no una certeza.

Esto significa que, en el día a día real:

1. Cada mañana, el sistema descarga la previsión meteorológica para mañana, pasado mañana, etc.
2. Calcula las mismas variables que usamos para entrenar (temperatura máxima, humedad mínima, viento, lluvia acumulada en la franja de 12h-18h).
3. Se las pasa al modelo ya entrenado.
4. El modelo devuelve el riesgo estimado para cada zona.

**Por qué esto es importante mencionarlo:** el modelo aprendió con datos perfectos (lo que de verdad ocurrió), pero en producción trabaja con una previsión que puede fallar. Si la previsión meteorológica se equivoca, el riesgo calculado también se verá afectado, aunque el modelo en sí sea bueno. Es una limitación real de cualquier sistema de predicción que dependa de pronósticos, y la vamos a medir: compararemos cómo de bien funciona el modelo cuando usamos datos meteorológicos reales frente a cuando usamos solo la previsión, para cuantificar cuánto se degrada el resultado.

Esta es la parte que convierte el proyecto en un sistema que de verdad podría usarse cada día, y no solo en un análisis de datos del pasado.

---

## 8. Limitaciones que reconocemos desde el principio

- El satélite puede no detectar un incendio si hay nubes ese día. No lo podemos arreglar, pero lo documentamos como una limitación conocida del estudio.
- España tiene climas muy distintos (Galicia no es lo mismo que Andalucía). Por eso incluimos la región como una variable más, y analizamos los resultados también por zona, no solo en global.
- No intentamos predecir cómo se va a propagar un incendio una vez empieza — eso sería otro proyecto entero. Nos centramos en predecir el inicio.
- La previsión meteorológica del día siguiente puede fallar, y eso afecta a la calidad de la predicción de riesgo en producción. Lo medimos comparando el rendimiento con datos reales frente a datos de previsión.

---

## Resumen en una frase

Construimos un dataset diario por zonas, con incendios reales como positivos y vecinos "casi incendio" como negativos, usando solo información disponible antes del momento de la predicción, validando con datos del futuro real (no mezclados), traducimos las probabilidades del modelo en niveles de riesgo calibrados con la propia historia de incendios, y en producción alimentamos ese modelo cada día con la previsión meteorológica para anticipar el riesgo de mañana antes de que ocurra.