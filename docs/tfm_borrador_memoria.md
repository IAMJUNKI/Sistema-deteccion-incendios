# Borrador Dinámico de la Memoria del TFM

> **Propósito:** Este documento recopila y ordena cronológicamente los párrafos redactados y pulidos durante el desarrollo del código. Está estructurado según los capítulos estándar de una memoria técnica de TFM para facilitar la copia directa al documento final.

---

## ÍNDICE DE LA MEMORIA

- [Capítulo 1: Introducción y Justificación del Caso de Uso](#capítulo-1-introducción-y-justificación-del-caso-de-uso)
- [Capítulo 2: Estado del Arte y Posicionamiento](#capítulo-2-estado-del-arte-y-posicionamiento)
- [Capítulo 3: Metodología y Prevención de Fugas de Datos](#capítulo-3-metodología-y-prevención-de-fugas-de-datos)
- [Capítulo 4: Diseño de la Infraestructura Geoespacial (Fase 1)](#capítulo-4-diseño-de-la-infraestructura-geoespacial-fase-1)

---

## Capítulo 1: Introducción y Justificación del Caso de Uso

### 1.1 Justificación del MVP Regional en Galicia
*(Párrafo para introducir por qué se acota el proyecto a Galicia antes de escalar a nivel nacional)*
> La escala de procesamiento espaciotemporal a nivel nacional para España (con una resolución de 1 km² diaria y un histórico de 5 años) arroja un volumen bruto superior a los 900 millones de registros. Para garantizar la viabilidad computacional en el desarrollo del TFM, la agilidad en la ingeniería de variables y un control de calidad geoespacial exhaustivo, se ha seleccionado la Comunidad Autónoma de Galicia como región piloto para el Producto Mínimo Viable (MVP). Galicia representa un caso de estudio idóneo debido a la extrema densidad e histórico de incendios forestales, el reto ecológico del minifundio y el acceso a una de las redes meteorológicas terrestres más densas de la península (MeteoGalicia).

---

## Capítulo 2: Estado del Arte y Posicionamiento

### 2.1 Posicionamiento respecto a soluciones y estudios previos
*(Párrafo de posicionamiento para la sección de antecedentes del TFM)*
> A diferencia de los trabajos precedentes que o bien construyen datasets de alta resolución (como el dataset IberFire, Ercibengoa et al., 2025) o bien operan sistemas de índices meteorológicos a escala continental (como EFFIS/GEFF de Copernicus), este TFM presenta un pipeline completo de extremo a extremo: desde la ingesta de datos históricos y la construcción de negativos difíciles hasta la inferencia diaria con previsiones meteorológicas regionales de MeteoGalicia, la calibración estadística de niveles de riesgo y el despliegue de un dashboard operativo. La cuantificación de la brecha de rendimiento entre el modo histórico (ERA5) y el modo predictivo operativo (MeteoGalicia) constituye la contribución científica central de este trabajo.

---

## Capítulo 3: Metodología y Prevención de Fugas de Datos

### 3.1 La paradoja de la causa humana en la predicción del riesgo
*(Redacción para justificar por qué predecimos incendios mediante variables ambientales cuando el 96% los causan los humanos)*
> El 96% de los incendios forestales en España tienen causa humana, ya sea de origen accidental o intencionado. Lejos de invalidar el enfoque predictivo, este hecho subraya su valor: el sistema no predice la intención humana —impredecible por definición— sino la vulnerabilidad ambiental del territorio en un momento dado. Los datos históricos de ignición muestran que los incendios, independientemente de su causa, se concentran sistemáticamente en condiciones de temperatura extrema, baja humedad, viento intenso y sequedad acumulada, y en zonas con alta accesibilidad humana y determinados tipos de combustible. El modelo aprende estas condiciones de riesgo, que son las mismas que transforman cualquier ignición —accidental o deliberada— en un incendio de propagación incontrolada. La variable objetivo no es "¿quemará alguien hoy?" sino "¿si hoy se produce una ignición, las condiciones del territorio favorecerán su propagación catastrófica?"

### 3.2 Asunción semi-estática de la capa de combustibles y trabajo futuro
*(Redacción para justificar el uso de CORINE Land Cover estático y proponer extensiones con satélite)*
> Asumimos la capa de combustibles de CORINE Land Cover 2018 como semi-estática para el periodo 2019-2024. Aunque la vegetación sufre cambios lentos debido a la actividad forestal o incendios previos, CLC 2018 representa la línea de base oficial más robusta disponible. Como trabajo futuro, se propone integrar índices dinámicos de vegetación por satélite (como el NDVI o NDWI de MODIS/Sentinel) calculados mensualmente. Esto permitiría actualizar en tiempo real el estado de sequedad y la pérdida de biomasa viva tras un incendio previo, corrigiendo los desfases temporales de la cartografía estática.

---

## Capítulo 4: Diseño de la Infraestructura Geoespacial (Fase 1)

### 4.1 Alineación temporal de la cobertura del suelo y prevención de Data Leakage
*(Redacción que justifica la implementación de la rejilla multi-temporal para evitar data leakage)*
> Para mitigar la fuga de datos temporales (temporal data leakage) derivada de los cambios en el uso del suelo, este proyecto rechaza el uso de una rejilla estática única. En su lugar, el pipeline geoespacial implementa una **estrategia de alineación temporal adaptativa**. Para ello, se generan tres versiones de la rejilla base indexadas por año:
> 1. **Grid Histórico 2012 (basado en CLC 2012):** Utilizado para cruzar los eventos de ignición e incendios históricos ocurridos entre los años 2013 y 2018.
> 2. **Grid Histórico 2018 (basado en CLC 2018):** Utilizado para cruzar el histórico de entrenamiento de los años 2019 a 2024.
> 3. **Grid de Producción 2024 (basado en CLC 2024):** Utilizado exclusivamente para el pipeline de inferencia operativa diaria a partir de 2025.
>
> Este diseño garantiza que el modelo de Machine Learning aprenda relaciones físicas correctas basándose en la foto de vegetación más cercana en el tiempo al evento de ignición. Por ejemplo, evita clasificar erróneamente un incendio de 2019 en un bosque talado posteriormente en 2022 (que en el mapa de 2024 figuraría como matorral o pastizal), preservando la coherencia causal del entrenamiento. La modularidad del pipeline de la Fase 1 permite generar cualquiera de las tres rejillas simplemente especificando el archivo de CORINE de entrada.
