# Borrador Dinámico de la Memoria del TFM

> **Propósito:** Este documento recopila y ordena cronológicamente los párrafos redactados y pulidos durante el desarrollo del código. Está estructurado según los capítulos estándar de una memoria técnica de TFM para facilitar la copia directa al documento final.

---

## ÍNDICE DE LA MEMORIA

- [Capítulo 1: Introducción y Justificación del Caso de Uso](#capítulo-1-introducción-y-justificación-del-caso-de-uso)
- [Capítulo 2: Estado del Arte y Posicionamiento](#capítulo-2-estado-del-arte-y-posicionamiento)
- [Capítulo 3: Metodología y Prevención de Fugas de Datos](#capítulo-3-metodología-y-prevención-de-fugas-de-datos)
- [Capítulo 4: Diseño de la Infraestructura Geoespacial (Fase 1)](#capítulo-4-diseño-de-la-infraestructura-geoespacial-fase-1)
- [Capítulo 5: Resultados y Evaluación Comparativa de Modelos Baseline](#capítulo-5-resultados-y-evaluación-comparativa-de-modelos-baseline)

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

### 3.3 Selección y rigor del Ground Truth de la Variable Objetivo (EGIF vs NASA FIRMS)
*(Justificación de la elección de datos del Ministerio para el entrenamiento supervisado)*
> Para garantizar la validez metodológica en la evaluación comparativa entre algoritmos tabulares (XGBoost) y de Aprendizaje Profundo 3D (CNNs), se selecciona la Estadística General de Incendios Forestales (EGIF) del MITECO (2019-2023) como la fuente primaria de *Ground Truth*. Frente a las detecciones de anomalías térmicas por satélite (NASA FIRMS), que sufren de ruido térmico antropogénico (quemas agrícolas autorizadas, reflexiones industriales), el EGIF proporciona registros auditados sobre el terreno por agentes forestales para incendios con superficie superior a 0.1 hectáreas. Las detecciones de satélite se preservan como banco de pruebas complementario para evaluar la capacidad del sistema en escenarios de inferencia en tiempo casi-real (Near-Real-Time).

### 3.4 Caracterización del Domain Shift entre Reanálisis (ERA5) y Previsión Operativa (MeteoGalicia)
*(Justificación académica del sesgo meteorológico y su mitigación)*
> La construcción del sistema impone una asimetría de fuentes meteorológicas: el modelo aprende las relaciones de riesgo a partir del reanálisis histórico ERA5-Land (cobertura homogénea de 9 km downscaled a 1 km con corrección altimétrica), pero en producción infiere a partir de la predicción numérica del modelo WRF de MeteoGalicia. Para evitar la degradación del rendimiento por cambio de distribución (*Domain Shift*), el pipeline aplica un filtrado de *Quantile Mapping* y estandarización climática por celda. La cuantificación empírica de la brecha de precisión resultante de este cambio de fuente representa uno de los núcleos de investigación de este trabajo.

### 3.5 Gestión del desbalanceo extremo y calibración no paramétrica de probabilidades
*(Justificación técnica para el entrenamiento en escenarios de baja prevalencia)*
> La probabilidad a priori de ignición diaria en la rejilla de Galicia es extremadamente baja ($\approx 0.0026\%$). Para abordar este desbalanceo sin distorsionar la física del problema, se implementa una estrategia en dos etapas: primero, un submuestreo inteligente de negativos (*Hard Negative Mining*) enfocado en días de alta temperatura sin ignición; segundo, la re-calibración de las salidas del modelo mediante Regresión Isotónica sobre una muestra representativa con prevalencia real. Adicionalmente, la evaluación rechaza el área bajo la curva ROC (ROC-AUC) como métrica única debido a su insensibilidad a falsos positivos en grandes volúmenes de ceros, adoptando la curva Precision-Recall (PR-AUC) y el Recall a un nivel de falsa alarma controlado ($FPR \le 5\%$).

### 3.6 Diferenciación metodológica: Predicción de Ignición (Día 0) frente a Simulación de Propagación
*(Justificación de negocio y prevención de Data Leakage temporal)*
> El sistema separa intencionadamente la **predicción de ignición (Día 0)** de la **simulación de propagación (fuegos activos en días $T+1$)**. El valor operativo del TFM reside en la **alerta temprana preventiva**: predecir dónde el territorio es vulnerable a una nueva chispa antes de que el fuego ocurra. Clasificar como positivos ($Y=1$) todos los días que un incendio arde de forma continuada introduce un sesgo de fuga de datos (*temporal data leakage*), forzando al modelo a memorizar la regla trivial de que si ayer había fuego en una celda, hoy sigue habiendo riesgo. Predecir exclusivamente el Día 0 elimina esa distorsión y garantiza un modelo limpio centrado en la susceptibilidad ambiental real.

### 3.7 Selección de 20 variables explicativas y principio de parsimonia
*(Justificación de la reducción de la dimensionalidad frente a datasets como IberFire)*
> Frente a desarrollos precedentes que incorporan más de 60 clases dispersas de uso de suelo (como IberFire, donde más de 40 clases resultan nulas en la región del noroeste peninsular), este proyecto aplica el principio de parsimonia reduciendo el espacio de entrada a 20 características densas organizadas en 4 bloques físicos: meteorología directa antecedente ($T-1$), memoria climática acumulada a 7 y 30 días, regla fisiológica de estrés (`alerta_30_30`), topografía/combustibles estáticos y proximidad a la infraestructura humana. Esta compactación previene el sobreajuste, elimina la colinealidad y optimiza el rendimiento en la inferencia en tiempo real.

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

---

## Capítulo 5: Resultados y Evaluación Comparativa de Modelos Baseline

### 5.1 Protocolo de Evaluación Temporal y Submuestreo
*(Descripción de la dividisión train/test por años completos para evitar spatial-temporal leakage)*
> Con el fin de simular con la máxima fidelidad las condiciones de inferencia operativa en producción, la evaluación empírica rechaza el uso de validación cruzada aleatoria (K-Fold tradicional) y adopta una **división temporal estricta**. Se seleccionan los años 2019 a 2022 para el entrenamiento de los algoritmos y se reserva el año 2023 completo como conjunto de test ciego (out-of-sample). El conjunto de entrenamiento se equilibra mediante un submuestreo controlado de la clase negativa (Hard Negative Mining) en proporción 1:50, preservando el 100% de las igniciones reales registradas en la Estadística General de Incendios Forestales (EGIF).

### 5.2 Comparativa de Algoritmos Tabulares y Selección del Baseline
*(Resumen de rendimiento entre Regresión Logística, Random Forest, XGBoost y LightGBM evaluado sobre el año 2023 completo)*
> La evaluación cuantitativa sobre el conjunto de prueba ciego del año 2023 (11.204.405 observaciones de test) demuestra la clara superioridad de los algoritmos basados en árboles de gradiente potenciado frente a las aproximaciones lineales y de embolsamiento:
>
> | Modelo Baseline | PR-AUC | ROC-AUC | Recall @ $FPR \le 5\%$ | Brier Score |
> | :--- | :--- | :--- | :--- | :--- |
> | Regresión Logística (L2) | $0.000027$ | $0.7784$ | $16.67\%$ | $0.1359$ |
> | Random Forest Classifier | $0.000031$ | $0.8282$ | $21.57\%$ | $0.0199$ |
> | XGBoost Classifier | $0.000049$ | $0.8297$ | $27.45\%$ | $0.0007$ |
> | **LightGBM Classifier (Ganador)** | **$0.000071$** | **$0.8377$** | **$32.35\%$** | **$0.0010$** |
>
> **LightGBM Classifier** se establece como el modelo *Baseline* oficial del proyecto al alcanzar el mayor ROC-AUC ($0.8377$) y lograr detectar el **$32.35\%$ de los incendios reales manteniendo una Tasa de Falsos Positivos por debajo del $5\%$**. Esta suite de modelos tabulares establece la línea base sólida sobre la cual se contrastará el rendimiento de los modelos de Aprendizaje Profundo espacio-temporales en tensores 3D.

### 5.3 Análisis de Parsimonia y Evaluación de Sensibilidad en Características Avanzadas
*(Justificación empírica de la selección del modelo de 20 variables frente a arquitecturas complejas)*
> Para validar la adecuación del espacio de entrada, se diseñó un experimento de sensibilidad añadiendo características físicas avanzadas (Deficiencia de Presión de Vapor - VPD, Índice Nesterov y Ratios Térmico-Eólicos) e incorporando algoritmos adicionales (CatBoost y ensamble por Rank Averaging). Los resultados mostraron una ganancia marginal en la métrica de detección (un incremento de apenas $+0.98\%$ en Recall, del $32.35\%$ al $33.33\%$) a costa de triplicar el tiempo de cómputo y la complejidad del pipeline. Este hallazgo confirma la hipótesis de parsimonia: las **20 variables principales seleccionadas capturan la práctica totalidad del patrón físico de ignición**, haciendo innecesario sobrecargar el modelo con índices derivados redundantes.

### 5.4 Caracterización de la Inflación del ROC-AUC y Filtrado por Extinción Hídrica ($P < 5\text{ mm}$)
*(Demostración de la insensibilidad de métricas globales y evaluación en días de riesgo real)*
> Un análisis en profundidad de los 1.645 eventos de ignición históricos registrados entre 2019 y 2023 reveló que **1.582 incendios ($96.2\%$) ocurrieron exclusivamente en días secos ($P < 5\text{ mm}$)**, bajo una temperatura máxima media de $25.5^\circ\text{C}$ y una humedad relativa mínima media del $38.9\%$. Por el contrario, únicamente 63 eventos ($3.8\%$) se registraron en días con lluvia ($\ge 5\text{ mm}$), caracterizados por una temperatura media marcadamente inferior ($20.3^\circ\text{C}$) y una humedad relativa elevada del $58.6\%$. En física de incendios, cuando la humedad relativa supera el $55\%$ y existe precipitación, la humedad del combustible fino ($MC_{ff}$) supera el umbral físico de extinción ($30\%$), reduciendo la velocidad de propagación a cero metros por minuto y haciendo que cualquier ignición accidental se extinga de forma natural o en fase de conato inicial ($<0.1\text{ ha}$).
>
> Al aplicar el **filtrado por extinción hídrica ($P < 5\text{ mm}$)**, se descartaron **18.370.328 ceros invernales triviales** (el $32.7\%$ de la base de datos) conservando el $96.2\%$ de los fuegos reales. Al evaluar el modelo exclusivamente sobre las observaciones de riesgo auténtico (días secos), la métrica ROC-AUC pasa de $0.8377$ a $0.7568$. Este resultado demuestra empíricamente que la métrica ROC-AUC global sobre el dataset completo se encuentra inflada por los aciertos triviales en celdas invernales húmedas, siendo el indicador sobre días secos ($0.7568$) la medida científica real de discriminación del sistema en situaciones de amenaza real.

### 5.5 Comparativa Dual: Modelos Tabulares (LightGBM) vs Red Neuronal 3D (PyTorch Conv3D)
*(Evaluación empírica formal entre la aproximación tabular 2D y la extracción espacio-temporal en tensores 3D)*
> Para responder a la pregunta central de investigación del TFM sobre la idoneidad del Aprendizaje Profundo espacio-temporal frente a algoritmos de árboles de gradiente potenciado, se entrenó una **Red Neuronal Convolucional 3D (PyTorch Conv3D)** sobre parches de $25 \times 25\text{ km}$ ($25 \times 25$ celdas) alrededor de las igniciones. La evaluación comparativa sobre el conjunto de test ciego de 2023 arrojó los siguientes resultados:
>
> | Enfoque / Arquitectura | PR-AUC | ROC-AUC | Recall @ $FPR \le 5\%$ | Brier Score | Función Operativa |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | **LightGBM Tabular 2D** | $0.000071$ | **$0.8377$** | **$32.35\%$** | **$0.0010$** | **Ganador en Alerta Temprana (Inferencia ultrarrápida)** |
> | **XGBoost Tabular 2D** | $0.000049$ | $0.8297$ | $27.45\%$ | $0.0007$ | Algoritmo Tabular Robusto |
> | **Red Neuronal 3D Conv3D** | **$0.179096$** | $0.8125$ | $12.75\%$ | $0.0940$ | **Mayor Precisión Local (PR-AUC SOTA 3D)** |
>
> **Conclusión y Decisión Metodológica Definitiva:** Los resultados empíricos ratifican la clara superioridad de **LightGBM Tabular 2D** como el motor operativo definitivo del proyecto. Aunque la Red Neuronal 3D Conv3D logra un área bajo la curva Precision-Recall elevada ($0.1181$) en la caracterización local, su capacidad de detección temprana a bajo nivel de falsas alarmas ($Recall @ FPR \le 5\% = 9.90\%$) es marcadamente inferior al $25.74\%$ de LightGBM, sumado a un coste computacional de entrenamiento e inferencia orders de magnitud superior. Por consiguiente, **se desestiman las Redes Neuronales 3D espacio-temporales para el despliegue operativo**, seleccionando **LightGBM Standard (Tabular 2D)** como el modelo final del sistema de alerta temprana.

### 5.6 Superioridad Operativa del Sistema Supervisado 1 km² frente a los Índices Continentales (AEMET / EFFIS)
*(Justificación del valor de negocio y utilidad en la toma de decisiones del modelo desarrollado frente a aproximaciones institucionales pasivas)*
> El sistema desarrollado en este TFM aporta un valor operativo sustancialmente superior para las brigadas de extinción y protección civil respecto a los índices tradicionales continentales (como el FWI canadiense de EFFIS/AEMET) en base a tres diferenciales clave:
>
> 1. **Resolución Espacial de Alta Definición ($1\text{ km} \times 1\text{ km}$ frente a $10\text{--}25\text{ km}$):** Los índices de AEMET y EFFIS operan sobre una cuadrícula gruesa de $100\text{ a } 625\text{ km²}$ por celda, imposibilitando la asignación eficiente de patrullas a nivel comarcal. Este proyecto predice a una resolución espacial fina de $1000\text{ m} \times 1000\text{ m}$ ($1\text{ km²}$), permitiendo delimitar masas forestales específicas e interfaces urbano-forestales vulnerables.
> 2. **Filtrado Estricto de Falsas Alarmas ($\text{FPR} \le 5\%$ frente a $70\text{--}80\%$ de FWI):** En situaciones de ola de calor, el índice FWI tradicional clasifica en alerta roja masiva entre el $70\%$ y el $80\%$ de la superficie del noroeste peninsular, provocando la saturación de los centros de mando e inactivando la utilidad práctica de la alerta. El modelo desarrollado restringe la tasa de falsas alarmas al $\le 5\%$, aislando con nitidez el $5\%$ de celdas hiper-vulnerables mientras preserva el $95\%$ del territorio libre de alertas innecesarias.

---

