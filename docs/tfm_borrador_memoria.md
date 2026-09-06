# Borrador Dinámico de la Memoria del TFM

> **Propósito:** Este documento recopila y ordena cronológicamente los párrafos redactados y pulidos durante el desarrollo del código. Está estructurado según los capítulos estándar de una memoria técnica de TFM para facilitar la copia directa al documento final.

> **Actualización metodológica — 5 de septiembre de 2026.** La implementación operativa actual
> congela el contrato `egif-2d-48-v1`: son exactamente 48 predictores, obtenidos del contrato de
> 50 variables del datacubo excluyendo `precipitation_sum` y `consecutive_dry_days`. El FWI no es
> predictor y se conserva únicamente como baseline físico. El datacubo continúa almacenando las
> 50 variables para trazabilidad y rollback, pero los artefactos de producción son
> `forecast_risk_egif_48_t1/t2/t3.joblib`. Las métricas históricas que aparecen más abajo deben
> leerse como resultados retrospectivos y no como una evaluación de un forecast operativo real.
>
> La semántica temporal de la familia 48 es: `issue_time` + meteorología del día objetivo
> procedente del forecast + memoria observada/prevista → `target_ignicion` del día objetivo. El
> benchmark histórico usa ERA5-Land perfecta y por eso se etiqueta `era5_perfect_benchmark`; no
> mide todavía el error de WRF. La partición comparable es entrenamiento 2019–2020, calibración
> 2021, validación 2022 y test ciego 2023. La ampliada usa entrenamiento 2016–2020 con el mismo
> resto de particiones. La familia 48 se calibra por horizonte con corrección de prior seguida de
> Platt; la familia histórica de 50 se conserva como rollback/shadow y no se mezclan ambas.

---

## ÍNDICE DE LA MEMORIA

- [Capítulo 1: Introducción y Justificación del Caso de Uso](#capítulo-1-introducción-y-justificación-del-caso-de-uso)
- [Capítulo 2: Estado del Arte y Posicionamiento](#capítulo-2-estado-del-arte-y-posicionamiento)
- [Capítulo 3: Metodología y Prevención de Fugas de Datos](#capítulo-3-metodología-y-prevención-de-fugas-de-datos)
- [Capítulo 4: Diseño de la Infraestructura Geoespacial (Fase 1)](#capítulo-4-diseño-de-la-infraestructura-geoespacial-fase-1)
- [Capítulo 5: Resultados y Evaluación Comparativa de Modelos Baseline](#capítulo-5-resultados-y-evaluación-comparativa-de-modelos-baseline)
- [Capítulo 6: Diseño e Implementación del Centro de Mando Táctico y Dashboard Operativo (Fase 5)](#capítulo-6-diseño-e-implementación-del-centro-de-mando-táctico-y-dashboard-operativo-fase-5)

---

## Capítulo 1: Introducción y Justificación del Caso de Uso

### 1.1 Justificación del MVP Regional en Galicia
*(Párrafo para introducir por qué se acota el proyecto a Galicia antes de escalar a nivel nacional)*
> La escala de procesamiento espaciotemporal a nivel nacional para España (con una resolución de 1 km² diaria y un histórico de 5 años) arroja un volumen bruto superior a los 900 millones de registros. Para garantizar la viabilidad computacional en el desarrollo del TFM, la agilidad en la ingeniería de variables y un control de calidad geoespacial exhaustivo, se ha seleccionado la Comunidad Autónoma de Galicia como región piloto para el Producto Mínimo Viable (MVP). Galicia representa un caso de estudio idóneo debido a la extrema densidad e histórico de incendios forestales, el reto ecológico del minifundio y el acceso a una de las redes meteorológicas terrestres más densas de la península (MeteoGalicia).

---

## Capítulo 2: Estado del Arte y Posicionamiento

### 2.1 Posicionamiento respecto a soluciones y estudios previos
*(Párrafo de posicionamiento para la sección de antecedentes del TFM)*
> A diferencia de los trabajos precedentes que se centran en construir datasets de alta resolución (como IberFire, con una resolución de 1 km × 1 día y 120 variables) o en operar índices meteorológicos a escala continental, este TFM presenta un pipeline completo de extremo a extremo: desde la ingesta histórica y la construcción de negativos difíciles hasta la inferencia diaria con previsiones regionales de MeteoGalicia, la calibración de probabilidades, la priorización preventiva de celdas y el despliegue de un dashboard operativo. IberFire se utiliza como referencia metodológica para la integración de fuentes y la prevención de fugas, pero no como una comparación directa de objetivos: su definición de incendio y su estrategia de evaluación no son idénticas a las de este sistema. La cuantificación de la brecha de rendimiento entre el benchmark histórico ERA5-Land y el modo predictivo operativo MeteoGalicia constituye una contribución científica central de este trabajo.

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

### 3.4 Caracterización del Domain Shift entre Reanálisis (ERA5-Land) y Previsión Operativa (MeteoGalicia)
*(Justificación académica del sesgo meteorológico y su mitigación)*
> La construcción del sistema impone una asimetría de fuentes meteorológicas: el modelo aprende las relaciones de riesgo a partir del reanálisis histórico ERA5-Land, mientras que en producción infiere a partir de la predicción numérica WRF de MeteoGalicia. La API MeteoSIX v5 ofrece una malla WRF de 1 km, que constituye la fuente preferida, y una malla de 4 km como fallback cuando la primera no está disponible o no supera la validación de cobertura. La rejilla de riesgo continúa siendo de 1 km, pero el manifiesto registra la malla meteorológica efectiva y marca `fresh_fallback` cuando se degrada la resolución. La primera versión archiva el forecast bruto y separa el benchmark ERA5 del rendimiento operativo; no aplica Quantile Mapping sin pares históricos forecast-observación. La cuantificación de la brecha de precisión resultante de este cambio de fuente representa uno de los núcleos de investigación de este trabajo.

> Para evitar una fuga temporal adicional, el datacubo retrospectivo y el dataset de entrenamiento operativo se mantienen como productos distintos. El primero conserva las acumulaciones inclusivas para auditoría; el segundo recalcula las memorias meteorológicas con una frontera `target_date - 1 day`, arrastrando el contexto de 30 días entre particiones anuales. La familia operativa de 48 variables y el control alineado de 50 solo se entrenan sobre esta segunda salida. Así, el benchmark `era5_perfect_benchmark` documenta la arquitectura con meteorología histórica conocida, pero no se presenta como una evaluación del error real de MeteoGalicia.

### 3.5 Gestión del desbalanceo extremo y calibración de probabilidades
*(Justificación técnica para el entrenamiento en escenarios de baja prevalencia)*
> La probabilidad a priori de ignición diaria en la rejilla de Galicia es extremadamente baja ($\approx 0.0026\%$). Para abordar este desbalanceo sin distorsionar la física del problema, se implementa una estrategia en dos etapas: primero, se conservan todas las igniciones y se submuestrean negativos de forma determinista; segundo, se corrige el prior de las probabilidades y se calibra sobre un año separado con la prevalencia real. En la familia operativa de 48 variables se utiliza Platt por horizonte, mientras que la familia histórica de 50 mantiene su calibrador isotónico para rollback. Adicionalmente, la evaluación rechaza el área bajo la curva ROC (ROC-AUC) como métrica única debido a su insensibilidad a falsos positivos en grandes volúmenes de ceros, adoptando la curva Precision-Recall (PR-AUC), Brier score y recall con presupuesto espacial.

### 3.6 Diferenciación metodológica: Predicción de Ignición (Día 0) frente a Simulación de Propagación
*(Justificación de negocio y prevención de Data Leakage temporal)*
> El sistema separa intencionadamente la **predicción de ignición (Día 0)** de la **simulación de propagación (fuegos activos en días $T+1$)**. El valor operativo del TFM reside en la **alerta temprana preventiva**: predecir dónde el territorio es vulnerable a una nueva chispa antes de que el fuego ocurra. Clasificar como positivos ($Y=1$) todos los días que un incendio arde de forma continuada introduce un sesgo de fuga de datos (*temporal data leakage*), forzando al modelo a memorizar la regla trivial de que si ayer había fuego en una celda, hoy sigue habiendo riesgo. Predecir exclusivamente el Día 0 elimina esa distorsión y garantiza un modelo limpio centrado en la susceptibilidad ambiental real.

### 3.7 Selección de 48 variables explicativas y principio de parsimonia
*(Justificación de la reducción de la dimensionalidad frente a datasets como IberFire)*
> Frente a desarrollos como IberFire, que integra más variables en varias categorías, este proyecto aplica inicialmente el principio de parsimonia con un contrato operativo de 48 características densas organizadas en bloques físicos: meteorología, memoria meteorológica, topografía/combustibles y contexto humano. Esta compactación facilita la trazabilidad y la inferencia en tiempo real. Las variables excluidas del contrato —`precipitation_sum` y `consecutive_dry_days`— permanecen en el datacubo para auditoría, pero no se entregan al modelo. El FWI se conserva como baseline y no como extensión silenciosamente incorporada.

> La comparación de 48 y 50 variables se realiza con un control de 50 alineado temporalmente. Los artefactos de 50 variables entrenados con la semántica histórica anterior se conservan únicamente para rollback/shadow, porque no constituyen una comparación experimental perfectamente pareada.

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
> Con el fin de simular con la máxima fidelidad las condiciones de inferencia operativa en producción, la evaluación empírica rechaza el uso de validación cruzada aleatoria (K-Fold tradicional) y adopta una **división temporal estricta**. Para la familia operativa de 48 variables se conservan dos experimentos: el comparable usa 2019–2020 para entrenar, 2021 para calibrar, 2022 para validar y 2023 como test ciego; el ampliado sustituye el entrenamiento por 2016–2020. El conjunto de entrenamiento se equilibra mediante submuestreo controlado de negativos, preservando el 100% de las igniciones EGIF. Las métricas se calculan sobre la población completa.

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

### 5.7 Separación entre baseline físico y modelo supervisado
*(Decisión metodológica para no confundir un índice de peligro con una probabilidad de ignición)*
> El sistema no fija determinísticamente la probabilidad del modelo mediante umbrales de lluvia,
> humedad o biomasa. El modelo LightGBM aprende la relación estadística entre las variables del
> contrato y `target_ignicion`, mientras que el Fire Weather Index (FWI) se calcula y presenta
> como baseline físico independiente. Esta separación permite comparar dos señales con objetivos
> distintos sin introducir reglas no validadas en la probabilidad calibrada ni presentar el FWI
> como predictor encubierto.
>
> El submuestreo de negativos se aplica únicamente durante el entrenamiento y se corrige en el
> calibrador. La publicación operativa conserva la población completa, la calidad del forecast y
> el presupuesto espacial de priorización. Cualquier filtro físico adicional deberá proponerse
> como una nueva versión, entrenarse y evaluarse temporalmente antes de incorporarse.

### 5.8 Objetivo preventivo y priorización de recursos públicos

> El producto no se diseña para sustituir el criterio del centro de mando ni para ordenar automáticamente el despliegue de agentes. Su función es reducir el espacio de decisión: para cada horizonte, entrega un ranking de celdas donde la combinación de meteorología, combustible, topografía e historial hace más probable una nueva ignición o un incendio detectable. Este ranking permite concentrar vigilancia, patrullas y medios de primera intervención cuando los recursos son limitados. La variable objetivo científica permanece definida a nivel de celda y día, mientras que la capa operativa transforma la probabilidad calibrada en cuatro acciones orientativas: vigilancia rutinaria, vigilancia reforzada, preposición de medios y preposición prioritaria. La decisión final debe incorporar accesibilidad, exposición, tiempos de respuesta, medios disponibles y confirmación humana.

> Esta formulación es más útil para prevención que intentar predecir directamente la causa humana. El sistema no estima si una persona decidirá provocar un incendio; estima si una ignición producida por cualquier causa encontrará condiciones ambientales favorables para consolidarse. El criterio de éxito operativo será, por tanto, cuántos eventos reales quedan cubiertos por el 1 %, 5 % y 10 % de celdas priorizadas, junto con el recall alcanzado para una tasa de falsas alarmas asumible.

### 5.9 Selección adaptativa de la previsión meteorológica

> La documentación de MeteoSIX v5 establece que la ejecución WRF de 1 km iniciada a las 00:00 UTC termina aproximadamente a las 07:30 UTC, aunque la disponibilidad real puede variar. Por ello, una ejecución fija a las 05:00 hora local no garantiza que la nueva salida de 1 km esté publicada. El pipeline implementado intenta WRF 1 km como fuente preferente y valida la cobertura completa de las horas y variables requeridas. Si esta descarga falla o está incompleta, intenta WRF 04 km. El resultado se etiqueta respectivamente como `fresh` o `fresh_fallback`; únicamente cuando ambas mallas fallan se reutiliza el último forecast archivado y se marca como `stale`. Esta degradación es visible para el usuario y nunca se sustituye silenciosamente por una fecha histórica.

> La decisión entre 1 km y 04 km no se basa solo en la resolución nominal. También se conservan la ejecución `modelRun`, el instante de descarga, la malla efectiva, la versión de la API, la distancia al punto consultado y la cobertura horaria. La consulta se realiza mediante puntos representativos agrupados en lotes de 20, manteniendo la rejilla final de riesgo de 1 km sin lanzar una petición independiente por cada celda.

### 5.10 Aprendizajes metodológicos de IberFire y límites de comparabilidad

> IberFire constituye una referencia valiosa para estructurar un cubo espacio-temporal, separar capas estáticas de capas dinámicas y armonizar fuentes con resoluciones diferentes. También muestra la utilidad de derivar estadísticas meteorológicas a partir de series horarias: humedad relativa calculada a partir de temperatura y punto de rocío, y velocidad del viento calculada desde sus componentes antes de agregarla. Estas decisiones se incorporan como criterios de calidad para el histórico ERA5-Land y para comprobar que las variables de entrenamiento y previsión tienen la misma semántica.

> No obstante, sus resultados no se trasladan directamente a este TFM. IberFire utiliza una definición de incendio basada en áreas quemadas superiores a 5 hectáreas, una muestra balanceada y un objetivo de evaluación diferente. Sus resultados de accuracy y AUROC deben interpretarse como evidencia de utilidad del dataset, no como una cota esperable para nuestro modelo. La comparación del presente trabajo se centrará en PR-AUC, Brier score, calibración, recall a tasa de falsa alarma controlada y cobertura de incendios dentro del presupuesto espacial de movilización.

### 5.11 Mejoras futuras de la capa de riesgo preventivo

> Una evolución posterior podrá combinar el peligro meteorológico con una capa explícita de exposición y capacidad de respuesta. Esta capa podría incluir población, interfaz urbano-forestal, espacios protegidos, distancia a carreteras, tiempo estimado de llegada y disponibilidad de medios. La combinación debe evaluarse como una función de pérdida esperada, no como una multiplicación arbitraria de variables. Mientras no existan datos operativos fiables de recursos y tiempos de respuesta, el sistema publicará el ranking de peligro y la acción orientativa, evitando presentarlo como una probabilidad de daño económico o como una orden de movilización.

### 5.12 Proveedor meteorológico alternativo durante el desarrollo

> La ausencia temporal de la clave de MeteoGalicia no bloquea la validación técnica del pipeline. Se ha implementado una selección por configuración mediante `FORECAST_PROVIDER`: `meteogalicia` utiliza MeteoSIX v5, `aemet` utiliza AEMET OpenData y `auto` ejecuta la cadena WRF 1 km → WRF 04 km → AEMET municipal degradado cuando está habilitado. Esta decisión permite probar la doble descarga de AEMET, la normalización horaria, la agregación de la ventana crítica, el archivado, la generación de los tres mapas y el comportamiento del dashboard sin fingir que ambas fuentes tienen la misma resolución.

> AEMET OpenData ofrece predicción horaria municipal hasta 48 horas y predicción diaria para varios días. Para mantener el contrato T+1/T+2/T+3, el adaptador utiliza la predicción diaria para completar 72 horas y superpone las horas de la predicción horaria cuando están disponibles. La expansión diaria se marca en los metadatos como `source_resolution=daily_expansion`; el resultado se etiqueta `fresh_aemet` si se selecciona explícitamente o `fresh_aemet_degraded` cuando actúa como fallback automático. La consulta se realiza sobre un catálogo configurado de municipios, no sobre cada celda de 1 km; las capitales provinciales son solo una configuración de prueba.

> La precipitación requiere una precaución adicional: si la respuesta diaria solo ofrece probabilidad de precipitación y no cantidad en milímetros, el pipeline no la transforma automáticamente en cero. Solo mediante `AEMET_MISSING_PRECIPITATION_FALLBACK=0` se puede crear un artefacto `fresh_aemet_proxy` para comprobar técnicamente la publicación; dicho artefacto queda excluido de las métricas de peligro.

> Esta salida no se incorpora sin más al benchmark de WRF. El rendimiento se separará por proveedor y se reportará junto con la resolución espacial, la cobertura temporal y el horizonte. AEMET sirve para verificar el circuito operativo y para una contingencia explícita; MeteoGalicia WRF 1 km sigue siendo la fuente objetivo para movilización preventiva en zonas rurales de Galicia. Si la ejecución usa AEMET, el dashboard debe advertirlo y el centro de mando debe confirmar cualquier actuación con información independiente.

### 5.13 Estado meteorológico previo y arranque sin precalentamiento

> La inferencia no depende únicamente del forecast futuro. Las features de memoria —lluvia acumulada, días secos y medias móviles— necesitan un estado diario de los 30 días completos anteriores a la emisión. Para evitar que el primer despliegue tenga que esperar treinta ciclos diarios, se implementa un backfill con la climatología diaria de AEMET. La API permite recuperar un rango de fechas para todas las estaciones; el sistema filtra Galicia, conserva el JSON original e interpola cada estación a la rejilla de 1 km mediante los cuatro vecinos más cercanos.

> El backfill no convierte una estación en una observación de cada celda. Es una reconstrucción espacial con una resolución y una calidad propias, por lo que se registran el proveedor, la distancia a estación, el número de estaciones y la estrategia IDW. Esta información se utilizará para distinguir un arranque `aemet_daily_climatology_idw` de un estado basado en una red observacional más densa.

> La documentación de MeteoSIX v5 establece que su operación numérica está orientada al forecast desde el día actual y limita la consulta a un máximo de siete días. Aunque `precipitation_amount` representa la precipitación prevista durante la hora anterior, no es un archivo de observaciones pasadas. AEMET tampoco resuelve por sí sola el cierre operativo de D-1 mediante su climatología diaria validada, que se publica con un retraso aproximado de cuatro días. Por ello, el diseño separa el backfill AEMET del colector de observaciones actuales implementado en `scripts/ingest_aemet_current_observations.py`: este acumula la ventana móvil de AEMET y cierra solo los días con cobertura suficiente. No se utilizará el forecast como observación retrospectiva.

---

## Capítulo 6: Diseño e Implementación del Centro de Mando Táctico y Dashboard Operativo (Fase 5)

### 6.1 Arquitectura del Centro de Mando de Alerta Temprana (Emergency Operations Center - EOC)
*(Justificación del diseño de interfaz para la toma de decisiones en tiempo real)*
> Para transformar las probabilidades predictivas en una herramienta de soporte a la decisión operativa, el sistema se despliega mediante un **Centro de Mando Táctico** interactivo desarrollado sobre Streamlit y enriquecido con un sistema de diseño visual propio para salas de crisis. La arquitectura desacopla la visualización en cinco áreas funcionales: (1) Centro de Mando Cartográfico GIS con capas térmicas continuas y geometrías disueltas sin fronteras artificiales, (2) Analítica Territorial Comarcal y Provincial con rankings de concellos y evolución multi-horizonte $T+1 \rightarrow T+3$, (3) Diagnóstico Biofísico con explicabilidad local TreeSHAP y un simulador *What-If* reactivo, (4) Matriz de Despacho y Protocolos Preventivos alineados con el PLADIGA, y (5) Panel de Auditoría y Trazabilidad de Manifiestos.

### 6.2 Explicabilidad Causal (TreeSHAP) y Simulación Interactiva de Escenarios (*What-If Simulator*)
*(Justificación metodológica de la explicabilidad biofísica para mandos de extinción)*
> En contextos de protección civil y seguridad pública, los modelos de "caja negra" carecen de viabilidad operativa si el analista no puede contrastar los motivos físicos detrás de una alerta. El módulo de diagnóstico implementa **TreeSHAP local aditivo**, descomponiendo la contribución neta de cada variable (en log-odds) entre factores aceleradores (déficit de humedad del combustible, insolación en laderas solanas, velocidad del viento en ventana crítica) y factores atenuantes (precipitación antecedente a 30 días, humedad relativa elevada). Adicionalmente, el **Simulador *What-If*** permite a los directores de extinción evaluar en tiempo real la sensibilidad del territorio ante cambios en la previsión meteorológica (ej. un incremento térmico de $+3^\circ\text{C}$ o rachas de viento de $+20\text{ km/h}$), recalculando instantáneamente la probabilidad calibrada con el artefacto LightGBM serializado.

### 6.3 Alineación con los Protocolos Operativos del PLADIGA y Generación de Briefings de Emergencia
*(Impacto de negocio y transferencia directa a la gestión forestal pública)*
> La interfaz traduce automáticamente los percentiles relativos de riesgo en niveles de activación táctica coherentes con el Plan de Prevención y Defensa contra los Incendios Forestales de Galicia (PLADIGA). Las celdas clasificadas en el percentil superior ($\ge 99.5\%$, Nivel Extremo) disparan recomendaciones de preposición de brigadas helitransportadas (ej. BRIF Laza, bases comarcales) y prohibición de quemas agrícolas. Asimismo, el sistema incorpora un generador automático en 1-click de **Informes Ejecutivos de Situación** en Markdown, estructurando los puntos críticos del territorio y los factores dominantes para las reuniones matinales de coordinación de emergencias.

### 6.4 Simbología Cartográfica del Riesgo: Escala Absoluta $P(Y=1)$ frente a Priorización Relativa por Percentil
*(Decisión metodológica sobre la representación visual del riesgo y corrección del falso concepto de "Criterio Térmico")*
> En el diseño cartográfico de emergencias, la representación cromática del peligro no es una mera elección estética, sino una decisión crítica de modelado visual. Durante las fases iniciales del proyecto se detectó que el término informal "Criterio Térmico" introducía una distorsión conceptual severa ante comités de extinción y tribunales técnicos, al sugerir erróneamente un filtrado por temperatura ambiente o detección de anomalías infrarrojas satelitales (canales térmicos de MODIS/VIIRS). El sistema formaliza dos modos de simbología rigurosamente diferenciados:
> 1. **Riesgo Absoluto Calibrado $P(Y=1)$:** Cartografía la probabilidad física real generada por el modelo supervisado tras la corrección de prior y calibración Platt. Este modo permite la comparabilidad temporal inter-diaria: en una jornada húmeda de primavera la práctica totalidad de Galicia se mostrará en tonalidades suaves o frías, reflejando fielmente la baja probabilidad objetiva de ignición.
> 2. **Priorización Relativa por Percentil (%):** Ordena de forma monotónica las celdas exclusivamente respecto a la distribución del riesgo en Galicia en ese día concreto. Este enfoque es eminentemente táctico y está orientado al despacho de recursos con presupuesto espacial restringido (ej. identificar invariablemente el 1% o 5% de celdas más calientes del día para asignar patrullas fijas, con independencia de que el día sea tranquilo o extremo).

### 6.5 Principio de Divulgación Progresiva y Mitigación de la Paradoja de Prevalencia en Emergencias
*(Diseño de experiencia de usuario y comunicación efectiva ante desbalanceo extremo)*
> Para resolver la sobrecarga cognitiva habitual en los paneles de control geoespaciales, la interfaz del centro de mando implementa el patrón arquitectónico de **Divulgación Progresiva (Progressive Disclosure)** en tres niveles escalonados:
> 1. *Nivel de Conciencia Situacional Inmediata (5 segundos):* Proporciona a los mandos una lectura ejecutiva compacta con tres indicadores determinantes (Nivel de Peligro general en escala semafórica 1–5, extensión territorial en alerta urgente y comarca de atención prioritaria) junto con el mapa táctico principal. Los controles de bajo nivel (capas base alternativas, exploración de archivos `.parquet` históricos) se relegan a contenedores secundarios colapsados.
> 2. *Nivel Táctico Municipal (30 segundos):* Permite la búsqueda ágil por concello y celda, desplegando la ficha semafórica con recomendaciones PLADIGA directas para autoridades comarcales y brigadas de intervención.
> 3. *Nivel Diagnóstico y Gobernanza:* Expone la explicabilidad causal TreeSHAP, el simulador *What-If* y los metadatos de trazabilidad meteorológica para analistas científicos.
>
> Asimismo, la cabecera del sistema aborda explícitamente la **paradoja de la baja prevalencia**: debido a que la frecuencia natural diaria de incendios forestales en una celda de 1 km² en Galicia es de apenas $\sim 0.02\%$ (1 evento por cada $\approx 5.000$ observaciones diarias), un valor predictivo del $8\%$ o $12\%$ puede ser erróneamente interpretado por un perfil no estadístico como "bajo". La interfaz clarifica de forma contextual que una probabilidad del $10\%$ representa una multiplicación del riesgo basal por más de 500 veces, legitimando plenamente la activación de protocolos de prealerta máxima.

### 6.6 Calibración de Severidad Institucional y Prevención de Falsas Alarmas Tautológicas
*(Resolución del desacoplamiento entre despacho táctico y severidad física)*
> En el diseño de sistemas de alerta temprana de protección civil, uno de los fallos operacionales más destructivos es el denominado «síndrome de Pedro y el lobo» (*cry wolf effect*), en el cual el sistema emite alarmas catastróficas en condiciones de riesgo físico moderado o nominal, induciendo a los operadores humanos a desconectar o ignorar la herramienta. 
>
> Durante la auditoría del centro de mando se detectó un error metodológico recurrente en la literatura aplicada: condicionar el nivel de severidad global autonómica a umbrales absolutos sobre conteos de percentiles (ej. `if num_celdas_top_05 > 100`). Dado que la cuadrícula fija de Galicia comprende 29.601 celdas de 1 km², el percentil $99.5\%$ (Top 0.5%) contendrá por definición matemática invariable $29.601 \times 0.005 = 148$ cuadrículas. Si el evaluador compara un cardinal fijo ($148 > 100$), el sistema activará la máxima alerta («Nivel 5 — Extremo») de forma tautológica todos los días del año, incluso en jornadas invernales donde la probabilidad física máxima sea inferior al $0.05\%$.
>
> La arquitectura implementada en este TFM corrige este desacoplamiento separando estrictamente dos conceptos:
> 1. **La Severidad Física Autonómica:** Determinada de forma determinista mediante la probabilidad calibrada real $P(Y=1)$ alcanzada en el territorio (Nivel 1 Bajo $<1.0\%$, Nivel 2 Moderado $1.0\%-2.5\%$, Nivel 3 Alto $2.5\%-6.0\%$, Nivel 4 Muy Alto $6.0\%-12.0\%$, Nivel 5 Extremo $\ge 12.0\%$). Bajo esta escala rigurosa, una jornada con máximas de $28^\circ\text{C}$ y rachas de $36\text{ km/h}$ que alcanza un $P_{\text{máx}} = 1.88\%$ es clasificada correctamente como **Nivel 2 — Moderado**, con tarjeta ámbar e indicaciones proporcionadas, eliminando el pánico operacional injustificado.
> 2. **La Priorización Relativa de Despacho:** Utilizada para distribuir eficientemente los medios de extinción disponibles (Top 0.5% o Top 1.0% de cuadrículas preferentes), reconociendo explícitamente que pertenecer al 0.5% superior de Galicia no equivale a riesgo extremo cuando la jornada en su conjunto es física y meteorológicamente templada.
> 3. **Sub-graduación de Percentiles Superiores:** Para evitar la generación de mapas monocromáticos donde todas las cuadrículas del Top 0.5% colapsaban al mismo valor cromático morado uniforme, se implementó una escala de alta resolución en los percentiles de cola ($99.8\%$ Crítico `#800026`, $99.5\%$ Muy Alto `#BD0026`, $98.0\%$ Prioritario `#E31A1C`), complementada con guías tácticas contextuales que instruyen al usuario a activar el modo de Riesgo Absoluto para visualizar contrastes continuos de probabilidad.

### 6.7 Armonización Semántica de Componentes y Ergonomía Operativa de las Pestañas de Apoyo
*(Unificación de métricas cuantitativas, matrices operativas PLADIGA y mitigación de sobrecarga cognitiva)*
> Para garantizar la coherencia táctica en todas las vistas de toma de decisiones del Centro de Mando, se llevó a cabo una auditoría y armonización transversal sobre los cinco módulos funcionales secundarios:
> 1. **Consulta Municipal y Ficha de Amenaza (`concello_lookup`):** La interfaz vincula la semaforización cualitativa con el valor cuantitativo exacto de la probabilidad calibrada $P(Y=1)$ (ej. «Nivel 2 — Moderado (1.88%)»). Esto permite a los técnicos de protección civil municipal evaluar de inmediato si un territorio clasificado en nivel moderado se sitúa próximo al límite inferior de activación ($1.0\%$) o al umbral de riesgo alto ($2.5\%$), aportando precisión granular a los comités locales.
> 2. **Analítica Territorial y Rankings Distritales (`territorial_analytics`):** Se eliminó la etiqueta de "Riesgo Extremo / Alerta Urgente" en las columnas de ordenación de distritos, sustituyéndola por una nomenclatura estadística transparente: `Cuadrículas Top 5%` y `Cuadrículas Top 0.5%`. Esta modificación erradica la falacia de asumir que el subconjunto de celdas relativamente más calientes de un distrito represente peligro crítico en días con baja carga térmica general.
> 3. **Diagnóstico Biofísico y Simulador *What-If* (`shap_simulator`):** Se integró la retroalimentación numérica directa en las tarjetas comparativas de impacto ($\text{Probabilidad Basal } P_{\text{base}} \rightarrow \text{Probabilidad Simulada } P_{\text{sim}}$), permitiendo a los analistas comprobar cuantitativamente el efecto marginal de variaciones térmicas ($+3^\circ\text{C}$), descensos de humedad relativa ($-15\%$) o aceleraciones de viento ($+15\text{ km/h}$) sobre el artefacto LightGBM calibrado.
> 4. **Matriz de Medidas y Despacho PLADIGA (`operational_protocols`):** Se sustituyó el esquema obsoleto de 4 niveles por la escala institucional canónica de 5 niveles del PLADIGA (Nivel 1 Bajo $<1.0\%$, Nivel 2 Moderado $1.0\%-2.5\%$, Nivel 3 Alto $2.5\%-6.0\%$, Nivel 4 Muy Alto $6.0\%-12.0\%$, Nivel 5 Extremo $\ge 12.0\%$). Con ello, los protocolos de movilización de retenes, brigadas helitransportadas (BRIF) y suspensión de permisos de quema quedan perfectamente sincronizados con la severidad autonómica computada en el módulo de KPIs.
> 5. **Ergonomía de Cabecera y Trazabilidad Técnica (`styles` y `system_audit`):** Se solventó el conflicto de solapamiento visual del título táctico respecto a la barra superior nativa de Streamlit mediante un ajuste de margen superior (`padding-top: 3.25rem !important;`), y se eliminaron los avisos técnicos de advertencia sobre proxies meteorológicos del panel operativo principal, recluyéndolos estrictamente en la pestaña de auditoría del sistema para preservar la concentración de los mandos en situaciones de crisis.

---

