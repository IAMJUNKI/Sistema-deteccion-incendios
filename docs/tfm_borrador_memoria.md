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
- [Capítulo 7: Arquitectura de Despliegue, MLOps y Estrategia de Reproducibilidad](#capítulo-7-arquitectura-de-despliegue-mlops-y-estrategia-de-reproducibilidad)

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

### 3.8 Resolución de la Latencia Observacional y Fusión Multi-Fuente (AEMET vs. MeteoGalicia EMA)
*(Justificación metodológica del cierre de la brecha $D-4 \rightarrow D-1$ en producción)*
> En la arquitectura operativa en tiempo real, el cálculo de las variables retrospectivas de memoria biometeorológica (`dias_sin_lluvia`, `prec_acum_3d`, `prec_acum_7d` y `tmax_media_7d`) impone un reto de ingeniería de datos crítico: la **latencia de publicación de las fuentes oficiales**. La climatología diaria validada de AEMET sufre un desfase estructural de 3 a 5 días debido a sus filtros centralizados de control de calidad institucional, lo que deja un vacío sistemático entre el día $D-4$ y la víspera de la predicción ($D-1$).
>
> Para resolver este desajuste sin recurrir a suposiciones sintéticas ni a la invención de ceros —lo que arruinaría las estimaciones de desecación del combustible fino—, el pipeline implementa una **estrategia de fusión multi-fuente escalonada**:
> 1. **Base Climatológica Consolidada ($< D-4$):** Registros validados de la red climatológica de AEMET, idóneos para la serie temporal de 30 días de fondo.
> 2. **Cierre de la Brecha Reciente ($D-4 \rightarrow D-1$, 72h–120h):** Ingesta automatizada de la Red de Estaciones Meteorológicas Automáticas (EMA) de MeteoGalicia a través de su servicio Open Data (`mgrss/observacion`). Con más de 140 estaciones distribuidas sobre el territorio gallego (una densidad de ~1 estación cada 175 km² frente a los ~857 km² por estación de AEMET), se recuperan las mediciones físicas reales de temperatura, humedad relativa mínima, precipitación diaria y rachas de viento, interpolándolas a la cuadrícula de 1 km² mediante IDW de 4 vecinos.
> 3. **Degradación Elegante (Fallback de Contingencia):** Si por incidencia de conectividad o mantenimiento de MeteoGalicia no se pudieran obtener las observaciones de estación, el sistema rescata de forma controlada el pronóstico numérico archivado previamente emitido, marcando la calidad de la feature como `forecast_proxy` para que la inferencia diaria nunca se detenga.
>
> **Impacto Físico Real (Resumen Negocio / Tribunal):** En la física de incendios, la pérdida de humedad de los combustibles finos (hojarasca, pasto seco) responde a una escala temporal ultracorta de 24 a 72 horas. Un error de 5 mm de lluvia o de 3 °C en los últimos tres días altera drásticamente la probabilidad de ignición. Disponer de mediciones físicas directas de la red rural de MeteoGalicia para los días $D-3$, $D-2$ y $D-1$ garantiza que el modelo evalúa la sequedad real del sotobosque gallego y no una aproximación numérica abstracta, alineando este trabajo con las recomendaciones del nuevo Índice de Peligro de Incendios Forestales (IPIF de AEMET, 2026) y el marco de alta resolución de IberFire (Ercibengoa et al., 2025).

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

### 5.14 Cierre de la Brecha Observacional Reciente con la Red de Estaciones de MeteoGalicia (EMA)

> Para superar la dependencia de procesos en bucle permanente (como el colector de la ventana móvil de 12 horas de AEMET) y asegurar un arranque determinista en cualquier instante, se ha incorporado el cliente e interpolador `src.ingestion.meteogalicia_observations` y el script operativo `scripts/ingest_meteogalicia_observations.py`. 
>
> Este componente consulta el servicio Open Data de datos diarios de MeteoGalicia (`datosDiariosEstacionsMeteo.action`), recuperando automáticamente las observaciones físicas consolidadas para las más de 140 estaciones de la red gallega entre la última fecha registrada en el estado y el día $D-1$ (típicamente las últimas 72 a 120 horas). Las lecturas se proyectan de UTM 29N a WGS84, se normalizan sus magnitudes físicas (transformando el viento de m/s a km/h y derivando el VPD crítico), y se interpolan a la malla de 1 km mediante IDW de 4 vecinos más cercanos.
>
> Esta solución dota al pipeline de una resiliencia operacional completa: permite que un despliegue en servidor o una máquina local se enciendan en cualquier momento del día, detecten automáticamente los días faltantes con `--auto-fill-gap` y cierren el estado meteorológico hasta la víspera sin vacíos temporales y con datos medidos en el terreno gallego.

### 5.15 Especialización Multi-Horizonte ($T+1, T+2, T+3$), Parsimonia Operativa (48 vs. 50 Variables) y Validación en Test Ciego 2023
*(Justificación de la arquitectura desacoplada de predictores y del diseño experimental pareado)*
> La evaluación de modelos predictivos sobre series temporales impone dos requerimientos metodológicos esenciales para garantizar la aplicabilidad en operaciones de emergencia:
>
> 1. **Especialización y desacoplo por horizonte ($T+1, T+2, T+3$):** Predecir a 24h, 48h o 72h no constituye una mera reevaluación del mismo modelo con datos desplazados. La estructura física de la información difiere: mientras que para $T+1$ la memoria antecedente de precipitación y humedad proviene al 100% de observaciones físicas cerradas de la red EMA, en $T+2$ y $T+3$ las memorias recientes deben encadenar predicciones numéricas intermedias, acumulando incertidumbre. En el entrenamiento, cada horizonte se formula como un modelo LightGBM independiente con su propia etiqueta objetivo ($issue\_date + h$), su muestra determinista de negativos (`seed = 42 + h`) y su calibrador sigmoide específico (Platt scaling ajustado sobre 2021). Como resultado empírico, la importancia relativa del déficit de presión de vapor (`vpd_mean`) disminuye de $>16\%$ en $T+1$ a $\approx 13\%$ en $T+3$, cediendo peso a ventanas acumuladas para estabilizar la señal.
> 2. **Principio de parsimonia y diseño experimental pareado (48 vs. 50 variables):** El contrato operativo `egif-2d-48-v1` suprime dos variables redundantes del prototipo canónico (`precipitation_sum` del día objetivo y `consecutive_dry_days`), cuya información queda capturada de forma continua y libre de fugas por las ventanas de 3 a 30 días y el VPD. Para contrastar esta compactación sin confundir el efecto de las variables con el volumen de entrenamiento, el benchmark del test ciego 2023 (10.804.365 celdas-día, 530 igniciones) evalúa un diseño pareado:
>    - `EGIF 50 control alineado`: 50 variables bajo la nueva semántica temporal estricta (entrenado con 2019–2020).
>    - `EGIF 48 comparable`: 48 variables bajo las mismas particiones (2019–2020), demostrando que la parsimonia preserva la capacidad discriminante.
>    - `EGIF 48 ampliado`: 48 variables entrenado sobre el histórico completo (2016–2020), erigiéndose como el candidato ganador al alcanzar un ROC-AUC de $0{,}8713$ y un Recall del $9{,}62\%\text{--}11{,}32\%$ en el $1\%$ superior de celdas diarias prioritarias.
>    - `FWI CEMS`: Baseline de referencia externo, superado ampliamente en todas las métricas operativas (Recall top 1% de $2{,}83\%$).

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

### 6.8 Auditoría Integral de Fidelidad Territorial, Normalización Cartográfica y Robustez Explicativa (TreeSHAP)
*(Corrección de asimetrías espaciales, distorsión planar, desajustes en explicabilidad local y alertas termo-higrométricas)*
> En la transición del prototipo de investigación a un Centro de Mando Táctico operable en tiempo real, se llevó a cabo una auditoría integral de extremo a extremo sobre la totalidad de los componentes y flujos de datos del dashboard, identificando y subsanando cinco deficiencias metodológicas y geoespaciales críticas:
>
> 1. **Normalización Territorial Exacta y Erradicación del Sesgo Provincial:**
>    En versiones preliminares, la asignación administrativa provincial se realizaba mediante heurísticas cartesianas planas aproximadas (`lat < 42.45` y `lon > -8.25`). Dicho enfoque amputaba más del $50\%$ de la superficie de la provincia de Pontevedra (incluyendo comarcas enteras como Deza, Tabeirós-Terra de Montes o O Salnés, tales como Lalín, A Estrada y Vilagarcía de Arousa), reclasificándolas erróneamente bajo A Coruña. Para erradicar este sesgo, se procedió a la integración topológica estricta con la capa vectorial oficial de provincias del Instituto Geográfico Nacional (IGN, formato GeoJSON), precomputando tablas de correspondencia indexadas por `cell_id` (`galicia_grid_1km_egif_admin.parquet` para las 29.601 cuadrículas canónicas y `grid_galicia_centroids_admin.parquet` para la serie histórica). Los conteos resultantes coinciden con exactitud milimétrica con la delimitación territorial oficial: Lugo (9.859 km²), A Coruña (7.949 km²), Ourense (7.282 km²) y Pontevedra (4.511 km²).
>
> 2. **Integración Canónica de los 19 Distritos Forestales del PLADIGA (I a XIX):**
>    Se incorporaron los centroides y demarcaciones de los 19 Distritos Forestales oficiales definidos por el Plan de Prevención y Defensa contra los Incendios Forestales de Galicia (PLADIGA, Xunta de Galicia), asegurando que tanto los resúmenes ejecutivos como la ordenación de comarcas reflejen las unidades territoriales operativas empleadas por los agentes ambientales y directores de extinción.
>
> 3. **Corrección de la Distorsión Métrica Planar en la Búsqueda Municipal:**
>    En la latitud media de Galicia ($\bar{\phi} \approx 42.6^\circ\text{ N}$), un grado de longitud abarca aproximadamente $82\text{ km}$, mientras que un grado de latitud equivale a $111\text{ km}$ ($\cos(42.6^\circ) \approx 0.7361$). El cómputo euclidiano plano no ponderado introducía una distorsión métrica del $36\%$ en la componente longitudinal, distorsionando la vinculación entre centroides de celda y cabeceras de concello. Se reescribió el algoritmo de proximidad aplicando la métrica esférica equirrectangular local:
>    $$\Delta d = \sqrt{(\Delta \text{lat})^2 + (\Delta \text{lon} \cdot \cos(\bar{\phi}))^2}$$
>    ampliando asimismo la base municipal a más de 45 concellos de referencia distribuidos homogéneamente sobre las cuatro provincias.
>
> 4. **Desacoplamiento Operativo: Regla 30-30-30 vs. Alerta Termo-Higrométrica 30-30:**
>    La regla clásica 30-30-30 exige simultáneamente $T \ge 30^\circ\text{C}$, $HR \le 30\%$ y racha de viento $V \ge 30\text{ km/h}$. En episodios de ola de calor bajo domos térmicos de alta presión o valles interiores con atmósfera en calma, el viento en superficie puede registrar velocidades inferiores a $25\text{ km/h}$. Reportar "0 km² bajo condición crítica" en la tarjeta de telemetría transmitía una falsa percepción de calma operacional ante situaciones con desecación extrema del combustible fino. La cabecera fue rediseñada para distinguir de forma dinámica dos umbrales: si concurre viento intenso, se activa la `Condición Crítica 30-30-30` (alerta roja); si el viento es moderado pero existe sequedad térmica acusada, la interfaz activa la `Alerta Termo-Higrométrica 30-30` (alerta ámbar), visibilizando con total transparencia los miles de kilómetros cuadrados en peligro de ignición.
>
> 5. **Robustez Numérica y Contrato Dimensional en Explicabilidad Local (TreeSHAP):**
>    El diagnóstico biofísico local mediante TreeSHAP (`explain_tree_prediction`) experimentaba interrupciones en tiempo de ejecución al evaluar celdas del modelo de producción, debido a la discrepancia dimensional con el contrato congelado `egif-2d-48-v1` y la persistencia de columnas con tipo de datos `object` en el DataFrame de inferencia. La rutina se refactorizó para asegurar el alineamiento estricto con las 48 variables canónicas mediante `ensure_feature_matrix_for_contract`, garantizando tipado numérico `float` puro y completitud defensiva. Adicionalmente, se implementaron etiquetas semánticas inteligibles en castellano (`FEATURE_LABELS`) y se acoplaron coherentemente las covariables meteorológicas medias en el simulador *What-If* (`temperature_mean`, `relative_humidity_mean`, `wind_speed_mean`, `vpd_mean`) para evitar configuraciones físicas contradictorias.
>
> 6. **Supresión Óptica del Entorno Exterior y Demarcación Orgánica de Sectores (PLADIGA):**
>    Para evitar la dispersión cognitiva de los operadores de emergencia ante territorios limítrofes fuera de su competencia competencial, se incorporó una máscara inversa exterior sobre el contorno oficial del Instituto Geográfico Nacional con una densidad de atenuación del $74\%\text{--}82\%$, prescindiendo de contornos perimetrales artificiales invasivos. Portugal, Castilla y León, Asturias y el medio marino quedan ensombrecidos en penumbra táctica, destacando a Galicia como el único teatro operativo activo. Asimismo, para la exploración sectorial, se descartaron los cuadrantes rectangulares rígidos (*bounding boxes*) y los bordes rayados artificiales, sustituyéndolos por un **overlay translúcido y sutil** sobre los **polígonos orgánicos continuos** construidos mediante la disolución espacial exacta de los 19 Distritos Forestales oficiales del PLADIGA (`galicia_sectors.geojson`, `fillOpacity: 0.08`, sin líneas de contorno). Al seleccionar un sector (ej. *Ourense Sur — Monterrei*, *Rías Baixas*, *Costa da Morte*), la interfaz resalta de forma limpia el área comarcal sin elementos obstructivos y prioriza las celdas críticas pertenecientes exclusivamente a dicho sector.
>
> 7. **Especialización Funcional y Selector Rápido de Cartografía en la Vista Táctica:**
>    Para evitar la dispersión de controles entre la barra lateral y el visor espacial, se reestructuró la cabecera de la pestaña cartográfica en una barra de herramientas de tres componentes esenciales: **Sector Territorial (PLADIGA)**, **Estilo de Mapa Base** y **Localizar Celda ID**. Al sustituir el selector municipal de centrado (cuyo análisis exhaustivo queda centralizado con métricas y rankings en la pestaña dedicada «Consulta por Concello»), el operador de emergencias puede alternar directamente en la propia vista activa entre el relieve físico orográfico, el lienzo táctico neutro (*Esri Gris Claro*), imágenes satelitales PNOA u ortofotos, sin necesidad de navegar por menús colapsados en el panel de configuración.
>
> 8. **Cartografía Base Institucional Oficial (IGN España) y Nomenclatura Descriptiva:**
>    Se adoptó como estándar visual predeterminado la ortofotografía aérea de máxima resolución del Instituto Geográfico Nacional (**«IGN España (Ortofoto Oficial)»**, capa PNOA de máxima actualidad), proporcionando una base fotorrealista canónica e institucional. Asimismo, se simplificó la taxonomía cartográfica eliminando marcas y siglas accesorias en favor de **nombres puramente funcionales y descriptivos**: *IGN España (Ortofoto Oficial)*, *Relieve Topográfico*, *Lienzo Claro*, *Lienzo Oscuro*, *Satélite* y *Callejero*. Esta capacidad multimodal permite a los directores de extinción contrastar al instante la distribución de las cuadrículas críticas con el territorio real gallego (combustibles forestales, fondos de valle térmicos, cañones de los ríos Sil y Miño, sierras escarpadas del Macizo Central Ourensán y orientación solana/umbría).
>
> ---
>
> **Impacto Físico Real y Alineación con el Estado del Arte (Resumen Negocio / Tribunal):**
> Los modelos continentales tradicionales de alerta de incendios, como el EFFIS/GEFF de Copernicus o las mallas globales de reanálisis, proporcionan resoluciones espaciales agregadas de entre $10\text{ km}$ y $25\text{ km}$, diluyendo la orografía abrupta y los microclimas gallegos. Por su parte, iniciativas pioneras como IberFire (Ercibengoa et al., 2025) han demostrado la validez de la resolución a $1\text{ km} \times 1\text{ km}$, y el nuevo Índice de Peligro de Incendios Forestales de AEMET (IPIF, 2026) confirma la necesidad institucional de sustituir los índices canadienses empíricos clásicos (FWI) por formulaciones multimodales calibradas que integren la humedad real del suelo, la inflamabilidad del estrato vegetal y la topografía.
>
> En la física de incendios, la pendiente acelera la velocidad de propagación exponencialmente (el frente asciende precalentando por radiación y convección el combustible fino superior), mientras que la orientación solana multiplica la evapotranspiración diurna frente a las umbrías húmedas. Resolver la asignación administrativa exacta y proyectar las predicciones supervisadas a $1\text{ km}^2$ con interpretabilidad TreeSHAP causal dota a los comités de emergencias y a los mandos del PLADIGA de un instrumento con base científica rigurosa: no solo saben qué cuadrícula concreta presenta una probabilidad calibrada crítica, sino exactamente qué balance entre déficit de presión de vapor, sequedad acumulada y masa de combustible está impulsando el riesgo en ese instante.

---

## Capítulo 7: Arquitectura de Despliegue, MLOps y Estrategia de Reproducibilidad

### 7.1 Desacoplamiento entre entrenamiento pesado y distribución operativa mediante Hugging Face Hub
*(Estrategia de reproducibilidad "Zero-Retrain" y empaquetado de artefactos)*
> Uno de los mayores retos en la transferencia de proyectos de Machine Learning geoespacial al ámbito operacional y evaluador es la barrera de entrada que suponen los datasets masivos. La reconstrucción retrospectiva del datacubo de Galicia (2016–2023) y el reentrenamiento supervisado de los modelos LightGBM calibrados requieren la descarga y procesamiento de más de 20 GB de ficheros raster NetCDF (ERA5-Land), capas vectoriales complejas (CORINE Land Cover y OpenStreetMap) y tabulares anuales particionados.
>
> Para garantizar la plena reproducibilidad del sistema ante el tribunal académico y facilitar su despliegue inmediato en nuevos entornos sin incurrir en horas de reentrenamiento, se ha adoptado una arquitectura desacoplada basada en **Hugging Face Hub**:
> 1. **Artefactos Estáticos Inmutables (~11 MB):** Agrupan los modelos serializados calibrados por horizonte temporal (`forecast_risk_egif_48_t1/t2/t3.joblib`), sus contratos de metadatos (`.json`), el manifiesto de trazabilidad activa (`active_model_manifest.json`) y la rejilla canónica de 29.601 celdas (`galicia_grid_1km_egif.parquet`).
> 2. **Artefactos Dinámicos Operativos (~160 MB):** Comprenden el pronóstico diario publicado (`predicciones_operativas.parquet`) y el estado meteorológico acumulado de 30 días (`weather_daily_state.parquet`).
>
> Mediante un cliente de aprovisionamiento desasistido (`scripts/download_artifacts.py`), cualquier usuario puede inicializar el Centro de Mando Táctico en local en menos de dos minutos con una descarga inferior a 25 MB (modo ligero) o 170 MB (modo completo), asegurando que el código clonado desde GitHub opere de manera idéntica al entorno de producción sin requerir credenciales complejas de nubes privadas.

### 7.2 Resolución del "Cold Start" Meteorológico y Pipeline de Sincronización Continua
*(Gobernanza del estado antecedente de 30 días y automatización diaria)*
> En la modelización del riesgo de incendio, variables críticas como la racha de días secos consecutivos o la precipitación acumulada a 30 días exigen una memoria retrospectiva continua. Un despliegue estándar que intente calcular una predicción diaria se enfrenta al problema clásico de **Cold Start (arranque en frío)**: para predecir el riesgo del día $T$, el sistema necesitaría consultar de golpe las observaciones horarias de 30 días pasados a través de las APIs institucionales de MeteoGalicia o AEMET, lo que provocaría latencias inaceptables y bloqueos por saturación de cuotas.
>
> El pipeline operativo resuelve esta limitación mediante una **estrategia de estado deslizante (Warm Start)** sincronizada bidireccionalmente:
> 1. **Actualización Incremental de Brecha (`auto-fill-gap`):** El script de ingesta de observaciones (`ingest_meteogalicia_observations.py`) examina la fecha máxima registrada en el parquet de estado local y descarga exclusivamente el intervalo temporal faltante hasta $D-1$ (típicamente 24 a 48 horas), interpolando las medidas de las estaciones de MeteoGalicia (EMA) y actualizando atómicamente la ventana móvil sin recalcular los 30 días previos.
> 2. **Sincronización Automática Servidor $\rightarrow$ Hugging Face Hub:** Cada mañana, tras completarse la inferencia predictiva en el servidor de producción Linux (programada mediante `systemd timer` a las 05:15 CET), un proceso automatizado (`scripts/publish_to_huggingface.py`) sube el estado meteorológico recién consolidado y las predicciones resultantes al repositorio de Hugging Face. De este modo, la comunidad y los evaluadores disponen de forma transparente y permanente de la última foto operativa del territorio gallego sin intervención manual.

### 7.3 Arquitectura y Flujo de Ejecución del Pipeline Operativo Diario en Producción
*(Descripción integral del ciclo diario de inferencia: desde la captura de datos hasta la alerta táctica)*
> Para materializar la transferencia tecnológica del modelo de investigación a un entorno de misión crítica, se ha diseñado un pipeline operativo de ejecución desasistida (*batch processing*) concebido bajo tres pilares fundamentales: **robustez ante fallos externos, ausencia estricta de fugas de datos (*data leakage*) e interpretabilidad física inmediata**.
>
> El pipeline se ejecuta de forma totalmente automatizada cada madrugada mediante un temporizador de sistema (`systemd timer`) programado a las **05:15 CET**, transformando las previsiones meteorológicas y el estado antecedente del territorio en **tres mapas predictivos diarios independientes ($T+1, T+2, T+3$)** para las 29.601 cuadrículas de $1\text{ km} \times 1\text{ km}$ que componen la Comunidad Autónoma de Galicia.
>
> ```mermaid
> flowchart TD
>     A[05:15 CET: Scheduler systemd timer] --> B[Adquisición del Lock de Concurrencia]
>     B --> C[Actualización de Memoria Antecedente D-30 a D-1\nRed EMA MeteoGalicia + Climatología AEMET]
>     C --> D[Ingesta y Validación de Previsión Futura a 72h\nCadena: WRF 1km -> WRF 4km -> AEMET]
>     D --> E[Proyección a Rejilla Canónica 1km²\n29.601 celdas de Galicia]
>     E --> F[Ingeniería de Variables en Ventana Crítica 12h-18h\nContrato egif-2d-48-v1 sin Data Leakage]
>     F --> G1[Inferencia T+1 LightGBM + Platt]
>     F --> G2[Inferencia T+2 LightGBM + Platt]
>     F --> G3[Inferencia T+3 LightGBM + Platt]
>     G1 --> H[Estratificación Táctica:\nSeveridad PLADIGA 1-5 + Percentiles Top 1%]
>     G2 --> H
>     G3 --> H
>     H --> I[Publicación Atómica y Criptográfica:\nParquet + Manifiesto SHA-256]
>     I --> J[Centro de Mando Táctico EOC\nDashboard Streamlit desacoplado]
>     I --> K[Sincronización Hugging Face Hub]
> ```
>
> A continuación se detalla la lógica de ejecución estructurada en siete fases consecutivas:
>
> 1. **Fase 1 — Control de Concurrencia y Bloqueo Atómico:** Al activarse el proceso, el orquestador (`scripts/run_daily_inference.py`) genera un archivo de exclusión mutua (`data/processed/.daily_inference.lock`) registrando el PID, host y timestamp de la instancia. Este mecanismo garantiza que ejecuciones solapadas o reintentos manuales nunca colisionen ni sobreescriban ficheros en disco durante una corrida activa.
> 2. **Fase 2 — Consolidación de la Memoria Ambiental Antecedente ($D-30 \rightarrow D-1$):** Ningún modelo de incendios puede predecir el peligro analizando únicamente el día presente; la susceptibilidad a la ignición depende de cuántos días consecutivos lleva el bosque sin llover y de la evaporación acumulada. El pipeline actualiza el archivo de estado móvil (`weather_daily_state.parquet`), cerrando el intervalo reciente ($D-4 \rightarrow D-1$) mediante las lecturas físicas de las más de 140 estaciones automáticas de MeteoGalicia (EMA) interpoladas por IDW de 4 vecinos, sobre la base consolidada de AEMET ($< D-4$). Siguiendo un estricto principio *anti-data-leakage*, la jornada de emisión ($D$) se excluye de la memoria antecedente al considerarse un día abierto e incompleto a las 05:00.
> 3. **Fase 3 — Ingesta Resiliente del Pronóstico Numérico Futuro (72 horas):** El pipeline consulta la previsión meteorológica para los días $T+1, T+2$ y $T+3$ a través de una cadena adaptativa de degradación controlada (*graceful degradation*):
>    - *Fuente Primaria (WRF 1 km):* Modelo numérico de alta resolución de MeteoGalicia (`1km`), consultado en lotes óptimos de 20 localizaciones cada 4 km para minimizar la latencia de red.
>    - *Fallback 1 (WRF 4 km):* Si la salida de 1 km aún no ha finalizado su corrida de cálculo numérico, el sistema conmuta automáticamente a la malla WRF de 4 km, registrando en los metadatos `fresh_fallback`.
>    - *Fallback 2 (AEMET Municipal) / Fallback 3 (Stale):* En caso de caída de la red regional, el pipeline puede recurrir a las predicciones municipales de AEMET o, como última contingencia, al último pronóstico archivado que cubra el horizonte, marcándolo con transparencia como `stale`.
>    - *Validación de Integridad Física:* Se verifica exhaustivamente que no existan valores nulos, horas ausentes o anomalías de formato (ej. humedades relativas fuera de $[0, 100]\%$ o valores centinela `-9999`). Si la integridad no es perfecta, el pipeline aborta la corrida antes de propagar datos corruptos.
> 4. **Fase 4 — Ingeniería de Características en la Ventana Crítica (Contrato `egif-2d-48-v1`):** Una vez proyectado el pronóstico sobre la cuadrícula territorial de 29.601 celdas, se extraen las variables correspondientes a la **ventana crítica de peligro (12:00 a 18:00 h local peninsular)**:
>    - *Extremos atmosféricos:* Temperatura máxima ($T_{\text{max, vc}}$), humedad relativa mínima ($RH_{\text{min, vc}}$), racha máxima de viento ($V_{\text{max, vc}}$) y déficit de presión de vapor ($VPD_{\text{vc}}$).
>    - *Acumulaciones hídricas y sequedad:* Precipitación total en 24h, días secos consecutivos y acumulados móviles a 3, 7, 14 y 30 días. Para $T+1$ la memoria proviene al 100% de observaciones pasadas; para $T+2$ y $T+3$ se encadenan coherentemente los días futuros precedentes, preservando el aislamiento causal.
>    - *Capas fisiográficas y antrópicas:* Elevación, pendiente e índice de radiación solana (DEM de Copernicus), fracciones de combustible vegetal por clases (CORINE Land Cover 2024 para producción) y distancias a redes viarias e interfaces urbanas.
> 5. **Fase 5 — Inferencia Multi-Horizonte y Calibración de Probabilidades:** Las matrices de 48 variables se alimentan de forma independiente a tres modelos LightGBM serializados (`forecast_risk_egif_48_t1/t2/t3.joblib`), especializados por horizonte temporal. Los *log-odds* generados por los árboles se transforman en probabilidades físicas reales de ignición $P(Y=1)$ mediante calibración sigmoide de Platt (ajustada sobre datos independientes con prevalencia natural), garantizando que las probabilidades reflejen la frecuencia estadística real de fuegos en el territorio gallego.
> 6. **Fase 6 — Estratificación Táctica y Despacho Operativo:** Para evitar que las probabilidades matemáticas abstractas confundan a los equipos de emergencia, el sistema traduce la salida a dos escalas complementarias:
>    - *Severidad Física Absoluta (PLADIGA 1–5):* Categorización objetiva del peligro basada en umbrales de probabilidad calibrada (Nivel 1 Bajo $<1.0\%$ hasta Nivel 5 Extremo $\ge 12.0\%$), vinculada directamente a los protocolos de movilización de medios y suspensión de quemas del Plan gallego.
>    - *Priorización Relativa de Despacho (Percentiles):* Identificación monotónica del Top 0.5%, Top 1% y Top 5% de cuadrículas relativamente más amenazadas del día, permitiendo a los directores de extinción optimizar la distribución de patrullas y retenes terrestres bajo presupuestos de recursos limitados.
> 7. **Fase 7 — Publicación Atómica, Trazabilidad y Visualización Desacoplada:** La salida definitiva se escribe en un archivo temporal y se publica de forma atómica mediante `os.replace` (`predicciones_operativas.parquet`), acompañada de un archivo de manifiesto criptográfico (`predicciones_operativas.manifest.json`). El manifiesto registra los identificadores de ejecución, tiempos de cálculo, procedencia de datos, versiones de contrato y firmas hash SHA-256 tanto de los modelos empleados como del fichero de salida. El Centro de Mando Táctico (Streamlit) opera como un consumidor pasivo que lee el artefacto ya generado en disco, asegurando que ningún usuario que acceda a la web sobrecargue la máquina recalculando predicciones. Paralelamente, los resultados se sincronizan automáticamente con Hugging Face Hub para su consulta pública y auditoría evaluadora.
>
> ---
>
> **Impacto Físico y Relevancia Operativa (Resumen Negocio / Tribunal):**
> Frente a las alertas meteorológicas continentales tradicionales (como el FWI europeo de EFFIS o AEMET a escala de $10\text{--}25\text{ km}$), que durante una ola de calor declaran en "alerta roja" a tres provincias enteras sin discriminar dónde se ubican los recursos, este pipeline operativo aporta una ventaja decisiva: **resolución hiperlocal a $1\text{ km} \times 1\text{ km}$ con una tasa de falsas alarmas controlada ($FPR \le 5\%$)**.
>
> Al evaluar no solo si hace calor o viento, sino la desecación física real de los combustibles finos (mediante el VPD y la lluvia acumulada de la red rural de MeteoGalicia), el tipo de masa forestal (coníferas y eucaliptales de alta combustibilidad frente a frondosas caducifolias) y la orientación de las laderas (solanas precalentadas frente a umbrías húmedas), el sistema entrega a las 06:00 de la mañana un mapa de intervención quirúrgico. Esto permite a los mandos de extinción preposicionar helicópteros y brigadas (BRIF) en el 1% del territorio con verdadero riesgo crítico antes de que se inicie la jornada laboral y las horas de máxima insolación vespertina, transformando la gestión del fuego de una respuesta reactiva a una prevención predictiva auditable.

