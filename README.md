# Sistema de estimación del peligro diario de incendio forestal en Galicia

Estima, para cada celda de 1 km² de Galicia y cada día, la probabilidad de que se inicie un
incendio. Se entrena con el registro oficial de incendios del Ministerio (EGIF) y con
meteorología, topografía, cobertura del suelo y actividad humana, y se compara contra el índice
FWI que publican hoy AEMET y el servicio europeo de emergencias.

El sistema está desplegado: cada madrugada publica cuatro mapas de peligro para toda Galicia —el
del día en curso y los de los tres siguientes— en un panel accesible por navegador.

Trabajo Fin de Máster · Máster en Big Data, Data Science e Inteligencia Artificial · Universidad
Complutense de Madrid · Curso 2025-2026.

---

## Índice

- [Qué hace y qué no hace](#qué-hace-y-qué-no-hace)
- [Resultados](#resultados)
- [Estado del proyecto](#estado-del-proyecto)
- [Empezar](#empezar)
- [El conjunto de datos](#el-conjunto-de-datos)
- [Fuentes de datos](#fuentes-de-datos)
- [Cómo está organizado el repositorio](#cómo-está-organizado-el-repositorio)
- [Ejecutar el sistema](#ejecutar-el-sistema)
- [Pruebas](#pruebas)
- [Trabajar en este repositorio](#trabajar-en-este-repositorio)
- [Documentación](#documentación)
- [Equipo](#equipo)

---

## Qué hace y qué no hace

**Hace:** ordena el territorio por peligro de ignición. Responde a la pregunta operativa de un
servicio de prevención —«si hoy puedo vigilar de forma reforzada el 5 % de Galicia, dónde la
coloco»— con una resolución de 1 km², frente a las casillas de 10 a 27,5 km del índice oficial.

**No hace:**

- **No predice el tamaño de un incendio**, solo dónde empieza. El tamaño final depende del viento
  y del tiempo de respuesta de los medios, y nada de eso está en el conjunto de datos.
- **No captura las igniciones que no dependen del tiempo.** El modelo ordena por severidad
  meteorológica, así que falla de forma sistemática en marzo y junio, cuando predominan las quemas
  agrícolas descontroladas.
- **No está validado fuera de Galicia.** Sobre territorio no visto en el entrenamiento, su
  capacidad de discriminar cae de 0,743 a 0,628 de ROC-AUC. Llevarlo a otra comunidad exigiría
  reentrenarlo con su propio historial.

---

## Resultados

Medidos sobre **2023**, el año que se reservó desde el principio y se abrió una sola vez con los
modelos ya serializados, sin reentrenarlos. Población completa: 10.804.365 celdas-día y 530
igniciones.

| Vigilando el 5 % del territorio | Igniciones detectadas (de 530) | Recall |
|---|---|---|
| **Este sistema** | **234** | **44,15 %** |
| FWI oficial del CEMS | 104 | 19,62 % |

Estrechando la vigilancia al 1 % de celdas más críticas de cada día, que es el escenario realista
cuando hay que repartir un número fijo de patrullas, el sistema alcanza el 9,62 % de las
igniciones frente al 2,83 % del índice oficial: **multiplica por 3,4 el estándar publicado en la
métrica más exigente del estudio.**

La progresión del ROC-AUC mide cuánto del acierto es mera estacionalidad y cuánto es discriminar
celdas dentro de un mismo día, que es la aportación real (validación 2022):

| Sistema | ROC global | ROC en temporada | ROC dentro del día |
|---|---|---|---|
| Este sistema | 0,866 | 0,805 | **0,743** |
| FWI Van Wagner a 1 km | 0,810 | 0,736 | 0,615 |
| FWI oficial del CEMS | 0,807 | 0,739 | 0,602 |

### Dos salvedades que conviene leer antes de citar las cifras

**Estas cifras se obtuvieron con meteorología de reanálisis, es decir, con una entrada
meteorológica perfecta.** Miden la arquitectura del sistema, no el servicio. En producción el
sistema se alimenta de previsión numérica, que incorpora su propio error, y esa degradación
**todavía no está cuantificada**. La validación de extremo a extremo —emparejar las predicciones
emitidas cada día con las igniciones oficiales posteriores— está en curso.

**Un 44 % de recall significa que 56 de cada 100 igniciones no se detectan.** El resultado es
bueno en términos comparativos, no absolutos.

---

## Estado del proyecto

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Rejilla de 1 km², Copernicus DEM y CORINE Land Cover | Completada |
| 2 | Ingesta de EGIF y ERA5-Land, construcción del target | Completada |
| 3 | Ingeniería de características y datacubo de 86,5 M de filas | Completada |
| 4 | Entrenamiento, calibración y evaluación sobre el test ciego | Completada |
| 5 | Panel Streamlit y pipeline de inferencia diaria | Desplegada |

Pendiente: cuantificar la degradación al pasar de meteorología observada a previsión numérica, y
cerrar la validación de extremo a extremo del sistema desplegado.

---

## Empezar

Hay tres caminos según lo que quieras hacer. Todos parten del entorno Conda.

```bash
git clone https://github.com/IAMJUNKI/Sistema-deteccion-incendios.git
cd Sistema-deteccion-incendios
conda env create -f environment.yml
conda activate incendios-forestales
pip install -e .
```

### A · Solo ver el panel (lo más rápido)

No hace falta descargar datos históricos ni reentrenar nada. Los artefactos operativos se
distribuyen desde el repositorio público
[`Junkii/galicia_wildfire_risk`](https://huggingface.co/Junkii/galicia_wildfire_risk) en Hugging
Face, y al ser público no necesitas `HF_TOKEN`.

```bash
python scripts/download_artifacts.py \
  --repo-id Junkii/galicia_wildfire_risk \
  --only-dashboard

python -m streamlit run app.py
```

Esto descarga los modelos, la rejilla, las predicciones y los metadatos del panel. Si además
quieres ejecutar inferencias locales, quita `--only-dashboard` para traer el estado meteorológico
acumulado de treinta días (unos 145 MB más).

Como alternativa, se puede activar la sincronización automática una vez por proceso de Streamlit
añadiendo a `.env`:

```dotenv
HF_REPO_ID=Junkii/galicia_wildfire_risk
HF_AUTO_DOWNLOAD=true
HF_DOWNLOAD_MODE=dashboard   # o "full" para incluir el estado meteorológico
HF_TARGET_DIR=.
```

El panel lee después los ficheros locales: Hugging Face no se consulta en cada visita.

> Si `streamlit run app.py` falla con `ModuleNotFoundError: No module named 'folium'`, es que se
> está ejecutando desde el entorno base. Usa `python -m streamlit run app.py` con el entorno
> `incendios-forestales` activado.

### B · Reproducir el entrenamiento y la evaluación

Necesita el datacubo tabular en `data/processed/tabular/`. El pipeline completo que produce todas
las tablas y figuras de la memoria es un único punto de entrada:

```bash
python scripts/pipeline_definitivo.py
```

Escribe sus resultados en `docs/technical/` (una tabla por etapa) y el modelo serializado en
`data/models/`. Cada cifra publicada en la memoria sale de uno de esos ficheros.

### C · Reconstruir el datacubo desde las fuentes originales

Es el camino largo: descarga ERA5-Land, recorta CORINE, cruza el DEM y construye el target desde
el XML de EGIF. Requiere claves de API y varias horas. Está documentado paso a paso en
[`docs/ejecutar_pipeline.md`](docs/ejecutar_pipeline.md).

### Claves de API

Solo hacen falta para los caminos B y C, o para ejecutar inferencia diaria real:

```bash
cp .env.example .env
```

| Variable | Para qué | Dónde obtenerla |
|---|---|---|
| `CDSAPI_KEY` | Descargar ERA5-Land | [Copernicus CDS](https://cds.climate.copernicus.eu/user/register) |
| `METEOGALICIA_API_KEY` | Previsión WRF y observaciones (fuente preferente) | [MeteoSIX](https://www.meteogalicia.gal/web/proxectos/meteosix) |
| `AEMET_API_KEY` | Climatología validada y contingencia | [AEMET OpenData](https://opendata.aemet.es/dist/index.html) |

---

## El conjunto de datos

Una rejilla regular divide Galicia en **29.601 celdas de 1 km²** en `EPSG:3035`, un sistema de
igual área para que ninguna celda quede infrarrepresentada. Cruzando esas celdas con los 2.922
días del periodo 2016-2023 se obtiene un conjunto de **86.494.122 filas**, de las que 12.699
corresponden a una ignición: aproximadamente **un caso positivo por cada 6.800 negativos**.

Cada fila describe una celda en un día concreto con 50 variables candidatas, de las que el modelo
final usa 48.

### Contrato temporal

Es la parte del diseño que más conviene entender antes de tocar nada.

- **El target es `target_ignicion`**, tomado del registro oficial de EGIF. Un cero significa
  ausencia de ignición registrada, no ausencia de peligro.
- **El datacubo histórico no aplica desfase:** las variables meteorológicas y sus acumulados
  describen el mismo día `T` que se evalúa. Es un **nowcast**, de la misma naturaleza que el FWI
  con el que se compara, no una previsión a 24 horas.
- **El contexto meteorológico arranca un mes antes** del primer año pedido, para poder calcular
  acumulados de hasta 30 días sin que el 1 de enero se quede sin historia detrás.
- **Existe una segunda exportación, alineada para operación**, en
  `data/processed/tabular/egif_operational/`. Recalcula las memorias meteorológicas cerrándolas en
  la víspera, de modo que no contienen ningún dato posterior al momento en que se emitiría el
  aviso. Es la que alimenta a los modelos desplegados, y **no sustituye al datacubo histórico**:
  las dos se mantienen separadas para no atribuir al sistema en producción un resultado medido
  sobre la otra. Ver
  [alineación y modelos operativos](docs/explanations/implementacion_alineacion_operativa.md).

### Partición temporal

Un reparto aleatorio de filas colocaría días consecutivos de la misma ola de calor a ambos lados
de la partición, y el modelo obtendría métricas excelentes sin haber aprendido nada transferible.
El reparto es estrictamente temporal:

| Años | Papel | Igniciones |
|---|---|---|
| 2016-2020 | Entrenamiento | 9.584 |
| 2021 | Parada temprana y ajuste del calibrador | 926 |
| 2022 | Validación: aquí se tomaron todas las decisiones | 1.659 |
| 2023 | **Test ciego**: se abrió una sola vez, al cerrar el trabajo | 530 |

> **2023 no debe volver a evaluarse.** Su valor como estimación insesgada depende de haberse
> abierto una sola vez con el sistema congelado. La configuración con la que se abrió está
> registrada en `docs/technical/egif48_test_2023_audit.json`.

### Decisiones de modelado

| Cuestión | Decisión | Por qué |
|---|---|---|
| Desbalance | Todos los positivos y 1 de cada 100 negativos, por hash determinista de celda y fecha | Con esta prevalencia los negativos son masivamente redundantes; el factor 1/100 da un recall indistinguible de 1/25 con la cuarta parte del coste |
| Sesgo del submuestreo | Corrección de prior de King y Zeng | El submuestreo infla las probabilidades en un factor conocido |
| Calibración | Regresión de **Platt**, no isotónica | La isotónica es escalonada y, sobre diez millones de filas, genera millones de empates que desplazan cualquier umbral definido por percentiles |
| Algoritmo | LightGBM | Gana a regresión logística, Random Forest y XGBoost con prueba de McNemar |
| Variables | 48 de 50; fuera la lluvia del día y los días consecutivos sin lluvia | Ambas contienen la precipitación del propio día `T`, a menudo posterior a la ignición. Excluirlas no tiene coste medible y elimina la objeción |
| Evaluación | Población completa del año, nunca submuestreada | Un guardián de cobertura aborta la ejecución si no se recorren exactamente las filas que el contrato declara |

**Suelo de ruido.** Repetir cinco veces el mismo entrenamiento cambiando solo la semilla produce
una oscilación de **1,93 puntos de recall**. Ninguna diferencia inferior se presenta como un
hallazgo en este proyecto.

---

## Fuentes de datos

| Fuente | Uso |
|---|---|
| [EGIF-MITECO](https://www.miteco.gob.es/es/biodiversidad/temas/incendios-forestales/estadisticas-incendios.html) | **Variable objetivo.** Registro oficial, con parte rellenado sobre el terreno por los agentes forestales |
| [ERA5-Land (Copernicus CDS)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Meteorología horaria histórica, descendida de escala a 1 km |
| [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) | Elevación, pendiente, rugosidad y orientación |
| [CORINE Land Cover 2018](https://land.copernicus.eu/en/products/corine-land-cover) | Fracción de cada tipo de cobertura por celda |
| [OpenStreetMap](https://download.geofabrik.de/europe/spain.html) | Red viaria y proximidad a núcleos: actividad humana |
| [CNIG/IGN](https://centrodedescargas.cnig.es/CentroDescargas/index.jsp) | Límites administrativos |
| [MeteoGalicia MeteoSIX v5](https://www.meteogalicia.gal/web/proxectos/meteosix) | Previsión WRF a 1 km y red de estaciones automáticas, para operación |
| [AEMET OpenData](https://opendata.aemet.es/dist/index.html) | Climatología diaria validada y contingencia |
| [CEMS / EFFIS](https://effis.jrc.ec.europa.eu/) | Producto oficial de FWI, usado como baseline de comparación |

**Por qué EGIF y no los focos de calor de satélite.** El proyecto empezó con NASA FIRMS y lo
descartó por tres motivos: un satélite pasa dos veces al día, así que un incendio que empieza por
la tarde puede quedar fechado al día siguiente y el modelo aprendería a relacionarlo con el tiempo
de otro día; FIRMS no da las hectáreas quemadas; y desde el espacio no siempre se ven los fuegos
pequeños, que son la mayoría. El trabajo con FIRMS se conserva en
[`archive/firms_mikel/`](archive/) por trazabilidad y no está en la ruta operativa.

**Por qué CORINE de 2018 y no una edición posterior.** CORINE incluye una categoría de «zonas
quemadas». Una edición posterior a los incendios permitiría al modelo reconocer como quemadas
precisamente las celdas que ardieron, un conocimiento del que jamás dispondría en un uso real. Con
la edición de 2018, anterior a los años evaluados, esa contaminación queda descartada por
construcción.

---

## Cómo está organizado el repositorio

```text
src/
  geospatial/      rejilla de 1 km, DEM y CORINE
  ingestion/       ERA5-Land, EGIF, FWI del CEMS, MeteoGalicia y AEMET
  features/        calendario, acumulados y exportación tabular
  entrenamiento/   contrato del dataset, muestreo, calibración y métricas
  modeling/        carga, selección y evaluación temporal de modelos
  models/          modelos serializados y familias por horizonte
  operational/     inferencia diaria y manifiesto de procedencia
  baselines/       FWI de Van Wagner calculado sobre nuestra propia rejilla
  webapp/          panel Streamlit: mapas, KPIs, SHAP y analítica territorial
  workflow.py      orquestación de la construcción del datacubo

scripts/           puntos de entrada ejecutables (ver la sección siguiente)
configs/           configuración de experimentos e hiperparámetros
notebooks/         exploración y validación; el 19 unifica la fase de descubrimiento
tests/             281 pruebas
docs/              documentación técnica, de despliegue y resultados publicados
  technical/       una tabla por etapa del pipeline: es la fuente de las cifras
deploy/            unidades systemd del servicio desplegado
knowledge/         alcance y plan del proyecto
archive/           trabajo histórico fuera de la ruta operativa
data/              NO versionado (ver .gitignore)
  raw/             fuentes originales
  processed/       datacubo y exportaciones tabulares
  models/          modelos entrenados
```

Los datos, los modelos y los artefactos pesados **nunca se suben al repositorio**. Se distribuyen
por Hugging Face (ver [Empezar](#empezar)) o por el procedimiento de
[despliegue de código y datos](docs/deployment/despliegue_codigo_y_datos.md).

---

## Ejecutar el sistema

### Investigación y evaluación

```bash
# Pipeline completo de la memoria: todas las tablas y figuras
python scripts/pipeline_definitivo.py

# Entrenar y evaluar variantes sobre el datacubo EGIF
python scripts/entrenar_egif.py --modelos lightgbm

# Selección de variables y búsqueda de hiperparámetros
python scripts/seleccionar_variables.py
python scripts/buscar_hiperparametros.py

# Auditar el test ciego de 2023 sobre los artefactos congelados
python scripts/audit_egif48_test.py
```

### Operación diaria

En el servidor lo lanza un temporizador de systemd a las 05:15. Manualmente:

```bash
# 1. Cerrar la memoria ambiental de los 30 días anteriores en la víspera
python scripts/update_weather_state.py

# 2. Ingerir la previsión de las 72 horas siguientes y publicar los cuatro mapas
python scripts/run_daily_inference.py

# 3. Comprobar que la ejecución fue completa y coherente
python scripts/check_operational_run.py
```

La previsión se degrada de forma explícita y nunca silenciosa: intenta WRF a 1 km, cae a la malla
de 4 km si falta, y solo si ambas fallan reutiliza el último pronóstico archivado. Cada resultado
se etiqueta como reciente, reciente degradado o caducado, y el manifiesto conserva la malla
empleada, la ejecución de origen y las firmas criptográficas del modelo y de la salida.

### Publicar artefactos

```bash
python scripts/publish_to_huggingface.py
```

Ver [distribución de artefactos](docs/tasks/10_distribucion_artefactos_huggingface.md) y la
[ficha del modelo](docs/huggingface_model_card.md).

---

## Pruebas

```bash
conda activate incendios-forestales
python -m pytest tests -q
```

Son **281 pruebas** y deben pasar todas. El conjunto no necesita el datacubo real: `tests/conftest.py`
escribe un datacubo EGIF en miniatura con la misma estructura que el de verdad, de modo que la
carga, el muestreo y los guardianes de cobertura se prueban sin depender de los 16 GB de datos.

Las pruebas del pipeline geoespacial importan `geopandas`, `rasterio` y `cdsapi`, que solo están en
el entorno `incendios-forestales`. Si no están instalados, `conftest.py` las omite en vez de dejar
que un `ModuleNotFoundError` durante la recolección impida ejecutar el resto.

Para validar los productos generados sin descargar ni modificar datos están los notebooks
`07_validacion_datacubo`, `08_validacion_dataset_parquet` y `18_validacion_fwi_cems`. El notebook
`19_descubrimiento_unificado` reúne la fase de descubrimiento completa y carga sus cifras desde los
CSV de `docs/technical/`, no las tiene escritas a mano.

---

## Trabajar en este repositorio

### Ramas

El trabajo se integra **directamente en `main`**, que es lo que el equipo hace en la práctica: de
133 commits solo 6 son merges, y la rama `Develop` del remoto está sin uso. Para cambios que
puedan romper algo, lo habitual ha sido una rama local corta y un merge cuando pasan las pruebas.

Antes de empezar a trabajar, trae lo que haya:

```bash
git pull --ff-only
```

Con seis personas tocando el repositorio a diario, esto evita la mayoría de los conflictos.

### Commits

Se usan [Conventional Commits](https://www.conventionalcommits.org/), con el módulo entre
paréntesis:

| Prefijo | Uso |
|---|---|
| `feat(módulo):` | Nueva funcionalidad |
| `fix(módulo):` | Corrección de un fallo |
| `data(módulo):` | Cambios en scripts de datos |
| `test:` | Pruebas |
| `docs:` | Documentación |
| `refactor(módulo):` | Refactor sin cambio de comportamiento |
| `chore:` | Mantenimiento: dependencias, configuración |

Módulos: `geospatial` · `ingestion` · `features` · `entrenamiento` · `modeling` · `models` ·
`operational` · `webapp`.

### Dos reglas que no se negocian

1. **Los datos no entran en Git.** Ni datasets, ni modelos, ni estados meteorológicos. Están en
   `.gitignore` y así deben quedarse.
2. **2023 no se reevalúa.** Es el test ciego y ya se abrió. Reabrirlo con un pipeline distinto
   invalida la única estimación insesgada del trabajo.

### Reproducibilidad

Las cifras publicadas en `docs/technical/` dependen de la versión del código con la que se
generaron. Si cambias el pipeline de entrenamiento, **regenera todas las etapas**, no solo la que
te interesa: una tabla actualizada junto a otras antiguas es peor que ninguna. El ejecutable fija
`deterministic: true` y `force_row_wise: true` en LightGBM para que dos ejecuciones idénticas
coincidan.

---

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/variables.md`](docs/variables.md) | Las 50 variables, una por una, con su procedencia |
| [`docs/ejecutar_pipeline.md`](docs/ejecutar_pipeline.md) | Reconstruir el datacubo desde las fuentes originales |
| [`docs/modeling.md`](docs/modeling.md) | Selección de variables y evaluación temporal |
| [`docs/explanations/implementacion_alineacion_operativa.md`](docs/explanations/implementacion_alineacion_operativa.md) | Por qué hay dos exportaciones y en qué se diferencian |
| [`docs/deployment/servidor_produccion.md`](docs/deployment/servidor_produccion.md) | Servidor, systemd, Nginx, TLS, copias y rollback |
| [`docs/deployment/despliegue_codigo_y_datos.md`](docs/deployment/despliegue_codigo_y_datos.md) | Releases y transferencia de datos pesados |
| [`docs/deployment/automatizacion_meteorologica.md`](docs/deployment/automatizacion_meteorologica.md) | Cierre de la brecha observacional de D-4 a D-1 |
| [`docs/huggingface_model_card.md`](docs/huggingface_model_card.md) | Ficha del modelo operativo |
| [`docs/technical/`](docs/technical/) | Resultados publicados: una tabla por etapa |
| [`docs/propuesta_reorganizacion.md`](docs/propuesta_reorganizacion.md) | Limpieza pendiente del repositorio, con su justificación |
| [`archive/vecindad/README.md`](archive/vecindad/) | Variables de contexto espacial: medidas, no adoptadas, y por qué |

---

## Equipo

Trabajo Fin de Máster — Máster en Big Data, Data Science e Inteligencia Artificial, Universidad
Complutense de Madrid.

Miquel Jiménez · Enrique Bravo · Alfonso García · Raúl Utrilla · Santi · Diego Junquera
