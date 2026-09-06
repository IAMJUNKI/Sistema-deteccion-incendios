# Propuesta de reorganización del repositorio

**Estado: propuesta. Nada de lo que sigue está aplicado.**

Documento preparado tras un análisis de dependencias del repositorio completo. Recoge qué
convendría mover al archivo, qué conviene dejar donde está y —lo más importante— **qué parecía
archivable y no lo es**, porque ahí es donde se rompen las cosas.

La propuesta parte de una idea que ya está en el repositorio: la carpeta `archive/firms_mikel/`
creada en agosto para conservar el trabajo sobre la fuente de target anterior. Se trata de
extender ese criterio al resto del código de exploración, manteniendo la misma estructura
interna para que nada se pierda y todo siga siendo localizable.

---

## 1 · Situación actual

| | |
|---|---|
| Módulos Python bajo `src/` | 78 |
| Alcanzables desde los puntos de entrada de producción | **26** |
| No alcanzables | 52 |
| Ficheros duplicados byte a byte | **8** |

Un tercio del código es la ruta operativa; dos tercios son herramientas de análisis, la
aplicación web y restos de exploración. Eso no es un problema en sí —un TFM necesita conservar
su trabajo experimental— pero conviene que la separación sea explícita.

Los puntos de entrada considerados producción son `scripts/run_daily_inference.py`,
`scripts/train_egif_operational.py`, `scripts/pipeline_definitivo.py`, `src/workflow.py`,
`src/pipeline.py` y `app.py`.

---

## 2 · Lo que sí conviene mover

Los ocho ficheros de esta sección se verificaron con cuatro comprobaciones independientes:
importación real de todos los módulos de producción, análisis de imports con AST sobre los nueve
puntos de entrada, búsqueda textual en todo el árbol vivo —incluido el código dentro de los
notebooks— y ejecución de la suite de pruebas antes y después.

### 2.1 · Duplicados exactos en `src/models/`

Estos cuatro ficheros tienen una **copia idéntica byte a byte** en
`archive/firms_mikel/src/models/`. Se archivaron en agosto y reaparecieron después en el árbol
vivo, probablemente en una fusión. Nadie los importa.

| Fichero | Líneas |
|---|---|
| `src/models/train_ensemble_advanced.py` | 153 |
| `src/models/entrena_2d_comparativa.py` | 202 |
| `src/models/evalua_2d_robusto.py` | 202 |
| `src/models/hyperparameter_tuning.py` | 81 |

**Acción:** retirarlos del árbol vivo con `git rm`. No hace falta moverlos a ninguna parte: la
copia archivada ya existe y crear una tercera solo empeoraría la confusión.

**Por qué:** tener el mismo fichero en dos rutas obliga a quien lo lea a averiguar cuál es el
bueno, y hace que una corrección aplicada en uno pase desapercibida en el otro.

### 2.2 · Restos de exploración sin ninguna referencia

| Fichero | Líneas | Qué es |
|---|---|---|
| `src/modeling/audit_xgboost.py` | 60 | Auditoría de XGBoost para revisar fuera de Jupyter |
| `src/modeling/run_ablation.py` | 45 | Réplica de la ablación en proceso aislado |
| `src/modeling/run_corrected_suite.py` | 17 | Ejecuta la suite corregida sin acumular memoria |
| `src/ingestion/forecast_provider.py` | 67 | Selector de proveedor de previsión, sustituido |

**Acción:** mover con `git mv` a `archive/descubrimiento/`, conservando la estructura:

```
archive/
├── README.md                 ← ampliar
├── firms_mikel/              ← existente, no tocar
└── descubrimiento/           ← nuevo
    ├── README.md             ← explicar qué hay y por qué
    └── src/
        ├── ingestion/forecast_provider.py
        └── modeling/audit_xgboost.py
                    run_ablation.py
                    run_corrected_suite.py
```

**Por qué `git mv` y no copiar y borrar:** conserva el historial del fichero, de modo que
`git log --follow` sigue funcionando y no se pierde quién escribió qué ni cuándo.

**Advertencia:** `audit_xgboost.py` y `run_ablation.py` importan módulos de `src/modeling/` que
se quedan en su sitio. Tras moverlos dejarán de ejecutarse. Es aceptable para código archivado
—se conserva como registro, no para correrlo— pero conviene anotarlo en el README del archivo.

---

## 3 · Lo que parece archivable y NO lo es

Esta es la sección importante. Los ocho ficheros siguientes aparecen como no alcanzables desde
producción, y aun así **moverlos rompería cosas**.

### 3.1 · `src/modeling/` — lo usan cuatro notebooks y una prueba

| Fichero | Lo referencian |
|---|---|
| `src/modeling/data.py` | notebooks 10, 11, 12 y 14 · `tests/test_modeling.py` · `configs/egif_v1.yaml` |
| `src/modeling/features.py` | notebooks 10, 11, 12 y 14 · `tests/test_modeling.py` · `docs/modeling.md` |
| `src/modeling/experiments.py` | notebooks 10, 11 y 12 |
| `src/modeling/evaluation.py` | notebooks 11 y 14 · `tests/test_modeling.py` |

El rastreo automático de dependencias no los ve porque **el código dentro de los notebooks no
aparece en un análisis de imports de ficheros `.py`**. Hay que extraer las celdas del JSON para
encontrarlo. Es la trampa más fácil de pisar al reorganizar un repositorio con notebooks.

### 3.2 · `src/models/` — dependencias cruzadas y la aplicación web

| Fichero | Lo referencian |
|---|---|
| `src/models/explainability.py` | **`src/webapp/components/shap_simulator.py`** |
| `src/models/metrics.py` | `hyperparameter_tuning`, `train_baseline`, `train_ensemble_advanced`, `train_nn_3d` |
| `src/models/train_baseline.py` | `train_ensemble_advanced` · `docs/explanations/` |
| `src/models/train_nn_3d.py` | `docs/explanations/justificacion_metodologica_dataset_y_modelado.md` |

`explainability.py` es el caso más claro: alimenta el simulador SHAP del panel web, que forma
parte del entregable.

Los otros tres se referencian entre sí y desde la documentación metodológica. Archivarlos
dejaría la memoria citando ficheros que ya no están donde dice.

### 3.3 · Excluidos por decisión expresa

`src/pipeline.py` (233 líneas) y `src/workflow.py` (384) aparecen como puntos de entrada que
nadie importa, pero `docs/ejecutar_pipeline.md` documenta ejecutarlos a mano con
`python -m src.workflow`. No se tocan sin confirmarlo con quien los mantiene.

`src/baselines/fwi_van_wagner.py` (313 líneas) tampoco: conviene comprobar antes su relación
con el baseline FWI del pipeline definitivo.

---

## 4 · Un hallazgo aparte: la suite de pruebas no se ejecutaba entera

Al verificar el movimiento salió a la luz un problema independiente y más urgente que la
reorganización.

`tests/conftest.py` lleva una lista de módulos de prueba que se omiten cuando falta el entorno
geoespacial —importan `geopandas`, `rasterio` o `cdsapi`, que solo están en el entorno
`incendios-forestales`—. Esa lista **no incluye tres ficheros de prueba añadidos después**:

```
tests/test_datacube_profile.py     ModuleNotFoundError: geopandas
tests/test_webapp_components.py    ModuleNotFoundError: geopandas
tests/test_fwi.py                  ModuleNotFoundError: cdsapi
```

Un fallo al importar durante la recolección **aborta la suite completa**, no solo ese fichero.
El efecto medido:

| | Pruebas ejecutadas |
|---|---|
| Con la lista actual | **0** — la recolección se interrumpe |
| Añadiendo los tres nombres | **234 pasan** |

Son 234 pruebas que ahora mismo no se están ejecutando en un entorno sin el stack geoespacial.
La corrección es añadir tres cadenas a `_TESTS_GEOESPACIALES` en `tests/conftest.py`.

**Esto merece arreglarse con independencia de que la reorganización se haga o no**, y es el
cambio de mayor valor y menor riesgo de todo este documento.

---

## 5 · Otros detalles menores

**Dos ficheros con el mismo nombre en `knowledge/`.** Aparecen dos entradas de
`Project Plan - Predicción incendios forestales.md`, de 15 KB cada una. Probablemente difieren
en la codificación del acento o en un carácter invisible. Conviene mirarlo y dejar uno.

**Los documentos HTML no entran en el repositorio.** La línea `*.html` del `.gitignore` es
anterior a este trabajo y excluye `docs/modelado_2d_egif.html` y `docs/resumen_modelado.html`.
Si se quiere que los entregables vivan en el repositorio hace falta una excepción explícita.

---

## 6 · Orden recomendado

1. **Arreglar `conftest.py`** — tres líneas, ningún riesgo, recupera 234 pruebas. Independiente
   de todo lo demás.
2. **Retirar los cuatro duplicados exactos** — la copia archivada ya existe.
3. **Mover los cuatro restos de exploración** a `archive/descubrimiento/` con `git mv`, y
   escribir su README.
4. **Consultar** con el equipo lo de `src/pipeline.py`, `src/workflow.py` y
   `fwi_van_wagner.py` antes de tocarlos.

Cada paso en su propio commit, y ejecutando `python -m pytest tests -q` después de cada uno.

---

## 7 · Cómo se verificó

Para que la propuesta sea auditable, estas son las comprobaciones que sostienen las
afirmaciones anteriores:

1. **Duplicados** — comparación por hash MD5 de todos los `.py` del repositorio.
2. **Alcance de producción** — recorrido del grafo de imports con AST desde los puntos de
   entrada, resolviendo `src.*` a rutas de fichero.
3. **Referencias reales** — búsqueda de patrones de import sobre `.py`, `.md`, `.yaml`, `.toml`,
   `.sh` **y el código extraído de los `.ipynb`**.
4. **Prueba de humo** — importación efectiva de los 22 módulos de producción y comparación de la
   suite de pruebas antes y después del movimiento.

El paso 3 es el que evitó archivar `src/modeling/`, y el 4 el que distinguió los fallos de
entorno de los provocados por el movimiento.
