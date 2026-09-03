# Contexto para una IA colaboradora

> Estado del proyecto: 2026-09-02. Este archivo resume las decisiones vigentes
> del TFM. Debe leerse antes de proponer cambios de datos, modelado o pipeline.

## 1. Objetivo y alcance

Este repositorio construye un datacubo histórico diario de **riesgo de ignición
forestal en Galicia**. La unidad de observación es una celda de 1 km × 1 km y
un día. A partir del cubo se genera un dataset tabular Parquet para modelos de
Machine Learning y, más adelante, mapas diarios de riesgo.

El objetivo actual es predecir **igniciones**, no área quemada ni focos FIRMS.
La fuente de referencia del target es el histórico oficial **EGIF-MITECO**.
No sustituir EGIF por FIRMS sin una decisión explícita del usuario.

## 2. Estado actual verificable

- Rama remota publicada: `main` en `origin`; `origin/dev` fue eliminada a
  petición del usuario. La rama local `dev` se utiliza para el trabajo en curso.
- Pruebas de contrato de datos y meteorología: `36 passed`.
- El contrato canónico actual produce 50 predictores y 55 variables de datos;
  los productos pesados se regeneran localmente cuando se necesitan.
- Los datos pesados están ignorados por Git: **no asumir que estén presentes en
  otro ordenador ni intentar subirlos al repositorio**.

Productos actuales locales:

```text
data/processed/datacube/galicia_1km.nc
data/processed/tabular/egif/metadata.json
data/processed/tabular/egif/year=2019/dataset_2019.parquet
...
data/processed/tabular/egif/year=2023/dataset_2023.parquet
```

El producto local histórico 2019–2023 debe considerarse una instantánea de
trabajo, no el contrato temporal del repositorio. El workflow permite construir
cualquier intervalo de años completos disponible en EGIF. Cada ejecución deja
su cobertura declarada y su estado (`running` o `completed`) en
`data/processed/datacube/run_manifest.json`.

## 3. Contrato espacial y temporal

| Concepto | Decisión vigente |
|---|---|
| Zona | Galicia |
| Rejilla | Rectangular, celdas de 1 km × 1 km |
| CRS de trabajo | EPSG:3035 |
| Celdas modelables | `is_galicia = 1`: el centro de la celda está dentro del límite de Galicia |
| Periodo del cubo | Años completos elegidos por el usuario; por defecto 2016-01-01 a 2023-12-31 |
| Contexto meteorológico | Desde el 1 de diciembre anterior al año inicial, para acumulados de hasta 30 días |
| Target | `target_ignicion`: 1 si hay una o más igniciones EGIF en celda/día; 0 si no |
| Formato espacial | NetCDF 3D `(time, y, x)`; el Parquet contiene solo celdas activas de Galicia |

El NetCDF conserva la rejilla rectangular completa para mapas y futuros modelos
espacio-temporales. El Parquet no es una rejilla: contiene una fila por fecha y
celda activa de Galicia.

### Decisión temporal importante

El producto histórico actual es un **nowcast/análisis diario**: meteorología,
acumulados y medias móviles describen el propio día `T` e incluyen `T`.
Esta decisión fue explícita del usuario. No introducir un desplazamiento a
`T-1` sin consultarlo.

IberFire entrena con variables de `T-1` para predecir el fuego de `T`. Si se
prepara inferencia operativa con predicciones de MeteoGalicia, habrá que definir
un contrato equivalente y revalidar el modelo; no basta con reutilizar ERA5.

## 4. Fuentes y transformación

| Bloque | Fuente | Procesamiento principal |
|---|---|---|
| Límite | CNIG/IGN | Máscara `is_galicia` sobre la rejilla |
| Topografía | Copernicus DEM GLO-30 | Estadísticos por celda: elevación, pendiente, rugosidad y orientación en ocho franjas |
| Cobertura | CORINE Land Cover 2018 | Fracciones agregadas de combustibles/usos de suelo por celda |
| Actividad humana | OpenStreetMap / Geofabrik, instantánea Galicia 2022-01-01 | Distancia a carreteras y zonas residenciales; longitud de carreteras |
| Meteorología histórica | ERA5-Land horario | Estadísticos diarios, ventana 12–18 h Europe/Madrid, acumulados y días secos |
| Incendios | XML EGIF-MITECO | Asignación de igniciones a celda/día; superficie y fuego grande como resultados auxiliares |

CORINE se resume deliberadamente en variables densas; no recrear sin motivo las
44 clases brutas de IberFire. Las celdas costeras sin píxel CORINE válido se
completan con el vector de la celda válida más próxima. En meteorología, las
celdas de borde sin interpolación lineal válida se completan con el píxel
terrestre ERA5-Land más cercano.

El extracto OSM se descarga automáticamente solo si falta y se conserva en
`data/raw/human_activity/`. Es una capa estática; no se redescarga ni se
recalcula salvo petición expresa con `--rebuild-human-activity`.

### Perfil configurable de variables (trabajo en `dev`)

`src/datacube_profile.py` contiene el único perfil de variables que usa
`src.workflow`: topografía, CORINE, calendario, meteorología y actividad humana.
No existe una ruta alternativa de prueba o final. No confundir este contrato de
almacenamiento con la selección posterior de predictores del modelo.

El perfil incluye VPD (`vpd_mean`, `vpd_max_12_18h`) y actividad humana
desagregada: longitudes viarias principal/local/pistas/otras y fracciones
residencial/de edificios. La longitud total es la suma de las categorías.

## 5. Variables disponibles

La lista explicada y actualizada está en [`docs/variables.md`](../docs/variables.md).
Resumen:

- Identificación: `cell_id`, `x`, `y`, `fecha` se derivan para exportación y
  trazabilidad. El calendario se deriva de la coordenada `time`; no se almacena
  como señal del cubo final.
- Topografía: `elevation_mean/std`, `slope_mean/std` y ocho fracciones de
  orientación. Se retiró la rugosidad por redundancia.
- Cobertura/combustible: `artificial`, `agriculture`, bosques de frondosas,
  coníferas y mixto, `scrub`, `open_spaces`, `wetlands`, `water` y
  sin `forest_cover_fraction`, que era una suma exacta de las tres fracciones
  forestales.
- Meteorología: temperatura, humedad relativa, viento y precipitación diaria;
  extremos 12–18 h; acumulados de precipitación de 3, 7, 14 y 30 días; medias
  de temperatura, viento y humedad a 7 días, humedad a 14 días; días secos
  consecutivos; VPD diario y máximo de 12–18 h.
- Actividad humana estática: `road_length_km`, sus categorías principal/local/
  pistas/otras, `residential_area_fraction` y `building_area_fraction`. Se
  retiraron las distancias por su saturación a cero a escala de 1 km.

Cada variable del NetCDF incluye `long_name`, `units` y, cuando aplique,
`description` o contrato temporal. La coordenada técnica `number` de ERA5
(identificador de miembro de ensemble) se elimina del cubo y del Parquet: no
tiene significado espacial, temporal ni predictivo.

No usar como predictores: `fecha`, `cell_id`, `x`, `y`, `year`, `is_galicia`,
`target_ignicion`, `burned_area_ha`, `large_fire_500ha` ni las columnas
auxiliares descritas a continuación. El contrato de predictores se publica en
`metadata.json` y se valida desde `src/modeling/features.py`.

## 6. Máscara auxiliar estilo IberFire: `is_near_ignition_25x25_10d`

Esta capa se implementó a partir de la idea `is_near_fire` de IberFire, adaptada
al target de ignición EGIF:

- Cada ignición marca un cuadrado centrado de **25 × 25 celdas** (radio de 12
  celdas).
- Marca el día del evento y los diez días anteriores: `t-10, …, t` (11 fechas).
- Si la ventana toca el borde de la rejilla o de Galicia, se recorta; las celdas
  fuera de Galicia quedan siempre a 0.
- La propia celda-día de ignición tiene valor 1: está dentro de su propio
  vecindario. Esto no afecta a la selección de negativos porque ya es positiva.

**Uso único permitido:** filtro histórico de negativos. Por ejemplo:

```python
negativos_limpios = datos[
    (datos["target_ignicion"] == 0)
    & (datos["is_near_ignition_25x25_10d"] == 0)
]
```

Nunca debe usarse como predictor ni como entrada de producción: se construye
con incendios observados, incluso futuros respecto al día que se está marcando.
El código la excluye de `predictor_columns` y la registra como auxiliar en los
metadatos del Parquet.

La construcción marca directamente las ventanas de las igniciones observadas,
en lugar de filtrar toda la matriz temporal; es equivalente al contrato anterior
pero evita que la regeneración completa del cubo sea innecesariamente lenta.

## 7. Pipeline y puntos de entrada

Arquitectura:

```text
datos raw
  ├─ src/geospatial/       rejilla, topografía, CORINE
  ├─ src/ingestion/        ERA5 y EGIF
  ├─ src/features/time.py  calendario
  ├─ src/pipeline.py       ensamblado NetCDF + máscara auxiliar
  └─ src/features/tabular.py  exportación Parquet anual + metadata.json
```

`src/workflow.py` orquesta el proceso histórico completo. Las rutas y el
periodo se centralizan en `src/config.py`. Crea o reutiliza también la capa OSM
de actividad humana antes del ensamblado.

Para instalar y ejecutar desde cero, la referencia es
[`docs/ejecutar_pipeline.md`](../docs/ejecutar_pipeline.md). Puntos esenciales:

1. Crear el entorno Conda desde `environment.yml`.
2. Crear `.env` desde `.env.example` y añadir la credencial de Copernicus
   únicamente en local.
3. Colocar CORINE 2018 y el XML EGIF en `data/raw/`.
4. Descargar ERA5 solo si faltan sus ficheros mensuales.
5. Ejecutar `python -m src.workflow --start-year AAAA --end-year AAAA`.

No descargar de nuevo un dato que ya cubra el periodo solicitado. Los scripts
de ERA5 omiten los meses existentes y FWI verifica su cobertura diaria; las
capas estáticas no deben reconstruirse salvo que se cambie la rejilla, el DEM,
CORINE o el límite.

## 8. Modelado: estado, reglas y cautelas

El modelado está separado del pipeline de datos en `src/modeling/` y
`configs/modeling.toml`.

División temporal fija:

```text
entrenamiento: 2019–2021
validación para decisiones: 2022
test ciego final: 2023
```

Métricas: PR-AUC como principal, además de ROC-AUC y `recall@1%`. No usar
accuracy como métrica principal por el fuerte desbalanceo.

Conjuntos de variables en `src/modeling/features.py`:

- `completo`: parte de los 50 predictores publicados en `metadata.json`.
- `temporal_compacto`: elimina `roughness_mean`, `roughness_std` y variables
  de calendario redundantes (`day_of_year`, `month`, `iso_week`, `month_sin`,
  `month_cos`). Es el candidato provisional de 44 variables.

### Hallazgo crítico sobre muestreo

Las primeras métricas extraordinariamente altas de XGBoost **no son válidas**:
el muestreo original mantenía negativos de un subconjunto fijo de `cell_id`,
creando fuga espacial. El muestreo actual en `src/modeling/data.py` usa un hash
determinista de `(cell_id, fecha)` y conserva todos los positivos más una
muestra de negativos repartida por espacio y tiempo.

No afirmar que XGBoost sea un “buen modelo” ni reutilizar las métricas antiguas
hasta completar una auditoría limpia, evaluación sobre distribución completa de
negativos y el test ciego de 2023. El filtro `is_near_ignition_25x25_10d` debe
evaluarse como protocolo alternativo, no como sustituto de la evaluación
operacional con negativos representativos.

El muestreo reutilizable de los experimentos usa un hash determinista de
`(cell_id, fecha)`; no usar el antiguo residuo fijo de `cell_id % 25`, porque
introducía una fuga espacial.

## 9. Notebooks útiles

| Notebook | Finalidad |
|---|---|
| `07_validacion_datacubo.ipynb` | Validar contrato, mapa y cobertura del NetCDF |
| `08_validacion_dataset_parquet.ipynb` | Validar esquema, particiones y calidad tabular |
| `09_exploracion_y_seleccion_features.ipynb` | Exploración, correlaciones y candidatos de variables |
| `10_experimentos_feature_selection.ipynb` | Ablaciones temporales con 2022 como validación |
| `11_validacion_robusta_modelo.ipynb` | Robustez frente a muestras de negativos |
| `12_comparacion_modelos.ipynb` | Comparación de familias de modelos |
| `14_auditoria_completa_xgboost.ipynb` | Auditoría de métricas, importancia y gráficos de XGBoost |

Los notebooks son herramientas de exploración y validación; la lógica reusable
debe vivir en `src/`, con tests. No introducir lógica crítica solo en una celda.

## 10. Validación y trabajo seguro

Ejecutar antes de un commit relevante:

```powershell
& C:\Users\alfon\anaconda3\envs\incendios-forestales\python.exe -m pytest tests -q
& C:\Users\alfon\anaconda3\envs\incendios-forestales\python.exe -m ruff check src tests
```

Para cambios de datos, validar además:

- cobertura temporal igual al periodo declarado en `run_manifest.json`;
- `target_ignicion` definido para cada fecha/celda activa;
- Parquet anual consolidado y suma de filas consistente con `metadata.json`;
- columnas auxiliares y outcomes fuera de `predictor_columns`;
- ausencia de NaN en los predictores publicados.

No borrar datos `raw` ni outputs canónicos sin autorización. Los backups locales
creados al regenerar la máscara se llaman `*_before_near_ignition` y son
recuperables; no se incluyen en Git.

## 11. Próximas prioridades recomendadas

1. Corregir y finalizar la auditoría de XGBoost con muestreo celda-día correcto.
2. Comparar de forma transparente negativos representativos frente a negativos
   filtrados por `is_near_ignition_25x25_10d`.
3. Mantener 2023 como test ciego hasta fijar variables, muestreo y parámetros.
4. Añadir un baseline Fire Weather Index (FWI) para cuantificar el valor real
   del ML.
5. Evaluar con validación temporal si merece la pena añadir VPD de la ventana
   12–18 h; debe calcularse hora a hora, no a partir de extremos diarios.
6. Estudiar fuentes operativas de MeteoGalicia y mantener un contrato temporal
   compatible antes de desplegar inferencia diaria.

## 12. Instrucciones para la IA que reciba este archivo

- Habla en español salvo que el usuario solicite otra lengua.
- Prioriza cambios pequeños, trazables, testeados y documentados.
- No inventes ejecuciones, métricas ni disponibilidad de datos.
- Antes de alterar contratos temporales, targets o predictores, explica la
  consecuencia y pide confirmación si cambia el alcance metodológico.
- No uses resultados con el antiguo muestreo de negativos para justificar la
  calidad del modelo.
- No trates `is_near_ignition_25x25_10d` como predictor bajo ninguna
  circunstancia.
