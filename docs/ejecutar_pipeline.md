# Ejecutar el pipeline histórico

Esta guía explica cómo construir el datacubo histórico de Galicia y sus Parquet. La vía canónica es siempre `python -m src.workflow`: no existe un segundo pipeline final o de prueba.

Conserva siempre `data/raw/`. El workflow reutiliza cualquier fichero original local que cubra el periodo solicitado y descarga únicamente lo que falte. En cambio, las salidas de `data/processed/` se reconstruyen al ejecutar el workflow.

## 1. Resultado

El proceso crea un cubo espacio-temporal de 1 km y un dataset tabular por año:

```text
data/processed/datacube/galicia_1km.nc
data/processed/datacube/run_manifest.json
data/processed/tabular/egif/metadata.json
data/processed/tabular/egif/year=AAAA/dataset_AAAA.parquet
```

Cada fila de Parquet representa una celda de Galicia y un día; incluye `fecha`, `x`, `y` y `cell_id`. Un año seleccionado abarca, por defecto, desde el 1 de enero hasta el 31 de diciembre. Los días sin igniciones son observaciones válidas con `target_ignicion = 0`.

### Capas estáticas y capas dinámicas

Una **capa estática** tiene un único valor por celda de 1 km y no cambia de un día a otro dentro del cubo. Describe el territorio: máscara de Galicia y geometría de la rejilla, elevación, pendiente, orientación, clases CORINE, carreteras, edificios y áreas residenciales. Se calculan una vez y se reutilizan en cada ejecución. Solo deben reconstruirse con `--rebuild-static` si cambia su fuente o la forma de calcularlas.

Una **capa dinámica** tiene un valor por celda **y por día**. En este proyecto son la meteorología derivada de ERA5-Land, sus acumulados, el FWI y el histórico/target de igniciones EGIF. Estas capas se vuelven a calcular para el intervalo temporal elegido, aunque las fuentes raw ya estén descargadas.

## 2. Preparación inicial

Abre PowerShell en la carpeta que contiene `README.md` y `environment.yml`, y ejecuta una sola vez:

```powershell
conda env create -f environment.yml
conda activate incendios-forestales
Copy-Item .env.example .env
```

### Obtener el token de Copernicus

1. Crea una cuenta o inicia sesión en [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
2. Abre tu [perfil de CDS](https://cds.climate.copernicus.eu/profile) y copia el **Personal Access Token** que muestra la sección de configuración de API. No es una contraseña ni el antiguo identificador numérico de usuario.
3. Para usar FWI, inicia sesión también en [CEMS Early Warning Data Store](https://ewds.climate.copernicus.eu/) y abre la página del conjunto `cems-fire-historical-v1`; acepta sus condiciones de uso en el formulario de descarga. Esto se hace una vez por cuenta.
4. Abre `.env` y pega el token sin comillas ni espacios:

```text
COPERNICUS_CDS_API_KEY=tu_token_personal
```

El mismo token sirve para ERA5-Land y para FWI de CEMS/EWDS. El cliente `cdsapi` ya forma parte del entorno del proyecto. No subas `.env` a Git.

Comprueba la instalación con:

```powershell
python -m pytest tests -q
```

## 3. Datos raw: qué guardar y cómo se llaman

No descargues a mano un dato que ya exista en estas carpetas. El workflow comprueba primero los archivos locales.

| Bloque | Carpeta | Nombre habitual | Origen |
| --- | --- | --- | --- |
| Incendios EGIF | `data/raw/fire_history/` | Cualquier `*.xml`, por ejemplo `Xml_20260903_010933_1.xml` | Manual: XML oficiales MITECO. Puede haber varios; se unen y se deduplican. |
| CORINE 2018 | `data/raw/corine/` | `U2018_CLC2018_V2020_20u1.tif` | Manual: descargar una vez. |
| ERA5-Land horario | `data/raw/meteorology/era5/` | Canónico: `era5_land_galicia_AAAA_MM.nc`; histórico aceptado: `era5land_galicia_AAAA_MM.nc` | Automático si falta; cualquiera de ambos nombres se reutiliza. |
| Fire Weather Index | `data/raw/meteorology/fwi/` | `fwi_galicia_AAAA.nc` | Automático desde CEMS/EWDS si falta o no cubre todas las fechas solicitadas. |
| Límite de Galicia | `data/raw/igm/` | `galicia_boundary.geojson` | Automático al reconstruir capas estáticas. |
| Modelo de elevación | No se conserva como raw | No hay un archivo que deba aportar la persona usuaria | Se obtiene de Copernicus DEM GLO-30 al reconstruir topografía y se integra directamente en la capa procesada. |
| OpenStreetMap | `data/raw/human_activity/` | `galicia-220101-free.shp.zip` y `galicia-220101-free-shp/` | Automático; extracto histórico versionado de Geofabrik. |

Para 2016--2023 se necesitan XML EGIF que cubran esos años. ERA5 necesita además diciembre del año anterior para los acumulados: para 2016--2023 se valida desde diciembre de 2015. No hay que obtener estos meses manualmente si el token está configurado.

> Si ya dispones de una copia válida, colócala en la carpeta raw correspondiente con el nombre indicado: es preferible a volver a descargarla. En ERA5 no dejes las dos variantes de nombre para el mismo mes; el workflow se detendrá para evitar una copia ambigua.

## 4. Comandos recomendados

### Primera construcción de un periodo

Para construir, por ejemplo, el cubo 2016--2023 en un equipo donde aún no existan las capas estáticas:

```powershell
python -m src.workflow --start-year 2016 --end-year 2023 --rebuild-static
```

`--rebuild-static` recalcula límite, rejilla, topografía, CORINE y actividad humana. Puede descargar límite, DEM y OpenStreetMap si faltan; por eso es el comando más lento.

### Regenerar cubo y Parquet con los raw locales

En las ejecuciones posteriores, normalmente basta con:

```powershell
python -m src.workflow --start-year 2016 --end-year 2023
```

Este comando reutiliza capas estáticas y raw presentes, completa solo ERA5/FWI faltante, genera las variables dinámicas, el NetCDF y los Parquet. Por defecto lee todos los XML de `data/raw/fire_history/`.

Para cambiar de periodo, cambia solamente los años:

```powershell
python -m src.workflow --start-year 2019 --end-year 2023
```

### Año final parcial (solo si se necesita)

Para terminar conscientemente en una fecha distinta de 31 de diciembre:

```powershell
python -m src.workflow `
  --start-year 2019 `
  --end-year 2024 `
  --partial-final-year `
  --final-date 2024-08-15
```

No uses este modo porque el último incendio de un XML sea anterior: los días posteriores sin incendios también deben estar en el cubo. Para el histórico, usa años completos.

## 5. Referencia de opciones

Muestra las opciones disponibles en tu versión con:

```powershell
python -m src.workflow --help
```

| Opción | Qué hace | Cuándo usarla |
| --- | --- | --- |
| `--start-year AAAA --end-year BBBB` | Define el periodo de salida. | Siempre. |
| `--rebuild-static` | Reconstruye límite, rejilla, topografía, CORINE y actividad humana. | Primera ejecución o cambio de fuentes/lógica estática. |
| `--rebuild-human-activity` | Recalcula solo carreteras, edificios y áreas residenciales desde OSM local. | Cambio de la lógica de actividad humana. |
| `--skip-daily-meteorology` | Reutiliza `data/processed/meteorology/era5_daily.nc`. | Solo uso avanzado; debe cubrir exactamente el periodo y contexto necesarios. |
| `--no-download-missing-era5` | Valida ERA5 local y prohíbe descargar meses que falten. | Sin conexión o para comprobar que la caché está completa. |
| `--skip-fwi` | No descarga ni añade el baseline FWI. | Diagnóstico si EWDS no está disponible; no produce el cubo final comparable. |
| `--egif-xml RUTA` | Usa un XML o carpeta distinta. | Normalmente innecesario. |
| `--partial-final-year --final-date AAAA-MM-DD` | Limita explícitamente el último año. | Periodo final incompleto. |
| `--grid RUTA` / `--spatial-cube RUTA` | Sustituye rutas de capas estáticas. | Desarrollo; no se recomienda en uso normal. |

Por ejemplo, para construir sin permitir peticiones ERA5, porque ya están todos los NetCDF locales:

```powershell
python -m src.workflow --start-year 2019 --end-year 2023 --no-download-missing-era5
```

## 6. Descargar solamente ERA5-Land

El workflow es el método habitual. Para completar la caché raw sin construir cubo ni Parquet:

```powershell
python -m src.ingestion.era5 `
  --output-dir data/raw/meteorology/era5 `
  --start-date 2015-12-01 `
  --end-date 2023-12-31
```

Esta descarga es mensual y omite los meses ya presentes. No procesa meteorología, no descarga FWI y no modifica `data/processed/`.

## 7. Seguimiento y validación

El avance se escribe en la terminal. Al finalizar revisa:

```text
data/processed/datacube/run_manifest.json  -> estado y periodo declarado
data/processed/datacube/galicia_1km.nc     -> cubo final
data/processed/tabular/egif/metadata.json  -> columnas del dataset tabular
```

En PyCharm, usando el kernel `incendios-forestales`, ejecuta:

1. `notebooks/07_validacion_datacubo.ipynb`.
2. `notebooks/08_validacion_dataset_parquet.ipynb`.

Si la ejecución se interrumpe, no borres `data/raw/`: al repetir, se reutilizarán los archivos ya completos. Es normal que `data/processed/` se regenere.

## 8. Problemas habituales

**Faltan XML EGIF.** Coloca uno o varios `*.xml` en `data/raw/fire_history/` y repite el comando. El pipeline lee todos los XML y elimina duplicados de periodos solapados.

**ERA5 se quiere descargar pese a existir.** Verifica carpeta, tamaño no nulo y nombre `era5_land_galicia_AAAA_MM.nc` o `era5land_galicia_AAAA_MM.nc`. No mantengas dos variantes para un mismo mes.

**Falla una solicitud FWI de EWDS.** Conserva los raw ya descargados y reintenta más tarde: el servicio puede rechazar solicitudes temporalmente. `--skip-fwi` permite comprobar el resto del flujo, pero no sustituye la ejecución final con FWI.

**Quiero empezar en otro año.** No hace falta borrar nada. Añade los XML EGIF necesarios y ejecuta con otro `--start-year`; el workflow validará y completará solo los raw meteorológicos que falten.
