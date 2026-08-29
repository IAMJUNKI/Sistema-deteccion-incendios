# Guía paso a paso: crear el datacubo desde cero

Esta guía está pensada para Windows y para alguien que empieza con el proyecto.
Los comandos se escriben en **PowerShell**, abierta en la carpeta del repositorio.

## Qué construye el proceso

El resultado son un cubo espacial-temporal y Parquet para Machine Learning de
Galicia, desde el 01-ene-2019 hasta la última fecha que exista en el XML EGIF.

```text
datos originales → capas geográficas → meteorología + incendios → cubo → Parquet
```

## Paso 1. Instalar lo necesario y descargar el repositorio

En un equipo nuevo instala primero:

- [Git for Windows](https://git-scm.com/download/win).
- [Miniconda](https://docs.anaconda.com/miniconda/) o Anaconda.
- PyCharm Community o Professional (opcional, pero recomendado para los notebooks).

Abre **PowerShell** y ejecuta estos comandos. Sustituye
`<URL_DEL_REPOSITORIO>` por la dirección que aparece en el botón **Code** del
repositorio en GitHub:

```powershell
git clone <URL_DEL_REPOSITORIO>
cd Sistema-deteccion-incendios
```

Todos los comandos que aparecen a partir de aquí deben ejecutarse en esa misma
ventana y carpeta: es la que contiene `README.md` y `environment.yml`.

## Paso 2. Crear el entorno de Python (solo una vez)

Instala Miniconda o Anaconda si no está instalado. Después ejecuta:

```powershell
conda env create -f environment.yml
conda activate incendios-forestales
```

Comprueba que funciona:

```powershell
python -m pytest tests -q
```

Debe terminar indicando que todos los tests han pasado.

## Paso 3. Crear el archivo de configuración (solo una vez)

Haz una copia de `.env.example` con este comando:

```powershell
Copy-Item .env.example .env
```

Abre `.env` con PyCharm o con el Bloc de notas. Regístrate en Copernicus Climate
Data Store, copia tu token personal desde su perfil y sustitúyelo:

```text
COPERNICUS_CDS_API_KEY=tu_token_personal
```

No subas `.env` a Git: contiene credenciales personales.

## Paso 4. Conseguir y colocar los datos originales

Dentro del repositorio crea, si no existen, estas carpetas:

```text
data/raw/
├── corine/
├── fire_history/
├── meteorology/era5/
└── human_activity/  # La crea el pipeline para OpenStreetMap
```

Coloca los dos ficheros que el proyecto **no descarga automáticamente**:

| Dato | Qué debe descargar la persona | Dónde guardarlo |
|---|---|---|
| CORINE Land Cover 2018 | GeoTIFF `U2018_CLC2018_V2020_20u1.tif` | `data/raw/corine/` |
| Histórico oficial EGIF | XML descargado de MITECO, desde 2018 | `data/raw/fire_history/` |

El límite de Galicia se descarga automáticamente si falta cuando se usa
`--rebuild-static`. El DEM Copernicus GLO-30 también se descarga automáticamente
en esa primera construcción. Ambos necesitan conexión a internet. La capa de
carreteras y zonas residenciales se descarga también automáticamente desde un
extracto Shapefile versionado de OpenStreetMap si falta; se guarda en
`data/raw/human_activity/` y no se vuelve a descargar.

## Paso 5. Descargar ERA5-Land (solo si aún no está en el equipo)

ERA5 no se descarga al ejecutar el workflow principal, para evitar descargas
largas involuntarias. Se descarga una vez con este comando:

```powershell
python -m src.ingestion.era5 `
  --output-dir data/raw/meteorology/era5 `
  --start-date 2018-12-01 `
  --end-date 2023-11-26
```

El script guarda un NetCDF por mes y omite los que ya existan, por lo que se
puede reanudar sin empezar de nuevo. Al terminar, deben verse ficheros como:

`data/raw/meteorology/era5/era5_land_galicia_2019_01.nc`.

## Paso 6. Ejecutar el pipeline completo

Primero mira el nombre exacto del XML que hayas puesto:

```powershell
Get-ChildItem data/raw/fire_history
```

Después sustituye `NOMBRE_DEL_ARCHIVO.xml` por ese nombre:

```powershell
python -m src.workflow `
  --egif-xml data/raw/fire_history/NOMBRE_DEL_ARCHIVO.xml `
  --rebuild-static
```

La primera vez tarda bastante: crea la rejilla, descarga el DEM, procesa CORINE,
descarga/procesa la capa estática de actividad humana, procesa ERA5, asigna las
igniciones EGIF, crea el NetCDF y exporta los Parquet.
No cierres PowerShell mientras se ejecuta.

## Paso 7. Ejecutarlo otra vez cuando ya existen los datos

Si cambias EGIF o quieres regenerar el resultado, no hace falta descargar de
nuevo ERA5 ni recalcular DEM/CORINE. Ejecuta:

```powershell
python -m src.workflow `
  --egif-xml data/raw/fire_history/NOMBRE_DEL_ARCHIVO.xml `
  --skip-daily-meteorology
```

El workflow reemplaza sus salidas generadas anteriores, pero nunca borra los
datos originales de `data/raw/`.

No existe un segundo script para construir un cubo "final" o "de prueba":
`src.workflow` es la única vía canónica y también vuelve a exportar los Parquet.

Para recalcular solo las variables estáticas de carreteras y zonas residenciales
con el mismo extracto OSM local, añade `--rebuild-human-activity` al comando.

## Paso 8. Comprobar el resultado

Al terminar deben existir:

```text
data/processed/datacube/galicia_1km.nc
data/processed/tabular/egif/metadata.json
data/processed/tabular/egif/year=2019/dataset_2019.parquet
…
data/processed/tabular/egif/year=2023/dataset_2023.parquet
```

En PyCharm abre y ejecuta con el kernel `incendios-forestales`:

1. `notebooks/07_validacion_datacubo.ipynb`
2. `notebooks/08_validacion_dataset_parquet.ipynb`

Si una ejecución falla, no borres `data/raw/`: guarda los datos descargados y
permite repetir el proceso sin volver a descargarlos.
