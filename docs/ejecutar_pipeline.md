# Guía paso a paso: crear el datacubo desde cero

Esta guía está pensada para Windows y para alguien que empieza con el proyecto.
Los comandos se escriben en **PowerShell**, abierta en la carpeta del repositorio.

## Qué construye el proceso

El resultado son un cubo espacio-temporal y Parquet para Machine Learning de
Galicia, para los años que elija la persona usuaria. Por defecto, cada año va
del 1 de enero al 31 de diciembre; las fechas de las igniciones que aparezcan
en los XML EGIF no modifican ese intervalo.

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
├── meteorology/fwi/  # La crea el workflow si necesita descargar FWI
└── human_activity/  # La crea el pipeline para OpenStreetMap
```

Coloca los dos ficheros que el proyecto **no descarga automáticamente**:

| Dato | Qué debe descargar la persona | Dónde guardarlo |
|---|---|---|
| CORINE Land Cover 2018 | GeoTIFF `U2018_CLC2018_V2020_20u1.tif` | `data/raw/corine/` |
| Histórico oficial EGIF | Uno o varios XML de MITECO de los años que quieras usar | `data/raw/fire_history/` |

El límite de Galicia se descarga automáticamente si falta cuando se usa
`--rebuild-static`. El DEM Copernicus GLO-30 también se descarga automáticamente
en esa primera construcción. Ambos necesitan conexión a internet. La capa de
carreteras y zonas residenciales se descarga también automáticamente desde un
extracto Shapefile versionado de OpenStreetMap si falta; se guarda en
`data/raw/human_activity/` y no se vuelve a descargar.

## Paso 5. Elegir los años y ejecutar el pipeline completo

No hay que abrir ningún XML ni indicar fechas de incendios. Elige los **años
completos** que descargaste de EGIF. El workflow acepta la carpeta, une todos
los XML y elimina duplicados de intervalos solapados.

Por ejemplo, para generar el cubo original 2019–2023:

```powershell
python -m src.workflow `
  --start-year 2019 `
  --end-year 2023 `
  --rebuild-static
```

Para un cubo ampliado 2016–2023, cambia solamente los años:

```powershell
python -m src.workflow --start-year 2016 --end-year 2023
```

El workflow comprueba los meses ERA5 necesarios desde diciembre del año
anterior y descarga **solo los que falten**. También descarga o reutiliza FWI
para los mismos años. Los ficheros FWI se verifican por fechas: si uno procede
de una ejecución parcial y no cubre el intervalo solicitado, se vuelve a
descargar ese año de forma segura. No se redescargan ficheros crudos que ya
cubren por completo el periodo pedido.

La primera vez tarda bastante: crea la rejilla, descarga el DEM, procesa CORINE,
descarga/procesa la capa estática de actividad humana, procesa ERA5, descarga
FWI desde CEMS si no está disponible, asigna las igniciones EGIF, crea el NetCDF
y exporta los Parquet.
No cierres PowerShell mientras se ejecuta.

La primera descarga de FWI requiere haber iniciado sesión una vez en el portal
EWDS de Copernicus y aceptado sus condiciones de CEMS. Usa el mismo
`COPERNICUS_CDS_API_KEY` definido en `.env`; no requiere otra credencial. Los
archivos anuales se guardan en `data/raw/meteorology/fwi/`. Se reutilizan si
ya cubren el año solicitado; un fichero parcial se completa automáticamente
cuando haga falta.

### Caso excepcional: año final parcial

Para un año actual todavía incompleto, indícalo de forma explícita. Por ejemplo,
si EGIF se ha descargado hasta el 15 de agosto de 2024:

```powershell
python -m src.workflow `
  --start-year 2019 `
  --end-year 2024 `
  --partial-final-year `
  --final-date 2024-08-15
```

Sin esa opción, `--end-year 2023` siempre significa hasta el 31 de diciembre de
2023, aunque el último incendio registrado en el XML haya sido anterior. Los
días sin incendios se conservan correctamente como `target_ignicion = 0`.

## Paso 6. Ejecutarlo otra vez cuando ya existen los datos

Si cambias EGIF o quieres regenerar el resultado, no hace falta descargar de
nuevo ERA5, FWI ni recalcular DEM/CORINE. Ejecuta:

```powershell
python -m src.workflow `
  --start-year 2019 `
  --end-year 2023
```

El workflow reemplaza sus salidas generadas anteriores, pero nunca borra los
datos originales de `data/raw/`.

No existe un segundo script para construir un cubo "final" o "de prueba":
`src.workflow` es la única vía canónica y también vuelve a exportar los Parquet.

Para recalcular solo las variables estáticas de carreteras y zonas residenciales
con el mismo extracto OSM local, añade `--rebuild-human-activity` al comando.

## Paso 7. Comprobar el resultado

Al terminar deben existir:

```text
data/processed/datacube/galicia_1km.nc
data/processed/datacube/run_manifest.json
data/processed/tabular/egif/metadata.json
data/processed/tabular/egif/year=2016/dataset_2016.parquet
…
data/processed/tabular/egif/year=2023/dataset_2023.parquet
```

En PyCharm abre y ejecuta con el kernel `incendios-forestales`:

1. `notebooks/07_validacion_datacubo.ipynb`
2. `notebooks/08_validacion_dataset_parquet.ipynb`

Si una ejecución falla, no borres `data/raw/`: guarda los datos descargados y
permite repetir el proceso sin volver a descargarlos.
