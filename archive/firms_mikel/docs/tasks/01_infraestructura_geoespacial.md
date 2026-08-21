# 01 — Infraestructura Geoespacial (Fase 1)

---

## 1. ¿Qué se ha hecho?
- Se ha creado la estructura modular de la Fase 1 en el paquete `src.geospatial` con cuatro módulos:
  - `grid.py`: Lógica de descarga de la frontera regional y construcción de la rejilla de celdas de 1 km x 1 km.
  - `topography.py`: Descarga y procesamiento de Copernicus DEM GLO-30 (altitud, cálculo de pendiente y orientación).
  - `vegetation.py`: Lógica para procesar y agrupar CORINE Land Cover en macro-clases de combustible de incendio.
  - `pipeline.py`: Script de orquestación de la fase.
- Se ha ejecutado el pipeline obteniendo la rejilla base de Galicia: **30.697 celdas terrestres activas**.
- Se ha generado la primera versión de la rejilla multi-temporal en formato GeoParquet: [galicia_grid_1km_2018.parquet](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/data/processed/grid/galicia_grid_1km_2018.parquet) (basada en CORINE 2018).
- Se ha configurado el pipeline para admitir la generación de las versiones `2012` y `2024` de forma paramétrica para soportar la estrategia multi-temporal (Opción A).
- Se ha generado un mapa térmico de control de calidad visual en: [verificacion_grid_fase1.png](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/docs/verificacion_grid_fase1.png).

---

## 2. ¿Por qué se ha hecho?
Este paso es el **cuello de botella y cimiento absoluto del TFM**. Para predecir el riesgo de incendio diario celda a celda (1 km²), necesitábamos transformar la geografía continua de Galicia en un "tablero de juego" discreto indexable por un identificador único (`cell_id`). Sin esta rejilla estática base y sus variables de terreno, es matemáticamente imposible cruzar los datos meteorológicos temporales (ERA5) y el histórico de focos activos (NASA FIRMS).

---

## 3. ¿Cómo se ha hecho?
El pipeline se ejecuta secuencialmente mediante los siguientes procesos:
1. **Descarga de límites (GADM):** Se descarga el GeoJSON de GADM España, se filtra Galicia y se proyecta al CRS métrico oficial `EPSG:25829` (UTM Huso 29N).
2. **Generación del Grid:** Se crea una malla base a partir del bounding box en pasos de 1.000 metros. Mediante una intersección espacial (`sjoin`), se conservan únicamente las celdas que tocan tierra gallega. A cada una se le calcula el centroide en coordenadas geográficas `EPSG:4326` (WGS84).
3. **Descarga y Cálculo de DEM:** Usando el bounding box de la rejilla, se descarga el Copernicus DEM GLO-30 desde AWS. Se reproyecta a 30m de resolución en UTM y se calcula la pendiente y la orientación del terreno a través de gradientes discretos en metros.
4. **Agregación Zonal Vectorizada:** Para evitar la lentitud de hacer 30.000 máscaras vectoriales individuales, el script rasteriza las geometrías de las celdas asignando su `cell_id` a la resolución del DEM. Posteriormente, aplana las matrices y realiza un agrupamiento (`groupby`) con Pandas sobre los píxeles activos, obteniendo promedios en segundos.
5. **Combustibles CLC (CORINE):** Si está presente, el GeoTIFF de CORINE Land Cover se reproyecta y rasteriza para calcular el tipo de combustible predominante de cada celda y su cobertura forestal. Si no se detecta localmente, el script posee tolerancia a fallos y autocompleta con datos por defecto (`otros`, `0%`) para no bloquear el desarrollo.

---

## 4. ¿Por qué se han elegido estas tecnologías?
- **GeoPandas & Shapely:** Es el estándar moderno de Python para manipulación de vectores geográficos, permitiendo hacer el `sjoin` de contorno y la creación de polígonos de forma eficiente.
- **dem-stitcher & AWS Open Data:** Evita la burocracia de registrar cuentas y gestionar tokens en portales de Copernicus Dataspace, permitiendo descargar los fragmentos de DEM directamente del S3 público de AWS de manera automatizada.
- **Rasterio & Rioxarray:** Facilitan la lectura y la reproyección rápida de formatos raster a través de GDAL.
- **Vectorized Groupby (Pandas + Rasterize):** Se descartó la librería clásica de estadísticas zonales (`rasterstats`) porque realizar 30.697 búsquedas espaciales secuenciales tardaba varios minutos. Al rasterizar y usar el `groupby` de Pandas a nivel de pixel plano, el cálculo completo se redujo a **2 segundos**.
- **Formatos GeoParquet & snappy:** El formato Parquet conserva perfectamente los tipos de datos geométricos de GeoPandas y ocupa un 90% menos de espacio en disco que un GeoJSON o Shapefile equivalente, acelerando las lecturas/escrituras en las siguientes fases.

---

## 5. ¿Qué conseguimos con ello?
- **Independencia y Paralelismo:** Todo el equipo de 6 personas puede descargar la rejilla `galicia_grid_1km_v1.parquet` (unos pocos megabytes) en su máquina y empezar a desarrollar las Fases 2 (ingesta de FIRMS y ERA5), 3 (feature engineering) y 5 (esqueleto del dashboard de Streamlit) de manera simultánea sin depender de que se terminen otros scripts.
- **Consistencia espacial:** Garantizamos que todo el modelo y la visualización de la webapp compartan exactamente el mismo índice espacial (`cell_id`), previniendo errores de alineación o coordenadas geográficas a futuro.
