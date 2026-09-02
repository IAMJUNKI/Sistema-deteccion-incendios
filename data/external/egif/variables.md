# Variables del datacubo

## Metadatos de las variables

Cada variable del NetCDF incluye `long_name` (nombre legible), `units` y, cuando
es necesario, `description` o el contrato temporal. Esto permite interpretar el
cubo sin depender del código. Las unidades principales son `degC` (temperatura),
`%` (humedad relativa), `km h-1` (viento), `mm` (precipitación), `m`/`km`
(distancias y topografía), `fraction` (proporciones) y `1` (indicadores binarios).

Cada fila del Parquet representa una celda de 1 km de Galicia y una fecha. El
NetCDF conserva además la rejilla rectangular completa y la máscara `is_galicia`.

## Identificación y calendario

| Variables | Significado |
|---|---|
| `cell_id`, `x`, `y`, `fecha` | Identificador y posición de la celda y fecha. |
| `is_galicia` | 1 si el centro de la celda pertenece a Galicia; fuera de Galicia no se exporta a Parquet. |
| `year` | Año derivado al exportar el Parquet, solo para particionar y realizar cortes temporales. |
| Calendario restante | Se deriva de `fecha` si un análisis lo necesita; no se almacena en el cubo canónico ni se usa como predictor. |

## Topografía (Copernicus DEM GLO-30)

`elevation_mean/std`, `slope_mean/std`, `roughness_mean/std` y las fracciones
de orientación `aspect_000_045_fraction` hasta `aspect_315_360_fraction`.

## Cobertura del suelo (CORINE 2018)

Las fracciones `artificial`, `agriculture`, `broadleaf_forest`,
`coniferous_forest`, `mixed_forest`, `scrub`, `open_spaces`, `wetlands` y
`water` suman aproximadamente 1 en cada celda. `forest_cover_fraction` es la
suma de los tres tipos de bosque. Cuando una celda costera no tiene píxeles
CORINE válidos, se copia el vector completo de la celda válida más próxima.

## Actividad humana (OpenStreetMap, instantánea 2022-01-01)

`road_length_km` suma los kilómetros de vías seleccionadas dentro de la celda.
La revisión de contrato añade longitudes por categoría
(`road_length_main_km`, `road_length_local_km`, `road_length_track_km` y
`road_length_other_km`), cuya suma es `road_length_km`, y las fracciones
`residential_area_fraction` y `building_area_fraction`. Son variables estáticas
calculadas una sola vez. Las dos distancias se excluyen porque se saturan
frecuentemente en cero a resolución de 1 km.

## Meteorología (ERA5-Land)

| Grupo | Variables |
|---|---|
| Día completo | `temperature_mean/min/max`, `relative_humidity_mean/min`, `wind_speed_mean/max`, `precipitation_sum` |
| Tarde crítica (12–18 h, Europe/Madrid) | `temperature_max_12_18h`, `relative_humidity_min_12_18h`, `wind_speed_max_12_18h`, `vpd_max_12_18h` |
| Déficit de presión de vapor | `vpd_mean` (día completo, kPa) |
| Memoria meteorológica | `temperature_mean_7d`, `wind_speed_mean_7d`, `relative_humidity_mean_7d/14d`, `precipitation_sum_3d/7d/14d/30d`, `consecutive_dry_days` |

Los acumulados y las medias móviles incluyen la fecha T y los días anteriores.
Las celdas de borde sin interpolación lineal válida se completan con el píxel
terrestre ERA5-Land más cercano.

La coordenada escalar `number` de ERA5 no se conserva: identificaba un único
miembro de ensemble y no tenía significado espacial, temporal ni predictivo.

## Resultados EGIF: no usar como predictores

- `target_ignicion`: 1 si hay una o más igniciones EGIF en la celda y fecha; 0
  si no hay ninguna.
- `burned_area_ha`: superficie quemada asociada en hectáreas.
- `large_fire_500ha`: 1 si la superficie asociada alcanza 500 ha.
- `is_near_ignition_25x25_10d`: auxiliar histórico. Vale 1 en el cuadrado
  de 25 × 25 celdas alrededor de una ignición durante el día del evento y los
  diez días anteriores (11 fechas en total; el cuadrado se recorta en la
  frontera de la rejilla y de Galicia).
  **Solo se usa como filtro de negativos históricos** al reproducir el
  protocolo de IberFire; nunca se usa como predictor ni estará disponible en
  producción.

`burned_area_ha`, `large_fire_500ha`, `target_ignicion` y
`is_near_ignition_25x25_10d` se excluyen de la matriz de predictores.

## Flags de inclusión

`src/datacube_profile.py` reúne un único perfil canónico para topografía,
cobertura, actividad humana, calendario y meteorología. No es una lista de
predictores del modelo: esa selección se valida después en ML.

## Revisión de variables aplicada al cubo final

La siguiente tabla separa la disponibilidad en el cubo de la posterior
selección de predictores del modelo. Los resultados EGIF y la máscara auxiliar
continúan almacenándose, pero nunca se usan como predictores.

| Decisión | Variables | Motivo breve |
|---|---|---|
| Añadidas | `vpd_mean`, `vpd_max_12_18h` | El déficit de presión de vapor (kPa) resume conjuntamente temperatura y sequedad del aire; la ventana de tarde representa la condición más desfavorable. |
| Añadidas | `wind_speed_mean_7d`, `relative_humidity_mean_14d`, `consecutive_dry_days` | Amplían la memoria meteorológica: viento medio reciente, humedad a dos semanas y duración interpretable de la racha seca (`precipitation_sum < 1 mm`). |
| Añadidas | `road_length_main_km`, `road_length_local_km`, `road_length_track_km`, `road_length_other_km` | Desagregan la intensidad y el tipo de acceso humano; su suma es `road_length_km`. |
| Añadidas | `residential_area_fraction`, `building_area_fraction` | Miden intensidad de ocupación humana dentro de la celda, evitando la saturación de una simple distancia. |
| Excluidas | `roughness_mean`, `roughness_std` | Redundantes con la pendiente y elevación; se conservan solo en las fuentes intermedias si hicieran falta auditorías. |
| Excluida | `forest_cover_fraction` | Es suma exacta de las tres fracciones de bosque ya disponibles. |
| Excluidas | `distance_to_road_m`, `distance_to_residential_area_m` | A 1 km se saturan frecuentemente en cero y pierden intensidad de red o edificación. |
| Excluidos | Calendario (`year`, `month`, semana, seno/coseno) | Se deriva de `time` cuando se necesita para particionar o analizar; no se almacena como señal del cubo. |
| Excluidas | `number`, `precipitation_sum_1d`, `aspect_no_data_fraction` | No representan una señal útil: miembro ERA5 residual, duplicado exacto y control de calidad constante, respectivamente. |
