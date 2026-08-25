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
| `year`, `month`, `iso_week`, `day_of_year`, `day_of_week`, `is_weekend` | Calendario. |
| `day_of_year_sin/cos`, `month_sin/cos` | Codificación cíclica de estacionalidad. |

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

`distance_to_road_m` es la distancia desde el polígono de la celda a la vía
seleccionada más cercana y vale 0 si esta la intersecta. `road_length_km` suma
los kilómetros de vías seleccionadas dentro de la celda.
`distance_to_residential_area_m` mide la distancia al área residencial o núcleo
poblado OSM más cercano. Son variables estáticas calculadas una sola vez y
estarán disponibles igualmente en producción.

## Meteorología (ERA5-Land)

| Grupo | Variables |
|---|---|
| Día completo | `temperature_mean/min/max`, `relative_humidity_mean/min`, `wind_speed_mean/max`, `precipitation_sum` |
| Tarde crítica (12–18 h, Europe/Madrid) | `temperature_max_12_18h`, `relative_humidity_min_12_18h`, `wind_speed_max_12_18h` |
| Memoria meteorológica | `temperature_mean_7d`, `relative_humidity_mean_7d`, `precipitation_sum_3d/7d/14d/30d`, `consecutive_dry_days` |

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
