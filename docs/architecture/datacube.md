# Arquitectura y diccionario del datacubo

## Flujo reproducible

```text
Límite Galicia + DEM + CORINE ──> capas estáticas 2D
ERA5-Land horario ──────────────> meteorología diaria y acumulados 3D
EGIF XML ───────────────────────> target de ignición 3D
Calendario ─────────────────────> variables temporales 1D
                                      │
                                      ▼
                    NetCDF final + Parquet tabular por año
```

`src.workflow` es la única entrada operativa. No usa ningún Parquet Mikel ni
ninguna fuente FIRMS.

## Variables

| Familia | Variables |
|---|---|
| Rejilla | `cell_id`, `x`, `y`, `is_galicia` |
| Topografía | `elevation_mean/std`, `slope_mean/std`, `roughness_mean/std`, ocho fracciones de orientación por intervalo angular y `aspect_no_data_fraction` |
| CORINE | nueve fracciones de cobertura y `forest_cover_fraction` |
| Calendario | año, mes, semana ISO, día del año, día de semana, fin de semana y codificaciones seno/coseno |
| ERA5 | temperatura, humedad relativa y viento diarios y en 12–18 h; precipitación diaria, acumulados 1/3/7/14/30 días y días secos consecutivos |
| Resultados EGIF | `target_ignicion`, `burned_area_ha`, `large_fire_500ha` |

`burned_area_ha` y `large_fire_500ha` son variables de resultado/auditoría y
nunca entran como predictores.

## Convenciones de datos

- Todas las fracciones están entre 0 y 1.
- `cell_id` es el índice estable de la malla rectangular, calculado como
  `row * n_x + column`.
- Fuera de Galicia las capas llevan `NaN`; no se exportan a Parquet.
- La exportación Parquet contiene una fila por `fecha` y `cell_id` activo con
  predictores completos, consolidada en un fichero por año.
- El directorio Parquet incluye `metadata.json` con las columnas predictoras y
  el contrato temporal.

## Decisiones explícitas

El producto actual no aplica desplazamiento temporal: un acumulado de 30 días
en `T` cubre `T-29` a `T`. Por ello solo debe presentarse como nowcast o
análisis del riesgo del día. Para una previsión operativa se creará en el futuro
una variante separada con información disponible hasta `T-1`; no se mezclará
con este dataset.
