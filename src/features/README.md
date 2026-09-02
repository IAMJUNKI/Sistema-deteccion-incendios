# Variables temporales y salida tabular

`time.py` crea el calendario diario. `tabular.py` transforma el NetCDF final en
Parquet por bloques, filtrando solo celdas activas y fechas cubiertas por EGIF.
No realiza muestreo ni ingeniería de variables del modelo.
**Fase 3 del proyecto.** Enriquecimiento del dataset con variables acumuladas y derivadas, y auditoría estricta contra el data leakage temporal.

## Responsabilidad

- Calcular acumulados meteorológicos en ventanas de 3, 7, 14 y 30 días.
- Extraer variables exclusivamente en la franja de mayor riesgo: **12h-18h** (temperatura máxima, humedad mínima, viento máximo).
- Calcular días consecutivos sin lluvia (estrés hídrico).
- Añadir variables de proximidad humana (distancia a carreteras, núcleos urbanos vía OSM).
- Calcular historial de incendios acumulado en el vecindario de cada celda.
- Garantizar que ninguna feature use información del día T o posterior para predecir T.
- En producción, combinar el estado observado hasta el último día completo con
  el forecast MeteoGalicia del día objetivo y, para T+2/T+3, de los días
  intermedios.

## Entregable

**Dataset Maestro** en formato `.parquet` con todas las features finales, listo para ML.

## Reglas críticas anti-leakage

1. Para predecir el riesgo del día `T`, solo usar datos hasta `T-1`.
2. En el benchmark histórico la ventana usa la meteorología real del día
   objetivo y se etiqueta `era5_perfect`; en producción usa el forecast del
   día objetivo. En ambos casos la hora local es 12:00–18:00 inclusiva.
3. Excluir como negativos cualquier celda a ±5 días de un incendio real.
