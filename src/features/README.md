# Variables temporales y salida tabular

`time.py` crea el calendario diario. `tabular.py` transforma el NetCDF final en
Parquet por bloques, filtrando solo celdas activas y fechas cubiertas por EGIF.
No realiza muestreo ni ingeniería de variables del modelo.
