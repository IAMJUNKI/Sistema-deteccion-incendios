"""
Descarga masiva ERA5-Land horario para Galicia, 2019-2024.
Un archivo NetCDF por mes (72 en total, ~250 MB). Reanudable:
si un mes ya esta descargado, lo salta — se puede cortar y relanzar.

Uso:
  conda activate incendios-forestales
  python descarga_era5_2019_2024.py
Requiere el archivo ~/.cdsapirc con el token personal (ver instrucciones).
"""
from pathlib import Path

import cdsapi

SALIDA = Path("Datos/era5_bruto")
SALIDA.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]
# Norte, Oeste, Sur, Este — rectangulo que envuelve Galicia
AREA = [43.8, -9.31, 41.8, -6.74]

cliente = cdsapi.Client()

for anio in range(2019, 2025):
    for mes in range(1, 13):
        destino = SALIDA / f"era5land_galicia_{anio}_{mes:02d}.nc"
        if destino.exists() and destino.stat().st_size > 100_000:
            print(f"[SKIP] {destino.name} ya existe")
            continue
        print(f"[PIDE] {destino.name} ...")
        cliente.retrieve(
            "reanalysis-era5-land",
            {
                "variable": VARIABLES,
                "year": str(anio),
                "month": f"{mes:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": AREA,
                "format": "netcdf",
                "download_format": "unarchived",
            },
            str(destino),
        )
        print(f"[OK]   {destino.name}")

print("Descarga completa.")
