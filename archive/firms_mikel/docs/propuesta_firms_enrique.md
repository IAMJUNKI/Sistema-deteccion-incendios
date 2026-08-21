# Propuesta: NASA FIRMS como fuente del target — Enrique Bravo

> Tarea de la reunión del 5/7: *"Probar los datasets individualmente y preparar una propuesta de dataset final"*. Esta es la parte de **NASA FIRMS** (la fuente de la variable objetivo).

---

## 1. Qué he hecho

He descargado y analizado el histórico completo de detecciones térmicas VIIRS de Galicia 2019-2024: **94.364 detecciones de España → 5.818 en Galicia → 5.132 tras filtro de calidad**.

Archivos generados (en el Drive/carpeta compartida):

- `firms_galicia_2019_2024_limpio.csv` — dataset limpio listo para cruzar con el grid
- `exploracion_firms_galicia.ipynb` — notebook reproducible con todo el proceso
- `analisis_firms_galicia.png` — gráficos resumen

## 2. De dónde salen los datos (reproducible por cualquiera)

1. Portal FIRMS de NASA: `firms.modaps.eosdis.nasa.gov` → menú → **Download Archived Data** → **Country Yearly Summary**.
2. Archivos anuales por país, sensor VIIRS S-NPP (375 m), sin registro ni API key:
   `https://firms.modaps.eosdis.nasa.gov/data/country/csv/viirs-snpp/<año>/viirs-snpp_<año>_Spain.csv`
3. Recorte a Galicia con la frontera oficial GADM 4.1 (la misma fuente que usa el grid de la Fase 1) — no un rectángulo, para no colar focos de Asturias/León/Portugal.
4. La API con clave existe pero es para tiempo casi real (últimos días) → la usaremos en la Fase 5 (producción), no para el histórico.

## 3. Filtro de calidad aplicado (propuesta a validar el domingo)

| Filtro | Se elimina | Motivo |
|---|---|---|
| `confidence` ∈ {n, h} | 205 filas (l = baja) | Descarta reflejos solares y falsos positivos |
| `type` == 0 (vegetación) | 470 filas (type 2 = fuente estática) | Chimeneas/industria que el satélite ve calientes a diario |
| | 19 filas (type 3 = agua) | Reflejos sobre agua |

## 4. Hallazgos que validan los datos

- **2022 concentra el 55%** de los focos (2.815) y los días pico —18 y 19 de julio de 2022 con ~500 focos/día— son exactamente la ola de O Courel y Valdeorras. El pico de sept-2020 coincide con los incendios de Ourense de ese año. Los datos cuadran con la realidad.
- **Estacionalidad doble**: julio-septiembre concentra el 82%, pero hay un pico secundario en **marzo** (quemas de primavera, patrón gallego documentado). El modelo tendrá que aprender ambas temporadas.
- **Concentración espacial en Ourense oriental** (O Courel, Valdeorras, Baixa Limia), como cabía esperar. Sin artefactos urbanos tras el filtrado.
- 2/3 de las detecciones son **nocturnas** (el sensor discrimina mejor de noche).
- La intensidad (`frp`) es muy asimétrica: mediana 6 MW, máximo 475 MW.

## 5. Propuesta de columnas FIRMS para el dataset conjunto

Clave de unión con el resto de fuentes: **(`cell_id`, `fecha`)** — el foco se asigna a la celda del grid de 1 km que contiene sus coordenadas.

| Columna propuesta | Origen | Uso |
|---|---|---|
| `cell_id` | lat/lon → grid Fase 1 | Clave espacial |
| `fecha` | `acq_date` | Clave temporal |
| `n_focos` | conteo por celda-día | Base del target |
| `frp_max`, `frp_sum` | `frp` | Intensidad (útil para filtrar eventos triviales) |
| `confidence_max` | `confidence` | Trazabilidad de calidad |

El target definitivo (Y=1 solo en la celda-día de **inicio** de cada incendio) se construye después agrupando focos contiguos en eventos únicos (clustering espacio-temporal, Fase 2 del plan) — un incendio grande genera cientos de focos durante días y no debe contarse cientos de veces.

## 6. Puntos abiertos para decidir el domingo

1. **¿Añadimos el satélite NOAA-20?** Lleva el mismo sensor VIIRS (disponible desde 2018) y duplicaría las pasadas diarias. Más detecciones = mejor target, a coste de deduplicar entre satélites.
2. **¿Umbral mínimo de evento?** Con FRP y nº de focos podemos descartar eventos triviales (1 foco aislado de 1 MW) si el grupo quiere acercarse a la definición normativa de incendio.
3. **Años pre-2019**: VIIRS S-NPP tiene datos desde 2012, por si queremos ampliar el histórico de entrenamiento (los docs del repo ya contemplan 2013-2018 con el grid CORINE 2012).
