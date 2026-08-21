# Módulo: geospatial — Infraestructura Geoespacial

**Fase 1 del proyecto.** Construcción del tablero de juego geográfico: rejilla de celdas de 1km×1km sobre Galicia, extracción de variables topográficas (DEM) y cobertura del suelo (CORINE Land Cover).

## Responsabilidad

Este módulo se ejecuta **una sola vez** para generar el dataset estático de referencia espacial. Sus outputs son la base sobre la que se construyen todas las fases posteriores.

## Entregable

DataFrame estático con las siguientes columnas por `cell_id`:

| Columna | Descripción | Fuente |
|---|---|---|
| `cell_id` | Identificador único de la celda | Generado |
| `lat_centroid` | Latitud del centroide | Generado |
| `lon_centroid` | Longitud del centroide | Generado |
| `altitud_media` | Altitud media de la celda (m) | Copernicus DEM GLO-30 |
| `pendiente_media` | Pendiente media (grados) | Copernicus DEM GLO-30 |
| `orientacion` | Orientación del terreno (N/S/E/O) | Copernicus DEM GLO-30 |
| `combustible_clase` | Clase de combustible predominante | CORINE Land Cover |
| `combustible_pct_bosque` | % cobertura forestal en la celda | CORINE Land Cover |

## Fuentes de Datos

- [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)
- [CORINE Land Cover](https://land.copernicus.eu/en/products/corine-land-cover)
- [CNIG/IGN Límites administrativos](https://centrodedescargas.cnig.es/CentroDescargas/index.jsp)
