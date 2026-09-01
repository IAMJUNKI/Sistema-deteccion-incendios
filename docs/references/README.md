# Documentación y materiales de referencia

Esta carpeta conserva los materiales que fundamentan la integración del modelo 2D EGIF y la previsión meteorológica operativa:

- `API_MeteoSIX_v5_es.pdf`: documentación de la API v5 de MeteoGalicia.
- `IberFire_2505.00837v2.pdf`: trabajo de IberFire compartido para comparar objetivos y metodología.
- `Datacube_memoria.ipynb`: notebook de construcción o exploración del datacubo.
- `modelado_2d_egif.html`: material del modelado 2D EGIF.

Estos ficheros son referencias del proyecto, no entradas directas de la inferencia diaria. El código operativo utiliza la interfaz de proveedores meteorológicos y el contrato canónico documentados en `docs/explanations/`.

Los materiales se versionan para que el razonamiento sea auditable. Los datasets y el NetCDF asociados se mantienen fuera de Git por su tamaño y están en `data/external/egif/` cuando se dispone de una copia local.
