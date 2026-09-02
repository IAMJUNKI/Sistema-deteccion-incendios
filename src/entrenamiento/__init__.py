"""Herramientas de análisis y modelado sobre el datacubo EGIF.

La ruta de producción del modelo 2D está en
``src.models.canonical_training`` y utiliza el contrato oficial
``src.features.canonical_contract``. Los módulos de este paquete se conservan
para reproducir análisis y selección de variables, pero su contrato flexible
no sustituye al esquema canónico congelado de producción.

Paquete separado de `src/modeling/`, que es el pipeline común de selección de variables del
equipo. La separación es deliberada: evita conflictos de merge y deja claro qué pertenece a
cada línea de trabajo.

Recorrido de los módulos, en el orden en que se ejecutan:

1. `contrato`   — lee `metadata.json` y resuelve el esquema por patrón, para que el pipeline
                  sobreviva a que el dataset gane columnas.
2. `datos`      — muestrea negativos al entrenar; recorre la población completa al evaluar.
3. `derivadas`  — anomalías climatológicas y z-scores espaciales diarios.
4. `seleccion`  — señal univariante, redundancia, permutación y ablación por grupos.
5. `modelos`    — los cuatro estimadores tras una interfaz común.
6. `calibracion`— corrección de prior más isotónica, para que la probabilidad signifique algo.
7. `metricas`   — recall a coste operativo fijo, recall diario e intervalos por bootstrap.
8. `experimento`— orquesta lo anterior y vigila cobertura y test ciego.
"""

__all__ = [
    "calibracion",
    "contrato",
    "datos",
    "derivadas",
    "experimento",
    "metricas",
    "modelos",
    "seleccion",
]
