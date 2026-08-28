"""Lectura del contrato del dataset EGIF.

Este módulo es la única puerta de entrada al esquema. Todo lo demás pregunta aquí en vez de
llevar listas de columnas escritas a mano, por una razón operativa: **el dataset todavía no es
el definitivo**. Faltan por incorporar `vpd_mean`, `vpd_max_12_18h` y las variables OSM
desagregadas. Cuando lleguen, el pipeline debe recogerlas sin que haya que editar código.

Por eso los grupos de variables se resuelven por patrón sobre el nombre y no por enumeración:
una variable nueva llamada `precipitation_sum_60d` cae sola en el grupo meteorológico, y una
`road_length_main_km` cae sola en actividad humana.

El contrato también fija qué columnas **no** pueden ser predictoras. Esa lista viene del propio
`metadata.json` (`outcome_columns_not_predictors` y `sampling_auxiliary_columns_not_predictors`)
y se refuerza aquí, de modo que una fuga de información tendría que superar dos barreras
independientes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
DIR_DATASET = RAIZ / "data" / "processed" / "tabular" / "egif"

COL_CELDA = "cell_id"
COL_FECHA = "fecha"
COL_TARGET = "target_ignicion"

#: Nunca pueden entrar como predictoras, se declaren como se declaren en el metadato.
#: `is_near_ignition_25x25_10d` es el caso delicado: describe si hubo una ignición cerca en los
#: diez días anteriores, así que usarla como predictor sería mirar la respuesta.
PROHIBIDAS = frozenset({
    COL_CELDA, COL_FECHA, COL_TARGET,
    "x", "y", "is_galicia", "year",
    "burned_area_ha", "large_fire_500ha", "is_near_ignition_25x25_10d",
})

#: Patrones que asignan cada variable a su grupo temático. El orden importa: se aplica el
#: primero que encaje.
PATRONES_GRUPO: tuple[tuple[str, str], ...] = (
    (r"^(temperature|relative_humidity|wind_speed|precipitation|vpd|consecutive_dry)", "meteorologia"),
    (r"^(elevation|slope|roughness|aspect)", "topografia"),
    (r"^(artificial|agriculture|broadleaf|coniferous|mixed_forest|scrub|open_spaces|wetlands|water|forest_cover)", "cobertura"),
    (r"^(distance_to|road_length|residential_area|building_area)", "actividad_humana"),
    (r"^(month|day_of|iso_week|is_weekend|year)", "calendario"),
)

#: Variables meteorológicas base sobre las que se calculan anomalías y z-scores diarios.
#: Se resuelven contra el dataset real: si alguna no está, simplemente no se genera su derivada.
BASE_DERIVADAS = ("temperature_max", "relative_humidity_min", "wind_speed_max", "precipitation_sum_7d")


def clasificar(variable: str) -> str:
    """Asigna una variable a su grupo temático a partir del nombre."""
    for patron, grupo in PATRONES_GRUPO:
        if re.match(patron, variable):
            return grupo
    return "otras"


@dataclass(frozen=True)
class Contrato:
    """Esquema del dataset tal y como está en disco, no como esperábamos que estuviera."""

    directorio: Path
    predictores: list[str]
    filas_por_anio: dict[int, int]
    contrato_temporal: Optional[str]
    filas_descartadas: int
    grupos: dict[str, list[str]] = field(default_factory=dict)

    @property
    def anios(self) -> list[int]:
        return sorted(self.filas_por_anio)

    def ruta(self, anio: int) -> Path:
        """Ruta del Parquet anual, comprobando que existe."""
        destino = self.directorio / f"year={anio}" / f"dataset_{anio}.parquet"
        if not destino.exists():
            raise FileNotFoundError(f"Falta la partición anual: {destino}")
        return destino

    def rutas(self, anios: Iterable[int]) -> list[Path]:
        return [self.ruta(a) for a in anios]

    def filas(self, anios: Iterable[int]) -> int:
        """Filas totales esperadas, para el guardián de cobertura."""
        return sum(self.filas_por_anio[a] for a in anios)

    def tiene(self, *variables: str) -> bool:
        disponibles = set(self.predictores)
        return all(v in disponibles for v in variables)

    def del_grupo(self, *grupos: str) -> list[str]:
        return sorted(v for g in grupos for v in self.grupos.get(g, []))

    def resumen(self) -> str:
        lineas = [f"Contrato EGIF — {len(self.predictores)} predictores, "
                  f"{self.filas(self.anios):,} filas en {self.anios}"]
        for grupo in sorted(self.grupos):
            lineas.append(f"  {grupo:<18} {len(self.grupos[grupo]):>3} variables")
        if self.contrato_temporal:
            lineas.append(f"  contrato temporal: {self.contrato_temporal}")
        return "\n".join(lineas)


def cargar(directorio: str | Path = DIR_DATASET, verificar_esquema: bool = True) -> Contrato:
    """Lee `metadata.json` y lo valida contra los Parquet que hay realmente en disco.

    Args:
        directorio: Carpeta que contiene `metadata.json` y las particiones `year=YYYY/`.
        verificar_esquema: Si es cierto, abre el primer Parquet y comprueba que las columnas
            declaradas existen de verdad. Cuesta milisegundos y evita descubrir una
            discrepancia a mitad de un entrenamiento de una hora.

    Raises:
        FileNotFoundError: Si falta el metadato.
        ValueError: Si el metadato declara predictores prohibidos o el esquema no cuadra.
    """
    directorio = Path(directorio)
    ruta_meta = directorio / "metadata.json"
    if not ruta_meta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_meta}.\n"
            "El dataset EGIF debe estar en data/processed/tabular/egif/ con la estructura "
            "year=YYYY/dataset_YYYY.parquet."
        )

    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    declarados = list(meta["predictor_columns"])

    coladas = sorted(set(declarados) & PROHIBIDAS)
    if coladas:
        raise ValueError(
            f"El metadato declara como predictoras columnas prohibidas: {coladas}. "
            "Son resultados del incendio o identificadores: usarlas sería fuga de información."
        )

    predictores = sorted(declarados)
    filas_por_anio = {int(a): int(n) for a, n in meta.get("annual_files", {}).items()}

    if verificar_esquema and filas_por_anio:
        primer_anio = min(filas_por_anio)
        esquema = pq.ParquetFile(directorio / f"year={primer_anio}" / f"dataset_{primer_anio}.parquet")
        reales = set(esquema.schema_arrow.names)
        esquema.close()
        ausentes = sorted(set(predictores) - reales)
        if ausentes:
            raise ValueError(
                f"El metadato declara variables que no están en dataset_{primer_anio}.parquet: "
                f"{ausentes}"
            )
        if COL_TARGET not in reales:
            raise ValueError(f"El Parquet no contiene la columna objetivo {COL_TARGET!r}.")

    grupos: dict[str, list[str]] = {}
    for variable in predictores:
        grupos.setdefault(clasificar(variable), []).append(variable)

    contrato = Contrato(
        directorio=directorio,
        predictores=predictores,
        filas_por_anio=filas_por_anio,
        contrato_temporal=meta.get("time_contract"),
        filas_descartadas=int(meta.get("dropped_rows_incomplete_predictors", 0)),
        grupos=grupos,
    )

    sin_clasificar = grupos.get("otras", [])
    if sin_clasificar:
        logger.warning(
            "Variables sin grupo temático (revisa PATRONES_GRUPO): %s", sin_clasificar
        )
    if contrato.filas_descartadas:
        logger.warning(
            "La exportación descartó %s filas por predictores incompletos.",
            f"{contrato.filas_descartadas:,}",
        )
    return contrato
