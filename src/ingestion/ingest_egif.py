"""Ingesta reproducible del histórico oficial EGIF de incendios de Galicia."""

import argparse
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

GALICIA_PROVINCES = {"15", "27", "32", "36"}
DEFAULT_RAW_DIR = Path("data/raw/fire_history")
DEFAULT_EVENTS_OUTPUT = Path("data/processed/target/egif_events_2018_2023.gpkg")
DEFAULT_TARGET_OUTPUT = Path("data/processed/target/egif_target_2018_2023.parquet")
DEFAULT_METADATA_OUTPUT = Path("data/processed/target/egif_target_2018_2023_metadata.json")
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2023-12-31"
EVENT_COLUMNS = [
    "egif_id",
    "fecha",
    "anio",
    "provincia",
    "municipio",
    "causa",
    "superficie_ha",
    "utm_zone",
    "utm_x",
    "utm_y",
    "latitud",
    "longitud",
    "fuente_coordenadas",
    "geometry",
]


def _local_name(tag: str) -> str:
    """Devuelve una etiqueta XML sin su posible espacio de nombres."""
    return tag.rsplit("}", maxsplit=1)[-1].lower()


def _find_text(element: ET.Element, tag: str) -> str | None:
    """Busca la primera etiqueta descendiente, admitiendo espacios de nombres XML."""
    for child in element.iter():
        if _local_name(child.tag) == tag.lower():
            return child.text.strip() if child.text else None
    return None


def _to_float(value: str | None) -> float:
    """Convierte un número EGIF, que puede usar coma decimal, a ``NaN``."""
    if value is None:
        return float("nan")
    return float(value.replace(",", "."))


def _event_record(pif: ET.Element) -> dict[str, Any] | None:
    """Extrae un registro EGIF y descarta incendios fuera de Galicia o inválidos."""
    province = _find_text(pif, "idprovincia")
    if province not in GALICIA_PROVINCES:
        return None

    try:
        detected_at = pd.to_datetime(_find_text(pif, "deteccion"), errors="coerce")
        latitude = _to_float(_find_text(pif, "latitud"))
        longitude = _to_float(_find_text(pif, "longitud"))
        utm_x = _to_float(_find_text(pif, "x"))
        utm_y = _to_float(_find_text(pif, "y"))
        utm_zone = int(_to_float(_find_text(pif, "huso")))
        forest_area = _to_float(_find_text(pif, "superficiearboladatotal"))
        non_forest_area = _to_float(_find_text(pif, "superficienoarboladatotal"))
    except (TypeError, ValueError):
        return None

    if pd.isna(detected_at):
        return None
    return {
        "egif_id": _find_text(pif, "idpif"),
        "fecha": detected_at.normalize(),
        "anio": int(detected_at.year),
        "provincia": province,
        "municipio": _find_text(pif, "idmunicipio"),
        "causa": _find_text(pif, "idcausa"),
        "superficie_ha": np.nansum([forest_area, non_forest_area]),
        "utm_zone": utm_zone,
        "utm_x": utm_x,
        "utm_y": utm_y,
        "latitud": latitude,
        "longitud": longitude,
    }


def parse_egif_xml(xml_path: str | Path) -> pd.DataFrame:
    """Parsea el XML EGIF y devuelve los registros gallegos sin geometría.

    La función no presupone que el primer hijo del XML sea un incendio: el
    fichero oficial incorpora un esquema XSD como primer elemento.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo EGIF XML en: {xml_path}")

    root = ET.parse(xml_path).getroot()
    records = [
        record
        for pif in root.iter()
        if _local_name(pif.tag) == "pif"
        if (record := _event_record(pif)) is not None
    ]
    return pd.DataFrame(records)


def crear_geodatos_incendios(fires: pd.DataFrame) -> gpd.GeoDataFrame:
    """Crea puntos EPSG:3035, usando UTM como respaldo de las coordenadas geográficas."""
    if fires.empty:
        return gpd.GeoDataFrame(columns=EVENT_COLUMNS, geometry="geometry", crs="EPSG:3035")

    events = fires.copy()
    is_wgs84 = events["latitud"].between(35, 44) & events["longitud"].between(-19, 5)
    geodata_parts: list[gpd.GeoDataFrame] = []

    if is_wgs84.any():
        geographic = events.loc[is_wgs84].copy()
        geographic["fuente_coordenadas"] = "wgs84"
        geodata_parts.append(
            gpd.GeoDataFrame(
                geographic,
                geometry=gpd.points_from_xy(geographic["longitud"], geographic["latitud"]),
                crs="EPSG:4326",
            ).to_crs("EPSG:3035")
        )

    utm_events = events.loc[~is_wgs84].copy()
    valid_utm = (
        utm_events["utm_zone"].isin([29, 30])
        & utm_events["utm_x"].notna()
        & utm_events["utm_y"].notna()
    )
    if valid_utm.any():
        utm_events = utm_events.loc[valid_utm].copy()
        geometry = []
        for _, row in utm_events.iterrows():
            transformer = Transformer.from_crs(
                f"EPSG:{25800 + int(row['utm_zone'])}", "EPSG:3035", always_xy=True
            )
            x, y = transformer.transform(row["utm_x"], row["utm_y"])
            geometry.append(Point(x, y))
        utm_events["fuente_coordenadas"] = "utm"
        geodata_parts.append(gpd.GeoDataFrame(utm_events, geometry=geometry, crs="EPSG:3035"))

    if not geodata_parts:
        return gpd.GeoDataFrame(columns=EVENT_COLUMNS, geometry="geometry", crs="EPSG:3035")
    return gpd.GeoDataFrame(pd.concat(geodata_parts, ignore_index=True), crs="EPSG:3035")[
        EVENT_COLUMNS
    ]


def map_fires_to_grid(fires_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Asigna incendios a celdas activas y crea un target único por celda y día."""
    columns = ["cell_id", "fecha", "target_ignicion", "n_incendios", "superficie_ha"]
    if fires_gdf.empty:
        return pd.DataFrame(columns=columns)

    grid = grid_gdf.loc[grid_gdf["is_galicia"] == 1, ["cell_id", "geometry"]]
    joined = gpd.sjoin(fires_gdf.to_crs(grid.crs), grid, how="inner", predicate="within")
    target = (
        joined.groupby(["cell_id", "fecha"], as_index=False)
        .agg(n_incendios=("egif_id", "size"), superficie_ha=("superficie_ha", "sum"))
        .assign(target_ignicion=1)
    )
    return target[["cell_id", "fecha", "target_ignicion", "n_incendios", "superficie_ha"]]


def procesar_egif(
    xml_path: str | Path,
    grid_path: str | Path,
    events_output_path: str | Path = DEFAULT_EVENTS_OUTPUT,
    target_output_path: str | Path = DEFAULT_TARGET_OUTPUT,
    metadata_output_path: str | Path = DEFAULT_METADATA_OUTPUT,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> dict[str, Any]:
    """Procesa EGIF, guarda eventos auditables y el target diario por celda."""
    fires = parse_egif_xml(xml_path)
    fires = fires.loc[fires["fecha"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
    fires_gdf = crear_geodatos_incendios(fires)
    grid = gpd.read_file(grid_path)
    target = map_fires_to_grid(fires_gdf, grid)

    events_output_path = Path(events_output_path)
    target_output_path = Path(target_output_path)
    metadata_output_path = Path(metadata_output_path)
    for path in (events_output_path, target_output_path, metadata_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    fires_gdf.to_file(events_output_path, driver="GPKG")
    target.to_parquet(target_output_path, index=False)

    last_event_date = fires["fecha"].max().date().isoformat() if not fires.empty else None
    metadata = {
        "source": "EGIF-MITECO XML",
        "requested_period": {"start": start_date, "end": end_date},
        "available_event_period": {
            "start": fires["fecha"].min().date().isoformat() if not fires.empty else None,
            "end": last_event_date,
        },
        "n_events_galicia": int(len(fires_gdf)),
        "n_positive_cell_days": int(len(target)),
        "warning": (
            "The source ends before the requested period. Do not label subsequent dates as negatives."
            if last_event_date and last_event_date < end_date
            else None
        ),
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metadata["warning"]:
        warnings.warn(metadata["warning"], stacklevel=2)
    return metadata


def main() -> None:
    """Ejecuta la ingesta EGIF desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Genera eventos y target diario desde EGIF.")
    parser.add_argument("--xml", required=True, help="Ruta del XML descargado de EGIF.")
    parser.add_argument("--grid", required=True, help="Rejilla vectorial de 1 km.")
    parser.add_argument("--events-output", default=str(DEFAULT_EVENTS_OUTPUT))
    parser.add_argument("--target-output", default=str(DEFAULT_TARGET_OUTPUT))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA_OUTPUT))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args()
    print(
        procesar_egif(
            args.xml,
            args.grid,
            args.events_output,
            args.target_output,
            args.metadata_output,
            args.start_date,
            args.end_date,
        )
    )


if __name__ == "__main__":
    main()
