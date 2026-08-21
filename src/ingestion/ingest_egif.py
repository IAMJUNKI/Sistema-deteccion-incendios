"""
Modulo de Ingesta y Procesamiento de la Estadística General de Incendios Forestales (EGIF - MITECO).

Convierte los registros oficiales de incendios forestales de Galicia (provincias 15, 27, 32, 36)
en la variable objetivo (Target Y ∈ {0, 1}) mapeada a la rejilla espacial 1km x 1km.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# Provincias de Galicia en la codificación oficial del MITECO
GALICIA_PROVINCES = {"15", "27", "32", "36", 15, 27, 32, 36}


def parse_egif_xml(xml_path: str | Path) -> pd.DataFrame:
    """
    Parsea el archivo XML del EGIF y extrae los incendios forestales ocurridos en Galicia
    con fecha de inicio (detección), coordenadas y superficie total afectada.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo EGIF XML en: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for pif in root.findall(".//Pif"):
        loc = pif.find("pif_localizacion")
        tiempos = pif.find("pif_tiempos")
        perdidas = pif.find("pif_perdidas")

        if loc is None or tiempos is None:
            continue

        prov = loc.findtext("idprovincia", "").strip()
        if prov in GALICIA_PROVINCES or (prov.isdigit() and int(prov) in GALICIA_PROVINCES):
            fecha_str = tiempos.findtext("deteccion", "")
            lat_str = loc.findtext("latitud", "")
            lon_str = loc.findtext("longitud", "")

            sup_arbolada = float(perdidas.findtext("superficiearboladatotal", "0") or 0) if perdidas is not None else 0.0
            sup_no_arbolada = float(perdidas.findtext("superficienoarboladatotal", "0") or 0) if perdidas is not None else 0.0
            sup_total = sup_arbolada + sup_no_arbolada

            if fecha_str and lat_str and lon_str:
                try:
                    lat = float(lat_str.replace(",", "."))
                    lon = float(lon_str.replace(",", "."))
                    fecha = pd.to_datetime(fecha_str, errors="coerce")

                    if pd.notnull(fecha) and lat != 0.0 and lon != 0.0:
                        records.append({
                            "fecha": fecha.strftime("%Y-%m-%d"),
                            "latitud": lat,
                            "longitud": lon,
                            "superficie_ha": sup_total,
                            "es_ignicion_egif": 1
                        })
                except (ValueError, AttributeError):
                    continue

    df_fires = pd.DataFrame(records)
    return df_fires


def map_fires_to_grid(df_fires: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Realiza el cruce espacial entre los puntos de ignición del EGIF y las celdas
    de 1km x 1km de la rejilla de Galicia.
    """
    if df_fires.empty:
        return pd.DataFrame(columns=["cell_id", "fecha", "target_ignicion"])

    geometry = [Point(xy) for xy in zip(df_fires["longitud"], df_fires["latitud"])]
    gdf_fires = gpd.GeoDataFrame(df_fires, geometry=geometry, crs="EPSG:4326")

    target_crs = grid_gdf.crs
    gdf_fires = gdf_fires.to_crs(target_crs)

    joined = gpd.sjoin(gdf_fires, grid_gdf[["cell_id", "geometry"]], how="inner", predicate="within")

    target_df = (
        joined.groupby(["cell_id", "fecha"])["es_ignicion_egif"]
        .max()
        .reset_index()
        .rename(columns={"es_ignicion_egif": "target_ignicion"})
    )

    return target_df


if __name__ == "__main__":
    print("Módulo ingest_egif listo para importar.")
