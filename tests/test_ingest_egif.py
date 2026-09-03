import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.ingestion.ingest_egif import crear_geodatos_incendios, map_fires_to_grid, parse_egif_xml


def test_parsear_egif_omite_esquema_y_filtra_galicia(tmp_path) -> None:
    xml_path = tmp_path / "egif.xml"
    xml_path.write_text(
        "<pifs><schema /><Pif><idpif>1</idpif><pif_localizacion>"
        "<idprovincia>15</idprovincia><huso>29</huso><x>550000</x><y>4800000</y>"
        "<latitud>42.0</latitud><longitud>-8.0</longitud></pif_localizacion>"
        "<pif_tiempos><deteccion>2019-01-02 15:00</deteccion></pif_tiempos>"
        "<pif_perdidas><superficiearboladatotal>1,5</superficiearboladatotal>"
        "<superficienoarboladatotal>0,5</superficienoarboladatotal></pif_perdidas></Pif>"
        "<Pif><pif_localizacion><idprovincia>24</idprovincia></pif_localizacion></Pif></pifs>",
        encoding="utf-8",
    )

    fires = parse_egif_xml(xml_path)

    assert len(fires) == 1
    assert fires.loc[0, "fecha"] == pd.Timestamp("2019-01-02")
    assert fires.loc[0, "superficie_ha"] == 2.0


def test_parsear_carpeta_egif_combina_xml_y_deduplica_por_identificador(tmp_path) -> None:
    """Los intervalos EGIF solapados se unen sin duplicar el mismo incendio."""
    folder = tmp_path / "fire_history"
    folder.mkdir()

    def xml_event(identifier: str, detected_at: str, forest_area: str) -> str:
        return (
            "<Pif><idpif>"
            f"{identifier}</idpif><pif_localizacion><idprovincia>15</idprovincia>"
            "<huso>29</huso><x>550000</x><y>4800000</y><latitud>42.0</latitud>"
            "<longitud>-8.0</longitud></pif_localizacion><pif_tiempos><deteccion>"
            f"{detected_at}</deteccion></pif_tiempos><pif_perdidas><superficiearboladatotal>"
            f"{forest_area}</superficiearboladatotal><superficienoarboladatotal>0</superficienoarboladatotal>"
            "</pif_perdidas></Pif>"
        )

    (folder / "egif_2016.xml").write_text(
        f"<pifs>{xml_event('old', '2016-01-02 15:00', '1')}</pifs>", encoding="utf-8"
    )
    (folder / "egif_2018.xml").write_text(
        "<pifs>"
        + xml_event("old", "2016-01-02 15:00", "2")
        + xml_event("new", "2018-01-03 15:00", "3")
        + xml_event("alternative_id", "2018-01-03 15:00", "3")
        + "</pifs>",
        encoding="utf-8",
    )

    fires = parse_egif_xml(folder)

    assert fires["egif_id"].tolist() == ["old", "alternative_id"]
    assert fires["superficie_ha"].tolist() == [2.0, 3.0]


def test_asignar_incendio_a_rejilla_activa() -> None:
    fires = pd.DataFrame(
        {
            "egif_id": ["1"],
            "fecha": [pd.Timestamp("2019-01-02")],
            "anio": [2019],
            "provincia": ["15"],
            "municipio": ["15001"],
            "causa": ["1"],
            "superficie_ha": [2.0],
            "utm_zone": [29],
            "utm_x": [550000.0],
            "utm_y": [4800000.0],
            "latitud": [42.0],
            "longitud": [-8.0],
        }
    )
    fires_gdf = crear_geodatos_incendios(fires)
    point = fires_gdf.geometry.iloc[0]
    grid = gpd.GeoDataFrame(
        {"cell_id": [7], "is_galicia": [1]},
        geometry=[box(point.x - 100, point.y - 100, point.x + 100, point.y + 100)],
        crs="EPSG:3035",
    )

    target = map_fires_to_grid(fires_gdf, grid)

    assert target.to_dict("records") == [
        {
            "cell_id": 7,
            "fecha": pd.Timestamp("2019-01-02"),
            "target_ignicion": 1,
            "n_incendios": 1,
            "superficie_ha": 2.0,
        }
    ]
