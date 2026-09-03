import json
from pathlib import Path

import pytest

from src.workflow import (
    PeriodoCubo,
    _guardar_manifiesto_periodo,
    _limpiar_salidas_dinamicas,
    _validar_entrada_egif,
)


def test_limpiar_salidas_dinamicas_no_toca_entradas(tmp_path: Path) -> None:
    meteorology = tmp_path / "meteorology.nc"
    datacube = tmp_path / "datacube.nc"
    output = tmp_path / "tabular"
    source = tmp_path / "era5_raw.nc"
    for path in (meteorology, datacube, source):
        path.write_text("test")
    output.mkdir()
    (output / "dataset.parquet").write_text("test")

    _limpiar_salidas_dinamicas(meteorology, datacube, output)

    assert not meteorology.exists()
    assert not datacube.exists()
    assert not output.exists()
    assert source.exists()


def test_periodo_anual_no_depende_de_las_fechas_de_los_incendios() -> None:
    periodo = PeriodoCubo(2017, 2023)

    assert periodo.start_date.isoformat() == "2017-01-01"
    assert periodo.end_date.isoformat() == "2023-12-31"
    assert periodo.meteorology_context_start.isoformat() == "2016-12-01"
    assert periodo.egif_coverage == "full_calendar_years"


def test_periodo_parcial_exige_fecha_final_del_mismo_ano() -> None:
    with pytest.raises(ValueError, match="requiere --final-date"):
        PeriodoCubo(2023, 2024, include_partial_final_year=True)

    periodo = PeriodoCubo(2023, 2024, include_partial_final_year=True, final_date="2024-08-15")
    assert periodo.end_date.isoformat() == "2024-08-15"


def test_validar_entrada_egif_acepta_una_carpeta_con_varios_xml(tmp_path: Path) -> None:
    (tmp_path / "2019.xml").write_text("<pifs />", encoding="utf-8")
    (tmp_path / "2020.xml").write_text("<pifs />", encoding="utf-8")

    assert _validar_entrada_egif(tmp_path) == tmp_path


def test_manifiesto_separa_periodo_declarado_y_estado(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    _guardar_manifiesto_periodo(PeriodoCubo(2019, 2023), manifest, status="completed")

    content = json.loads(manifest.read_text(encoding="utf-8"))
    assert content["status"] == "completed"
    assert content["period"]["end_date"] == "2023-12-31"
