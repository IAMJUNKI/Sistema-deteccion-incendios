from pathlib import Path

import pytest

from src.ingestion.pipeline import ejecutar_pipeline_egif, ejecutar_pipeline_meteorologia


def test_ejecutar_pipeline_procesa_e_interpola(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "src.ingestion.pipeline.procesar_era5", lambda *args: calls.append(("daily", *args))
    )
    monkeypatch.setattr(
        "src.ingestion.pipeline.interpolar_al_grid", lambda *args: calls.append(("grid", *args))
    )

    ejecutar_pipeline_meteorologia("static.nc", "grid.gpkg", raw_dir="raw", daily_output_path="daily.nc")

    assert calls[0] == ("daily", "raw", Path("daily.nc"), "2018-12-01", "2023-12-31")
    assert calls[1] == (
        "grid",
        Path("daily.nc"),
        "static.nc",
        "grid.gpkg",
        Path("data/processed/meteorology_2018_2023.nc"),
        "2018-12-01",
        "2023-12-31",
    )


def test_ejecutar_pipeline_no_omite_diario_inexistente() -> None:
    with pytest.raises(FileNotFoundError, match="No se puede omitir"):
        ejecutar_pipeline_meteorologia(
            "static.nc", "grid.gpkg", daily_output_path="missing_daily.nc", skip_daily=True
        )


def test_ejecutar_pipeline_egif_delega_en_el_procesador(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "src.ingestion.pipeline.procesar_egif", lambda *args: calls.append(args) or {"ok": True}
    )

    result = ejecutar_pipeline_egif("egif.xml", "grid.gpkg")

    assert result == {"ok": True}
    assert calls[0][0:2] == ("egif.xml", "grid.gpkg")
