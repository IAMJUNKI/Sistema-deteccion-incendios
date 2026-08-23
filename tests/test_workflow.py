from pathlib import Path

from src.workflow import _limpiar_salidas_dinamicas


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
