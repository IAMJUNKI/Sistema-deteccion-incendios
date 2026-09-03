from datetime import date
from pathlib import Path

import pytest

from src.ingestion.era5 import (
    crear_peticion_mensual,
    iterar_meses,
    localizar_archivo_mensual,
    rutas_archivo_mensual,
)


def test_iterar_meses_incluye_ambos_extremos() -> None:
    assert iterar_meses(date(2018, 12, 1), date(2019, 2, 28)) == [
        (2018, 12),
        (2019, 1),
        (2019, 2),
    ]


def test_iterar_meses_rechaza_intervalo_invertido() -> None:
    with pytest.raises(ValueError):
        iterar_meses(date(2020, 1, 1), date(2019, 12, 31))


def test_peticion_febrero_bisiesto() -> None:
    request = crear_peticion_mensual(2020, 2)
    assert request["day"][0] == "01"
    assert request["day"][-1] == "29"
    assert request["time"] == [f"{hour:02d}:00" for hour in range(24)]
    assert request["area"] == [43.8, -9.3, 41.8, -6.7]


def test_localizar_archivo_mensual_reutiliza_el_nombre_historico(tmp_path: Path) -> None:
    _, historical = rutas_archivo_mensual(tmp_path, 2018, 12)
    historical.write_bytes(b"ERA5")

    assert localizar_archivo_mensual(tmp_path, 2018, 12) == historical


def test_localizar_archivo_mensual_rechaza_dos_copias(tmp_path: Path) -> None:
    canonical, historical = rutas_archivo_mensual(tmp_path, 2018, 12)
    canonical.write_bytes(b"ERA5 nuevo")
    historical.write_bytes(b"ERA5 historico")

    with pytest.raises(ValueError, match="dos archivos ERA5-Land"):
        localizar_archivo_mensual(tmp_path, 2018, 12)
