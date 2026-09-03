from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.ingestion.fwi import _agrupar_fechas_por_mes, _cubre_intervalo


def _crear_fwi(path: Path, dates: pd.DatetimeIndex) -> None:
    dataset = xr.Dataset(
        {"fwinx": ("valid_time", np.zeros(len(dates), dtype=np.float32))},
        coords={"valid_time": dates},
    )
    dataset.to_netcdf(path)


def test_cubre_intervalo_exige_todos_los_dias(tmp_path: Path) -> None:
    path = tmp_path / "fwi_galicia_2023.nc"
    _crear_fwi(path, pd.date_range("2023-01-01", "2023-01-02", freq="D"))

    assert _cubre_intervalo(path, date(2023, 1, 1), date(2023, 1, 2))
    assert not _cubre_intervalo(path, date(2023, 1, 1), date(2023, 1, 3))


def test_agrupar_fechas_por_mes_evitar_dias_invalidos() -> None:
    grouped = _agrupar_fechas_por_mes(
        [date(2023, 2, 27), date(2023, 2, 28), date(2023, 3, 1), date(2023, 3, 31)]
    )

    assert grouped == {
        2: [date(2023, 2, 27), date(2023, 2, 28)],
        3: [date(2023, 3, 1), date(2023, 3, 31)],
    }
