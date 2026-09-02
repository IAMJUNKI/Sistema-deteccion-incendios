"""
Sistema canadiense de indices de peligro de incendio (FWI)
===========================================================

Implementacion de las ecuaciones de Van Wagner (1974, 1987), tal como se
recogen en el Canadian Forest Fire Danger Rating System (CFFDRS).

POR QUE LO CALCULAMOS EN VEZ DE DESCARGARLO
  IberFire descarga el FWI de CEMS a 27,5 km y lo interpola a 1 km. Nosotros
  ya tenemos temperatura, humedad, viento y lluvia a 1 km, asi que podemos
  calcularlo directamente a NUESTRA resolucion. El resultado es un baseline
  mas exigente que el de la referencia, y ademas evita depender de descargas.

COMPONENTES
  FFMC  humedad del combustible fino (capa superficial, responde en horas)
  DMC   humedad de la capa organica intermedia (responde en dias)
  DC    sequia profunda (responde en semanas/meses)
  ISI   indice de propagacion inicial = f(FFMC, viento)
  BUI   combustible disponible = f(DMC, DC)
  FWI   indice final = f(ISI, BUI)

CARACTER SECUENCIAL
  FFMC, DMC y DC son ACUMULADOS: el valor de hoy depende del de ayer en esa
  misma celda. No se pueden calcular fila a fila de forma independiente.
  Aqui se vectoriza sobre las celdas y se itera solo sobre los dias.

DESVIACION RESPECTO AL ESTANDAR - LEER ANTES DE USAR
  El FWI oficial se define con observaciones de MEDIODIA solar:
  temperatura, humedad y viento a las 12:00 hora local, y lluvia acumulada
  de 24 h.

  Nosotros disponemos de los extremos de la ventana critica 12-18 h:
  tmax_vc (maxima), rhmin_vc (minima) y vmax_vc (racha maxima). Usar
  extremos en lugar de valores de mediodia SOBREESTIMA el FWI de forma
  sistematica.

  Esto es aceptable si el FWI se usa como BASELINE COMPARATIVO (nos interesa
  su capacidad de ordenar celdas por riesgo, no su valor absoluto ni la
  categoria oficial de peligro). Si en algun momento se quieren usar los
  umbrales oficiales de la tabla de peligro, habria que recalcularlo con
  valores de mediodia.

  Debe quedar escrito asi en la memoria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Factores de duracion del dia (hemisferio norte, latitudes medias)
LE_DMC = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
LF_DC = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])

# Valores de arranque estandar (primavera, tras el deshielo)
FFMC_INICIAL, DMC_INICIAL, DC_INICIAL = 85.0, 6.0, 15.0


def _paso_ffmc(ffmc_prev, T, H, W, P):
    """FFMC: humedad del combustible fino. Responde en horas."""
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    # Efecto de la lluvia (solo por encima de 0,5 mm)
    lluvia = P > 0.5
    if np.any(lluvia):
        rf = np.where(lluvia, P - 0.5, 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            delta = (42.5 * rf * np.exp(-100.0 / (251.0 - mo))
                     * (1.0 - np.exp(-6.93 / np.where(rf > 0, rf, 1.0))))
            extra = np.where(mo > 150.0, 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), 0.0)
        mo = np.where(lluvia, np.minimum(mo + delta + extra, 250.0), mo)

    # Equilibrios de desorcion (ed) y adsorcion (ew)
    ed = (0.942 * H ** 0.679 + 11.0 * np.exp((H - 100.0) / 10.0)
          + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H)))
    ew = (0.618 * H ** 0.753 + 10.0 * np.exp((H - 100.0) / 10.0)
          + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H)))

    m = mo.copy()

    # Secado (mo por encima del equilibrio de desorcion)
    seca = mo > ed
    if np.any(seca):
        ko = (0.424 * (1.0 - (H / 100.0) ** 1.7)
              + 0.0694 * np.sqrt(W) * (1.0 - (H / 100.0) ** 8))
        kd = ko * 0.581 * np.exp(0.0365 * T)
        m = np.where(seca, ed + (mo - ed) / (10.0 ** kd), m)

    # Humidificacion (mo por debajo del equilibrio de adsorcion)
    moja = (mo < ed) & (mo < ew)
    if np.any(moja):
        kl = (0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7)
              + 0.0694 * np.sqrt(W) * (1.0 - ((100.0 - H) / 100.0) ** 8))
        kw = kl * 0.581 * np.exp(0.0365 * T)
        m = np.where(moja, ew - (ew - mo) / (10.0 ** kw), m)

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return np.clip(ffmc, 0.0, 101.0)


def _paso_dmc(dmc_prev, T, H, P, mes):
    """DMC: humedad de la capa organica intermedia. Responde en dias."""
    dmc = dmc_prev.copy()

    lluvia = P > 1.5
    if np.any(lluvia):
        re = 0.92 * P - 1.27
        mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
        b = np.where(
            dmc_prev <= 33.0,
            100.0 / (0.5 + 0.3 * dmc_prev),
            np.where(dmc_prev <= 65.0,
                     14.0 - 1.3 * np.log(np.maximum(dmc_prev, 1e-6)),
                     6.2 * np.log(np.maximum(dmc_prev, 1e-6)) - 17.2),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mr = mo + 1000.0 * re / (48.77 + b * re)
            pr = 244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 1e-6))
        dmc = np.where(lluvia, np.maximum(pr, 0.0), dmc)

    le = LE_DMC[mes - 1]
    k = np.where(T > -1.1, 1.894 * (T + 1.1) * (100.0 - H) * le * 1e-6, 0.0)
    return np.maximum(dmc + 100.0 * k, 0.0)


def _paso_dc(dc_prev, T, P, mes):
    """DC: sequia profunda. Responde en semanas o meses."""
    dc = dc_prev.copy()

    lluvia = P > 2.8
    if np.any(lluvia):
        rd = 0.83 * P - 1.27
        qo = 800.0 * np.exp(-dc_prev / 400.0)
        qr = qo + 3.937 * rd
        with np.errstate(divide="ignore", invalid="ignore"):
            dr = 400.0 * np.log(800.0 / np.maximum(qr, 1e-6))
        dc = np.where(lluvia, np.maximum(dr, 0.0), dc)

    lf = LF_DC[mes - 1]
    v = np.where(T > -2.8, 0.36 * (T + 2.8) + lf, lf)
    v = np.maximum(v, 0.0)
    return np.maximum(dc + 0.5 * v, 0.0)


def _isi(ffmc, W):
    """Indice de propagacion inicial: combustible fino seco + viento."""
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 19.115 * np.exp(-0.1386 * mo) * (1.0 + mo ** 5.31 / 49_300_000.0)
    return ff * np.exp(0.05039 * W)


def _bui(dmc, dc):
    """Indice de combustible disponible."""
    denom = dmc + 0.4 * dc
    seguro = np.where(denom > 0, denom, 1e-6)
    bui = np.where(
        dmc <= 0.4 * dc,
        0.8 * dmc * dc / seguro,
        dmc - (1.0 - 0.8 * dc / seguro) * (0.92 + (0.0114 * dmc) ** 1.7),
    )
    return np.maximum(bui, 0.0)


def _fwi(isi, bui):
    """Indice final."""
    fd = np.where(bui <= 80.0,
                  0.626 * np.maximum(bui, 0.0) ** 0.809 + 2.0,
                  1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
    b = 0.1 * isi * fd
    with np.errstate(divide="ignore", invalid="ignore"):
        alto = np.exp(2.72 * (0.434 * np.log(np.maximum(b, 1e-9))) ** 0.647)
    return np.where(b > 1.0, alto, b)


def calcular_fwi(df: pd.DataFrame,
                 col_celda="cell_id", col_fecha="fecha",
                 col_t="tmax_vc", col_h="rhmin_vc",
                 col_w="vmax_vc", col_p="prec_dia",
                 verbose=True) -> pd.DataFrame:
    """
    Calcula la serie completa de indices FWI.

    Entrada: DataFrame con una fila por celda y dia, ordenable por fecha.
             Temperatura en C, humedad en %, viento en km/h, lluvia en mm.
             La meteorologia debe estar SIN desplazar (calendario real): los
             indices son acumulados y necesitan la secuencia verdadera.

    Salida:  el mismo indice de filas mas ffmc, dmc, dc, isi, bui, fwi.

    Reinicio anual: cada 1 de enero se vuelve a los valores de arranque
    estandar. Es la practica habitual del CFFDRS en climas con invierno
    marcado y evita arrastrar estados irreales entre temporadas.
    """
    d = df[[col_celda, col_fecha, col_t, col_h, col_w, col_p]].copy()
    d[col_fecha] = pd.to_datetime(d[col_fecha])

    # Se pivota a matrices (dias x celdas) una sola vez. Filtrar por fecha
    # dentro del bucle seria O(n) por dia y resulta inviable a escala real.
    celdas = np.sort(d[col_celda].unique())
    fechas = np.sort(d[col_fecha].unique())
    nd, nc = len(fechas), len(celdas)
    if verbose:
        print(f"    FWI sobre {nc:,} celdas x {nd:,} dias")

    ic = pd.Series(np.arange(nc), index=celdas).reindex(d[col_celda]).to_numpy()
    idia = pd.Series(np.arange(nd), index=fechas).reindex(d[col_fecha]).to_numpy()

    def matriz(col, minimo=None, maximo=None):
        m = np.full((nd, nc), np.nan, dtype=np.float32)
        v = d[col].to_numpy(dtype=np.float32)
        m[idia, ic] = v
        if minimo is not None or maximo is not None:
            m = np.clip(m, minimo, maximo)
        return m

    T = matriz(col_t)
    H = matriz(col_h, 0.0, 100.0)
    W = matriz(col_w, 0.0, None)
    P = matriz(col_p, 0.0, None)

    ffmc = np.full(nc, FFMC_INICIAL)
    dmc = np.full(nc, DMC_INICIAL)
    dc = np.full(nc, DC_INICIAL)

    out_ffmc = np.empty((nd, nc), dtype=np.float32)
    out_dmc = np.empty((nd, nc), dtype=np.float32)
    out_dc = np.empty((nd, nc), dtype=np.float32)
    out_isi = np.empty((nd, nc), dtype=np.float32)
    out_bui = np.empty((nd, nc), dtype=np.float32)
    out_fwi = np.empty((nd, nc), dtype=np.float32)

    anio_prev = None
    for i in range(nd):
        ts = pd.Timestamp(fechas[i])
        if anio_prev is not None and ts.year != anio_prev:
            ffmc[:], dmc[:], dc[:] = FFMC_INICIAL, DMC_INICIAL, DC_INICIAL
        anio_prev = ts.year
        mes = np.full(nc, ts.month)

        t, h, w, p = T[i], H[i], W[i], P[i]
        ok = np.isfinite(t) & np.isfinite(h) & np.isfinite(w) & np.isfinite(p)
        t = np.nan_to_num(t, nan=0.0).astype(np.float64)
        h = np.nan_to_num(h, nan=50.0).astype(np.float64)
        w = np.nan_to_num(w, nan=0.0).astype(np.float64)
        p = np.nan_to_num(p, nan=0.0).astype(np.float64)

        # Si falta algun dato ese dia, se conserva el estado del dia anterior
        ffmc = np.where(ok, _paso_ffmc(ffmc, t, h, w, p), ffmc)
        dmc = np.where(ok, _paso_dmc(dmc, t, h, p, mes), dmc)
        dc = np.where(ok, _paso_dc(dc, t, p, mes), dc)

        isi = _isi(ffmc, w)
        bui = _bui(dmc, dc)
        out_ffmc[i], out_dmc[i], out_dc[i] = ffmc, dmc, dc
        out_isi[i], out_bui[i], out_fwi[i] = isi, bui, _fwi(isi, bui)

        if verbose and nd > 200 and i % max(1, nd // 10) == 0:
            print(f"      {i / nd:.0%}")

    # Se devuelve solo para los pares (celda, dia) que existian en la entrada
    return pd.DataFrame({
        col_celda: d[col_celda].to_numpy(),
        col_fecha: d[col_fecha].to_numpy(),
        "ffmc": out_ffmc[idia, ic], "dmc": out_dmc[idia, ic],
        "dc": out_dc[idia, ic], "isi": out_isi[idia, ic],
        "bui": out_bui[idia, ic], "fwi": out_fwi[idia, ic],
    })


# Categorias oficiales de peligro (EFFIS / ECMWF)
CATEGORIAS = [(5.2, "muy bajo"), (11.2, "bajo"), (21.3, "moderado"),
              (38.0, "alto"), (50.0, "muy alto"), (np.inf, "extremo")]


def categoria_fwi(v):
    """Traduce un valor de FWI a su categoria oficial de peligro."""
    for umbral, nombre in CATEGORIAS:
        if v < umbral:
            return nombre
    return "extremo"


if __name__ == "__main__":
    # Prueba de humo: una celda, 60 dias, ola de calor y despues tormenta.
    print("Prueba: 30 dias secos y calurosos, luego 30 dias de lluvia\n")
    fechas = pd.date_range("2023-07-01", periods=60)
    seco = np.arange(60) < 30
    prueba = pd.DataFrame({
        "cell_id": 0, "fecha": fechas,
        "tmax_vc": np.where(seco, 34.0, 18.0),
        "rhmin_vc": np.where(seco, 22.0, 85.0),
        "vmax_vc": np.where(seco, 25.0, 12.0),
        "prec_dia": np.where(seco, 0.0, 12.0),
    })
    r = calcular_fwi(prueba, verbose=False)
    for dia in (0, 7, 14, 29, 32, 40, 59):
        f = r.iloc[dia]
        print(f"  dia {dia:2d} ({'seco ' if seco[dia] else 'lluvia'}): "
              f"FFMC {f.ffmc:6.2f}  DMC {f.dmc:7.2f}  DC {f.dc:7.2f}  "
              f"ISI {f.isi:6.2f}  BUI {f.bui:7.2f}  FWI {f.fwi:7.2f}  "
              f"[{categoria_fwi(f.fwi)}]")

    print("\n  Comprobaciones:")
    pico = r.fwi[:30].max()
    tras_lluvia = r.fwi[45:].mean()
    print(f"    FWI maximo en sequia   : {pico:.2f}")
    print(f"    FWI medio tras lluvias : {tras_lluvia:.2f}")
    assert r.fwi[:30].is_monotonic_increasing or pico > r.fwi.iloc[0], \
        "El FWI deberia subir durante la sequia"
    assert tras_lluvia < pico / 2, "El FWI deberia desplomarse tras la lluvia"
    assert (r.ffmc.between(0, 101)).all(), "FFMC fuera de rango"
    assert (r.fwi >= 0).all(), "FWI negativo"
    print("    OK: sube en sequia, cae con la lluvia, rangos validos.")
