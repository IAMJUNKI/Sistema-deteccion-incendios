"""Recalcula por fuerza bruta lo que dice el contexto, sobre los datos reales.

Las pruebas unitarias demuestran la ausencia de fuga sobre mallas sintéticas, donde la respuesta
correcta se conoce de antemano. Esto comprueba lo mismo contra el dataset de verdad y por un
camino completamente distinto: contando igniciones una a una desde la tabla cruda, sin imágenes
integrales ni sumas acumuladas.

Un resultado de +5,67 pp en un problema con prevalencia de 1,5·10⁻⁴ es exactamente el tipo de
cifra que suele venir de una fuga, así que verificarla por un segundo algoritmo no es
ceremonial.

Uso:
    python archive/vecindad/verificar_sin_fuga.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.entrenamiento import contrato as mc, vecindad  # noqa: E402

ANIOS = [2019, 2020, 2021, 2022]
c = mc.cargar()

# ─── Contexto, igual que en el script ───────────────────────────────────────────
COLS = ["x", "y", "fecha", "target_ignicion",
        "broadleaf_forest", "coniferous_forest", "mixed_forest"]
ds = pads.dataset([str(p) for p in c.rutas(ANIOS)], format="parquet")
pos = ds.to_table(columns=COLS, filter=(pads.field("target_ignicion") == 1)).to_pandas()

bordes = []
for anio, ext in ((min(ANIOS), "min"), (max(ANIOS), "max")):
    uno = pads.dataset([str(c.ruta(anio))], format="parquet")
    f = uno.to_table(columns=["fecha"]).to_pandas()["fecha"]
    lim = f.min() if ext == "min" else f.max()
    bordes.append(uno.to_table(columns=COLS,
                               filter=(pads.field("fecha") == lim)).to_pandas())

marco = pd.concat([pos, *bordes], ignore_index=True).drop_duplicates(
    subset=["x", "y", "fecha"], keep="first", ignore_index=True)
ctx = vecindad.ajustar_contexto_espacial(marco, radios=(5, 12), ventanas=(7, 30))

# ─── Tabla cruda de igniciones, para contar a mano ──────────────────────────────
ig = pos.drop_duplicates(subset=["x", "y", "fecha"])
ix = ig.x.to_numpy(np.float64)
iy = ig.y.to_numpy(np.float64)
idia = pd.to_datetime(ig.fecha).to_numpy("datetime64[D]").astype(np.int64)
print(f"igniciones unicas 2019-2022: {len(ig):,}")

# ─── Muestra de filas del ano de validacion ─────────────────────────────────────
val = pads.dataset([str(c.ruta(2022))], format="parquet")
muestra = val.to_table(columns=["x", "y", "fecha", "target_ignicion"]).to_pandas()
rng = np.random.default_rng(7)
# Mitad positivos (donde una fuga se notaria) y mitad negativos al azar.
sel = pd.concat([
    muestra[muestra.target_ignicion == 1].sample(400, random_state=7),
    muestra[muestra.target_ignicion == 0].sample(400, random_state=7),
], ignore_index=True)

calculado = vecindad.anadir_vecindad(sel, ctx)
sx = sel.x.to_numpy(np.float64)
sy = sel.y.to_numpy(np.float64)
sd = pd.to_datetime(sel.fecha).to_numpy("datetime64[D]").astype(np.int64)

print("\ncomprobando 800 filas por fuerza bruta...\n")
fallos = 0
for radio in (5, 12):
    metros = radio * 1000.0
    for ventana in (7, 30):
        esperado = np.empty(len(sel), dtype=np.int64)
        for i in range(len(sel)):
            # Ventana de Chebyshev (cuadrada), que es la que usa la imagen integral,
            # y estrictamente [D-ventana, D-1].
            dentro = (np.abs(ix - sx[i]) <= metros + 1) & (np.abs(iy - sy[i]) <= metros + 1)
            dt = sd[i] - idia
            esperado[i] = int((dentro & (dt >= 1) & (dt <= ventana)).sum())

        obtenido = calculado[f"igniciones_{radio}km_{ventana}d"].to_numpy(np.int64)
        malas = int((esperado != obtenido).sum())
        fallos += malas
        print(f"  igniciones_{radio}km_{ventana}d:  "
              f"{'OK' if malas == 0 else f'{malas} DISCREPANCIAS'}   "
              f"(media {obtenido.mean():.3f}, max {obtenido.max()})")

    # dias desde la ultima, tambien hasta D-1
    esperado = np.empty(len(sel), dtype=np.int64)
    for i in range(len(sel)):
        dentro = (np.abs(ix - sx[i]) <= metros + 1) & (np.abs(iy - sy[i]) <= metros + 1)
        dt = sd[i] - idia
        previas = dt[dentro & (dt >= 1)]
        esperado[i] = min(int(previas.min()), 90) if len(previas) else 90
    obtenido = calculado[f"dias_desde_ignicion_{radio}km"].to_numpy(np.int64)
    malas = int((esperado != obtenido).sum())
    fallos += malas
    print(f"  dias_desde_ignicion_{radio}km:  "
          f"{'OK' if malas == 0 else f'{malas} DISCREPANCIAS'}")

print(f"\n{'=' * 70}")
print("SIN FUGA: todos los valores coinciden con el recuento manual hasta D-1"
      if fallos == 0 else f"REVISAR: {fallos} discrepancias")
print("=" * 70)

# ─── La prueba directa: los positivos no se ven a si mismos ─────────────────────
solos = 0
for i in range(len(sel)):
    if sel.target_ignicion.iloc[i] == 1:
        dentro = (np.abs(ix - sx[i]) <= 12_001) & (np.abs(iy - sy[i]) <= 12_001)
        dt = sd[i] - idia
        if not (dentro & (dt >= 1) & (dt <= 30)).any():
            assert calculado[f"igniciones_12km_30d"].iloc[i] == 0
            solos += 1
print(f"\nigniciones de la muestra sin ninguna otra cerca en los 30 dias previos: {solos}")
print("todas ellas ven 0 en su propio historial (si vieran 1, seria fuga)")
