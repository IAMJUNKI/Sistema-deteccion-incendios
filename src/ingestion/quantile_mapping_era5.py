"""
Correccion de sesgo ERA5 -> observaciones reales mediante Quantile Mapping empirico.
Primer modulo del ANALISIS DE DEGRADACION (contribucion cientifica central del TFM).

Contexto: ERA5-Land subestima las temperaturas maximas en Galicia (sesgo -1.5 a -1.7 C,
medido contra 3.566 observaciones de estaciones MeteoGalicia, julio-2022). El mismo tipo
de correccion se aplicara al gap ERA5 (entrenamiento) vs WRF-MeteoGalicia (produccion).

Metodo: se aprende el mapa que transforma los cuantiles de la distribucion del modelo
en los de la distribucion observada. Validacion honesta: split 70/30 POR ESTACIONES
(la correccion se evalua en estaciones nunca vistas).

Resultado (julio-2022, Tmax): MAE 2.23 -> 1.61 C (-28%) | sesgo -1.69 -> +0.21 C

Entrada: Datos/validacion_mapeo_estaciones.parquet (del experimento de mapeo 9km->1km)
Salida:  Datos/quantile_mapping_tmax_julio2022.csv (tabla de correccion por cuantiles)

Trabajo futuro: extender a todas las estaciones/meses del anio (la correccion es
estacional), a las demas variables (humedad, viento) y al par ERA5<->WRF.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
v = pd.read_parquet("Datos/validacion_mapeo_estaciones.parquet")

ests = v.idEstacion.unique()
rng.shuffle(ests)
n_tr = int(len(ests) * 0.7)
tr = v[v.idEstacion.isin(ests[:n_tr])]
te = v[v.idEstacion.isin(ests[n_tr:])].copy()

qs = np.linspace(0.01, 0.99, 99)
q_mod = np.quantile(tr.tC, qs)
q_obs = np.quantile(tr.tmax_obs, qs)

te["tC_qm"] = np.interp(te.tC, q_mod, q_obs)

for nombre, serie in [("Sin corregir", te.tC), ("Con Quantile Mapping", te.tC_qm)]:
    err = serie - te.tmax_obs
    print(f"{nombre:22s} MAE {err.abs().mean():.2f} C | sesgo {err.mean():+.2f} C")

pd.DataFrame({"quantil": qs, "valor_era5C": q_mod, "valor_observado": q_obs}).to_csv(
    "Datos/quantile_mapping_tmax_julio2022.csv", index=False)
print("Tabla de correccion guardada.")
