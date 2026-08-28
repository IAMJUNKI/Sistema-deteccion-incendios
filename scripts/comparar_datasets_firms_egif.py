"""Compara el dataset FIRMS (Parquet de Mikel) con el contrato del datacubo EGIF.

Responde a una sola pregunta: qué hace falta para repetir sobre EGIF el trabajo
de modelado que se hizo sobre FIRMS.

El lado FIRMS se **mide** leyendo los Parquet de `data/raw/dataset_maestro/`.
El lado EGIF se **declara** a partir del código publicado en `origin/main`
(`src/ingestion/meteorology.py`, `src/geospatial/{topography,vegetation,
human_activity}.py`, `src/features/{time,tabular}.py`), porque el Parquet aún no
está disponible en local. Si se pasa `--egif-dir` apuntando al dataset real, el
contrato declarado se reconcilia contra el esquema de verdad y se informa de
cualquier discrepancia.

Uso:
    python scripts/comparar_datasets_firms_egif.py
    python scripts/comparar_datasets_firms_egif.py --egif-dir data/processed/tabular/egif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[1]
FIRMS_DIR = RAIZ / "data" / "raw" / "dataset_maestro"
EGIF_DIR = RAIZ / "data" / "processed" / "tabular" / "egif"

# Positivos EGIF de 2022 segun la salida ya ejecutada de
# notebooks/10_experimentos_feature_selection.ipynb:
#   "validation_2022: 432,668 filas | 1,659 igniciones | 0.383% positivos"
# Solo se usa como referencia mientras el Parquet no este disponible en local.
EGIF_POSITIVOS_2022_REFERENCIA = 1_659

# ─────────────────────────────────────────────────────────────────────────────
# Contrato EGIF declarado, transcrito del código de origin/main.
# Cada grupo apunta al fichero del que procede para poder reauditarlo.
# ─────────────────────────────────────────────────────────────────────────────
ASPECTO = [
    f"aspect_{i:03d}_{i + 45:03d}_fraction" for i in range(0, 360, 45)
]

CONTRATO_EGIF: dict[str, tuple[str, list[str]]] = {
    "identificacion": (
        "src/features/tabular.py",
        ["cell_id", "x", "y", "fecha", "is_galicia", "year"],
    ),
    "calendario": (
        "src/features/time.py",
        [
            "month", "iso_week", "day_of_year", "day_of_week", "is_weekend",
            "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos",
        ],
    ),
    "topografia": (
        "src/geospatial/topography.py",
        ["elevation_mean", "elevation_std", "slope_mean", "slope_std",
         "roughness_mean", "roughness_std", *ASPECTO, "aspect_no_data_fraction"],
    ),
    "cobertura": (
        "src/geospatial/vegetation.py",
        ["artificial", "agriculture", "broadleaf_forest", "coniferous_forest",
         "mixed_forest", "scrub", "open_spaces", "wetlands", "water",
         "forest_cover_fraction"],
    ),
    "actividad_humana": (
        "src/geospatial/human_activity.py",
        ["distance_to_road_m", "road_length_km", "distance_to_residential_area_m",
         "road_length_main_km", "road_length_local_km", "road_length_track_km",
         "road_length_other_km", "residential_area_fraction", "building_area_fraction"],
    ),
    "meteorologia": (
        "src/ingestion/meteorology.py",
        ["temperature_mean", "temperature_min", "temperature_max",
         "temperature_max_12_18h", "relative_humidity_mean", "relative_humidity_min",
         "relative_humidity_min_12_18h", "wind_speed_mean", "wind_speed_max",
         "wind_speed_max_12_18h", "precipitation_sum", "precipitation_sum_3d",
         "precipitation_sum_7d", "precipitation_sum_14d", "precipitation_sum_30d",
         "temperature_mean_7d", "relative_humidity_mean_7d", "vpd_mean",
         "vpd_max_12_18h", "consecutive_dry_days"],
    ),
    "target": (
        "src/ingestion/ingest_egif.py",
        ["target_ignicion", "burned_area_ha", "large_fire_500ha",
         "is_near_ignition_25x25_10d"],
    ),
}

# Variables que el cubo almacena pero el contrato prohíbe como predictor.
NO_PREDICTORAS = {
    "cell_id", "x", "y", "fecha", "is_galicia", "year",
    "target_ignicion", "burned_area_ha", "large_fire_500ha",
    "is_near_ignition_25x25_10d", "precipitation_sum_1d", "aspect_no_data_fraction",
}

# Variables que el perfil de revisión (TEST_DATACUBE_VARIABLE_FLAGS) desactiva.
MARCADAS_PARA_EXCLUIR = {
    "roughness_mean": "topografía redundante con pendiente/elevación",
    "roughness_std": "topografía redundante con pendiente/elevación",
    "forest_cover_fraction": "suma exacta de las tres fracciones de bosque",
    "consecutive_dry_days": "se solapa con los acumulados de precipitación",
    "distance_to_road_m": "se satura en cero a 1 km",
    "distance_to_residential_area_m": "se satura en cero a 1 km",
    "aspect_no_data_fraction": "constante a cero en la malla de Galicia",
    **{v: "calendario: se deriva de `time` cuando haga falta" for v in
       ["month", "iso_week", "day_of_year", "month_sin", "month_cos",
        "day_of_week", "is_weekend", "day_of_year_sin", "day_of_year_cos", "year"]},
}

# ─────────────────────────────────────────────────────────────────────────────
# Mapeo FIRMS -> EGIF.  El tercer campo es la acción necesaria.
#   directo    : renombrar y listo
#   derivar    : hay que calcularla desde columnas EGIF existentes
#   revisar    : existe algo equivalente pero no idéntico
#   ausente    : no hay equivalente
# ─────────────────────────────────────────────────────────────────────────────
MAPEO: list[tuple[str, str, str, str]] = [
    ("cell_id",                 "cell_id",                    "directo", ""),
    ("fecha",                   "fecha",                      "directo", ""),
    ("target",                  "target_ignicion",            "directo",
     "EGIF usa la fecha de deteccion normalizada a dia"),
    ("tmax_vc",                 "temperature_max",            "directo", ""),
    ("rhmin_vc",                "relative_humidity_min",      "directo", ""),
    ("vmax_vc",                 "wind_speed_max",             "directo", ""),
    ("prec_dia",                "precipitation_sum",          "directo", ""),
    ("prec_acum_7d",            "precipitation_sum_7d",       "directo", ""),
    ("prec_acum_30d",           "precipitation_sum_30d",      "directo", ""),
    ("tmax_media_7d",           "temperature_mean_7d",        "revisar",
     "EGIF promedia la temperatura MEDIA; FIRMS promediaba la MAXIMA"),
    ("alerta_30_30",            "temperature_max + relative_humidity_min", "derivar",
     "indicador (tmax>30 y rhmin<30); trivial de recalcular"),
    ("altitud_media",           "elevation_mean",             "directo", ""),
    ("pendiente_media",         "slope_mean",                 "directo", ""),
    ("orientacion_media",       "aspect_*_fraction (x8)",     "revisar",
     "EGIF da 8 fracciones en vez de una media circular: es mejor"),
    ("orientacion_clase",       "aspect_*_fraction (x8)",     "revisar",
     "la categorica desaparece; las fracciones la sustituyen"),
    ("combustible_clase",       "fracciones CORINE (x9)",     "revisar",
     "de categorica a 9 continuas: es mejor, pero cambia el preprocesado"),
    ("combustible_pct_forestal", "broadleaf+coniferous+mixed_forest", "derivar",
     "forest_cover_fraction existe pero esta marcada para excluir"),
    ("mes",                     "month",                      "directo", ""),
    ("dia_semana",              "day_of_week",                "directo", ""),
    ("es_finde",                "is_weekend",                 "directo", ""),
    ("dia_anio_sin/cos",        "day_of_year_sin/cos",        "directo",
     "OJO: el perfil de revision desactiva todo el calendario"),
]

# Variables derivadas por src/features/derived.py y sus entradas en EGIF.
DERIVADAS: list[tuple[str, list[str], str]] = [
    ("vpd", ["vpd_mean", "vpd_max_12_18h"],
     "YA EXISTE en EGIF, y con ventana de tarde: mejor que la nuestra"),
    ("dias_sin_lluvia", ["consecutive_dry_days"],
     "existe, pero marcada para excluir del perfil final"),
    ("nesterov", ["temperature_max", "relative_humidity_min", "precipitation_sum"],
     "recalculable"),
    ("ratio_termico_eolico", ["temperature_max", "wind_speed_max"], "recalculable"),
    ("vpd_x_forestal", ["vpd_mean", "broadleaf_forest", "coniferous_forest", "mixed_forest"],
     "recalculable"),
    ("seq_x_forestal", ["consecutive_dry_days", "broadleaf_forest", "coniferous_forest",
                        "mixed_forest"],
     "recalculable si sobrevive consecutive_dry_days"),
    ("*_anom (climatologia celda x mes)",
     ["temperature_max", "relative_humidity_min", "wind_speed_max", "precipitation_sum_7d"],
     "la calcula nuestro codigo, no depende del cubo"),
    ("*_z_dia (z-score espacial diario)",
     ["temperature_max", "relative_humidity_min", "wind_speed_max", "precipitation_sum_7d"],
     "la calcula nuestro codigo, no depende del cubo"),
]


def titulo(texto: str) -> None:
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def inspeccionar_firms(directorio: Path) -> dict[str, object]:
    """Lee esquema y conteos reales de los Parquet FIRMS sin cargarlos en memoria."""
    ficheros = sorted(directorio.glob("dataset_maestro_*.parquet"))
    if not ficheros:
        return {"disponible": False, "anios": {}, "columnas": []}

    columnas: list[str] = []
    anios: dict[int, dict[str, int]] = {}
    for ruta in ficheros:
        anio = int(ruta.stem.split("_")[-1])
        parquet = pq.ParquetFile(ruta)
        if not columnas:
            columnas = list(parquet.schema_arrow.names)
        filas = parquet.metadata.num_rows
        positivos = 0
        if "target" in parquet.schema_arrow.names:
            tabla = parquet.read(columns=["target"])
            positivos = int(sum(tabla.column("target").to_pylist()))
        anios[anio] = {"filas": filas, "positivos": positivos}
        parquet.close()
    return {"disponible": True, "anios": anios, "columnas": columnas}


def inspeccionar_egif(directorio: Path) -> dict[str, object]:
    """Lee el contrato real del dataset EGIF si está presente en disco."""
    metadata = directorio / "metadata.json"
    if not metadata.exists():
        return {"disponible": False}

    contenido = json.loads(metadata.read_text(encoding="utf-8"))
    anios: dict[int, dict[str, int]] = {}
    for anio_txt, filas in sorted(contenido.get("annual_files", {}).items()):
        ruta = directorio / f"year={anio_txt}" / f"dataset_{anio_txt}.parquet"
        positivos = 0
        if ruta.exists():
            parquet = pq.ParquetFile(ruta)
            if "target_ignicion" in parquet.schema_arrow.names:
                tabla = parquet.read(columns=["target_ignicion"])
                positivos = int(sum(tabla.column("target_ignicion").to_pylist()))
            parquet.close()
        anios[int(anio_txt)] = {"filas": int(filas), "positivos": positivos}
    return {
        "disponible": True,
        "anios": anios,
        "predictores": contenido.get("predictor_columns", []),
        "contrato_temporal": contenido.get("time_contract"),
        "filas_descartadas": contenido.get("dropped_rows_incomplete_predictors"),
    }


def tabla_anios(anios: dict[int, dict[str, int]], etiqueta: str) -> None:
    if not anios:
        print("  (sin datos)")
        return
    print(f"  {'anio':>6} {'filas':>14} {'positivos':>11} {'prevalencia':>14}")
    print(f"  {'-' * 6} {'-' * 14} {'-' * 11} {'-' * 14}")
    for anio, datos in sorted(anios.items()):
        filas, positivos = datos["filas"], datos["positivos"]
        prev = positivos / filas if filas else 0.0
        una_de = f"1/{round(1 / prev):,}" if prev else "-"
        print(f"  {anio:>6} {filas:>14,} {positivos:>11,} {prev:>13.6%}  {una_de}")
    total_f = sum(d["filas"] for d in anios.values())
    total_p = sum(d["positivos"] for d in anios.values())
    print(f"  {'-' * 6} {'-' * 14} {'-' * 11} {'-' * 14}")
    print(f"  {etiqueta:>6} {total_f:>14,} {total_p:>11,} "
          f"{total_p / total_f if total_f else 0:>13.6%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms-dir", default=str(FIRMS_DIR))
    parser.add_argument("--egif-dir", default=str(EGIF_DIR))
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    firms = inspeccionar_firms(Path(args.firms_dir))
    egif = inspeccionar_egif(Path(args.egif_dir))

    # ── 1. Lo que hay ────────────────────────────────────────────────────────
    titulo("1. VOLUMEN Y PREVALENCIA")
    print("\nFIRMS (medido en data/raw/dataset_maestro/):")
    if firms["disponible"]:
        tabla_anios(firms["anios"], "TOTAL")
    else:
        print(f"  NO DISPONIBLE en {args.firms_dir}")

    print("\nEGIF (data/processed/tabular/egif/):")
    if egif["disponible"]:
        tabla_anios(egif["anios"], "TOTAL")
        print(f"\n  contrato temporal : {egif.get('contrato_temporal')}")
        print(f"  filas descartadas por predictores incompletos: "
              f"{egif.get('filas_descartadas'):,}")
    else:
        print(f"  NO DISPONIBLE en {args.egif_dir}")
        print("  -> hay que pedirle el Parquet a Alfonso o regenerarlo con")
        print("     docs/ejecutar_pipeline.md (implica descargar ERA5-Land entero).")
        print("  El resto del informe usa el contrato declarado en el codigo.")

    # Punto de referencia publicado por el equipo, para poder contrastar el
    # volumen de positivos aunque el Parquet EGIF no este en local.
    if firms["disponible"] and not egif["disponible"]:
        firms_2022 = firms["anios"].get(2022, {}).get("positivos", 0)
        firms_total = sum(d["positivos"] for d in firms["anios"].values())
        egif_2022 = EGIF_POSITIVOS_2022_REFERENCIA
        print("\n  CONTRASTE DE COBERTURA DEL TARGET")
        print(f"    FIRMS 2022 .................. {firms_2022:>7,} igniciones")
        print(f"    EGIF  2022 .................. {egif_2022:>7,} igniciones"
              f"   (notebooks/10, celda ejecutada)")
        print(f"    ratio EGIF/FIRMS en 2022 .... {egif_2022 / firms_2022:>7.2f}x")
        print(f"\n    FIRMS 2019-2024 (6 anios) ... {firms_total:>7,} igniciones")
        print(f"    EGIF  solo 2022 ............. {egif_2022:>7,} igniciones")
        print("\n    Un solo anio de EGIF tiene casi tantos positivos como los seis")
        print("    anios de FIRMS juntos. La deteccion satelital solo ve los fuegos")
        print("    suficientemente grandes y calientes en el momento del paso del")
        print("    satelite; EGIF registra el parte oficial de cada incidente.")

    # ── 2. Contrato EGIF ─────────────────────────────────────────────────────
    titulo("2. CONTRATO EGIF DECLARADO (transcrito del codigo de origin/main)")
    total = 0
    for grupo, (origen, variables) in CONTRATO_EGIF.items():
        predictoras = [v for v in variables if v not in NO_PREDICTORAS]
        total += len(predictoras)
        print(f"\n  {grupo.upper()}  ({len(variables)} vars, "
              f"{len(predictoras)} predictoras)   <- {origen}")
        for variable in variables:
            if variable in NO_PREDICTORAS:
                marca, nota = "  --  ", "no predictora por contrato"
            elif variable in MARCADAS_PARA_EXCLUIR:
                marca, nota = "  !!  ", MARCADAS_PARA_EXCLUIR[variable]
            else:
                marca, nota = "  ok  ", ""
            print(f"    {marca}{variable:<34}{nota}")
    print(f"\n  TOTAL predictoras disponibles: {total}")
    print("  Leyenda:  ok = disponible   !! = marcada para excluir   -- = no predictora")

    # Reconciliacion contra el dataset real, si esta.
    if egif["disponible"]:
        declaradas = {v for _, vs in CONTRATO_EGIF.values() for v in vs}
        reales = set(egif["predictores"])
        faltan = sorted(reales - declaradas)
        sobran = sorted(v for v in declaradas - reales if v not in NO_PREDICTORAS)
        titulo("2b. RECONCILIACION CONTRATO DECLARADO vs DATASET REAL")
        print(f"  predictores reales: {len(reales)}")
        if faltan:
            print(f"  en el dataset y NO en nuestro contrato: {faltan}")
        if sobran:
            print(f"  en nuestro contrato y NO en el dataset: {sobran}")
        if not faltan and not sobran:
            print("  sin discrepancias")

    # ── 3. Mapeo ─────────────────────────────────────────────────────────────
    titulo("3. MAPEO FIRMS -> EGIF")
    columnas_firms = set(firms.get("columnas", []))
    print(f"\n  {'FIRMS':<26}{'EGIF':<40}{'accion'}")
    print(f"  {'-' * 26}{'-' * 40}{'-' * 10}")
    for origen, destino, accion, nota in MAPEO:
        aviso = ""
        if columnas_firms and origen not in columnas_firms and "/" not in origen:
            aviso = "   [no vista en el Parquet FIRMS]"
        print(f"  {origen:<26}{destino:<40}{accion}{aviso}")
        if nota:
            print(f"  {'':<26}{'-> ' + nota}")

    resumen: dict[str, int] = {}
    for _, _, accion, _ in MAPEO:
        resumen[accion] = resumen.get(accion, 0) + 1
    print("\n  resumen: " + "   ".join(f"{k}={v}" for k, v in sorted(resumen.items())))

    # ── 4. Nuestras derivadas ────────────────────────────────────────────────
    titulo("4. VARIABLES DERIVADAS PROPIAS (src/features/derived.py)")
    disponibles = {v for _, vs in CONTRATO_EGIF.values() for v in vs}
    for nombre, entradas, comentario in DERIVADAS:
        faltantes = [e for e in entradas if e not in disponibles]
        estado = "BLOQUEADA" if faltantes else "OK"
        print(f"\n  [{estado:^9}] {nombre}")
        print(f"              entradas: {', '.join(entradas)}")
        print(f"              {comentario}")
        if faltantes:
            print(f"              FALTAN: {faltantes}")

    # ── 5. Lo que NO migra solo ──────────────────────────────────────────────
    titulo("5. QUE HAY QUE HACER PARA EL TRABAJO ANALOGO")
    tareas = [
        ("dataset.py", "diccionario de renombrado FIRMS->EGIF y lectura de "
                       "year=YYYY/dataset_YYYY.parquet en vez de dataset_maestro_YYYY"),
        ("dataset.py", "las categoricas desaparecen: quitar CATEGORICAL_COLS y su "
                       "manejo de tipos; CORINE ya viene como fracciones continuas"),
        ("dataset.py", "el desplazamiento t-1 sigue siendo un groupby(cell_id).shift(1): "
                       "la variante prevision NO esta bloqueada por el dataset"),
        ("derived.py", "quitar vpd propio: EGIF ya trae vpd_mean y vpd_max_12_18h"),
        ("derived.py", "combustible_pct_forestal = broadleaf + coniferous + mixed"),
        ("derived.py", "anomalias y z-scores no cambian: se calculan sobre las "
                       "columnas base renombradas"),
        ("calibration.py", "sin cambios; con muestreo 1/25 el neg_sampling_rate es 25"),
        ("evaluate.py", "sin cambios"),
        ("configs/", "reescribir la lista explicita de columnas del YAML"),
        ("PROTOCOLO", "el equipo valida sobre negativos submuestreados (1/25): "
                      "la prevalencia de validacion sale inflada x25. Evaluar sobre "
                      "el anio completo antes de publicar cualquier cifra"),
    ]
    for modulo, tarea in tareas:
        print(f"\n  [{modulo}]")
        print(f"     {tarea}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
