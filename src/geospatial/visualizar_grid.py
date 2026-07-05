"""Script de utilidad para visualizar e inspeccionar las celdas del grid Parquet generado.

Carga el archivo GeoParquet de la rejilla, imprime estadísticas de altitud, pendiente
y distribución de combustibles, y opcionalmente dibuja el mapa interactivo de Galicia.
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.geospatial.visualizar_grid")




def visualizar_grid(
    ruta_parquet: Path, mostrar_mapa: bool = False, columna_grafico: str = "both"
) -> None:
    """Carga e imprime estadísticas descriptivas del grid geoespacial y lo representa en un mapa.

    Args:
        ruta_parquet: Ruta del archivo .parquet a inspeccionar.
        mostrar_mapa: Si es True, dibuja el mapa con matplotlib.
        columna_grafico: Columna a pintar ('altitud_media', 'pendiente_media',
            'combustible_clase', 'both').
    """
    logger.info("=== INSPECTOR DE REJILLA GEOESPACIAL ===")

    if not ruta_parquet.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de rejilla en: {ruta_parquet}. "
            "Asegúrate de ejecutar el pipeline de la Fase 1 o tener el archivo sincronizado."
        )

    logger.info(f"Cargando dataset: {ruta_parquet}...")
    gdf = gpd.read_parquet(ruta_parquet)

    print("\n" + "=" * 50)
    print(f" RESUMEN DE LA REJILLA: {ruta_parquet.name}")
    print("=" * 50)
    print(f"• Número de celdas terrestres activas: {len(gdf):,}")
    print(f"• CRS del dataset: {gdf.crs}")
    print(f"• Dimensiones de la tabla: {gdf.shape[0]} filas x {gdf.shape[1]} columnas")
    print(f"• Columnas disponibles: {gdf.columns.tolist()}")

    print("\n" + "-" * 50)
    print(" ESTADÍSTICAS DE TOPOGRAFÍA")
    print("-" * 50)
    print(gdf[["altitud_media", "pendiente_media", "combustible_pct_forestal"]].describe().round(2))

    print("\n" + "-" * 50)
    print(" DISTRIBUCIÓN DE MODELOS DE COMBUSTIBLE (CORINE)")
    print("-" * 50)
    print(gdf["combustible_clase"].value_counts(dropna=False))

    print("\n" + "-" * 50)
    print(" ORIENTACIÓN PREDOMINANTE DE LAS LADERAS")
    print("-" * 50)
    print(gdf["orientacion_clase"].value_counts(dropna=False))

    if mostrar_mapa:
        import matplotlib.pyplot as plt

        logger.info(f"Generando visualización gráfica (Modo: {columna_grafico})...")

        # Modo 1: Mostrar ambas variables cara a cara (Subplots 1x2)
        if columna_grafico == "both":
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Mapa de altitudes
            gdf.plot(
                column="altitud_media",
                ax=axes[0],
                legend=True,
                cmap="terrain",
                legend_kwds={"label": "Altitud media (m)"},
            )
            axes[0].set_title("Rejilla Galicia 1km — Altitud Media (DEM)", fontsize=12)
            axes[0].axis("off")

            # Mapa de combustibles
            gdf.plot(
                column="combustible_clase",
                ax=axes[1],
                legend=True,
                cmap="tab10",
            )
            axes[1].set_title("Rejilla Galicia 1km — Tipo de Combustible (CORINE)", fontsize=12)
            axes[1].axis("off")

            plt.suptitle(f"Análisis Geoespacial Estático — {ruta_parquet.name}", fontsize=14, y=0.98)

        # Modo 2: Pintar solo una columna seleccionada
        else:
            if columna_grafico not in gdf.columns:
                raise ValueError(
                    f"La columna '{columna_grafico}' no existe en el dataset. Columnas: {gdf.columns.tolist()}"
                )

            fig, ax = plt.subplots(figsize=(10, 10))
            cmap = "terrain" if "altitud" in columna_grafico else "viridis" if "pendiente" in columna_grafico else "tab10"
            
            gdf.plot(
                column=columna_grafico,
                ax=ax,
                legend=True,
                cmap=cmap,
            )
            ax.set_title(f"Rejilla Galicia 1km — {columna_grafico} ({ruta_parquet.name})", fontsize=14)
            ax.axis("off")

        plt.tight_layout()
        logger.info("Cerrar la ventana del mapa para finalizar el script.")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspecciona y visualiza los atributos de una rejilla Parquet."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/grid/galicia_grid_1km_2018.parquet",
        help="Ruta al archivo GeoParquet de la rejilla (default: versión 2018).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Dibuja e ilustra el mapa espacial con Matplotlib.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="both",
        help="Columna a pintar en el mapa: 'altitud_media', 'pendiente_media', 'combustible_clase' o 'both' (default: 'both').",
    )

    args = parser.parse_args()

    visualizar_grid(
        ruta_parquet=Path(args.input),
        mostrar_mapa=args.plot,
        columna_grafico=args.column,
    )

