#!/usr/bin/env python3
"""Descarga los modelos y artefactos necesarios desde Hugging Face Hub.

Permite a cualquier usuario nuevo configurar el entorno para ejecutar el
Dashboard Streamlit o lanzar inferencias operativas sin tener que reentrenar.

Modo de uso:
    # Descarga estándar de todos los artefactos:
    python scripts/download_artifacts.py --repo-id tu-usuario/galicia-wildfire-risk

    # Si el repo está configurado en .env (HF_REPO_ID):
    python scripts/download_artifacts.py

    # Descarga ligera solo para ver el Dashboard (sin los 145 MB del estado meteorológico):
    python scripts/download_artifacts.py --only-dashboard
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descargar artefactos operativos del sistema de incendios desde Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Identificador del repositorio en Hugging Face (ej. usuario/galicia-wildfire-risk).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token de Hugging Face (solo necesario si el repositorio es privado).",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset"],
        help="Tipo de repositorio en Hugging Face (por defecto: 'model').",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=".",
        help="Directorio raíz del proyecto donde depositar los archivos (por defecto: '.').",
    )
    parser.add_argument(
        "--only-dashboard",
        action="store_true",
        help="Descargar únicamente lo indispensable para el Dashboard (~25 MB), omitiendo el estado meteorológico de 30 días.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    repo_id = args.repo_id or os.getenv("HF_REPO_ID")
    token = args.token or os.getenv("HF_TOKEN")

    if not repo_id:
        logger.error(
            "Falta el identificador del repositorio. Pásalo con --repo-id <usuario/repo> "
            "o configúralo en tu archivo .env como HF_REPO_ID."
        )
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "La librería 'huggingface_hub' no está instalada. "
            "Instálala con: pip install huggingface_hub"
        )
        return 1

    # Definir qué patrones descargar
    if args.only_dashboard:
        allow_patterns = [
            "data/models/*",
            "data/processed/grid/*",
            "data/processed/predicciones_operativas.parquet",
            "data/processed/predicciones_operativas.manifest.json",
        ]
        logger.info("Modo ligero activado: descargando modelos, rejilla y predicción para Dashboard...")
    else:
        allow_patterns = [
            "data/models/*",
            "data/processed/grid/*",
            "data/processed/state/*",
            "data/processed/predicciones_operativas.parquet",
            "data/processed/predicciones_operativas.manifest.json",
        ]
        logger.info("Modo completo: descargando modelos, rejilla, predicciones y estado meteorológico...")

    target_path = Path(args.target_dir).resolve()
    logger.info("📥 Descargando desde '%s' hacia '%s'...", repo_id, target_path)

    try:
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=args.repo_type,
            token=token,
            local_dir=str(target_path),
            allow_patterns=allow_patterns,
        )
        logger.info("✅ Descarga completada con éxito en %s.", downloaded_dir)
        print("\n🚀 ¡Todo listo! Puedes ejecutar:")
        print("   python -m streamlit run app.py")
        return 0
    except Exception as exc:
        logger.error("❌ Error descargando artefactos desde Hugging Face: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
