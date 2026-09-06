#!/usr/bin/env python3
"""Publica modelos y artefactos operativos en un repositorio de Hugging Face Hub.

Modo de uso:
    # 1. Subida completa inicial (modelos + rejilla + estado meteorológico + predicciones):
    python scripts/publish_to_huggingface.py --all

    # 2. Subida diaria desde servidor (solo estado meteorológico y predicciones actualizadas):
    python scripts/publish_to_huggingface.py

    # 3. Pasar repo-id o token explícitamente:
    python scripts/publish_to_huggingface.py --all --repo-id tu-usuario/galicia-wildfire-risk
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
logger = logging.getLogger("publish_hf")

# Artefactos estáticos (modelos, metadatos y rejilla base)
STATIC_ARTIFACTS = [
    "data/models/active_model_manifest.json",
    "data/models/forecast_risk_egif_48_t1.joblib",
    "data/models/forecast_risk_egif_48_t1.json",
    "data/models/forecast_risk_egif_48_t2.joblib",
    "data/models/forecast_risk_egif_48_t2.json",
    "data/models/forecast_risk_egif_48_t3.joblib",
    "data/models/forecast_risk_egif_48_t3.json",
    "data/processed/grid/galicia_grid_1km_egif.parquet",
]

# Artefactos dinámicos (pronóstico diario y estado meteorológico acumulado)
DYNAMIC_ARTIFACTS = [
    "data/processed/predicciones_operativas.parquet",
    "data/processed/predicciones_operativas.manifest.json",
    "data/processed/state/weather_daily_state.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publicar artefactos operativos del sistema de incendios a Hugging Face."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Subir todos los artefactos (estáticos + dinámicos). Usar para inicializar o actualizar modelos.",
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
        help="Token de Hugging Face con permisos de escritura (o definir HF_TOKEN en .env).",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset"],
        help="Tipo de repositorio en Hugging Face (por defecto: 'model').",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    repo_id = args.repo_id or os.getenv("HF_REPO_ID")
    token = args.token or os.getenv("HF_TOKEN")

    if not repo_id:
        logger.error(
            "Falta el identificador del repositorio. Configura HF_REPO_ID en tu archivo .env "
            "o pasa --repo-id <usuario/repositorio>."
        )
        return 1

    if not token:
        logger.error(
            "Falta el token de autenticación. Configura HF_TOKEN en tu archivo .env "
            "o pasa --token <hf_token> (debe tener permisos de 'Write')."
        )
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error(
            "La librería 'huggingface_hub' no está instalada. "
            "Instálala con: pip install huggingface_hub"
        )
        return 1

    api = HfApi(token=token)

    # Verificar o crear repositorio si no existe
    try:
        api.repo_info(repo_id=repo_id, repo_type=args.repo_type)
        logger.info("Repositorio conectado: %s (%s)", repo_id, args.repo_type)
    except Exception:
        logger.info("El repositorio no existe o no es accesible. Intentando crearlo como privado...")
        try:
            api.create_repo(repo_id=repo_id, repo_type=args.repo_type, private=True, exist_ok=True)
            logger.info("Repositorio creado con éxito: %s", repo_id)
        except Exception as exc:
            logger.error("No se pudo crear/acceder al repositorio %s: %s", repo_id, exc)
            return 1

    files_to_upload = list(DYNAMIC_ARTIFACTS)
    if args.all:
        files_to_upload = STATIC_ARTIFACTS + DYNAMIC_ARTIFACTS
        logger.info("Modo COMPLETO (--all): Subiendo modelos, rejilla y estado operativo...")
    else:
        logger.info("Modo DIARIO: Sincronizando estado operativo y predicciones...")

    uploaded_count = 0
    missing_count = 0

    for rel_path in files_to_upload:
        p = Path(rel_path)
        if not p.exists():
            logger.warning("  ⚠️ Archivo local no encontrado, omitiendo: %s", rel_path)
            missing_count += 1
            continue

        size_mb = p.stat().st_size / (1024 * 1024)
        logger.info("  ⬆️ Subiendo %s (%.2f MB)...", rel_path, size_mb)
        try:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type=args.repo_type,
                commit_message=f"Sync artefacto: {p.name}",
            )
            uploaded_count += 1
        except Exception as exc:
            logger.error("  ❌ Error subiendo %s: %s", rel_path, exc)

    logger.info(
        "✅ Sincronización completada. Archivos subidos: %d, no encontrados: %d.",
        uploaded_count,
        missing_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
