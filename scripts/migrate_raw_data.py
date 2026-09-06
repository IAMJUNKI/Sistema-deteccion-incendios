#!/usr/bin/env python3
"""Normaliza los datos compartidos del equipo a las rutas del workflow.

La operación predeterminada es ``--dry-run``. Con ``--apply`` se copian los
archivos, se verifica el SHA-256 byte a byte y se escribe
``data/raw/migration_manifest.json``. ``--remove-source`` solo es válido junto
con ``--apply`` y elimina los archivos de origen después de comprobar que cada
destino coincide; no elimina directorios ni archivos no incluidos en el plan.

Ejemplos::

    PYTHONPATH=. python scripts/migrate_raw_data.py
    PYTHONPATH=. python scripts/migrate_raw_data.py --apply
    PYTHONPATH=. python scripts/migrate_raw_data.py --apply --remove-source
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MigrationEntry:
    source: str
    destination: str
    sha256: str
    bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(source_root: Path) -> list[tuple[Path, Path]]:
    """Devuelve solo fuentes conocidas y evita incluir metadatos de macOS."""

    mappings: list[tuple[Path, Path]] = []

    def add_tree(source: Path, destination: Path) -> None:
        if not source.exists():
            return
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                mappings.append((path, destination / path.relative_to(source)))

    add_tree(source_root / "meteorology" / "era5", Path("meteorology/era5"))
    add_tree(source_root / "meteorology" / "FWI", Path("meteorology/fwi"))
    add_tree(source_root / "fire_history", Path("fire_history"))
    add_tree(source_root / "human_activity", Path("human_activity/galicia-220101-free-shp"))

    boundary = source_root / "galicia_boundary" / "galicia_boundary.geojson"
    if boundary.exists():
        mappings.append((boundary, Path("igm/galicia_boundary.geojson")))
    corine = source_root / "landcover" / "U2018_CLC2018_V2020_20u1.tif"
    if corine.exists():
        mappings.append((corine, Path("corine/U2018_CLC2018_V2020_20u1.tif")))
    return sorted(mappings, key=lambda item: str(item[1]))


def build_plan(source_root: str | Path, raw_root: str | Path) -> list[MigrationEntry]:
    source_root = Path(source_root)
    raw_root = Path(raw_root)
    entries: list[MigrationEntry] = []
    for source, relative_destination in _files(source_root):
        destination = raw_root / relative_destination
        entries.append(
            MigrationEntry(
                source=str(source),
                destination=str(destination),
                sha256=_sha256(source),
                bytes=source.stat().st_size,
            )
        )
    return entries


def migrate(
    source_root: str | Path = "misc/Datos/raw",
    raw_root: str | Path = "data/raw",
    *,
    apply: bool = False,
    remove_source: bool = False,
) -> Path:
    if remove_source and not apply:
        raise ValueError("--remove-source requiere --apply.")
    entries = build_plan(source_root, raw_root)
    if not entries:
        raise FileNotFoundError(f"No se encontraron fuentes migrables en {source_root}.")

    for entry in entries:
        status = "planned"
        destination = Path(entry.destination)
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and _sha256(destination) != entry.sha256:
                raise FileExistsError(
                    f"El destino existe con otro contenido y no se sobrescribe: {destination}"
                )
            if not destination.exists():
                shutil.copy2(entry.source, destination)
            if _sha256(destination) != entry.sha256:
                raise IOError(f"Falló la verificación SHA-256 de {destination}")
            status = "copied_and_verified"
        print(f"{status:20} {entry.source} -> {entry.destination}")

    manifest_path = Path(raw_root) / "migration_manifest.json"
    if apply:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_root": str(Path(source_root)),
                    "destination_root": str(Path(raw_root)),
                    "entries": [asdict(entry) for entry in entries],
                    "verified": True,
                    "source_removed": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if remove_source:
            for entry in entries:
                source = Path(entry.source)
                if _sha256(Path(entry.destination)) != entry.sha256:
                    raise IOError(f"No se elimina una fuente no verificada: {source}")
                source.unlink()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_removed"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="misc/Datos/raw")
    parser.add_argument("--destination", default="data/raw")
    parser.add_argument("--apply", action="store_true", help="Copia y verifica los archivos")
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Elimina las fuentes individuales después de verificar sus destinos",
    )
    args = parser.parse_args()
    manifest = migrate(
        args.source,
        args.destination,
        apply=args.apply,
        remove_source=args.remove_source,
    )
    print(f"Plan de migración: {manifest}")


if __name__ == "__main__":
    main()
