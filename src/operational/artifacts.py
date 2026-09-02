"""Atomic artifact writes, run locks and checksums for production jobs."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping
from uuid import uuid4

import pandas as pd


class RunLockError(RuntimeError):
    """The operational job is already running or left a lock behind."""


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")


def atomic_write_parquet(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Write a Parquet file and publish it only after the write succeeds."""

    path = Path(destination)
    temporary = _temporary_path(path)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_json(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Write JSON metadata atomically using UTF-8 and stable indentation."""

    path = Path(destination)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a local artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def run_lock(path: str | Path) -> Iterator[None]:
    """Create an exclusive lock file for the lifetime of an operational run.

    A stale lock is deliberately not removed automatically: removing it could
    allow two jobs to publish results at the same time. Operators can remove a
    lock after checking the recorded PID and timestamp.
    """

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "created_at": pd.Timestamp.now(tz="UTC").isoformat()}
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            details = lock_path.read_text(encoding="utf-8")
        except OSError:
            details = "sin metadatos legibles"
        raise RunLockError(f"Ya existe el lock operativo {lock_path}: {details}") from exc

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def lock_age_hours(path: str | Path) -> float | None:
    """Return lock age in hours, or ``None`` when the lock does not exist."""

    lock_path = Path(path)
    if not lock_path.exists():
        return None
    return max(0.0, (time.time() - lock_path.stat().st_mtime) / 3600.0)
