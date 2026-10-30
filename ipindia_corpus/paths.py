"""Resolve legacy Windows and container archive paths within this workspace."""

from __future__ import annotations

from pathlib import Path
import hashlib
import os
import uuid


def immutable_write(path: Path, data: bytes):
    """Publish complete bytes atomically; an interruption never leaves a partial blob."""
    path = Path(path)
    expected = hashlib.sha256(data).digest()
    def verify_existing():
        if hashlib.sha256(path.read_bytes()).digest() != expected:
            raise ValueError("immutable raw archive collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify_existing()
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            verify_existing()
        else:
            try:
                os.rename(temporary, path)
            except FileExistsError:
                verify_existing()
    finally:
        temporary.unlink(missing_ok=True)


class ArchiveResolver:
    """Reuse containment bases for one validation, never verification results."""
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.archive_root = (self.root / "data" / "raw").resolve()

    def resolve(self, stored: str) -> Path:
        value = stored.replace("\\", "/")
        if value.startswith("data/raw/"):
            candidate = (self.root / value).resolve()
        else:
            marker = "/data/raw/"
            position = value.lower().find(marker)
            if position < 0:
                raise ValueError("archive path is outside data/raw")
            candidate = (self.archive_root / value[position + len(marker):]).resolve()
        if not candidate.is_relative_to(self.archive_root):
            raise ValueError("archive path escapes data/raw")
        return candidate


def archive_path(root: Path, stored: str) -> Path:
    return ArchiveResolver(root).resolve(stored)


def portable_archive_path(root: Path, path: Path) -> str:
    root = Path(root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to((root / "data" / "raw").resolve()):
        raise ValueError("archive path is outside data/raw")
    return resolved.relative_to(root).as_posix()
