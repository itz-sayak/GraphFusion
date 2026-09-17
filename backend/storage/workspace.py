"""Per-session workspace layout.

    workspace/sessions/<session_id>/
        raw/        immutable byte-for-byte copies of the uploaded sources (read-only)
        processed/  normalised Parquet representation of each dataset
        merges/     merge outputs, one directory per merge execution
        exports/    user-facing export bundles
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path, session_id: str):
        self.root = Path(root) / "sessions" / session_id
        self.raw = self.root / "raw"
        self.processed = self.root / "processed"
        self.merges = self.root / "merges"
        self.exports = self.root / "exports"
        for d in (self.raw, self.processed, self.merges, self.exports):
            d.mkdir(parents=True, exist_ok=True)

    def preserve_original(self, src: str | Path, dataset_id: str) -> tuple[Path, str]:
        """Copy the source into raw/ and mark it read-only. Returns (path, sha256)."""
        src = Path(src)
        dest = self.raw / f"{dataset_id}{src.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(src, dest)
            os.chmod(dest, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        return dest, sha256_file(dest)

    def processed_path(self, dataset_id: str) -> Path:
        return self.processed / f"{dataset_id}.parquet"

    def merge_dir(self, merge_id: str) -> Path:
        d = self.merges / merge_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def remove(self) -> None:
        def _onerror(func, path, _exc):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(self.root, onerror=_onerror)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
