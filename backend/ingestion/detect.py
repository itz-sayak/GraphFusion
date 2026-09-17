"""Automatic source-format detection.

Order of evidence: magic bytes (authoritative for binary formats) → content
probe (JSON vs JSONL vs CSV) → file extension (fallback). Extensions lie in
practice (``.txt`` CSVs, ``.json`` files that are really JSON Lines), so the
content probe wins over the extension for text formats.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.core.errors import UnsupportedFormatError
from backend.core.models import SourceType

_EXTENSION_MAP = {
    ".csv": SourceType.CSV,
    ".tsv": SourceType.CSV,
    ".txt": SourceType.CSV,
    ".json": SourceType.JSON,
    ".jsonl": SourceType.JSONL,
    ".ndjson": SourceType.JSONL,
    ".parquet": SourceType.PARQUET,
    ".pq": SourceType.PARQUET,
    ".xlsx": SourceType.XLSX,
    ".xlsm": SourceType.XLSX,
    ".sqlite": SourceType.SQLITE,
    ".sqlite3": SourceType.SQLITE,
    ".db": SourceType.SQLITE,
}


def detect_format(path: str | Path, probe_bytes: int = 65536) -> SourceType:
    path = Path(path)
    if not path.exists():
        raise UnsupportedFormatError(f"File not found: {path}")
    with open(path, "rb") as fh:
        head = fh.read(probe_bytes)

    if head[:4] == b"PAR1":
        return SourceType.PARQUET
    if head[:16] == b"SQLite format 3\x00":
        return SourceType.SQLITE
    if head[:4] == b"PK\x03\x04":
        # Office Open XML is a zip; only accept it as xlsx when the extension agrees.
        if path.suffix.lower() in (".xlsx", ".xlsm"):
            return SourceType.XLSX
        raise UnsupportedFormatError(f"Zip archive is not a supported spreadsheet: {path.name}")

    if _is_binary(head):
        raise UnsupportedFormatError(f"{path.name} looks like binary data in an unsupported format")

    text = head.decode("utf-8-sig", errors="ignore").lstrip()
    if text[:1] in ("[", "{"):
        return _probe_json(text, path)

    ext = _EXTENSION_MAP.get(path.suffix.lower())
    if ext in (SourceType.CSV, None) and _looks_delimited(text):
        return SourceType.CSV
    if ext is not None:
        return ext
    raise UnsupportedFormatError(f"Could not determine format of {path.name}")


def _is_binary(head: bytes) -> bool:
    """NUL bytes or a high share of control characters mean this is not a text format."""
    if not head:
        return False
    if b"\x00" in head:
        return True
    control = sum(1 for b in head if b < 32 and b not in (9, 10, 13))
    return control / len(head) > 0.05


def _probe_json(text: str, path: Path) -> SourceType:
    if text.startswith("["):
        return SourceType.JSON
    lines = [ln for ln in text.splitlines() if ln.strip()][:5]
    parsed = 0
    for ln in lines:
        try:
            json.loads(ln)
            parsed += 1
        except json.JSONDecodeError:
            break
    # A single object spanning lines is JSON; several one-line objects are JSONL.
    if parsed >= 2 or (parsed == 1 and len(lines) == 1 and path.suffix.lower() in (".jsonl", ".ndjson")):
        return SourceType.JSONL
    return SourceType.JSON


def _looks_delimited(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()][:10]
    if not lines:
        return False
    for delim in (",", "\t", ";", "|"):
        counts = [ln.count(delim) for ln in lines]
        if counts[0] > 0 and len(set(counts[: max(2, len(counts) - 1)])) <= 2:
            return True
    return False
