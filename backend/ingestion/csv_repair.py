"""Repair of malformed delimited files that a CSV sniffer cannot read.

Real exports often contain the delimiter inside an unquoted text value
("Rua do Paço, 67"), so some rows have more fields than the header. A sniffer
then fails and the whole line becomes one column. This module detects that
case and rebuilds a well-formed file:

1. delimiter: the candidate (``,`` ``;`` tab ``|``) whose field count equals the
   header's on the most rows;
2. column profile: for each column, the value class (integer, number, date,
   empty, text) that dominates the well-formed rows;
3. a row with too few fields is padded with empty values;
4. a row with k extra fields had a delimiter inside one text value. Every way of
   merging k+1 adjacent fields back into one column is scored by
       2 · (fields whose class fits their column's profile)
     + 1 · (the merged column is a text column)
     + 0.5 · (each merged-in field starts with a space: ", " is prose, not a separator)
   and the best candidate wins. Ties are counted as ambiguous.

Rows and repairs are counted, never silently dropped.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

DELIMITERS = [",", ";", "\t", "|"]
NULL_MARKERS = {"", "NULL", "\\N"}
_INT = re.compile(r"^-?\d+$")
_NUM = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")
PROFILE_ROWS = 20_000


def value_class(v: str) -> str:
    v = v.strip()
    if v in NULL_MARKERS:
        return "empty"
    if _INT.match(v):
        return "int"
    if _NUM.match(v):
        return "num"
    if _DATE.match(v):
        return "date"
    return "text"


def _fits(cls: str, profile: str) -> bool:
    return cls == "empty" or cls == profile or (profile == "num" and cls == "int") or profile == "text"


def _rows(path: Path, delim: str, limit: int | None = None):
    with open(path, encoding="utf-8-sig", newline="", errors="replace") as fh:
        for i, row in enumerate(csv.reader(fh, delimiter=delim)):
            if limit is not None and i >= limit:
                return
            yield row


def needs_repair(path: Path) -> str | None:
    """The delimiter to repair with, when the header clearly has several fields; else None."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    best, best_hits = None, -1
    for d in DELIMITERS:
        n = len(next(csv.reader(io.StringIO(header), delimiter=d), []))
        if n < 2:
            continue
        hits = sum(1 for r in _rows(path, d, 2000) if len(r) == n)
        if hits > best_hits:
            best, best_hits = d, hits
    return best


def repair(src: Path, dest: Path, delim: str) -> dict[str, Any]:
    header = next(_rows(src, delim, 1))
    width = len(header)
    classes: list[Counter] = [Counter() for _ in range(width)]
    for i, row in enumerate(_rows(src, delim, PROFILE_ROWS + 1)):
        if i and len(row) == width:
            for j, v in enumerate(row):
                c = value_class(v)
                if c != "empty":
                    classes[j][c] += 1
    profile = []
    for cnt in classes:
        if not cnt:
            profile.append("text")
            continue
        top, n = cnt.most_common(1)[0]
        total = sum(cnt.values())
        if top == "int" and cnt["num"]:
            top = "num"
        profile.append(top if n / total >= 0.9 or (top == "num" and (cnt["int"] + cnt["num"]) / total >= 0.9) else "text")

    stats = {"delimiter": delim, "rows": 0, "padded": 0, "merged": 0, "ambiguous": 0, "unrepairable": 0}
    with open(dest, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(_rows(src, delim)):
            if i == 0:
                w.writerow(row)
                continue
            if not any(v.strip() for v in row):
                continue
            stats["rows"] += 1
            if len(row) < width:
                row = row + [""] * (width - len(row))
                stats["padded"] += 1
            elif len(row) > width:
                fixed, tie = _merge(row, width, profile, delim)
                if fixed is None:
                    stats["unrepairable"] += 1
                    fixed = row[: width - 1] + [delim.join(row[width - 1:])]
                else:
                    stats["merged"] += 1
                    stats["ambiguous"] += int(tie)
                row = fixed
            w.writerow(row)
    stats["column_profile"] = dict(zip(header, profile))
    return stats


def _merge(row: list[str], width: int, profile: list[str], delim: str) -> tuple[list[str] | None, bool]:
    k = len(row) - width
    scored = []
    for j in range(width):  # fields j .. j+k collapse into column j
        if profile[j] != "text":
            continue
        cand = row[:j] + [delim.join(row[j:j + k + 1])] + row[j + k + 1:]
        score = 2 * sum(_fits(value_class(v), p) for v, p in zip(cand, profile)) + 1
        score += 0.5 * sum(1 for v in row[j + 1:j + k + 1] if v[:1].isspace())
        scored.append((score, j, cand))
    if not scored:
        return None, False
    scored.sort(key=lambda s: (-s[0], s[1]))
    tie = len(scored) > 1 and scored[0][0] == scored[1][0]
    return scored[0][2], tie
