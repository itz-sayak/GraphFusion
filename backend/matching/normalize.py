"""Column-name normalisation (linguistic pre-processing, Cupid-style).

``customerIdentifier`` → ``['customer', 'identifier']``
``cust_no``           → ``['customer', 'number']``
``PULocationID``      → ``['pickup', 'location', 'identifier']``
"""
from __future__ import annotations

import re
from functools import lru_cache

from backend.config import abbreviations, glossary

_CAMEL_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_DIGIT_BOUNDARY = re.compile(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])")

STOPWORDS = {"the", "of", "a", "an", "and", "in", "for", "to", "by", "at", "on", "is"}


@lru_cache
def abbreviation_map() -> dict[str, str]:
    return {str(k).lower(): str(v).lower() for k, v in abbreviations()["abbreviations"].items()}


@lru_cache
def synonym_index() -> dict[str, int]:
    """token -> concept-set id. Multi-word synonyms are indexed by each token and by the joined form."""
    index: dict[str, int] = {}
    for i, group in enumerate(abbreviations()["synonyms"]):
        for term in group:
            term = str(term).lower()
            index.setdefault(term, i)
            for tok in term.split("_"):
                index.setdefault(tok, i)
    return index


def split_identifier(name: str) -> list[str]:
    """Split snake_case, kebab-case, camelCase, PascalCase, dotted and digit boundaries."""
    s = _CAMEL_1.sub(r"\1_\2", name)
    s = _CAMEL_2.sub(r"\1_\2", s)
    s = _DIGIT_BOUNDARY.sub("_", s)
    return [t.lower() for t in _NON_ALNUM.split(s) if t]


def expand_tokens(tokens: list[str]) -> list[str]:
    abbr = abbreviation_map()
    out: list[str] = []
    for tok in tokens:
        expanded = abbr.get(tok, tok)
        out.extend(expanded.split("_"))
    return [t for t in out if t not in STOPWORDS]


def normalize_name(name: str, vendor_prefixes: tuple[str, ...] = ()) -> str:
    return "_".join(name_tokens(name, vendor_prefixes))


def name_tokens(name: str, vendor_prefixes: tuple[str, ...] = ()) -> list[str]:
    label = glossary().get(name)
    tokens = split_identifier(label) if label else split_identifier(name)
    if len(tokens) > 1 and tokens[0] in vendor_prefixes:
        tokens = tokens[1:]
    return expand_tokens(tokens)


def concepts(tokens: list[str]) -> set[int]:
    idx = synonym_index()
    found = {idx[t] for t in tokens if t in idx}
    joined = "_".join(tokens)
    if joined in idx:
        found.add(idx[joined])
    return found


def to_snake(name: str) -> str:
    return "_".join(split_identifier(name)) or "column"
