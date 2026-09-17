"""Dynamic canonical naming. Nothing here knows about customers or taxis:
names are derived from the source schemas and the integration structure."""
from __future__ import annotations

import re

from backend.config import glossary
from backend.matching.normalize import abbreviation_map, name_tokens, split_identifier


def snake(name: str, vendor_prefixes: tuple[str, ...] = ()) -> str:
    toks = split_identifier(glossary().get(name, name))
    if len(toks) > 1 and toks[0] in vendor_prefixes:
        toks = toks[1:]
    s = "_".join(toks)
    s = re.sub(r"_id$", "_id", s)
    return s or "column"


def abbreviation_count(name: str) -> int:
    abbr = abbreviation_map()
    return sum(1 for t in split_identifier(name) if t in abbr and abbr[t] != t)


def canonical_attribute_name(members: list[tuple[str, str]], dataset_priority: list[str], vendor_prefixes: tuple[str, ...] = ()) -> str:
    """Pick the most descriptive member name: fewest abbreviations, then the highest-priority dataset."""
    rank = {d: i for i, d in enumerate(dataset_priority)}
    best = min(members, key=lambda m: (abbreviation_count(m[1]), rank.get(m[0], 99), len(m[1])))
    return snake(best[1], vendor_prefixes)


def role_for(reference_column: str, key_column: str, vendor_prefixes: tuple[str, ...] = ()) -> str | None:
    """Role of a foreign key relative to the key it references.

    PULocationID → LocationID  : tokens {pickup, location, identifier} − {location, identifier} = pickup
    cust_id      → customer_id : {customer, identifier} − {customer, identifier} = ∅ → no role
    """
    ref = name_tokens(reference_column, vendor_prefixes)
    key = set(name_tokens(key_column, vendor_prefixes))
    extra = [t for t in ref if t not in key and not t.isdigit()]
    # a role is a short qualifier (pickup, billing, home); long remainders are just
    # a verbose column name, not a role
    return "_".join(extra) if extra and len(extra) <= 2 else None


class NameAllocator:
    """Guarantees unique output column names while recording collisions."""

    def __init__(self, reserved: set[str] | None = None):
        self.used: set[str] = set(reserved or ())
        self.collisions: list[tuple[str, str]] = []

    def allocate(self, desired: str, qualifier: str) -> str:
        name = desired
        if name in self.used:
            name = f"{qualifier}_{desired}"
            i = 2
            while name in self.used:
                name = f"{qualifier}_{desired}_{i}"
                i += 1
            self.collisions.append((desired, name))
        self.used.add(name)
        return name
