"""Semantic similarity between columns.

Semantic similarity S_E combines three views:

1. **Thesaurus concepts** (Cupid-style): name tokens are mapped to synonym
   concept sets (``city ~ location ~ town``, ``amount ~ spent ~ fare``) and
   compared by weighted Jaccard.
2. **Semantic-type compatibility** from profiling (EMAIL vs DATE → 0).
3. **Embedding cosine** of a textual "column document"
   ``"<expanded name> | <semantic type> | <top values>"``.

The embedding encoder is pluggable:

* ``tfidf`` (default) — character 3–5-gram TF-IDF fitted on all column
  documents of the session. No download, deterministic, fast, and captures
  morphological similarity of both names and values.
* ``sentence_transformers`` (optional) — any Sentence-Transformers model
  (default ``all-MiniLM-L6-v2``) for true distributional semantics
  (``earnings`` ~ ``income``). Install with ``pip install .[embeddings]``.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from backend.core.models import ColumnProfile, SemanticType
from backend.logging_conf import get_logger
from backend.matching.normalize import concepts, name_tokens
from backend.matching.type_sim import semantic_type_compatibility

log = get_logger(__name__)


class ColumnEncoder(Protocol):
    name: str

    def fit_transform(self, documents: list[str]) -> np.ndarray: ...


class TfidfEncoder:
    name = "tfidf"

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not documents:
            return np.zeros((0, 1))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        mat = vec.fit_transform(documents)
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        return (mat.multiply(1 / norms[:, None])).tocsr()


_MODEL_CACHE: dict[str, object] = {}


class SentenceTransformerEncoder:
    name = "sentence_transformers"

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        # loading a model takes seconds; every discovery would otherwise reload it
        if model not in _MODEL_CACHE:
            _MODEL_CACHE[model] = SentenceTransformer(model)
        self._model = _MODEL_CACHE[model]
        self.model_name = model

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        return self._model.encode(documents, normalize_embeddings=True, show_progress_bar=False)


def make_encoder(config: dict) -> ColumnEncoder:
    backend = config.get("embeddings", {}).get("backend", "tfidf")
    if backend == "sentence_transformers":
        try:
            return SentenceTransformerEncoder(config["embeddings"]["model"])
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the session
            log.warning("Sentence-transformers unavailable, falling back to tfidf", error=str(exc))
    return TfidfEncoder()


def column_document(profile: ColumnProfile, vendor_prefixes: tuple[str, ...] = ()) -> str:
    tokens = " ".join(name_tokens(profile.name, vendor_prefixes))
    values = " ".join(str(v) for v, _ in profile.top_values[:5]) if profile.semantic_type not in (SemanticType.ID, SemanticType.NUMERIC, SemanticType.CURRENCY) else ""
    return f"{tokens} | {profile.semantic_type.value.lower()} | {values}".strip()


class EmbeddingIndex:
    """Holds one embedding per column across every dataset in a session."""

    def __init__(self, encoder: ColumnEncoder):
        self.encoder = encoder
        self._keys: dict[str, int] = {}
        self._matrix = None

    def build(self, columns: dict[str, ColumnProfile], vendor_prefixes: tuple[str, ...] = ()) -> "EmbeddingIndex":
        keys = list(columns)
        self._keys = {k: i for i, k in enumerate(keys)}
        self._matrix = self.encoder.fit_transform([column_document(columns[k], vendor_prefixes) for k in keys])
        return self

    def cosine(self, a: str, b: str) -> float:
        if self._matrix is None or a not in self._keys or b not in self._keys:
            return 0.0
        va, vb = self._matrix[self._keys[a]], self._matrix[self._keys[b]]
        if hasattr(va, "multiply"):
            return float(max(0.0, min(1.0, va.multiply(vb).sum())))
        return float(max(0.0, min(1.0, np.dot(va, vb))))


def concept_similarity(a: str, b: str, vendor_prefixes: tuple[str, ...] = ()) -> float:
    ta, tb = name_tokens(a, vendor_prefixes), name_tokens(b, vendor_prefixes)
    ca, cb = concepts(ta), concepts(tb)
    # tokens outside the thesaurus are their own concept
    fa = {f"c{c}" for c in ca} | {t for t in ta if not concepts([t])}
    fb = {f"c{c}" for c in cb} | {t for t in tb if not concepts([t])}
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def semantic_similarity(pa: ColumnProfile, pb: ColumnProfile, key_a: str, key_b: str, index: EmbeddingIndex | None, vendor_prefixes: tuple[str, ...] = ()) -> tuple[float, dict[str, float]]:
    concept = concept_similarity(pa.name, pb.name, vendor_prefixes)
    semtype = semantic_type_compatibility(pa.semantic_type, pb.semantic_type)
    emb = index.cosine(key_a, key_b) if index is not None else 0.0
    score = 0.4 * concept + 0.3 * semtype + 0.3 * emb
    return round(score, 4), {"concept": round(concept, 4), "semantic_type": semtype, "embedding": round(emb, 4)}
