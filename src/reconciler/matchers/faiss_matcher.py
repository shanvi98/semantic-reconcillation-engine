"""Tier 3: dense semantic matching via sentence embeddings + nearest-neighbor search.

Catches paraphrase-level matches TF-IDF misses, e.g. a settlement merchant_ref
built from a legal entity abbreviation that shares almost no n-grams with the
customer name on the invoice, but is semantically the same company.

Search runs through FAISS when it is installed and falls back to an equivalent
numpy computation when it is not. That is a faithful substitution rather than a
downgrade: ``IndexFlatIP`` is itself an exhaustive inner-product scan, so both
paths return the same neighbours in the same order. It means the tier — and the
tests covering it — work in a core-only install without the multi-hundred-
megabyte faiss/torch wheels, while still using the real index when available.
"""
from __future__ import annotations

import logging
import zlib
from typing import Protocol

import numpy as np

from ..schemas import MatchTier, PairMatch
from ..utils.normalization import amounts_match, clean_text
from .base import BaseMatcher, MatchCandidate

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalized embeddings."""
        ...


class SentenceTransformerEmbedder:
    """Default production embedder. Model is loaded lazily so importing this
    module never requires a network call / model download."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(texts, normalize_embeddings=True)
        return np.ascontiguousarray(vecs, dtype="float32")


class HashingEmbedder:
    """Deterministic, offline, dependency-light embedder used in unit tests and
    CI environments without model-download access. Not semantically strong —
    it's a char n-gram hashing bag-of-features, not a learned embedding — but
    it exercises the exact same index/search code path."""

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (2, 4)):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.ngram_range = ngram_range

    def _ngrams(self, text: str) -> list[str]:
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams.extend(text[i:i + n] for i in range(max(len(text) - n + 1, 0)))
        return grams

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for gram in self._ngrams(clean_text(text)):
                # crc32, not the builtin hash(): PYTHONHASHSEED salting would make
                # hash() differ across processes, breaking reproducibility across CI runs.
                bucket = zlib.crc32(gram.encode("utf-8")) % self.dim
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def _search(query: np.ndarray, corpus: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k inner-product neighbours of each query row within corpus.

    Uses a FAISS IndexFlatIP when available and an equivalent numpy argpartition
    otherwise. Both are exhaustive exact searches over the same normalized
    vectors, so results agree.
    """
    try:
        import faiss
    except ImportError:
        logger.debug("faiss not installed — using the equivalent numpy search path")
    else:
        index = faiss.IndexFlatIP(corpus.shape[1])
        index.add(np.ascontiguousarray(corpus, dtype="float32"))
        return index.search(np.ascontiguousarray(query, dtype="float32"), k)

    scores = query @ corpus.T
    # argpartition for the top-k, then sort just that slice — O(n) rather than
    # sorting every candidate, which matters once a batch gets large.
    top = np.argpartition(-scores, kth=min(k, scores.shape[1]) - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    indices = np.take_along_axis(top, ordered, axis=1)
    return np.take_along_axis(scores, indices, axis=1), indices


class FaissMatcher(BaseMatcher):
    tier = MatchTier.FAISS
    confidence_threshold = 0.55

    def __init__(
        self,
        embedder: Embedder | None = None,
        amount_bonus: float = 0.1,
        neighbours: int = 3,
    ):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.amount_bonus = amount_bonus
        self.neighbours = neighbours

    def find_matches(self, left: list[MatchCandidate], right: list[MatchCandidate]) -> list[PairMatch]:
        if not left or not right:
            return []

        left_vecs = self.embedder.encode([clean_text(c.text) for c in left])
        right_vecs = self.embedder.encode([clean_text(c.text) for c in right])

        k = min(self.neighbours, len(right))
        scores, indices = _search(left_vecs, right_vecs, k)

        # Rank all (left, candidate) pairs globally, same greedy strategy as the
        # TF-IDF tier, so a strong pair is never lost to input ordering.
        candidates = []
        for li in range(len(left)):
            for rank in range(k):
                ri = int(indices[li, rank])
                if ri < 0:  # faiss pads with -1 when fewer than k neighbours exist
                    continue
                candidates.append((float(scores[li, rank]), li, ri))
        # Ties break on ids so the assignment is stable across runs.
        candidates.sort(key=lambda t: (-t[0], left[t[1]].id, right[t[2]].id))

        matches: list[PairMatch] = []
        used_left: set[int] = set()
        used_right: set[int] = set()

        for score, li, ri in candidates:
            if li in used_left or ri in used_right:
                continue
            confidence = score
            if amounts_match(left[li].amount, right[ri].amount):
                confidence = min(1.0, confidence + self.amount_bonus)

            if confidence >= self.confidence_threshold:
                matches.append(PairMatch(
                    left_id=left[li].id,
                    right_id=right[ri].id,
                    tier=self.tier,
                    confidence=round(confidence, 4),
                    rationale=f"embedding cosine={score:.3f}",
                ))
                used_left.add(li)
                used_right.add(ri)

        return matches
