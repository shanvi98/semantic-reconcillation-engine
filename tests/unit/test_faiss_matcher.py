"""Uses HashingEmbedder (offline, deterministic) so this exercises the real
index/search code path without requiring a model download in CI.
"""
import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest

from reconciler.matchers.base import MatchCandidate
from reconciler.matchers.faiss_matcher import (
    FaissMatcher,
    HashingEmbedder,
    _search,
)
from reconciler.schemas import SourceType


@pytest.fixture
def no_faiss(monkeypatch):
    """Make `import faiss` fail, forcing the numpy search path."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("faiss is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


def _matcher(confidence_threshold: float = 0.55) -> FaissMatcher:
    matcher = FaissMatcher(embedder=HashingEmbedder())
    # HashingEmbedder is a raw n-gram bag, not a dense semantic embedding, so its
    # cosine scores run much lower than production sentence-transformer output.
    # Isolate ranking correctness here; the real 0.55 bar is validated end-to-end
    # in tests/evaluation with the production stack (or offline against ground truth).
    matcher.confidence_threshold = confidence_threshold
    return matcher


def _normalized(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((rows, dim)).astype("float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---- search backends --------------------------------------------------------

def test_numpy_and_faiss_search_paths_agree(no_faiss):
    """The fallback is only legitimate if it is an exact substitution.
    IndexFlatIP is itself an exhaustive inner-product scan, so both paths must
    return the same neighbours in the same order — otherwise a core-only
    install would silently reconcile differently from a full one.
    """
    query, corpus = _normalized(12, 32, seed=0), _normalized(40, 32, seed=1)

    numpy_scores, numpy_indices = _search(query, corpus, 3)

    faiss = pytest.importorskip("faiss", reason="faiss not installed; nothing to compare against")
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)
    faiss_scores, faiss_indices = index.search(query, 3)

    assert np.array_equal(numpy_indices, faiss_indices)
    assert np.allclose(numpy_scores, faiss_scores, atol=1e-6)


def test_search_returns_neighbours_in_descending_score_order():
    query, corpus = _normalized(5, 16, seed=2), _normalized(20, 16, seed=3)
    scores, _ = _search(query, corpus, 4)
    for row in scores:
        assert list(row) == sorted(row, reverse=True)


def test_matcher_works_without_faiss_installed(no_faiss):
    left = [MatchCandidate(id="INV-001", text="Acme Corp Invoice #012", amount=1000.0)]
    right = [
        MatchCandidate(id="STL-001", text="RAZORP*ACME012", amount=1000.0),
        MatchCandidate(id="STL-002", text="RAZORP*ZENITH099", amount=9999.0),
    ]
    matches = _matcher(0.05).resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]
    assert [m.right_id for m in matches] == ["STL-001"]


# ---- ranking ----------------------------------------------------------------

def test_finds_nearest_neighbor_above_threshold():
    left = [MatchCandidate(id="INV-001", text="Acme Corp Invoice #012", amount=1000.0)]
    right = [
        MatchCandidate(id="STL-001", text="RAZORP*ACME012", amount=1000.0),
        MatchCandidate(id="STL-002", text="RAZORP*ZENITH099", amount=9999.0),
    ]

    matches = _matcher(confidence_threshold=0.05).resolve(
        left, right, SourceType.INVOICE, SourceType.RAZORPAY
    )[0]
    assert len(matches) == 1
    assert matches[0].right_id == "STL-001"


def test_one_right_record_is_never_claimed_twice():
    left = [
        MatchCandidate(id="INV-001", text="Acme Corp Invoice #012", amount=1000.0),
        MatchCandidate(id="INV-002", text="Acme Corp Invoice #012", amount=1000.0),
    ]
    right = [MatchCandidate(id="STL-001", text="RAZORP*ACME012", amount=1000.0)]
    matches = _matcher(0.01).resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]
    assert len(matches) == 1


def test_more_neighbours_requested_than_exist_is_handled():
    left = [MatchCandidate(id="INV-001", text="Acme", amount=1.0)]
    right = [MatchCandidate(id="STL-001", text="Acme", amount=1.0)]
    matcher = FaissMatcher(embedder=HashingEmbedder(), neighbours=10)
    matcher.confidence_threshold = 0.01
    assert len(matcher.resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]) == 1


def test_empty_inputs_return_no_matches():
    assert _matcher().resolve([], [], SourceType.INVOICE, SourceType.RAZORPAY)[0] == []
    left = [MatchCandidate(id="INV-001", text="anything", amount=100.0)]
    assert _matcher().resolve(left, [], SourceType.INVOICE, SourceType.RAZORPAY)[0] == []


# ---- embedder ---------------------------------------------------------------

def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder()
    v1 = embedder.encode(["Acme Corp Invoice #012"])
    v2 = embedder.encode(["Acme Corp Invoice #012"])
    assert (v1 == v2).all()


def test_hashing_embedder_output_is_normalized():
    vecs = HashingEmbedder().encode(["Acme Corp", "Zenith Furniture"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_hashing_embedder_handles_empty_text():
    """An empty narration yields a zero vector; normalizing it must not divide
    by zero and produce NaNs that poison every downstream comparison."""
    vecs = HashingEmbedder().encode([""])
    assert not np.isnan(vecs).any()


def test_hashing_embedder_rejects_a_zero_dimension():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
