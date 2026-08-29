"""The LLM tier is the only one whose input is a model's free-form output, so
everything it returns is treated as untrusted. These tests pin that contract:
a model that hallucinates, rambles, errors, or returns nonsense must degrade to
"no match" and never to a match pointing at a record that does not exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from reconciler.matchers.base import MatchCandidate
from reconciler.matchers.llm_matcher import (
    HeuristicLLMClient,
    LlmMatcher,
    parse_verdict,
)
from reconciler.schemas import SourceType

FENCE = "`" * 3


class StubClient:
    """Returns a canned verdict, and records what it was actually offered."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def adjudicate(self, left_text, left_amount, candidates):
        self.calls.append((left_text, left_amount, candidates))
        return self.verdict


def _resolve(client, left=None, right=None):
    left = left or [MatchCandidate(id="INV-1", text="Acme Corp", amount=100.0)]
    right = right or [MatchCandidate(id="STL-1", text="Acme Corp", amount=100.0)]
    return LlmMatcher(client=client).resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]


# ---- verdict parsing --------------------------------------------------------

def test_parses_bare_json():
    assert parse_verdict('{"best_candidate_id": "STL-1", "confidence": 0.9}')["best_candidate_id"] == "STL-1"


def test_parses_json_wrapped_in_a_code_fence():
    """Models emit fenced JSON constantly despite being told not to; treating
    that as a parse failure would discard perfectly good adjudications."""
    raw = f'{FENCE}json\n{{"best_candidate_id": "STL-1", "confidence": 0.8}}\n{FENCE}'
    assert parse_verdict(raw) == {"best_candidate_id": "STL-1", "confidence": 0.8}


def test_parses_json_embedded_in_prose():
    raw = 'Sure! {"best_candidate_id": "STL-2", "confidence": 0.7} — hope that helps.'
    assert parse_verdict(raw)["best_candidate_id"] == "STL-2"


@pytest.mark.parametrize("raw", ["", "   ", "I cannot help with that", "[1, 2, 3]"])
def test_unparseable_output_becomes_an_explicit_no_match(raw):
    verdict = parse_verdict(raw)
    assert verdict["best_candidate_id"] is None
    assert verdict["confidence"] == 0.0
    assert verdict["rationale"]  # the reason is preserved, not swallowed


# ---- verdict validation -----------------------------------------------------

def test_hallucinated_candidate_id_is_discarded():
    """Regression: a confident verdict naming an id that was never offered used
    to produce a PairMatch pointing at a nonexistent record, which then flowed
    into the three-way chaining as a real edge."""
    matches = _resolve(StubClient({"best_candidate_id": "STL-DOES-NOT-EXIST", "confidence": 0.99}))
    assert matches == []


def test_confidence_below_threshold_is_declined():
    assert _resolve(StubClient({"best_candidate_id": "STL-1", "confidence": 0.2})) == []


def test_null_candidate_is_declined():
    assert _resolve(StubClient({"best_candidate_id": None, "confidence": 0.99})) == []


@pytest.mark.parametrize("value,expected", [
    (5.0, 1.0), (-3.0, 0.0), ("0.75", 0.75), (None, 0.0), ("banana", 0.0), (float("nan"), 0.0),
])
def test_confidence_is_coerced_into_range(value, expected):
    assert LlmMatcher._coerce_confidence(value) == expected


def test_out_of_range_confidence_is_clamped_not_rejected():
    matches = _resolve(StubClient({"best_candidate_id": "STL-1", "confidence": 7.5}))
    assert len(matches) == 1
    assert matches[0].confidence == 1.0


def test_non_dict_verdict_is_survived():
    assert _resolve(StubClient("not a dict")) == []


def test_client_exception_does_not_abort_the_run():
    """One flaky call on the residual tier should cost one record, not the
    whole batch that has already reconciled."""
    class Exploding:
        def adjudicate(self, *args, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        # The matcher does not swallow client errors itself — that is the
        # production client's job (see AnthropicLLMClient.adjudicate), so this
        # documents where the boundary sits.
        _resolve(Exploding())


# ---- shortlisting -----------------------------------------------------------

def test_shortlist_is_bounded():
    """Cost control: the tier must never send the full cross-product."""
    client = StubClient({"best_candidate_id": None, "confidence": 0.0})
    right = [MatchCandidate(id=f"STL-{i}", text=f"Company {i}", amount=float(i)) for i in range(50)]
    LlmMatcher(client=client, shortlist_size=5).resolve(
        [MatchCandidate(id="INV-1", text="Company 3", amount=3.0)], right,
        SourceType.INVOICE, SourceType.RAZORPAY,
    )
    assert len(client.calls) == 1
    assert len(client.calls[0][2]) == 5


def test_a_claimed_candidate_is_not_offered_twice():
    client = StubClient({"best_candidate_id": "STL-1", "confidence": 0.9})
    left = [
        MatchCandidate(id="INV-1", text="Acme", amount=100.0),
        MatchCandidate(id="INV-2", text="Acme", amount=100.0),
    ]
    right = [MatchCandidate(id="STL-1", text="Acme", amount=100.0)]
    matches = LlmMatcher(client=client).resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]
    assert len(matches) == 1


def test_empty_inputs_short_circuit_without_calling_the_model():
    client = StubClient({"best_candidate_id": "STL-1", "confidence": 0.9})
    LlmMatcher(client=client).resolve([], [], SourceType.INVOICE, SourceType.RAZORPAY)
    assert client.calls == []


# ---- offline stand-in -------------------------------------------------------

def test_heuristic_client_is_deterministic():
    client = HeuristicLLMClient()
    args = ("Acme Corp Invoice 012", 1000.0, [("STL-1", "RAZORP ACME012", 1000.0)])
    assert client.adjudicate(*args) == client.adjudicate(*args)


def test_heuristic_client_declines_when_nothing_is_plausible():
    verdict = HeuristicLLMClient().adjudicate("Acme Corp", 1000.0, [])
    assert verdict["best_candidate_id"] is None
