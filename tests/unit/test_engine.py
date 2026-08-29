"""The engine's contract is the escalation itself: each tier sees only what the
previous one could not resolve, nothing is dropped between tiers, and what
survives all of them is handed on with an explanation rather than in silence.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from reconciler.engine import (
    ReconciliationEngine,
    bank_to_candidate,
    find_near_misses,
    invoice_to_candidate,
    settlement_to_candidate,
)
from reconciler.matchers.base import BaseMatcher, MatchCandidate
from reconciler.schemas import (
    BankRecord,
    InvoiceRecord,
    MatchTier,
    PairMatch,
    SettlementRecord,
    SourceType,
)


class ScriptedMatcher(BaseMatcher):
    """Matches exactly the pairs it is told to, and records what it was handed."""

    def __init__(self, tier: MatchTier, pairs: list[tuple[str, str]]):
        self.tier = tier
        self.confidence_threshold = 0.0
        self.pairs = pairs
        self.seen_left: list[list[str]] = []

    def find_matches(self, left, right):
        self.seen_left.append([c.id for c in left])
        left_ids = {c.id for c in left}
        right_ids = {c.id for c in right}
        return [
            PairMatch(left_id=left_id, right_id=right_id, tier=self.tier, confidence=0.9)
            for left_id, right_id in self.pairs
            if left_id in left_ids and right_id in right_ids
        ]


def _candidates(prefix, n, amount=100.0):
    return [MatchCandidate(id=f"{prefix}-{i}", text=f"text {i}", amount=amount) for i in range(n)]


# ---- escalation -------------------------------------------------------------

def test_each_tier_only_sees_what_the_previous_one_left():
    left, right = _candidates("L", 3), _candidates("R", 3)
    tier1 = ScriptedMatcher(MatchTier.RULE, [("L-0", "R-0")])
    tier2 = ScriptedMatcher(MatchTier.TFIDF, [("L-1", "R-1")])
    tier3 = ScriptedMatcher(MatchTier.FAISS, [])

    engine = ReconciliationEngine(matchers=[tier1, tier2, tier3])
    result = engine.match_pair(left, right, SourceType.INVOICE, SourceType.RAZORPAY)

    assert tier1.seen_left == [["L-0", "L-1", "L-2"]]
    assert tier2.seen_left == [["L-1", "L-2"]]
    assert tier3.seen_left == [["L-2"]]
    assert len(result.matches) == 2
    assert [c.id for c in result.unmatched_left] == ["L-2"]


def test_escalation_stops_early_when_everything_resolves():
    left, right = _candidates("L", 1), _candidates("R", 1)
    tier1 = ScriptedMatcher(MatchTier.RULE, [("L-0", "R-0")])
    tier2 = ScriptedMatcher(MatchTier.LLM, [])

    ReconciliationEngine(matchers=[tier1, tier2]).match_pair(
        left, right, SourceType.INVOICE, SourceType.RAZORPAY
    )
    assert tier2.seen_left == [], "the expensive tier ran despite nothing being left for it"


def test_tier_stats_cover_every_tier_even_ones_that_never_ran():
    """A breakdown that silently omits short-circuited tiers reads as though
    those tiers do not exist, rather than as though they had nothing to do."""
    left, right = _candidates("L", 1), _candidates("R", 1)
    engine = ReconciliationEngine(matchers=[
        ScriptedMatcher(MatchTier.RULE, [("L-0", "R-0")]),
        ScriptedMatcher(MatchTier.LLM, []),
    ])
    result = engine.match_pair(left, right, SourceType.INVOICE, SourceType.RAZORPAY)
    assert {s.tier for s in result.tier_stats} == {"rule", "llm"}


def test_matches_are_tagged_with_their_source_and_tier():
    left, right = _candidates("L", 1), _candidates("R", 1)
    engine = ReconciliationEngine(matchers=[ScriptedMatcher(MatchTier.RULE, [("L-0", "R-0")])])
    match = engine.match_pair(left, right, SourceType.INVOICE, SourceType.RAZORPAY).matches[0]
    assert match.left_source is SourceType.INVOICE
    assert match.right_source is SourceType.RAZORPAY
    assert match.tier is MatchTier.RULE


def test_engine_requires_at_least_one_matcher():
    with pytest.raises(ValueError):
        ReconciliationEngine(matchers=[])


# ---- near misses ------------------------------------------------------------

def test_near_miss_is_reported_for_unresolved_records():
    left = [MatchCandidate(id="INV-1", text="ACME CORP INVOICE 012", amount=1000.0)]
    right = [
        MatchCandidate(id="STL-1", text="ACME CORP INVOICE 012", amount=1000.0),
        MatchCandidate(id="STL-2", text="TOTALLY UNRELATED", amount=9.0),
    ]
    near = find_near_misses(left, right)
    assert near["INV-1"].candidate_id == "STL-1"
    assert near["INV-1"].score > 0
    assert "amount agrees" in near["INV-1"].reason


def test_near_miss_search_includes_already_claimed_records():
    """The most useful hint is often 'this looks like a record another invoice
    already took' — which is invisible if you only search the leftovers."""
    left, right = _candidates("L", 3), _candidates("R", 3)
    engine = ReconciliationEngine(matchers=[ScriptedMatcher(MatchTier.RULE, [("L-0", "R-0")])])
    result = engine.match_pair(left, right, SourceType.INVOICE, SourceType.RAZORPAY)
    assert result.near_misses, "unmatched records surfaced no leads at all"


def test_near_misses_can_be_switched_off():
    left, right = _candidates("L", 2), _candidates("R", 2)
    engine = ReconciliationEngine(
        matchers=[ScriptedMatcher(MatchTier.RULE, [])], collect_near_misses=False
    )
    assert engine.match_pair(left, right, SourceType.INVOICE, SourceType.RAZORPAY).near_misses == {}


def test_near_misses_are_empty_when_nothing_is_close():
    left = [MatchCandidate(id="INV-1", text="ZZZZ", amount=1.0)]
    right = [MatchCandidate(id="STL-1", text="QQQQ", amount=999_999.0)]
    assert find_near_misses(left, right) == {}


# ---- record adapters --------------------------------------------------------

def test_invoice_adapter_extracts_the_invoice_number():
    record = InvoiceRecord("INV-012", "Acme Corp", date(2026, 6, 1), 1000.0, "Acme Corp Invoice #012")
    assert invoice_to_candidate(record).numeric_ref == "012"


def test_settlement_adapter_uses_the_right_amount_per_leg():
    """The invoice leg compares against what was billed; the bank leg against
    what actually landed after PG fees. Using one for both is the classic
    three-way reconciliation error."""
    record = SettlementRecord(
        "STL-1", "order_X", "UTR123456789012", date(2026, 6, 2),
        gross_amount=1000.0, fee=20.0, tax=3.6, settled_amount=976.4,
        merchant_ref="RAZORP*ACME012",
    )
    assert settlement_to_candidate(record, side="invoice_leg").amount == 1000.0
    assert settlement_to_candidate(record, side="bank_leg").amount == 976.4


def test_settlement_adapter_rejects_an_unknown_leg():
    record = SettlementRecord(
        "STL-1", "order_X", None, date(2026, 6, 2), 1.0, 0.0, 0.0, 1.0, "ref",
    )
    with pytest.raises(ValueError):
        settlement_to_candidate(record, side="sideways")


def test_bank_adapter_folds_the_utr_into_the_identifiers():
    record = BankRecord("BNK-1", date(2026, 6, 3), "NEFT-RAZORP*ACME012", 976.4, "UTR123456789012")
    assert "UTR123456789012" in bank_to_candidate(record).alnum_ids


def test_adapters_survive_records_with_empty_free_text():
    record = BankRecord("BNK-1", date(2026, 6, 3), "", 100.0, None)
    candidate = bank_to_candidate(record)
    assert candidate.id == "BNK-1"
    assert candidate.alnum_ids == frozenset()
