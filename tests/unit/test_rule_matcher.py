import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reconciler.matchers.base import MatchCandidate
from reconciler.matchers.rule_matcher import RuleMatcher
from reconciler.schemas import SourceType


def _resolve(left, right):
    return RuleMatcher().resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)


def test_matches_on_shared_alnum_id_with_amount_agreement():
    left = [MatchCandidate(id="STL-001", text="settlement", amount=1000.0, alnum_ids=frozenset({"UTR123456789"}))]
    right = [MatchCandidate(id="BNK-001", text="bank", amount=1000.0, alnum_ids=frozenset({"UTR123456789"}))]

    matches = RuleMatcher().resolve(left, right, SourceType.RAZORPAY, SourceType.BANK)[0]
    assert len(matches) == 1
    assert matches[0].confidence >= 0.95


def test_matches_on_numeric_ref_embedded_in_merchant_token():
    left = [MatchCandidate(id="INV-012", text="invoice", amount=5000.0, numeric_ref="012")]
    right = [MatchCandidate(id="STL-012", text="settlement", amount=5000.0, numeric_ref="012")]

    matches = _resolve(left, right)[0]
    assert len(matches) == 1
    assert matches[0].left_id == "INV-012" and matches[0].right_id == "STL-012"


def test_no_match_when_no_shared_identifier():
    left = [MatchCandidate(id="INV-001", text="invoice", amount=1000.0, numeric_ref="001")]
    right = [MatchCandidate(id="STL-999", text="settlement", amount=1000.0, numeric_ref="999")]

    matches, unmatched_left, unmatched_right = _resolve(left, right)
    assert matches == []
    assert unmatched_left == left
    assert unmatched_right == right


def test_falls_through_to_next_tier_when_id_overlaps_but_amount_disagrees():
    """ID overlap present but amount is wildly off — confidence drops below the
    tier's 0.95 auto-accept bar, so this tier declines it and passes it on
    unresolved rather than auto-accepting a shaky match."""
    left = [MatchCandidate(id="INV-001", text="invoice", amount=1000.0, numeric_ref="001")]
    right = [MatchCandidate(id="STL-001", text="settlement", amount=5000.0, numeric_ref="001")]

    matches, unmatched_left, unmatched_right = _resolve(left, right)
    assert matches == []
    assert unmatched_left == left
    assert unmatched_right == right


def test_declined_match_does_not_consume_its_candidate():
    """Regression: a pair scored below the auto-accept bar must leave both
    records free.

    A previous implementation marked the right candidate as used *before*
    filtering on confidence, so INV-A's rejected 0.90 match silently starved
    INV-B of the only settlement that genuinely matched it — the record was
    never matched here and never offered to a later tier either.
    """
    left = [
        MatchCandidate(id="INV-A", text="a", amount=1_000.0, numeric_ref="001"),   # id hit, amount contradicts
        MatchCandidate(id="INV-B", text="b", amount=5_000.0, numeric_ref="001"),   # id hit, amount agrees
    ]
    right = [MatchCandidate(id="STL-1", text="s", amount=5_000.0, numeric_ref="001")]

    matches, unmatched_left, _ = _resolve(left, right)

    assert [(m.left_id, m.right_id) for m in matches] == [("INV-B", "STL-1")]
    assert [c.id for c in unmatched_left] == ["INV-A"]


def test_assignment_is_global_best_first_not_input_order():
    """A weaker left record appearing first must not claim a right candidate
    that a stronger pair needs."""
    left = [
        MatchCandidate(id="INV-WEAK", text="w", amount=999.0, alnum_ids=frozenset({"SHARED01"})),
        MatchCandidate(id="INV-EXACT", text="e", amount=1_000.0, alnum_ids=frozenset({"SHARED01"})),
    ]
    right = [MatchCandidate(id="STL-1", text="s", amount=1_000.0, alnum_ids=frozenset({"SHARED01"}))]

    matches = _resolve(left, right)[0]
    assert [(m.left_id, m.right_id) for m in matches] == [("INV-EXACT", "STL-1")]


def test_one_right_record_is_never_claimed_twice():
    left = [
        MatchCandidate(id="INV-1", text="a", amount=1_000.0, alnum_ids=frozenset({"SHARED01"})),
        MatchCandidate(id="INV-2", text="b", amount=1_000.0, alnum_ids=frozenset({"SHARED01"})),
    ]
    right = [MatchCandidate(id="STL-1", text="s", amount=1_000.0, alnum_ids=frozenset({"SHARED01"}))]

    matches = _resolve(left, right)[0]
    assert len(matches) == 1


def test_empty_inputs_return_no_matches():
    assert _resolve([], [])[0] == []
    assert _resolve([MatchCandidate(id="INV-1", text="a", amount=1.0, numeric_ref="001")], [])[0] == []


def test_result_is_stable_regardless_of_input_ordering():
    """Reconciliation output must not depend on the order rows happened to
    arrive in — a bank statement is not sorted in invoice order."""
    left = [
        MatchCandidate(id="INV-1", text="a", amount=1_000.0, alnum_ids=frozenset({"IDAAA1"})),
        MatchCandidate(id="INV-2", text="b", amount=2_000.0, alnum_ids=frozenset({"IDBBB2"})),
        MatchCandidate(id="INV-3", text="c", amount=3_000.0, alnum_ids=frozenset({"IDCCC3"})),
    ]
    right = [
        MatchCandidate(id="STL-1", text="a", amount=1_000.0, alnum_ids=frozenset({"IDAAA1"})),
        MatchCandidate(id="STL-2", text="b", amount=2_000.0, alnum_ids=frozenset({"IDBBB2"})),
        MatchCandidate(id="STL-3", text="c", amount=3_000.0, alnum_ids=frozenset({"IDCCC3"})),
    ]

    forward = {m.left_id: m.right_id for m in _resolve(left, right)[0]}
    reversed_ = {m.left_id: m.right_id for m in _resolve(left[::-1], right[::-1])[0]}
    assert forward == reversed_ == {"INV-1": "STL-1", "INV-2": "STL-2", "INV-3": "STL-3"}
