import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reconciler.matchers.base import MatchCandidate
from reconciler.matchers.tfidf_matcher import TfidfMatcher
from reconciler.schemas import SourceType


def test_matches_lexically_similar_truncated_text():
    left = [MatchCandidate(id="INV-001", text="Acme Corp Invoice #012", amount=1000.0)]
    right = [MatchCandidate(id="STL-001", text="RAZORP*ACME012", amount=1000.0)]

    matches = TfidfMatcher().resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]
    assert len(matches) == 1
    assert matches[0].left_id == "INV-001" and matches[0].right_id == "STL-001"


def test_no_match_for_unrelated_text():
    left = [MatchCandidate(id="INV-001", text="Zenith Furniture Invoice #099", amount=1000.0)]
    right = [MatchCandidate(id="STL-001", text="RAZORP*BLUE045", amount=1000.0)]

    matches = TfidfMatcher(amount_bonus=0.0).resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]
    assert matches == []


def test_greedy_assignment_avoids_double_booking_best_candidate():
    left = [
        MatchCandidate(id="INV-001", text="Acme Corp Invoice #012", amount=1000.0),
        MatchCandidate(id="INV-002", text="Acme Corp Invoice #013", amount=2000.0),
    ]
    right = [
        MatchCandidate(id="STL-001", text="RAZORP*ACME012", amount=1000.0),
        MatchCandidate(id="STL-002", text="RAZORP*ACME013", amount=2000.0),
    ]

    matches = TfidfMatcher().resolve(left, right, SourceType.INVOICE, SourceType.RAZORPAY)[0]
    pairing = {m.left_id: m.right_id for m in matches}
    assert pairing == {"INV-001": "STL-001", "INV-002": "STL-002"}
