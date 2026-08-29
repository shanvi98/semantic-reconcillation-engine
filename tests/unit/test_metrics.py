"""Metrics are the claim this project makes about itself, so they get tested
like any other output. The cases that matter most are the ones where a metric
could flatter the engine: a false positive against a ground-truth row with no
triangle, and a match rate measured against an unreachable ceiling.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from reconciler.engine import PairResolution
from reconciler.pipeline import PipelineResult
from reconciler.reporting.metrics import (
    compute_metrics,
    score_against_ground_truth,
    tier_breakdown,
    tier_timings,
)
from reconciler.schemas import (
    ExceptionCategory,
    ExceptionRecord,
    MatchTier,
    PairMatch,
    SourceType,
    ThreeWayMatch,
)


def _result(triangles=(), exceptions=(), matches=()):
    empty = PairResolution(matches=list(matches), unmatched_left=[], unmatched_right=[], tier_stats=[])
    blank = PairResolution(matches=[], unmatched_left=[], unmatched_right=[], tier_stats=[])
    return PipelineResult(
        three_way_matches=list(triangles),
        exceptions=list(exceptions),
        invoice_settlement=empty,
        settlement_bank=blank,
    )


def _triangle(inv, stl, bank, confidence=0.9):
    return ThreeWayMatch(invoice_id=inv, settlement_id=stl, bank_txn_id=bank, overall_confidence=confidence)


# ---- accuracy ---------------------------------------------------------------

def test_perfect_prediction_scores_perfectly():
    gt = [{"invoice_id": "INV-1", "settlement_id": "STL-1", "bank_txn_id": "BNK-1"}]
    report = score_against_ground_truth(_result([_triangle("INV-1", "STL-1", "BNK-1")]), gt)
    assert (report.correct, report.incorrect, report.missed) == (1, 0, 0)
    assert report.precision == report.recall == report.f1 == 1.0


def test_wrong_counterparty_counts_as_incorrect_not_missed():
    gt = [{"invoice_id": "INV-1", "settlement_id": "STL-1", "bank_txn_id": "BNK-1"}]
    report = score_against_ground_truth(_result([_triangle("INV-1", "STL-9", "BNK-9")]), gt)
    assert (report.correct, report.incorrect, report.missed) == (0, 1, 0)
    assert report.precision == 0.0


def test_missing_a_real_triangle_counts_as_missed():
    gt = [{"invoice_id": "INV-1", "settlement_id": "STL-1", "bank_txn_id": "BNK-1"}]
    report = score_against_ground_truth(_result([]), gt)
    assert (report.correct, report.incorrect, report.missed) == (0, 0, 1)
    assert report.recall == 0.0


def test_matching_a_record_that_has_no_ground_truth_triangle_is_a_false_positive():
    """Regression: an invoice the generator deliberately left un-settleable used
    to be skipped entirely, so inventing a match for it cost nothing. That let
    precision stay at 1.0 while the engine fabricated matches for records with
    no counterparty at all — the exact failure precision exists to catch.
    """
    gt = [{"invoice_id": "INV-1", "settlement_id": None, "bank_txn_id": None}]
    report = score_against_ground_truth(_result([_triangle("INV-1", "STL-7", "BNK-7")]), gt)
    assert report.incorrect == 1
    assert report.precision == 0.0


def test_correctly_declining_an_unresolvable_record_is_not_penalised():
    gt = [{"invoice_id": "INV-1", "settlement_id": None, "bank_txn_id": None}]
    report = score_against_ground_truth(_result([]), gt)
    assert (report.correct, report.incorrect, report.missed) == (0, 0, 0)


def test_empty_ground_truth_does_not_divide_by_zero():
    report = score_against_ground_truth(_result([]), [])
    assert report.precision == report.recall == report.f1 == 0.0


# ---- match rates ------------------------------------------------------------

def test_achievable_match_rate_ignores_unreachable_triangles():
    """The raw match rate is capped below 1.0 by design-dropped legs, so it can
    never answer 'did we find everything findable'. The achievable rate does."""
    gt = [
        {"invoice_id": "INV-1", "settlement_id": "STL-1", "bank_txn_id": "BNK-1"},
        {"invoice_id": "INV-2", "settlement_id": None, "bank_txn_id": None},
    ]
    metrics = compute_metrics(
        _result([_triangle("INV-1", "STL-1", "BNK-1")]),
        total_invoices=2, total_settlements=1, total_bank_records=1,
        elapsed_seconds=1.0, ground_truth=gt,
    )
    assert metrics.match_rate == 0.5           # 1 of 2 invoices
    assert metrics.achievable_match_rate == 1.0  # 1 of 1 resolvable triangles


def test_achievable_match_rate_is_none_without_ground_truth():
    metrics = compute_metrics(_result([]), 1, 1, 1, elapsed_seconds=1.0)
    assert metrics.achievable_match_rate is None
    assert metrics.accuracy is None


def test_zero_elapsed_time_does_not_divide_by_zero():
    metrics = compute_metrics(_result([]), 1, 1, 1, elapsed_seconds=0.0)
    assert metrics.throughput_records_per_sec == 0.0


def test_zero_invoices_does_not_divide_by_zero():
    assert compute_metrics(_result([]), 0, 0, 0, elapsed_seconds=1.0).match_rate == 0.0


# ---- breakdowns -------------------------------------------------------------

def test_tier_breakdown_lists_every_tier_including_empty_ones():
    breakdown = tier_breakdown(_result(matches=[PairMatch("L", "R", tier=MatchTier.RULE)]))
    assert breakdown == {"rule": 1, "tfidf": 0, "faiss": 0, "llm": 0}


def test_exception_value_and_categories_are_reported():
    exceptions = [
        ExceptionRecord("INV-1", SourceType.INVOICE, "r", 100.0, ExceptionCategory.PENDING_SETTLEMENT),
        ExceptionRecord("INV-2", SourceType.INVOICE, "r", 250.0, ExceptionCategory.PENDING_SETTLEMENT),
        ExceptionRecord("BNK-1", SourceType.BANK, "r", 50.0, ExceptionCategory.UNMATCHED_BANK_CREDIT),
    ]
    metrics = compute_metrics(_result(exceptions=exceptions), 2, 0, 1, elapsed_seconds=1.0)
    assert metrics.exception_value == pytest.approx(400.0)
    assert metrics.exceptions_by_category == {"pending_settlement": 2, "unmatched_bank_credit": 1}


# ── Cascade shape ────────────────────────────────────────────────────────────
# The escalation diagram is a claim about what flowed through the engine, so the
# numbers behind it have to come from the engine. An earlier version of the
# diagram inferred them by summing match counts and working backwards, which
# silently overstated what reached the later tiers: the derivation cannot know
# that a leg leaves the cascade early once one side is exhausted.

def test_tier_timings_carry_the_measured_candidate_counts(pipeline_result):
    timings = {t.tier: t for t in tier_timings(pipeline_result)}

    legs = (pipeline_result.invoice_settlement, pipeline_result.settlement_bank)
    for tier, timing in timings.items():
        expected_in = sum(
            stat.candidates_in for leg in legs for stat in leg.tier_stats if stat.tier == tier
        )
        expected_out = sum(
            stat.candidates_out for leg in legs for stat in leg.tier_stats if stat.tier == tier
        )
        assert timing.candidates_in == expected_in, f"{tier} entry count is not the measured one"
        assert timing.candidates_out == expected_out, f"{tier} exit count is not the measured one"


def test_the_first_tier_is_handed_every_left_hand_record(pipeline_result, synthetic_batch):
    """Both legs drive from the left: invoices, then settlements."""
    rule = next(t for t in tier_timings(pipeline_result) if t.tier == "rule")
    expected = len(synthetic_batch["invoices"]) + len(synthetic_batch["settlements"])
    assert rule.candidates_in == expected


def test_a_tier_never_receives_more_than_the_previous_one_passed_on(pipeline_result):
    timings = tier_timings(pipeline_result)
    for earlier, later in itertools.pairwise(timings):
        assert later.candidates_in <= earlier.candidates_out, (
            f"{later.tier} claims {later.candidates_in} in, but {earlier.tier} "
            f"only passed on {earlier.candidates_out}"
        )


def test_unresolved_counts_come_from_the_legs_not_the_exception_categories(pipeline_result):
    metrics = compute_metrics(pipeline_result, 60, 57, 63, elapsed_seconds=1.0)
    legs = (pipeline_result.invoice_settlement, pipeline_result.settlement_bank)

    assert metrics.unresolved_left == sum(len(leg.unmatched_left) for leg in legs)
    assert metrics.unresolved_right == sum(len(leg.unmatched_right) for leg in legs)
    # Everything left over on either side is what the reviewer is handed.
    assert metrics.unresolved_left + metrics.unresolved_right == metrics.unmatched_exceptions
