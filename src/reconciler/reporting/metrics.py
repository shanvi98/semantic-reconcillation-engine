"""Throughput, match-rate, and (when ground truth is available) real accuracy metrics.

This is what turns "we built a matcher" into "we measured a matcher": match
rate alone is easy to game by matching everything at low confidence, so we also
report precision against synthetic ground truth, which we control exactly
because we generated the data ourselves.

Two match rates are reported, and the distinction matters. The raw
``match_rate`` divides complete triangles by *all* invoices, including the ones
the generator deliberately left un-settleable — so it can never reach 1.0 and a
reader comparing it against 100% is comparing against an unreachable ceiling.
``achievable_match_rate`` divides by the invoices that have a complete triangle
in ground truth, which is the number that actually answers "did the engine find
everything it could have found".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..pipeline import PipelineResult
from ..schemas import MatchTier


@dataclass
class AccuracyReport:
    correct: int
    incorrect: int
    missed: int  # ground truth says matched, engine said unmatched
    precision: float
    recall: float
    f1: float


@dataclass
class TierTiming:
    tier: str
    matches: int
    seconds: float
    # How many records this tier was actually handed, and how many it passed on.
    # The engine measures both; reporting them is what lets the escalation
    # diagram draw the real cascade instead of inferring one from match counts.
    candidates_in: int = 0
    candidates_out: int = 0


@dataclass
class MetricsReport:
    total_invoices: int
    total_settlements: int
    total_bank_records: int
    fully_matched_triangles: int
    partial_triangles: int
    unmatched_exceptions: int
    match_rate: float  # fully_matched_triangles / total_invoices
    tier_breakdown: dict[str, int]
    elapsed_seconds: float
    throughput_records_per_sec: float
    achievable_match_rate: float | None = None  # vs. triangles that can close at all
    exception_value: float = 0.0  # total money sitting in the review queue
    exceptions_by_category: dict[str, int] = field(default_factory=dict)
    tier_timings: list[TierTiming] = field(default_factory=list)
    # Records that survived all four tiers, split by which side of a pairing they
    # sat on. Left-hand records are the ones the cascade was driving; right-hand
    # records are counterparties no match ever claimed. Both become exceptions,
    # but only the left-hand count is what "fell through the funnel" means.
    unresolved_left: int = 0
    unresolved_right: int = 0
    accuracy: AccuracyReport | None = None


def tier_breakdown(result: PipelineResult) -> dict[str, int]:
    counts = {tier.value: 0 for tier in MatchTier if tier != MatchTier.UNMATCHED}
    for res in (result.invoice_settlement, result.settlement_bank):
        for match in res.matches:
            counts[match.tier.value] += 1
    return counts


def tier_timings(result: PipelineResult) -> list[TierTiming]:
    """Wall-clock cost per tier, summed across both legs.

    The whole argument for the escalating design is that the expensive tiers
    stay cheap because they see almost nothing — this is the measurement that
    either supports that claim or refutes it.
    """
    totals: dict[str, TierTiming] = {}
    for res in (result.invoice_settlement, result.settlement_bank):
        for stat in res.tier_stats:
            existing = totals.get(stat.tier)
            if existing is None:
                totals[stat.tier] = TierTiming(
                    tier=stat.tier, matches=stat.matched_count, seconds=stat.seconds,
                    candidates_in=stat.candidates_in, candidates_out=stat.candidates_out,
                )
            else:
                existing.matches += stat.matched_count
                existing.seconds += stat.seconds
                existing.candidates_in += stat.candidates_in
                existing.candidates_out += stat.candidates_out

    order = [tier.value for tier in MatchTier if tier != MatchTier.UNMATCHED]
    return [
        TierTiming(
            tier=t, matches=totals[t].matches, seconds=round(totals[t].seconds, 5),
            candidates_in=totals[t].candidates_in, candidates_out=totals[t].candidates_out,
        )
        for t in order if t in totals
    ]


def unresolved_counts(result: PipelineResult) -> tuple[int, int]:
    """(left, right) records still unmatched after every tier, across both legs."""
    legs = (result.invoice_settlement, result.settlement_bank)
    return (
        sum(len(leg.unmatched_left) for leg in legs),
        sum(len(leg.unmatched_right) for leg in legs),
    )


def exceptions_by_category(result: PipelineResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for exception in result.exceptions:
        counts[exception.category.value] = counts.get(exception.category.value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def score_against_ground_truth(result: PipelineResult, ground_truth: list[dict]) -> AccuracyReport:
    """ground_truth rows: {"invoice_id": ..., "settlement_id": ..., "bank_txn_id": ...}
    with None for legs that were deliberately dropped by the generator."""
    gt_by_invoice = {row["invoice_id"]: row for row in ground_truth}

    predicted: dict[str, tuple[str | None, str | None]] = {
        twm.invoice_id: (twm.settlement_id, twm.bank_txn_id)
        for twm in result.three_way_matches if twm.invoice_id
    }

    correct = incorrect = missed = 0
    for invoice_id, gt in gt_by_invoice.items():
        expected = (gt["settlement_id"], gt["bank_txn_id"])
        should_match = expected != (None, None)
        got = predicted.get(invoice_id)

        if not should_match:
            # Nothing to score: ground truth itself has no triangle here. But a
            # prediction against it is a real false positive, not a free pass.
            if got is not None:
                incorrect += 1
            continue
        if got is None:
            missed += 1
        elif got == expected:
            correct += 1
        else:
            incorrect += 1

    denom_p = correct + incorrect
    denom_r = correct + missed
    precision = correct / denom_p if denom_p else 0.0
    recall = correct / denom_r if denom_r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return AccuracyReport(
        correct=correct, incorrect=incorrect, missed=missed,
        precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
    )


def _achievable_match_rate(result: PipelineResult, ground_truth: list[dict] | None) -> float | None:
    """Complete triangles found / complete triangles that exist in ground truth."""
    if not ground_truth:
        return None
    resolvable = [
        row for row in ground_truth
        if row.get("settlement_id") and row.get("bank_txn_id")
    ]
    if not resolvable:
        return None

    found = {t.invoice_id for t in result.three_way_matches if t.is_complete}
    hit = sum(1 for row in resolvable if row["invoice_id"] in found)
    return round(hit / len(resolvable), 4)


def compute_metrics(
    result: PipelineResult,
    total_invoices: int,
    total_settlements: int,
    total_bank_records: int,
    elapsed_seconds: float,
    ground_truth: list[dict] | None = None,
) -> MetricsReport:
    fully_matched = sum(1 for t in result.three_way_matches if t.is_complete)
    partial = sum(1 for t in result.three_way_matches if not t.is_complete)
    total_records = total_invoices + total_settlements + total_bank_records
    unresolved_left, unresolved_right = unresolved_counts(result)

    return MetricsReport(
        total_invoices=total_invoices,
        total_settlements=total_settlements,
        total_bank_records=total_bank_records,
        fully_matched_triangles=fully_matched,
        partial_triangles=partial,
        unmatched_exceptions=len(result.exceptions),
        match_rate=round(fully_matched / total_invoices, 4) if total_invoices else 0.0,
        achievable_match_rate=_achievable_match_rate(result, ground_truth),
        exception_value=result.exception_value,
        exceptions_by_category=exceptions_by_category(result),
        tier_breakdown=tier_breakdown(result),
        tier_timings=tier_timings(result),
        unresolved_left=unresolved_left,
        unresolved_right=unresolved_right,
        elapsed_seconds=round(elapsed_seconds, 4),
        throughput_records_per_sec=round(total_records / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0,
        accuracy=score_against_ground_truth(result, ground_truth) if ground_truth else None,
    )
