"""
The core proof-of-work for the buildathon submission: run the full
Rule -> TF-IDF -> FAISS -> LLM pipeline over a 50+ record synthetic batch and
assert throughput, match rate, and *measured* precision/recall against
ground truth — not a cherry-picked example.

Run `pytest tests/evaluation -s` to see the metrics printed for a demo.
"""
from __future__ import annotations

import json
import time

from reconciler.pipeline import ThreeWayReconciliationPipeline
from reconciler.reporting.metrics import compute_metrics

MIN_BATCH_SIZE = 50
MIN_MATCH_RATE = 0.75   # fraction of invoices resolved into a complete triangle
MIN_PRECISION = 0.95    # of triangles we claim are complete, how many actually are
MIN_RECALL = 0.70       # of triangles that truly exist, how many we found


def test_batch_meets_track_size_bar(synthetic_batch):
    assert len(synthetic_batch["invoices"]) >= MIN_BATCH_SIZE


def test_match_rate_throughput_and_accuracy(synthetic_batch, offline_engine):
    pipeline = ThreeWayReconciliationPipeline(engine=offline_engine)

    start = time.perf_counter()
    result = pipeline.run(synthetic_batch["invoices"], synthetic_batch["settlements"], synthetic_batch["bank"])
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(
        result,
        total_invoices=len(synthetic_batch["invoices"]),
        total_settlements=len(synthetic_batch["settlements"]),
        total_bank_records=len(synthetic_batch["bank"]),
        elapsed_seconds=elapsed,
        ground_truth=synthetic_batch["ground_truth"],
    )

    print("\n" + "=" * 60)
    print("RECONCILIATION METRICS")
    print("=" * 60)
    print(json.dumps(metrics, default=lambda o: o.__dict__, indent=2))
    print("=" * 60)

    assert metrics.throughput_records_per_sec > 0
    assert metrics.match_rate >= MIN_MATCH_RATE, (
        f"match rate {metrics.match_rate} below bar {MIN_MATCH_RATE}"
    )

    assert metrics.accuracy is not None
    assert metrics.accuracy.precision >= MIN_PRECISION, (
        f"precision {metrics.accuracy.precision} below bar — engine is confidently "
        f"wrong, not just incomplete, on {metrics.accuracy.incorrect} triangle(s)"
    )
    assert metrics.accuracy.recall >= MIN_RECALL, (
        f"recall {metrics.accuracy.recall} below bar {MIN_RECALL}"
    )


def test_rule_tier_carries_the_exact_id_matches(pipeline_result):
    """Sanity check on tier ordering: UTR-bearing settlement/bank pairs should
    resolve at the cheapest tier rather than falling all the way to the LLM."""
    from reconciler.reporting.metrics import tier_breakdown
    breakdown = tier_breakdown(pipeline_result)
    assert breakdown["rule"] > 0


def test_tier_breakdown_is_internally_consistent(pipeline_result):
    """Every accepted pair match came from exactly one tier, so the tier
    breakdown must sum to the total number of pair matches across both legs.

    This batch's UTR/merchant-token noise happens to resolve entirely at the
    Rule tier for a given seed — which is the *correct* outcome (cheapest
    tier first, escalate only when it can't resolve something). That each
    later tier CAN catch what an earlier one misses is proven directly, with
    controlled inputs, in tests/unit/test_tfidf_matcher.py and
    tests/unit/test_faiss_matcher.py rather than left to chance here.
    """
    from reconciler.reporting.metrics import tier_breakdown
    breakdown = tier_breakdown(pipeline_result)
    total_pair_matches = len(pipeline_result.invoice_settlement.matches) + len(pipeline_result.settlement_bank.matches)
    assert sum(breakdown.values()) == total_pair_matches
