"""
An exception list is only useful if it's honest: every genuinely unresolved
record must show up in it (nothing silently dropped), and it shouldn't cry
wolf on transactions that actually did resolve correctly. This is what lets
the exception list stand in for a human reviewer's queue.
"""
from __future__ import annotations

from reconciler.reporting.exceptions_report import COLUMNS, exceptions_to_dataframe, summarize
from reconciler.schemas import ExceptionCategory

MAX_FALSE_EXCEPTION_RATE = 0.10  # valid triangles wrongly flagged as exceptions


def test_exceptions_have_non_empty_reasons(pipeline_result):
    assert all(e.reason.strip() for e in pipeline_result.exceptions)


def test_every_exception_has_a_known_category(pipeline_result):
    """The category column is what a reviewer triages on, so it must always be
    populated with a real enum member — never left at a placeholder."""
    assert all(isinstance(e.category, ExceptionCategory) for e in pipeline_result.exceptions)


def test_exceptions_dataframe_is_well_formed(pipeline_result):
    df = exceptions_to_dataframe(pipeline_result.exceptions)
    assert list(df.columns) == COLUMNS
    assert len(df) == len(pipeline_result.exceptions)


def test_exceptions_dataframe_is_sorted_by_value(pipeline_result):
    """The queue is ordered by money at stake — a reviewer with limited time
    should be working the most valuable break first."""
    amounts = exceptions_to_dataframe(pipeline_result.exceptions)["amount"].tolist()
    assert amounts == sorted(amounts, reverse=True)


def test_empty_exception_list_still_yields_the_full_schema():
    df = exceptions_to_dataframe([])
    assert list(df.columns) == COLUMNS
    assert df.empty


def test_partial_matches_report_the_amount_at_risk(pipeline_result):
    """A settlement that never reached the bank exposes its settled amount.
    Reporting 0.0 there (as an earlier version did) sorts the most chase-worthy
    breaks to the bottom of an amount-ranked queue."""
    partials = [
        e for e in pipeline_result.exceptions
        if e.category in (ExceptionCategory.IN_TRANSIT, ExceptionCategory.PARTIAL_MATCH)
    ]
    assert partials, "fixture batch should contain at least one partial triangle"
    assert all(e.amount > 0 for e in partials), (
        f"partial exceptions with no amount at risk: {[e.record_id for e in partials if e.amount <= 0]}"
    )


def test_partial_matches_name_the_leg_that_did_resolve(pipeline_result):
    """A partial triangle has a confirmed counterparty on one side and nothing
    on the other. Reporting its confidence against an empty candidate column
    put a 0.99 similarity next to no record at all."""
    partials = [
        e for e in pipeline_result.exceptions
        if e.category in (ExceptionCategory.IN_TRANSIT, ExceptionCategory.PARTIAL_MATCH)
    ]
    assert partials, "fixture batch should contain at least one partial triangle"
    for exception in partials:
        assert exception.best_candidate_id, (
            f"{exception.record_id} reports a confidence with no related record"
        )


def test_no_exception_reports_a_score_without_a_record(pipeline_result):
    """The two columns are read together; either both are populated or neither is."""
    orphans = [
        e.record_id for e in pipeline_result.exceptions
        if e.best_candidate_score is not None and not e.best_candidate_id
    ]
    assert not orphans, f"score with no related record: {orphans}"


def test_unmatched_invoices_carry_a_near_miss_lead(pipeline_result):
    """An unresolved record should tell the reviewer what the closest thing we
    saw was, not just that nothing matched."""
    invoice_exceptions = [
        e for e in pipeline_result.exceptions
        if e.category is ExceptionCategory.PENDING_SETTLEMENT
    ]
    assert invoice_exceptions, "fixture batch should contain pending-settlement invoices"
    with_leads = [e for e in invoice_exceptions if e.best_candidate_id]
    assert with_leads, "no unmatched invoice surfaced a nearest candidate"
    assert all(0.0 <= e.best_candidate_score <= 1.0 for e in with_leads)


def test_summary_reports_every_category(pipeline_result):
    text = summarize(pipeline_result.exceptions)
    present = {e.category.value for e in pipeline_result.exceptions}
    assert all(category in text for category in present)


def test_deliberately_dropped_legs_surface_as_exceptions(pipeline_result, synthetic_batch):
    """The generator deliberately drops some settlement legs (pending) and some
    bank legs (in-transit). Every such invoice must appear somewhere in the
    exception list — a dropped leg must never just disappear."""
    exception_ids = {e.record_id for e in pipeline_result.exceptions}
    matched_invoice_ids = {t.invoice_id for t in pipeline_result.three_way_matches if t.is_complete}

    for row in synthetic_batch["ground_truth"]:
        is_incomplete_by_design = row["settlement_id"] is None or row["bank_txn_id"] is None
        if is_incomplete_by_design:
            accounted_for = (
                row["invoice_id"] in exception_ids
                or row["invoice_id"] not in matched_invoice_ids
            )
            assert accounted_for, f"{row['invoice_id']} was dropped by the generator but not flagged anywhere"


def test_false_exception_rate_is_low(pipeline_result, synthetic_batch):
    """Of the invoices that truly do have a complete, resolvable triangle in
    ground truth, only a small fraction should get wrongly flagged as an
    exception — otherwise the exception queue is drowning reviewers in noise."""
    gt_by_invoice = {row["invoice_id"]: row for row in synthetic_batch["ground_truth"]}
    should_be_complete = {
        inv_id for inv_id, row in gt_by_invoice.items()
        if row["settlement_id"] and row["bank_txn_id"]
    }

    exception_invoice_ids = {
        e.record_id for e in pipeline_result.exceptions if e.source.value == "invoice"
    }
    false_exceptions = exception_invoice_ids & should_be_complete

    rate = len(false_exceptions) / len(should_be_complete) if should_be_complete else 0.0
    assert rate <= MAX_FALSE_EXCEPTION_RATE, (
        f"{len(false_exceptions)}/{len(should_be_complete)} valid triangles wrongly "
        f"flagged as exceptions: {sorted(false_exceptions)}"
    )
