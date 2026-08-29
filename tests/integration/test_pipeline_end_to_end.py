"""Wires all three sources through the real pipeline and checks structural
invariants — not accuracy (that's tests/evaluation), just that the plumbing
between the two pairwise legs and the three-way chaining is correct."""
from __future__ import annotations


def test_every_input_record_is_accounted_for(pipeline_result, synthetic_batch):
    """Nothing should silently vanish: every invoice/settlement/bank record must
    end up in a three-way match or in the exception list — never neither."""
    invoice_ids = {r.invoice_id for r in synthetic_batch["invoices"]}
    settlement_ids = {r.settlement_id for r in synthetic_batch["settlements"]}
    bank_ids = {r.bank_txn_id for r in synthetic_batch["bank"]}

    matched_invoice_ids = {t.invoice_id for t in pipeline_result.three_way_matches if t.invoice_id}
    matched_settlement_ids = {t.settlement_id for t in pipeline_result.three_way_matches if t.settlement_id}
    matched_bank_ids = {t.bank_txn_id for t in pipeline_result.three_way_matches if t.bank_txn_id}

    exception_ids_by_source = {}
    for e in pipeline_result.exceptions:
        exception_ids_by_source.setdefault(e.source.value, set()).add(e.record_id)

    assert invoice_ids <= matched_invoice_ids | exception_ids_by_source.get("invoice", set())
    assert settlement_ids <= matched_settlement_ids | exception_ids_by_source.get("razorpay", set())
    assert bank_ids <= matched_bank_ids | exception_ids_by_source.get("bank", set())


def test_no_record_is_used_in_two_different_triangles(pipeline_result):
    seen_invoice, seen_settlement, seen_bank = set(), set(), set()
    for t in pipeline_result.three_way_matches:
        if t.invoice_id:
            assert t.invoice_id not in seen_invoice
            seen_invoice.add(t.invoice_id)
        assert t.settlement_id not in seen_settlement
        seen_settlement.add(t.settlement_id)
        if t.bank_txn_id:
            assert t.bank_txn_id not in seen_bank
            seen_bank.add(t.bank_txn_id)


def test_complete_triangles_have_full_confidence_chain(pipeline_result):
    for t in pipeline_result.three_way_matches:
        if t.is_complete:
            assert len(t.pair_matches) == 2
            assert 0.0 < t.overall_confidence <= 1.0


def test_pure_bank_noise_never_enters_a_triangle(pipeline_result, synthetic_batch):
    """Salary/rent/unrelated-vendor bank rows the generator injects with no
    settlement counterpart at all must never get absorbed into a match."""
    matched_bank_ids = {t.bank_txn_id for t in pipeline_result.three_way_matches if t.bank_txn_id}
    noise_ids = {r.bank_txn_id for r in synthetic_batch["bank"] if r.bank_txn_id.startswith("BNK-N")}
    assert not (noise_ids & matched_bank_ids)
