"""
ThreeWayReconciliationPipeline: chains two pairwise resolutions
(Invoice <-> Settlement, Settlement <-> Bank) into a single three-way match,
and produces an honest exception list for everything that didn't fully close.

This is the piece that makes this a *three*-way reconciler rather than two
independent two-way matchers: a settlement only counts as "resolved" once it
has a bank leg, and an invoice only counts as fully closed once its
settlement also cleared the bank.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .engine import (
    NearMiss,
    PairResolution,
    ReconciliationEngine,
    bank_to_candidate,
    invoice_to_candidate,
    settlement_to_candidate,
)
from .schemas import (
    BankRecord,
    ExceptionCategory,
    ExceptionRecord,
    InvoiceRecord,
    SettlementRecord,
    SourceType,
    ThreeWayMatch,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    three_way_matches: list[ThreeWayMatch]
    exceptions: list[ExceptionRecord]
    invoice_settlement: PairResolution
    settlement_bank: PairResolution

    @property
    def complete_triangles(self) -> list[ThreeWayMatch]:
        return [t for t in self.three_way_matches if t.is_complete]

    @property
    def partial_triangles(self) -> list[ThreeWayMatch]:
        return [t for t in self.three_way_matches if not t.is_complete]

    @property
    def exception_value(self) -> float:
        """Total monetary value sitting in the review queue."""
        return round(sum(e.amount for e in self.exceptions), 2)


def _index_edges(matches, key: str, leg: str) -> dict:
    """Index pair matches by settlement id, warning on any collision.

    Every matcher tier enforces one-to-one assignment internally, so a duplicate
    here means a tier regressed. Overwriting silently would hide that behind a
    slightly-wrong match count, so it gets logged.
    """
    indexed: dict = {}
    for match in matches:
        settlement_id = getattr(match, key)
        if settlement_id in indexed:
            logger.warning(
                "%s leg produced two edges for settlement %s (%s and %s) — keeping the higher-confidence one",
                leg, settlement_id, indexed[settlement_id], match,
            )
            if match.confidence <= indexed[settlement_id].confidence:
                continue
        indexed[settlement_id] = match
    return indexed


class ThreeWayReconciliationPipeline:
    def __init__(self, engine: ReconciliationEngine | None = None):
        self.engine = engine or ReconciliationEngine()

    def run(
        self,
        invoices: list[InvoiceRecord],
        settlements: list[SettlementRecord],
        bank: list[BankRecord],
    ) -> PipelineResult:
        invoice_candidates = [invoice_to_candidate(r) for r in invoices]
        settlement_inv_side = [settlement_to_candidate(r, side="invoice_leg") for r in settlements]
        settlement_bank_side = [settlement_to_candidate(r, side="bank_leg") for r in settlements]
        bank_candidates = [bank_to_candidate(r) for r in bank]

        inv_stl = self.engine.match_pair(
            invoice_candidates, settlement_inv_side, SourceType.INVOICE, SourceType.RAZORPAY
        )
        stl_bank = self.engine.match_pair(
            settlement_bank_side, bank_candidates, SourceType.RAZORPAY, SourceType.BANK
        )

        settlement_to_invoice = _index_edges(inv_stl.matches, "right_id", "invoice<->settlement")
        settlement_to_bank = _index_edges(stl_bank.matches, "left_id", "settlement<->bank")

        three_way: list[ThreeWayMatch] = []
        matched_invoice_ids: set[str] = set()
        matched_settlement_ids: set[str] = set()
        matched_bank_ids: set[str] = set()

        for settlement in settlements:
            inv_edge = settlement_to_invoice.get(settlement.settlement_id)
            bank_edge = settlement_to_bank.get(settlement.settlement_id)
            if inv_edge is None and bank_edge is None:
                continue

            pair_matches = [e for e in (inv_edge, bank_edge) if e is not None]
            confidences = [e.confidence for e in pair_matches]
            twm = ThreeWayMatch(
                invoice_id=inv_edge.left_id if inv_edge else None,
                settlement_id=settlement.settlement_id,
                bank_txn_id=bank_edge.right_id if bank_edge else None,
                pair_matches=pair_matches,
                # A chain is as strong as its weakest link, so the triangle takes
                # the minimum rather than the mean of its edge confidences.
                overall_confidence=round(min(confidences), 4),
            )
            three_way.append(twm)

            if twm.invoice_id:
                matched_invoice_ids.add(twm.invoice_id)
            matched_settlement_ids.add(settlement.settlement_id)
            if twm.bank_txn_id:
                matched_bank_ids.add(twm.bank_txn_id)

        exceptions = self._build_exceptions(
            invoices, settlements, bank,
            matched_invoice_ids, matched_settlement_ids, matched_bank_ids,
            three_way,
            near_misses={**inv_stl.near_misses, **stl_bank.near_misses},
        )

        return PipelineResult(
            three_way_matches=three_way,
            exceptions=exceptions,
            invoice_settlement=inv_stl,
            settlement_bank=stl_bank,
        )

    @staticmethod
    def _build_exceptions(
        invoices: list[InvoiceRecord],
        settlements: list[SettlementRecord],
        bank: list[BankRecord],
        matched_invoice_ids: set[str],
        matched_settlement_ids: set[str],
        matched_bank_ids: set[str],
        three_way: list[ThreeWayMatch],
        near_misses: dict[str, NearMiss],
    ) -> list[ExceptionRecord]:
        exceptions: list[ExceptionRecord] = []

        def attach(record_id: str) -> tuple[str | None, float | None, str]:
            """Nearest candidate considered for this record, for the reviewer."""
            hit = near_misses.get(record_id)
            if hit is None:
                return None, None, ""
            return hit.candidate_id, hit.score, f" — closest candidate {hit.candidate_id} ({hit.reason})"

        for inv in invoices:
            if inv.invoice_id in matched_invoice_ids:
                continue
            candidate_id, score, suffix = attach(inv.invoice_id)
            exceptions.append(ExceptionRecord(
                record_id=inv.invoice_id,
                source=SourceType.INVOICE,
                category=ExceptionCategory.PENDING_SETTLEMENT,
                reason=f"No settlement or bank leg found — likely awaiting settlement{suffix}",
                amount=inv.amount,
                best_candidate_id=candidate_id,
                best_candidate_score=score,
            ))

        for stl in settlements:
            if stl.settlement_id in matched_settlement_ids:
                continue
            candidate_id, score, suffix = attach(stl.settlement_id)
            exceptions.append(ExceptionRecord(
                record_id=stl.settlement_id,
                source=SourceType.RAZORPAY,
                category=ExceptionCategory.UNMATCHED_SETTLEMENT,
                reason=f"Settlement with no matching invoice and no bank credit{suffix}",
                amount=stl.gross_amount,
                best_candidate_id=candidate_id,
                best_candidate_score=score,
            ))

        for bnk in bank:
            if bnk.bank_txn_id in matched_bank_ids:
                continue
            exceptions.append(ExceptionRecord(
                record_id=bnk.bank_txn_id,
                source=SourceType.BANK,
                category=ExceptionCategory.UNMATCHED_BANK_CREDIT,
                reason=f"Unidentified bank credit — no settlement explains it ({bnk.narration})",
                amount=bnk.amount,
            ))

        # Partial triangles (matched on one leg, missing the other) are exceptions
        # too, even though the settlement itself is technically "matched" above.
        settlement_by_id = {s.settlement_id: s for s in settlements}
        for twm in three_way:
            if twm.is_complete:
                continue
            settlement = settlement_by_id.get(twm.settlement_id)
            missing = twm.missing_legs
            category = (
                ExceptionCategory.IN_TRANSIT if missing == ["bank"]
                else ExceptionCategory.PARTIAL_MATCH
            )
            explanation = (
                "settled by the PG but not yet credited by the bank"
                if category is ExceptionCategory.IN_TRANSIT
                else f"missing {', '.join(missing)} leg"
            )
            # The amount at risk is the leg that is actually missing: an
            # un-credited settlement exposes the settled amount, not zero. The
            # previous version reported 0.0 here, which sorted every partial
            # match to the bottom of an amount-ranked review queue — exactly
            # backwards for the cases a controller most needs to chase.
            amount = 0.0
            if settlement is not None:
                amount = settlement.settled_amount if missing == ["bank"] else settlement.gross_amount

            # Name the leg that *did* resolve. A partial triangle has no
            # "closest candidate" in the near-miss sense — it has a confirmed
            # counterparty on one side and nothing on the other, and that
            # counterparty is what a reviewer needs in order to chase it.
            # Reporting the triangle confidence against an empty candidate
            # column, as an earlier version did, showed a 0.99 similarity score
            # next to no record at all.
            linked_id = twm.invoice_id or twm.bank_txn_id

            exceptions.append(ExceptionRecord(
                record_id=twm.settlement_id,
                source=SourceType.RAZORPAY,
                category=category,
                reason=f"Partial match — {explanation} (resolved via {twm.weakest_tier.value} tier)",
                amount=amount,
                best_candidate_id=linked_id,
                best_candidate_score=twm.overall_confidence,
            ))

        return exceptions
