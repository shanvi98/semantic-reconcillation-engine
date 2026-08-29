"""Typed data contracts shared across the three sources and the matching pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class SourceType(str, Enum):
    BANK = "bank"
    RAZORPAY = "razorpay"
    INVOICE = "invoice"


class MatchTier(str, Enum):
    RULE = "rule"
    TFIDF = "tfidf"
    FAISS = "faiss"
    LLM = "llm"
    UNMATCHED = "unmatched"


class ExceptionCategory(str, Enum):
    """Why a record landed in the review queue.

    A finance-ops reviewer works these categories very differently — a pending
    settlement is a "wait and re-run tomorrow", a short settlement is a dispute
    to raise with the PG, and an unmatched bank credit is an unidentified
    receipt. Collapsing them into one free-text reason column, as an earlier
    version did, forces the reviewer to re-derive that triage from prose.
    """
    PENDING_SETTLEMENT = "pending_settlement"
    IN_TRANSIT = "in_transit"
    UNMATCHED_SETTLEMENT = "unmatched_settlement"
    UNMATCHED_BANK_CREDIT = "unmatched_bank_credit"
    PARTIAL_MATCH = "partial_match"


@dataclass(frozen=True)
class BankRecord:
    bank_txn_id: str
    txn_date: date
    narration: str
    amount: float
    utr: str | None = None


@dataclass(frozen=True)
class SettlementRecord:
    settlement_id: str
    order_id: str
    utr: str | None
    settlement_date: date
    gross_amount: float
    fee: float
    tax: float
    settled_amount: float
    merchant_ref: str  # noisy free-text e.g. "RAZORP*ACME012"


@dataclass(frozen=True)
class InvoiceRecord:
    invoice_id: str
    customer_name: str
    invoice_date: date
    amount: float
    description: str  # noisy free-text e.g. "Acme Corp Invoice #012"


@dataclass
class PairMatch:
    """One resolved edge between two records from different sources."""
    left_id: str
    right_id: str
    left_source: SourceType | None = None
    right_source: SourceType | None = None
    tier: MatchTier = MatchTier.UNMATCHED
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class ThreeWayMatch:
    """A fully or partially resolved triangle: invoice <-> settlement <-> bank."""
    invoice_id: str | None
    settlement_id: str | None
    bank_txn_id: str | None
    pair_matches: list[PairMatch] = field(default_factory=list)
    overall_confidence: float = 0.0

    @property
    def is_complete(self) -> bool:
        return all([self.invoice_id, self.settlement_id, self.bank_txn_id])

    @property
    def weakest_tier(self) -> MatchTier:
        """The least-certain tier anywhere in this triangle.

        A triangle is only as trustworthy as its shakiest edge, so an
        LLM-adjudicated leg dominates a rule-matched one when reporting how
        this match was actually resolved.
        """
        if not self.pair_matches:
            return MatchTier.UNMATCHED
        order = [MatchTier.LLM, MatchTier.FAISS, MatchTier.TFIDF, MatchTier.RULE]
        present = {p.tier for p in self.pair_matches}
        for tier in order:
            if tier in present:
                return tier
        return MatchTier.UNMATCHED

    @property
    def missing_legs(self) -> list[str]:
        return [
            leg for leg, value in (
                ("invoice", self.invoice_id),
                ("settlement", self.settlement_id),
                ("bank", self.bank_txn_id),
            ) if value is None
        ]


@dataclass
class ExceptionRecord:
    record_id: str
    source: SourceType
    reason: str
    amount: float
    category: ExceptionCategory = ExceptionCategory.PARTIAL_MATCH
    best_candidate_id: str | None = None
    best_candidate_score: float | None = None
