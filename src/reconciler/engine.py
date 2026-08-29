"""
ReconciliationEngine: runs one pair of record sets through the escalating
Rule -> TF-IDF -> FAISS -> LLM tiers. Each tier only ever sees what the
previous tier left unresolved, and only auto-accepts matches at/above its
own confidence threshold — everything else falls through.

What survives all four tiers is not simply discarded. The engine records the
closest candidate it *considered* for each unresolved record, so the exception
queue can tell a reviewer "nothing matched, and here is the nearest thing we
saw and how close it got" rather than only "nothing matched".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .matchers.base import BaseMatcher, MatchCandidate
from .matchers.faiss_matcher import FaissMatcher
from .matchers.llm_matcher import LlmMatcher
from .matchers.rule_matcher import RuleMatcher
from .matchers.tfidf_matcher import TfidfMatcher
from .schemas import (
    BankRecord,
    InvoiceRecord,
    PairMatch,
    SettlementRecord,
    SourceType,
)
from .utils.normalization import (
    amounts_match,
    clean_text,
    extract_alnum_ids,
    extract_invoice_number,
    extract_trailing_digits,
)


def invoice_to_candidate(rec: InvoiceRecord) -> MatchCandidate:
    ids = extract_alnum_ids(rec.description) | extract_alnum_ids(rec.customer_name)
    return MatchCandidate(
        id=rec.invoice_id,
        text=f"{rec.customer_name} {rec.description}",
        amount=rec.amount,
        alnum_ids=frozenset(ids),
        numeric_ref=extract_invoice_number(rec.description),
    )


def settlement_to_candidate(rec: SettlementRecord, *, side: str) -> MatchCandidate:
    """side='invoice_leg' uses gross_amount (what the merchant invoiced);
    side='bank_leg' uses settled_amount (what actually hits the bank)."""
    if side not in ("invoice_leg", "bank_leg"):
        raise ValueError(f"side must be 'invoice_leg' or 'bank_leg', got {side!r}")

    ids = extract_alnum_ids(rec.merchant_ref)
    if rec.order_id:
        ids.add(clean_text(rec.order_id))
    if rec.utr:
        ids.add(clean_text(rec.utr))

    amount = rec.gross_amount if side == "invoice_leg" else rec.settled_amount
    # The invoice number is only embedded in the merchant ref on the invoice
    # side; on the bank side the narration carries the UTR instead.
    numeric_ref = extract_trailing_digits(rec.merchant_ref) if side == "invoice_leg" else None
    return MatchCandidate(
        id=rec.settlement_id,
        text=rec.merchant_ref,
        amount=amount,
        alnum_ids=frozenset(ids),
        numeric_ref=numeric_ref,
    )


def bank_to_candidate(rec: BankRecord) -> MatchCandidate:
    ids = extract_alnum_ids(rec.narration)
    if rec.utr:
        ids.add(clean_text(rec.utr))
    return MatchCandidate(
        id=rec.bank_txn_id,
        text=rec.narration,
        amount=rec.amount,
        alnum_ids=frozenset(ids),
    )


@dataclass
class TierRunStats:
    tier: str
    matched_count: int
    seconds: float
    candidates_in: int = 0  # left records this tier was handed
    candidates_out: int = 0  # left records it passed on unresolved


@dataclass
class NearMiss:
    """The closest candidate considered for a record that stayed unmatched."""
    candidate_id: str
    score: float
    reason: str


@dataclass
class PairResolution:
    matches: list[PairMatch]
    unmatched_left: list[MatchCandidate]
    unmatched_right: list[MatchCandidate]
    tier_stats: list[TierRunStats]
    near_misses: dict[str, NearMiss] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return sum(stat.seconds for stat in self.tier_stats)


def _similarity(left: MatchCandidate, right: MatchCandidate) -> tuple[float, str]:
    """Cheap explainable closeness score, used only to explain non-matches.

    Deliberately not one of the tier scorers: this runs after every tier has
    already declined, and its job is to be legible to a human reviewer
    ("shared 2 of 6 tokens, amount agrees"), not to make a matching decision.
    """
    left_tokens = set(clean_text(left.text).split())
    right_tokens = set(clean_text(right.text).split())
    union = left_tokens | right_tokens
    overlap = len(left_tokens & right_tokens) / len(union) if union else 0.0

    shared_ids = left.alnum_ids & right.alnum_ids
    amount_agrees = amounts_match(left.amount, right.amount, abs_tol=5.0, rel_tol=0.02)

    score = overlap * 0.5 + (0.3 if shared_ids else 0.0) + (0.2 if amount_agrees else 0.0)

    reasons = [f"text overlap {overlap:.0%}"]
    if shared_ids:
        reasons.append(f"shared id {sorted(shared_ids)[0]}")
    reasons.append("amount agrees" if amount_agrees else f"amount differs by {abs(left.amount - right.amount):,.2f}")
    return round(score, 4), ", ".join(reasons)


def find_near_misses(
    unmatched_left: list[MatchCandidate],
    all_right: list[MatchCandidate],
    *,
    min_score: float = 0.15,
) -> dict[str, NearMiss]:
    """Best runner-up candidate for each still-unmatched record.

    Scored against the *full* right-hand pool rather than only the leftovers:
    the most useful hint for a reviewer is often "this looks like a record that
    another invoice already claimed", which points at a duplicate or a
    mis-assignment, and is invisible if you only search the unclaimed remainder.
    """
    if not unmatched_left or not all_right:
        return {}

    near_misses: dict[str, NearMiss] = {}
    for left in unmatched_left:
        best_score, best_id, best_reason = 0.0, None, ""
        for right in all_right:
            score, reason = _similarity(left, right)
            # Ties break on id so the reported near-miss is stable across runs.
            if score > best_score or (score == best_score and best_id and right.id < best_id):
                best_score, best_id, best_reason = score, right.id, reason
        if best_id and best_score >= min_score:
            near_misses[left.id] = NearMiss(candidate_id=best_id, score=best_score, reason=best_reason)
    return near_misses


class ReconciliationEngine:
    def __init__(self, matchers: list[BaseMatcher] | None = None, *, collect_near_misses: bool = True):
        # `is None` rather than a truthiness check: an explicitly empty matcher
        # list is a caller error worth reporting, not a request for the defaults.
        if matchers is None:
            matchers = [RuleMatcher(), TfidfMatcher(), FaissMatcher(), LlmMatcher()]
        if not matchers:
            raise ValueError("ReconciliationEngine needs at least one matcher")
        self.matchers = matchers
        self.collect_near_misses = collect_near_misses

    def match_pair(
        self,
        left: list[MatchCandidate],
        right: list[MatchCandidate],
        left_source: SourceType,
        right_source: SourceType,
    ) -> PairResolution:
        all_matches: list[PairMatch] = []
        tier_stats: list[TierRunStats] = []
        remaining_left, remaining_right = left, right

        for matcher in self.matchers:
            handed_in = len(remaining_left)
            start = time.perf_counter()
            matches, remaining_left, remaining_right = matcher.resolve(
                remaining_left, remaining_right, left_source, right_source
            )
            elapsed = time.perf_counter() - start
            tier_stats.append(TierRunStats(
                tier=matcher.tier.value,
                matched_count=len(matches),
                seconds=elapsed,
                candidates_in=handed_in,
                candidates_out=len(remaining_left),
            ))
            all_matches.extend(matches)
            if not remaining_left or not remaining_right:
                break

        # Record the tiers that never ran, so a tier breakdown is always complete
        # rather than silently missing the rows we short-circuited past.
        ran = {stat.tier for stat in tier_stats}
        for matcher in self.matchers:
            if matcher.tier.value not in ran:
                tier_stats.append(TierRunStats(
                    tier=matcher.tier.value, matched_count=0, seconds=0.0,
                    candidates_in=0, candidates_out=0,
                ))

        near_misses = (
            find_near_misses(remaining_left, right) if self.collect_near_misses else {}
        )

        return PairResolution(
            matches=all_matches,
            unmatched_left=remaining_left,
            unmatched_right=remaining_right,
            tier_stats=tier_stats,
            near_misses=near_misses,
        )
