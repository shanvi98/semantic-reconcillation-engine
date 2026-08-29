"""Tier 1: exact alphanumeric-ID overlap (order_id / invoice no. / UTR) + amount tolerance.

Assignment is global best-first rather than left-to-right greedy: every
(left, right) pair that shares an identifier is scored, the whole set is sorted
by confidence, and pairs are then claimed in descending order. Two properties
fall out of that which a per-left loop does not give you:

- A high-confidence pair is never lost because an earlier, weaker left record
  happened to reach the same right candidate first.
- A pair scored *below* this tier's auto-accept bar never consumes its right
  candidate. Declining a match must leave both records free for the next tier;
  a rejected match that still burns a candidate silently starves later tiers.
"""
from __future__ import annotations

from ..schemas import MatchTier, PairMatch
from ..utils.normalization import amounts_match
from .base import BaseMatcher, MatchCandidate

# An ID overlap alone is strong evidence; amount agreement removes the remaining
# doubt. The gap between these two values is deliberate: it straddles the 0.95
# auto-accept bar so an ID hit with a contradicted amount falls through to the
# lexical/semantic tiers instead of being force-accepted here.
_CONFIDENCE_ID_AND_AMOUNT = 0.99
_CONFIDENCE_ID_ONLY = 0.90

# Wider than the default tolerance: the invoice leg compares against gross_amount
# while PG fee/tax rounding drifts a little, and an ID already carries the match.
_AMOUNT_ABS_TOL = 1.0
_AMOUNT_REL_TOL = 0.03


class RuleMatcher(BaseMatcher):
    tier = MatchTier.RULE
    confidence_threshold = 0.95

    def _score_pair(self, left: MatchCandidate, right: MatchCandidate) -> tuple[float, str] | None:
        """Confidence + human-readable rationale for one pair, or None if no ID overlap."""
        shared = left.alnum_ids & right.alnum_ids
        numeric_hit = bool(left.numeric_ref) and left.numeric_ref == right.numeric_ref
        if not shared and not numeric_hit:
            return None

        if shared:
            rationale = f"shared identifier(s): {sorted(shared)}"
        else:
            rationale = f"invoice-ref match: {left.numeric_ref}"

        if amounts_match(left.amount, right.amount, abs_tol=_AMOUNT_ABS_TOL, rel_tol=_AMOUNT_REL_TOL):
            return _CONFIDENCE_ID_AND_AMOUNT, f"{rationale}; amounts agree"
        return _CONFIDENCE_ID_ONLY, f"{rationale}; amount mismatch ({left.amount:.2f} vs {right.amount:.2f})"

    def find_matches(self, left: list[MatchCandidate], right: list[MatchCandidate]) -> list[PairMatch]:
        if not left or not right:
            return []

        scored: list[tuple[float, str, str, str]] = []
        for left_candidate in left:
            if not left_candidate.alnum_ids and not left_candidate.numeric_ref:
                continue
            for right_candidate in right:
                result = self._score_pair(left_candidate, right_candidate)
                if result is None:
                    continue
                confidence, rationale = result
                # Only pairs this tier would actually accept are eligible to claim a
                # candidate — a declined pair must not block a later, better one.
                if confidence >= self.confidence_threshold:
                    scored.append((confidence, left_candidate.id, right_candidate.id, rationale))

        # Sort by descending confidence; ties break on ids so the result is stable
        # across runs regardless of input ordering.
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))

        matches: list[PairMatch] = []
        used_left: set[str] = set()
        used_right: set[str] = set()

        for confidence, left_id, right_id, rationale in scored:
            if left_id in used_left or right_id in used_right:
                continue
            matches.append(PairMatch(
                left_id=left_id,
                right_id=right_id,
                tier=self.tier,
                confidence=confidence,
                rationale=rationale,
            ))
            used_left.add(left_id)
            used_right.add(right_id)

        return matches
