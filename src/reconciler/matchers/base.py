"""Shared matcher contract. Every tier consumes only what previous tiers left unresolved."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schemas import MatchTier, PairMatch, SourceType


@dataclass(frozen=True)
class MatchCandidate:
    """Source-agnostic view of a record, built by the engine's adapter functions."""
    id: str
    text: str
    amount: float
    alnum_ids: frozenset[str] = field(default_factory=frozenset)
    numeric_ref: str | None = None  # e.g. invoice number embedded in a merchant ref


class BaseMatcher(ABC):
    tier: MatchTier
    confidence_threshold: float

    @abstractmethod
    def find_matches(
        self,
        left: list[MatchCandidate],
        right: list[MatchCandidate],
    ) -> list[PairMatch]:
        """Return only matches at/above this tier's confidence_threshold."""
        raise NotImplementedError

    def resolve(
        self,
        left: list[MatchCandidate],
        right: list[MatchCandidate],
        left_source: SourceType,
        right_source: SourceType,
    ) -> tuple[list[PairMatch], list[MatchCandidate], list[MatchCandidate]]:
        """
        Run this tier and split inputs into (matches, unmatched_left, unmatched_right)
        so the engine can hand the leftovers to the next tier.
        """
        matches = self.find_matches(left, right)
        for m in matches:
            m.left_source = left_source
            m.right_source = right_source
            m.tier = self.tier

        matched_left_ids = {m.left_id for m in matches}
        matched_right_ids = {m.right_id for m in matches}
        unmatched_left = [c for c in left if c.id not in matched_left_ids]
        unmatched_right = [c for c in right if c.id not in matched_right_ids]
        return matches, unmatched_left, unmatched_right
