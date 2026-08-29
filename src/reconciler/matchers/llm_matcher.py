"""Tier 4: LLM adjudication for the residual, genuinely ambiguous cases only.

By design this tier only ever sees what Rule -> TF-IDF -> FAISS could not
resolve — typically a handful of records out of 50+, each scored against a
short shortlist of plausible candidates (not the full cross-product). That
keeps latency/cost bounded and keeps the earlier, cheaper tiers doing most
of the work, which is the point of the escalating-tier design.

Everything a model returns is treated as untrusted: the verdict is parsed
defensively, the chosen id is checked against the shortlist that was actually
offered, and the confidence is clamped into range. A model that returns prose,
a fenced code block, or an id it invented degrades to "no match" — never to a
PairMatch pointing at a record that does not exist.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Protocol

from ..schemas import MatchTier, PairMatch
from ..utils.normalization import amounts_match, clean_text
from .base import BaseMatcher, MatchCandidate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a finance-ops reconciliation adjudicator. Given one source record "
    "and a shortlist of candidate records from a different ledger, decide which "
    "candidate (if any) represents the same underlying transaction. Free text may "
    "be truncated, abbreviated, or contain a masked merchant reference. "
    "Choose only from the candidate ids provided, or null if none is a genuine "
    "match — a wrong match is more costly than no match. "
    "Respond ONLY with JSON: "
    '{"best_candidate_id": "<id or null>", "confidence": <0.0-1.0>, "rationale": "<short reason>"}'
)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Models routinely wrap JSON in fenced code blocks despite instructions not to.
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _no_match(reason: str) -> dict:
    return {"best_candidate_id": None, "confidence": 0.0, "rationale": reason}


def parse_verdict(raw: str) -> dict:
    """Pull the verdict JSON out of a raw model response.

    Tolerates fenced code blocks and leading/trailing prose, both of which
    models emit often enough that treating them as failures would throw away
    good adjudications. Anything genuinely unparseable becomes an explicit
    no-match with the reason preserved, so it surfaces in the exception list
    rather than disappearing.
    """
    if not raw or not raw.strip():
        return _no_match("empty LLM response")

    text = raw.strip()
    fenced = _JSON_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    embedded = _JSON_OBJECT.search(text)
    for attempt in (text, embedded.group(0) if embedded else None):
        if not attempt:
            continue
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return _no_match(f"unparseable LLM output: {raw[:200]}")


class LLMClient(Protocol):
    def adjudicate(self, left_text: str, left_amount: float,
                   candidates: list[tuple[str, str, float]]) -> dict:
        """candidates: list of (id, text, amount). Returns the JSON dict described above."""
        ...


class AnthropicLLMClient:
    """Production client. Reads ANTHROPIC_API_KEY from the environment.

    An API failure is downgraded to a no-match rather than raised: one flaky
    call on the residual tier should leave a single record in the exception
    queue for a human, not abort a reconciliation run that has already
    resolved the rest of the batch.
    """

    def __init__(self, model: str | None = None, *, max_tokens: int = 256):
        self.model = model or os.environ.get("RECON_LLM_MODEL", _DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client = None

    def _load(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _render_prompt(left_text: str, left_amount: float,
                       candidates: list[tuple[str, str, float]]) -> str:
        candidate_block = "\n".join(
            f"- id={cid} amount={amt} text={text!r}" for cid, text, amt in candidates
        )
        return (
            f"Source record: amount={left_amount} text={left_text!r}\n\n"
            f"Candidates:\n{candidate_block}"
        )

    def adjudicate(self, left_text: str, left_amount: float,
                   candidates: list[tuple[str, str, float]]) -> dict:
        try:
            response = self._load().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": self._render_prompt(left_text, left_amount, candidates),
                }],
            )
        except Exception as exc:
            logger.warning("LLM adjudication call failed, treating as no-match: %s", exc)
            return _no_match(f"LLM call failed: {type(exc).__name__}")

        blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return parse_verdict("".join(blocks))


class HeuristicLLMClient:
    """Deterministic, offline stand-in used in tests / CI without an API key.
    Mirrors the AnthropicLLMClient interface so the matcher code path under
    test is identical to production; only the adjudication logic differs."""

    def adjudicate(self, left_text: str, left_amount: float,
                   candidates: list[tuple[str, str, float]]) -> dict:
        left_tokens = set(clean_text(left_text).split())
        best_id, best_score, best_reason = None, 0.0, "no plausible candidate"

        for cid, text, amount in candidates:
            cand_tokens = set(clean_text(text).split())
            overlap = len(left_tokens & cand_tokens) / max(len(left_tokens | cand_tokens), 1)
            score = overlap * 0.7
            if amounts_match(left_amount, amount, abs_tol=5.0, rel_tol=0.02):
                score += 0.3
            if score > best_score:
                best_id, best_score = cid, score
                best_reason = f"token overlap={overlap:.2f}, amount_match={amounts_match(left_amount, amount)}"

        return {"best_candidate_id": best_id, "confidence": round(best_score, 4), "rationale": best_reason}


class LlmMatcher(BaseMatcher):
    tier = MatchTier.LLM
    confidence_threshold = 0.6

    def __init__(self, client: LLMClient | None = None, shortlist_size: int = 5):
        self.client = client or HeuristicLLMClient()
        self.shortlist_size = shortlist_size

    def _shortlist(self, left: MatchCandidate, right: list[MatchCandidate]) -> list[MatchCandidate]:
        def rough_score(r: MatchCandidate) -> tuple[float, str]:
            lt, rt = set(clean_text(left.text).split()), set(clean_text(r.text).split())
            overlap = len(lt & rt) / max(len(lt | rt), 1)
            bonus = 0.5 if amounts_match(left.amount, r.amount, abs_tol=5.0, rel_tol=0.02) else 0.0
            # id is the tiebreaker so an equal-scoring shortlist is stable run to run.
            return overlap + bonus, r.id

        return sorted(right, key=rough_score, reverse=True)[: self.shortlist_size]

    @staticmethod
    def _coerce_confidence(value) -> float:
        """Clamp whatever the model returned into a usable [0, 1] float."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        if confidence != confidence:  # NaN
            return 0.0
        return max(0.0, min(1.0, confidence))

    def find_matches(self, left: list[MatchCandidate], right: list[MatchCandidate]) -> list[PairMatch]:
        if not left or not right:
            return []

        matches: list[PairMatch] = []
        used_right: set[str] = set()

        for candidate in left:
            pool = [r for r in right if r.id not in used_right]
            shortlist = self._shortlist(candidate, pool)
            if not shortlist:
                break  # every right record is claimed; no later left record can match

            verdict = self.client.adjudicate(
                left_text=candidate.text,
                left_amount=candidate.amount,
                candidates=[(r.id, r.text, r.amount) for r in shortlist],
            )
            if not isinstance(verdict, dict):
                logger.warning(
                    "LLM client returned %s, expected dict — skipping %s",
                    type(verdict).__name__, candidate.id,
                )
                continue

            best_id = verdict.get("best_candidate_id")
            confidence = self._coerce_confidence(verdict.get("confidence", 0.0))

            if not best_id or confidence < self.confidence_threshold:
                continue

            # The id must be one we actually offered. A model returning an id
            # from a previous call, an invented one, or a record already claimed
            # would otherwise produce a PairMatch pointing at a record that is
            # not part of this leg at all.
            offered = {r.id for r in shortlist}
            if best_id not in offered:
                logger.warning(
                    "LLM returned id %r for %s which was not in the shortlist %s — discarding",
                    best_id, candidate.id, sorted(offered),
                )
                continue

            matches.append(PairMatch(
                left_id=candidate.id,
                right_id=best_id,
                tier=self.tier,
                confidence=round(confidence, 4),
                rationale=str(verdict.get("rationale", ""))[:300],
            ))
            used_right.add(best_id)

        return matches
