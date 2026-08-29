"""Tier 2: lexical fuzzy matching via char n-gram TF-IDF + cosine similarity.

Char n-grams (not word n-grams) are deliberate: bank/PG narrations are
truncated mid-token ("RAZORP*ACME012-XXXX"), so whole-word overlap fails
where sub-word overlap ("ACME012") still works.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schemas import MatchTier, PairMatch
from ..utils.normalization import amounts_match, clean_text, strip_noise_tokens
from .base import BaseMatcher, MatchCandidate


class TfidfMatcher(BaseMatcher):
    tier = MatchTier.TFIDF
    confidence_threshold = 0.45

    def __init__(self, ngram_range: tuple[int, int] = (2, 4), amount_bonus: float = 0.15):
        self.ngram_range = ngram_range
        self.amount_bonus = amount_bonus

    def _prep(self, text: str) -> str:
        return strip_noise_tokens(clean_text(text))

    def find_matches(self, left: list[MatchCandidate], right: list[MatchCandidate]) -> list[PairMatch]:
        if not left or not right:
            return []

        left_texts = [self._prep(c.text) for c in left]
        right_texts = [self._prep(c.text) for c in right]

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range, min_df=1)
        corpus = left_texts + right_texts
        tfidf = vectorizer.fit_transform(corpus)
        left_vecs, right_vecs = tfidf[: len(left)], tfidf[len(left):]

        sim_matrix = cosine_similarity(left_vecs, right_vecs)

        matches: list[PairMatch] = []
        used_right: set[int] = set()

        # Greedy best-first assignment over the full similarity matrix avoids one
        # early left record "stealing" the best right candidate for a later one.
        order = np.dstack(np.unravel_index(np.argsort(-sim_matrix, axis=None), sim_matrix.shape))[0]
        used_left: set[int] = set()

        for li, ri in order:
            li, ri = int(li), int(ri)
            if li in used_left or ri in used_right:
                continue
            score = float(sim_matrix[li, ri])
            if score <= 0:
                continue

            confidence = score
            if amounts_match(left[li].amount, right[ri].amount):
                confidence = min(1.0, confidence + self.amount_bonus)

            if confidence >= self.confidence_threshold:
                matches.append(PairMatch(
                    left_id=left[li].id,
                    right_id=right[ri].id,
                    tier=self.tier,
                    confidence=round(confidence, 4),
                    rationale=f"tfidf cosine={score:.3f} (amount_bonus={'yes' if confidence > score else 'no'})",
                ))
                used_left.add(li)
                used_right.add(ri)

        return matches
