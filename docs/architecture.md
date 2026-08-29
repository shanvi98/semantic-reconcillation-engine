# Architecture

## Data flow

```
data/raw/invoices.csv ────────────┐
data/raw/razorpay_settlements.csv ──► ReconciliationEngine (Rule→TF-IDF→FAISS→LLM)
data/raw/bank_statement.csv ──────┘             │
                                                 ▼
                                  ThreeWayReconciliationPipeline
                                                 │
                            ┌────────────────────┴────────────────────┐
                            ▼                                         ▼
                  data/outputs/metrics.json              data/outputs/exceptions.csv
```

`ThreeWayReconciliationPipeline.run()` performs two pairwise resolutions —
`invoice ↔ settlement` (on `gross_amount`) and `settlement ↔ bank` (on `settled_amount`,
since PG fees/tax are deducted before the bank credit) — then chains them on
`settlement_id` into a `ThreeWayMatch`.

Using one amount for both legs is the classic three-way reconciliation error: the invoice
leg must compare against what was billed, and the bank leg against what actually landed.
`settlement_to_candidate(record, side=...)` exists to make that distinction explicit and
un-skippable, and it raises rather than guessing on an unknown side.

A settlement with only one leg resolved becomes a **partial match**, reported as an
exception even though the settlement itself technically "matched" on one side.

## Tier contract (`matchers/base.py`)

Every tier implements `find_matches(left, right) -> list[PairMatch]` and returns only
matches at or above its own `confidence_threshold`. `BaseMatcher.resolve()` then splits the
input into `(matches, unmatched_left, unmatched_right)`, which the engine feeds to the next
tier. Two invariants make the escalation sound:

**Declining must be free.** A tier that scores a pair below its bar has to leave *both*
records available downstream. An earlier version of the rule tier marked the right-hand
candidate as consumed before filtering on confidence, so a rejected 0.90 match silently
starved the record that genuinely matched it — the pair was neither matched here nor
offered to any later tier. It is pinned now by
`tests/unit/test_rule_matcher.py::test_declined_match_does_not_consume_its_candidate`.

**Assignment must not depend on input order.** Every tier scores all eligible pairs, sorts
globally by confidence, and claims greedily from the top, with ids as tiebreaker. A bank
statement does not arrive sorted in invoice order, so a left-to-right loop would make the
result a function of upstream file ordering.

The consequence of both: the rule tier's exact-ID matches never touch the LLM, and the LLM
tier only ever sees a small residual, each scored against a short heuristically-ranked
shortlist rather than the full cross-product — keeping cost and latency bounded regardless
of batch size.

## Why char n-gram TF-IDF before dense embeddings

Bank/PG narrations are truncated mid-token (`RAZORP*ACME012-XXXX`), so word-level lexical
matching fails where sub-word overlap still works. TF-IDF with `analyzer="char_wb"` catches
these cheaply; dense embeddings are reserved for the harder paraphrase-level cases (an
abbreviated legal entity name sharing almost no characters with the invoice's customer name
but semantically the same company).

## Treating model output as untrusted

The LLM tier is the only one whose input is free-form model output, so nothing it returns
is taken at face value:

- The response is parsed defensively — fenced code blocks and surrounding prose are both
  tolerated, because models emit them often enough that rejecting them would discard good
  adjudications.
- The returned id is checked against the shortlist that was actually offered. An invented
  id, or one carried over from a previous call, is discarded with a warning rather than
  becoming a `PairMatch` pointing at a record that is not in this leg at all.
- Confidence is coerced and clamped into `[0, 1]`; `NaN`, strings and nulls degrade to 0.
- An API failure is downgraded to a no-match. One flaky call on the residual tier should
  cost one record in the review queue, not abort a run that has already reconciled the rest
  of the batch.

## Search backends

`FaissMatcher` uses a FAISS `IndexFlatIP` when faiss is installed and an equivalent numpy
`argpartition` when it is not. This is a faithful substitution, not a degraded mode:
`IndexFlatIP` is itself an exhaustive inner-product scan over the same normalized vectors,
so both paths return identical neighbours in identical order — asserted directly in
`tests/unit/test_faiss_matcher.py::test_numpy_and_faiss_search_paths_agree`.

The practical effect is that the semantic tier, and every test covering it, runs in a
core-only install without the multi-hundred-megabyte faiss/torch wheels, while still using
the real index when it is available.

## Offline-vs-production stack

`FaissMatcher` and `LlmMatcher` both take an injectable backend (`Embedder` / `LLMClient`
protocol) so the exact same tier logic runs against either the production stack
(sentence-transformers + Claude API) or a deterministic offline stand-in (`HashingEmbedder`
/ `HeuristicLLMClient`). CI always runs offline — a green tick has to mean the logic is
correct, not that a hosted model happened to be reachable.

## Exception model

`ExceptionRecord` carries a `category` (`ExceptionCategory`) alongside the free-text
reason, because a reviewer works the categories very differently: a pending settlement is
"wait and re-run tomorrow", a short settlement is a dispute to raise with the PG, and an
unidentified bank credit is an unknown receipt to investigate. Collapsing those into one
prose column forces a human to re-derive triage the pipeline already knows.

Two further properties keep the queue actionable:

- **The amount is the amount at risk.** A settlement that never reached the bank exposes
  its settled amount, not zero. Reporting 0.0 sorted the most chase-worthy breaks to the
  bottom of an amount-ranked queue.
- **A score never appears without a record.** Unmatched records carry the nearest candidate
  the engine considered (`engine.find_near_misses`); partial triangles carry the leg that
  *did* resolve. Both populate the same two columns, read together as "what we associated
  this with, and how sure we are".

Near-miss search runs against the *full* opposite-side pool rather than only the leftovers,
because the most useful hint is often "this looks like a record another invoice already
claimed" — which points at a duplicate or a mis-assignment and is invisible if you only
search what is unclaimed.

## Configuration

Precedence is dataclass defaults → `config/settings.yaml` → environment variables.
Booleans get an explicit coercion pass in `config.py` because both YAML and env vars can
supply the *string* `"false"`, which is truthy in Python — silently flipping a run onto the
paid production stack is exactly the failure a config layer exists to prevent. Unknown keys
warn rather than passing silently, since a typo'd setting that is quietly dropped is
indistinguishable from one that did not take effect.

## Presentation

`src/reconciler/ui/` holds the dashboard's design system: `theme.py` (tokens + stylesheet)
and `components.py` (rendered blocks). `app.py` is orchestration only.

One Streamlit-specific hazard is handled centrally in `components._html()`: Streamlit
renders through a CommonMark parser, where a raw HTML block ends at the first blank line
and any line indented four spaces or more becomes a code block. Concatenating two readable,
indented templates therefore puts a whitespace-only line between them — closing the HTML
block and dumping the rest on screen as *source text*. Flattening markup before it reaches
the parser keeps the templates readable without exposing their indentation.
