# Evaluation Report

Reproduce every number below with:

```bash
python scripts/generate_data.py --n-records 60 --seed 42
python scripts/run_reconciliation.py
pytest tests/evaluation -s
```

The same figures are asserted as pass/fail CI gates in `tests/evaluation/`, so a
regression fails the build rather than quietly degrading this document.

## Latest run — offline deterministic stack, `seed=42`

Stack: `HashingEmbedder` + `HeuristicLLMClient` (no API key, no model download, no network).

| Metric | Value |
|---|---|
| Total records | 180 (60 invoices · 57 settlements · 63 bank lines) |
| Complete triangles | 53 |
| Partial triangles | 4 |
| Match rate | 88.3% |
| **Achievable match rate** | **100%** |
| Precision | 1.00 (0 incorrect) |
| Recall | 1.00 (0 missed) |
| F1 | 1.00 |
| Throughput | 1,554 records/sec |
| Elapsed | 0.12 s |
| Tier breakdown | rule=106 · tfidf=4 · faiss=0 · llm=0 |
| Unresolved exceptions | 17, worth ₹12,91,651.57 |

### Reading the two match rates

`match_rate` (88.3%) divides complete triangles by *all* 60 invoices — including the seven
the generator deliberately left un-settleable (3 pending settlement, 4 in transit). It is
therefore capped below 100% by construction, and measuring it against a 100% target
measures the fixture rather than the matcher.

`achievable_match_rate` (100%) divides by the 53 triangles that can close at all. That is
the number that answers "did the engine find everything findable", and here it did.

### Tier cost

| Tier | Matches | Time | Cost characteristic |
|---|---|---|---|
| Rule | 106 | 1.4 ms | exact, deterministic, free |
| TF-IDF | 4 | 30 ms | one vectoriser fit over the residual |
| FAISS | 0 | 80 ms | embedding + index build, still on the residual |
| LLM | 0 | 0.2 ms | never invoked — nothing reached it |

100% of matches resolved on the two cheapest tiers. That ratio is the design working: the
expensive tier stays cheap because it almost never runs, and its cost scales with the
residual rather than with batch size.

The FAISS tier's 80 ms is index construction over records it ultimately did not need to
match — visible here precisely because `tier_timings` reports wall-clock cost per tier
rather than only match counts. Under the production stack this figure is dominated by
sentence-transformer model loading (~18 s cold), which is why the cheap tiers running first
matters in practice and not just in principle.

### Exception queue

| Category | Count | Value |
|---|---|---|
| `unmatched_bank_credit` | 10 | ₹4,44,772.18 |
| `in_transit` | 4 | ₹6,01,834.44 |
| `pending_settlement` | 3 | ₹2,45,044.95 |

All 10 unmatched bank credits are the generator's injected noise rows (salary, rent, GST
refunds, chargebacks) that have no invoice or settlement counterpart by construction —
correctly declined rather than force-matched. The remaining 7 are the deliberately dropped
legs, each surfaced with the counterparty it *did* resolve against.

`tests/integration/test_pipeline_end_to_end.py::test_pure_bank_noise_never_enters_a_triangle`
asserts that the noise rows never get absorbed into a match.

## Effect of the tier-consumption fix

Before the rule tier stopped consuming candidates for pairs it declined, this batch
reported `rule=113, tfidf=0, faiss=0, llm=0`. Those four records were not resolved at the
rule tier — they were scored below its 0.95 bar, rejected, and their counterparties marked
used anyway, so no later tier was ever offered them.

They now correctly fall through and resolve at TF-IDF (`rule=106, tfidf=4`). The visible
match rate barely moved, which is the point: the failure was invisible in the headline
number and only showed up as an escalation path that could never demonstrate a second tier
doing work.

## Interpreting a different result

- **Low match rate, high precision** — the engine is appropriately conservative, declining
  ambiguous cases rather than guessing. Check `achievable_match_rate` before concluding
  anything: a low raw match rate on a fixture with many dropped legs is expected.
- **High match rate, low precision** — thresholds are too permissive. Tune the per-tier
  `confidence_threshold` values; the tier that gained the most matches is the one to look
  at first.
- **Anything reaching the LLM tier in volume** — the cheap tiers are underperforming on
  this data shape. That is a signal about normalization (`utils/normalization.py`), not a
  reason to raise the LLM budget.

## Notes

- Precision and recall are computed against the generator's own ground truth
  (`data/processed/ground_truth.json`), not eyeballed — see
  `src/reconciler/reporting/metrics.py::score_against_ground_truth`.
- A predicted triangle for an invoice that has *no* triangle in ground truth counts as a
  false positive rather than being skipped, so fabricating matches for un-settleable
  records costs precision instead of being free.
- The generator draws from a single seeded `random.Random` and never touches global RNG
  state, so a given seed always produces byte-identical CSVs.
