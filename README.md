# Semantic Reconciliation Engine

**Created by Shanvi Kumari.**

Three-way finance-ops reconciliation — **Bank Statement ↔ Razorpay Settlement ↔ Invoice** —
built for Razorpay Buildathon Track 04 ("AI Finance Controller").

Most reconciliation demos do a two-way match (settlement ↔ bank, or invoice ↔ settlement).
This engine closes the full loop across all three legs, because a settlement that never
reaches the bank, or an invoice that was never settled, is exactly the kind of gap
finance-ops actually loses time chasing.

```bash
pip install -e ".[dev]"          # core only — no ML wheels, no network
python scripts/generate_data.py  # synthetic 3-source batch + ground truth
python scripts/run_reconciliation.py
pytest                           # 163 tests, offline and deterministic
```

## Why this design

The track brief frames the bottleneck as **verification capacity**: free-text noise
(truncated bank narrations, masked merchant refs, typo'd customer names) is what makes
naive exact-match reconciliation break down at scale. The pipeline escalates through four
tiers, cheapest first, so most of the batch resolves before ever touching an LLM:

```
Rule (exact ID / UTR overlap)
  → TF-IDF (char n-gram lexical fuzzy match)
    → FAISS (dense semantic embedding match)
      → LLM (adjudicates only the residual, genuinely ambiguous cases)
```

Every tier sees only what the previous tier left unresolved, and only auto-accepts matches
at or above its own confidence threshold. Anything that survives all four becomes an
honest, reasoned exception rather than a forced guess.

Two properties make that escalation trustworthy rather than merely plausible:

- **Declining is free.** A tier that scores a pair below its own bar leaves *both* records
  available to the next tier. A tier that consumed a candidate it then rejected would
  silently starve every tier behind it.
- **Model output is untrusted.** The LLM tier validates the id it gets back against the
  shortlist it actually offered. A hallucinated id becomes a no-match, never a
  `PairMatch` pointing at a record that does not exist.

## Results

On the default 60-invoice batch (`seed=42`), running the fully offline stack:

| Metric | Value |
|---|---|
| Complete triangles | 53 / 60 invoices |
| Match rate | 88.3% |
| **Achievable match rate** | **100%** |
| Precision / Recall / F1 | 1.00 / 1.00 / 1.00 |
| Resolved on the two cheapest tiers | 100% (106 rule, 4 TF-IDF) |
| Exceptions | 17, worth ₹12.92L |

The two match rates answer different questions, and only the second one is about the
engine. `match_rate` divides by *all* invoices, including the ones the generator
deliberately left un-settleable, so it is capped below 100% by construction — comparing it
against a 100% target measures the test fixture, not the matcher.
`achievable_match_rate` divides by the triangles that can close at all, and that is the
number that says whether the engine found everything findable.

Reproduce with `make run`; the numbers are asserted as CI gates in `tests/evaluation/`.

## Dashboard

```bash
pip install -e ".[app]"
make app          # or: streamlit run app.py
```

A dark control-room view of the last run: escalation diagram, per-tier cost, measured
accuracy, and a filterable exception queue ranked by value at risk. The visual language
lives in `src/reconciler/ui/` (design tokens, stylesheet, components) so it is a real
module rather than inline markup scattered through the app.

## Install profiles

The core install is deliberately light — the engine and the **entire test suite** run on
pandas / numpy / scikit-learn / pyyaml alone, with no network access. The heavyweight
wheels are opt-in:

| Extra | Pulls in | Needed for |
|---|---|---|
| *(core)* | pandas, numpy, scikit-learn, pyyaml | everything, including all tests |
| `[embeddings]` | faiss-cpu, sentence-transformers | production semantic tier |
| `[llm]` | anthropic | production adjudication tier |
| `[app]` | streamlit | the dashboard |
| `[dev]` | pytest, pytest-cov, ruff | development |
| `[all]` | all of the above | one-command demo environment |

The semantic tier works in either case: it uses a FAISS `IndexFlatIP` when faiss is
installed and an equivalent numpy computation when it is not. `IndexFlatIP` is itself an
exhaustive inner-product scan, so both paths return the same neighbours in the same order
— an equivalence pinned by a test rather than assumed.

By default everything runs on an **offline, deterministic stack**: a hashing-based embedder
standing in for the sentence-transformer, and a heuristic client standing in for the Claude
adjudication call. Set `RECON_USE_REAL_LLM=true` / `RECON_USE_REAL_EMBEDDINGS=true` (and
`ANTHROPIC_API_KEY`) for the production stack. See `.env.example`.

## The exception queue

An exception list is only useful if a reviewer can act on it, so every row carries the
reason it fell through, the money at stake, and the nearest record the engine considered:

| record_id | category | reason | amount | related | confidence |
|---|---|---|---|---|---|
| STL-0006 | `in_transit` | settled by the PG, not yet credited by the bank | 235,574.96 | INV-006 | 0.99 |
| INV-052 | `pending_settlement` | no settlement or bank leg — closest STL-0027 (amount agrees) | 155,760.32 | STL-0027 | 0.20 |
| BNK-N0000 | `unmatched_bank_credit` | unidentified credit (REVERSAL CHGBACK) | 76,442.17 | — | — |

Rows are ranked by value at risk, because review capacity is finite and the first row a
controller sees should be the one worth the most. The `category` column exists because a
pending settlement ("wait and re-run"), a short settlement ("raise a dispute") and an
unidentified credit ("find out what this is") are three different jobs, and making a human
re-derive that triage from prose is work the pipeline should have already done.

## Repository layout

```
semantic-reconciliation-engine/
├── app.py                          # Streamlit dashboard (orchestration only)
├── config/settings.yaml            # tier thresholds, model names, offline/online toggles
├── data/
│   ├── raw/                        # generated 3-source CSV batch
│   ├── processed/                  # ground_truth.json for evaluation
│   └── outputs/                    # exceptions.csv, metrics.json from the last run
├── src/reconciler/
│   ├── schemas.py                  # records, PairMatch, ThreeWayMatch, ExceptionRecord
│   ├── config.py                   # defaults → settings.yaml → env, with real coercion
│   ├── data_loader.py              # CSV → typed records, validated with located errors
│   ├── cli.py                      # entry points (importable, so error paths are tested)
│   ├── generators/synthetic_data.py
│   ├── matchers/                   # one file per tier, common BaseMatcher contract
│   ├── engine.py                   # one pair of record sets through the 4 tiers
│   ├── pipeline.py                 # chains both legs into three-way matches
│   ├── reporting/                  # metrics.py, exceptions_report.py
│   ├── ui/                         # theme.py (design tokens + CSS), components.py
│   └── utils/                      # normalization.py, logging_config.py
├── scripts/                        # thin shims over reconciler.cli
├── tests/
│   ├── unit/                       # one component at a time, controlled inputs
│   ├── integration/                # pipeline invariants, CLI exit codes, app smoke tests
│   └── evaluation/                 # match rate + measured precision/recall as CI gates
└── .github/workflows/ci.yml        # lint · test matrix (3.10–3.12) · accuracy gates
```

## How accuracy is measured, not claimed

The synthetic generator produces 60 records across all three sources with realistic noise —
truncated bank narrations, PG merchant-ref masking, fee/tax-adjusted settlement amounts,
settlement lag, typo'd company names — and **also emits the ground-truth mapping** it used
to generate that noise. `tests/evaluation` runs the real pipeline against that ground truth
and asserts match rate, precision, and recall as pass/fail CI gates.

Precision counts a match against a record with no ground-truth triangle as a false
positive rather than skipping it, so fabricating a match for a deliberately un-settleable
invoice costs precision instead of being free.

The generator draws from a single seeded `random.Random` and never touches global RNG
state, so the same seed always yields byte-identical CSVs and the gates measure the
matcher rather than the weather.

## Development

```bash
make help          # list every target
make lint          # ruff
make test          # full suite
make coverage      # suite + HTML coverage report
make run           # generate a batch, then reconcile it
make app           # launch the dashboard
```

## Extending

- **New matcher tier:** implement `BaseMatcher.find_matches` in `src/reconciler/matchers/`,
  add it to the `matchers=[...]` list passed to `ReconciliationEngine`. Return only matches
  at or above your tier's `confidence_threshold`, and do not consume candidates for pairs
  you decline.
- **New source:** add a schema in `schemas.py`, a loader in `data_loader.py`, and a
  `..._to_candidate` adapter in `engine.py`.

## Author

Built and maintained by **Shanvi Kumari** — creator of the Semantic Reconciliation Engine,
including the four-tier escalation design, the synthetic generator with emitted ground
truth, the evaluation gates, and the dashboard.

Released under the MIT License — see [LICENSE](LICENSE).
