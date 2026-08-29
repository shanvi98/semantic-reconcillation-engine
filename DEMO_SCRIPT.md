# 5-Minute Demo Script — Semantic Reconciliation Engine

Companion to `demo_runsheet.csv` (same content, machine-readable). Read this one on stage.

**The one thing you must not forget:**

```powershell
$env:RECON_USE_REAL_EMBEDDINGS="false"
```

`config/settings.yaml` ships with `use_real_embeddings: true`, which makes the FAISS tier
load MiniLM on every run: **12.54s instead of 0.048s**, for byte-identical matches and
identical precision/recall. That is 13 seconds of dead air out of 300. Set the variable
in the same shell you launch Streamlit from — the dashboard shells out to the CLI and
inherits the environment.

---

## Pre-demo checklist

Do all of this before anyone is watching. Nothing here happens on stage.

| When | Command | Why |
|---|---|---|
| T-10 | `.venv\Scripts\activate` | A demo that starts with an install ends at minute five. |
| T-9 | `$env:RECON_USE_REAL_EMBEDDINGS="false"` | Sub-second live run, same accuracy. |
| T-8 | `python scripts/generate_data.py`<br>`python scripts/run_reconciliation.py` | Dashboard opens on **Live results**, not the empty state. |
| T-7 | `python -m pytest -q` | Confirms `167 passed`. Leave the terminal open on a second tab — it is your closing slide. |
| T-5 | `python -m streamlit run app.py` | Zoom the browser to **80%** so the KPI row and the score rings fit one screen. Dark theme is already forced in `.streamlit/config.toml`. |
| T-2 | Open `data/raw/razorpay_settlements.csv` and `data/raw/bank_statement.csv` in a second editor tab | You jump here for 25 seconds at 3:50. No file browsing on stage. |

**Tabs you should have open, in order:** dashboard → CSV tab → terminal with green tests.

---

## The script

### 0:00 — Hook · the problem (40s)

**Show:** the hero banner — title, tagline, and the four-node chain `Rule → TF-IDF → FAISS → LLM`.

> Finance ops has a verification problem, not a matching problem. Bank statements truncate
> narrations, payment gateways mask merchant refs, invoices carry typo'd customer names.
> Exact-match reconciliation breaks on all three, and a human ends up eyeballing spreadsheets.

*Point at the four-node chain.*

> Most demos do a two-way match — settlement against bank, or invoice against settlement.
> This closes the full **triangle**: Invoice → Razorpay Settlement → Bank. A settlement that
> never reaches the bank, and an invoice that was never settled, are exactly the gaps finance
> ops loses days chasing — and a two-way match cannot see either of them.

**Why it lands:** the hero states your thesis before a single number appears. Judges should
know what problem you own inside 20 seconds. Say the word *triangle* — it is what they remember.

---

### 0:40 — The data (45s)

**Show:** the sidebar *Last run* block — Records / Elapsed / Throughput / Exceptions.

> 180 records across three ledgers: 60 invoices, 57 settlements, 63 bank lines. Deliberately
> not clean — the generator injects truncated narrations, masked merchant refs, gateway fee
> and tax deductions, settlement lag, and typo'd company names.

**Point at the mismatch.** 60 vs 57 vs 63 *is* the problem statement. If the ledgers tied,
there would be nothing to reconcile.

> And critically, the generator also emits the **ground-truth mapping** it used to create that
> noise. So every accuracy number on this screen is measured against a known answer key,
> not asserted.

**Why it lands:** this is the credibility hook for the accuracy rings 90 seconds from now.
Plant it here and collect on it at 2:40.

---

### 1:25 — Run it live (35s)

**Do:** click **Run reconciliation** in the sidebar.

> Let me just run it. This is not a cached screenshot — the dashboard shells out to the exact
> same CLI that CI runs, `scripts/run_reconciliation.py`.

*KPI row repaints.*

> 180 records, under 50 milliseconds — a few thousand records a second. And the accuracy did not move —
> I will show you in a moment why that matters.

**Why it lands:** same entry point as CI means the demo cannot diverge from the tested path.
The engineers on the panel will notice.

**If it fails:** expand the **Run log** — it is right there in the sidebar, and showing it
calmly reads as confidence. Then fall back to the pre-warmed numbers already on screen.

---

### 2:00 — Run summary (40s)

**Show:** the four KPI cards.

> 180 records reconciled. **53 complete triangles** — invoice, settlement and bank, all three
> legs closed — plus 4 partials where one leg is still missing.

*Point at **Value in review**.*

> And the number a controller actually cares about: **₹12.9 lakh** sitting in 17 exceptions.
> Not 17 rows to triage — 12.9 lakh of exposure, ranked.

**Why it lands:** the complete triangle is your unit of work, not "matches." And translating
a row count into money is what separates a dashboard from a report.

---

### 2:40 — Measured accuracy (45s) ← your strongest moment

**Show:** the five score rings.

> Two match rates, and only the second one is about the engine. **88.3%** divides by all 60
> invoices *including the ones the generator deliberately left un-settleable*, so it is capped
> below 100% by construction — comparing it against a 100% target measures my test fixture,
> not my matcher. **Achievable match rate** divides by the triangles that can close at all.
> That is **100%**: the engine found everything findable.

*Point at precision.*

> Precision, recall, F1 — all **1.00**, zero wrong, zero missed, scored against that emitted
> ground truth. And precision counts a match against a record with no true triangle as a false
> positive rather than skipping it, so fabricating a match for a deliberately un-settleable
> invoice costs me precision instead of being free.

**Why it lands:** you are volunteering the weakness of your own headline metric before anyone
can ask. Judges reward that far more than a clean number. And anyone can hit 100% match rate
by matching everything at low confidence — precision scored this way is what makes it
un-gameable.

---

### 3:25 — The escalation path (50s) ← the core idea

**Show:** the pipeline flow diagram and the tier funnel. Run your finger down it: **106 / 4 / 0 / 0**.

> Here is the actual design. Four tiers, cheapest first, and every tier sees only what the tier
> before it **declined**. 117 records enter — 60 invoices against settlements, then 57
> settlements against bank lines. Rule matching, exact ID or UTR overlap, took 106 of them in
> 1.4 milliseconds and passed 8 on. TF-IDF took 4. FAISS and the LLM tier saw the last 4 and
> took none. **100% of matches resolved on the two cheapest tiers**, and 7 records survived
> all four.

**Switch to the CSV tab.** Show `STL-0001` → `RAZORP*UDAA001`, then `BNK-00001` → `RTGS-RAZORP*UDAA00`.

> This is one of the four TF-IDF catches. The settlement ref is `RAZORP*UDAA001`. The bank
> narration is `RTGS-RAZORP*UDAA00` — truncated, last character gone, **and the UTR column is
> empty**. Tier 1 has nothing to hook onto. Char n-gram TF-IDF still matches it at 0.89,
> because sub-word overlap survives a truncation that whole-word overlap does not.

**Back to the dashboard**, the note line under the funnel.

> Two properties make that escalation trustworthy rather than just plausible. **Declining is
> free** — a tier that scores a pair below its own bar leaves *both* records available to the
> next tier, so a rejection never starves the tiers behind it. And **model output is untrusted**
> — the LLM tier validates the id it gets back against the shortlist it actually offered, so a
> hallucinated id becomes a no-match, never a match pointing at a record that does not exist.

**Why it lands:** the expensive tier stays cheap because it almost never runs — that is the
whole economic argument, in one screen. The `RAZORP*UDAA00` record is the single most
memorable 25 seconds of the demo: one concrete row beats any amount of architecture talk.
And the hallucination-safety line pre-answers a question a judge was going to ask anyway.

---

### 4:30 — Exception queue (35s)

**Show:** the queue and the category legend pills. Point at the top row, `STL-0006`.

> Whatever survives all four tiers becomes a **reasoned exception**, not a forced guess. Ranked
> by value at risk, because review capacity is finite and the first row a controller sees should
> be the one worth the most. Top row: ₹2.36 lakh, settled by the gateway, not yet credited by
> the bank — and it names `INV-006` as the related record, so the reviewer knows where to start.

**Do:** open the **Category** filter, keep only *Pending settlement*. Then click **Download queue (CSV)**.

> The category column exists because these are three different jobs. A pending settlement means
> *wait and re-run*. A short settlement means *raise a dispute*. An unidentified credit means
> *go find out what this is*. Making a human re-derive that triage from prose is work the
> pipeline should already have done. And it exports, so it lands in whatever the team actually
> works out of.

**Why it lands:** ranked by money rather than ID order, every row carrying its reason and the
nearest record considered. Use the filter — do not just describe it. The export is what makes
this a workflow instead of a poster.

---

### 5:05 — Close (15s)

**Show:** the terminal tab reading `167 passed in 2.34s`.

> All of it is asserted. **167 tests**, offline and deterministic, no network and no API key —
> including the match rate and the precision and recall you just saw, which run as **pass/fail
> CI gates** on Python 3.10 through 3.12. The accuracy is not a claim in a README. It is a
> build failure if it regresses.

**Why it lands:** ending on green tests is the strongest possible close, and accuracy gates in
CI are rare in a hackathon build.

---

## If you are running long

Cut in this order. Each cut buys you the time in brackets.

1. **[25s] The CSV tab detour at 3:50.** The funnel already made the point. Painful but survivable.
2. **[20s] The second data beat at 1:05.** Fold "emits its own ground truth" into the accuracy section instead.
3. **[20s] KPI cards 1 and 2 at 2:00.** The rings re-state the same story with better numbers.

**Never cut:** the achievable-match-rate explanation (2:40) or the escalation funnel (3:25).
Those two are the demo.

---

## Q&A — the seven you will actually get

**Q: FAISS and the LLM found nothing. Isn't that a failure?**
No — it is the design succeeding. The cheap tiers cleared everything, so the expensive tiers
had nothing left to adjudicate. On a noisier batch they engage, and the tier order guarantees
they only ever cost you what the cheaper tiers could not solve. I would be worried if the LLM
tier were carrying the batch.

**Q: The README says 12.5 seconds, you showed 0.05. Which is real?**
Both, and the difference is honest: 12.5 seconds is loading the MiniLM sentence-transformer,
not matching. With the offline hashing embedder it is 48 milliseconds with byte-identical
accuracy. The semantic tier is an exhaustive inner-product scan either way — the FAISS
`IndexFlatIP` path and the numpy path return the same neighbours in the same order, and a test
pins that equivalence rather than assuming it.

**Q: Is this real data?**
Synthetic, seeded, and deliberately so — because the generator emits the ground-truth mapping
it used to inject the noise, which is the only reason I can show you a *measured* precision
instead of an asserted one. Same seed, byte-identical CSVs, every run. Swapping in real ledgers
is a loader change in `data_loader.py`, not an engine change.

**Q: What happens when the LLM hallucinates a match?**
It cannot produce one. The tier parses the verdict defensively — fenced code blocks and leading
prose are both tolerated — then validates the returned id against the shortlist it actually
offered and clamps the confidence into range. An invented id degrades to a no-match and lands
in the exception queue. A wrong match is more costly than no match, and the code is written
that way. *(Have `src/reconciler/matchers/llm_matcher.py` open if you want to show `parse_verdict`.)*

**Q: How do you add a fourth source, or a fifth tier?**
A tier is one class implementing `BaseMatcher.find_matches`, added to the matchers list. The
contract is: return only matches at or above your own threshold, and do not consume candidates
for pairs you decline. A source is a schema, a loader, and a `..._to_candidate` adapter. Both
are written up in the README under *Extending*.

**Q: Why is triangle confidence the minimum of its edges, not the mean?**
A chain is as strong as its weakest link. If the invoice→settlement edge is 0.99 and the
settlement→bank edge is 0.62, the triangle is 0.62. Averaging would let a strong edge launder
a weak one into looking trustworthy.

**Q: Why does an exception show a related record at 0.20 confidence?**
That is the near-miss reporter, and it runs only *after* all four tiers have declined. Its job
is to be legible to a reviewer — "shared 2 of 6 tokens, amount agrees" — not to make a matching
decision. It also scores against the *full* candidate pool rather than just the leftovers,
because the most useful hint is often that a record another invoice already claimed looks like
yours. That points at a duplicate.

---

## Numbers cheat-sheet

Keep this visible. Every figure verified against a live run.

| | |
|---|---|
| Records | **180** — 60 invoices · 57 settlements · 63 bank lines |
| Complete triangles | **53** (+ 4 partial) |
| Match rate | **88.3%** — capped by construction |
| **Achievable match rate** | **100%** — the number about the engine |
| Precision / Recall / F1 | **1.00 / 1.00 / 1.00** — 0 wrong, 0 missed |
| Tier split | rule **106** · tfidf **4** · faiss **0** · llm **0** |
| Cascade | **117** in → 8 to TF-IDF → 4 to FAISS/LLM → **7** unresolved |
| Queue arithmetic | 7 unresolved + 10 unclaimed counterparties = **17** exceptions |
| Resolved on two cheapest tiers | **100%** |
| Runtime | ~**0.04s** → **3,700-4,500 rec/s**, varies run to run (offline embedder) |
| Exceptions | **17**, worth **₹12.92L** |
| By category | 10 unmatched bank credit · 4 in-transit · 3 pending settlement |
| Tests | **167 passed in 2.34s**, Python 3.10–3.12 |
| Top exception | `STL-0006` · ₹2,35,574.96 · in-transit · related `INV-006` |
| The TF-IDF hero record | `RAZORP*UDAA001` → `RTGS-RAZORP*UDAA00`, no UTR, matched 0.89 |
