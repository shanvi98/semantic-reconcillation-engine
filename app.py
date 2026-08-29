"""Streamlit dashboard for the Semantic Reconciliation Engine.

Run with:  streamlit run app.py   (or `make app`)

This file is orchestration only — layout, data loading, and wiring. The visual
language lives in `reconciler.ui`, so the two can change independently.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from reconciler.ui import inject_theme  # noqa: E402
from reconciler.ui.components import (  # noqa: E402
    Stat,
    category_legend,
    exception_queue,
    hero,
    kpi_row,
    pipeline_flow,
    score_ring,
    score_rings,
    section,
    tier_funnel,
)

METRICS_PATH = ROOT / "data" / "outputs" / "metrics.json"
EXCEPTIONS_PATH = ROOT / "data" / "outputs" / "exceptions.csv"

RAW_DIR = ROOT / "data" / "raw"
RAW_FILES = ("invoices.csv", "razorpay_settlements.csv", "bank_statement.csv")
GROUND_TRUTH_PATH = ROOT / "data" / "processed" / "ground_truth.json"

BATCHES_DIR = ROOT / "data" / "batches"
# Which batch was last loaded. A file rather than session state: the watcher
# thread and every browser session need the same answer, and it should survive
# a restart mid-demo.
ACTIVE_BATCH_PATH = ROOT / "data" / "outputs" / ".active_batch"

WATCH_INTERVAL_SECONDS = 2.0
# An editor writing a CSV is not atomic, so a save can be observed half-written.
# Waiting for the file to stop changing costs one poll and avoids reconciling a
# truncated batch.
WRITE_SETTLE_SECONDS = 1.0

st.set_page_config(
    page_title="Semantic Reconciliation Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

# Runs are triggered from three places — the button, a batch switch, and the
# watcher thread — and nothing stops two of them coinciding. Two pipelines
# writing the same outputs at once produced a half-written metrics.json and a
# JSONDecodeError on the next read, so runs are serialised here.
_RUN_LOCK = threading.Lock()


def run_pipeline() -> tuple[bool, str]:
    """Shell out to the CLI so the dashboard runs exactly what CI runs."""
    with _RUN_LOCK:
        result = subprocess.run(
            [sys.executable, "scripts/run_reconciliation.py", "--quiet"],
            cwd=ROOT, capture_output=True, text=True,
        )
    return result.returncode == 0, f"{result.stdout or ''}\n{result.stderr or ''}".strip()


# `mtime` is deliberately un-prefixed: Streamlit skips hashing any cache
# argument whose name starts with an underscore, so the previous `_mtime`
# parameter was never part of the cache key and edits on disk were served stale.
@st.cache_data(show_spinner=False)
def load_metrics(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_exceptions(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(Path(path))


# ── Batches ──────────────────────────────────────────────────────────────────
# Selecting a batch copies it into data/raw rather than pointing the engine
# somewhere else. That keeps exactly one set of inputs in play, so the CLI, the
# tests, `make run` and the watcher all keep working unchanged — and the batch
# switch shows up to the watcher as an ordinary edit to the files it watches.

def discover_batches() -> list[dict]:
    """Batch folders that hold a complete set of inputs, in name order."""
    if not BATCHES_DIR.is_dir():
        return []

    found = []
    for directory in sorted(BATCHES_DIR.iterdir()):
        if not directory.is_dir() or not all((directory / n).exists() for n in RAW_FILES):
            continue
        manifest = {}
        manifest_path = directory / "batch.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        found.append({
            "key": directory.name,
            "label": manifest.get("label", directory.name),
            "description": manifest.get("description", ""),
            "counts": manifest.get("counts", {}),
            "dir": directory,
        })
    return found


def active_batch_key() -> str | None:
    if not ACTIVE_BATCH_PATH.exists():
        return None
    return ACTIVE_BATCH_PATH.read_text(encoding="utf-8").strip() or None


def load_batch(batch: dict) -> None:
    """Copy a batch's inputs into data/raw and its answer key into data/processed.

    The ground truth moves with the data. Copying the CSVs alone would leave the
    previous batch's key in place, and every accuracy figure on the dashboard
    would then be scored against a mapping describing different records.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)

    for name in RAW_FILES:
        shutil.copy2(batch["dir"] / name, RAW_DIR / name)

    key = batch["dir"] / "ground_truth.json"
    if key.exists():
        shutil.copy2(key, GROUND_TRUTH_PATH)

    ACTIVE_BATCH_PATH.write_text(batch["key"], encoding="utf-8")


# ── Freshness ────────────────────────────────────────────────────────────────
# The dashboard watches data/raw rather than being told when it changed, so
# edits made anywhere — an editor, a generate_data run, a script — are noticed
# the same way. mtime comparison is the whole mechanism: if any input is newer
# than metrics.json, the results on screen predate the data behind them.

def inputs_mtime() -> float:
    stamps = [p.stat().st_mtime for p in (RAW_DIR / n for n in RAW_FILES) if p.exists()]
    return max(stamps) if stamps else 0.0


def results_mtime() -> float:
    return METRICS_PATH.stat().st_mtime if METRICS_PATH.exists() else 0.0


def results_are_stale() -> bool:
    """Inputs newer than the run that produced what is on screen.

    False when no run exists at all: nothing is stale yet, and the empty state
    asking for a first run is deliberate. Auto-run re-runs a batch that moved
    underneath it — it does not decide to start one on the user's behalf.
    """
    if not METRICS_PATH.exists():
        return False
    return inputs_mtime() > results_mtime()


def ground_truth_is_stale() -> bool:
    """True when an input was edited after the answer key was written.

    `generate_data` writes the CSVs and then the ground truth, so the key is
    normally the newer file. A hand-edited CSV inverts that, and every accuracy
    figure is then scored against a mapping that no longer describes the batch —
    which is worth saying out loud rather than letting a stale 1.00 stand.
    """
    if not GROUND_TRUTH_PATH.exists():
        return False
    return inputs_mtime() > GROUND_TRUTH_PATH.stat().st_mtime


# ── The watcher ──────────────────────────────────────────────────────────────

class InputWatcher:
    """Polls data/raw in a background thread and reconciles when it changes.

    Deliberately server-side rather than a `run_every` fragment. A fragment's
    timer lives in the browser, and browsers throttle timers in tabs that are
    not visible — to roughly once a minute after a few minutes hidden. That is
    precisely the state the dashboard is in while someone edits a CSV in their
    editor, so the browser-timer version stalled exactly when it was needed and
    looked, from the outside, like a feature that did not work.

    A daemon thread owned by the server process has no such dependency: the run
    happens whether or not anything is watching. The browser is then only
    responsible for repainting, which is a much weaker thing to depend on.

    No Streamlit calls happen on this thread — it has no ScriptRunContext. It
    touches the filesystem and a lock-guarded status block, nothing else.
    """

    def __init__(self, interval: float, settle: float):
        self.interval = interval
        self.settle = settle
        self.enabled = True
        self._lock = threading.Lock()
        self._status = "starting"
        self._error: str | None = None
        self._runs = 0
        self._thread = threading.Thread(target=self._loop, name="input-watcher", daemon=True)
        self._thread.start()

    @property
    def state(self) -> tuple[str, str | None, int]:
        with self._lock:
            return self._status, self._error, self._runs

    def _set(self, status: str, error: str | None = None, counted: bool = False) -> None:
        with self._lock:
            self._status = status
            self._error = error
            if counted:
                self._runs += 1

    def _loop(self) -> None:
        # A failed run leaves metrics.json untouched, so the inputs stay stale
        # and an unguarded loop would retry the same broken CSV forever.
        # Remembering the timestamp already tried makes it one attempt per edit.
        attempted: float | None = None
        while True:
            try:
                if not self.enabled:
                    self._set("paused")
                elif not METRICS_PATH.exists():
                    # Nothing to be stale against yet; the empty state asks for
                    # a first run and that choice stays with the user.
                    self._set("no run yet")
                elif not results_are_stale():
                    self._set("up to date")
                else:
                    pending = inputs_mtime()
                    if time.time() - pending < self.settle:
                        self._set("waiting for write to settle")
                    elif pending == attempted:
                        pass  # already tried this edit; status left as the failure
                    else:
                        attempted = pending
                        self._set("reconciling")
                        ok, log = run_pipeline()
                        self._set("up to date" if ok else "run failed",
                                  None if ok else log, counted=ok)
            # Broad by intent: a watcher that dies on one unexpected error stops
            # watching silently, which is worse than reporting the error and
            # carrying on.
            except Exception as exc:
                self._set("watcher error", f"{type(exc).__name__}: {exc}")
            time.sleep(self.interval)


@st.cache_resource(show_spinner=False)
def get_watcher() -> InputWatcher:
    """One watcher per server process, shared by every browser session."""
    return InputWatcher(WATCH_INTERVAL_SECONDS, WRITE_SETTLE_SECONDS)


def money(value: float) -> str:
    for cutoff, suffix in ((1e7, "Cr"), (1e5, "L"), (1e3, "K")):
        if abs(value) >= cutoff:
            return f"₹{value / cutoff:,.2f}{suffix}"
    return f"₹{value:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    batches = discover_batches()
    if batches:
        st.markdown('<div class="side-title">Batch</div>', unsafe_allow_html=True)

        keys = [b["key"] for b in batches]
        by_key = {b["key"]: b for b in batches}
        current = active_batch_key()

        index = keys.index(current) if current in keys else 0
        # Compared against the selection this session last *rendered*, not against
        # the batch on disk. Comparing against disk would treat the first render
        # as a change and copy a batch over data/raw the moment anyone opened the
        # dashboard, discarding whatever was already there.
        previous = st.session_state.setdefault("selected_batch", keys[index])

        picked = st.selectbox(
            "Source batch",
            keys,
            index=index,
            format_func=lambda k: by_key[k]["label"],
            label_visibility="collapsed",
        )
        st.session_state["selected_batch"] = picked

        inputs_missing = not all((RAW_DIR / name).exists() for name in RAW_FILES)
        if picked != previous or (inputs_missing and current is None):
            with st.spinner(f"Loading {by_key[picked]['label']}…"):
                load_batch(by_key[picked])
                ok, log = run_pipeline()
            st.cache_data.clear()
            if not ok:
                st.session_state["batch_load_error"] = log
            else:
                st.session_state.pop("batch_load_error", None)
            st.rerun()

        if st.session_state.get("batch_load_error"):
            st.error("Batch loaded, but reconciling it failed.")
            with st.expander("Log", expanded=True):
                st.code(st.session_state["batch_load_error"], language="text")

        chosen = by_key[picked]
        counts = chosen.get("counts") or {}
        if counts:
            st.markdown(
                '<div class="side-kv"><span>Records</span>'
                f'<b>{counts.get("invoices", 0)} · {counts.get("settlements", 0)} · '
                f'{counts.get("bank", 0)}</b></div>'
                '<div class="side-kv"><span>Closeable</span>'
                f'<b>{counts.get("closeable_triangles", 0)}</b></div>',
                unsafe_allow_html=True,
            )
        if chosen.get("description"):
            st.markdown(f'<div class="note">{chosen["description"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-title">Pipeline control</div>', unsafe_allow_html=True)

    if st.button("Run reconciliation", type="primary"):
        with st.spinner("Rule → TF-IDF → FAISS → LLM over the batch…"):
            ok, log = run_pipeline()
        st.cache_data.clear()
        if ok:
            st.success("Run complete.")
        else:
            st.error("Run failed — see the log.")
        with st.expander("Run log", expanded=not ok):
            st.code(log or "(no output)", language="text")

    watcher = get_watcher()
    watcher.enabled = st.toggle(
        "Auto-run on CSV change",
        value=watcher.enabled,
        help=f"A background thread polls data/raw every {WATCH_INTERVAL_SECONDS:.0f}s "
             "and reconciles again whenever an input file is newer than the last run. "
             "It runs in the server, so it works while this tab is hidden.",
    )

    # The reconciliation itself does not need the browser; repainting does. This
    # fragment's only job is to notice the results changed underneath the page
    # and rerun the view. If the tab is hidden its timer is throttled, which now
    # costs a late repaint rather than a missed run.
    @st.fragment(run_every=WATCH_INTERVAL_SECONDS)
    def show_watcher_state() -> None:
        status, error, runs = watcher.state
        st.markdown(
            f'<div class="side-kv"><span>Watching data/raw</span><b>{status}</b></div>'
            f'<div class="side-kv"><span>Auto-runs</span><b>{runs}</b></div>',
            unsafe_allow_html=True,
        )
        if error:
            st.error("Auto-run failed — the last edit was not reconciled.")
            with st.expander("Auto-run log", expanded=True):
                st.code(error, language="text")

        # The marker is claimed *before* the rerun, not after. The sidebar runs
        # ahead of the main body, so a marker the main body owned would still
        # look stale on the very next pass and rerun again, forever.
        current = results_mtime()
        if current != st.session_state.get("seen_results_mtime"):
            st.session_state["seen_results_mtime"] = current
            st.cache_data.clear()
            st.rerun(scope="app")

    show_watcher_state()

    st.markdown('<div class="side-title">Last run</div>', unsafe_allow_html=True)
    if METRICS_PATH.exists():
        snapshot = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
        rows = {
            "Records": f"{snapshot['total_invoices'] + snapshot['total_settlements'] + snapshot['total_bank_records']:,}",
            "Elapsed": f"{snapshot['elapsed_seconds']:.2f}s",
            "Throughput": f"{snapshot['throughput_records_per_sec']:,.0f}/s",
            "Exceptions": f"{snapshot['unmatched_exceptions']:,}",
        }
        st.markdown(
            "".join(f'<div class="side-kv"><span>{k}</span><b>{v}</b></div>' for k, v in rows.items()),
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="side-kv"><span>No run yet</span><b>—</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="side-title">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Runs <b>scripts/run_reconciliation.py</b> against '
        "<b>data/raw/*.csv</b> and writes metrics and the exception queue to "
        "<b>data/outputs/</b>. With auto-run on, any edit to those CSVs "
        "reconciles again on its own. The default stack is offline and "
        "deterministic — no API key and no model download required.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

has_results = METRICS_PATH.exists() and EXCEPTIONS_PATH.exists()
hero(status="Live results" if has_results else "Awaiting first run")

if not has_results:
    st.markdown(
        '<div class="empty"><div class="mark">◈</div>'
        "<h3>No results yet</h3>"
        "<p>Press <code>Run reconciliation</code> in the sidebar, or generate a batch "
        "from the terminal with <code>make run</code>.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

metrics = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
exceptions_df = load_exceptions(str(EXCEPTIONS_PATH), EXCEPTIONS_PATH.stat().st_mtime)

total_records = (
    metrics["total_invoices"] + metrics["total_settlements"] + metrics["total_bank_records"]
)
accuracy = metrics.get("accuracy")
achievable = metrics.get("achievable_match_rate")

# ---- Headline ---------------------------------------------------------------
section(
    "Run summary",
    "Every figure below is measured against the generator's own ground truth, "
    "not sampled or estimated.",
)

kpi_row([
    Stat(
        label="Records reconciled",
        value=f"{total_records:,}",
        foot=f"<b>{metrics['total_invoices']}</b> invoices · "
             f"<b>{metrics['total_settlements']}</b> settlements · "
             f"<b>{metrics['total_bank_records']}</b> bank lines",
        color="#5B8CFF",
    ),
    Stat(
        label="Complete triangles",
        value=f"{metrics['fully_matched_triangles']:,}",
        foot=f"<b>{metrics['partial_triangles']}</b> partial · closed invoice → settlement → bank",
        color="#34D399",
        fill=metrics["match_rate"],
    ),
    Stat(
        label="Throughput",
        value=f"{metrics['throughput_records_per_sec']:,.0f}",
        unit="rec/s",
        foot=f"batch completed in <b>{metrics['elapsed_seconds']:.2f}s</b>",
        color="#38BDF8",
    ),
    Stat(
        label="Value in review",
        value=money(metrics.get("exception_value", 0.0)),
        foot=f"across <b>{metrics['unmatched_exceptions']}</b> exception(s)",
        color="#FBBF24",
    ),
])

# ---- Accuracy ---------------------------------------------------------------
if accuracy:
    section(
        "Measured accuracy",
        "Match rate alone is trivially gamed by matching everything at low confidence, "
        "so precision and recall are scored against ground truth the generator emitted "
        "alongside the data.",
    )

    if ground_truth_is_stale():
        st.markdown(
            '<div class="note warn"><b>Ground truth is older than the inputs.</b> '
            "A file under <b>data/raw/</b> has been edited since the answer key was "
            "written, so precision, recall and F1 below are scored against a mapping "
            "that no longer describes this batch. Match rate and the exception queue "
            "are still accurate. Run <b>python scripts/generate_data.py</b> to emit a "
            "matching key, or read the accuracy figures as stale.</div>",
            unsafe_allow_html=True,
        )

    rings = [
        score_ring(
            metrics["match_rate"], "match rate",
            "Complete triangles over all invoices — capped below 100% by legs the "
            "generator dropped on purpose.",
            "#5B8CFF", 0.0,
        ),
    ]
    if achievable is not None:
        rings.append(score_ring(
            achievable, "achievable",
            "Of the triangles that can close at all, the share the engine actually found.",
            "#34D399", 0.1,
        ))
    rings += [
        score_ring(
            accuracy["precision"], "precision",
            f"Of the triangles claimed complete, the share that are right. "
            f"{accuracy['incorrect']} wrong.",
            "#A78BFA", 0.2,
        ),
        score_ring(
            accuracy["recall"], "recall",
            f"Of the triangles that truly exist, the share found. {accuracy['missed']} missed.",
            "#38BDF8", 0.3,
        ),
        score_ring(accuracy["f1"], "f1", "Harmonic mean of precision and recall.", "#FBBF24", 0.4),
    ]
    score_rings(rings)

# ---- Escalation -------------------------------------------------------------
section(
    "Escalation path",
    "Each tier is handed only what the tier before it declined. A batch that resolves "
    "mostly on the left is a batch that closed cheaply — the model tier exists for the "
    "residual, not the bulk.",
)

breakdown = metrics["tier_breakdown"]
timings = metrics.get("tier_timings") or []
# The records that fell through every tier. Counted from the cascade itself
# rather than from exception categories: the categories mix both sides of a
# pairing together, which is the right view for a reviewer and the wrong one
# for a diagram about what flowed through the funnel.
unresolved_left = int(metrics.get("unresolved_left", 0))
unresolved_right = int(metrics.get("unresolved_right", 0))

pipeline_flow(timings, unresolved_left)
tier_funnel(breakdown, timings)

resolved_cheaply = breakdown.get("rule", 0) + breakdown.get("tfidf", 0)
total_matches = sum(breakdown.values()) or 1
entered = next((int(t.get("candidates_in", 0)) for t in timings if t.get("tier") == "rule"), 0)
st.markdown(
    f'<div class="note"><b>{resolved_cheaply / total_matches:.0%}</b> of matches resolved on the '
    f"two cheapest tiers, with <b>{breakdown.get('llm', 0)}</b> reaching the model. "
    "That ratio is the design working: the expensive tier stays cheap because it "
    "almost never runs.<br>"
    f"The cascade was handed <b>{entered:,}</b> records across both legs — invoices against "
    "settlements, then settlements against bank lines — so a settlement is counted once per "
    f"leg it takes part in. <b>{unresolved_left:,}</b> survived all four tiers, and a further "
    f"<b>{unresolved_right:,}</b> counterparties were never claimed by any match; together "
    f"they are the <b>{metrics['unmatched_exceptions']:,}</b> rows in the queue below.</div>",
    unsafe_allow_html=True,
)

# ---- Exceptions -------------------------------------------------------------
section(
    "Exception queue",
    "Ranked by value at risk, because review capacity is finite and the first row "
    "should be the one worth the most.",
)

by_category = metrics.get("exceptions_by_category", {})
if by_category:
    category_legend(by_category)

filtered = exception_queue(exceptions_df)

if not filtered.empty:
    st.download_button(
        "Download queue (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="exceptions_filtered.csv",
        mime="text/csv",
    )
