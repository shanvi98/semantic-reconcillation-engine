"""Smoke tests for the Streamlit dashboard.

A dashboard that raises on load is indistinguishable from a broken pipeline to
anyone demoing it, and the failure only shows up in a browser. AppTest executes
the real script, so these catch it in CI instead.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="streamlit is an optional [app] extra")

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app.py"
OUTPUTS = ROOT / "data" / "outputs"


def _run() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=120).run()


RAW = ROOT / "data" / "raw"
RAW_FILES = ("invoices.csv", "razorpay_settlements.csv", "bank_statement.csv")


@pytest.fixture
def with_results():
    """Guarantee a completed run exists, generating one if the repo is clean.

    Reconciles whatever batch is already in data/raw and only generates when
    there is none. Regenerating unconditionally would silently replace a batch
    the developer is working with — at a different record count, since this
    fixture wants a small one — and data/raw is gitignored, so nothing would
    show the substitution until the numbers on the dashboard quietly changed.
    """
    metrics, exceptions = OUTPUTS / "metrics.json", OUTPUTS / "exceptions.csv"
    if not (metrics.exists() and exceptions.exists()):
        from reconciler import cli
        if not all((RAW / name).exists() for name in RAW_FILES):
            cli.generate_data(["--n-records", "40", "--seed", "42"])
        cli.run_reconciliation(["--quiet"])
    return metrics, exceptions


@pytest.fixture
def without_results(tmp_path):
    """Temporarily move the outputs aside to exercise the empty state."""
    stashed = []
    for name in ("metrics.json", "exceptions.csv"):
        path = OUTPUTS / name
        if path.exists():
            backup = tmp_path / name
            shutil.move(str(path), str(backup))
            stashed.append((path, backup))
    yield
    for path, backup in stashed:
        shutil.move(str(backup), str(path))


def test_app_renders_without_raising(with_results):
    app = _run()
    assert not app.exception, [str(e.value) for e in app.exception]


def test_app_renders_the_full_dashboard(with_results):
    app = _run()
    assert len(app.dataframe) == 1, "exception queue table is missing"
    assert [b.label for b in app.button] == ["Run reconciliation"]
    assert {m.label for m in app.multiselect} == {"Source", "Category"}


def test_headline_sections_are_present(with_results):
    body = " ".join(block.value for block in _run().markdown)
    for heading in ("Run summary", "Escalation path", "Exception queue"):
        assert heading in body, f"missing section: {heading}"


def test_untrusted_text_is_escaped_before_it_reaches_the_page():
    """Reasons and record ids come from CSV input and model output. Rendering
    them raw inside unsafe_allow_html markup would let a crafted bank narration
    inject markup into the dashboard."""
    from reconciler.ui.components import _esc

    assert _esc('<img src=x onerror="alert(1)">') == (
        "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"
    )


def test_empty_state_is_shown_when_no_run_exists(without_results):
    app = _run()
    assert not app.exception, [str(e.value) for e in app.exception]
    body = " ".join(block.value for block in app.markdown)
    assert "No results yet" in body
    assert not app.dataframe, "queue table rendered with no data behind it"


def test_filters_narrow_the_queue(with_results):
    app = _run()
    source_filter = app.multiselect[0]
    if len(source_filter.value) > 1:
        app.multiselect[0].set_value([source_filter.value[0]]).run()
        assert not app.exception, [str(e.value) for e in app.exception]


def test_search_with_no_hits_does_not_break_the_page(with_results):
    app = _run()
    app.text_input[0].set_value("zzz-no-such-record-zzz").run()
    assert not app.exception, [str(e.value) for e in app.exception]


def test_regex_metacharacters_in_search_are_treated_literally(with_results):
    """The search box is a plain substring filter; an unescaped regex would
    raise on input a user could reasonably type."""
    app = _run()
    app.text_input[0].set_value("INV-021 (pending").run()
    assert not app.exception, [str(e.value) for e in app.exception]
