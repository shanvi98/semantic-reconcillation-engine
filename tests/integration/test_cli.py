"""The CLI is the surface most people touch first, and its error paths are the
ones that matter: a bad input file should produce a clear message and a non-zero
exit code, not a traceback and an exit code of 0 that CI reads as success.
"""
from __future__ import annotations

import json

import pytest

from reconciler import cli


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point every configured path at a temp dir so nothing touches the repo."""
    raw, processed, outputs = tmp_path / "raw", tmp_path / "processed", tmp_path / "outputs"
    monkeypatch.setenv("RECON_DATA_RAW_DIR", str(raw))
    monkeypatch.setenv("RECON_DATA_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("RECON_DATA_OUTPUTS_DIR", str(outputs))
    # Offline stack: no model download, no API key, no network.
    monkeypatch.setenv("RECON_USE_REAL_EMBEDDINGS", "false")
    monkeypatch.setenv("RECON_USE_REAL_LLM", "false")
    return {"raw": raw, "processed": processed, "outputs": outputs}


def test_generate_then_run_produces_both_artifacts(workspace, capsys):
    assert cli.generate_data(["--n-records", "40", "--seed", "42"]) == 0
    assert (workspace["raw"] / "invoices.csv").exists()
    assert (workspace["processed"] / "ground_truth.json").exists()

    assert cli.run_reconciliation(["--quiet"]) == 0
    metrics_path = workspace["outputs"] / "metrics.json"
    assert metrics_path.exists()
    assert (workspace["outputs"] / "exceptions.csv").exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["total_invoices"] == 40
    assert metrics["accuracy"]["precision"] >= 0.95


def test_metrics_json_is_fully_serializable(workspace):
    """The metrics tree is nested dataclasses and enums; a default= encoder that
    misses a type fails at write time, after the run has already been paid for."""
    cli.generate_data(["--n-records", "20", "--seed", "1"])
    cli.run_reconciliation(["--quiet"])

    metrics = json.loads((workspace["outputs"] / "metrics.json").read_text(encoding="utf-8"))
    assert isinstance(metrics["tier_timings"], list)
    assert isinstance(metrics["exceptions_by_category"], dict)
    assert isinstance(metrics["tier_breakdown"], dict)


def test_run_without_input_data_exits_nonzero_with_a_clear_message(workspace, caplog):
    with caplog.at_level("ERROR"):
        code = cli.run_reconciliation([])
    assert code == 2
    assert "does not exist" in caplog.text


def test_run_against_a_malformed_csv_exits_nonzero(workspace, caplog):
    cli.generate_data(["--n-records", "10"])
    (workspace["raw"] / "invoices.csv").write_text("wrong,columns\n1,2\n", encoding="utf-8")

    with caplog.at_level("ERROR"):
        code = cli.run_reconciliation([])
    assert code == 2
    assert "missing required column" in caplog.text


def test_generate_rejects_invalid_parameters(workspace, caplog):
    with caplog.at_level("ERROR"):
        assert cli.generate_data(["--n-records", "0"]) == 2
    assert "n_records" in caplog.text


def test_run_without_ground_truth_still_succeeds(workspace, caplog):
    """Reconciliation of real data has no ground truth. Accuracy metrics drop
    out; the run itself must not."""
    cli.generate_data(["--n-records", "20"])
    (workspace["processed"] / "ground_truth.json").unlink()

    with caplog.at_level("WARNING"):
        assert cli.run_reconciliation(["--quiet"]) == 0
    assert "No ground truth" in caplog.text

    metrics = json.loads((workspace["outputs"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["accuracy"] is None
    assert metrics["achievable_match_rate"] is None


def test_output_directories_are_created_on_demand(workspace):
    cli.generate_data(["--n-records", "10"])
    assert not workspace["outputs"].exists()
    assert cli.run_reconciliation(["--quiet"]) == 0
    assert workspace["outputs"].is_dir()


def test_offline_settings_select_the_offline_backends(workspace):
    from reconciler.config import load_settings
    from reconciler.matchers.faiss_matcher import HashingEmbedder
    from reconciler.matchers.llm_matcher import HeuristicLLMClient

    engine = cli.build_engine(load_settings())
    faiss_tier = next(m for m in engine.matchers if m.tier.value == "faiss")
    llm_tier = next(m for m in engine.matchers if m.tier.value == "llm")

    assert isinstance(faiss_tier.embedder, HashingEmbedder)
    assert isinstance(llm_tier.client, HeuristicLLMClient)
