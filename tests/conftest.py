"""
Shared fixtures for the whole suite.

Everything defaults to the *offline, deterministic* stack (HashingEmbedder +
HeuristicLLMClient) so `pytest` runs green with no network access, no model
download, and no ANTHROPIC_API_KEY — required for CI to be trustworthy.
Set RECON_TEST_USE_REAL_STACK=true locally to run against the real
sentence-transformers embedder / Claude API instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconciler.data_loader import load_bank, load_invoices, load_settlements
from reconciler.engine import ReconciliationEngine
from reconciler.generators.synthetic_data import generate_dataset
from reconciler.matchers.faiss_matcher import FaissMatcher, HashingEmbedder
from reconciler.matchers.llm_matcher import HeuristicLLMClient, LlmMatcher
from reconciler.matchers.rule_matcher import RuleMatcher
from reconciler.matchers.tfidf_matcher import TfidfMatcher
from reconciler.pipeline import ThreeWayReconciliationPipeline

N_RECORDS = 60  # exceeds the track's 50+ record bar with room for injected noise
SEED = 42
USE_REAL_STACK = os.environ.get("RECON_TEST_USE_REAL_STACK", "false").lower() == "true"


@pytest.fixture(scope="session")
def synthetic_batch(tmp_path_factory):
    data = generate_dataset(n_records=N_RECORDS, seed=SEED)
    out_dir = tmp_path_factory.mktemp("synthetic_data")

    data["invoices"].to_csv(out_dir / "invoices.csv", index=False)
    data["settlements"].to_csv(out_dir / "razorpay_settlements.csv", index=False)
    data["bank"].to_csv(out_dir / "bank_statement.csv", index=False)

    return {
        "invoices": load_invoices(out_dir / "invoices.csv"),
        "settlements": load_settlements(out_dir / "razorpay_settlements.csv"),
        "bank": load_bank(out_dir / "bank_statement.csv"),
        "ground_truth": data["ground_truth"],
    }


@pytest.fixture(scope="session")
def offline_engine():
    faiss_tier = FaissMatcher() if USE_REAL_STACK else FaissMatcher(embedder=HashingEmbedder())
    llm_tier = LlmMatcher(client=None if USE_REAL_STACK else HeuristicLLMClient())
    return ReconciliationEngine(matchers=[RuleMatcher(), TfidfMatcher(), faiss_tier, llm_tier])


@pytest.fixture(scope="session")
def pipeline_result(synthetic_batch, offline_engine):
    pipeline = ThreeWayReconciliationPipeline(engine=offline_engine)
    return pipeline.run(synthetic_batch["invoices"], synthetic_batch["settlements"], synthetic_batch["bank"])
