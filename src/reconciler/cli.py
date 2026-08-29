"""Command-line entry points.

The logic lives here rather than in ``scripts/`` so it is importable and
testable like the rest of the package — a CLI whose behaviour can only be
exercised by shelling out is a CLI whose error paths never get tested. The
files under ``scripts/`` are thin shims over these functions, kept for the
familiar ``python scripts/run_reconciliation.py`` invocation.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

from .config import ConfigError, Settings, load_settings
from .data_loader import DataLoadError, load_bank, load_invoices, load_settlements
from .engine import ReconciliationEngine
from .generators.synthetic_data import generate_dataset
from .matchers.faiss_matcher import (
    FaissMatcher,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from .matchers.llm_matcher import AnthropicLLMClient, LlmMatcher
from .matchers.rule_matcher import RuleMatcher
from .matchers.tfidf_matcher import TfidfMatcher
from .pipeline import ThreeWayReconciliationPipeline
from .reporting.exceptions_report import summarize, write_atomically, write_exceptions_csv
from .reporting.metrics import compute_metrics
from .utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def json_default(obj):
    """JSON encoder for the nested dataclass/enum metrics tree."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def build_engine(settings: Settings) -> ReconciliationEngine:
    """Assemble the four tiers against whichever stack the settings select."""
    embedder = (
        SentenceTransformerEmbedder(settings.embedding_model)
        if settings.use_real_embeddings
        else HashingEmbedder()
    )
    llm_client = AnthropicLLMClient(settings.llm_model) if settings.use_real_llm else None
    return ReconciliationEngine(matchers=[
        RuleMatcher(),
        TfidfMatcher(),
        FaissMatcher(embedder=embedder),
        LlmMatcher(client=llm_client),
    ])


# ---- generate-data ----------------------------------------------------------

def _generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic reconciliation batch")
    parser.add_argument("--n-records", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pending-settlement-rate", type=float, default=0.08,
                        help="fraction of invoices with no settlement leg")
    parser.add_argument("--in-transit-rate", type=float, default=0.08,
                        help="fraction of settlements with no bank leg")
    parser.add_argument("--dispute-rate", type=float, default=0.06,
                        help="fraction of settlements short-paid beyond fee tolerance")
    parser.add_argument("--bank-noise-records", type=int, default=10,
                        help="unrelated bank credits with no invoice or settlement")
    return parser


def generate_data(argv: list[str] | None = None) -> int:
    args = _generate_parser().parse_args(argv)
    setup_logging()
    settings = load_settings()
    out_dir = args.out_dir or settings.data_raw_dir

    try:
        data = generate_dataset(
            n_records=args.n_records,
            seed=args.seed,
            pending_settlement_rate=args.pending_settlement_rate,
            in_transit_rate=args.in_transit_rate,
            dispute_rate=args.dispute_rate,
            bank_noise_records=args.bank_noise_records,
        )
    except ValueError as exc:
        logger.error("Invalid generator parameters: %s", exc)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    data["invoices"].to_csv(out_dir / "invoices.csv", index=False)
    data["settlements"].to_csv(out_dir / "razorpay_settlements.csv", index=False)
    data["bank"].to_csv(out_dir / "bank_statement.csv", index=False)

    gt_path = settings.data_processed_dir / "ground_truth.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(data["ground_truth"], indent=2), encoding="utf-8")

    resolvable = sum(1 for row in data["ground_truth"] if row["settlement_id"] and row["bank_txn_id"])

    print(f"invoices:    {len(data['invoices'])} rows")
    print(f"settlements: {len(data['settlements'])} rows")
    print(f"bank:        {len(data['bank'])} rows")
    print(f"complete triangles in ground truth: {resolvable}/{len(data['invoices'])}")
    print(f"written to {out_dir} (seed={args.seed})")
    print(f"ground truth -> {gt_path}")
    return 0


# ---- run-reconciliation -----------------------------------------------------

def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the three-way reconciliation pipeline")
    parser.add_argument("--raw-dir", type=Path, default=None, help="override data/raw location")
    parser.add_argument("--out-dir", type=Path, default=None, help="override data/outputs location")
    parser.add_argument("--quiet", action="store_true", help="print only the summary, not the metrics JSON")
    return parser


def run_reconciliation(argv: list[str] | None = None) -> int:
    args = _run_parser().parse_args(argv)
    setup_logging()

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    raw_dir = args.raw_dir or settings.data_raw_dir
    out_dir = args.out_dir or settings.data_outputs_dir
    logger.info("Stack: %s", settings.stack_description)

    try:
        invoices = load_invoices(raw_dir / "invoices.csv")
        settlements = load_settlements(raw_dir / "razorpay_settlements.csv")
        bank = load_bank(raw_dir / "bank_statement.csv")
    except DataLoadError as exc:
        logger.error("Could not load input data: %s", exc)
        return 2

    logger.info(
        "Loaded %d invoices, %d settlements, %d bank records",
        len(invoices), len(settlements), len(bank),
    )

    pipeline = ThreeWayReconciliationPipeline(engine=build_engine(settings))

    start = time.perf_counter()
    result = pipeline.run(invoices, settlements, bank)
    elapsed = time.perf_counter() - start

    gt_path = settings.data_processed_dir / "ground_truth.json"
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None
    if ground_truth is None:
        logger.warning("No ground truth at %s — accuracy metrics will be omitted", gt_path)

    metrics = compute_metrics(
        result, len(invoices), len(settlements), len(bank), elapsed, ground_truth=ground_truth
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    exceptions_path = write_exceptions_csv(result.exceptions, out_dir / "exceptions.csv")
    metrics_path = out_dir / "metrics.json"
    payload = json.dumps(metrics, default=json_default, indent=2)
    write_atomically(metrics_path, lambda target: target.write_text(payload, encoding="utf-8"))

    if not args.quiet:
        print(json.dumps(metrics, default=json_default, indent=2))

    print()
    print(summarize(result.exceptions))
    print()
    print(f"metrics    -> {metrics_path}")
    print(f"exceptions -> {exceptions_path}")
    return 0


# ---- build-index ------------------------------------------------------------

def build_faiss_index(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a FAISS index over bank narrations")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    setup_logging()
    settings = load_settings()

    try:
        import faiss
    except ImportError:
        logger.error("faiss is not installed — install the embeddings extra: pip install -e '.[embeddings]'")
        return 2

    try:
        bank = load_bank(settings.data_raw_dir / "bank_statement.csv")
    except DataLoadError as exc:
        logger.error("%s", exc)
        return 2

    from .utils.normalization import clean_text

    # Mirrors the pipeline's switch so this utility is runnable offline too.
    embedder = (
        SentenceTransformerEmbedder(settings.embedding_model)
        if settings.use_real_embeddings
        else HashingEmbedder()
    )
    logger.info("Embedding %d narrations with %s", len(bank), type(embedder).__name__)

    vectors = embedder.encode([clean_text(r.narration) for r in bank])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out_dir = args.out_dir or settings.data_outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "bank_narrations.index"
    faiss.write_index(index, str(index_path))
    (out_dir / "bank_narrations_ids.json").write_text(
        json.dumps([r.bank_txn_id for r in bank], indent=2), encoding="utf-8"
    )
    print(f"Indexed {len(bank)} bank narrations ({vectors.shape[1]}-dim) -> {index_path}")
    return 0
