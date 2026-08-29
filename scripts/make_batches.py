#!/usr/bin/env python
"""Build the demo batches the dashboard can switch between.

Each batch is a self-contained folder under ``data/batches/<name>/`` holding the
three source CSVs, the ground truth emitted alongside them, and a small
``batch.json`` describing what the batch is for.

Keeping the answer key *inside* the batch is the point: the accuracy figures on
the dashboard are only meaningful against the mapping that generated that exact
data, so a batch that travelled without its key would show a confident, wrong
precision the moment it was selected.

Run with:  python scripts/make_batches.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconciler.generators.synthetic_data import generate_dataset

BATCHES_DIR = ROOT / "data" / "batches"

# Three batches that look genuinely different on the dashboard, rather than three
# reseeds of the same shape: a reference month, a bad month, and a big month.
BATCHES = [
    {
        "key": "batch1",
        "label": "Batch 1 · Standard",
        "description": (
            "The reference month. 60 invoices with ordinary noise — this is the "
            "batch every number in the demo script was measured against."
        ),
        "params": {"n_records": 60, "seed": 42},
    },
    {
        "key": "batch2",
        "label": "Batch 2 · High exceptions",
        "description": (
            "A bad month. A quarter of invoices never settle, a fifth of "
            "settlements never reach the bank, and the statement is full of "
            "unrelated credits. Match rate falls; the engine still finds "
            "everything findable."
        ),
        "params": {
            "n_records": 60,
            "seed": 7,
            "pending_settlement_rate": 0.25,
            "in_transit_rate": 0.20,
            "dispute_rate": 0.20,
            "bank_noise_records": 30,
        },
    },
    {
        "key": "batch3",
        "label": "Batch 3 · Scale",
        "description": (
            "A big month. 300 invoices, roughly 900 records across the three "
            "ledgers — the batch to select when someone asks whether it holds "
            "up beyond a toy dataset."
        ),
        "params": {"n_records": 300, "seed": 42},
    },
]


def build(spec: dict) -> dict:
    out_dir = BATCHES_DIR / spec["key"]
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_dataset(**spec["params"])

    data["invoices"].to_csv(out_dir / "invoices.csv", index=False)
    data["settlements"].to_csv(out_dir / "razorpay_settlements.csv", index=False)
    data["bank"].to_csv(out_dir / "bank_statement.csv", index=False)
    (out_dir / "ground_truth.json").write_text(
        json.dumps(data["ground_truth"], indent=2), encoding="utf-8"
    )

    resolvable = sum(
        1 for row in data["ground_truth"] if row["settlement_id"] and row["bank_txn_id"]
    )
    manifest = {
        "key": spec["key"],
        "label": spec["label"],
        "description": spec["description"],
        "params": spec["params"],
        "counts": {
            "invoices": len(data["invoices"]),
            "settlements": len(data["settlements"]),
            "bank": len(data["bank"]),
            "closeable_triangles": resolvable,
        },
    }
    (out_dir / "batch.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    for spec in BATCHES:
        manifest = build(spec)
        counts = manifest["counts"]
        print(
            f"{manifest['label']:<28} "
            f"{counts['invoices']:>4} inv · {counts['settlements']:>4} stl · "
            f"{counts['bank']:>4} bank · {counts['closeable_triangles']:>4} closeable"
        )
    print(f"\nwritten to {BATCHES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
