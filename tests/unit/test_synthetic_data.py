"""The generator is the source of ground truth for every accuracy number this
project reports, so its determinism is load-bearing: if the same seed can
produce different data, the evaluation gates are measuring noise.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from reconciler.generators.synthetic_data import generate_dataset


def test_same_seed_produces_identical_data():
    a = generate_dataset(n_records=30, seed=7)
    b = generate_dataset(n_records=30, seed=7)
    assert a["invoices"].equals(b["invoices"])
    assert a["settlements"].equals(b["settlements"])
    assert a["bank"].equals(b["bank"])
    assert a["ground_truth"] == b["ground_truth"]


def test_different_seeds_produce_different_data():
    a = generate_dataset(n_records=30, seed=1)
    b = generate_dataset(n_records=30, seed=2)
    assert not a["invoices"].equals(b["invoices"])


def test_generation_does_not_touch_the_global_rng():
    """Regression: the generator used to call `random.seed()`, so anything else
    in the process drawing from the global RNG had its stream silently reset.
    That turns unrelated tests into order-dependent flakes.
    """
    random.seed(999)
    expected = [random.random() for _ in range(3)]

    random.seed(999)
    generate_dataset(n_records=10, seed=42)
    actual = [random.random() for _ in range(3)]

    assert actual == expected


def test_ground_truth_covers_every_invoice():
    data = generate_dataset(n_records=40, seed=3)
    assert len(data["ground_truth"]) == len(data["invoices"]) == 40
    assert {row["invoice_id"] for row in data["ground_truth"]} == set(data["invoices"]["invoice_id"])


def test_ground_truth_ids_actually_exist_in_the_data():
    """A ground truth that references records the generator never emitted would
    make every accuracy number meaningless."""
    data = generate_dataset(n_records=40, seed=3)
    settlement_ids = set(data["settlements"]["settlement_id"])
    bank_ids = set(data["bank"]["bank_txn_id"])

    for row in data["ground_truth"]:
        if row["settlement_id"]:
            assert row["settlement_id"] in settlement_ids
        if row["bank_txn_id"]:
            assert row["bank_txn_id"] in bank_ids


def test_a_bank_leg_never_exists_without_its_settlement_leg():
    """The lifecycle is invoice -> settlement -> bank; a bank credit with no
    settlement would be an impossible state, not realistic noise."""
    for row in generate_dataset(n_records=60, seed=11)["ground_truth"]:
        if row["bank_txn_id"]:
            assert row["settlement_id"], f"{row['invoice_id']} has a bank leg but no settlement leg"


def test_noise_records_have_no_ground_truth_counterpart():
    data = generate_dataset(n_records=30, seed=5, bank_noise_records=8)
    referenced = {row["bank_txn_id"] for row in data["ground_truth"] if row["bank_txn_id"]}
    noise = {b for b in data["bank"]["bank_txn_id"] if b.startswith("BNK-N")}
    assert len(noise) == 8
    assert not (noise & referenced)


def test_dropping_every_leg_still_produces_a_valid_batch():
    data = generate_dataset(n_records=20, seed=4, pending_settlement_rate=1.0)
    assert data["settlements"].empty
    assert all(row["settlement_id"] is None for row in data["ground_truth"])


def test_settled_amount_is_net_of_fee_and_tax():
    data = generate_dataset(n_records=40, seed=6, dispute_rate=0.0)
    for row in data["settlements"].to_dict("records"):
        expected = round(row["gross_amount"] - row["fee"] - row["tax"], 2)
        assert row["settled_amount"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("kwargs", [
    {"n_records": 0},
    {"bank_noise_records": -1},
    {"dispute_rate": 1.5},
    {"in_transit_rate": -0.1},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_dataset(**kwargs)
