"""
Synthetic three-way reconciliation dataset generator.

Simulates the full lifecycle of a payment:
    Invoice (internal) -> Razorpay Settlement (PG) -> Bank Statement credit

and deliberately injects the noise patterns seen in real finance-ops data:
free-text truncation, merchant-ref masking, fee/tax deductions, settlement
lag, dropped legs (pending settlement, in-transit credit), amount disputes,
and unrelated bank noise (salary/rent/other vendors).

Ground truth for every injected record is preserved so the evaluation suite
can compute a *real* accuracy number instead of eyeballing a sample.

All randomness flows through a single seeded ``random.Random`` instance passed
explicitly down the call chain. Nothing here touches the global ``random``
module state: a generator that reseeds the process-wide RNG makes every *other*
component's randomness depend on whether data was generated first, which turns
unrelated tests into order-dependent flakes.
"""
from __future__ import annotations

import random
import string
from datetime import date, timedelta

import pandas as pd

_COMPANIES = [
    "Acme Corp", "Bluewave Logistics", "Crescent Retail", "Delta Textiles",
    "Everest Foods", "Falcon Analytics", "Ganges Pharma", "Harbor Steel",
    "Indus Motors", "Jaipur Handicrafts", "Kestrel Media", "Lotus Apparel",
    "Meridian Consulting", "Nimbus Cloud", "Orion Freight", "Pinnacle Realty",
    "Quartz Energy", "Rangoli Foods", "Summit Traders", "Tundra Robotics",
    "Udaan Airlines", "Vertex Systems", "Wavefront Labs", "Xenon Chemicals",
    "Yashoda Hospitals", "Zenith Furniture",
]

_NOISE_NARRATIONS = [
    "SALARY CREDIT FEB", "RENT PAYMENT HO", "GST REFUND ADJ",
    "INTEREST CREDIT SB", "REVERSAL CHGBACK", "VENDOR PAYOUT MISC",
    "ATM CASH DEPOSIT", "CHEQUE CLEARING", "TDS REFUND AY26",
]

_BANK_PREFIXES = ["NEFT-", "RTGS-", "IMPS-", "UPI-", ""]

_PG_FEE_RATE = 0.02
_GST_ON_FEE_RATE = 0.18

Row = dict


def _rand_id(prefix: str, n: int, digits: int = 6) -> str:
    return f"{prefix}{n:0{digits}d}"


def _rand_alpha_suffix(rng: random.Random, k: int = 6) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=k))


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len].rstrip()


def _merchant_token(company: str, invoice_no: int) -> str:
    """e.g. 'Acme Corp' -> 'ACME012' — how a PG typically mangles the legal name."""
    letters = "".join(ch for ch in company.upper() if ch.isalpha())[:4]
    return f"{letters}{invoice_no:03d}"


def _company_typo(rng: random.Random, name: str, rate: float = 0.35) -> str:
    """Introduce a light OCR/keying-style transposition for invoice free-text noise."""
    if len(name) < 4 or rng.random() > rate:
        return name
    idx = rng.randrange(1, len(name) - 1)
    chars = list(name)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def generate_dataset(
    n_records: int = 60,
    seed: int = 42,
    pending_settlement_rate: float = 0.08,
    in_transit_rate: float = 0.08,
    dispute_rate: float = 0.06,
    bank_noise_records: int = 10,
) -> dict:
    """Build a synthetic three-way batch plus its ground truth.

    Returns a dict with keys 'invoices', 'settlements', 'bank' (all pandas
    DataFrames) and 'ground_truth' (a list of dicts mapping
    invoice_id -> settlement_id -> bank_txn_id, with None for the legs the
    generator deliberately dropped).

    Deterministic for a given ``seed``: the same seed always yields byte-identical
    CSVs, which is what lets the evaluation suite assert on exact numbers.
    """
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")
    if bank_noise_records < 0:
        raise ValueError(f"bank_noise_records must be >= 0, got {bank_noise_records}")
    for name, rate in (
        ("pending_settlement_rate", pending_settlement_rate),
        ("in_transit_rate", in_transit_rate),
        ("dispute_rate", dispute_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {rate}")

    rng = random.Random(seed)

    invoices: list[Row] = []
    settlements: list[Row] = []
    bank: list[Row] = []
    ground_truth: list[Row] = []

    base_date = date(2026, 6, 1)

    for i in range(1, n_records + 1):
        company = rng.choice(_COMPANIES)
        invoice_id = _rand_id("INV-", i, 3)
        invoice_date = base_date + timedelta(days=rng.randint(0, 45))
        amount = round(rng.uniform(5_000, 250_000), 2)

        description = f"{_company_typo(rng, company)} Invoice #{i:03d}"
        invoices.append({
            "invoice_id": invoice_id,
            "customer_name": company,
            "invoice_date": invoice_date.isoformat(),
            "amount": amount,
            "description": description,
        })

        gt_row: Row = {"invoice_id": invoice_id, "settlement_id": None, "bank_txn_id": None}

        # -- Leg 2: Razorpay settlement (may be deliberately dropped: "pending") --
        if rng.random() < pending_settlement_rate:
            ground_truth.append(gt_row)
            continue

        fee = round(amount * _PG_FEE_RATE, 2)
        tax = round(fee * _GST_ON_FEE_RATE, 2)
        settled_amount = round(amount - fee - tax, 2)
        if rng.random() < dispute_rate:
            # short-settlement dispute: PG paid less than invoiced, beyond normal fee tolerance
            settled_amount = round(settled_amount - rng.uniform(50, 500), 2)

        settlement_id = _rand_id("STL-", i, 4)
        order_id = f"order_{_rand_alpha_suffix(rng, 10)}"
        utr = f"UTR{rng.randint(10**11, 10**12 - 1)}"
        settlement_date = invoice_date + timedelta(days=rng.randint(1, 3))
        merchant_ref = f"RAZORP*{_merchant_token(company, i)}"

        settlements.append({
            "settlement_id": settlement_id,
            "order_id": order_id,
            "utr": utr,
            "settlement_date": settlement_date.isoformat(),
            "gross_amount": amount,
            "fee": fee,
            "tax": tax,
            "settled_amount": settled_amount,
            "merchant_ref": merchant_ref,
        })
        gt_row["settlement_id"] = settlement_id

        # -- Leg 3: Bank credit (may be deliberately dropped: "in transit") --
        if rng.random() < in_transit_rate:
            ground_truth.append(gt_row)
            continue

        bank_txn_id = _rand_id("BNK-", i, 5)
        bank_date = settlement_date + timedelta(days=rng.randint(0, 1))
        prefix = rng.choice(_BANK_PREFIXES)
        narration_full = f"{prefix}{merchant_ref}-{_rand_alpha_suffix(rng, 6)}"
        narration = _truncate(narration_full, rng.choice([18, 24, 32, 40]))

        bank.append({
            "bank_txn_id": bank_txn_id,
            "txn_date": bank_date.isoformat(),
            "narration": narration,
            "amount": settled_amount,
            "utr": utr if rng.random() > 0.4 else None,  # UTR often absent in narration-only feeds
        })
        gt_row["bank_txn_id"] = bank_txn_id
        ground_truth.append(gt_row)

    # -- Unrelated bank noise: salary, rent, other vendors with no invoice/settlement leg --
    for j in range(bank_noise_records):
        bank.append({
            "bank_txn_id": _rand_id("BNK-N", j, 4),
            "txn_date": (base_date + timedelta(days=rng.randint(0, 45))).isoformat(),
            "narration": rng.choice(_NOISE_NARRATIONS),
            "amount": round(rng.uniform(2_000, 90_000), 2),
            "utr": None,
        })

    # Shuffled with the same seeded rng — a real bank statement does not arrive
    # sorted in invoice order, and matching must not depend on input ordering.
    rng.shuffle(bank)

    return {
        "invoices": pd.DataFrame(invoices),
        "settlements": pd.DataFrame(settlements),
        "bank": pd.DataFrame(bank),
        "ground_truth": ground_truth,
    }
