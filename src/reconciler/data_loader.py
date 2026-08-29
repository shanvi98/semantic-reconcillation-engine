"""CSV -> typed record loaders for the three sources.

Loading is where a bad input file should fail, loudly and specifically. A CSV
with a renamed column or a non-ISO date is the single most common way a
reconciliation run goes wrong in practice, and the default pandas failure for
both is an opaque ``AttributeError``/``ValueError`` many frames away from the
actual cause. Every loader here validates its columns up front and reports the
file, the row, and the offending value.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .schemas import BankRecord, InvoiceRecord, SettlementRecord

INVOICE_COLUMNS = ("invoice_id", "customer_name", "invoice_date", "amount", "description")
SETTLEMENT_COLUMNS = (
    "settlement_id", "order_id", "utr", "settlement_date",
    "gross_amount", "fee", "tax", "settled_amount", "merchant_ref",
)
BANK_COLUMNS = ("bank_txn_id", "txn_date", "narration", "amount", "utr")

# Accepted beyond ISO-8601 — the formats bank/PG exports actually ship with.
_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y")


class DataLoadError(ValueError):
    """Raised when a source CSV is missing columns or holds unparseable values."""


def _read_csv(path: str | Path, required: tuple[str, ...]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"{path} does not exist — run scripts/generate_data.py first")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"{path} is empty") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"{path} is not valid CSV: {exc}") from exc

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataLoadError(
            f"{path} is missing required column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )
    return df


def _to_date(value, *, path: Path | str, row: int, column: str) -> date:
    """Parse a date, accepting the handful of formats real exports use."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise DataLoadError(f"{path} row {row}: {column} is empty")

    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise DataLoadError(
        f"{path} row {row}: {column}={text!r} is not a recognised date. "
        f"Expected one of {list(_DATE_FORMATS)}"
    )


def _to_float(value, *, path: Path | str, row: int, column: str) -> float:
    """Parse an amount, tolerating thousands separators and currency symbols."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise DataLoadError(f"{path} row {row}: {column} is empty")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    for symbol in ("₹", "INR", "Rs.", "Rs", "$"):
        text = text.replace(symbol, "")
    try:
        return float(text.strip())
    except ValueError as exc:
        raise DataLoadError(f"{path} row {row}: {column}={value!r} is not a number") from exc


def _to_str(value, *, path: Path | str, row: int, column: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise DataLoadError(f"{path} row {row}: {column} is empty")
    text = str(value).strip()
    if not text:
        raise DataLoadError(f"{path} row {row}: {column} is empty")
    return text


def _opt_str(value) -> str | None:
    """Optional free-text field. A blank UTR is normal, not an error: many bank
    feeds are narration-only and carry no UTR at all."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def load_invoices(path: str | Path) -> list[InvoiceRecord]:
    df = _read_csv(path, INVOICE_COLUMNS)
    return [
        InvoiceRecord(
            invoice_id=_to_str(row["invoice_id"], path=path, row=i, column="invoice_id"),
            customer_name=_to_str(row["customer_name"], path=path, row=i, column="customer_name"),
            invoice_date=_to_date(row["invoice_date"], path=path, row=i, column="invoice_date"),
            amount=_to_float(row["amount"], path=path, row=i, column="amount"),
            description=_opt_str(row["description"]) or "",
        )
        for i, row in enumerate(df.to_dict("records"), start=2)  # start=2: row 1 is the header
    ]


def load_settlements(path: str | Path) -> list[SettlementRecord]:
    df = _read_csv(path, SETTLEMENT_COLUMNS)
    return [
        SettlementRecord(
            settlement_id=_to_str(row["settlement_id"], path=path, row=i, column="settlement_id"),
            order_id=_opt_str(row["order_id"]) or "",
            utr=_opt_str(row["utr"]),
            settlement_date=_to_date(row["settlement_date"], path=path, row=i, column="settlement_date"),
            gross_amount=_to_float(row["gross_amount"], path=path, row=i, column="gross_amount"),
            fee=_to_float(row["fee"], path=path, row=i, column="fee"),
            tax=_to_float(row["tax"], path=path, row=i, column="tax"),
            settled_amount=_to_float(row["settled_amount"], path=path, row=i, column="settled_amount"),
            merchant_ref=_opt_str(row["merchant_ref"]) or "",
        )
        for i, row in enumerate(df.to_dict("records"), start=2)
    ]


def load_bank(path: str | Path) -> list[BankRecord]:
    df = _read_csv(path, BANK_COLUMNS)
    return [
        BankRecord(
            bank_txn_id=_to_str(row["bank_txn_id"], path=path, row=i, column="bank_txn_id"),
            txn_date=_to_date(row["txn_date"], path=path, row=i, column="txn_date"),
            narration=_opt_str(row["narration"]) or "",
            amount=_to_float(row["amount"], path=path, row=i, column="amount"),
            utr=_opt_str(row["utr"]),
        )
        for i, row in enumerate(df.to_dict("records"), start=2)
    ]
