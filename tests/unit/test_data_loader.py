"""Loading is where a bad input file should fail — loudly, and pointing at the
row and column that caused it. A reconciliation that dies twenty frames deep in
a matcher because a date column was formatted differently is the single most
expensive kind of failure to diagnose in an ops setting.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from reconciler.data_loader import (
    DataLoadError,
    load_bank,
    load_invoices,
    load_settlements,
)

INVOICE_CSV = (
    "invoice_id,customer_name,invoice_date,amount,description\n"
    "INV-001,Acme Corp,2026-06-01,1000.50,Acme Corp Invoice #001\n"
)
BANK_CSV = (
    "bank_txn_id,txn_date,narration,amount,utr\n"
    "BNK-001,2026-06-03,NEFT-RAZORP*ACME001,980.25,UTR123456789012\n"
    "BNK-002,2026-06-04,SALARY CREDIT,5000.00,\n"
)
SETTLEMENT_CSV = (
    "settlement_id,order_id,utr,settlement_date,gross_amount,fee,tax,settled_amount,merchant_ref\n"
    "STL-0001,order_ABC123,UTR123456789012,2026-06-02,1000.50,20.01,3.60,976.89,RAZORP*ACME001\n"
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_a_well_formed_invoice_file(tmp_path):
    records = load_invoices(_write(tmp_path, "invoices.csv", INVOICE_CSV))
    assert len(records) == 1
    assert records[0].invoice_id == "INV-001"
    assert records[0].invoice_date == date(2026, 6, 1)
    assert records[0].amount == pytest.approx(1000.50)


def test_loads_settlements_and_bank(tmp_path):
    settlements = load_settlements(_write(tmp_path, "s.csv", SETTLEMENT_CSV))
    bank = load_bank(_write(tmp_path, "b.csv", BANK_CSV))
    assert settlements[0].settled_amount == pytest.approx(976.89)
    assert len(bank) == 2


def test_blank_utr_is_none_not_an_error(tmp_path):
    """Narration-only bank feeds carry no UTR at all — that is normal input,
    not a malformed file."""
    bank = load_bank(_write(tmp_path, "b.csv", BANK_CSV))
    assert bank[0].utr == "UTR123456789012"
    assert bank[1].utr is None


def test_missing_file_names_the_path_and_the_fix(tmp_path):
    with pytest.raises(DataLoadError) as exc:
        load_invoices(tmp_path / "nope.csv")
    assert "does not exist" in str(exc.value)
    assert "generate_data" in str(exc.value)


def test_missing_column_is_reported_by_name(tmp_path):
    broken = INVOICE_CSV.replace("amount", "value")
    with pytest.raises(DataLoadError) as exc:
        load_invoices(_write(tmp_path, "invoices.csv", broken))
    message = str(exc.value)
    assert "amount" in message and "value" in message


def test_unparseable_date_names_the_row_and_value(tmp_path):
    broken = INVOICE_CSV.replace("2026-06-01", "not-a-date")
    with pytest.raises(DataLoadError) as exc:
        load_invoices(_write(tmp_path, "invoices.csv", broken))
    message = str(exc.value)
    assert "row 2" in message and "not-a-date" in message


@pytest.mark.parametrize("raw,expected", [
    ("01-06-2026", date(2026, 6, 1)),
    ("01/06/2026", date(2026, 6, 1)),
    ("2026/06/01", date(2026, 6, 1)),
    ("01-Jun-2026", date(2026, 6, 1)),
])
def test_accepts_the_date_formats_real_exports_ship_with(tmp_path, raw, expected):
    csv = INVOICE_CSV.replace("2026-06-01", raw)
    assert load_invoices(_write(tmp_path, "invoices.csv", csv))[0].invoice_date == expected


def test_amounts_tolerate_separators_and_currency_symbols(tmp_path):
    csv = INVOICE_CSV.replace("1000.50", '"1,000.50"')
    assert load_invoices(_write(tmp_path, "invoices.csv", csv))[0].amount == pytest.approx(1000.50)


def test_non_numeric_amount_names_the_row(tmp_path):
    csv = INVOICE_CSV.replace("1000.50", "abc")
    with pytest.raises(DataLoadError) as exc:
        load_invoices(_write(tmp_path, "invoices.csv", csv))
    assert "row 2" in str(exc.value) and "amount" in str(exc.value)


def test_empty_file_is_reported_as_empty(tmp_path):
    with pytest.raises(DataLoadError) as exc:
        load_invoices(_write(tmp_path, "invoices.csv", ""))
    assert "empty" in str(exc.value)


def test_blank_required_id_is_rejected(tmp_path):
    csv = INVOICE_CSV.replace("INV-001,", ",")
    with pytest.raises(DataLoadError) as exc:
        load_invoices(_write(tmp_path, "invoices.csv", csv))
    assert "invoice_id" in str(exc.value)
