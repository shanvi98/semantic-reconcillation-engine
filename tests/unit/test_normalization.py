import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reconciler.utils.normalization import (
    amounts_match,
    clean_text,
    extract_alnum_ids,
    extract_invoice_number,
    extract_trailing_digits,
)


def test_clean_text_strips_punctuation_and_casing():
    assert clean_text("RAZORP*ACME012-XK3F9A") == "RAZORP ACME012 XK3F9A"


def test_extract_alnum_ids_ignores_short_tokens():
    ids = extract_alnum_ids("RAZORP*ACME012")
    assert "ACME012" in ids
    assert "CORP" not in ids  # 4 chars, below the 5-char threshold


def test_extract_alnum_ids_excludes_brand_boilerplate():
    """'RAZORP' is >= 5 chars and would otherwise pass the alnum-token regex,
    but every settlement/bank record in a batch shares it — as a rule-tier
    identifier it would act as a false universal join key, not a
    distinguishing ID. See the noise-token filter in extract_alnum_ids."""
    ids = extract_alnum_ids("RAZORP*ACME012")
    assert "RAZORP" not in ids
    assert "ACME012" in ids


def test_extract_invoice_number_from_free_text():
    assert extract_invoice_number("Acme Corp Invoice #012") == "012"
    assert extract_invoice_number("no invoice reference here") is None


def test_extract_trailing_digits_from_merchant_ref():
    assert extract_trailing_digits("RAZORP*ACME012") == "012"
    assert extract_trailing_digits("NO-DIGITS-HERE") is None


def test_amounts_match_within_tolerance():
    assert amounts_match(1000.0, 1000.4)
    assert not amounts_match(1000.0, 1050.0)
    assert amounts_match(100000.0, 100450.0)  # within relative tolerance
