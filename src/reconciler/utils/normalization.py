"""Text and amount normalization shared by every matcher tier."""
from __future__ import annotations

import re

_ALNUM_ID = re.compile(r"[A-Z0-9]{5,}")
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]+")
_MULTI_SPACE = re.compile(r"\s+")

_NOISE_TOKENS = {
    "NEFT", "RTGS", "IMPS", "UPI", "TXN", "REF", "PAYMENT", "PAY",
    "SETTLEMENT", "RAZORP", "RAZORPAY", "INV", "INVOICE", "NO", "NUMBER",
}


def clean_text(raw: str) -> str:
    """Uppercase, strip punctuation noise, collapse whitespace."""
    text = raw.upper()
    text = _NON_ALNUM.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def strip_noise_tokens(text: str) -> str:
    tokens = [t for t in text.split() if t not in _NOISE_TOKENS]
    return " ".join(tokens)


def extract_alnum_ids(raw: str) -> set[str]:
    """Pull candidate order/invoice/UTR-like identifiers out of free text.

    Excludes _NOISE_TOKENS (rail codes, brand boilerplate like "RAZORP") even
    though some are >=5 chars and would otherwise pass the alnum-token regex:
    a token every record in the batch shares (e.g. the PG's own brand prefix)
    is not a distinguishing identifier and must never drive a tier-1 "exact
    ID hit" — it would turn into a false universal join key across records.
    """
    text = clean_text(raw)
    return {tok for tok in _ALNUM_ID.findall(text) if tok not in _NOISE_TOKENS}


def extract_invoice_number(raw: str) -> str | None:
    """Best-effort zero-padded invoice number, e.g. 'Acme Corp Invoice #012' -> '012'."""
    match = re.search(r"(?:INV(?:OICE)?\s*#?\s*)(\d{2,})", raw.upper())
    return match.group(1) if match else None


def extract_trailing_digits(raw: str, min_len: int = 2, max_len: int = 4) -> str | None:
    """Pull the trailing digit run off a merchant-ref-style token, e.g.
    'RAZORP*ACME012' -> '012'. This is how a PG typically embeds an invoice
    number inside a mangled/abbreviated merchant identifier."""
    text = clean_text(raw).replace(" ", "")
    match = re.search(rf"(\d{{{min_len},{max_len}}})$", text)
    return match.group(1) if match else None


def amounts_match(a: float, b: float, *, abs_tol: float = 1.0, rel_tol: float = 0.005) -> bool:
    """Tolerant amount comparison to absorb rounding / fee timing noise."""
    diff = abs(a - b)
    return diff <= abs_tol or diff <= rel_tol * max(abs(a), abs(b), 1e-9)
