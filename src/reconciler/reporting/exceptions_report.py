"""Renders the exception list finance-ops actually needs to act on."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..schemas import ExceptionRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

COLUMNS = [
    "record_id",
    "source",
    "category",
    "reason",
    "amount",
    "best_candidate_id",
    "best_candidate_score",
]


def exceptions_to_dataframe(exceptions: list[ExceptionRecord]) -> pd.DataFrame:
    """Exception list as a DataFrame, sorted highest-value first.

    Review capacity is finite, so the queue is ordered by the money at stake
    rather than by input order — the first row a controller sees should be the
    one worth the most.
    """
    if not exceptions:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame([
        {
            "record_id": e.record_id,
            "source": e.source.value,
            "category": e.category.value,
            "reason": e.reason,
            "amount": e.amount,
            "best_candidate_id": e.best_candidate_id,
            "best_candidate_score": e.best_candidate_score,
        }
        for e in exceptions
    ])
    return df.sort_values("amount", ascending=False, kind="stable").reset_index(drop=True)[COLUMNS]


def write_atomically(path: Path, write: Callable[[Path], None]) -> Path:
    """Write via a temp file in the same directory, then rename into place.

    The dashboard reads these files while runs write them, and a reader that
    catches a half-written file gets a parse error rather than stale data. Rename
    within a directory is atomic on both POSIX and Windows, so a reader sees
    either the previous complete file or the new one, never a partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_exceptions_csv(exceptions: list[ExceptionRecord], path: str | Path) -> Path:
    frame = exceptions_to_dataframe(exceptions)
    return write_atomically(Path(path), lambda target: frame.to_csv(target, index=False))


def summarize(exceptions: list[ExceptionRecord]) -> str:
    """One-paragraph plain-text summary, for CLI output and run logs."""
    if not exceptions:
        return "No exceptions — every record fully reconciled."

    by_category: dict[str, list[ExceptionRecord]] = {}
    for e in exceptions:
        by_category.setdefault(e.category.value, []).append(e)

    lines = [f"{len(exceptions)} exception(s) totalling {sum(e.amount for e in exceptions):,.2f}:"]
    for category, items in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        value = sum(e.amount for e in items)
        lines.append(f"  - {category:<22} {len(items):>3} record(s), {value:>14,.2f}")
    return "\n".join(lines)
