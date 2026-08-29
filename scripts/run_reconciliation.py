#!/usr/bin/env python
"""Run the full three-way reconciliation over data/raw/ and save metrics.

Thin shim over reconciler.cli — the logic lives in the package so it can be
imported and tested. Installed as a console script too: see [project.scripts].
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reconciler.cli import run_reconciliation

if __name__ == "__main__":
    raise SystemExit(run_reconciliation())
