#!/usr/bin/env python3
"""Phase 1 entry point: freeze Dataset.ET into data/ + metadata/.

Usage::

    python prepare_dataset.py
    python prepare_dataset.py --metadata-only
"""

from __future__ import annotations

from data.prepare import main

if __name__ == "__main__":
    raise SystemExit(main())
