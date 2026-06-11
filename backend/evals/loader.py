"""Dataset loading for the eval suite (JSONL, one case per line)."""

from __future__ import annotations

import json
from typing import Any

from evals._config import DATASET_DIR


def load_dataset(filename: str) -> list[dict[str, Any]]:
    """
    Read a ``datasets/*.jsonl`` file into a list of case dicts.

    Blank lines are skipped; every other line must be a JSON object. The schema
    is dimension-specific (see the dataset files / README).
    """
    path = DATASET_DIR / filename
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - dataset author aid
                raise ValueError(f"{filename}:{lineno}: invalid JSON: {exc}") from exc
    return rows
