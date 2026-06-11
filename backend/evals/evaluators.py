"""
Pure evaluators for the offline eval suite — no I/O, no network, no LLM, no
eval-only deps. Safe to unit-test and to reuse from both the pytest gate and
``run.py``.

Extraction is scored in two parts, by design:

- **Scalar fields** (the enum-valued ``budget_range`` / ``timeline`` /
  ``company_size`` / ``decision_maker`` / ``current_solution`` plus the contact
  fields) are compared by **exact value**. This is where contract #1 lives — the
  ``.value`` strings in ``models/schemas.py`` are simultaneously the scorer keys,
  the Salesforce picklist values, and the Apex parser's inputs, so an extraction
  that returns the wrong string silently produces the wrong Lead. precision/recall/F1.

- **List fields** (``pain_points`` / ``goals`` / ``objections``) are compared by
  **count**, not verbatim text. The model's free-text wording won't match the gold
  strings, but the *cardinality* is the real contract: "enumerate EACH distinct
  problem rather than consolidating" (one multi-impact sentence -> 3 pain points).
"""

from __future__ import annotations

from typing import Any

# Defaults that mean "not captured" — normalised away before scoring so an
# explicit "Unknown" / None from the model counts the same as an omission.
_QUAL_SCALAR_DEFAULTS: dict[str, Any] = {
    "budget_range": "Unknown",
    "timeline": "Unknown",
    "company_size": "Unknown",
    "decision_maker": None,
    "current_solution": None,
}
_LEAD_SCALARS = ("first_name", "last_name", "email", "company", "phone", "title")
_QUAL_SCALARS = ("budget_range", "timeline", "company_size", "decision_maker", "current_solution")
_LIST_FIELDS = ("pain_points", "goals")


def _scalars(result: dict[str, Any]) -> dict[tuple[str, str], Any]:
    """Flatten a result's *meaningful* scalar fields to ``{(section, field): value}``."""
    out: dict[tuple[str, str], Any] = {}

    lead = result.get("lead_data") or {}
    for field in _LEAD_SCALARS:
        value = lead.get(field)
        if value not in (None, ""):
            out[("lead", field)] = value

    qual = result.get("qualification_data") or {}
    for field in _QUAL_SCALARS:
        value = qual.get(field)
        if value is not None and value != _QUAL_SCALAR_DEFAULTS[field]:
            out[("qual", field)] = value

    return out


def _list_counts(result: dict[str, Any]) -> dict[str, int]:
    """Counts for the list-valued fields (``pain_points`` / ``goals`` / ``objections``)."""
    qual = result.get("qualification_data") or {}
    counts = {field: len(qual.get(field) or []) for field in _LIST_FIELDS}
    counts["objections"] = len(result.get("objections") or [])
    return counts


def field_accuracy(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """
    Score one extraction result against its gold ``expected``.

    Returns scalar TP/FP/FN (a wrong value is one FP + one FN) plus per-list-field
    count matches. Aggregate with ``aggregate_field_metrics``.
    """
    pred = _scalars(predicted)
    gold = _scalars(expected)

    tp = sum(1 for key, value in gold.items() if pred.get(key) == value)
    fn = sum(1 for key, value in gold.items() if pred.get(key) != value)
    fp = sum(1 for key, value in pred.items() if gold.get(key) != value)

    pred_counts = _list_counts(predicted)
    gold_counts = _list_counts(expected)
    lists = [
        {"field": field, "pred_count": pred_counts[field], "gold_count": gold_counts[field],
         "match": pred_counts[field] == gold_counts[field]}
        for field in (*_LIST_FIELDS, "objections")
    ]

    return {
        "scalar": {"tp": tp, "fp": fp, "fn": fn},
        "lists": lists,
        "wrong_or_missing": sorted(
            f"{sect}.{field}" for (sect, field), value in gold.items() if pred.get((sect, field)) != value
        ),
        "hallucinated": sorted(
            f"{sect}.{field}" for (sect, field), value in pred.items() if gold.get((sect, field)) != value
        ),
    }


def parse_success(predicted: dict[str, Any], expected: dict[str, Any]) -> bool | None:
    """
    Did extraction return data when it should have?

    Returns ``None`` for rows that legitimately yield nothing (e.g. a bare "yes,
    looks correct"), so the caller can exclude them. Otherwise True/False —
    catching the ``_safe_parse_json -> {}`` silent-drop risk.
    """
    expects_data = bool(_scalars(expected)) or any(_list_counts(expected).values())
    if not expects_data:
        return None
    got_data = bool(_scalars(predicted)) or any(_list_counts(predicted).values())
    return got_data


def aggregate_field_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-row ``field_accuracy`` dicts into precision/recall/F1 + list-match rate."""
    tp = sum(r["scalar"]["tp"] for r in results)
    fp = sum(r["scalar"]["fp"] for r in results)
    fn = sum(r["scalar"]["fn"] for r in results)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    list_matches = [entry["match"] for r in results for entry in r["lists"]]
    list_match_rate = sum(list_matches) / len(list_matches) if list_matches else 1.0

    return {
        "n": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "scalar_precision": round(precision, 4),
        "scalar_recall": round(recall, 4),
        "scalar_f1": round(f1, 4),
        "list_count_match_rate": round(list_match_rate, 4),
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
# Scored against an ``acceptable_next`` *set*, not a single label: discovery vs
# qualification (and qualification vs lead_capture) are legitimately ambiguous, so
# a brittle exact-match would punish correct calls. The first element of each
# case's acceptable set is treated as the "primary" expected stage for the
# confusion matrix.


def routing_match(predicted_stage: str, acceptable: list[str]) -> dict[str, Any]:
    """Did the router pick an acceptable next stage?"""
    return {
        "predicted": predicted_stage,
        "acceptable": list(acceptable),
        "correct": predicted_stage in acceptable,
    }


def routing_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up ``routing_match`` dicts into an accuracy + the list of misses."""
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 1.0,
        "misses": [
            {"predicted": r["predicted"], "acceptable": r["acceptable"]}
            for r in results
            if not r["correct"]
        ],
    }


def confusion_matrix(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """``{primary_expected_stage: {predicted_stage: count}}`` for the report."""
    matrix: dict[str, dict[str, int]] = {}
    for r in results:
        primary = r["acceptable"][0] if r["acceptable"] else "?"
        row = matrix.setdefault(primary, {})
        row[r["predicted"]] = row.get(r["predicted"], 0) + 1
    return matrix
