"""
Judge calibration (PR D2) — validate the LLM judges against human-confirmed labels
before trusting them.

A judge is only worth listening to if it agrees with a human. This module measures
that agreement with **Cohen's kappa** (chance-corrected, unlike raw accuracy) over a
small hand-labeled gold set (``datasets/judge_labels.jsonl``, maintainer-confirmed) and
decides, per judge, whether it clears a documented bar. That is the difference between
"I added a judge" and "I *validated* a judge" — the whole point of this commit.

Two halves, split on the suite's hermeticity line (mirrors ``scorecard.py`` vs the live
sim):

- ``cohen_kappa`` / ``is_validated`` / score normalisation are **pure** → unit-tested
  offline under ``eval_recorded`` (``test_calibration_recorded.py``), no key, no network.
- ``run_calibration`` makes **real** ``claude-sonnet-4-6`` calls → ``eval_live``. It is
  what ``python -m evals.calibration`` runs to *produce* the recorded kappa below.

``CALIBRATED_KAPPA`` is the **recorded** result of that live run — committed, like the
extraction/routing floors travel with their cassette. Re-run ``python -m evals.calibration``
and update it on any change to the judge prompts (``judges.py``), the judge model, or the
labels. A judge counts toward the scorecard (PR D2 commit 2, ``run.py --judges``) **only**
when its recorded kappa clears ``KAPPA_BAR``; below the bar it is reported as advisory.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from typing import Any

# Make ``app`` / ``evals`` importable however this is invoked (mirrors run.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.judges import persona_adherence_judge, summary_faithfulness_judge  # noqa: E402
from evals.loader import load_dataset  # noqa: E402

LABELS_DATASET = "judge_labels.jsonl"

# Landis & Koch (1977): kappa >= 0.61 is "substantial agreement". 0.6 is the documented
# bar a judge must clear to count toward the scorecard; below it the judge is advisory
# only. A deliberately strict, defensible line — not tuned to make a judge pass.
KAPPA_BAR = 0.6

# Recorded by ``python -m evals.calibration`` (judge=claude-sonnet-4-6, 2026-06-15) against
# the 20 maintainer-confirmed labels in datasets/judge_labels.jsonl. Re-run + update on any
# judge-prompt / judge-model / label change (the kappa and the labels are a pair).
#
# - persona_adherence (0.74) CLEARS the 0.6 bar → validated; counts toward the scorecard.
# - summary_faithfulness (0.25) is BELOW the bar → advisory only. The generic RAG
#   groundedness rubric over-flags the asked-for sentiment / conversion-likelihood gloss
#   (TRANSCRIPT_SUMMARY_PROMPT requests it) as "ungrounded" even when no fact is fabricated;
#   it still caught every genuinely hallucinated summary. A task-specific faithfulness
#   prompt is the follow-up — until then this judge does not count toward the scorecard.
CALIBRATED_KAPPA: dict[str, float] = {
    "persona_adherence": 0.7447,
    "summary_faithfulness": 0.25,
}


# ---------------------------------------------------------------------------
# Pure statistics — offline-gated
# ---------------------------------------------------------------------------

def cohen_kappa(human: Sequence[Any], judge: Sequence[Any]) -> float:
    """
    Cohen's kappa between two equal-length sequences of categorical labels.

    ``kappa = (po - pe) / (1 - pe)`` — observed agreement ``po`` corrected for the
    agreement ``pe`` expected by chance given each rater's own class frequencies.
    1.0 = perfect, 0 = chance, negative = worse than chance. Works for any hashable
    label space (the 3-class persona scores and the 2-class faithfulness booleans
    both flow through unchanged). +0 deps on purpose — no sklearn in the eval set.
    """
    if len(human) != len(judge):
        raise ValueError("human and judge label sequences must be the same length")
    n = len(human)
    if n == 0:
        return 0.0
    po = sum(h == j for h, j in zip(human, judge, strict=True)) / n
    classes = set(human) | set(judge)
    pe = sum(
        (sum(h == c for h in human) / n) * (sum(j == c for j in judge) / n)
        for c in classes
    )
    if pe >= 1.0:
        # Both raters collapsed onto a single shared class → po is necessarily 1.0;
        # the chance correction is undefined (0/0). Define as perfect agreement.
        return 1.0
    return (po - pe) / (1.0 - pe)


def is_validated(feedback_key: str) -> bool:
    """True when the judge's **recorded** kappa clears ``KAPPA_BAR`` → it counts."""
    return CALIBRATED_KAPPA.get(feedback_key, 0.0) >= KAPPA_BAR


def _persona_class(score: Any) -> float:
    """Snap a persona judge score to the nearest {0.0, 0.5, 1.0} human-label class."""
    return min((0.0, 0.5, 1.0), key=lambda option: abs(option - float(score)))


def _faithful_class(score: Any) -> bool:
    """Groundedness judge score → boolean (True grounded / False not)."""
    return bool(score)


# ---------------------------------------------------------------------------
# Live calibration run — eval_live (real claude-sonnet-4-6 calls)
# ---------------------------------------------------------------------------

def _score(result: Any) -> Any:
    """Pull the score out of an openevals EvaluatorResult (a dict at runtime)."""
    return result["score"] if isinstance(result, dict) else result.score


async def _grade_persona(evaluator: Any, row: dict[str, Any]) -> float:
    res = await evaluator(inputs=row["transcript"], outputs=row["tars_turns"])
    return _persona_class(_score(res))


async def _grade_faithful(evaluator: Any, row: dict[str, Any]) -> bool:
    res = await evaluator(outputs=row["summary"], context=row["transcript"])
    return _faithful_class(_score(res))


def _summarise(key: str, rows: list[dict], human: list, predicted: list) -> dict[str, Any]:
    kappa = cohen_kappa(human, predicted)
    n = len(human)
    agree = sum(h == p for h, p in zip(human, predicted, strict=True))
    return {
        "feedback_key": key,
        "kappa": round(kappa, 4),
        "n": n,
        "observed_agreement": round(agree / n, 4) if n else 0.0,
        "bar": KAPPA_BAR,
        "validated": kappa >= KAPPA_BAR,
        "disagreements": [
            {"id": rows[i]["id"], "human": human[i], "judge": predicted[i]}
            for i in range(n)
            if human[i] != predicted[i]
        ],
    }


async def run_calibration(judge: Any | None = None) -> dict[str, Any]:
    """
    Grade every labeled example with the real judges and return kappa per dimension.

    LIVE — builds the real ``claude-sonnet-4-6`` judge (or accepts an injected one for a
    cheaper smoke) and grades each row concurrently. Returns, per ``feedback_key``:
    ``{kappa, n, observed_agreement, bar, validated, disagreements:[...]}``. The
    ``disagreements`` make every human/judge split inspectable — that is where a low
    kappa gets explained (and where the labels or the prompt get refined).
    """
    rows = list(load_dataset(LABELS_DATASET))
    persona_rows = [r for r in rows if r["judge"] == "persona_adherence"]
    faith_rows = [r for r in rows if r["judge"] == "summary_faithfulness"]

    persona_ev = persona_adherence_judge(judge)
    faith_ev = summary_faithfulness_judge(judge)

    persona_pred = list(await asyncio.gather(*(_grade_persona(persona_ev, r) for r in persona_rows)))
    faith_pred = list(await asyncio.gather(*(_grade_faithful(faith_ev, r) for r in faith_rows)))

    return {
        "persona_adherence": _summarise(
            "persona_adherence", persona_rows, [r["label"] for r in persona_rows], persona_pred
        ),
        "summary_faithfulness": _summarise(
            "summary_faithfulness", faith_rows, [r["label"] for r in faith_rows], faith_pred
        ),
    }


def _format_report(results: dict[str, Any]) -> str:
    lines = ["judge calibration — judge=claude-sonnet-4-6, labels=judge_labels.jsonl", ""]
    for key, r in results.items():
        verdict = "VALIDATED" if r["validated"] else "BELOW BAR (advisory only)"
        lines.append(
            f"[{key}] kappa={r['kappa']} (bar {r['bar']}) "
            f"agreement={r['observed_agreement']} n={r['n']} -> {verdict}"
        )
        for d in r["disagreements"]:
            lines.append(f"    disagree {d['id']}: human={d['human']} judge={d['judge']}")
    lines.append("")
    lines.append("Update CALIBRATED_KAPPA in calibration.py with these values and commit.")
    return "\n".join(lines)


def main() -> int:
    # Live run: needs ANTHROPIC_API_KEY (from .env). Never upload these judge calls to
    # LangSmith — force tracing off regardless of .env (the labels carry no real PII,
    # but calibration is local-by-design).
    os.environ.setdefault("LLM_PROVIDER", "xai")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    results = asyncio.run(run_calibration())
    print(_format_report(results))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
