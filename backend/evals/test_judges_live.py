"""
Live judge + calibration tests (``judges.py`` / ``calibration.py``) — eval_live.

Hits the real ``claude-sonnet-4-6`` judge (needs ``ANTHROPIC_API_KEY``; spends tokens),
so it is manual/nightly, never the per-commit gate. Asserts **structure and score
domains**, not exact verdicts — a judge is a model, and its grade on a borderline case is
not bit-stable. The pure kappa math + the validation gate are covered offline in
``test_calibration_recorded.py``; the *recorded* kappa is produced by
``python -m evals.calibration``.

    pytest backend/evals -m eval_live      # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import pytest

from evals.calibration import KAPPA_BAR, run_calibration
from evals.judges import persona_adherence_judge, summary_faithfulness_judge

pytestmark = pytest.mark.eval_live


def _score(result):
    return result["score"] if isinstance(result, dict) else result.score


@pytest.mark.asyncio
async def test_persona_judge_scores_in_choice_set():
    ev = persona_adherence_judge()  # real claude-sonnet-4-6
    res = await ev(
        inputs="Visitor: hi\nTARS: Hi, I'm TARS, here because the humans wanted weekends. What's broken?",
        outputs="Hi, I'm TARS, here because the humans wanted weekends. What's broken?",
    )
    assert float(_score(res)) in (0.0, 0.5, 1.0)


@pytest.mark.asyncio
async def test_faithfulness_judge_returns_boolean():
    ev = summary_faithfulness_judge()
    res = await ev(
        outputs="The visitor mentioned a $100K budget.",
        context="Visitor: our budget is approved north of 100K.",
    )
    assert bool(_score(res)) is True


@pytest.mark.asyncio
async def test_calibration_runs_and_reports_kappa():
    results = await run_calibration()
    for key in ("persona_adherence", "summary_faithfulness"):
        r = results[key]
        assert r["n"] > 0
        assert -1.0 <= r["kappa"] <= 1.0
        assert r["bar"] == KAPPA_BAR
        assert isinstance(r["validated"], bool)
