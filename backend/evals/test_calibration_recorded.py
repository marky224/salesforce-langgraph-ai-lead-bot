"""
Offline unit tests for judge calibration (``calibration.py``) — eval_recorded.

Pure: Cohen's kappa math, the validation gate, score normalisation, and the gold-label
dataset's schema. No key, no network, no model call — the live calibration run that
*produces* ``CALIBRATED_KAPPA`` is eval_live (``test_judges_live.py``). Mirrors how
``test_scorecard.py`` keeps the deterministic slice on the per-commit gate.
"""

from __future__ import annotations

import math

import pytest

from evals import calibration
from evals.calibration import KAPPA_BAR, cohen_kappa, is_validated
from evals.loader import load_dataset

pytestmark = pytest.mark.eval_recorded


# --- cohen_kappa ---

def test_kappa_perfect_agreement():
    assert cohen_kappa([1.0, 0.5, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0]) == 1.0


def test_kappa_degenerate_single_shared_class_is_perfect():
    # Both raters say True for everything → chance correction is 0/0; define as 1.0.
    assert cohen_kappa([True, True, True], [True, True, True]) == 1.0


def test_kappa_known_value():
    # human=[T,T,F,F] judge=[T,F,F,F]: po=3/4=.75; pe = .5*.25 + .5*.75 = .5;
    # kappa = (.75 - .5) / (1 - .5) = 0.5
    assert math.isclose(cohen_kappa([True, True, False, False], [True, False, False, False]), 0.5)


def test_kappa_worse_than_chance_is_negative():
    assert cohen_kappa([True, False, True, False], [False, True, False, True]) < 0


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


def test_kappa_empty_is_zero():
    assert cohen_kappa([], []) == 0.0


# --- validation gate ---

def test_is_validated_threshold(monkeypatch):
    monkeypatch.setattr(
        calibration, "CALIBRATED_KAPPA", {"clears": KAPPA_BAR, "below": KAPPA_BAR - 0.01}
    )
    assert is_validated("clears")           # exactly at the bar counts
    assert not is_validated("below")
    assert not is_validated("unknown_key")  # missing → not validated


def test_recorded_kappa_in_range():
    # Whatever the committed calibration result is, it must be a valid kappa.
    for value in calibration.CALIBRATED_KAPPA.values():
        assert -1.0 <= value <= 1.0


# --- score normalisation ---

def test_persona_class_snaps_to_nearest_choice():
    assert calibration._persona_class(0.0) == 0.0
    assert calibration._persona_class(0.5) == 0.5
    assert calibration._persona_class(1.0) == 1.0
    assert calibration._persona_class(0.4) == 0.5  # tolerate a near-miss from the judge


def test_faithful_class_is_boolean():
    assert calibration._faithful_class(True) is True
    assert calibration._faithful_class(False) is False
    assert calibration._faithful_class(1.0) is True
    assert calibration._faithful_class(0.0) is False


# --- gold-label dataset schema (locks the calibration set on the gate) ---

def test_judge_labels_dataset_wellformed():
    rows = list(load_dataset("judge_labels.jsonl"))
    assert len(rows) == 20
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["transcript"]
        if r["judge"] == "persona_adherence":
            assert r["label"] in (0.0, 0.5, 1.0)
            assert r["tars_turns"]
        elif r["judge"] == "summary_faithfulness":
            assert isinstance(r["label"], bool)
            assert r["summary"]
        else:
            pytest.fail(f"unknown judge {r['judge']!r} in {r['id']}")
