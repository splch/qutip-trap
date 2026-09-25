"""The learning layer: every explanation ends in a retrieval prompt, support tracks prior knowledge, the spacing rule is the
declining-share rule, and the six-click tour climbs the ladder."""

from __future__ import annotations

import math

import pytest

from qutip_trap_app.viewmodel.learn import (
    BELL_TOUR,
    CONCEPTS,
    FREE_EXERCISE,
    Attempt,
    MasteryLog,
    explain,
    level_concepts,
    plan_for,
    review_gap_days,
    score_choice,
    score_histogram_prediction,
    sketch_distribution,
)
from qutip_trap_app.views.level4 import DEVICE_PAGES


def test_concepts_are_well_formed() -> None:
    for c in CONCEPTS.values():
        assert c.prompts, f"{c.id}: every explanation is followed by a retrieval prompt"
        assert c.can_do and c.explain.sentence and c.explain.picture and c.explain.equation
        for pr in c.prompts:
            if pr.kind == "choose":
                assert pr.answer in pr.options, pr.id
            elif pr.kind == "free_text":
                assert pr.rubric, pr.id
            else:
                assert pr.where, f"{pr.id}: the learner is told where to look in plain words, never the route"
            if pr.kind == "locate":
                assert pr.answer is not None and pr.answer.removeprefix("/device/") in DEVICE_PAGES, pr.id
    assert {c.level for c in CONCEPTS.values()} == {0, 1, 2, 3, 4}
    assert all(level_concepts(level) for level in range(4)), "every level's drawer has its concepts"
    for name, page in DEVICE_PAGES.items():
        assert page.concepts and all(i in CONCEPTS for i in page.concepts), name
    closure = [p for c in CONCEPTS.values() for p in c.prompts if p.kind == "predict_closure"]
    assert len(closure) == 1 and closure[0].options and closure[0].where


def test_explain_cards_deepen_and_end_in_a_prompt() -> None:
    card = explain("loop_closure", "sentence")
    assert card.deeper == "picture" and card.prompt.id == "loop_closure.q1"
    assert "alpha" not in card.text, "the sentence depth has no symbols"
    eq = explain("loop_closure", "equation")
    assert eq.deeper is None and "alpha_m" in eq.text and eq.concept.section == "4.4.3"


def test_support_tracks_prior_knowledge() -> None:
    novice = plan_for("newcomer", 0)
    expert = plan_for("physicist", 0)
    assert novice.explain_open and novice.explain_depth == "sentence" and novice.plain_labels_first
    assert not novice.expanded, "a newcomer's tables start closed"
    assert not expert.explain_open and expert.explain_depth == "equation" and not expert.plain_labels_first
    assert expert.expanded
    assert plan_for("unknown", 3) == plan_for("newcomer", 3), (
        "when prior knowledge is unknown the app assists"
    )
    circuits = [plan_for("circuits", level) for level in range(5)]
    assert [p.explain_open for p in circuits] == [False, False, True, True, True], (
        "the hardware levels explained"
    )
    assert [p.expanded for p in circuits] == [False, False, False, True, True]


@pytest.mark.parametrize(
    ("days", "lo", "hi"),
    [
        (7.0, 0.20 * 7.0, 0.40 * 7.0),
        (365.0, 0.05 * 365.0, 0.10 * 365.0),
        (1.0, 0.2, 0.4),
        (3650.0, 0.05 * 3650.0, 0.10 * 3650.0),
    ],
)
def test_spacing_rule_anchors(days: float, lo: float, hi: float) -> None:
    a, b = review_gap_days(days)
    assert math.isclose(a, lo) and math.isclose(b, hi)


def test_spacing_share_declines_between_the_anchors() -> None:
    a90, b90 = review_gap_days(90.0)
    assert 0.05 * 90 < a90 < 0.20 * 90 and 0.10 * 90 < b90 < 0.40 * 90
    shares = [review_gap_days(d)[0] / d for d in (7.0, 30.0, 90.0, 180.0, 365.0)]
    assert all(x > y for x, y in zip(shares, shares[1:])), "the optimal share falls as the target grows"


def test_mastery_log_separates_session_from_delayed_accuracy() -> None:
    log = MasteryLog(retention_days=90.0)
    log.record(Attempt("histogram", "histogram.q1", 0.0, True, unaided=False))
    log.record(Attempt("histogram", "histogram.q1", 0.1, True, unaided=True))
    assert log.session_accuracy("histogram", now_days=0.2) == 1.0
    assert log.delayed_unaided_accuracy("histogram") is None, "ten minutes later is not a delayed test"
    gap_lo, gap_hi = review_gap_days(90.0)
    assert log.due(now_days=0.5) == ()
    assert "histogram" in log.due(now_days=gap_hi + 0.2)
    log.record(Attempt("histogram", "histogram.q1", gap_hi + 0.2, False, unaided=True))
    assert log.delayed_unaided_accuracy("histogram") == 0.0
    assert log.due(now_days=gap_hi + 0.3) == (), "the latest exposure restarts the gap"
    assert gap_lo < gap_hi


def test_scoring_is_against_the_record_and_about_the_task() -> None:
    sim = {"00": 0.505, "11": 0.495}
    bars = {"00": 0.035, "11": 0.035}
    good = score_histogram_prediction({"00": 1.0, "11": 1.0}, sim, bars)
    assert good.within_error_bars and good.total_variation < 0.01 and "you" not in good.feedback.lower()
    bad = score_histogram_prediction({"00": 1.0}, sim, bars)
    assert not bad.within_error_bars and bad.bars_off == ("00", "11") and bad.total_variation > 0.4
    prompt = CONCEPTS["native_gate"].prompts[0]
    assert score_choice(prompt, "1") and not score_choice(prompt, "2")
    assert sketch_distribution("uniform", sim, 2) == {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    assert sketch_distribution("zero", sim, 2) == {"00": 1.0} and sketch_distribution("ideal", sim, 2) == sim


def test_tour_and_the_free_exercise() -> None:
    assert len(BELL_TOUR) == 6 and all(stop.concept_id in CONCEPTS for stop in BELL_TOUR)
    levels = [CONCEPTS[s.concept_id].level for s in BELL_TOUR]
    assert levels == sorted(levels) and levels[0] == 0 and levels[-1] == 4, "the tour climbs the ladder"
    assert len(FREE_EXERCISE) == 5 and "predict" in FREE_EXERCISE[1].lower()
