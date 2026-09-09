"""The learning layer: the concept ladder is well formed, every explanation ends in a retrieval prompt, support tracks
prior knowledge, the spacing rule is the verified declining-share rule, and the six-click tour is complete."""

from __future__ import annotations

import math

import pytest

from qutip_trap_app.viewmodel import learn
from qutip_trap_app.viewmodel.learn import (
    BELL_TOUR,
    CONCEPTS,
    Attempt,
    MasteryLog,
    explain,
    plan_for,
    prerequisites_closure,
    review_gap_days,
    route_matches,
    score_choice,
    score_direction,
    score_histogram_prediction,
)


def test_concept_ladder_is_well_formed() -> None:
    for c in CONCEPTS.values():
        assert c.prompts, f"{c.id}: every explanation is followed by a retrieval prompt"
        assert c.can_do and c.explain.sentence and c.explain.picture and c.explain.equation
        for p in c.prerequisites:
            assert p in CONCEPTS, (c.id, p)
            assert CONCEPTS[p].level <= c.level, f"{c.id} rests on a deeper concept {p}"
            assert c.id not in prerequisites_closure(p), f"cycle between {c.id} and {p}"
        for pr in c.prompts:
            if pr.kind == "choose":
                assert pr.answer in pr.options, pr.id
            elif pr.kind == "free_text":
                assert pr.rubric, pr.id
            elif pr.kind in ("predict_histogram", "predict_direction", "predict_closure"):
                assert pr.checks, pr.id
                assert pr.where, (
                    f"{pr.id}: the learner is told where to look in plain words, never the machine reference"
                )
            elif pr.kind == "locate":
                assert pr.answer and route_matches(pr.answer), pr.id
                assert pr.where, f"{pr.id}: the learner is told where to look in plain words, never the route"
    assert {c.level for c in CONCEPTS.values()} == {0, 1, 2, 3, 4}
    for page, ids in learn.PAGE_CONCEPTS.items():
        assert ids and all(i in CONCEPTS for i in ids), page
        assert page in learn.PAGE_SECTIONS and page in learn.DEVICE_PAGES
    closure = [p for c in CONCEPTS.values() for p in c.prompts if p.kind == "predict_closure"]
    assert len(closure) == 1 and closure[0].options and closure[0].where
    kinds = {c.kind for c in CONCEPTS.values()}
    assert {"fact", "concept", "procedure", "discrimination"} <= kinds


def test_explain_cards_deepen_and_end_in_a_prompt() -> None:
    card = explain("loop_closure", "sentence")
    assert card.deeper == "picture" and card.prompt.id == "loop_closure.q1"
    assert "alpha" not in card.text, "the sentence depth has no symbols"
    eq = explain("loop_closure", "equation")
    assert eq.deeper is None and "alpha_m" in eq.text and eq.section == "4.4.3"


def test_support_tracks_prior_knowledge() -> None:
    novice = plan_for("newcomer", 0)
    expert = plan_for("physicist", 0)
    unknown = plan_for("unknown", 3)
    assert novice.explain_open and novice.worked_example_first and novice.explain_depth == "sentence"
    assert not expert.explain_open and not expert.worked_example_first and expert.explain_depth == "equation"
    assert unknown.explain_open, "when prior knowledge is unknown the app assists (the asymmetric rule)"
    assert novice.prompt_before_reveal and expert.prompt_before_reveal, "retrieval is asked of everyone"
    assert expert.numerics_open and expert.chips_expanded


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
    assert math.isclose(score_direction((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), math.pi)
    assert math.isclose(score_direction((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 0.0)
    assert math.isclose(score_direction((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), math.pi / 2)


def test_six_click_tour_covers_the_ladder() -> None:
    assert len(BELL_TOUR) == 6
    assert [s.index for s in BELL_TOUR] == [1, 2, 3, 4, 5, 6]
    for stop in BELL_TOUR:
        assert stop.concept_id in CONCEPTS
        assert route_matches(
            stop.route.replace("{id}", "j1")
            .replace("{gate}", "ms[2]")
            .replace("{pulse}", "4")
            .replace("{sample}", "0")
        )
    levels = [CONCEPTS[s.concept_id].level for s in BELL_TOUR]
    assert levels == sorted(levels) and levels[0] == 0 and levels[-1] == 4
    assert not route_matches("/device/nowhere") and route_matches("/learn")
    assert learn.DEFAULT_RETENTION_DAYS == 90.0
