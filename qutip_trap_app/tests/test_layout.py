"""The layout follows the window (DESIGN.md Section 4, R12): a compact window gives the content its whole width, the rail
and the drawer taking none; the drawer docks under content it would crowd; a card's room is the column it is in."""

from __future__ import annotations

import dataclasses

from qutip_trap_app.views import theme
from qutip_trap_app.views.common import TWO_COLUMN_MIN_WIDTH, card_room, compact, content_width, drawer_docked
from qutip_trap_app.views.state import Store


class _Window:
    """What the layout reads of a page: its width."""

    def __init__(self, width: float) -> None:
        self.width = width


def _store(*, drawer: bool) -> Store:
    store = Store()
    store.learner = dataclasses.replace(store.learner, explain_open=drawer)
    return store


def test_a_compact_window_gives_the_content_its_whole_width() -> None:
    phone, laptop = _Window(390.0), _Window(1280.0)
    assert compact(phone) and not compact(laptop)
    assert content_width(_store(drawer=True), phone) == 390.0 - 2 * theme.PAGE_PADDING_COMPACT, (
        "the drawer is docked under the content and the rail is a bar: neither takes width"
    )
    beside = theme.RAIL_WIDTH + theme.DRAWER_WIDTH
    assert content_width(_store(drawer=True), laptop) == 1280.0 - beside - 2 * theme.PAGE_PADDING
    assert content_width(_store(drawer=False), laptop) == 1280.0 - theme.RAIL_WIDTH - 2 * theme.PAGE_PADDING


def test_the_drawer_docks_under_the_content_it_would_crowd() -> None:
    """Beside an 800 px window's content the drawer would leave it under 300 px: it docks under the content, which keeps
    the width, and a stacked card has the whole column."""
    tight, wide = _Window(800.0), _Window(1280.0)
    assert drawer_docked(tight) and not drawer_docked(wide) and drawer_docked(_Window(390.0))
    store = _store(drawer=True)
    content = content_width(store, tight)
    assert content == 800.0 - theme.RAIL_WIDTH - 2 * theme.PAGE_PADDING
    assert content < TWO_COLUMN_MIN_WIDTH, "the stacks go one above the other"
    assert card_room(store, tight, 0, share=0.5) == content - 2 * theme.CARD_PADDING, (
        "a stacked card has the column"
    )
