"""The paths of the zoom ladder (PLAN.md Section 14.6), built in one place. Every segment is percent-encoded, so that an id
with a bracket or a slash (``ms[2]``, the ZZ wrapper's ``zz[3]/ms``) arrives whole in ``shell.parse_route``, which decodes
each segment; a browser delivers a typed or reloaded URL encoded the same way."""

from __future__ import annotations

from urllib.parse import quote


def path_of(*segments: object) -> str:
    """``/a/b/c`` from the segments, each percent-encoded (a slash inside a segment included)."""
    return "/" + "/".join(quote(str(s), safe="") for s in segments)


def job(key: str) -> str:
    """Level 0 for the record ``key``."""
    return path_of("job", key)


def circuit(key: str, gate_id: str) -> str:
    """Level 1 at the gate piece ``gate_id``."""
    return path_of("job", key, "circuit", gate_id)


def schedule(key: str, pulse: int | str) -> str:
    """Level 2 at the pulse with index ``pulse``."""
    return path_of("job", key, "schedule", pulse)


def dynamics(key: str, pulse: int | str, sample: int | str = 0) -> str:
    """Level 3 inside the pulse ``pulse`` for the dynamical sample ``sample``."""
    return path_of("job", key, "dynamics", pulse, sample)


def device(page: str) -> str:
    """The Level 4 physics page ``page``."""
    return path_of("device", page)


def learn(tab: str | None = None) -> str:
    """The Learn view, at the activity ``tab`` when given."""
    return path_of("learn") if tab is None else path_of("learn", tab)


def preset(preset_id: str) -> str:
    """One published-experiment preset in Learn."""
    return path_of("learn", "preset", preset_id)
