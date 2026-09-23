"""The Hamiltonian page: the terms of H(t) and the collapse operators the engine integrates for one step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import HamiltonianRecord
from qutip_trap_app.viewmodel.shown import Shown

FORMULA = (
    "H/hbar = sum_m omega_m a_m^dag a_m + sum_i (Delta_i/2) sigma_z^i + sum_terms (Omega(t)/2) e^{-i(mu t - phi)} "
    "sigma_+ prod_m D_m(i eta) + h.c. + light shifts"
)

_OPERATORS = {
    "heating_down": "sqrt(Gamma (N + 1)) a_m",
    "heating_up": "sqrt(Gamma N) a_m^dag",
    "motional_dephasing": "sqrt(2/tau) a_m^dag a_m",
    "qubit_dephasing": "sqrt(gamma_phi/2) sigma_z",
    "rayleigh_dephasing": "(1/2) sqrt(Gamma_el) sigma_z",
}
"""The collapse operator of each device channel, by the channel's name."""


@dataclass(frozen=True)
class TermView:
    index: int
    title: str
    operator: str
    tiles: tuple[Shown, ...]
    tones: tuple[tuple[Shown, Shown, Shown], ...]
    """Per tone: (detuning from the carrier, phase at the start, peak Omega/2 pi)."""
    etas: tuple[Shown, ...]
    frozen: tuple[Shown, ...]
    matrix_elements: dict[int, np.ndarray]


@dataclass(frozen=True)
class CollapseView:
    channel: str
    rate: Shown
    ion: int | None
    mode: int | None
    integrated: bool
    note: str
    operator: str


@dataclass(frozen=True)
class HamiltonianView:
    gate_id: str
    header: tuple[Shown, ...]
    free_modes: tuple[tuple[int, str, Shown, Shown], ...]
    """Per mode: (index, class, frequency, the sample's offset of it)."""
    offsets: tuple[Shown, ...]
    """The qubit-frequency offsets and light shifts of the ions."""
    caps: tuple[Shown, ...]
    segments: tuple[tuple[Shown, Shown, str, int, str], ...]
    """Per segment: (duration, fastest frequency, the pulses playing, drive terms, kernel)."""
    terms: tuple[TermView, ...]
    collapse: tuple[CollapseView, ...]
    approximations: tuple[str, ...]


def hamiltonian_view(ham: HamiltonianRecord) -> HamiltonianView:
    resolved = sorted(m for m, c in ham.mode_classes.items() if c == "resolved")
    terms = []
    for k, d in enumerate(ham.drives):
        # a term with no resolved mode (every mode frozen or dropped) is the bare spin operator
        operator = " (x) ".join(
            [f"sigma_+^({d.ion})"] + [f"D_{m}(i {d.etas.get(m, 0.0):+.4f})" for m in resolved]
        )
        crosstalk = d.ion != d.primary_ion
        terms.append(
            TermView(
                index=k,
                title=f"term {k}: {'crosstalk onto' if crosstalk else 'addressed'} ion {d.ion} ({d.pulse})",
                operator=operator,
                tiles=(
                    Shown("Peak Rabi frequency", d.omega_peak_hz, "Hz", f"Omega/2 pi seen by ion {d.ion}"),
                    Shown("Share of the addressed ion's light", abs(d.crosstalk)),
                    Shown("Rabi-frequency scale of this sample", d.rabi_scale),
                    Shown(
                        "Carrier factor",
                        d.carrier_factor,
                        detail="the micromotion correction of the coupling",
                    ),
                    Shown("Debye-Waller factor of the frozen modes", d.debye_waller),
                    Shown("Non-zero operator entries", d.operator_nnz),
                ),
                tones=tuple(
                    (
                        Shown(f"Tone {j} detuning", mu, "Hz"),
                        Shown(f"Tone {j} phase", phase, "rad"),
                        Shown(f"Tone {j} peak", peak, "Hz"),
                    )
                    for j, (mu, phase, peak) in enumerate(d.tones)
                ),
                etas=tuple(
                    Shown(f"eta, mode {m}", e, detail=ham.mode_classes.get(m, "?"))
                    for m, e in sorted(d.etas.items())
                    if e != 0.0
                ),
                frozen=tuple(
                    Shown(
                        f"Frozen mode {m} at n = {d.frozen_n.get(m, 0)}", v, detail="e^{-eta^2/2} L_n(eta^2)"
                    )
                    for m, v in sorted(d.frozen_debye_waller.items())
                ),
                matrix_elements=d.matrix_elements,
            )
        )
    return HamiltonianView(
        gate_id=ham.gate_id,
        header=(
            Shown("Joint dimension", ham.dimension, detail=f"dims {list(ham.dims)}"),
            Shown("Fastest frequency in the frame", ham.omega_max_hz, "Hz"),
            Shown("Drive terms", ham.n_drive_terms, detail="coefficient-bearing terms, conjugates included"),
            Shown("Frame", ham.frame),
            Shown("Kernel", ham.kernel, detail="how the drive operators are applied"),
        ),
        free_modes=tuple(
            (
                m,
                ham.mode_classes.get(m, "?"),
                Shown(f"omega_{m}/2 pi", w, "Hz"),
                Shown(f"Offset of mode {m} in this sample", ham.mode_offsets_hz.get(m, 0.0), "Hz"),
            )
            for m, w in sorted(ham.mode_frequencies_hz.items())
        ),
        offsets=tuple(
            Shown(f"Qubit offset, ion {i}", v, "Hz") for i, v in sorted(ham.qubit_offsets_hz.items())
        )
        + tuple(Shown(f"Light shift, ion {i}", v, "Hz") for i, v in sorted(ham.stark_shifts_hz.items())),
        caps=tuple(Shown(f"Cap, mode {m}", d) for m, d in sorted(ham.caps.items())),
        segments=tuple(
            (
                Shown(
                    "Duration",
                    s.t_end_s - s.t_start_s,
                    "s",
                    f"{s.t_start_s * 1e6:.2f} to {s.t_end_s * 1e6:.2f} us",
                ),
                Shown("Fastest frequency", s.omega_max_hz, "Hz"),
                ", ".join(s.pulses),
                s.n_drive_terms,
                s.kernel,
            )
            for s in ham.segments
        ),
        terms=tuple(terms),
        collapse=tuple(
            CollapseView(
                channel=c.channel,
                rate=Shown("Rate", c.rate_hz, "1/s"),
                ion=c.ion,
                mode=c.mode,
                integrated=c.integrated,
                note=c.note,
                operator=_OPERATORS.get(c.channel.split("[")[0], c.channel),
            )
            for c in ham.collapse
        ),
        approximations=ham.approximations,
    )
