"""The absolute SIGN of the excess-micromotion amplitude and modulation index (PLAN.md Section 4.1.1; Section 13 row
"Floquet function and rf phase origin"; Section 9.17 row "Mathieu sign of the Floquet function"; audit items E.2, E.3).

The adopted Mathieu origin is d^2x/dxi^2 + [a - 2q cos 2xi] x = 0, under which u(t) ~ e^{i nu t}[1 - (q/2) cos omega_rf t]
and "the in-phase micromotion at the rf phase origin is a CONTRACTION, x_mu(t) = -(q_x/2) x_sec(t) cos(omega_rf t), and
the phase of every micromotion sideband and the sign with which a dc shim adds to Berkeland's irreducible phi_ac term
follow from that one statement". The signed amplitude is therefore u_1 = -(1/2) Q u_0, and the regression the plan names
is that the modulation index changes sign - a phase step of pi - as a shim crosses the compensated value.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.special import jv

from qutip_trap.api import HilbertSpace, ModeTruncation, Pulse, RfDrive
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian, micromotion_index
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.run.job import detection_micromotion
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.micromotion import MicromotionIndex, excess_amplitude_m, modulation_index
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import DcElectrodes
from qutip_trap.units import TWO_PI
from tests.m2_fixtures import single_ion_raman_device

KX = 1  # the 3 MHz x mode of the single-ion fixture


@dataclasses.dataclass(frozen=True)
class _Mass:
    """The only ``Species`` field ``Trap.mathieu`` and ``micromotion_amplitude_m`` read."""

    mass_u: float


def _yb_trap(e_x: float) -> Trap:
    """Berkeland's 9.12 example: 171Yb+ at 1 MHz radially, Omega_rf/2pi = 10.1 MHz, so q_x = +0.2785."""
    return Trap(
        omega_hz=(1.0e6, 1.0e6, 0.2e6),
        axis_angle_rad=0.0,
        rf=RfDrive(0.0, 10.1e6),
        dc=None,
        geometry=None,
        stray_field_v_per_m=(e_x, 0.0, 0.0),
        shim_voltages_v={},
    )


def test_in_phase_amplitude_is_minus_half_q_u0_and_keeps_the_radial_antiphase() -> None:
    """sign(u_1 . x_hat) = -sign(q_x E_x); q_y = -q_x leaves the two radial micromotions in antiphase."""
    yb = _Mass(170.93578162009095)
    params = _yb_trap(1.0).mathieu(yb)
    q_x = float(params.q[0, 0])
    assert q_x > 0.0, "the explicit path's inversion gives q_x = +2 sqrt 2 omega_x/Omega"
    amp = _yb_trap(1.0).micromotion_amplitude_m(yb)
    assert amp[0] < 0.0, "the in-phase micromotion at the rf phase origin is a contraction (Section 4.1.1)"
    assert math.copysign(1.0, amp[0]) == -math.copysign(1.0, q_x * 1.0)
    assert _yb_trap(-1.0).micromotion_amplitude_m(yb)[0] == pytest.approx(-amp[0], rel=1e-12)
    # the two Appendix-E routes to u_1 agree, including the sign and the x/y antiphase
    from qutip_trap.trap.micromotion import displacement_m
    from qutip_trap.units import ATOMIC_MASS_KG

    mass = yb.mass_u * ATOMIC_MASS_KG
    spring = params.pseudopotential_spring()
    u0_x = float(np.linalg.solve(mass * spring, np.array([1.0, 0.0, 0.0]) * 1.602176634e-19)[0])
    assert excess_amplitude_m(u0_x, q_x)[0] == pytest.approx(amp[0], rel=1e-12)
    assert excess_amplitude_m(u0_x, -q_x)[0] == pytest.approx(-amp[0], rel=1e-12)
    # Berkeland's own magnitude is unchanged: (1/2)|q| u_0 with u_0 = Q E/(m omega^2) at the pseudopotential frequency
    u0_berkeland = displacement_m(1.0, mass, TWO_PI * 1e6)[0]
    assert abs(excess_amplitude_m(u0_berkeland, q_x)[0]) == pytest.approx(
        0.5 * abs(q_x) * u0_berkeland, rel=1e-12
    )


def test_modulation_index_is_signed_and_its_quadrature_pair_reproduces_both_terms() -> None:
    """beta = delta_k . u_1 (Appendix E), not |delta_k . u_1|; ``as_modulation`` turns (ip, op) into (beta, offset)."""
    yb = _Mass(170.93578162009095)
    k = TWO_PI / 369.5e-9
    dk = np.array([k, 0.0, 0.0])
    plus = _yb_trap(1.0).micromotion_beta(yb, dk)
    minus = _yb_trap(-1.0).micromotion_beta(yb, dk)
    assert plus.in_phase < 0.0 < minus.in_phase
    assert minus.in_phase == pytest.approx(-plus.in_phase, rel=1e-12)
    assert modulation_index(dk, np.array([1e-9, 0.0, 0.0])) == pytest.approx(k * 1e-9, rel=1e-12)
    assert modulation_index(dk, np.array([-1e-9, 0.0, 0.0])) == pytest.approx(-k * 1e-9, rel=1e-12)
    # beta cos(theta + offset) = ip cos theta + op sin theta for every theta, both quadratures and every sign of ip
    for ip, op in ((-0.4, 0.0), (0.4, 0.0), (0.3, 0.4), (-0.3, 0.4), (0.3, -0.4), (0.0, 0.0)):
        beta, offset = MicromotionIndex(ip, op, "peak").as_modulation()
        assert beta == pytest.approx(math.hypot(ip, op), rel=1e-12, abs=1e-15)
        for theta in np.linspace(0.0, TWO_PI, 17):
            assert beta * math.cos(theta + offset) == pytest.approx(
                ip * math.cos(theta) + op * math.sin(theta), rel=1e-12, abs=1e-15
            )
    # the rms tag scales both quadratures before the pair is formed
    assert MicromotionIndex(-0.3, 0.4, "rms").as_peak().in_phase == pytest.approx(-0.3 * math.sqrt(2))


def test_micromotion_index_refuses_a_missing_geometry_instead_of_reporting_beta_zero() -> None:
    """beta = 0 means "no micromotion", not "the number is unavailable": ``micromotion_index`` used to swallow every
    ValueError from ``Trap.micromotion_beta``, so an rf phase imbalance with no rod geometry factors (R_m, alpha) - the
    one input PLAN 4.1.1 says no source supplies - silently switched the whole micromotion comb off. A trap with no rf
    record at all is the legitimately-zero case and stays zero (Section 4.1.1: beta = 0 and C0 = 1 there)."""
    from qutip_trap.trap.surface import Electrodes

    base = single_ion_raman_device(
        rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0), stray=(50.0, 0.0, 0.0)
    )
    dd = derive_raman_drive(base, 0, (0, 1), scattering=False)
    dk = np.asarray(dd.delta_k, dtype=float)
    assert micromotion_index(base, 0, dk)[0] > 0.0

    # no rf record: zero by construction, not by fallback
    no_rf = dataclasses.replace(base, trap=dataclasses.replace(base.trap, rf=None))
    assert micromotion_index(no_rf, 0, dk) == (0.0, 0.0)
    assert micromotion_index(base, 0, np.zeros(3)) == (0.0, 0.0), "a drive with no wavevector"

    # an rf phase imbalance with no rod geometry: the out-of-phase term (1/4) q_x R alpha phi_ac is unavailable
    imbalanced = dataclasses.replace(
        base,
        trap=dataclasses.replace(
            base.trap, rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0, phase_imbalance_rad=1e-3)
        ),
    )
    with pytest.raises(ValueError, match="R_m and alpha"):
        micromotion_index(imbalanced, 0, dk)
    with pytest.raises(ValueError, match="R_m and alpha"):
        detection_micromotion(imbalanced, 0, imbalanced.beams[:1])
    # and a rod record whose dc voltages the linear map cannot read: also an error, not beta = 0
    rod = dataclasses.replace(
        base,
        trap=dataclasses.replace(
            base.trap,
            omega_hz=None,
            rf=RfDrive(frequency_hz=20e6, voltage_peak_v=300.0),
            dc=DcElectrodes({"endcaps": 8.0, "extra": 1.0}),
            geometry=Electrodes("rod_quadrupole", {"R_m": 1e-3, "Z0_m": 2e-3, "kappa": 0.3}),
        ),
    )
    with pytest.raises(ValueError, match="one endcap voltage"):
        micromotion_index(rod, 0, dk)


def test_the_modulated_builder_first_micromotion_sideband_changes_sign_across_the_compensating_shim() -> None:
    """Section 9.17: the first micromotion sideband of the exact modulated drive steps by pi as the residual field
    crosses zero. J_1 is odd, so its amplitude in e^{i beta cos(Omega_rf t + delta)} carries the sign of beta; the
    builder therefore has to pass the index's own quadrature offset, not only |beta|."""
    base = single_ion_raman_device(
        rf=RfDrive(frequency_hz=30e6, voltage_peak_v=100.0), stray=(50.0, 0.0, 0.0)
    )
    space = HilbertSpace((2,), (ModeTruncation(KX, 6, (0, 2), 0.2),), None, (0, 2))
    omega_rf = base.trap.rf.omega_rad_s  # type: ignore[union-attr]
    period = TWO_PI / omega_rf
    times = np.linspace(0.0, period, 129)[:-1]

    element: tuple[int, int] | None = None

    def first_harmonic(stray_x: float) -> complex:
        """(2/T) int c(t) e^{-i Omega_rf t} dt / <c>: the drive coefficient's first micromotion-sideband weight.

        e^{i beta cos(Omega t + delta)} = sum_n i^n J_n(beta) e^{in(Omega t + delta)}, and the period mean is J_0(beta),
        so this projection returns 2 i J_1(beta) e^{i delta}/J_0(beta) - odd in the SIGNED index, since J_1 is odd and
        delta steps by pi while J_0 is even."""
        nonlocal element
        trap = dataclasses.replace(base.trap, stray_field_v_per_m=(stray_x, 0.0, 0.0))
        dev = dataclasses.replace(base, trap=trap, crystal=solve_crystal(trap, base.crystal.species))
        dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
        drive = square_drive(dd, include_stark=False, rf_locked=True, rf_phase_rad=0.0)
        built = build_hamiltonian(
            dev,
            [Pulse(drive, 0.0, 5.0 * period, "p", ())],
            space,
            options=BuilderOptions(micromotion="modulated"),
        )
        part = built.drive_parts["p"]
        matrices = [part(float(t)).full() for t in times]
        if element is None:
            flat = int(np.argmax(np.abs(matrices[0])))
            element = (flat // matrices[0].shape[1], flat % matrices[0].shape[1])
        vals = np.array([complex(m[element]) for m in matrices])
        dc = complex(np.mean(vals))
        assert abs(dc) > 0.0, "the carrier does not average to zero over one rf period"
        return complex(2.0 * np.mean(vals * np.exp(-1j * omega_rf * times)) / dc)

    s_plus = first_harmonic(+50.0)
    s_minus = first_harmonic(-50.0)
    assert abs(s_plus) > 1e-3, "there is a first micromotion sideband at all"
    assert s_minus == pytest.approx(-s_plus, rel=5e-3, abs=1e-9), "a pi step across compensation"
    assert first_harmonic(0.0) == pytest.approx(0j, abs=1e-9), "no residual field, no micromotion sideband"
    # and the weight is 2 i J_1(beta) e^{i delta} with delta = -pi at a positive q_x and a positive field, so the
    # SIGN of the sideband is the one the plan's rf phase origin fixes
    dd = derive_raman_drive(base, 0, (0, 1), scattering=False)
    assert dd.micromotion is not None
    beta, offset = dd.micromotion.as_peak().as_modulation()
    assert dd.micromotion.in_phase < 0.0 and offset == pytest.approx(-math.pi, abs=1e-12)
    assert (beta, offset) == micromotion_index(base, 0, np.asarray(dd.delta_k, dtype=float))
    # normalized by the period mean of e^{i beta cos}, which is J_0(beta), so the projection is 2 i J_1 e^{i delta}/J_0
    expected = 2j * jv(1, beta) * np.exp(1j * offset) / jv(0, beta)
    got = s_plus if abs(s_plus - expected) < abs(s_plus.conjugate() - expected) else s_plus.conjugate()
    assert got == pytest.approx(expected, rel=5e-3, abs=1e-9)
