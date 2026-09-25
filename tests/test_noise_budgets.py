"""Gate error budgets through the exact gate check, and the depolarizing and twirl summaries of a channel (PLAN.md Section
4.4.7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from qutip_trap.calibration.entangling import exact_gate_check
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.channels import heating_channels, motional_dephasing_channels
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.summary import (
    average_gate_infidelity,
    choi_from_unitary,
    depolarizing_choi,
    depolarizing_rate,
    entanglement_infidelity,
    pauli_string,
    pauli_twirl,
)
from qutip_trap.options import Numerics
from tests.fixtures import (
    X_COM_TWO_IONS,
    chain_device,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)
from tests.oracles import ballance_dephasing_error, ballance_heating_error

FAST = Numerics(mesolve_dimension_max=4096)


def _gate(loops: int, epsilon_hz: float):  # type: ignore[no-untyped-def]
    """A K-loop symmetric MS gate closed on the COM alone with the rocking mode frozen: dims 2 x 2 x 12."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    wf = Waveform.symmetric(
        modes, gate_mode=X_COM_TWO_IONS, loops=loops, epsilon_hz=epsilon_hz, all_modes=False
    )
    space = HilbertSpace((2, 2), (ModeTruncation(X_COM_TWO_IONS, 12, (0, 4), 0.15),), None, (0, 1, 2, 4, 5))
    table = table_with_waveform((0, 1), wf, device=dev, drives=raman_gate_drives(2))
    return dev, wf, space, table


def _check(dev, wf, space, table, channels=()):  # type: ignore[no-untyped-def]
    check, _ = exact_gate_check(
        dev, wf, (0, 1), raman_gate_drives(2), table, space=space, channels=channels, options=FAST
    )
    return check


@pytest.mark.slow
@pytest.mark.parametrize("loops", [1, 2])
def test_heating_during_the_gate_costs_ndot_tg_over_2k(loops: int) -> None:
    """Heating at 400 quanta/s on the gate mode costs eps_h = ndot t_g/(2K) of the K-loop gate to 8 % (Ballance 2016)."""
    dev, wf, space, table = _gate(loops, 20e3 * loops)
    base = _check(dev, wf, space, table)
    ndot = 400.0
    noisy = _check(dev, wf, space, table, channels=heating_channels(space, {X_COM_TWO_IONS: ndot}))
    loss = base.fidelity - noisy.fidelity
    assert loss == pytest.approx(ballance_heating_error(ndot, wf.duration_s, loops), rel=0.08)
    assert loss > 2e-3


@pytest.mark.slow
@pytest.mark.parametrize("loops", [1, 2])
def test_motional_dephasing_costs_alpha_k_tg_over_tau(loops: int) -> None:
    """L = a^dag a sqrt(2/tau) costs eps_d = alpha_K t_g/tau with alpha_K = (8K + 3)/(16 K^2) to 10 % (Ballance 2016)."""
    dev, wf, space, table = _gate(loops, 20e3 * loops)
    base = _check(dev, wf, space, table)
    t_g = wf.duration_s
    tau = t_g / 5e-3
    noisy = _check(dev, wf, space, table, channels=motional_dephasing_channels(space, {X_COM_TWO_IONS: tau}))
    assert base.fidelity - noisy.fidelity == pytest.approx(ballance_dephasing_error(t_g, tau, loops), rel=0.1)


def test_depolarizing_summary_and_over_rotation_twirl() -> None:
    """Lambda_eps has entanglement infidelity and depolarizing rate eps (1e-10, Chen 2023), the twirl of exp(-i alpha XX) is
    p_xx = sin^2 alpha (1e-12, Trout 2018), and a two-qubit average gate infidelity is 4/5 of the entanglement one."""
    for n in (1, 2):
        for eps in (1e-3, 0.05):
            c = depolarizing_choi(eps, n)
            ideal = choi_from_unitary(np.eye(2**n))
            assert entanglement_infidelity(c, ideal) == pytest.approx(eps, rel=1e-10)
            assert depolarizing_rate(c, ideal) == pytest.approx(eps, rel=1e-10)
            assert np.allclose(c, c.conj().T) and np.trace(c).real == pytest.approx(1.0)
    alpha = 0.23
    u = expm(-1j * alpha * pauli_string("XX"))
    tw = pauli_twirl(choi_from_unitary(u), 2)
    assert tw["XX"] == pytest.approx(math.sin(alpha) ** 2, abs=1e-12) and tw["II"] == pytest.approx(
        math.cos(alpha) ** 2, abs=1e-12
    )
    assert sum(tw.values()) == pytest.approx(1.0)
    assert average_gate_infidelity(1e-3, 4) == pytest.approx(0.8e-3)
    assert entanglement_infidelity(choi_from_unitary(u), choi_from_unitary(np.eye(4))) == pytest.approx(
        math.sin(alpha) ** 2, abs=1e-12
    )
