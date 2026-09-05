"""Units, constants and the Hz / rad-s convention enforced by types (PLAN.md Sections 5.6, 13; M0)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qutip_trap import units
from qutip_trap.units import Hz, RadPerS, hz_from_rad_s, lande_g_j, rad_s_from_hz


def test_two_pi_at_the_boundary() -> None:
    assert rad_s_from_hz(Hz(1.0)) == pytest.approx(2.0 * math.pi)
    assert hz_from_rad_s(RadPerS(2.0 * math.pi)) == pytest.approx(1.0)


@given(st.floats(min_value=1e-6, max_value=1e15, allow_nan=False, allow_infinity=False))
def test_round_trip_hz_rad_s(f: float) -> None:
    assert hz_from_rad_s(rad_s_from_hz(Hz(f))) == pytest.approx(f, rel=1e-15)


def test_codata_2022_values_are_pinned() -> None:
    """A silent change of CODATA edition in scipy must fail loudly (Section 5.6)."""
    assert units.CODATA_EDITION == "CODATA 2022"
    assert units.MU_B_J_PER_T == pytest.approx(9.2740100657e-24, rel=1e-12)
    assert units.MU_B_OVER_H_HZ_PER_T == pytest.approx(13996244917.1, rel=1e-12)
    assert units.MU_N_J_PER_T == pytest.approx(5.0507837393e-27, rel=1e-12)
    assert units.A_0_M == pytest.approx(5.29177210544e-11, rel=1e-12)
    assert units.ELECTRON_MASS_U == pytest.approx(0.0005485799090441, rel=1e-12)
    assert units.G_S == pytest.approx(2.00231930436092, rel=1e-13)
    assert units.G_S > 0.0, "Steck's sign convention: g_S positive (Section 13)"


def test_mu_b_over_h_matches_section_4_5_1() -> None:
    """Section 4.5.1 prints mu_B/h = 1.399624 MHz/G."""
    assert units.MU_B_OVER_H_MHZ_PER_G == pytest.approx(1.399624, abs=5e-7)


def test_lande_factors_of_the_40ca_levels() -> None:
    """Section 4.5.7: g_J = 2, 2/3, 4/3, 4/5, 6/5 for S1/2, P1/2, P3/2, D3/2, D5/2 with g_S = 2 exactly."""
    half = 0.5
    assert lande_g_j(0, half, half, g_s=2.0) == pytest.approx(2.0)
    assert lande_g_j(1, half, half, g_s=2.0) == pytest.approx(2.0 / 3.0)
    assert lande_g_j(1, half, 1.5, g_s=2.0) == pytest.approx(4.0 / 3.0)
    assert lande_g_j(2, half, 1.5, g_s=2.0) == pytest.approx(0.8)
    assert lande_g_j(2, half, 2.5, g_s=2.0) == pytest.approx(1.2)
    # with the measured g_S the S1/2 factor is Steck's 2.0023193 itself
    assert lande_g_j(0, half, half) == pytest.approx(units.G_S)


def test_wavenumber_to_wavelength_reproduces_the_729_nm_line() -> None:
    """NIST Ca II D5/2 at 13710.88 cm^-1 is the 729.347 nm vacuum line of Section 4.5.7."""
    lam = units.wavelength_vac_m_from_hz(units.hz_from_wavenumber_cm(13710.88))
    assert lam == pytest.approx(729.347e-9, rel=2e-6)


def test_wavelength_from_non_positive_frequency_is_refused() -> None:
    with pytest.raises(ValueError):
        units.wavelength_vac_m_from_hz(Hz(0.0))


def test_type_checker_refuses_hz_where_rad_s_is_expected(tmp_path: Path) -> None:
    """The convention is 'enforced by types' (M0): mypy must reject passing an Hz where a RadPerS is expected."""
    snippet = textwrap.dedent(
        """
        from qutip_trap.units import Hz, RadPerS, rad_s_from_hz

        def needs_angular(w: RadPerS) -> float:
            return float(w)

        f = Hz(1.0e6)
        needs_angular(f)                  # error: Hz is not RadPerS
        needs_angular(rad_s_from_hz(f))   # fine: converted at the boundary
        """
    )
    path = tmp_path / "misuse.py"
    path.write_text(snippet, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-error-summary",
            "--follow-imports=silent",
            str(path),
        ],
        capture_output=True,
        env={k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "CLICOLOR_FORCE")}
        | {"NO_COLOR": "1"},
        text=True,
        check=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert 'incompatible type "Hz"; expected "RadPerS"' in proc.stdout, proc.stdout
    assert proc.stdout.count("error:") == 1, proc.stdout
