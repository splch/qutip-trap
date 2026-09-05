"""Readout, SPAM and results (PLAN.md Section 8; milestone M5)."""

from __future__ import annotations

from qutip_trap.readout.detection import Detector
from qutip_trap.readout.discriminate import POVM
from qutip_trap.readout.fluorescence import DarkStateReport

__all__ = ["POVM", "DarkStateReport", "Detector"]
