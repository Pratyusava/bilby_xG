# Licensed under an MIT style license -- see LICENSE

"""Utilities for next-generation-detector inference.

Provides the fast time-to-merger estimate used by the frequency-dependent
antenna response, and helpers that locate the data files shipped with
:mod:`bilby_xG` (next-generation detector definitions and amplitude spectral
densities). The data helpers use :mod:`importlib.resources`, so they work
whether the package is installed as a wheel, an editable install, or run from
the source tree.
"""
from importlib.resources import files

import numpy as np

from bilby.core.utils.constants import (
    solar_mass, gravitational_constant, speed_of_light,
)

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]

#: Root of the packaged data directory.
DATA_ROOT = files("bilby_xG") / "data"
#: Directory of shipped ``*.interferometer`` files.
DETECTORS_DIR = DATA_ROOT / "detectors"
#: Directory of shipped ASD ``*.txt`` files.
NOISE_CURVES_DIR = DATA_ROOT / "noise_curves"


def detector_file(name):
    """Return the path to ``<name>.interferometer`` if shipped, else ``None``."""
    path = DETECTORS_DIR / f"{name}.interferometer"
    return str(path) if path.is_file() else None


def noise_curve_file(filename):
    """Return the path to a shipped noise-curve file, else ``None``.

    ``filename`` may be a bare name (e.g. ``"ce20_asd.txt"``).
    """
    path = NOISE_CURVES_DIR / filename
    return str(path) if path.is_file() else None


def calculate_time_to_merger_for_any_mode(frequency, mass_1, mass_2, chi_1=0,
                                          chi_2=0, mode=2, safety=1.1):
    """Time to merger from a given frequency, to 2PN, for an arbitrary mode.

    This uses Eq. (3.3) of arXiv:gr-qc/9502040 and is much faster than the
    ``XLALSimInspiralTaylorF2ReducedSpinChirpTime`` routine. The mode argument
    rescales the frequency by ``2 / |mode|`` so the time-to-coalescence track is
    correct for higher-order modes.

    Parameters
    ==========
    frequency: float or array_like
        Frequency (Hz) at which to evaluate the time to merger.
    mass_1, mass_2: float
        Detector-frame component masses (solar masses).
    chi_1, chi_2: float
        Dimensionless aligned-spin parameters.
    mode: int
        Azimuthal mode number ``m`` (e.g. 2 for the dominant 22 mode).
    safety: float
        Multiplicative safety factor (kept for API compatibility).

    Returns
    =======
    time_to_merger: float or array_like
        Time to merger from ``frequency``, in seconds.
    """
    msun_to_seconds = solar_mass * gravitational_constant / speed_of_light ** 3.
    frequency = 2 * frequency / np.abs(mode)
    total_mass = mass_1 + mass_2
    total_mass_in_seconds = total_mass * msun_to_seconds
    chirp_mass = (mass_1 * mass_2) ** (3. / 5.) / (mass_1 + mass_2) ** (1. / 5.)
    chirp_mass_in_seconds = chirp_mass * msun_to_seconds
    eta = mass_1 * mass_2 / (mass_1 + mass_2) ** 2.
    x = np.pi * total_mass_in_seconds * frequency
    # spin combinations
    beta = ((113. * (mass_1 / total_mass) ** 2. + 75. * eta) * chi_1
            + (113. * (mass_2 / total_mass) ** 2. + 75. * eta) * chi_2) / 12.
    sigma = (-247. * chi_1 * chi_2 + 721. * chi_1 * chi_2) * eta / 48.
    # up to 2PN
    tau0 = 5. / 256. * chirp_mass_in_seconds \
        * (np.pi * chirp_mass_in_seconds * frequency) ** (-8. / 3.)
    tau2 = 4. / 3. * (743. / 336. + 11. * eta / 4.) * x ** (2. / 3.) * tau0
    tau3 = -8. / 5. * (4. * np.pi - beta) * x * tau0
    tau4 = 2. * (3058673. / 1016064. + 5429. * eta / 1008.
                 + 617. * eta ** 2. / 144. - sigma) * x ** (4. / 3.) * tau0
    return tau0 + tau2 + tau3 + tau4
