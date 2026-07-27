# Licensed under an MIT style license -- see LICENSE

"""Gravitational-wave propagation models.

This module unifies the two beyond-GR propagation effects that were previously
implemented on separate, mutually-conflicting branches of the bilby fork:

* the ``SoG`` branch added a single constant "speed of gravity" ``vG`` that
  rescaled the detector-tensor / time-shift terms of the frequency-dependent
  antenna response, and
* the ``dispersion`` branch added a frequency-dependent modified dispersion
  relation parameterised by ``(a, A)`` that rescaled the *same* terms with a
  frequency-dependent group/phase velocity **and** applied an additional
  cumulative propagation phase ``dphi`` accrued over the cosmological distance.

Both are expressed here through a single :class:`Propagation` interface that
exposes

* :meth:`~Propagation.phase_velocity` -- ``v_p / c`` (used in the finite-size
  detector-tensor term ``fL / (c v_p)``),
* :meth:`~Propagation.group_velocity` -- ``v_g / c`` (used in the propagation
  time-shift term ``dts = -omega.vertex / (c v_g)``), and
* :meth:`~Propagation.propagation_phase` -- an additive phase ``Delta Psi(f)``
  applied as ``exp(1j * Delta Psi)`` to the strain.

General relativity is the default and recovers ``v_p = v_g = 1`` and a zero
propagation phase, so the next-generation likelihoods reduce *exactly* to the
standard frequency-dependent-response result when no beyond-GR parameter is
sampled.
"""
import numpy as np

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>"]

# h_bar in units of eV * s scaled so that E = h_bar * (2 pi f) is in peV when
# combined with the 1e-21 normalisation of A (see the original branch code).
_H_BAR = 0.0006582119569000001
# Planck constant h, same unit convention as the original ``dphi`` branch code.
_H = 0.0041356677
_KM_BY_MPC = 3.240779289444365e-20
# 1 Mpc / speed_of_light, in seconds.
_MPC_TO_S = 1.0292713e14


class Propagation:
    """General-relativity propagation: ``v_p = v_g = c`` and zero extra phase.

    Subclasses override the methods that differ. All methods are vectorised
    over ``frequencies`` and return ``numpy`` arrays of the same shape.
    """

    def phase_velocity(self, frequencies):
        """Phase velocity ``v_p / c`` at each frequency (GR: ones)."""
        return np.ones_like(np.asarray(frequencies, dtype=float))

    def group_velocity(self, frequencies):
        """Group velocity ``v_g / c`` at each frequency (GR: ones)."""
        return np.ones_like(np.asarray(frequencies, dtype=float))

    def propagation_phase(self, frequencies, mode=2):
        """Additive propagation phase ``Delta Psi(f)`` (GR: zeros)."""
        return np.zeros_like(np.asarray(frequencies, dtype=float))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SpeedOfGravity(Propagation):
    """Constant "speed of gravity" model (the ``SoG`` branch).

    A single dimensionless ``vG = v / c`` rescales both the phase and group
    velocities; there is no additional propagation phase. ``vG = 1`` recovers
    general relativity.
    """

    def __init__(self, vG=1.0):
        self.vG = float(vG)

    def phase_velocity(self, frequencies):
        return np.full_like(np.asarray(frequencies, dtype=float), self.vG)

    def group_velocity(self, frequencies):
        return np.full_like(np.asarray(frequencies, dtype=float), self.vG)

    def __repr__(self):
        return f"SpeedOfGravity(vG={self.vG})"


class ModifiedDispersion(Propagation):
    """Lorentz-violating modified dispersion relation (the ``dispersion`` branch).

    The dispersion relation ``E^2 = p^2 c^2 + A p^a c^a`` (with ``A`` in units of
    ``1e-21 * peV**(2 - a)``) yields a frequency-dependent group and phase
    velocity and a cumulative propagation phase accrued over the comoving
    distance to the source. The phase additionally depends on the source
    luminosity distance, redshift, detector-frame chirp mass and ``H0``.

    Parameters
    ==========
    a: float
        Power-law index of the modified dispersion relation. ``a == 2`` is
        excluded (degenerate). ``a == 1`` uses the logarithmic-phase branch.
    A: float
        Dispersion amplitude in units of ``1e-21 * peV**(2 - a)``.
    luminosity_distance: float, optional
        Source luminosity distance in Mpc. Required for the propagation phase.
    mass_1, mass_2: float, optional
        Detector-frame component masses (solar masses). Used for the
        detector-frame chirp mass in the ``a == 1`` logarithmic branch.
    H0: float, optional
        Hubble constant in km/s/Mpc. Defaults to the active bilby cosmology.
    redshift: float, optional
        Source redshift. Computed from ``luminosity_distance`` if not given.
    """

    def __init__(self, a, A, luminosity_distance=None, mass_1=None,
                 mass_2=None, H0=None, redshift=None):
        # Imported lazily so that importing this module never forces bilby.gw
        # to import (keeps the dependency surface small and import-order safe).
        from bilby.gw.conversion import (
            luminosity_distance_to_redshift,
            component_masses_to_chirp_mass,
        )
        from bilby.gw.cosmology import get_cosmology
        from bilby.core.utils.constants import (
            solar_mass, gravitational_constant, speed_of_light,
        )

        if a == 2:
            raise ValueError("ModifiedDispersion is undefined for a == 2.")
        self.a = float(a)
        self.A = float(A)
        self.luminosity_distance = luminosity_distance
        self.H0 = get_cosmology().H0.value if H0 is None else float(H0)

        if redshift is not None:
            self.redshift = float(redshift)
        elif luminosity_distance is not None:
            self.redshift = float(luminosity_distance_to_redshift(luminosity_distance))
        else:
            self.redshift = None

        if None not in (mass_1, mass_2) and self.redshift is not None:
            detector_frame_chirp_mass = (
                component_masses_to_chirp_mass(mass_1, mass_2) * (1 + self.redshift)
            )
            self._chirp_mass_in_seconds = (
                detector_frame_chirp_mass
                * solar_mass * gravitational_constant / speed_of_light ** 3.
            )
        else:
            self._chirp_mass_in_seconds = None

    def phase_velocity(self, frequencies):
        energy = _H_BAR * 2 * np.pi * np.asarray(frequencies, dtype=float)
        amplitude = self.A * 1e-21
        return 1 - amplitude / 2 * energy ** (self.a - 2)

    def group_velocity(self, frequencies):
        energy = _H_BAR * 2 * np.pi * np.asarray(frequencies, dtype=float)
        amplitude = self.A * 1e-21
        return 1 + (self.a - 1) * amplitude / 2 * energy ** (self.a - 2)

    def propagation_phase(self, frequencies, mode=2):
        """Cumulative dephasing accrued during cosmological propagation.

        Returns zeros if the source distance was not supplied. Note: the
        original ``dispersion`` branch hard-coded ``f = 100`` inside this
        calculation, collapsing the (physical) frequency dependence to a
        constant. That line is a debugging leftover and is intentionally
        omitted here so the dephasing is correctly frequency dependent.
        """
        from bilby.gw.cosmology import get_cosmology

        frequencies = np.asarray(frequencies, dtype=float)
        if self.luminosity_distance is None or self.redshift is None:
            return np.zeros_like(frequencies)

        a, z = self.a, self.redshift
        amplitude = self.A * 1e-21
        lambda_A = _H * np.abs(amplitude) ** (1 / (a - 2))
        H_0 = self.H0 * _KM_BY_MPC
        Omega_m = get_cosmology().Om0
        Omega_de = 1 - Omega_m

        z_array = np.linspace(0, z, 1000)
        integrand = (1 + z_array) ** (a - 2) / np.sqrt(
            Omega_m * (1 + z_array) ** 3 + Omega_de
        )
        D_a = (1 + z) ** (1 - a) / H_0 * np.trapz(integrand, z_array)

        D_L = self.luminosity_distance * _MPC_TO_S
        lambda_A_eff = ((1 + z) ** (1 - a) * D_L / D_a) ** (1 / (a - 2)) * lambda_A

        if a != 1:
            return (
                np.sign(amplitude) * np.pi * D_L * lambda_A_eff ** (a - 2)
                * (2 * frequencies / mode) ** (a - 1) / (a - 1)
            )
        Mc = self._chirp_mass_in_seconds
        return (
            np.sign(amplitude) * np.pi * D_L / lambda_A_eff
            * np.log(Mc * (2 * frequencies / mode) * np.pi)
        )

    def __repr__(self):
        return f"ModifiedDispersion(a={self.a}, A={self.A})"


def build_propagation(parameters):
    """Select a :class:`Propagation` model from a parameter dictionary.

    Priority: a modified dispersion relation (``a`` and ``A`` both present)
    takes precedence over a constant speed of gravity (``vG``); if neither is
    present, general relativity is used. This lets a single likelihood support
    all three physics cases purely through which parameters are sampled.
    """
    if "a" in parameters and "A" in parameters:
        return ModifiedDispersion(
            a=parameters["a"],
            A=parameters["A"],
            luminosity_distance=parameters.get("luminosity_distance"),
            mass_1=parameters.get("mass_1"),
            mass_2=parameters.get("mass_2"),
            H0=parameters.get("H0"),
        )
    if "vG" in parameters:
        return SpeedOfGravity(vG=parameters["vG"])
    return Propagation()
