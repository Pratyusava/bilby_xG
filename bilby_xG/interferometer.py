# Licensed under an MIT style license -- see LICENSE

"""Interferometer with a frequency-dependent, finite-size antenna response.

Extends :class:`bilby.gw.detector.interferometer.Interferometer` with the
finite-size, Earth-rotation-aware antenna response needed for
kilometre-to-tens-of-kilometre next-generation detectors (Cosmic Explorer,
Einstein Telescope), following Baral et al. (2023), arXiv:2304.09889 and
Nishizawa et al. (2009), arXiv:0903.0528.

The single, physics-agnostic response method takes an optional
:class:`~bilby_xG.propagation.Propagation` model. With the default (general
relativity) it reproduces the standard frequency-dependent response exactly;
supplying a :class:`~bilby_xG.propagation.SpeedOfGravity` or
:class:`~bilby_xG.propagation.ModifiedDispersion` model enables the
corresponding beyond-GR physics through a single code path.
"""
import numpy as np
from bilby_cython.geometry import greenwich_mean_sidereal_time
from bilby_cython.geometry import time_delay_from_geocenter as _cython_time_delay

from bilby.core.utils import ra_dec_to_theta_phi, speed_of_light
from bilby.gw.detector.calibration import Recalibrate
from bilby.gw.detector.interferometer import Interferometer as _Interferometer

from .geometry import InterferometerGeometry
from .propagation import Propagation, build_propagation
from .utils import calculate_time_to_merger_for_any_mode

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]


def _mode_integer(mode_key):
    """Parse a waveform-mode dict key into an integer azimuthal number ``m``.

    Accepts either bare strings such as ``"2"`` / ``"3"`` (keyed by ``m``) or
    ``"l,m"`` strings such as ``"2,2"`` produced by the relative-binning
    individual-mode source models.
    """
    if ',' in mode_key:
        return int(mode_key.split(',')[-1])
    return int(mode_key[-1])


class Interferometer(_Interferometer):
    """An interferometer with a frequency-dependent antenna response.

    Parameters are as in
    :class:`bilby.gw.detector.interferometer.Interferometer`. The geometry is
    a :class:`bilby_xG.geometry.InterferometerGeometry`, which additionally
    exposes the per-arm detector tensors used by the finite-size response.
    """

    def __init__(self, name, power_spectral_density, minimum_frequency,
                 maximum_frequency, length, latitude, longitude, elevation,
                 xarm_azimuth, yarm_azimuth, xarm_tilt=0., yarm_tilt=0.,
                 calibration_model=None):
        if calibration_model is None:
            calibration_model = Recalibrate()
        super(Interferometer, self).__init__(
            name=name, power_spectral_density=power_spectral_density,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency, length=length,
            latitude=latitude, longitude=longitude, elevation=elevation,
            xarm_azimuth=xarm_azimuth, yarm_azimuth=yarm_azimuth,
            xarm_tilt=xarm_tilt, yarm_tilt=yarm_tilt,
            calibration_model=calibration_model)
        self.geometry = InterferometerGeometry(
            length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
            xarm_tilt, yarm_tilt)

    @staticmethod
    def _finite_size_factor(x, y):
        """Single-arm finite-size response factor (Baral et al. 2023, Eq. 2.13)."""
        return 0.5 * (
            np.exp(-np.pi * 1j * x * (1. + y)) * np.sinc(x * (1 - y))
            + np.exp(np.pi * 1j * x * (1. - y)) * np.sinc(x * (1 + y))
        )

    def frequency_dependent_antenna_response(
            self, ra, dec, time, psi, frequencies, start_time,
            times_to_coalescence, propagation=None,
            earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
            finite_size=True):
        """Frequency-dependent plus/cross antenna response.

        See Nishizawa et al. (2009) arXiv:0903.0528 for the polarisation
        tensors and Baral et al. (2023) arXiv:2304.09889 for the finite-size
        implementation. ``[u, v, w]`` are the Earth-frame and ``[m, n, omega]``
        the wave-frame basis vectors.

        Parameters
        ==========
        ra, dec: float
            Source right ascension and declination (radians).
        time: float
            Geocentric coalescence time (GPS seconds).
        psi: float
            Polarisation angle (radians).
        frequencies: array_like
            Frequencies at which to evaluate the response.
        start_time: float
            Start time of the data segment.
        times_to_coalescence: array_like
            Time-to-coalescence at each frequency (sets the Earth orientation).
        propagation: bilby_xG.propagation.Propagation, optional
            Propagation model providing ``phase_velocity``/``group_velocity``.
            Defaults to general relativity (``v_p = v_g = c``).
        earth_rotation_time_delay, earth_rotation_beam_patterns, finite_size: bool
            Toggle Earth-rotation time delay, Earth-rotation beam patterns and
            finite-size detector effects respectively.

        Returns
        =======
        (fps, fcs): tuple of array_like
            Complex plus and cross antenna response at each frequency.

        Notes
        =====
        Only the plus and cross modes are computed. The detector-position time
        delay is incorporated directly in the returned beam patterns.
        """
        if propagation is None:
            propagation = Propagation()

        if earth_rotation_time_delay or earth_rotation_beam_patterns:
            gmst_at_tc = greenwich_mean_sidereal_time(time)
            day = 24. * 60. * 60.
            gmst_day_after = greenwich_mean_sidereal_time(time + day)
            one_second_to_gmst = (gmst_day_after - gmst_at_tc) / day
            gmsts = gmst_at_tc - one_second_to_gmst * times_to_coalescence
        else:
            gmsts = np.ones(len(frequencies)) * greenwich_mean_sidereal_time(time)

        # basis vectors of the GW frame
        thetas, phis = ra_dec_to_theta_phi(ra, dec, gmsts)
        cosphis = np.cos(phis)
        costhetas = np.cos(thetas)
        sinphis = np.sin(phis)
        sinthetas = np.sin(thetas)
        u = np.zeros(shape=(3, len(phis)))
        u[0] = cosphis * costhetas
        u[1] = costhetas * sinphis
        u[2] = -sinthetas
        v = np.zeros(shape=(3, len(phis)))
        v[0] = -sinphis
        v[1] = cosphis
        m = -u * np.sin(psi) - v * np.cos(psi)
        n = -u * np.cos(psi) + v * np.sin(psi)
        omegas = np.zeros(shape=(3, len(phis)))
        omegas[0] = sinthetas * cosphis
        omegas[1] = sinthetas * sinphis
        omegas[2] = costhetas

        # beam patterns
        tmp = np.einsum('ik,jk->ijk', m, n)
        pol_plus = np.einsum('ik,jk->ijk', m, m) - np.einsum('ik,jk->ijk', n, n)
        pol_cross = tmp + np.transpose(tmp, axes=(1, 0, 2))

        if not finite_size:
            fps = np.einsum('ij,ijk->k', self.geometry.detector_tensor, pol_plus)
            fcs = np.einsum('ij,ijk->k', self.geometry.detector_tensor, pol_cross)
            if not earth_rotation_beam_patterns:
                fps = fps[-1] * np.ones(len(fps))
                fcs = fcs[-1] * np.ones(len(fcs))
        else:
            fpxx = np.einsum('ij,ijk->k', self.geometry.xx, pol_plus)
            fpyy = np.einsum('ij,ijk->k', self.geometry.yy, pol_plus)
            fcxx = np.einsum('ij,ijk->k', self.geometry.xx, pol_cross)
            fcyy = np.einsum('ij,ijk->k', self.geometry.yy, pol_cross)
            if not earth_rotation_beam_patterns:
                fpxx = fpxx[-1] * np.ones(len(fpxx))
                fpyy = fpyy[-1] * np.ones(len(fpyy))
                fcxx = fcxx[-1] * np.ones(len(fcxx))
                fcyy = fcyy[-1] * np.ones(len(fcyy))

            px = -np.dot(omegas.T, self.geometry.x)
            py = -np.dot(omegas.T, self.geometry.y)
            phase_velocity = propagation.phase_velocity(frequencies)
            fL_over_c = (frequencies * self.geometry.length * 10. ** 3.
                         / (speed_of_light * phase_velocity))
            # Always recompute the single-arm factors: with a non-trivial
            # propagation model they depend on frequency through v_p, so
            # caching across evaluations would be incorrect.
            self.Dxx = self._finite_size_factor(fL_over_c, px)
            self.Dyy = self._finite_size_factor(fL_over_c, py)
            fps = fpxx * self.Dxx - fpyy * self.Dyy
            fcs = fcxx * self.Dxx - fcyy * self.Dyy

        # propagation time-shift factor (group velocity)
        group_velocity = propagation.group_velocity(frequencies)
        dts = -np.dot(omegas.T, self.geometry.vertex) / (speed_of_light * group_velocity)
        ifo_times = time - start_time + dts
        if not earth_rotation_time_delay:
            ifo_times = ifo_times[-1]

        exp_fac = np.exp(-1j * 2. * np.pi * frequencies * ifo_times)
        fps = fps * exp_fac
        fcs = fcs * exp_fac
        return fps, fcs

    def get_detector_response_for_frequency_dependent_antenna_response(
            self, waveform_polarizations, parameters, start_time, frequencies,
            earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
            finite_size=True):
        """Combine waveform polarisations with the frequency-dependent response.

        Handles both the standard ``{"plus": ..., "cross": ...}`` polarisation
        dict and the per-mode nested dict
        ``{mode_key: {"plus": ..., "cross": ...}}`` produced by the
        individual-mode source models. The propagation model is selected from
        ``parameters`` via :func:`bilby_xG.propagation.build_propagation`, so
        a sampled ``vG`` or ``(a, A)`` automatically enables the corresponding
        beyond-GR physics, while their absence recovers general relativity.

        Note: the calibration model is not applied here; only plus and cross
        modes are used.
        """
        propagation = build_propagation(parameters)

        if 'plus' in waveform_polarizations.keys():
            times_to_coalescence = calculate_time_to_merger_for_any_mode(
                frequencies, parameters['mass_1'], parameters['mass_2'],
                parameters['chi_1'], parameters['chi_2'], mode=2, safety=1)
            correction_factor = np.exp(
                1j * propagation.propagation_phase(frequencies, mode=2))
            fps, fcs = self.frequency_dependent_antenna_response(
                parameters['ra'], parameters['dec'], parameters['geocent_time'],
                parameters['psi'],
                times_to_coalescence=times_to_coalescence,
                propagation=propagation,
                frequencies=frequencies,
                start_time=start_time,
                earth_rotation_time_delay=earth_rotation_time_delay,
                finite_size=finite_size,
                earth_rotation_beam_patterns=earth_rotation_beam_patterns,
            )
            signal_ifo = correction_factor * (
                waveform_polarizations['plus'] * fps
                + waveform_polarizations['cross'] * fcs
            )
        else:
            signal_ifo = np.zeros(len(frequencies), dtype=complex)
            for mode_key in waveform_polarizations.keys():
                mode = _mode_integer(mode_key)
                times_to_coalescence = calculate_time_to_merger_for_any_mode(
                    frequencies, parameters['mass_1'], parameters['mass_2'],
                    parameters['chi_1'], parameters['chi_2'], mode=mode, safety=1)
                correction_factor = np.exp(
                    1j * propagation.propagation_phase(frequencies, mode=mode))
                fps, fcs = self.frequency_dependent_antenna_response(
                    parameters['ra'], parameters['dec'], parameters['geocent_time'],
                    parameters['psi'],
                    times_to_coalescence=times_to_coalescence,
                    propagation=propagation,
                    frequencies=frequencies,
                    start_time=start_time,
                    earth_rotation_time_delay=earth_rotation_time_delay,
                    finite_size=finite_size,
                    earth_rotation_beam_patterns=earth_rotation_beam_patterns,
                )
                signal_ifo += correction_factor * (
                    waveform_polarizations[mode_key]['plus'] * fps
                    + waveform_polarizations[mode_key]['cross'] * fcs
                )
        return signal_ifo

    def time_delay_from_geocenter(self, ra, dec, time, vG=1):
        """Detector time delay from geocentre, optionally rescaled by ``vG``.

        Backward compatible with bilby: ``vG=1`` reproduces the upstream result
        exactly. ``vG`` (the speed of gravity as a fraction of ``c``) rescales
        the propagation speed, so the delay -- which scales as
        distance / speed -- is the geometric delay divided by ``vG``.
        """
        return _cython_time_delay(self.geometry.vertex, ra, dec, time) / vG
