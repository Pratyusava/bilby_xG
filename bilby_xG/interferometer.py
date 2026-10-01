# Licensed under an MIT style license -- see LICENSE

"""Interferometer with a frequency-dependent, finite-size antenna response.

Extends :class:`bilby.gw.detector.interferometer.Interferometer` with the
finite-size, Earth-rotation-aware antenna response needed for
kilometre-to-tens-of-kilometre next-generation detectors (Cosmic Explorer,
Einstein Telescope), following Baral et al. (2023), arXiv:2304.09889 and
Nishizawa et al. (2009), arXiv:0903.0528. The arrival time follows the
barycentric position of each detector, including the Earth's orbit, relative
to a chosen center (:mod:`bilby_xG.motion`).

The single, physics-agnostic response method takes an optional
:class:`~bilby_xG.propagation.Propagation` model. With the default (general
relativity) it reproduces the standard frequency-dependent response exactly;
supplying a :class:`~bilby_xG.propagation.SpeedOfGravity` or
:class:`~bilby_xG.propagation.ModifiedDispersion` model enables the
corresponding beyond-GR physics through a single code path.
"""
import os

import numpy as np
from bilby_cython.geometry import greenwich_mean_sidereal_time
from bilby_cython.geometry import time_delay_from_geocenter as _cython_time_delay

from bilby.core.utils import logger, ra_dec_to_theta_phi, speed_of_light
from bilby.gw.detector.calibration import Recalibrate
from bilby.gw.detector.interferometer import Interferometer as _Interferometer

from .geometry import InterferometerGeometry
from .motion import geocentre_delay as _geocentre_delay, resolve_center, sky_direction
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


def compute_wave_frame(ra, dec, time, psi, frequencies, times_to_coalescence,
                       earth_rotation=True):
    """The detector-independent part of the frequency-dependent response.

    The propagation direction ``omegas`` (shape ``(3, n)``) and the plus and
    cross polarisation tensors (shape ``(3, 3, n)``) at each frequency, in
    the Earth frame at the time that frequency is emitted (at the
    coalescence time if ``earth_rotation`` is false). Every detector sees
    the same frame, so it can be shared between detectors, see
    :meth:`Interferometer.frequency_dependent_antenna_response`.
    """
    if earth_rotation:
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

    # polarisation tensors
    tmp = np.einsum('ik,jk->ijk', m, n)
    pol_plus = np.einsum('ik,jk->ijk', m, m) - np.einsum('ik,jk->ijk', n, n)
    pol_cross = tmp + np.transpose(tmp, axes=(1, 0, 2))
    return omegas, pol_plus, pol_cross


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

    def optimal_snr_squared(self, signal, max_bins=2 ** 20):
        """Approximate optimal SNR^2 via strided frequency-bin decimation.

        ``signal`` and this interferometer's PSD/frequency arrays are
        full-band (``duration * sampling_frequency`` bins -- O(1e8) at ET's
        low ``minimum_frequency``, ~GB-scale per array). The upstream
        implementation (``bilby.gw.detector.interferometer.Interferometer``)
        boolean-masks and squares full-size copies internally, multiplying
        peak transient memory several-fold over the persistent per-detector
        arrays already held; on this codebase's low-``minimum_frequency``
        runs that is enough to OOM a memory-constrained host (observed:
        killed mid-way through the injected-SNR log lines on a 3-detector ET
        triangle at minimum_frequency=3Hz). This method is only used for
        logging/diagnostics, not the likelihood -- the relative-binning
        likelihood computes its own reduced-data SNR -- so it is safe to
        approximate: |h(f)|^2/S_n(f) varies smoothly with frequency (unlike
        the oscillatory complex waveform itself), so summing over a strided
        subset of frequency bins and rescaling by the stride gives an SNR
        accurate to well under a percent while only ever operating on
        decimated-size arrays (all indexing below is done after striding, so
        no full-size copy is ever materialized). Falls back to the exact
        upstream computation when the unmasked band is already small enough
        that decimation isn't needed.
        """
        mask = self.strain_data.frequency_mask
        n_unmasked = int(np.count_nonzero(mask))
        stride = max(1, n_unmasked // max_bins)
        if stride == 1:
            return super().optimal_snr_squared(signal=signal)

        strided_mask = mask[::stride]
        s = signal[::stride][strided_mask]
        freqs = self.strain_data.frequency_array[::stride][strided_mask]
        psd = (self.power_spectral_density.get_power_spectral_density_array(
            frequency_array=freqs) * self._window_power_correction)
        integrand = np.conj(s) * s / psd
        return 4 / self.strain_data.duration * np.sum(integrand) * stride

    def plot_data(self, signal=None, outdir='.', label=None, n_points=2000):
        """Memory-safe characteristic-strain frequency-domain data plot.

        The upstream implementation
        (``bilby.gw.detector.interferometer.Interferometer.plot_data``)
        computes the full-band ASD array and calls ``loglog`` on every
        in-band frequency bin -- at low ``minimum_frequency`` (e.g. 3Hz on
        ET) that is O(1e8) points, each contributing array several GB, plus
        matplotlib's own vertex-buffer allocation for that many points. This
        is the dominant remaining memory cost once
        :meth:`optimal_snr_squared` is fixed the same way (observed: several
        GB retained after the SNR lines print, on the injected-strain plot
        call immediately after). Since the plot is log-log, a geometric
        (log-spaced) subsample of ``n_points`` in-band frequencies loses no
        visible structure while only ever touching O(n_points)-sized arrays
        -- the PSD is queried directly at those frequencies rather than via
        the cached full-band ``amplitude_spectral_density_array`` property,
        so no full-size array is built just to be subsampled.

        Also plots characteristic strain rather than raw ASD: ``2 f |h(f)|``
        for the (injected) signal and ``sqrt(f) * ASD(f) = sqrt(f S(f))``
        for the noise floor -- the standard convention that puts both on a
        directly comparable, frequency-weighted scale.
        """
        import matplotlib.pyplot as plt
        from bilby.core import utils as core_utils
        if core_utils.command_line_args.bilby_test_mode:
            return

        mask = self.strain_data.frequency_mask
        idxs = np.flatnonzero(mask)
        f_full = self.strain_data.frequency_array
        f_geom = np.geomspace(f_full[idxs[0]], f_full[idxs[-1]], min(n_points, idxs.size))
        sel = np.unique(idxs[np.searchsorted(f_full[idxs], f_geom)])

        f_plot = f_full[sel]
        strain_hc = 2 * f_plot * np.abs(self.strain_data.frequency_domain_strain[sel])
        psd = (self.power_spectral_density.get_power_spectral_density_array(
            frequency_array=f_plot) * self._window_power_correction)
        asd_hc = np.sqrt(f_plot * psd)

        fig, ax = plt.subplots()
        ax.loglog(f_plot, strain_hc, color='C0', label=self.name)
        ax.loglog(f_plot, asd_hc, color='C1', lw=1.0, label=self.name + ' ASD')
        if signal is not None:
            signal_hc = 2 * f_plot * np.abs(signal[sel])
            ax.loglog(f_plot, signal_hc, color='C2', label='Signal')
        ax.grid(True)
        ax.set_ylabel(r'Characteristic strain $h_c(f)$')
        ax.set_xlabel(r'Frequency [Hz]')
        ax.legend(loc='best')
        fig.tight_layout()
        if label is None:
            fig.savefig('{}/{}_frequency_domain_data.png'.format(outdir, self.name))
        else:
            fig.savefig('{}/{}_{}_frequency_domain_data.png'.format(
                outdir, self.name, label))
        plt.close(fig)

    def offload_frequency_domain_strain(self, cache_dir):
        """Move the full-band frequency-domain strain to a disk-backed array.

        ``strain_data._frequency_domain_strain`` is the one full-band array
        (O(1e8) complex128 bins at ET's low ``minimum_frequency``) that
        cannot be cheaply regenerated -- it's the actual (injected or real)
        data. Writing it to disk once and reopening it memory-mapped means
        it no longer costs resident memory except for the pages a caller
        actually touches (e.g. one relative-binning summary-data chunk at a
        time), and it pickles as a small path/dtype/shape marker instead of
        its full contents (see :meth:`__getstate__`). Idempotent: a no-op if
        this interferometer's strain is already memory-mapped.
        """
        strain = self.strain_data._frequency_domain_strain
        if isinstance(strain, np.memmap):
            return
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{self.name}_frequency_domain_strain.npy")
        np.save(path, strain)
        self.strain_data._frequency_domain_strain = np.load(path, mmap_mode='r')
        logger.info(
            f"{self.name}: offloaded frequency-domain strain to {path} "
            f"({strain.nbytes / 1e9:.2f} GB).")

    def discard_regenerable_frequency_caches(self):
        """Drop full-band caches that bilby regenerates lazily and cheaply.

        ``frequency_array``, ``frequency_mask`` (from duration/
        sampling_frequency/start_time) and the PSD's ``psd_array``/
        ``asd_array`` (from the PSD file) are all deterministically
        recomputed by bilby's own properties on next access -- there is no
        need to keep O(1e8)-element copies of them resident just because
        something touched them once (e.g. during summary-data computation).
        """
        times_and_frequencies = self.strain_data._times_and_frequencies
        times_and_frequencies._frequency_array = None
        times_and_frequencies._frequency_array_updated = False
        self.strain_data._frequency_mask = None
        self.strain_data._frequency_mask_updated = False
        self.power_spectral_density._cache = dict(
            psd_array=None, asd_array=None, frequency_array=None)

    def __getstate__(self):
        """Pickle the memory-mapped strain as a path, not its full contents.

        ``numpy.memmap`` has no special pickling behaviour of its own -- by
        default it pickles like any other ``ndarray``, i.e. embeds the full
        buffer. That would defeat :meth:`offload_frequency_domain_strain`
        entirely (both for on-disk pickles and for the arrays that travel
        through a sampler's worker-pool ``initargs``), so this intercepts
        it explicitly.
        """
        state = self.__dict__.copy()
        strain = state["strain_data"].__dict__.get("_frequency_domain_strain")
        if isinstance(strain, np.memmap):
            state = dict(state)
            state["strain_data"] = state["strain_data"].__class__.__new__(
                state["strain_data"].__class__)
            state["strain_data"].__dict__.update(self.strain_data.__dict__)
            state["strain_data"].__dict__["_frequency_domain_strain"] = (
                "__diskarray__", strain.filename, str(strain.dtype), strain.shape)
        return state

    def __setstate__(self, state):
        strain_data = state.get("strain_data")
        if strain_data is not None:
            marker = strain_data.__dict__.get("_frequency_domain_strain")
            if isinstance(marker, tuple) and marker[:1] == ("__diskarray__",):
                path = marker[1]
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"{self.__class__.__name__}.__setstate__: disk-backed "
                        f"frequency-domain strain for {state.get('name')} is "
                        f"missing at {path} (was the array_cache_dir cleaned "
                        "up between runs?).")
                strain_data.__dict__["_frequency_domain_strain"] = np.load(
                    path, mmap_mode='r')
        self.__dict__.update(state)

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
            finite_size=True, wave_frame=None, orbital_motion=True,
            center=None, geocentre_delay=None, split=False):
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
            Coalescence time at the center (GPS seconds); with the default
            center, at the geocentre.
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
        wave_frame: tuple, optional
            The output of :func:`compute_wave_frame` for these arguments, if already
            computed (e.g. for another detector); computed here otherwise.
        orbital_motion: bool
            Include the orbital motion of the Earth in the position of the
            detector, see :mod:`bilby_xG.motion`.
        center: bilby_xG.motion.Center, optional
            The point ``time`` refers to; the geocentre by default.
        geocentre_delay: array_like, optional
            The output of :func:`bilby_xG.motion.geocentre_delay` for these
            arguments, if already computed; computed here otherwise.
        split: bool
            Return the arrival-time phase separately (see below).

        Returns
        =======
        (fps, fcs): tuple of array_like
            Complex plus and cross antenna response at each frequency.
            With ``split``, ``(fps, fcs, delay_phase)``: the beam patterns
            without the arrival-time phase, and that phase
            :math:`e^{-2 \\pi i f t_{\\rm ifo}(f)}`, so that the response is
            ``fps * delay_phase``, ``fcs * delay_phase``.

        Notes
        =====
        Only the plus and cross modes are computed. The arrival-time delay
        :math:`-\\hat{n}\\cdot[\\mathbf{R}_d(t_f) - \\mathbf{C}]/c` of the
        detector at its barycentric position :math:`\\mathbf{R}_d` relative
        to the center :math:`\\mathbf{C}` is incorporated directly in the
        returned beam patterns; it is the sum of the rotating-vertex term
        (from ``omegas``, the source direction in Earth-fixed axes) and the
        detector-independent :func:`bilby_xG.motion.geocentre_delay`.
        """
        if propagation is None:
            propagation = Propagation()

        if wave_frame is None:
            omegas, pol_plus, pol_cross = compute_wave_frame(
                ra, dec, time, psi, frequencies, times_to_coalescence,
                earth_rotation=earth_rotation_time_delay or earth_rotation_beam_patterns)
        else:
            omegas, pol_plus, pol_cross = wave_frame

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

        # arrival-time delay at the detector's position, relative to the
        # center, at the propagation (group) velocity
        if earth_rotation_time_delay:
            vertex_distance = omegas.T @ self.geometry.vertex
        else:
            # the vertex held at its position at coalescence
            gmst = greenwich_mean_sidereal_time(time)
            vertex_distance = sky_direction(ra - gmst, dec) @ self.geometry.vertex
        if geocentre_delay is None:
            geocentre_delay = _geocentre_delay(
                ra, dec, time, times_to_coalescence, center=center,
                orbital_motion=orbital_motion)
        delays = geocentre_delay - vertex_distance / speed_of_light
        ifo_times = time - start_time + delays / propagation.group_velocity(frequencies)

        exp_fac = np.exp(-1j * 2. * np.pi * frequencies * ifo_times)
        if split:
            return fps, fcs, exp_fac
        fps = fps * exp_fac
        fcs = fcs * exp_fac
        return fps, fcs

    def get_detector_response_for_frequency_dependent_antenna_response(
            self, waveform_polarizations, parameters, start_time, frequencies,
            earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
            finite_size=True, shared=None, orbital_motion=True, center=None,
            split=False):
        """Combine waveform polarisations with the frequency-dependent response.

        Handles both the standard ``{"plus": ..., "cross": ...}`` polarisation
        dict and the per-mode nested dict
        ``{mode_key: {"plus": ..., "cross": ...}}`` produced by the
        individual-mode source models. The propagation model is selected from
        ``parameters`` via :func:`bilby_xG.propagation.build_propagation`, so
        a sampled ``vG`` or ``(a, A)`` automatically enables the corresponding
        beyond-GR physics, while their absence recovers general relativity.

        With ``split`` (per-mode polarisations with the twist,
        ``"coprecessing"``, ``"c_plus"``, ``"c_cross"``; see
        :func:`bilby_xG.source.mlgw_bns_precessing_individual_modes`) the
        response of each mode is returned as two factors, ``{mode_key:
        (fast, slow)}``: ``fast`` the co-precessing mode with the
        arrival-time and propagation phases, ``slow`` the beam patterns
        contracted with the twist, ``c_plus * F_+ + c_cross * F_x``. Their
        product is the mode's response. This is the split of scheme 1 of
        Leslie, Dai & Pratten 2021 (arXiv:2109.09872).

        ``shared``, if given, is a dict in which the detector-independent
        part of the response (time to coalescence, propagation phase and
        :func:`compute_wave_frame` and :func:`bilby_xG.motion.geocentre_delay`,
        per azimuthal mode number) is stored, and reused by later calls for
        other detectors. The caller must pass a fresh dict whenever
        ``parameters``, ``frequencies``, ``center`` or the Earth-rotation/
        orbital-motion options change.

        The arrival time follows the barycentric position of the detector
        (see :mod:`bilby_xG.motion`): ``orbital_motion`` includes the Earth's
        orbit in it (the detector-frame masses are then those of the
        Solar-System-barycentre frame), and ``parameters['geocent_time']`` is
        the arrival time at ``center`` (a :class:`bilby_xG.motion.Center`, or
        anything :func:`bilby_xG.motion.resolve_center` accepts; the
        geocentre by default).

        Note: the calibration model is not applied here; only plus and cross
        modes are used.
        """
        propagation = build_propagation(parameters)
        center = resolve_center(center, [self])
        if shared is None:
            shared = {}
        earth_rotation = earth_rotation_time_delay or earth_rotation_beam_patterns

        def response(mode, polarizations, split=False):
            if mode not in shared:
                times_to_coalescence = calculate_time_to_merger_for_any_mode(
                    frequencies, parameters['mass_1'], parameters['mass_2'],
                    parameters['chi_1'], parameters['chi_2'], mode=mode, safety=1)
                shared[mode] = (
                    times_to_coalescence,
                    np.exp(1j * propagation.propagation_phase(frequencies, mode=mode)),
                    compute_wave_frame(
                        parameters['ra'], parameters['dec'],
                        parameters['geocent_time'], parameters['psi'],
                        frequencies, times_to_coalescence,
                        earth_rotation=earth_rotation),
                    _geocentre_delay(
                        parameters['ra'], parameters['dec'],
                        parameters['geocent_time'], times_to_coalescence,
                        center=center, orbital_motion=orbital_motion))
            times_to_coalescence, correction_factor, frame, geocentre_delay = shared[mode]
            factors = self.frequency_dependent_antenna_response(
                parameters['ra'], parameters['dec'], parameters['geocent_time'],
                parameters['psi'],
                times_to_coalescence=times_to_coalescence,
                propagation=propagation,
                frequencies=frequencies,
                start_time=start_time,
                earth_rotation_time_delay=earth_rotation_time_delay,
                finite_size=finite_size,
                earth_rotation_beam_patterns=earth_rotation_beam_patterns,
                wave_frame=frame,
                orbital_motion=orbital_motion,
                center=center,
                geocentre_delay=geocentre_delay,
                split=split,
            )
            if split:
                fps, fcs, delay_phase = factors
                return (correction_factor * delay_phase * polarizations['coprecessing'],
                        polarizations['c_plus'] * fps + polarizations['c_cross'] * fcs)
            fps, fcs = factors
            return correction_factor * (
                polarizations['plus'] * fps + polarizations['cross'] * fcs)

        if split:
            return {mode_key: response(_mode_integer(mode_key), polarizations, split=True)
                    for mode_key, polarizations in waveform_polarizations.items()}
        if 'plus' in waveform_polarizations.keys():
            return response(2, waveform_polarizations)
        signal_ifo = np.zeros(len(frequencies), dtype=complex)
        for mode_key in waveform_polarizations.keys():
            signal_ifo += response(_mode_integer(mode_key), waveform_polarizations[mode_key])
        return signal_ifo

    def time_delay_from_geocenter(self, ra, dec, time, vG=1):
        """Detector time delay from geocentre, optionally rescaled by ``vG``.

        Backward compatible with bilby: ``vG=1`` reproduces the upstream result
        exactly. ``vG`` (the speed of gravity as a fraction of ``c``) rescales
        the propagation speed, so the delay -- which scales as
        distance / speed -- is the geometric delay divided by ``vG``.
        """
        return _cython_time_delay(self.geometry.vertex, ra, dec, time) / vG
