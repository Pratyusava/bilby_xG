# Licensed under an MIT style license -- see LICENSE

"""Gravitational-wave transient likelihoods for next-generation detectors.

All likelihoods project the waveform through
:meth:`bilby_xG.interferometer.Interferometer.get_detector_response_for_frequency_dependent_antenna_response`,
the frequency-dependent, finite-size, Earth-rotation-aware antenna response
needed for next-generation detectors (Cosmic Explorer, Einstein Telescope).
The interferometers must therefore be
:class:`bilby_xG.interferometer.Interferometer` instances, e.g. built with
:class:`bilby_xG.networks.InterferometerList`. The beyond-GR propagation
model (general relativity, speed of gravity ``vG``, or modified dispersion
``a, A``) is selected automatically from the sampled parameters.

Provides:

* :class:`GravitationalWaveTransientNextGeneration` -- the standard transient
  likelihood with the frequency-dependent response,
* :class:`MBGravitationalWaveTransientNextGeneration` -- its multi-banded
  variant (S. Morisaki 2021, arXiv:2104.07813),
* :class:`RelativeBinningGravitationalWaveTransientNextGeneration` -- relative
  binning (Zackay et al. 2018, arXiv:1806.08792), summing modes onto a single
  waveform, and
* :class:`RelativeBinningGravitationalWaveTransientNextGenerationModebyMode` --
  a relative-binning variant that keeps the per-mode decomposition through the
  binning summary data (Leslie, Dai & Pratten 2021, arXiv:2109.09872), with
  bins chosen either by the closed-form arXiv:1806.08792 prescription or,
  optionally, that paper's adaptive bin-selection algorithm.
"""
import inspect
import tempfile
import time

import numpy as np
from scipy.optimize import differential_evolution

from bilby.core.utils import logger
from bilby.core.prior import Constraint, DeltaFunction
from bilby.gw.likelihood import (
    GravitationalWaveTransient,
    MBGravitationalWaveTransient,
    RelativeBinningGravitationalWaveTransient,
)
from bilby.gw.utils import noise_weighted_inner_product

from .injection import complex_gaussian
from .motion import resolve_center

# bilby >= 2.8 passes ``parameters`` explicitly through log_likelihood_ratio
# and calculate_snrs, and warns (FutureWarning) on every read of the
# deprecated ``likelihood.parameters`` state. Older bilby only has the state.
_PARAMETERS_AS_ARGUMENT = "parameters" in inspect.signature(
    GravitationalWaveTransient.log_likelihood_ratio).parameters


def _resolve_parameters(likelihood, parameters):
    """Use the explicitly passed ``parameters``; fall back to the legacy
    ``likelihood.parameters`` state only when none were given (bilby < 2.8
    callers)."""
    if parameters is None:
        return likelihood.parameters
    return parameters


def _log_likelihood_ratio_at(likelihood, parameters):
    """Evaluate ``likelihood.log_likelihood_ratio`` at ``parameters``
    without touching the deprecated parameter state when bilby supports it."""
    if _PARAMETERS_AS_ARGUMENT:
        return likelihood.log_likelihood_ratio(parameters=dict(parameters))
    likelihood.parameters.update(parameters)
    return likelihood.log_likelihood_ratio()

__author__ = [
    "Pratyusava Baral <pbaral@uwm.edu>",
    "Soichiro Morisaki",
    "Ish Gupta",
]

_DEFAULT_GAMMA = np.array([-5 / 3, -2 / 3, 1, 5 / 3, 7 / 3])


#: The parameters the detector-independent part of the response depends on
#: (sky position, time, polarisation, the time-to-coalescence masses and spins,
#: and the propagation model's parameters).
_SHARED_RESPONSE_KEYS = (
    "ra", "dec", "geocent_time", "psi", "mass_1", "mass_2", "chi_1", "chi_2",
    "vG", "a", "A", "luminosity_distance", "H0")


def _clear_waveform_cache(waveform_generator):
    """Forget the generator's last waveform: bilby's cache is keyed by the
    parameters alone, so after a change of its waveform arguments (bin edges,
    return_twist) it would return the waveform made with the old ones."""
    if hasattr(waveform_generator, "_cache"):
        waveform_generator._cache = dict(parameters=None, waveform=None, model=None)


def split_scheme_coefficients(ratio, slow, bin_widths):
    """The per-bin coefficients of scheme 1 of Leslie, Dai & Pratten 2021
    (arXiv:2109.09872): ``ratio`` and ``slow`` given at the bin edges (last
    axis) are each linearised in every bin, as ``r0 + r1 (f - f_c)`` and
    ``c0 + c1 (f - f_c)``, and their product is kept to first order,
    ``r0 c0 + (r0 c1 + r1 c0) (f - f_c)``. Returns its two coefficients, which
    take the place of the ratio's in the summary-data sums. Works on numpy
    and JAX arrays."""
    r0 = 0.5 * (ratio[..., 1:] + ratio[..., :-1])
    r1 = (ratio[..., 1:] - ratio[..., :-1]) / bin_widths
    c0 = 0.5 * (slow[..., 1:] + slow[..., :-1])
    c1 = (slow[..., 1:] - slow[..., :-1]) / bin_widths
    return r0 * c0, r0 * c1 + r1 * c0


def relative_binning_bin_freqs(frequency_array, minimum_frequency, maximum_frequency,
                                chi=1, epsilon=0.5, gamma=None):
    """Relative-binning frequency bin edges (Zackay et al. 2018, arXiv:1806.08792).

    A pure function of the candidate frequency grid and the search band --
    no waveform generator, interferometer or likelihood needed.
    ``d_phi`` below is a closed-form function of frequency (evaluated
    pointwise at each ``frequency_array`` entry, not an integral or
    cumulative sum over it), so ``frequency_array`` only has to be dense
    enough to place the returned bin edges accurately; it need not be a
    real data segment's true (uniform-``df``) frequency resolution -- e.g. a
    log-spaced grid works fine for exploring how ``epsilon``/``chi`` affect
    the bin count and placement.

    Returns
    -------
    bin_freqs: ndarray
        The bin-edge frequencies (``len(bin_freqs) - 1`` bins).
    """
    if gamma is None:
        gamma = _DEFAULT_GAMMA
    gamma = gamma[:, np.newaxis]

    frequency_array_useful = frequency_array[
        (frequency_array >= minimum_frequency)
        & (frequency_array <= maximum_frequency)
    ]

    d_alpha = chi * 2 * np.pi / np.abs(
        (minimum_frequency ** gamma) * np.heaviside(-gamma, 1)
        - (maximum_frequency ** gamma) * np.heaviside(gamma, 1)
    )
    d_phi = np.sum(
        np.sign(gamma) * d_alpha * frequency_array_useful ** gamma,
        axis=0
    )
    d_phi_from_start = d_phi - d_phi[0]
    number_of_bins = int(d_phi_from_start[-1] // epsilon)

    bin_edges = np.linspace(0, d_phi_from_start[-1], num=number_of_bins + 1)
    bin_indices = np.searchsorted(d_phi_from_start, bin_edges)
    unique_bin_indices = np.unique(bin_indices)

    return frequency_array_useful[unique_bin_indices]


class GravitationalWaveTransientNextGeneration(GravitationalWaveTransient):
    """Transient likelihood using the frequency-dependent antenna response.

    Identical to :class:`bilby.gw.likelihood.GravitationalWaveTransient` except
    that the waveform is projected onto each detector through the
    frequency-dependent, finite-size, Earth-rotation-aware antenna response
    required for next-generation detectors (Cosmic Explorer / Einstein
    Telescope). The beyond-GR propagation model (general relativity, speed of
    gravity, or modified dispersion) is selected automatically from the sampled
    parameters by the interferometer response method.

    Parameters
    ==========
    interferometers, waveform_generator, time_marginalization,
    distance_marginalization, phase_marginalization, calibration_marginalization,
    priors, distance_marginalization_lookup_table, calibration_lookup_table,
    number_of_response_curves, starting_index, jitter_time, reference_frame,
    time_reference:
        As in :class:`bilby.gw.likelihood.GravitationalWaveTransient`.
    earth_rotation_beam_patterns: bool, optional
        Include the Earth-rotation modulation of the beam patterns. Default True.
    earth_rotation_time_delay: bool, optional
        Include the Earth-rotation modulation of the propagation time delay.
        Default True.
    finite_size: bool, optional
        Include finite-size detector effects. Default True.
    orbital_motion: bool, optional
        Include the Earth's orbit in the motion of the detectors (masses are
        then in the Solar-System-barycentre frame, see
        :mod:`bilby_xG.motion`). Default True.
    center: str, array_like or bilby_xG.motion.Center, optional
        The point the time parameter refers to: ``"geocenter"`` (default),
        an interferometer name, a fixed barycentric position or a
        :class:`~bilby_xG.motion.Center` (e.g. an SNR-weighted average of
        the detectors). See :func:`bilby_xG.motion.resolve_center`.
    """

    # Fallbacks for instances pickled before the detector motion was added:
    # their data were built without the orbital motion, about the geocentre.
    orbital_motion = False
    center = None

    def __init__(self, interferometers, waveform_generator, time_marginalization=False,
                 distance_marginalization=False, phase_marginalization=False,
                 calibration_marginalization=False, priors=None,
                 distance_marginalization_lookup_table=None, calibration_lookup_table=None,
                 number_of_response_curves=1000, starting_index=0, jitter_time=True,
                 reference_frame="sky", time_reference="geocenter",
                 earth_rotation_beam_patterns=True, earth_rotation_time_delay=True,
                 finite_size=True, orbital_motion=True, center=None):

        super().__init__(
            interferometers, waveform_generator, time_marginalization,
            distance_marginalization, phase_marginalization,
            calibration_marginalization, priors,
            distance_marginalization_lookup_table, calibration_lookup_table,
            number_of_response_curves, starting_index, jitter_time, reference_frame,
            time_reference)

        self.earth_rotation_beam_patterns = earth_rotation_beam_patterns
        self.earth_rotation_time_delay = earth_rotation_time_delay
        self.finite_size = finite_size
        self.orbital_motion = orbital_motion
        self.center = resolve_center(center, interferometers)

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters=None):
        """Project the waveform onto the frequency-dependent detector response.

        Parameters
        ==========
        signal_polarizations: dict
            Waveform evaluated at ``interferometer.frequency_array``. Either a
            ``{"plus", "cross"}`` dict or a per-mode nested dict.
        interferometer: bilby.gw.detector.Interferometer
            Interferometer to compute the response with respect to.
        parameters: dict, optional
            Parameters to project with; current bilby core's
            GravitationalWaveTransient.calculate_snrs passes this explicitly
            (its own ``_compute_full_waveform`` takes the same kwarg). Falls
            back to ``self.parameters`` when not given, as before.
        """
        frequencies = interferometer.frequency_array
        idxs_above_minimum_frequency = frequencies > \
            (interferometer.minimum_frequency - (frequencies[1] - frequencies[0]))
        freqs = frequencies[idxs_above_minimum_frequency]
        waveform_polarizations_red = {}

        try:
            waveform_polarizations_red['plus'] = \
                signal_polarizations['plus'][idxs_above_minimum_frequency]
            waveform_polarizations_red['cross'] = \
                signal_polarizations['cross'][idxs_above_minimum_frequency]
        except KeyError:
            for key in signal_polarizations.keys():
                waveform_polarizations_red[key] = {}
                waveform_polarizations_red[key]['plus'] = \
                    signal_polarizations[key]['plus'][idxs_above_minimum_frequency]
                waveform_polarizations_red[key]['cross'] = \
                    signal_polarizations[key]['cross'][idxs_above_minimum_frequency]
        h = np.zeros_like(frequencies, dtype=complex)
        parameters = _resolve_parameters(self, parameters)
        parameters, _ = self.waveform_generator.parameter_conversion(parameters)
        h[idxs_above_minimum_frequency] = \
            interferometer.get_detector_response_for_frequency_dependent_antenna_response(
                waveform_polarizations=waveform_polarizations_red,
                parameters=parameters,
                start_time=interferometer.strain_data.start_time,
                frequencies=freqs,
                earth_rotation_time_delay=self.earth_rotation_time_delay,
                earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
                finite_size=self.finite_size,
                orbital_motion=self.orbital_motion,
                center=self.center)

        return h


class MBGravitationalWaveTransientNextGeneration(MBGravitationalWaveTransient):
    """Multi-banded likelihood with a frequency-dependent antenna response.

    Combines the multi-banding of S. Morisaki (2021), arXiv:2104.07813 with the
    frequency-dependent, finite-size antenna response for next-generation
    detectors. All multi-banding arguments are as in
    :class:`bilby.gw.likelihood.MBGravitationalWaveTransient`; the additional
    arguments below control the response.

    Parameters
    ==========
    earth_rotation_beam_patterns: bool, optional
        Apply the Earth-rotation modulation to the beam patterns. Default True.
    earth_rotation_time_delay: bool, optional
        Apply the Earth-rotation modulation to the propagation delay. Default True.
    finite_size: bool, optional
        Apply finite-size detector effects. Default True.
    orbital_motion: bool, optional
        Include the Earth's orbit in the motion of the detectors (see
        :mod:`bilby_xG.motion`). Default True.
    center: str, array_like or bilby_xG.motion.Center, optional
        The point the time parameter refers to, see
        :func:`bilby_xG.motion.resolve_center`. Default the geocentre.
    response_update: float, optional
        Time interval for updating the detector response. **Not implemented** in
        this release: only the default (``None``, update at every evaluation) is
        supported. Passing a value raises :class:`NotImplementedError`.
    """

    # Fallbacks for instances pickled before the detector motion was added:
    # their data were built without the orbital motion, about the geocentre.
    orbital_motion = False
    center = None

    def __init__(
        self, interferometers, waveform_generator, reference_chirp_mass, highest_mode=2,
        linear_interpolation=True, accuracy_factor=5, time_offset=None, delta_f_end=None,
        maximum_banding_frequency=None, minimum_banding_duration=0.,
        distance_marginalization=False, phase_marginalization=False, priors=None,
        distance_marginalization_lookup_table=None, reference_frame="sky",
        time_reference="geocenter", earth_rotation_beam_patterns=True,
        earth_rotation_time_delay=True, finite_size=True, response_update=None,
        orbital_motion=True, center=None
    ):
        super().__init__(
            interferometers=interferometers, waveform_generator=waveform_generator,
            reference_chirp_mass=reference_chirp_mass, highest_mode=highest_mode,
            linear_interpolation=linear_interpolation, accuracy_factor=accuracy_factor,
            time_offset=time_offset, delta_f_end=delta_f_end,
            maximum_banding_frequency=maximum_banding_frequency,
            minimum_banding_duration=minimum_banding_duration,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            reference_frame=reference_frame, time_reference=time_reference
        )
        self.earth_rotation_beam_patterns = earth_rotation_beam_patterns
        self.earth_rotation_time_delay = earth_rotation_time_delay
        self.finite_size = finite_size
        self.orbital_motion = orbital_motion
        self.center = resolve_center(center, interferometers)
        if response_update is not None:
            raise NotImplementedError(
                "response_update is not supported; the detector response is "
                "updated at every waveform evaluation (response_update=None)."
            )
        self.response_update = response_update

    def calculate_snrs(self, waveform_polarizations, interferometer, parameters=None):
        """Compute the SNRs for multi-banding with the frequency-dependent response.

        Parameters
        ==========
        waveform_polarizations: dict
            Waveform at the banded frequency points (plus/cross or per-mode).
        interferometer: bilby.gw.detector.Interferometer

        Returns
        =======
        snrs: named tuple of SNRs
        """
        parameters = _resolve_parameters(self, parameters)
        converted_parameters, _ = self.waveform_generator.parameter_conversion(parameters)
        waveform_polarizations_red = {}
        try:
            waveform_polarizations_red['plus'] = \
                waveform_polarizations['plus'][self.unique_to_original_frequencies]
            waveform_polarizations_red['cross'] = \
                waveform_polarizations['cross'][self.unique_to_original_frequencies]
        except KeyError:
            for key in waveform_polarizations.keys():
                waveform_polarizations_red[key] = {}
                waveform_polarizations_red[key]['plus'] = \
                    waveform_polarizations[key]['plus'][self.unique_to_original_frequencies]
                waveform_polarizations_red[key]['cross'] = \
                    waveform_polarizations[key]['cross'][self.unique_to_original_frequencies]
        h = interferometer.get_detector_response_for_frequency_dependent_antenna_response(
            waveform_polarizations=waveform_polarizations_red,
            parameters=converted_parameters,
            start_time=interferometer.strain_data.start_time,
            frequencies=self.banded_frequency_points,
            earth_rotation_time_delay=self.earth_rotation_time_delay,
            earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
            finite_size=self.finite_size,
            orbital_motion=self.orbital_motion,
            center=self.center)

        d_inner_h = np.dot(h, self.linear_coeffs[interferometer.name])

        if self.linear_interpolation:
            optimal_snr_squared = np.vdot(
                np.real(h * np.conjugate(h)), self.quadratic_coeffs[interferometer.name])
        else:
            optimal_snr_squared = 0.
            for b in range(len(self.fb_dfb) - 1):
                Ks, Ke = self.Ks_Ke[b]
                start_idx, end_idx = self.start_end_idxs[b]
                Mb = self.Mbs[b]
                if b == 0:
                    optimal_snr_squared += (4. / self.interferometers.duration) * np.vdot(
                        np.real(h[start_idx:end_idx + 1] * np.conjugate(h[start_idx:end_idx + 1])),
                        interferometer.frequency_mask[Ks:Ke + 1] * self.windows[start_idx:end_idx + 1]
                        / interferometer.power_spectral_density_array[Ks:Ke + 1])
                else:
                    self.wths[interferometer.name][b][Ks:Ke + 1] = \
                        self.square_root_windows[start_idx:end_idx + 1] * h[start_idx:end_idx + 1]
                    self.hbcs[interferometer.name][b][-Mb:] = np.fft.irfft(self.wths[interferometer.name][b])
                    thbc = np.fft.rfft(self.hbcs[interferometer.name][b])
                    optimal_snr_squared += (4. / self.Tbhats[b]) * np.vdot(
                        np.real(thbc * np.conjugate(thbc)), self.Ibcs[interferometer.name][b])

        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=None,
            optimal_snr_squared_array=None)


class RelativeBinningGravitationalWaveTransientNextGeneration(RelativeBinningGravitationalWaveTransient):
    """A gravitational-wave transient likelihood object capaple of handling
    frequency-dependent antenna responses which uses the relative
    binning procedure to calculate a fast likelihood. See Zackay et al.
    arXiv1806.08792

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    fiducial_parameters: dict, optional
        A starting guess for initial parameters of the event for finding the
        maximum likelihood (fiducial) waveform. These should be specified in
        the same parameter basis as the one that sampling is carried out in.
        For example, if sampling in `mass_1` and `mass_2`, the fiducial
        parameters should also be provided in `mass_1` and `mass_2.`
    parameter_bounds: dict, optional
        Dictionary of bounds (lists) for the initial parameters when finding
        the initial maximum likelihood (fiducial) waveform.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT to calculate the likelihood over a regularly spaced
        grid.
        In order to cover the whole space the prior is set to be uniform over
        the spacing of the array of times.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
        This is done analytically using a Bessel function.
        The phase prior is set to be a delta function at phase=0.
    priors: dict, optional
        If given, used in the distance and phase marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is False, however using this parameter is strongly encouraged.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center,
            this is the default
        - e.g., "H1": sample in the time of arrival at H1
    chi: float, optional
        Tunable parameter which limits the perturbation of alpha when setting
        up the bin range. See https://arxiv.org/abs/1806.08792.
    epsilon: float, optional
        Tunable parameter which limits the differential phase change in each
        bin when setting up the bin range. See https://arxiv.org/abs/1806.08792.
    earth_rotation_beam_patterns: bool, optional
        If true, the beam patterns are rotated with the Earth. Default is True.
    earth_rotation_time_delay: bool, optional
        If true, the time delay is rotated with the Earth. Default is True.
    finite_size: bool, optional
        If true, the finite size effect is included. Default is True.
    orbital_motion: bool, optional
        If true, the Earth's orbit is included in the motion of the detectors
        (see :mod:`bilby_xG.motion`). Default is True.
    center: str, array_like or bilby_xG.motion.Center, optional
        The point the time parameter refers to, see
        :func:`bilby_xG.motion.resolve_center`. Default the geocentre.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    The relative binning likelihood does not currently support calibration marginalization.
    """

    # Fallbacks for instances pickled before the detector motion was added:
    # their data were built without the orbital motion, about the geocentre.
    orbital_motion = False
    center = None

    def __init__(
            self, interferometers,
            waveform_generator,
            fiducial_parameters=None,
            parameter_bounds=None,
            maximization_kwargs=None,
            update_fiducial_parameters=False,
            distance_marginalization=False,
            time_marginalization=False,
            phase_marginalization=False,
            priors=None,
            distance_marginalization_lookup_table=None,
            jitter_time=True,
            reference_frame="sky",
            time_reference="geocenter",
            chi=1,
            epsilon=0.5,
            earth_rotation_beam_patterns=True,
            earth_rotation_time_delay=True,
            finite_size=True,
            orbital_motion=True,
            center=None):

        self.earth_rotation_beam_patterns = earth_rotation_beam_patterns
        self.earth_rotation_time_delay = earth_rotation_time_delay
        self.finite_size = finite_size
        self.orbital_motion = orbital_motion
        self.center = resolve_center(center, interferometers)

        super(RelativeBinningGravitationalWaveTransientNextGeneration, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            fiducial_parameters=fiducial_parameters,
            parameter_bounds=parameter_bounds,
            maximization_kwargs=maximization_kwargs,
            update_fiducial_parameters=update_fiducial_parameters,
            distance_marginalization=distance_marginalization,
            time_marginalization=time_marginalization,
            phase_marginalization=phase_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
            chi=chi,
            epsilon=epsilon)

    def set_fiducial_waveforms(self, parameters):
        parameters = parameters.copy()
        parameters["fiducial"] = 1
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        maximum_nonzero_index = np.where(self.fiducial_polarizations["plus"] != 0j)[0][-1]
        logger.debug(f"Maximum Nonzero Index is {maximum_nonzero_index}")
        maximum_nonzero_frequency = self.waveform_generator.frequency_array[maximum_nonzero_index]
        logger.debug(f"Maximum Nonzero Frequency is {maximum_nonzero_frequency}")
        self.maximum_frequency = maximum_nonzero_frequency

        if self.fiducial_polarizations is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")

        for interferometer in self.interferometers:
            logger.debug(f"Maximum Frequency is {interferometer.maximum_frequency}")
            converted_parameters, _ = self.waveform_generator.parameter_conversion(parameters)

            frequencies = interferometer.frequency_array
            idxs_above_minimum_frequency = frequencies > \
                (interferometer.minimum_frequency - (frequencies[1] - frequencies[0]))
            freqs = frequencies[idxs_above_minimum_frequency]

            waveform_polarizations_reduced = {}
            waveform_polarizations_reduced['plus'] = self.fiducial_polarizations['plus'][idxs_above_minimum_frequency]
            waveform_polarizations_reduced['cross'] = self.fiducial_polarizations['cross'][idxs_above_minimum_frequency]

            wf = np.zeros_like(interferometer.frequency_array, dtype=complex)
            wf[idxs_above_minimum_frequency] = \
                interferometer.get_detector_response_for_frequency_dependent_antenna_response(
                waveform_polarizations=waveform_polarizations_reduced,
                parameters=converted_parameters,
                start_time=interferometer.strain_data.start_time,
                frequencies=freqs,
                earth_rotation_time_delay=self.earth_rotation_time_delay,
                earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
                finite_size=self.finite_size,
                orbital_motion=self.orbital_motion,
                center=self.center)

            wf[frequencies > self.maximum_frequency] = 0
            self.per_detector_fiducial_waveforms[interferometer.name] = wf

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer, parameters=None):
        name = interferometer.name
        parameters = _resolve_parameters(self, parameters)
        converted_parameters, _ = self.waveform_generator.parameter_conversion(parameters)
        strain = interferometer.get_detector_response_for_frequency_dependent_antenna_response(
            waveform_polarizations=waveform_polarizations,
            parameters=converted_parameters,
            start_time=interferometer.strain_data.start_time,
            frequencies=self.bin_freqs,
            earth_rotation_time_delay=self.earth_rotation_time_delay,
            earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
            finite_size=self.finite_size,
            orbital_motion=self.orbital_motion,
            center=self.center)

        reference_strain = self.per_detector_fiducial_waveform_points[name]
        waveform_ratio = strain / reference_strain
        r0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
        r1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_widths

        return [r0, r1]


class RelativeBinningGravitationalWaveTransientNextGenerationModebyMode(GravitationalWaveTransient):
    """
    This modifies code written by Ish Gupta and others to implement the
    relative binning likelihood for the HM precessing model.
    """

    """A gravitational-wave transient likelihood object which uses the relative
    binning procedure to calculate a fast likelihood. See Zackay et al.
    arXiv1806.08792

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    fiducial_parameters: dict, optional
        A starting guess for initial parameters of the event for finding the
        maximum likelihood (fiducial) waveform. These should be specified in
        the same parameter basis as the one that sampling is carried out in.
        For example, if sampling in `mass_1` and `mass_2`, the fiducial
        parameters should also be provided in `mass_1` and `mass_2.`
    parameter_bounds: dict, optional
        Dictionary of bounds (lists) for the initial parameters when finding
        the initial maximum likelihood (fiducial) waveform.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT to calculate the likelihood over a regularly spaced
        grid.
        In order to cover the whole space the prior is set to be uniform over
        the spacing of the array of times.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
        This is done analytically using a Bessel function.
        The phase prior is set to be a delta function at phase=0.
    priors: dict, optional
        If given, used in the distance and phase marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is False, however using this parameter is strongly encouraged.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center,
          this is the default
        - e.g., "H1": sample in the time of arrival at H1
    chi: float, optional
        Tunable parameter which limits the perturbation of alpha when setting
        up the bin range. See https://arxiv.org/abs/1806.08792.
    epsilon: float, optional
        Tunable parameter which limits the differential phase change in each
        bin when setting up the bin range. See https://arxiv.org/abs/1806.08792.
    bin_selection_test_parameters: dict, optional
        If given, frequency bins are instead chosen by the adaptive
        mode-by-mode relative-binning bin-selection algorithm (Algorithm 1
        of Leslie, Dai & Pratten 2021, https://arxiv.org/abs/2109.09872),
        using this parameter set (which should differ from
        ``fiducial_parameters`` -- e.g. another sample from the region of
        parameter space favoured by the data) to probe the accuracy of
        candidate bins. This adapts the bin placement to the signal under
        analysis and typically needs far fewer bins than ``chi``/``epsilon``
        for comparable accuracy. If not given, bins follow the closed-form
        ``chi``/``epsilon`` prescription instead.
    bin_selection_eta: float, optional
        Only used when ``bin_selection_test_parameters`` is given. Target
        total absolute error in the log-likelihood ratio from mode-by-mode
        relative binning, evaluated at the test parameters. The paper finds
        0.1 is small enough not to noticeably affect the posterior. Default
        0.1.
    bin_selection_target_number_of_bins: int, optional
        Only used when ``bin_selection_test_parameters`` is given. Initial
        guess for the number of bins, used to set the per-bin error budget
        ``bin_selection_eta / target_number_of_bins`` before the algorithm's
        outer loop converges on the actual number of bins needed. Default
        200, as in the paper.
    bin_freqs: array_like, optional
        Precomputed bin-edge frequencies, e.g. from
        :func:`bilby_xG.bin_selection.select_relative_binning_bins`. They are
        snapped to the data's frequency grid and clipped to the band, whose
        edges are always included. Takes precedence over
        ``bin_selection_test_parameters`` and ``chi``/``epsilon``.
    injection: bilby_xG.injection.SummaryDataInjection, optional
        Simulated data: the summary data are built from this injection's
        signal and a Gaussian-noise draw (or zero noise), and the
        interferometers get the data grid but no strain. Without it the
        summary data come from the interferometers' strain.
    precessing_scheme: str, optional
        How the modes of a precessing source (keyed by co-precessing
        multipole, see
        :func:`bilby_xG.source.mlgw_bns_precessing_individual_modes`) are
        linearised in each bin, in the terms of Leslie, Dai & Pratten 2021
        (arXiv:2109.09872). ``"ratio"`` (the default, their scheme 2, the
        only one for other sources): the ratio of each mode's detector
        response to the fiducial's. ``"split"`` (their scheme 1): the
        response of each mode is ``fast * slow`` (see ``split`` in
        :meth:`~bilby_xG.interferometer.Interferometer.get_detector_response_for_frequency_dependent_antenna_response`);
        the ratio of ``fast`` to the fiducial's and ``slow`` itself are
        linearised separately, the summary data are those of the fiducial
        ``fast``, and their product is kept to first order in the
        frequency offset from the bin centre. Its reference does not vanish
        where the fiducial's twist and beam patterns do. The bins are
        chosen with scheme 2 either way.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    The relative binning likelihood does not currently support calibration marginalization.
    """

    # Fallbacks for instances pickled before the detector motion was added:
    # their data were built without the orbital motion, about the geocentre.
    orbital_motion = False
    center = None
    precessing_scheme = "ratio"

    def __init__(self, interferometers,
                 waveform_generator,
                 fiducial_parameters=None,
                 parameter_bounds=None,
                 maximization_kwargs=None,
                 update_fiducial_parameters=False,
                 distance_marginalization=False,
                 time_marginalization=False,
                 phase_marginalization=False,
                 priors=None,
                 distance_marginalization_lookup_table=None,
                 jitter_time=True,
                 reference_frame="sky",
                 time_reference="geocenter",
                 mode_array=[[2, 2], [3, 3], [4, 4], [2, 1], [3, 2]],  # FIXME
                 chi=1,
                 epsilon=0.5,
                 bin_selection_test_parameters=None,
                 bin_selection_eta=0.1,
                 bin_selection_target_number_of_bins=200,
                 bin_freqs=None,
                 earth_rotation_time_delay=True,
                 earth_rotation_beam_patterns=True,
                 finite_size=True,
                 orbital_motion=True,
                 center=None,
                 summary_data_chunk_size=2 ** 20,
                 array_cache_dir=None,
                 injection=None,
                 precessing_scheme="ratio"):

        if precessing_scheme not in ("ratio", "split"):
            raise ValueError(f"unknown precessing_scheme {precessing_scheme!r}")
        self.precessing_scheme = precessing_scheme
        if precessing_scheme == "split":
            # the source gives each mode's co-precessing part and twist
            waveform_generator.waveform_arguments["return_twist"] = True
            _clear_waveform_cache(waveform_generator)
        if injection is not None:
            if time_marginalization:
                raise ValueError(
                    "time marginalization needs the full data, which an "
                    "injection as summary data does not have")
            if update_fiducial_parameters and injection.noise:
                raise ValueError(
                    "the noise realisation is drawn for fixed fiducial "
                    "waveforms and bins, so they cannot be updated")
            injection.setup_interferometers(interferometers)
        self.injection = injection

        super(RelativeBinningGravitationalWaveTransientNextGenerationModebyMode, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference)

        if fiducial_parameters is None:
            logger.info("Drawing fiducial parameters from prior.")
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = fiducial_parameters.copy()
        self.fiducial_parameters["fiducial"] = 0
        if self.time_marginalization:
            self.fiducial_parameters["geocent_time"] = interferometers.start_time
        if self.distance_marginalization:
            self.fiducial_parameters["luminosity_distance"] = self._ref_dist
        if self.phase_marginalization:
            self.fiducial_parameters["phase"] = 0.0
        if "mode_array" in waveform_generator.waveform_arguments.keys():
            self.mode_array = waveform_generator.waveform_arguments["mode_array"]
        else:
            self.mode_array = mode_array
        self.fiducial_likelihood = 0.
        self.chi = chi
        self.epsilon = epsilon
        self.gamma = np.array([-5 / 3, -2 / 3, 1, 5 / 3, 7 / 3])
        self.bin_selection_test_parameters = bin_selection_test_parameters
        self.bin_selection_eta = bin_selection_eta
        self.bin_selection_target_number_of_bins = bin_selection_target_number_of_bins
        self.precomputed_bin_freqs = None if bin_freqs is None else np.asarray(bin_freqs)
        self.maximum_frequency = waveform_generator.frequency_array[-1]
        self.fiducial_waveform_obtained = False
        self.check_if_bins_are_setup = False
        self.earth_rotation_time_delay = earth_rotation_time_delay
        self.earth_rotation_beam_patterns = earth_rotation_beam_patterns
        self.finite_size = finite_size
        self.orbital_motion = orbital_motion
        self.center = resolve_center(center, interferometers)
        # Frequency samples processed at once when building the summary data.
        # Keeping this well below the full band means the full-resolution
        # fiducial waveform never has to be held in memory at once.
        self.summary_data_chunk_size = summary_data_chunk_size
        if array_cache_dir is None and injection is None:
            array_cache_dir = tempfile.mkdtemp(prefix="bilby_xG_array_cache_")
        self.array_cache_dir = array_cache_dir
        if injection is None:
            logger.info(
                f"Disk-backed full-band array cache: {self.array_cache_dir} "
                "(must remain until this likelihood is no longer used).")
        self._fiducial_converted_parameters = None
        # self.per_detector_fiducial_waveforms = dict()
        # self.per_detector_fiducial_waveform_points = dict()
        # Only the fiducial waveform sampled at the bin edges is kept; the
        # full-resolution fiducial waveform is re-projected in chunks on demand.
        self.per_detector_per_mode_fiducial_waveform_points = dict()
        self.bin_freqs = dict()
        self.bin_inds = dict()
        self.bin_widths = dict()
        self.bin_centers = dict()
        self.set_fiducial_waveforms(self.fiducial_parameters)
        logger.info("Initial fiducial waveforms set up")
        self.setup_bins()
        self.compute_summary_data()
        logger.info("Summary Data Obtained")

        if update_fiducial_parameters:
            # write a check to make sure prior is not None
            logger.info("Using scipy optimization to find maximum likelihood parameters.")
            self.parameters_to_be_updated = [key for key in priors if not isinstance(
                priors[key], (DeltaFunction, Constraint, float, int))]
            logger.info(f"Parameters over which likelihood is maximized: {self.parameters_to_be_updated}")
            if parameter_bounds is None:
                logger.info("No parameter bounds were given. Using priors instead.")
                self.parameter_bounds = self.get_bounds_from_priors(priors)
            else:
                self.parameter_bounds = self.get_parameter_list_from_dictionary(parameter_bounds)
            self.fiducial_parameters = self.find_maximum_likelihood_parameters(
                self.parameter_bounds, maximization_kwargs=maximization_kwargs)
        self.fiducial_likelihood = _log_likelihood_ratio_at(self, self.fiducial_parameters)
        logger.info(f"Fiducial likelihood: {self.fiducial_likelihood:.2f}")
        # Legacy parameter state for bilby < 2.8, where the sampler updates
        # ``self.parameters`` in place rather than passing parameters.
        if _PARAMETERS_AS_ARGUMENT:
            self._parameters = dict(fiducial=0)
        else:
            self.parameters = dict(fiducial=0)

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={},\n\fiducial_parameters={},' \
            .format(self.interferometers, self.waveform_generator, self.fiducial_parameters)

    def setup_bins(self):
        """
        Setup the frequency bins.

        If ``self.bin_selection_test_parameters`` is set, bins are instead
        chosen adaptively by the mode-by-mode relative-binning bin-selection
        algorithm (Algorithm 1 of Leslie, Dai & Pratten 2021,
        https://arxiv.org/abs/2109.09872, "GETBINS"/"BISECTBINSEARCH"): the
        full frequency range is recursively bisected, starting as a single
        candidate bin, until every candidate bin's contribution to the
        log-likelihood-ratio error (this class's own mode-by-mode
        relative-binning approximation compared to the exact per-frequency
        sum, both evaluated at the test parameters) falls below a shrinking
        per-bin budget. See :meth:`mode_by_mode_bin_freqs`.

        Otherwise, bins follow the closed-form post-Newtonian-phase
        prescription of https://arxiv.org/abs/1806.08792. If :code:`epsilon`
        is too small, the naive bins can be smaller than the frequency
        spacing of the data. We require that bins are at least as wide as
        this spacing.
        """
        frequency_array = self.waveform_generator.frequency_array
        # Bin over the intersection of the interferometer frequency ranges,
        # matching bilby's RelativeBinningGravitationalWaveTransient.setup_bins.
        minimum_frequency = np.maximum.reduce(
            [ifo.minimum_frequency for ifo in self.interferometers], initial=0)
        maximum_frequency = np.minimum.reduce(
            [ifo.maximum_frequency for ifo in self.interferometers], initial=frequency_array[-1])
        maximum_frequency = min(maximum_frequency, self.maximum_frequency)

        if self.precomputed_bin_freqs is not None:
            ref_freqs = frequency_array[
                (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)]
            inner = self.precomputed_bin_freqs[
                (self.precomputed_bin_freqs > ref_freqs[0])
                & (self.precomputed_bin_freqs < ref_freqs[-1])]
            indices = np.unique(np.concatenate(
                [[0], np.searchsorted(ref_freqs, inner), [len(ref_freqs) - 1]]))
            # an inner edge right at the band's end (the given edges may end
            # slightly below this likelihood's maximum frequency) would leave
            # a sliver bin: merge it into its neighbour
            if len(indices) > 2 and ref_freqs[indices[-1]] - ref_freqs[indices[-2]] < 0.01 * (
                    ref_freqs[indices[-2]] - ref_freqs[indices[-3]]):
                indices = np.delete(indices, -2)
            if len(indices) > 2 and ref_freqs[indices[1]] - ref_freqs[indices[0]] < 0.01 * (
                    ref_freqs[indices[2]] - ref_freqs[indices[1]]):
                indices = np.delete(indices, 1)
            bin_freqs = ref_freqs[indices]
            selection_description = (
                f"precomputed, {len(self.precomputed_bin_freqs) - 1} bins given")
        elif self.bin_selection_test_parameters is not None:
            bin_freqs = self.mode_by_mode_bin_freqs(
                self.bin_selection_test_parameters, frequency_array,
                minimum_frequency, maximum_frequency)
            selection_description = (
                f"mode-by-mode adaptive bin selection, eta="
                f"{self.bin_selection_eta}")
        else:
            bin_freqs = relative_binning_bin_freqs(
                frequency_array, minimum_frequency, maximum_frequency,
                chi=self.chi, epsilon=self.epsilon, gamma=self.gamma)
            selection_description = (
                f"epsilon={self.epsilon}, chi={self.chi}, "
                f"gamma={self.gamma.tolist()}")

        bin_inds = np.searchsorted(frequency_array, bin_freqs)
        self.bin_inds = bin_inds
        self.bin_freqs = bin_freqs
        self.number_of_bins = len(bin_inds) - 1

        logger.info(
            f"Constructed {self.number_of_bins} relative-binning bins over "
            f"[{minimum_frequency:.3g}, {maximum_frequency:.3g}] Hz "
            f"({selection_description}).")

        self.waveform_generator.waveform_arguments["frequency_bin_edges"] = self.bin_freqs
        _clear_waveform_cache(self.waveform_generator)
        self.bin_widths = self.bin_freqs[1:] - self.bin_freqs[:-1]
        self.bin_centers = (self.bin_freqs[1:] + self.bin_freqs[:-1]) / 2

        for interferometer in self.interferometers:
            name = interferometer.name
            self.per_detector_per_mode_fiducial_waveform_points[name] = {}
            for mode in self.mode_array:
                mode_key = f"{mode[0]},{mode[1]}"
                self.per_detector_per_mode_fiducial_waveform_points[name][mode_key] = \
                    self._reference_mode(interferometer, mode_key, self.bin_freqs)

    def mode_by_mode_bin_freqs(self, test_parameters, frequency_array,
                                minimum_frequency, maximum_frequency):
        """Adaptive frequency bin edges for mode-by-mode relative binning.

        Implements the bin-selection algorithm (Algorithm 1, "GETBINS") of
        Leslie, Dai & Pratten 2021, https://arxiv.org/abs/2109.09872: starting
        from the full ``[minimum_frequency, maximum_frequency]`` range as a
        single candidate bin, :meth:`_bisect_bin_search` recursively bisects
        candidate bins until each one's log-likelihood-ratio error is below
        ``self.bin_selection_eta / target_number_of_bins``. The resulting
        number of bins is then fed back in as the new target and the process
        repeated until it converges (the number of bins produced equals the
        number assumed to set the per-bin error budget).

        Parameters
        ----------
        test_parameters: dict
            A representative parameter set (e.g. another posterior sample,
            distinct from ``self.fiducial_parameters``) used to probe the
            error of candidate bins. The paper finds the choice of bins is
            not very sensitive to which sample is used.
        frequency_array: ndarray
            The candidate frequency grid to choose bin edges from (a subset
            of it is returned).
        minimum_frequency, maximum_frequency: float
            The frequency range to cover with bins.

        Returns
        -------
        bin_freqs: ndarray
            The bin-edge frequencies (``len(bin_freqs) - 1`` bins).
        """
        test_parameters = test_parameters.copy()
        test_parameters["fiducial"] = 1
        test_parameters.update(self.get_sky_frame_parameters(parameters=test_parameters))
        test_converted_parameters, _ = self.waveform_generator.parameter_conversion(test_parameters)

        ref_freqs = frequency_array[
            (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)]

        eta = self.bin_selection_eta
        target_number_of_bins = self.bin_selection_target_number_of_bins
        indices = [0, len(ref_freqs) - 1]
        previous_number_of_bins = None
        iterations = 0
        while previous_number_of_bins != target_number_of_bins and iterations < 50:
            target_bin_error = eta / target_number_of_bins
            self._bisect_bin_search_calls = 0
            self._bisect_bin_search_last_log = time.time()
            indices = self._bisect_bin_search(
                test_converted_parameters, ref_freqs, 0, len(ref_freqs) - 1, target_bin_error)
            previous_number_of_bins = target_number_of_bins
            target_number_of_bins = len(indices) - 1
            iterations += 1
            logger.info(
                f"Mode-by-mode bin selection: iteration {iterations}, "
                f"{target_number_of_bins} bins (target log-likelihood "
                f"error {eta}, {eta / previous_number_of_bins:.3g} per bin).")

        return ref_freqs[indices]

    def _bisect_bin_search(self, test_converted_parameters, ref_freqs, lo, hi, target_bin_error):
        """"BISECTBINSEARCH" (Algorithm 1 of arXiv:2109.09872): recursively
        bisect the candidate bin spanning ``ref_freqs[lo:hi+1]`` until its
        log-likelihood-ratio error is within ``target_bin_error``, or it
        cannot be split further (it already spans a single frequency sample).
        Returns the list of ``ref_freqs`` indices bounding the resulting
        (possibly many) bins, e.g. ``[lo, m1, m2, hi]``.
        """
        if hi - lo <= 1:
            return [lo, hi]
        error = self._log_likelihood_error_for_candidate_bin(
            test_converted_parameters, ref_freqs[lo], ref_freqs[hi])
        self._bisect_bin_search_calls += 1
        now = time.time()
        if now - self._bisect_bin_search_last_log > 5:
            self._bisect_bin_search_last_log = now
            logger.info(
                f"Mode-by-mode bin selection: {self._bisect_bin_search_calls} "
                f"candidate bins evaluated so far, current candidate "
                f"[{ref_freqs[lo]:.3g}, {ref_freqs[hi]:.3g}] Hz, error={error:.3g} "
                f"(target {target_bin_error:.3g}).")
        if error <= target_bin_error:
            return [lo, hi]
        mid = (lo + hi) // 2
        left = self._bisect_bin_search(test_converted_parameters, ref_freqs, lo, mid, target_bin_error)
        right = self._bisect_bin_search(test_converted_parameters, ref_freqs, mid, hi, target_bin_error)
        return left[:-1] + right

    def _log_likelihood_error_for_candidate_bin(self, test_converted_parameters, f_lo, f_hi):
        """Absolute error in the log-likelihood-ratio contribution from the
        candidate frequency bin ``[f_lo, f_hi]``, summed over interferometers:
        the exact per-frequency sum (both evaluated at
        ``test_converted_parameters``) minus this class's own mode-by-mode
        relative-binning approximation for that single bin (a linear
        interpolant using only the two bin edges). This is the
        "LOGLIKELIHOODERROR" primitive of Algorithm 1 in
        https://arxiv.org/abs/2109.09872.
        """
        total_exact = 0.0
        total_approx = 0.0
        for interferometer in self.interferometers:
            result = self._accumulate_bin_sums(interferometer, test_converted_parameters, f_lo, f_hi)
            if result is None:
                continue
            exact_ll, approx_ll = result
            total_exact += exact_ll
            total_approx += approx_ll
        return abs(total_exact - total_approx)

    def _accumulate_bin_sums(self, interferometer, test_converted_parameters, f_lo, f_hi):
        """Exact and mode-by-mode-relative-binning-approximated
        log-likelihood-ratio contributions from one candidate bin, on one
        interferometer, used by :meth:`_log_likelihood_error_for_candidate_bin`.

        The full-resolution fiducial and test-parameter mode waveforms
        needed for the exact sum are evaluated in chunks of
        ``self.summary_data_chunk_size`` samples (mirroring
        :meth:`compute_summary_data`), so they are never held in memory at
        once; the linear approximation only needs two edge evaluations per
        mode, done once outside the chunk loop.

        Returns ``None`` if the candidate bin is empty for this
        interferometer (its edges fall between two consecutive samples),
        otherwise ``(exact_log_likelihood, approximate_log_likelihood)``.
        """
        full_frequencies = interferometer.frequency_array
        mask = interferometer.frequency_mask
        masked_frequency_array = full_frequencies[mask]
        lo_idx, hi_idx = np.searchsorted(masked_frequency_array, [f_lo, f_hi])
        if hi_idx <= lo_idx:
            return None

        raw_offset = int(np.argmax(mask))
        # an injection as summary data: the bins are chosen on its noiseless signal
        raw_strain = (None if self.injection is not None
                      else interferometer.strain_data._frequency_domain_strain)
        psd_model = interferometer.power_spectral_density
        window_power_correction = interferometer._window_power_correction
        duration = interferometer.duration
        central_frequency = (f_lo + f_hi) / 2
        mode_keys = [f"{ell},{emm}" for ell, emm in self.mode_array]

        a0 = {key: 0j for key in mode_keys}
        a1 = {key: 0j for key in mode_keys}
        b0 = {key: {keyp: 0j for keyp in mode_keys} for key in mode_keys}
        b1 = {key: {keyp: 0j for keyp in mode_keys} for key in mode_keys}
        exact_dh = 0j
        exact_hh = 0j

        for chunk_lo in range(lo_idx, hi_idx, self.summary_data_chunk_size):
            chunk_hi = min(chunk_lo + self.summary_data_chunk_size, hi_idx)
            chunk_frequencies = masked_frequency_array[chunk_lo:chunk_hi]
            if raw_strain is None:
                chunk_strain = self.injection.signal(interferometer, chunk_frequencies)
            else:
                chunk_strain = raw_strain[raw_offset + chunk_lo:raw_offset + chunk_hi]
            chunk_psd = (
                psd_model.get_power_spectral_density_array(chunk_frequencies)
                * window_power_correction)
            delta_frequency = chunk_frequencies - central_frequency

            chunk_h0 = {
                key: self._project_fiducial_mode(
                    interferometer, self._fiducial_converted_parameters, key, chunk_frequencies)
                for key in mode_keys}
            chunk_h_test = {
                key: self._project_fiducial_mode(
                    interferometer, test_converted_parameters, key, chunk_frequencies)
                for key in mode_keys}

            h_test_total = sum(chunk_h_test.values())
            exact_dh += noise_weighted_inner_product(h_test_total, chunk_strain, chunk_psd, duration)
            exact_hh += noise_weighted_inner_product(h_test_total, h_test_total, chunk_psd, duration)

            for key in mode_keys:
                h0 = chunk_h0[key]
                a0[key] += noise_weighted_inner_product(h0, chunk_strain, chunk_psd, duration)
                a1[key] += noise_weighted_inner_product(
                    h0, chunk_strain * delta_frequency, chunk_psd, duration)
                for keyp in mode_keys:
                    h0p = chunk_h0[keyp]
                    b0[key][keyp] += noise_weighted_inner_product(h0p, h0, chunk_psd, duration)
                    b1[key][keyp] += noise_weighted_inner_product(
                        h0p, h0 * delta_frequency, chunk_psd, duration)

        edge_frequencies = np.array([f_lo, f_hi])
        r0, r1 = {}, {}
        for key in mode_keys:
            h0_edges = self._project_fiducial_mode(
                interferometer, self._fiducial_converted_parameters, key, edge_frequencies)
            h_test_edges = self._project_fiducial_mode(
                interferometer, test_converted_parameters, key, edge_frequencies)
            ratio_edges = h_test_edges / h0_edges
            r0[key] = 0.5 * (ratio_edges[0] + ratio_edges[1])
            r1[key] = (ratio_edges[1] - ratio_edges[0]) / (f_hi - f_lo)

        approx_dh = sum(
            a0[key] * np.conj(r0[key]) + a1[key] * np.conj(r1[key])
            for key in mode_keys)
        approx_hh = sum(
            b0[key][keyp] * r0[key] * np.conj(r0[keyp])
            + b1[key][keyp] * (r0[key] * np.conj(r1[keyp]) + np.conj(r0[keyp]) * r1[key])
            for key in mode_keys for keyp in mode_keys)

        exact_ll = np.real(exact_dh) - 0.5 * np.real(exact_hh)
        approx_ll = np.real(approx_dh) - 0.5 * np.real(approx_hh)
        return exact_ll, approx_ll

    def set_fiducial_waveforms(self, parameters):
        """Set fiducial waveforms based on the given parameters.

        No full-resolution waveform is built or stored here: only the sky-frame
        converted parameters (re-used to re-project the fiducial modes in
        chunks) and ``self.maximum_frequency`` are computed.

        Parameters
        ----------
        parameters: dict
            The parameter set for which to compute the fiducial waveforms.
        """
        parameters = parameters.copy()
        parameters["fiducial"] = 1
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))

        converted_parameters, _ = self.waveform_generator.parameter_conversion(parameters)
        # Stashed so the fiducial detector waveforms can be re-projected in
        # chunks later (setup_bins, compute_summary_data, _compute_full_waveform)
        # instead of storing them at full resolution here.
        self._fiducial_converted_parameters = converted_parameters
        self.maximum_frequency = self._fiducial_maximum_frequency(converted_parameters)
        logger.debug(f"Maximum fiducial frequency: {self.maximum_frequency}")

    def _fiducial_mode_polarizations(self, converted_parameters, mode_key, frequencies):
        """Evaluate one fiducial mode's plus/cross polarizations at ``frequencies``.

        Uses the waveform generator's frequency-sequence evaluation path
        directly (mirroring how :class:`~bilby.gw.WaveformGenerator` filters and
        forwards parameters), so an arbitrary sub-band can be produced without
        building the full-resolution frequency-domain waveform and without
        disturbing the generator's parameter cache.
        """
        ell, emm = (int(part) for part in mode_key.split(","))
        wfg = self.waveform_generator
        source_parameters = {
            key: value for key, value in converted_parameters.items()
            if key in wfg.source_parameter_keys
        }
        source_parameters["fiducial"] = 0
        waveform_arguments = dict(wfg.waveform_arguments)
        waveform_arguments["mode_array"] = [[ell, emm]]
        waveform_arguments["frequency_bin_edges"] = np.asarray(frequencies)
        source_parameters.update(waveform_arguments)
        polarizations = wfg.frequency_domain_source_model(
            wfg.frequency_array, **source_parameters)
        return polarizations[mode_key]

    def _fiducial_maximum_frequency(self, converted_parameters):
        """Largest frequency at which any fiducial mode is non-zero.

        Evaluated in ``summary_data_chunk_size`` chunks through the
        frequency-sequence path so no full-resolution waveform is built. This
        reproduces the legacy full-grid ``!= 0`` criterion (both paths share the
        waveform model's hard high-frequency cut-off) without its historical
        off-by-``minimum_frequency`` indexing error.
        """
        frequency_array = self.waveform_generator.frequency_array
        df = frequency_array[1] - frequency_array[0]
        minimum_frequency = min(ifo.minimum_frequency for ifo in self.interferometers)
        search_frequencies = frequency_array[frequency_array > (minimum_frequency - df)]

        per_mode_maxima = []
        for ell, emm in self.mode_array:
            mode_key = f"{ell},{emm}"
            last_nonzero_frequency = None
            for start in range(0, len(search_frequencies), self.summary_data_chunk_size):
                chunk = search_frequencies[start:start + self.summary_data_chunk_size]
                plus = self._fiducial_mode_polarizations(
                    converted_parameters, mode_key, chunk)["plus"]
                nonzero = np.nonzero(plus)[0]
                if len(nonzero):
                    last_nonzero_frequency = chunk[nonzero[-1]]
            if last_nonzero_frequency is not None:
                per_mode_maxima.append(last_nonzero_frequency)

        if per_mode_maxima:
            return min(per_mode_maxima)
        return min(ifo.maximum_frequency for ifo in self.interferometers)

    def find_maximum_likelihood_parameters(self, parameter_bounds,
                                           iterations=5, maximization_kwargs=None):
        """Find the maximum likelihood parameters using scipy optimization.

        Parameters
        ----------
        parameter_bounds: dict
            Dictionary of bounds (lists) for the initial parameters.
        iterations: int, optional
            Number of optimization iterations.
        maximization_kwargs: dict, optional
            Additional keyword arguments for the optimization.

        Returns
        -------
        updated_parameters: dict
            The updated parameter set based on the maximum likelihood optimization.
        """

        if maximization_kwargs is None:
            maximization_kwargs = dict()
        current_parameters = dict(self.fiducial_parameters, fiducial=0)
        updated_parameters_list = self.get_parameter_list_from_dictionary(self.fiducial_parameters)
        old_fiducial_ln_likelihood = _log_likelihood_ratio_at(self, current_parameters)
        logger.info(f"Fiducial ln likelihood ratio: {old_fiducial_ln_likelihood:.2f}")
        for it in range(iterations):
            logger.info(f"Optimizing fiducial parameters. Iteration : {it + 1}")
            output = differential_evolution(
                self.lnlike_scipy_maximize,
                bounds=parameter_bounds,
                x0=updated_parameters_list,
                **maximization_kwargs,
            )
            updated_parameters_list = output['x']
            updated_parameters = self.get_parameter_dictionary_from_list(updated_parameters_list)
            current_parameters.update(updated_parameters)
            self.set_fiducial_waveforms(updated_parameters)
            self.setup_bins()
            self.compute_summary_data()
            new_fiducial_ln_likelihood = _log_likelihood_ratio_at(self, current_parameters)
            logger.info(f"Fiducial ln likelihood ratio: {new_fiducial_ln_likelihood:.2f}")
            if new_fiducial_ln_likelihood - old_fiducial_ln_likelihood < 0.1:
                break
            old_fiducial_ln_likelihood = new_fiducial_ln_likelihood

        logger.info("Fiducial waveforms updated")
        logger.info("Summary Data updated")
        return updated_parameters

    def lnlike_scipy_maximize(self, parameter_list):
        """Compute the log likelihood for the given parameter list using scipy maximize.

        Parameters
        ----------
        parameter_list: list
            List of parameters for which to compute the log likelihood.

        Returns
        -------
        log_likelihood: float
            The log likelihood value.
        """
        parameters = dict(self.get_parameter_dictionary_from_list(parameter_list), fiducial=0)
        return -_log_likelihood_ratio_at(self, parameters)

    def get_parameter_dictionary_from_list(self, parameter_list):
        parameter_dictionary = dict(zip(self.parameters_to_be_updated, parameter_list))
        excluded_parameter_keys = set(self.fiducial_parameters) - set(self.parameters_to_be_updated)
        for key in excluded_parameter_keys:
            parameter_dictionary[key] = self.fiducial_parameters[key]
        return parameter_dictionary

    def get_parameter_list_from_dictionary(self, parameter_dict):
        return [parameter_dict[k] for k in self.parameters_to_be_updated]

    def get_bounds_from_priors(self, priors):
        bounds = []
        for key in self.parameters_to_be_updated:
            bounds.append([priors[key].minimum, priors[key].maximum])
        return bounds

    def _reference_mode(self, interferometer, mode_key, frequencies):
        """The fiducial waveform the ratios and the summary data of one mode
        refer to: its detector response, or with ``precessing_scheme="split"``
        its ``fast`` factor."""
        return self._project_fiducial_mode(
            interferometer, self._fiducial_converted_parameters, mode_key, frequencies,
            part="fast" if self.precessing_scheme == "split" else "strain")

    def _project_fiducial_mode(self, interferometer, converted_parameters, mode_key, frequencies,
                               part="strain"):
        """Project one mode's fiducial polarizations onto ``interferometer``.

        The fiducial polarizations for the requested ``frequencies`` are
        evaluated on demand through the waveform generator's frequency-sequence
        path, so callers can walk the band in chunks rather than building and
        storing a full-resolution detector waveform for every detector and mode.
        ``part="fast"`` gives the ``fast`` factor of the response instead
        (see ``precessing_scheme``).
        """
        full_frequencies = interferometer.frequency_array
        frequencies = np.asarray(frequencies)
        df = full_frequencies[1] - full_frequencies[0]
        above_minimum_frequency = frequencies > (interferometer.minimum_frequency - df)
        freqs = frequencies[above_minimum_frequency]

        mode_polarizations = self._fiducial_mode_polarizations(
            converted_parameters, mode_key, freqs)
        if part == "fast":
            keys = ('coprecessing', 'c_plus', 'c_cross')
        else:
            keys = ('plus', 'cross')
        waveform_polarizations_reduced = {
            mode_key: {key: mode_polarizations[key] for key in keys}}

        wf = np.zeros(len(frequencies), dtype=complex)
        response = interferometer.get_detector_response_for_frequency_dependent_antenna_response(
            waveform_polarizations=waveform_polarizations_reduced,
            parameters=converted_parameters,
            start_time=interferometer.strain_data.start_time,
            frequencies=freqs,
            earth_rotation_time_delay=self.earth_rotation_time_delay,
            earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
            finite_size=self.finite_size,
            orbital_motion=self.orbital_motion,
            center=self.center,
            split=part == "fast")
        wf[above_minimum_frequency] = response[mode_key][0] if part == "fast" else response
        wf[frequencies > self.maximum_frequency] = 0
        return wf

    def _bin_chunks(self, masked_bin_inds):
        """Yield ``(start_bin, end_bin)`` ranges of consecutive bins whose
        combined frequency span stays within ``summary_data_chunk_size``
        samples (always at least one bin per chunk)."""
        number_of_bins = self.number_of_bins
        start = 0
        while start < number_of_bins:
            end = start + 1
            while (end < number_of_bins and
                   masked_bin_inds[end + 1] - masked_bin_inds[start]
                   <= self.summary_data_chunk_size):
                end += 1
            yield start, end
            start = end

    def compute_summary_data(self):
        """Compute summary data for the likelihood.

        The per-mode fiducial detector waveforms are projected in frequency
        chunks of ``summary_data_chunk_size`` samples rather than all at once,
        so the full-resolution waveform never has to sit in memory. The result
        is identical to an unchunked computation.

        The actual (injected/real) strain data is also read in chunks,
        straight off a disk-backed memory-mapped copy
        (:meth:`bilby_xG.interferometer.Interferometer.offload_frequency_domain_strain`)
        rather than through the ``frequency_domain_strain``/
        ``power_spectral_density_array`` properties -- those recompute and
        boolean-mask a full-band array (O(1e8) bins) on every access, which
        is expensive both in itself and because it is done on top of the
        persistent full-band arrays already held per detector. The PSD is
        cheap to regenerate from the PSD file, so it is simply recomputed at
        each chunk's (small) frequency sub-array instead of being cached at
        full band.

        With ``self.injection`` (an injection as summary data) the strain is
        instead the injected signal, evaluated per chunk, and each bin's
        a0/a1 get a draw of their noise terms, sampled together with the
        noise term of <s|d>, which gives :meth:`noise_log_likelihood`. See
        :class:`bilby_xG.injection.SummaryDataInjection`.
        """
        summary_data = dict()
        injection = self.injection
        n_modes = len(self.mode_array)
        network_s_inner_s, network_s_inner_n = 0.0, 0.0

        for interferometer in self.interferometers:
            if injection is None:
                interferometer.offload_frequency_domain_strain(self.array_cache_dir)
            interferometer.discard_regenerable_frequency_caches()
            logger.info(
                f"Computing summary data for {interferometer.name} "
                f"({self.number_of_bins} bins).")
            summary_data[interferometer.name] = {
                'a0': dict(),
                'a1': dict(),
                'b0': dict(),
                'b1': dict()
            }

            mask = interferometer.frequency_mask
            n_in_band = int(np.count_nonzero(mask))
            raw_offset = int(np.argmax(mask))
            raw_last = len(mask) - 1 - int(np.argmax(mask[::-1]))
            is_contiguous = (raw_last - raw_offset + 1) == n_in_band
            # Only materialise the full index array (an O(1e8)-element int
            # allocation at ET's low minimum_frequency) in the fallback case;
            # the masks this pipeline uses (a single [minimum_frequency,
            # maximum_frequency] band) are always contiguous.
            in_band_indices = None if is_contiguous else np.flatnonzero(mask)
            masked_frequency_array = interferometer.frequency_array[mask]
            masked_bin_inds = np.searchsorted(masked_frequency_array, self.bin_freqs)
            duration = interferometer.duration
            raw_strain = (None if injection is not None
                          else interferometer.strain_data._frequency_domain_strain)
            psd_model = interferometer.power_spectral_density
            window_power_correction = interferometer._window_power_correction
            a0, b0, a1, b1 = dict(), dict(), dict(), dict()
            s_inner_s, s_inner_n = 0.0, 0j
            if injection is not None and injection.noise:
                # standard circular complex normals, one per basis function
                # (h0_k, h0_k (f - f_m) for every mode k, and the signal) per
                # bin, drawn up front so the realisation does not depend on
                # the chunking
                rng = injection.noise_generator(interferometer)
                z = (rng.standard_normal((self.number_of_bins, 2 * n_modes + 1))
                     + 1j * rng.standard_normal((self.number_of_bins, 2 * n_modes + 1))
                     ) / np.sqrt(2)

            for j, (ell, emm) in enumerate(self.mode_array):
                key = f'{ell},{emm}'
                b0[key] = dict()
                b1[key] = dict()
                a0[key], a1[key] = np.zeros((2, self.number_of_bins), dtype=complex)

                for ellp, emmp in self.mode_array[j::]:
                    b0[key][f'{ellp},{emmp}'], b1[key][f'{ellp},{emmp}'] = \
                        np.zeros((2, self.number_of_bins), dtype=complex)

            for chunk_start, chunk_end in self._bin_chunks(masked_bin_inds):
                chunk_lo = masked_bin_inds[chunk_start]
                chunk_hi = masked_bin_inds[chunk_end]
                chunk_frequencies = masked_frequency_array[chunk_lo:chunk_hi]
                if raw_strain is None:
                    chunk_strain = injection.signal(interferometer, chunk_frequencies)
                elif is_contiguous:
                    chunk_strain = raw_strain[
                        raw_offset + chunk_lo:raw_offset + chunk_hi]
                else:
                    chunk_strain = raw_strain[in_band_indices[chunk_lo:chunk_hi]]
                chunk_psd = (
                    psd_model.get_power_spectral_density_array(chunk_frequencies)
                    * window_power_correction)
                chunk_h0 = {
                    f'{ell},{emm}': self._reference_mode(
                        interferometer, f'{ell},{emm}', chunk_frequencies)
                    for ell, emm in self.mode_array
                }

                logger.info(
                    f"{interferometer.name}: summary data bins "
                    f"{chunk_start}-{chunk_end}/{self.number_of_bins} "
                    f"({chunk_end / self.number_of_bins:.0%}), "
                    f"frequencies [{chunk_frequencies[0]:.3g}, "
                    f"{chunk_frequencies[-1]:.3g}] Hz.")

                for i in range(chunk_start, chunk_end):
                    idxs = slice(masked_bin_inds[i] - chunk_lo,
                                 masked_bin_inds[i + 1] - chunk_lo)

                    frequencies = chunk_frequencies[idxs]
                    # the centre the ratios' r0 refer to (the midpoint of the
                    # bin edges), not that of the samples in the bin: they
                    # differ by up to a sample spacing, an error of r1 times
                    # it which more bins do not reduce
                    delta_frequency = frequencies - self.bin_centers[i]

                    strain = chunk_strain[idxs]
                    psd = chunk_psd[idxs]

                    for j, (ell, emm) in enumerate(self.mode_array):
                        key = f'{ell},{emm}'
                        h0 = chunk_h0[key][idxs]
                        a0[key][i] = noise_weighted_inner_product(h0, strain, psd, duration)
                        a1[key][i] = noise_weighted_inner_product(h0, strain * delta_frequency, psd, duration)

                        for ellp, emmp in self.mode_array[j::]:
                            keyp = f'{ellp},{emmp}'
                            h0_p = chunk_h0[keyp][idxs]
                            b0[key][keyp][i] = noise_weighted_inner_product(h0_p, h0, psd, duration)
                            b1[key][keyp][i] = noise_weighted_inner_product(h0_p, h0 * delta_frequency, psd, duration)

                    if injection is None:
                        continue
                    s_inner_s += noise_weighted_inner_product(strain, strain, psd, duration).real
                    if not injection.noise:
                        continue
                    h0s = np.array([chunk_h0[f'{ell},{emm}'][idxs]
                                    for ell, emm in self.mode_array])
                    basis = np.concatenate([h0s, h0s * delta_frequency, strain[None]])
                    # gram[i, j] = <g_i | g_j>; the noise terms <g_i | n> have
                    # E[x_i x_j^*] = 2 gram[i, j] for bilby's noise normalisation
                    gram = 4 / duration * (basis.conj() / psd) @ basis.T
                    noise = complex_gaussian(2 * gram, z[i])
                    for j, (ell, emm) in enumerate(self.mode_array):
                        a0[f'{ell},{emm}'][i] += noise[j]
                        a1[f'{ell},{emm}'][i] += noise[n_modes + j]
                    s_inner_n += noise[-1]

            if injection is not None:
                snr = np.sqrt(s_inner_s)
                message = f"{interferometer.name}: injected optimal SNR {snr:.2f}"
                if injection.noise:
                    message += f", matched-filter SNR {(s_inner_s + s_inner_n.real) / snr:.2f}"
                logger.info(message)
                network_s_inner_s += s_inner_s
                network_s_inner_n += s_inner_n.real

            for i, (ell, emm) in enumerate(self.mode_array):
                key = f'{ell},{emm}'
                summary_data[interferometer.name]['a0'][key] = a0[key]
                summary_data[interferometer.name]['a1'][key] = a1[key]
                if key not in summary_data[interferometer.name]['b0']:
                    summary_data[interferometer.name]['b0'][key] = dict()
                    summary_data[interferometer.name]['b1'][key] = dict()
                for ellp, emmp in self.mode_array[i::]:
                    keyp = f'{ellp},{emmp}'
                    if keyp not in summary_data[interferometer.name]['b0']:
                        summary_data[interferometer.name]['b0'][keyp] = dict()
                        summary_data[interferometer.name]['b1'][keyp] = dict()
                    summary_data[interferometer.name]['b0'][key][keyp] = b0[key][keyp]
                    summary_data[interferometer.name]['b1'][key][keyp] = b1[key][keyp]
                    summary_data[interferometer.name]['b0'][keyp][key] = np.conj(b0[key][keyp])
                    summary_data[interferometer.name]['b1'][keyp][key] = np.conj(b1[key][keyp])

            # Free this detector's full-band frequency array/mask/PSD cache
            # (repopulated above by the frequency_mask/frequency_array
            # accesses) before moving to the next interferometer -- without
            # this they stay attached to `interferometer` (and so to
            # self.interferometers, held for the likelihood's lifetime) and
            # accumulate across detectors instead of each one's being
            # released once its summary data is done.
            interferometer.discard_regenerable_frequency_caches()
            del mask, masked_frequency_array, masked_bin_inds, raw_strain
            if in_band_indices is not None:
                del in_band_indices

        self.summary_data = summary_data
        if injection is not None:
            snr = np.sqrt(network_s_inner_s)
            message = f"Network injected optimal SNR {snr:.2f}"
            if injection.noise:
                message += f", matched-filter SNR {(network_s_inner_s + network_s_inner_n) / snr:.2f}"
            logger.info(message)
            # -<d|d>/2 without the parameter-independent -<n|n>/2
            self._noise_log_likelihood_value = -network_s_inner_s / 2 - network_s_inner_n

    def _shared_response(self, converted_parameters):
        """The detector-independent part of the response at the bin edges
        (see ``shared`` in
        :meth:`~bilby_xG.interferometer.Interferometer.get_detector_response_for_frequency_dependent_antenna_response`),
        computed for the first interferometer of a likelihood evaluation
        and reused for the others."""
        key = tuple(converted_parameters.get(name) for name in _SHARED_RESPONSE_KEYS)
        cache = getattr(self, "_shared_response_cache", None)
        if cache is None or cache[0] != key or cache[1] is not self.bin_freqs:
            cache = (key, self.bin_freqs, {})
            self._shared_response_cache = cache
        return cache[2]

    def __getstate__(self):
        # the shared-response cache is per evaluation, and its layout may
        # change between versions
        state = self.__dict__.copy()
        state.pop("_shared_response_cache", None)
        return state

    def __setstate__(self, state):
        state.pop("_shared_response_cache", None)
        self.__dict__.update(state)

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer, parameters=None):
        name = interferometer.name
        r0, r1 = {}, {}
        parameters = _resolve_parameters(self, parameters)
        converted_parameters, _ = self.waveform_generator.parameter_conversion(parameters)
        shared = self._shared_response(converted_parameters)
        if self.precessing_scheme == "split":
            return self._split_ratio_per_interferometer(
                waveform_polarizations, interferometer, converted_parameters, shared)

        for ell, emm in self.mode_array:
            mode_key = f"{ell},{emm}"
            waveform_polarizations_reduced = {mode_key: waveform_polarizations[mode_key]}
            strain = interferometer.get_detector_response_for_frequency_dependent_antenna_response(
                waveform_polarizations=waveform_polarizations_reduced,
                parameters=converted_parameters,
                start_time=interferometer.strain_data.start_time,
                frequencies=self.bin_freqs,
                earth_rotation_time_delay=self.earth_rotation_time_delay,
                earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
                finite_size=self.finite_size,
                orbital_motion=self.orbital_motion,
                center=self.center,
                shared=shared)
            reference_strain = self.per_detector_per_mode_fiducial_waveform_points[name][mode_key]
            waveform_ratio = strain / reference_strain
            r0[mode_key] = 0.5 * (waveform_ratio[1:] + waveform_ratio[:-1])
            r1[mode_key] = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_widths

        return r0, r1

    def _split_ratio_per_interferometer(self, waveform_polarizations, interferometer,
                                        converted_parameters, shared):
        """``compute_waveform_ratio_per_interferometer`` for
        ``precessing_scheme="split"``: in each bin, the linearised ratio
        ``r = fast / fast_0`` times the linearised ``slow``, to first order
        in the offset from the bin centre."""
        name = interferometer.name
        r0, r1 = {}, {}
        if any("coprecessing" not in waveform_polarizations[f"{ell},{emm}"]
               for ell, emm in self.mode_array):
            raise ValueError(
                "precessing_scheme='split' needs the twist of each mode: a precessing "
                "source with the return_twist waveform argument (set by this likelihood "
                "on its waveform generator; was it changed or shared?)")
        factors = interferometer.get_detector_response_for_frequency_dependent_antenna_response(
            waveform_polarizations={f"{ell},{emm}": waveform_polarizations[f"{ell},{emm}"]
                                    for ell, emm in self.mode_array},
            parameters=converted_parameters,
            start_time=interferometer.strain_data.start_time,
            frequencies=self.bin_freqs,
            earth_rotation_time_delay=self.earth_rotation_time_delay,
            earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
            finite_size=self.finite_size,
            orbital_motion=self.orbital_motion,
            center=self.center,
            shared=shared,
            split=True)
        for mode_key, (fast, slow) in factors.items():
            ratio = fast / self.per_detector_per_mode_fiducial_waveform_points[name][mode_key]
            r0[mode_key], r1[mode_key] = split_scheme_coefficients(ratio, slow, self.bin_widths)
        return r0, r1

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters=None):
        """Reconstruct the full-resolution waveform ratio (only needed for
        time marginalisation's FFT, ``calculate_snrs``).

        Everything but the final scatter back to full band is done over
        ``interferometer.frequency_mask`` (the in-band subset) rather than
        the raw ``interferometer.frequency_array`` -- mirrors
        :meth:`compute_summary_data`'s ``masked_bin_inds`` remap, needed so
        ``self.bin_inds`` (built from the full-band array in
        :meth:`setup_bins`) isn't used to index a differently-sized array.
        The FFT downstream still needs a full-band, zero-outside-band array
        aligned with ``interferometer.frequency_domain_strain`` /
        ``power_spectral_density_array``, so that scatter still happens --
        this only avoids holding ``2 + 3 * len(mode_array)`` full-band
        buffers (a real cost once ``minimum_frequency`` is high enough that
        the masked band is meaningfully smaller than the full one; at a low
        ``minimum_frequency`` relative to ``maximum_frequency`` the two are
        close in size and the saving is small, but this is still the
        correct thing to index against).
        """
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            signal_polarizations, interferometer, parameters=parameters)
        f = interferometer.frequency_array
        mask = interferometer.frequency_mask
        f_masked = f[mask]
        masked_bin_inds = np.searchsorted(f_masked, self.bin_freqs)
        full_waveform_ratio = np.zeros_like(f_masked, dtype=complex)
        full_waveform_masked = np.zeros_like(f_masked, dtype=complex)

        for ell, emm in self.mode_array:
            mode_key = f"{ell},{emm}"
            duplicated_r0, duplicated_r1, duplicated_fm = np.zeros((3, f_masked.shape[0]), dtype=complex)

            for i in range(self.number_of_bins):
                idxs = slice(masked_bin_inds[i], masked_bin_inds[i + 1])
                duplicated_fm[idxs] = self.bin_centers[i]
                duplicated_r0[idxs] = r0[mode_key][i]
                duplicated_r1[idxs] = r1[mode_key][i]

            full_waveform_ratio += duplicated_r0 + duplicated_r1 * (f_masked - duplicated_fm)
            fiducial_waveform = self._reference_mode(interferometer, mode_key, f_masked)
            full_waveform_masked += full_waveform_ratio * fiducial_waveform

        full_waveform = np.zeros_like(f, dtype=complex)
        full_waveform[mask] = full_waveform_masked
        return full_waveform

    def calculate_snrs(self, waveform_polarizations, interferometer, return_array=True, parameters=None):
        """Calculate the SNRs (Signal-to-Noise Ratios).

        Parameters
        ----------
        waveform_polarizations: dict
            The waveform polarizations.
        interferometer: bilby.gw.detector.Interferometer
            The interferometer for which to calculate the SNRs.
        return_array: bool, optional
            If True, return the full waveform.
        parameters: dict, optional
            The parameters to evaluate at (bilby >= 2.8 always passes
            these). Falls back to the deprecated ``self.parameters`` state
            only when not given.

        Returns
        -------
        calculated_snrs: namedtuple
            A named tuple containing calculated SNR values.
        """
        parameters = _resolve_parameters(self, parameters)
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        a0 = self.summary_data[interferometer.name]['a0'].copy()
        a1 = self.summary_data[interferometer.name]['a1'].copy()
        b0 = self.summary_data[interferometer.name]['b0'].copy()
        b1 = self.summary_data[interferometer.name]['b1'].copy()

        d_inner_h, h_inner_h = np.zeros(2, dtype=complex)
        for ell, emm in self.mode_array:
            key = f"{ell},{emm}"
            d_inner_h += np.sum(a0[key] * np.conj(r0[key]) + a1[key] * np.conj(r1[key]))

            for ellp, emmp in self.mode_array:
                keyp = f"{ellp},{emmp}"
                h_inner_h += np.sum(
                    b0[key][keyp] * r0[key] * np.conj(r0[keyp])
                    + b1[key][keyp] * (r0[key] * np.conj(r1[keyp]) + np.conj(r0[keyp]) * r1[key]))

        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        if return_array and self.time_marginalization:
            full_waveform = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            d_inner_h_array = 4 / self.waveform_generator.duration * np.fft.fft(
                full_waveform[0:-1]
                * interferometer.frequency_domain_strain.conjugate()[0:-1]
                / interferometer.power_spectral_density_array[0:-1])

        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array
        )
