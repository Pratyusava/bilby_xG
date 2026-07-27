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
  binning summary data.
"""
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

__author__ = [
    "Pratyusava Baral <pbaral@uwm.edu>",
    "Soichiro Morisaki",
    "Ish Gupta",
]


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
    """

    def __init__(self, interferometers, waveform_generator, time_marginalization=False,
                 distance_marginalization=False, phase_marginalization=False,
                 calibration_marginalization=False, priors=None,
                 distance_marginalization_lookup_table=None, calibration_lookup_table=None,
                 number_of_response_curves=1000, starting_index=0, jitter_time=True,
                 reference_frame="sky", time_reference="geocenter",
                 earth_rotation_beam_patterns=True, earth_rotation_time_delay=True,
                 finite_size=True):

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

    def _compute_full_waveform(self, signal_polarizations, interferometer):
        """Project the waveform onto the frequency-dependent detector response.

        Parameters
        ==========
        signal_polarizations: dict
            Waveform evaluated at ``interferometer.frequency_array``. Either a
            ``{"plus", "cross"}`` dict or a per-mode nested dict.
        interferometer: bilby.gw.detector.Interferometer
            Interferometer to compute the response with respect to.
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
        parameters, _ = self.waveform_generator.parameter_conversion(self.parameters)
        h[idxs_above_minimum_frequency] = \
            interferometer.get_detector_response_for_frequency_dependent_antenna_response(
                waveform_polarizations=waveform_polarizations_red,
                parameters=parameters,
                start_time=interferometer.strain_data.start_time,
                frequencies=freqs,
                earth_rotation_time_delay=self.earth_rotation_time_delay,
                earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
                finite_size=self.finite_size)

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
    response_update: float, optional
        Time interval for updating the detector response. **Not implemented** in
        this release: only the default (``None``, update at every evaluation) is
        supported. Passing a value raises :class:`NotImplementedError`.
    """

    def __init__(
        self, interferometers, waveform_generator, reference_chirp_mass, highest_mode=2,
        linear_interpolation=True, accuracy_factor=5, time_offset=None, delta_f_end=None,
        maximum_banding_frequency=None, minimum_banding_duration=0.,
        distance_marginalization=False, phase_marginalization=False, priors=None,
        distance_marginalization_lookup_table=None, reference_frame="sky",
        time_reference="geocenter", earth_rotation_beam_patterns=True,
        earth_rotation_time_delay=True, finite_size=True, response_update=None
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
        if response_update is not None:
            raise NotImplementedError(
                "response_update is not supported; the detector response is "
                "updated at every waveform evaluation (response_update=None)."
            )
        self.response_update = response_update

    def calculate_snrs(self, waveform_polarizations, interferometer):
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
        converted_parameters, _ = self.waveform_generator.parameter_conversion(self.parameters)
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
            finite_size=self.finite_size)

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

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    The relative binning likelihood does not currently support calibration marginalization.
    """

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
            finite_size=True):

        self.earth_rotation_beam_patterns = earth_rotation_beam_patterns
        self.earth_rotation_time_delay = earth_rotation_time_delay
        self.finite_size = finite_size

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
                finite_size=self.finite_size)

            wf[frequencies > self.maximum_frequency] = 0
            self.per_detector_fiducial_waveforms[interferometer.name] = wf

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer):
        name = interferometer.name
        converted_parameters, _ = self.waveform_generator.parameter_conversion(self.parameters)
        strain = interferometer.get_detector_response_for_frequency_dependent_antenna_response(
            waveform_polarizations=waveform_polarizations,
            parameters=converted_parameters,
            start_time=interferometer.strain_data.start_time,
            frequencies=self.bin_freqs,
            earth_rotation_time_delay=self.earth_rotation_time_delay,
            earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
            finite_size=self.finite_size)

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

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    The relative binning likelihood does not currently support calibration marginalization.
    """

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
                 earth_rotation_time_delay=True,
                 earth_rotation_beam_patterns=True,
                 finite_size=True):

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
        self.maximum_frequency = waveform_generator.frequency_array[-1]
        self.fiducial_waveform_obtained = False
        self.check_if_bins_are_setup = False
        self.earth_rotation_time_delay = earth_rotation_time_delay
        self.earth_rotation_beam_patterns = earth_rotation_beam_patterns
        self.finite_size = finite_size
        # self.fiducial_polarizations = None
        self.fiducial_polarizations = dict()  # change
        # self.per_detector_fiducial_waveforms = dict()
        # self.per_detector_fiducial_waveform_points = dict()
        self.per_detector_per_mode_fiducial_waveforms = dict()
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
        self.parameters.update(self.fiducial_parameters)
        logger.info(f"Fiducial likelihood: {self.log_likelihood_ratio():.2f}")
        self.fiducial_likelihood = self.log_likelihood_ratio()
        self.parameters = dict(fiducial=0)

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={},\n\fiducial_parameters={},' \
            .format(self.interferometers, self.waveform_generator, self.fiducial_parameters)

    def setup_bins(self):
        """
        Setup the frequency bins following the method in
        https://arxiv.org/abs/1806.08792.

        If :code:`epsilon` is too small, the naive bins can be smaller than
        the frequency spacing of the data. We require that bins are at least
        as wide as this spacing.
        """
        frequency_array = self.waveform_generator.frequency_array
        gamma = self.gamma[:, np.newaxis]
        # Bin over the intersection of the interferometer frequency ranges,
        # matching bilby's RelativeBinningGravitationalWaveTransient.setup_bins.
        minimum_frequency = np.maximum.reduce(
            [ifo.minimum_frequency for ifo in self.interferometers], initial=0)
        maximum_frequency = np.minimum.reduce(
            [ifo.maximum_frequency for ifo in self.interferometers], initial=frequency_array[-1])
        maximum_frequency = min(maximum_frequency, self.maximum_frequency)

        frequency_array_useful = frequency_array[
            (frequency_array >= minimum_frequency)
            & (frequency_array <= maximum_frequency)
        ]

        d_alpha = self.chi * 2 * np.pi / np.abs(
            (minimum_frequency ** gamma) * np.heaviside(-gamma, 1)
            - (maximum_frequency ** gamma) * np.heaviside(gamma, 1)
        )
        d_phi = np.sum(
            np.sign(gamma) * d_alpha * frequency_array_useful ** gamma,
            axis=0
        )
        d_phi_from_start = d_phi - d_phi[0]
        number_of_bins = int(d_phi_from_start[-1] // self.epsilon)

        bin_edges = np.linspace(0, d_phi_from_start[-1], num=number_of_bins + 1)
        bin_indices = np.searchsorted(d_phi_from_start, bin_edges)
        unique_bin_indices = np.unique(bin_indices)

        bin_inds = np.searchsorted(frequency_array, frequency_array_useful[unique_bin_indices])
        bin_freqs = frequency_array_useful[unique_bin_indices]
        self.bin_inds = bin_inds
        self.bin_freqs = bin_freqs
        self.number_of_bins = len(bin_inds) - 1

        self.waveform_generator.waveform_arguments["frequency_bin_edges"] = self.bin_freqs
        self.bin_widths = self.bin_freqs[1:] - self.bin_freqs[:-1]
        self.bin_centers = (self.bin_freqs[1:] + self.bin_freqs[:-1]) / 2

        for interferometer in self.interferometers:
            name = interferometer.name
            self.per_detector_per_mode_fiducial_waveform_points[name] = {}
            for mode in self.mode_array:
                mode_key = f"{mode[0]},{mode[1]}"
                self.per_detector_per_mode_fiducial_waveform_points[name][mode_key] = \
                    self.per_detector_per_mode_fiducial_waveforms[name][mode_key][bin_inds]

    def set_fiducial_waveforms(self, parameters):
        """Set fiducial waveforms based on the given parameters.

        Parameters
        ----------
        parameters: dict
            The parameter set for which to compute the fiducial waveforms.
        """
        parameters = parameters.copy()
        parameters["fiducial"] = 1
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))

        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(parameters)

        if self.fiducial_polarizations is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")

        maximum_nonzero_index = []
        maximum_nonzero_frequency = []

        waveform_args = self.waveform_generator.waveform_arguments.copy()

        for interferometer in self.interferometers:
            self.per_detector_per_mode_fiducial_waveforms[interferometer.name] = dict()
            logger.debug(f"Maximum Frequency is {interferometer.maximum_frequency}")

            converted_parameters, _ = self.waveform_generator.parameter_conversion(parameters)
            frequencies = interferometer.frequency_array
            idxs_above_minimum_frequency = frequencies > \
                (interferometer.minimum_frequency - (frequencies[1] - frequencies[0]))
            freqs = frequencies[idxs_above_minimum_frequency]

            for ell, emm in self.mode_array:
                mode_key = f"{ell},{emm}"
                waveform_polarizations_reduced = {mode_key: {
                    'plus': self.fiducial_polarizations[mode_key]['plus'][idxs_above_minimum_frequency],
                    'cross': self.fiducial_polarizations[mode_key]['cross'][idxs_above_minimum_frequency]
                }}

                nonzero_indices = np.where(waveform_polarizations_reduced[mode_key]["plus"] != 0j)[0]
                if len(nonzero_indices) > 0:
                    max_idx = nonzero_indices[-1]
                    maximum_nonzero_index.append(max_idx)
                    max_freq = self.waveform_generator.frequency_array[max_idx]
                    maximum_nonzero_frequency.append(max_freq)

                    logger.debug(f"Maximum Nonzero Index for mode ({ell},{emm}): {max_idx}")
                    logger.debug(f"Maximum Nonzero Frequency for mode ({ell},{emm}): {max_freq}")

                wf = np.zeros_like(interferometer.frequency_array, dtype=complex)
                wf[idxs_above_minimum_frequency] = \
                    interferometer.get_detector_response_for_frequency_dependent_antenna_response(
                        waveform_polarizations=waveform_polarizations_reduced,
                        parameters=converted_parameters,
                        start_time=interferometer.strain_data.start_time,
                        frequencies=freqs,
                        earth_rotation_time_delay=self.earth_rotation_time_delay,
                        earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
                        finite_size=self.finite_size
                    )

                if maximum_nonzero_frequency:
                    self.maximum_frequency = min(maximum_nonzero_frequency)
                else:
                    self.maximum_frequency = interferometer.maximum_frequency

                wf[frequencies > self.maximum_frequency] = 0
                self.per_detector_per_mode_fiducial_waveforms[interferometer.name][mode_key] = wf
            self.waveform_generator.waveform_arguments = waveform_args.copy()

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
        self.parameters.update(self.fiducial_parameters)
        self.parameters["fiducial"] = 0
        updated_parameters_list = self.get_parameter_list_from_dictionary(self.fiducial_parameters)
        old_fiducial_ln_likelihood = self.log_likelihood_ratio()
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
            self.parameters.update(updated_parameters)
            self.set_fiducial_waveforms(updated_parameters)
            self.setup_bins()
            self.compute_summary_data()
            new_fiducial_ln_likelihood = self.log_likelihood_ratio()
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
        self.parameters.update(self.get_parameter_dictionary_from_list(parameter_list))
        return -self.log_likelihood_ratio()

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

    def compute_summary_data(self):
        """Compute summary data for the likelihood."""
        summary_data = dict()

        for interferometer in self.interferometers:
            summary_data[interferometer.name] = {
                'a0': dict(),
                'a1': dict(),
                'b0': dict(),
                'b1': dict()
            }

            mask = interferometer.frequency_mask
            masked_frequency_array = interferometer.frequency_array[mask]
            masked_bin_inds = np.searchsorted(masked_frequency_array, self.bin_freqs)
            masked_strain = interferometer.frequency_domain_strain[mask]
            masked_psd = interferometer.power_spectral_density_array[mask]
            duration = interferometer.duration
            a0, b0, a1, b1 = dict(), dict(), dict(), dict()

            for j, (ell, emm) in enumerate(self.mode_array):
                key = f'{ell},{emm}'
                b0[key] = dict()
                b1[key] = dict()
                a0[key], a1[key] = np.zeros((2, self.number_of_bins), dtype=complex)

                for ellp, emmp in self.mode_array[j::]:
                    b0[key][f'{ellp},{emmp}'], b1[key][f'{ellp},{emmp}'] = \
                        np.zeros((2, self.number_of_bins), dtype=complex)

            for i in range(self.number_of_bins):
                start_idx = masked_bin_inds[i]
                end_idx = masked_bin_inds[i + 1]
                idxs = slice(start_idx, end_idx)

                frequencies = masked_frequency_array[idxs]
                central_frequency = (frequencies[0] + frequencies[-1]) / 2
                delta_frequency = frequencies - central_frequency

                strain = masked_strain[idxs]
                psd = masked_psd[idxs]

                for j, (ell, emm) in enumerate(self.mode_array):
                    key = f'{ell},{emm}'
                    masked_h0 = self.per_detector_per_mode_fiducial_waveforms[interferometer.name][key][mask]
                    h0 = masked_h0[idxs]
                    a0[key][i] = noise_weighted_inner_product(h0, strain, psd, duration)
                    a1[key][i] = noise_weighted_inner_product(h0, strain * delta_frequency, psd, duration)

                    for ellp, emmp in self.mode_array[j::]:
                        keyp = f'{ellp},{emmp}'
                        masked_h0_p = \
                            self.per_detector_per_mode_fiducial_waveforms[interferometer.name][keyp][mask]
                        h0_p = masked_h0_p[idxs]
                        b0[key][keyp][i] = noise_weighted_inner_product(h0_p, h0, psd, duration)
                        b1[key][keyp][i] = noise_weighted_inner_product(h0_p, h0 * delta_frequency, psd, duration)

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

        self.summary_data = summary_data

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer):
        name = interferometer.name
        r0, r1 = {}, {}
        waveform_args = self.waveform_generator.waveform_arguments.copy()

        for ell, emm in self.mode_array:
            mode_key = f"{ell},{emm}"
            waveform_polarizations_reduced = {mode_key: waveform_polarizations[mode_key]}
            converted_parameters, _ = self.waveform_generator.parameter_conversion(self.parameters)
            strain = interferometer.get_detector_response_for_frequency_dependent_antenna_response(
                waveform_polarizations=waveform_polarizations_reduced,
                parameters=converted_parameters,
                start_time=interferometer.strain_data.start_time,
                frequencies=self.bin_freqs,
                earth_rotation_time_delay=self.earth_rotation_time_delay,
                earth_rotation_beam_patterns=self.earth_rotation_beam_patterns,
                finite_size=self.finite_size)
            reference_strain = self.per_detector_per_mode_fiducial_waveform_points[name][mode_key]
            waveform_ratio = strain / reference_strain
            r0[mode_key] = 0.5 * (waveform_ratio[1:] + waveform_ratio[:-1])
            r1[mode_key] = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_widths

        self.waveform_generator.waveform_arguments = waveform_args.copy()
        return r0, r1

    def _compute_full_waveform(self, signal_polarizations, interferometer):
        r0, r1 = self.compute_waveform_ratio_per_interferometer(signal_polarizations, interferometer)
        f = interferometer.frequency_array
        full_waveform_ratio = np.zeros_like(f, dtype=complex)
        full_waveform = np.zeros_like(f, dtype=complex)

        for ell, emm in self.mode_array:
            mode_key = f"{ell},{emm}"
            duplicated_r0, duplicated_r1, duplicated_fm = np.zeros((3, f.shape[0]), dtype=complex)

            for i in range(self.number_of_bins):
                idxs = slice(self.bin_inds[i], self.bin_inds[i + 1])
                duplicated_fm[idxs] = self.bin_centers[i]
                duplicated_r0[idxs] = r0[mode_key][i]
                duplicated_r1[idxs] = r1[mode_key][i]

            full_waveform_ratio += duplicated_r0 + duplicated_r1 * (f - duplicated_fm)
            fiducial_waveform = self.per_detector_per_mode_fiducial_waveforms[interferometer.name][mode_key]
            full_waveform += full_waveform_ratio * fiducial_waveform

        return full_waveform

    def calculate_snrs(self, waveform_polarizations, interferometer, return_array=True):
        """Calculate the SNRs (Signal-to-Noise Ratios).

        Parameters
        ----------
        waveform_polarizations: dict
            The waveform polarizations.
        interferometer: bilby.gw.detector.Interferometer
            The interferometer for which to calculate the SNRs.
        return_array: bool, optional
            If True, return the full waveform.

        Returns
        -------
        calculated_snrs: namedtuple
            A named tuple containing calculated SNR values.
        """
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
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
