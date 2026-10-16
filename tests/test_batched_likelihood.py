"""The batched (JAX) mode-by-mode likelihood against the numpy one."""
import numpy as np
import pytest

pytest.importorskip("mlgw_bns")
pytest.importorskip("jax")


@pytest.fixture(scope="module")
def likelihoods():
    from bilby_xG.batched import BatchedRelativeBinningLikelihood
    from bilby_xG.injection import SummaryDataInjection
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    )
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import (MLGW_BNS_MODES, convert_to_mlgw_bns_parameters,
                                 mlgw_bns_individual_modes)
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 30.0, 2048.0, 128
    injection = dict(
        chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
        lambda_2=600.0, luminosity_distance=400.0, theta_jn=0.4, psi=1.0, phase=1.3,
        geocent_time=1187008882.4, ra=3.4, dec=-0.4)
    modes = [list(mode) for mode in MLGW_BNS_MODES]
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=4096,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                mode_array=modes))
    start_time = injection["geocent_time"] + 2 - duration
    data = SummaryDataInjection(wfg, injection, start_time, fmin, fmax, noise=False)
    numpy_likelihood = Likelihood(
        interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
        fiducial_parameters=injection, epsilon=0.03, time_reference="geocent",
        mode_array=modes, injection=data)
    return injection, numpy_likelihood, BatchedRelativeBinningLikelihood(
        numpy_likelihood, batch_size=8)


def test_batched_likelihood_matches_numpy(likelihoods):
    injection, numpy_likelihood, batched = likelihoods
    rng = np.random.default_rng(1)
    samples = [dict(injection)]
    for _ in range(9):
        sample = dict(injection)
        sample["chirp_mass"] *= 1 + 2e-6 * rng.normal()
        sample["lambda_1"] += 50 * rng.normal()
        sample["theta_jn"] += 0.05 * rng.normal()
        sample["ra"] += 1e-2 * rng.normal()
        sample["geocent_time"] += 1e-4 * rng.normal()
        samples.append(sample)
    expected = np.array([numpy_likelihood.log_likelihood_ratio(parameters=dict(s))
                         for s in samples])
    # ten points: one batch of 8 and a padded one
    got = batched.log_likelihood_ratio({key: np.array([s[key] for s in samples])
                                        for key in samples[0]})
    assert got.shape == (len(samples),)
    # a scalar dict gives a float
    single = batched.log_likelihood_ratio(dict(injection))
    assert isinstance(single, float)
    np.testing.assert_allclose(single, got[0], rtol=1e-6)
    # The mlgw_bns regressors round differently in a JAX batch than in a
    # single numpy call, by up to ~1e-2 rad in phase near the merger (see
    # mlgw_bns.batched); at this SNR (~50) that moves ln L by O(0.1).
    assert np.max(np.abs(got - expected)) < 0.5
    assert abs(got[0] - expected[0]) < 1e-2


def test_batched_likelihood_out_of_range_is_minus_infinity(likelihoods):
    injection, _, batched = likelihoods
    assert batched.log_likelihood_ratio(dict(injection, chi_1=0.9)) == -np.inf


def _precessing_setup(precessing_scheme="ratio", wfg=None):
    from bilby_xG.batched import BatchedRelativeBinningLikelihood
    from bilby_xG.injection import SummaryDataInjection
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    )
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import (MLGW_BNS_MODES, convert_to_mlgw_bns_precessing_parameters,
                                 mlgw_bns_precessing_individual_modes)
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 30.0, 2048.0, 128
    injection = dict(
        chirp_mass=1.1975, mass_ratio=0.9, a_1=0.3, a_2=0.1, tilt_1=1.2, tilt_2=2.0,
        phi_12=1.0, phi_jl=0.7, lambda_1=400.0, lambda_2=600.0, luminosity_distance=400.0,
        theta_jn=0.4, psi=1.0, phase=1.3, geocent_time=1187008882.4, ra=3.4, dec=-0.4)
    modes = [list(mode) for mode in MLGW_BNS_MODES]
    if wfg is None:
        wfg = WaveformGenerator(
            duration=duration, sampling_frequency=4096,
            frequency_domain_source_model=mlgw_bns_precessing_individual_modes,
            parameter_conversion=convert_to_mlgw_bns_precessing_parameters,
            waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                    reference_frequency=fmin, mode_array=modes))
    start_time = injection["geocent_time"] + 2 - duration
    data = SummaryDataInjection(wfg, injection, start_time, fmin, fmax, noise=False)
    numpy_likelihood = Likelihood(
        interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
        fiducial_parameters=injection, epsilon=0.03, time_reference="geocent",
        mode_array=modes, injection=data, precessing_scheme=precessing_scheme)
    return injection, numpy_likelihood, BatchedRelativeBinningLikelihood(
        numpy_likelihood, batch_size=4)


@pytest.fixture(scope="module")
def precessing_likelihoods():
    return _precessing_setup()


@pytest.fixture(scope="module")
def split_likelihoods(precessing_likelihoods):
    # a used generator, as from a pickled run: its cached waveform at the
    # fiducial point (without the twist, on other bin edges) must not be reused
    import copy
    wfg = copy.deepcopy(precessing_likelihoods[1].waveform_generator)
    assert wfg._cache["waveform"] is not None
    return _precessing_setup("split", wfg=wfg)


def _precessing_samples(injection, n, seed):
    rng = np.random.default_rng(seed)
    samples = [dict(injection)]
    for _ in range(n):
        sample = dict(injection)
        sample["chirp_mass"] *= 1 + 2e-6 * rng.normal()
        sample["a_1"] += 0.02 * rng.normal()
        sample["tilt_1"] += 0.1 * rng.normal()
        sample["phi_jl"] += 0.3 * rng.normal()
        sample["phi_12"] += 0.3 * rng.normal()
        sample["theta_jn"] += 0.05 * rng.normal()
        samples.append(sample)
    return samples


def test_batched_precessing_likelihood_matches_numpy(precessing_likelihoods):
    injection, numpy_likelihood, batched = precessing_likelihoods
    samples = _precessing_samples(injection, 4, seed=2)
    expected = np.array([numpy_likelihood.log_likelihood_ratio(parameters=dict(s))
                         for s in samples])
    got = batched.log_likelihood_ratio({key: np.array([s[key] for s in samples])
                                        for key in samples[0]})
    # at the injection, zero noise: ln L ratio = <h|h> / 2
    assert expected[0] > 100
    assert np.all(expected[1:] < expected[0])
    # see test_batched_likelihood_matches_numpy for the tolerance
    assert np.max(np.abs(got - expected)) < 0.5
    assert abs(got[0] - expected[0]) < 0.05


def test_batched_precessing_likelihood_out_of_range_is_minus_infinity(precessing_likelihoods):
    injection, _, batched = precessing_likelihoods
    assert batched.log_likelihood_ratio(dict(injection, a_1=0.9, tilt_1=0.0)) == -np.inf


def _exact_log_likelihood_ratio(likelihood, interferometer, converted, f_lo, f_hi):
    """ln L ratio summed exactly over the data grid in [f_lo, f_hi]."""
    from bilby.gw.utils import noise_weighted_inner_product

    frequencies = interferometer.frequency_array[interferometer.frequency_mask]
    lo, hi = np.searchsorted(frequencies, [f_lo, f_hi])
    frequencies = frequencies[lo:hi]
    if likelihood.injection is not None:
        strain = likelihood.injection.signal(interferometer, frequencies)
    else:
        offset = int(np.argmax(interferometer.frequency_mask))
        strain = interferometer.strain_data._frequency_domain_strain[offset + lo:offset + hi]
    psd = (interferometer.power_spectral_density.get_power_spectral_density_array(frequencies)
           * interferometer._window_power_correction)
    h = sum(likelihood._project_fiducial_mode(interferometer, converted, f"{ell},{emm}", frequencies)
            for ell, emm in likelihood.mode_array)
    duration = interferometer.duration
    return (np.real(noise_weighted_inner_product(h, strain, psd, duration))
            - 0.5 * np.real(noise_weighted_inner_product(h, h, psd, duration)))


def test_split_scheme(precessing_likelihoods, split_likelihoods):
    """Scheme 1 (``precessing_scheme="split"``) agrees with scheme 2 and with
    the exact likelihood, and the batched version with the numpy one."""
    injection, ratio_likelihood, _ = precessing_likelihoods
    _, split_likelihood, batched = split_likelihoods
    samples = _precessing_samples(injection, 3, seed=3)
    ratio = np.array([ratio_likelihood.log_likelihood_ratio(parameters=dict(s))
                      for s in samples])
    split = np.array([split_likelihood.log_likelihood_ratio(parameters=dict(s))
                      for s in samples])
    # the exact ln L ratio: one bin over the whole band, summed exactly
    exact = []
    for sample in samples:
        sample = dict(sample, fiducial=1)
        sample.update(split_likelihood.get_sky_frame_parameters(parameters=sample))
        converted, _ = split_likelihood.waveform_generator.parameter_conversion(sample)
        exact.append(sum(_exact_log_likelihood_ratio(
            split_likelihood, ifo, converted, split_likelihood.bin_freqs[0],
            split_likelihood.bin_freqs[-1])
            for ifo in split_likelihood.interferometers))
    np.testing.assert_allclose(ratio, exact, atol=0.1)
    np.testing.assert_allclose(split, exact, atol=0.1)
    got = batched.log_likelihood_ratio({key: np.array([s[key] for s in samples])
                                        for key in samples[0]})
    assert np.max(np.abs(got - split)) < 0.5


def test_truncated_mixture_likelihood(likelihoods):
    from scipy.special import logsumexp

    from bilby_xG.batched import (BatchedRelativeBinningLikelihood,
                                  BatchedTruncatedMixtureLikelihood)

    injection, numpy_likelihood, batched = likelihoods
    n_bins = numpy_likelihood.number_of_bins
    edges = [n_bins // 4, n_bins // 2, n_bins]
    log_evidences = [3.0, -1.0, 20.0]
    mixture = BatchedTruncatedMixtureLikelihood(
        numpy_likelihood, edges, log_evidences, batch_size=8)
    rng = np.random.default_rng(2)
    points = {key: np.full(5, value, dtype=float) for key, value in injection.items()}
    points["chirp_mass"] *= 1 + 2e-6 * rng.normal(size=5)
    points["ra"] += 1e-2 * rng.normal(size=5)

    # reference: the data truncated at an edge, by zeroing the summary data
    # of the bins above it
    truncated = []
    for edge in edges:
        reference = BatchedRelativeBinningLikelihood(numpy_likelihood, batch_size=8)
        for key in ("a0", "a1", "b0", "b1"):
            reference._arrays[key] = reference._arrays[key].copy()
            reference._arrays[key][..., edge:] = 0
        truncated.append(reference.log_likelihood_ratio(points))
    truncated = np.array(truncated)
    np.testing.assert_allclose(truncated[-1], batched.log_likelihood_ratio(points), rtol=1e-12)
    expected = logsumexp(truncated - np.array(log_evidences)[:, None], axis=0)
    np.testing.assert_allclose(mixture.log_likelihood_ratio(points), expected,
                               rtol=1e-10, atol=1e-8)
    with pytest.raises(ValueError):
        BatchedTruncatedMixtureLikelihood(numpy_likelihood, [n_bins, n_bins // 2], [0, 0])
