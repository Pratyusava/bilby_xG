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
