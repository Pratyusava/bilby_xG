"""Greedy bin selection with eta tuned to an evidence-bias tolerance."""
import json
import os

import numpy as np
import pytest
from scipy.stats import gamma

from bilby_xG.bin_selection import (evidence_bias, nested_sampling_resolution,
                                    sampled_parameter_count)


def test_evidence_bias_is_ln_mean_exp_or_mean_abs():
    rng = np.random.default_rng(0)
    errors = 1e-3 * rng.standard_normal(1000)
    assert evidence_bias(errors) == pytest.approx(np.mean(np.abs(errors)))
    # one-signed errors: ln E[e^eps] > E[eps] = E|eps|
    errors = np.abs(errors) * 100
    assert evidence_bias(errors) == pytest.approx(np.log(np.mean(np.exp(errors))))
    assert evidence_bias(-errors) == pytest.approx(np.mean(errors))


def test_nested_sampling_resolution():
    # Gaussian L(X): d ln L / d ln X = 2 d / D, one dead point per ln X step 1 / n_live
    d = np.array([0.5, 5.5, 50])
    assert np.allclose(nested_sampling_resolution(d, 11, 1000), 2 * d / 11e3)


def test_sampled_parameter_count():
    import bilby
    priors = bilby.core.prior.PriorDict(dict(
        a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Sine(), c=1.0,
        d=bilby.core.prior.DeltaFunction(2.0), e=bilby.core.prior.Constraint(0, 1)))
    assert sampled_parameter_count(priors) == 2


@pytest.fixture(scope="module")
def selection(tmp_path_factory):
    pytest.importorskip("mlgw_bns.batched")
    import bilby

    from bilby_xG.bin_selection import select_relative_binning_bins
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import (MLGW_BNS_MODES, convert_to_mlgw_bns_parameters,
                                 mlgw_bns_individual_modes)
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 30.0, 2048.0, 128
    injection = dict(
        chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
        lambda_2=600.0, luminosity_distance=400.0, theta_jn=0.4, psi=1.0, phase=1.3,
        geocent_time=1187008882.4, ra=3.4, dec=-0.4)
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=4096,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                mode_array=[list(mode) for mode in MLGW_BNS_MODES]))
    ifos = InterferometerList(["ET-EMR"])
    ifos.set_strain_data_from_zero_noise(4096, duration, injection["geocent_time"] + 2 - duration)
    priors = bilby.core.prior.PriorDict(dict(injection))
    priors["mass_ratio"] = bilby.core.prior.Uniform(0.5, 1)
    priors["lambda_1"] = bilby.core.prior.Uniform(0, 2000)
    priors["lambda_2"] = bilby.core.prior.Uniform(0, 2000)
    priors["luminosity_distance"] = bilby.core.prior.Uniform(200, 800)
    priors["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi)

    def run(name, **kwargs):
        bilby.core.utils.random.seed(1)
        outdir = str(tmp_path_factory.mktemp(name))
        bin_freqs = select_relative_binning_bins(
            ifos, wfg, injection, priors, fmin, fmax, eta=3.0, n_train=4, n_validation=2,
            n_evidence=16, n_scan_draws=1, scan_betas=np.geomspace(1e-4, 1, 5),
            n_grid=5001, npool=4, outdir=outdir, **kwargs)
        with open(os.path.join(outdir, "bin_selection.json")) as f:
            return bin_freqs, json.load(f), outdir

    return dict(tuned=run("tuned", evidence_tolerance=1e-5),
                fixed=run("fixed", evidence_tolerance=None))


def test_tuned_eta_meets_tolerance(selection):
    bin_freqs, summary, outdir = selection["tuned"]
    assert summary["evidence_bias"] <= 1e-5
    assert summary["eta"] == max(h["eta"] for h in summary["eta_history"]
                                 if h["evidence_bias"] <= 1e-5)
    assert summary["n_bins"] == len(bin_freqs) - 1
    assert os.path.exists(os.path.join(outdir, "bin_selection.png"))
    # evidence points spread over the Gamma(D_eff / 2) posterior of d
    d = -np.array(summary["evidence_delta_log_likelihood"])
    # mass_ratio, lambda_1, lambda_2, luminosity_distance, phase
    assert summary["effective_dimension"] == 5
    lo, hi = gamma.ppf([0.01, 0.99], summary["effective_dimension"] / 2)
    assert np.all((d > 0.9 * lo) & (d < 1.1 * hi))


def test_tolerance_none_keeps_eta(selection):
    _, fixed, _ = selection["fixed"]
    _, tuned, _ = selection["tuned"]
    assert fixed["eta"] == 3.0 and len(fixed["eta_history"]) == 1
    # the same first trial, the tuned one goes on to finer bins
    assert tuned["eta_history"][0] == fixed["eta_history"][0]
    assert fixed["evidence_bias"] > 1e-5 and tuned["n_bins"] > fixed["n_bins"]
