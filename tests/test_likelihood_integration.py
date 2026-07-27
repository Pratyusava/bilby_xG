"""Integration tests for the next-generation likelihoods.

The central correctness check: with all frequency-dependent effects disabled
and a general-relativity propagation model, the response used by
``GravitationalWaveTransientNextGeneration`` must reduce to the standard bilby
detector response, so the two likelihoods agree.
"""
import numpy as np
import pytest

import bilby
from bilby_xG.likelihood import GravitationalWaveTransientNextGeneration
from bilby_xG.networks import InterferometerList

lalsim = pytest.importorskip("lalsimulation")

DURATION = 4.0
SAMPLING_FREQUENCY = 1024.0
INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0, luminosity_distance=2000.0, theta_jn=0.4,
    psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
    chi_1=0.0, chi_2=0.0,
)


@pytest.fixture(scope="module")
def setup():
    bilby.core.utils.random.seed(42)
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP", reference_frequency=50.0,
        minimum_frequency=20.0,
    )
    wfg = bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    ifos = InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION,
        start_time=INJECTION["geocent_time"] - 2,
    )
    ifos.inject_signal(parameters=INJECTION, waveform_generator=wfg)
    return ifos, wfg


def test_nextgen_likelihood_runs(setup):
    ifos, wfg = setup
    like = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg,
        earth_rotation_beam_patterns=True, earth_rotation_time_delay=True,
        finite_size=True,
    )
    like.parameters.update(INJECTION)
    logl = like.log_likelihood_ratio()
    assert np.isfinite(logl)


def test_gr_reduction_matches_standard(setup):
    """With effects off + GR, NextGen ~= standard GravitationalWaveTransient."""
    ifos, wfg = setup

    standard = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg,
    )
    nextgen = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg,
        earth_rotation_beam_patterns=False, earth_rotation_time_delay=False,
        finite_size=False,
    )
    standard.parameters.update(INJECTION)
    nextgen.parameters.update(INJECTION)

    logl_standard = standard.log_likelihood_ratio()
    logl_nextgen = nextgen.log_likelihood_ratio()

    assert np.isfinite(logl_nextgen)
    # Loose tolerance: the two response code paths differ in masking/edge
    # handling but must agree to well within a few percent at the injection.
    assert logl_nextgen == pytest.approx(logl_standard, rel=0.05)


def test_vG_unity_matches_gr(setup):
    """Sampling vG=1 must give the same likelihood as not sampling it."""
    ifos, wfg = setup
    nextgen = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg,
        earth_rotation_beam_patterns=True, earth_rotation_time_delay=True,
        finite_size=True,
    )
    nextgen.parameters.update(INJECTION)
    logl_gr = nextgen.log_likelihood_ratio()
    nextgen.parameters.update(dict(vG=1.0))
    logl_vg1 = nextgen.log_likelihood_ratio()
    assert logl_vg1 == pytest.approx(logl_gr, rel=1e-10)
