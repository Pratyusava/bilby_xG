"""Tests for the individual-mode (higher-order-mode) source models."""
import numpy as np
import pytest

import bilby
from bilby_xG.likelihood import GravitationalWaveTransientNextGeneration
from bilby_xG.networks import InterferometerList
from bilby_xG.source import lal_binary_black_hole_individual_modes

lalsim = pytest.importorskip("lalsimulation")

DURATION = 4.0
SAMPLING_FREQUENCY = 1024.0
PARAMS = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0, luminosity_distance=2000.0, theta_jn=0.4,
    psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
    chi_1=0.0, chi_2=0.0,
)


def test_individual_modes_returns_per_mode_dict():
    frequency_array = np.linspace(0, SAMPLING_FREQUENCY / 2, 2049)
    modes = lal_binary_black_hole_individual_modes(
        frequency_array,
        mass_1=PARAMS["mass_1"], mass_2=PARAMS["mass_2"],
        luminosity_distance=PARAMS["luminosity_distance"],
        a_1=0.0, tilt_1=0.0, phi_12=0.0, a_2=0.0, tilt_2=0.0, phi_jl=0.0,
        theta_jn=PARAMS["theta_jn"], phase=PARAMS["phase"],
        waveform_approximant="IMRPhenomXHM", reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=SAMPLING_FREQUENCY / 2,
        mode_array=[[2, 2], [3, 3], [4, 4]],
    )
    # One entry per azimuthal number requested, each with plus/cross.
    assert set(modes.keys()) == {"2", "3", "4"}
    for value in modes.values():
        assert set(value.keys()) == {"plus", "cross"}
        assert value["plus"].shape == frequency_array.shape


def test_nextgen_likelihood_with_higher_modes_runs():
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXHM", reference_frequency=50.0,
        minimum_frequency=20.0, mode_array=[[2, 2], [3, 3], [4, 4]],
    )
    wfg = bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=lal_binary_black_hole_individual_modes,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    ifos = InterferometerList(["CE", "CE20"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION,
        start_time=PARAMS["geocent_time"] - 2,
    )
    like = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg,
    )
    like.parameters.update(PARAMS)
    logl = like.log_likelihood_ratio()
    assert np.isfinite(logl)
