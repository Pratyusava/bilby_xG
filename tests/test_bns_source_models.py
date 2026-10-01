"""Tests for the surrogate / EOB BNS source models (optional dependencies)."""
import numpy as np
import pytest

PARAMETERS = dict(chirp_mass=1.2, mass_ratio=0.9, chi_1=0.01, chi_2=0.0,
                  lambda_1=400.0, lambda_2=900.0, luminosity_distance=40.0,
                  theta_jn=0.35, phase=1.57)
SOURCE_KEYS = ["M", "q", "chi1z", "chi2z", "LambdaAl2", "LambdaBl2", "distance",
               "inclination", "coalescence_angle"]


def test_mlgw_bns_modes():
    pytest.importorskip("mlgw_bns")
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes

    params, added = convert_to_mlgw_bns_parameters(dict(PARAMETERS))
    assert {"M", "q", "LambdaAl2", "coalescence_angle"} <= set(added)
    freqs = np.array([0.0, 5.0, 20.0, 100.0, 1000.0])
    kwargs = {key: params[key] for key in SOURCE_KEYS}
    modes = mlgw_bns_individual_modes(freqs, **kwargs, mode_array=[[2, 2], [3, 3]])
    assert set(modes) == {"2,2", "3,3"}
    for pols in modes.values():
        assert pols["plus"][0] == 0 and np.all(np.abs(pols["plus"][1:]) > 0)
    # frequency_bin_edges takes precedence over the frequency array
    on_edges = mlgw_bns_individual_modes(freqs, **kwargs, mode_array=[[2, 2]],
                                         frequency_bin_edges=freqs[2:])
    np.testing.assert_array_equal(on_edges["2,2"]["cross"], modes["2,2"]["cross"][2:])
    with pytest.raises(ValueError):
        mlgw_bns_individual_modes(freqs, **kwargs, mode_array=[[3, 2]])


def test_mlgw_bns_modes_match_the_surrogate():
    """Per mode, the model's own h_+ - i h_x (at azimuth 0, i.e. phase pi/2)."""
    pytest.importorskip("mlgw_bns")
    from mlgw_bns import ParametersWithExtrinsic
    from bilby_xG.source import (_mlgw_bns_model, convert_to_mlgw_bns_parameters,
                                 mlgw_bns_individual_modes)

    params, _ = convert_to_mlgw_bns_parameters(dict(PARAMETERS, phase=np.pi / 2))
    freqs = np.geomspace(3.0, 2048.0, 300)
    modes = mlgw_bns_individual_modes(freqs, **{key: params[key] for key in SOURCE_KEYS})
    expected = _mlgw_bns_model().predict_modes_dict(freqs, ParametersWithExtrinsic(
        mass_ratio=1 / params["q"], lambda_1=params["LambdaAl2"],
        lambda_2=params["LambdaBl2"], chi_1=params["chi1z"], chi_2=params["chi2z"],
        distance_mpc=params["distance"], inclination=params["inclination"],
        total_mass=params["M"], coalescence_phase=0.0, merger_time=0.0))
    for key, pols in modes.items():
        ell, emm = (int(part) for part in key.split(","))
        # mlgw_bns' batched evaluation reproduces predict_modes_dict to the
        # rounding of phases of up to ~1e6 rad
        scale = np.max(np.abs(expected[(ell, emm)]))
        np.testing.assert_allclose((pols["plus"] - 1j * pols["cross"]) / scale,
                                   expected[(ell, emm)] / scale, rtol=0, atol=1e-7)


def test_mlgw_bns_modes_accept_small_lambda():
    """The surrogate's training guard lambda >= 5 is relaxed for every mode."""
    pytest.importorskip("mlgw_bns")
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes

    params, _ = convert_to_mlgw_bns_parameters(dict(PARAMETERS, lambda_1=0.0, lambda_2=1.0))
    modes = mlgw_bns_individual_modes(np.geomspace(5.0, 1000.0, 20),
                                      **{key: params[key] for key in SOURCE_KEYS})
    assert all(np.all(np.isfinite(pols["plus"])) for pols in modes.values())
    with pytest.raises(ValueError):
        mlgw_bns_individual_modes(np.geomspace(5.0, 1000.0, 20), **dict(
            {key: params[key] for key in SOURCE_KEYS}, chi1z=0.9))


PRECESSING = dict(chirp_mass=1.2, mass_ratio=0.8, a_1=0.3, a_2=0.2, tilt_1=1.2, tilt_2=2.0,
                  phi_12=1.0, phi_jl=0.7, lambda_1=400.0, lambda_2=900.0,
                  luminosity_distance=40.0, theta_jn=0.9, phase=0.4)
PRECESSING_KEYS = ["M", "q", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
                   "LambdaAl2", "LambdaBl2", "distance", "theta_jn", "phase"]
PRECESSING_ARGUMENTS = dict(reference_frequency=20.0, minimum_frequency=10.0)


def _precessing_modes(frequencies, parameters=PRECESSING, **kwargs):
    from bilby_xG.source import (convert_to_mlgw_bns_precessing_parameters,
                                 mlgw_bns_precessing_individual_modes)

    params, _ = convert_to_mlgw_bns_precessing_parameters(dict(parameters))
    return mlgw_bns_precessing_individual_modes(
        frequencies, **{key: params[key] for key in PRECESSING_KEYS},
        **(PRECESSING_ARGUMENTS | kwargs))


def test_mlgw_bns_precessing_conversion():
    pytest.importorskip("mlgw_bns")
    from bilby_xG.source import convert_to_mlgw_bns_precessing_parameters

    params, added = convert_to_mlgw_bns_precessing_parameters(dict(PRECESSING))
    assert params["chi_1"] == pytest.approx(0.3 * np.cos(1.2))
    assert params["chi_2"] == pytest.approx(0.2 * np.cos(2.0))
    assert params["mass_1"] / params["mass_2"] == pytest.approx(1 / 0.8)
    # missing spins are zero
    aligned, _ = convert_to_mlgw_bns_precessing_parameters(
        {key: value for key, value in PRECESSING.items() if key not in ("a_2", "phi_jl")})
    assert aligned["a_2"] == 0 and aligned["phi_jl"] == 0 and aligned["chi_2"] == 0


def test_mlgw_bns_precessing_modes_sum_to_the_surrogate():
    """The modes add up to mlgw_bns' precessing h_+ and (with LALSuite's
    sign) h_x, and are zero below the minimum frequency."""
    pytest.importorskip("jax")
    pytest.importorskip("mlgw_bns")
    from mlgw_bns.batched_precession import batch_arguments
    from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic
    from bilby_xG.source import _mlgw_bns_model

    freqs = np.concatenate([[0.0, 5.0], np.geomspace(10.0, 1500.0, 300)])
    modes = _precessing_modes(freqs, return_twist=True)
    assert set(modes) == {"2,2", "2,1", "3,3", "4,4"}
    for pols in modes.values():
        assert np.all(pols["plus"][:2] == 0) and np.all(np.abs(pols["plus"][2:]) > 0)
        np.testing.assert_allclose(pols["plus"], pols["c_plus"] * pols["coprecessing"])
        np.testing.assert_allclose(pols["cross"], pols["c_cross"] * pols["coprecessing"])

    q = PRECESSING["mass_ratio"]
    total_mass = PRECESSING["chirp_mass"] / (q / (1 + q) ** 2) ** (3 / 5)
    params = PrecessingParametersWithExtrinsic.from_lvk(
        mass_1=total_mass / (1 + q), mass_2=total_mass * q / (1 + q),
        **{key: PRECESSING[key] for key in (
            "theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2", "phase",
            "lambda_1", "lambda_2")},
        reference_frequency_hz=20.0, distance_mpc=40.0)
    plus, cross = (np.asarray(x)[0] for x in PrecessingModel(_mlgw_bns_model()).jax_predict()(
        *batch_arguments([params], freqs[2:])))
    scale = np.max(np.abs(plus))
    # to the rounding of the surrogate's regressors in different batches
    np.testing.assert_allclose(sum(p["plus"] for p in modes.values())[2:] / scale,
                               plus / scale, rtol=0, atol=2e-3)
    np.testing.assert_allclose(sum(p["cross"] for p in modes.values())[2:] / scale,
                               -cross / scale, rtol=0, atol=2e-3)


def test_mlgw_bns_precessing_modes_do_not_depend_on_the_grid():
    """The precession angles are integrated from the minimum frequency
    whatever the frequencies (bin edges, chunks) the modes are evaluated at."""
    pytest.importorskip("jax")
    pytest.importorskip("mlgw_bns")
    freqs = np.geomspace(10.0, 1500.0, 300)
    modes = _precessing_modes(freqs)
    on_edges = _precessing_modes(freqs, frequency_bin_edges=freqs[150:])
    for key, pols in on_edges.items():
        scale = np.max(np.abs(modes[key]["plus"]))
        np.testing.assert_allclose(pols["plus"] / scale, modes[key]["plus"][150:] / scale,
                                   rtol=0, atol=2e-3)


def test_mlgw_bns_precessing_aligned_limit():
    """With the spins along L, each mode is the aligned source's up to a
    constant phase (the two are referenced to different points)."""
    pytest.importorskip("jax")
    pytest.importorskip("mlgw_bns")
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes

    freqs = np.geomspace(10.0, 800.0, 200)
    parameters = dict(PRECESSING, tilt_1=0.0, tilt_2=np.pi)
    precessing = _precessing_modes(freqs, parameters)
    params, _ = convert_to_mlgw_bns_parameters(dict(
        parameters, chi_1=parameters["a_1"], chi_2=-parameters["a_2"]))
    aligned = mlgw_bns_individual_modes(freqs, **{key: params[key] for key in SOURCE_KEYS})
    for key, pols in precessing.items():
        for pol in ("plus", "cross"):
            ratio = pols[pol] / aligned[key][pol]
            np.testing.assert_allclose(np.abs(ratio), 1, atol=1e-6)
            np.testing.assert_allclose(ratio / ratio[0], 1, atol=1e-2)


def test_mlgw_bns_precessing_modes_out_of_range():
    pytest.importorskip("jax")
    pytest.importorskip("mlgw_bns")
    with pytest.raises(ValueError):
        _precessing_modes(np.geomspace(10.0, 1000.0, 20), dict(PRECESSING, a_1=0.9, tilt_1=0.0))


def test_lal_conversion_gives_aligned_components():
    from bilby_xG.conversion import convert_to_lal_binary_neutron_star_parameters

    converted, added = convert_to_lal_binary_neutron_star_parameters(dict(
        mass_1=1.4, mass_2=1.3, a_1=0.3, tilt_1=1.2, a_2=0.1, tilt_2=0.5, phi_12=0.0,
        phi_jl=0.0, lambda_1=0.0, lambda_2=0.0, luminosity_distance=40.0, theta_jn=0.1,
        phase=0.0))
    assert converted["chi_1"] == pytest.approx(0.3 * np.cos(1.2))
    assert "chi_2" in added


def test_teobresums_spa_modes():
    pytest.importorskip("EOBRun_module")
    from bilby_xG.source import (
        convert_to_teobresums_parameters,
        teobresums_spa_individual_modes,
    )

    params, added = convert_to_teobresums_parameters(dict(PARAMETERS))
    assert "use_spins" not in added and params["chi1"] == PARAMETERS["chi_1"]
    kwargs = {key: params[key] for key in SOURCE_KEYS}
    freqs = np.geomspace(20, 1024, 200)
    modes = teobresums_spa_individual_modes(
        freqs, **kwargs, reference_frequency=20.0, minimum_frequency=20.0,
        maximum_frequency=1024.0, mode_array=[[2, 2], [3, 3]],
        frequency_bin_edges=freqs)
    assert set(modes) == {"2,2", "3,3"}
    h22, h33 = (np.abs(modes[k]["plus"]) for k in ("2,2", "3,3"))
    assert np.all(h22 > 0) and np.all(h22 > h33)
    # SPA amplitude of the (2,2) inspiral falls as f^(-7/6)
    slope = np.polyfit(np.log(freqs[:50]), np.log(h22[:50]), 1)[0]
    assert slope == pytest.approx(-7 / 6, abs=0.05)
