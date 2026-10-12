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


def test_mlgw_bns_time_to_merger():
    """Each mode carries -t(f) = phi'(f) / 2 pi, close to the 2PN estimate
    early in the inspiral."""
    pytest.importorskip("mlgw_bns")
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes
    from bilby_xG.utils import calculate_time_to_merger_for_any_mode

    params, _ = convert_to_mlgw_bns_parameters(dict(PARAMETERS))
    freqs = np.linspace(40.0, 40.1, 1001)
    modes = mlgw_bns_individual_modes(freqs, **{key: params[key] for key in SOURCE_KEYS})
    for key, pols in modes.items():
        emm = int(key.split(",")[1])
        phase = np.unwrap(np.angle(pols["plus"] - 1j * pols["cross"]))
        np.testing.assert_allclose(pols["time_to_merger"][1:-1],
                                   np.gradient(phase, freqs)[1:-1] / (2 * np.pi), rtol=1e-6)
        post_newtonian = calculate_time_to_merger_for_any_mode(
            freqs, params["mass_1"], params["mass_2"], params["chi1z"], params["chi2z"],
            mode=emm, safety=1)
        np.testing.assert_allclose(pols["time_to_merger"], post_newtonian, rtol=1e-2)
