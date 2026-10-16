"""Tests for the network/detector layer and shipped-data discovery."""
import numpy as np
import pytest

import bilby
from bilby_cython.geometry import time_delay_from_geocenter as _cython_tdfg

from bilby_xG.geometry import InterferometerGeometry
from bilby_xG.interferometer import Interferometer
from bilby_xG.networks import (
    InterferometerList,
    PowerSpectralDensity,
    TriangularInterferometer,
    get_empty_interferometer,
)


def test_interferometer_subclasses_bilby():
    ifo = get_empty_interferometer("CE")
    assert isinstance(ifo, Interferometer)
    assert isinstance(ifo, bilby.gw.detector.Interferometer)
    for method in [
        "_finite_size_factor",
        "frequency_dependent_antenna_response",
        "get_detector_response_for_frequency_dependent_antenna_response",
    ]:
        assert hasattr(ifo, method)


def test_geometry_properties():
    ifo = get_empty_interferometer("H1")
    assert isinstance(ifo.geometry, InterferometerGeometry)
    xx = ifo.geometry.xx
    yy = ifo.geometry.yy
    assert xx.shape == (3, 3)
    assert yy.shape == (3, 3)
    assert np.allclose(xx, 0.5 * np.einsum("i,j->ij", ifo.geometry.x, ifo.geometry.x))


def test_time_delay_backward_compatible():
    """vG=1 must reproduce the upstream geocentre time delay exactly."""
    ifo = get_empty_interferometer("H1")
    ra, dec, t = 1.2, -0.3, 1234567890.0
    expected = _cython_tdfg(ifo.geometry.vertex, ra, dec, t)
    assert np.isclose(ifo.time_delay_from_geocenter(ra, dec, t), expected)
    # vG scales the delay as 1 / vG.
    assert np.isclose(
        ifo.time_delay_from_geocenter(ra, dec, t, vG=0.5), expected / 0.5
    )


def test_shipped_detectors_resolve_by_name():
    ce20 = get_empty_interferometer("CE20")
    assert ce20.name == "CE20"
    assert ce20.length == 20
    ce = get_empty_interferometer("CE")
    assert ce.name == "CE"
    # bilby_xG ships the 40 km next-generation CE (takes precedence).
    assert ce.length == 40


def test_bilby_builtin_detectors_load_as_bilby_xG():
    """Names not shipped with bilby_xG fall back to bilby's definitions."""
    h1 = get_empty_interferometer("H1")
    assert isinstance(h1, Interferometer)
    assert h1.name == "H1"
    assert h1.length == 4


def test_interferometer_list_resolves_names():
    ifos = InterferometerList(["CE", "CE20", "H1"])
    assert [ifo.name for ifo in ifos] == ["CE", "CE20", "H1"]
    assert all(isinstance(ifo, Interferometer) for ifo in ifos)


def test_shipped_psd_resolves_by_bare_name():
    psd = PowerSpectralDensity(asd_file="ce20_asd.txt")
    assert psd.asd_array is not None
    assert np.all(np.isfinite(psd.asd_array[psd.asd_array > 0]))


def test_einstein_telescope_l_shaped_detectors_resolve_by_name():
    for name, lat in [("ET_1L_IT", 40 + 31.0 / 60), ("ET_1L_DE", 51.275)]:
        ifo = get_empty_interferometer(name)
        assert isinstance(ifo, Interferometer)
        assert ifo.name == name
        assert ifo.length == 15
        assert np.isclose(ifo.latitude, lat)
        # Ships with the HFLF baseline sensitivity down to 3 Hz.
        assert ifo.minimum_frequency == 3
        assert "HFLF" in ifo.power_spectral_density.psd_file
        psd = ifo.power_spectral_density.psd_array
        assert np.all(np.isfinite(psd[psd > 0]))


def test_einstein_telescope_hflf_psd_resolves_by_bare_name():
    for length in (10, 15):
        psd = PowerSpectralDensity(psd_file=f"ET_{length}_HFLF_psd.txt")
        assert psd.psd_array is not None
        assert np.all(np.isfinite(psd.psd_array[psd.psd_array > 0]))
        # HFLF curve extends below the high-frequency f_min.
        assert psd.frequency_array.min() <= 2


def test_einstein_telescope_triangle_resolves_by_name():
    et = get_empty_interferometer("ET-EMR")
    assert isinstance(et, bilby.gw.detector.TriangularInterferometer)
    assert len(et) == 3
    assert [ifo.name for ifo in et] == ["ET-EMR1", "ET-EMR2", "ET-EMR3"]
    assert all(isinstance(ifo, Interferometer) for ifo in et)
    assert all(ifo.length == 10 for ifo in et)
    assert all(ifo.minimum_frequency == 3 for ifo in et)
    assert all("HFLF" in ifo.power_spectral_density.psd_file for ifo in et)


@pytest.mark.parametrize("latitude, longitude, xarm_azimuth", [
    (50 + 43 / 60 + 23 / 3600, 5 + 55 / 60 + 14 / 3600, 250.5674),
    (-31.5, 115.9, 10.0),
])
def test_triangle_arms_lie_along_the_edges(latitude, longitude, xarm_azimuth):
    # an equilateral triangle of straight 10 km arms, in which each channel's
    # x arm ends at the next vertex and its y arm at the previous one
    et = TriangularInterferometer(
        "T", PowerSpectralDensity(psd_file="ET_10_HFLF_psd.txt"), 3, 4096, 10,
        latitude, longitude, 100.0, xarm_azimuth, xarm_azimuth + 60)
    vertices = [ifo.vertex for ifo in et]
    for i, ifo in enumerate(et):
        for arm, vertex in [(ifo.geometry.x, vertices[(i + 1) % 3]),
                            (ifo.geometry.y, vertices[(i - 1) % 3])]:
            edge = vertex - vertices[i]
            np.testing.assert_allclose(np.linalg.norm(edge), 10e3, rtol=1e-9)
            np.testing.assert_allclose(arm @ edge / 10e3, 1, atol=1e-12)


@pytest.mark.parametrize("latitude", [-89.0, -43.6, 0.0, 0.3, 50.7, 89.0])
def test_geocentric_to_geodetic_inverts_bilby(latitude):
    from bilby.gw.utils import get_vertex_position_geocentric
    from bilby_xG.networks import _geocentric_to_geodetic

    position = (np.radians(latitude), np.radians(-120.4), 1234.5)
    np.testing.assert_allclose(
        _geocentric_to_geodetic(get_vertex_position_geocentric(*position)),
        position, atol=1e-8)


def test_einstein_telescope_network():
    ifos = InterferometerList(["ET_1L_IT", "ET_1L_DE"])
    assert [ifo.name for ifo in ifos] == ["ET_1L_IT", "ET_1L_DE"]
    assert all(isinstance(ifo, Interferometer) for ifo in ifos)


def test_log10_luminosity_distance_conversion():
    from bilby_xG.conversion import convert_to_lal_binary_black_hole_parameters
    converted, _ = convert_to_lal_binary_black_hole_parameters(
        dict(mass_1=30.0, mass_2=30.0, log10_luminosity_distance=np.log10(400.0))
    )
    assert np.isclose(converted["luminosity_distance"], 400.0)


def test_colocated_override_keeps_one_vertex():
    import numpy as np
    from bilby_xG.networks import InterferometerList, get_empty_interferometer

    spread = InterferometerList(["ET-EMR"])
    colocated = InterferometerList([get_empty_interferometer("ET-EMR", colocated=True)])
    assert not np.allclose(spread[0].vertex, spread[1].vertex)
    for ifo in colocated[1:]:
        np.testing.assert_allclose(ifo.vertex, colocated[0].vertex)
    for a, b in zip(spread, colocated):
        np.testing.assert_allclose(a.detector_tensor, b.detector_tensor, atol=1e-2)


def test_read_frequency_domain_data_chunked(tmp_path):
    import numpy as np
    from bilby_xG.networks import InterferometerList
    from bilby_xG.utils import read_frequency_domain_data_chunked

    ifos = InterferometerList(["ET-EMR"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=256, duration=4, start_time=0)
    ifos.save_data(str(tmp_path))
    ifo = ifos[0]
    strain = read_frequency_domain_data_chunked(
        tmp_path / f"{ifo.name}_frequency_domain_data.dat", ifo.frequency_array,
        tmp_path / "strain.npy", chunksize=37)
    np.testing.assert_allclose(strain, ifo.frequency_domain_strain)
    assert isinstance(strain, np.memmap)


def test_detector_independent_response_is_shared():
    """Reusing one interferometer's wave frame for the others changes nothing."""
    ifos = InterferometerList(["ET-EMR"])
    frequencies = np.geomspace(3.0, 2048.0, 50)
    parameters = dict(mass_1=1.45, mass_2=1.3, chi_1=0.01, chi_2=0.0, ra=3.4,
                      dec=-0.4, psi=1.0, geocent_time=1187008882.4,
                      luminosity_distance=200.0, vG=0.99)
    polarizations = {key: {"plus": np.exp(1j * frequencies * k),
                           "cross": np.exp(2j * frequencies * k)}
                     for k, key in enumerate(["2,2", "2,1", "3,3", "3,2"], 1)}
    for flags in [(True, True, True), (False, False, True), (True, False, False)]:
        kwargs = dict(parameters=parameters, start_time=parameters["geocent_time"] - 1000,
                      frequencies=frequencies, earth_rotation_time_delay=flags[0],
                      earth_rotation_beam_patterns=flags[1], finite_size=flags[2])
        shared = {}
        for ifo in ifos:
            response = ifo.get_detector_response_for_frequency_dependent_antenna_response
            np.testing.assert_array_equal(response(polarizations, shared=shared, **kwargs),
                                          response(polarizations, **kwargs))
        assert set(shared) == {1, 2, 3}
