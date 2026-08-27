"""Tests for the network/detector layer and shipped-data discovery."""
import numpy as np

import bilby
from bilby_cython.geometry import time_delay_from_geocenter as _cython_tdfg

from bilby_xG.geometry import InterferometerGeometry
from bilby_xG.interferometer import Interferometer
from bilby_xG.networks import (
    InterferometerList,
    PowerSpectralDensity,
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
