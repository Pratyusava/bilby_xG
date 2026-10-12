"""Tests for the motion of the detectors in the Solar System."""
import numpy as np
import pytest

from bilby.core.utils import speed_of_light
from bilby_xG.networks import InterferometerList
from bilby_xG.motion import (
    Center,
    EarthEphemeris,
    default_ephemeris,
    detector_position,
    doppler_factor,
    earth_barycentric_position_velocity,
    geocentre_delay,
    hermite_interpolate,
    precession_matrix,
    resolve_center,
    sky_direction,
)
from bilby_xG.utils import calculate_time_to_merger_for_any_mode

TC = 1187008882.4


def test_ephemeris_matches_astropy():
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time

    times = TC + np.linspace(0, 3e7, 5)
    position, velocity = earth_barycentric_position_velocity(times)
    ap_position, ap_velocity = get_body_barycentric_posvel("earth", Time(times, format="gps"))
    # astropy uses TDB, we neglect TDB - TT (< 2 ms, i.e. < 60 m)
    np.testing.assert_allclose(position, ap_position.xyz.to("m").value.T, rtol=0, atol=100)
    np.testing.assert_allclose(velocity, ap_velocity.xyz.to("m/s").value.T, rtol=0, atol=1e-2)


def test_interpolation_error():
    rng = np.random.default_rng(1)
    times = TC + rng.uniform(-3 * 86400, 86400, 2000)
    exact, _ = earth_barycentric_position_velocity(times)
    assert np.abs(EarthEphemeris().position(times) - exact).max() < 0.05


def test_lazy_extension():
    ephemeris = EarthEphemeris(step=600.0)
    ephemeris.position(np.array([TC]))
    ephemeris.position(np.array([TC + 5 * 86400]))
    ephemeris.position(np.array([TC - 86400]))
    times = np.linspace(TC - 86400, TC + 5 * 86400, 1001)
    assert ephemeris.t0 <= times[0]
    assert len(ephemeris._positions) <= 6 * 144 + 3
    np.testing.assert_allclose(ephemeris.position(times),
                               EarthEphemeris(step=600.0).position(times), rtol=0, atol=1e-4)
    table = ephemeris.table(TC, TC + 3600)
    inside = np.linspace(TC, TC + 3600, 20)
    np.testing.assert_allclose(hermite_interpolate(inside, **table),
                               ephemeris.position(inside), rtol=0, atol=1e-4)



RA, DEC, PSI = 1.2, -0.3, 0.7


def test_geocentre_delay():
    ttc = np.geomspace(1e-3, 1e5, 200)
    delay = geocentre_delay(RA, DEC, TC, ttc)
    assert delay.shape == ttc.shape
    _, velocity = earth_barycentric_position_velocity(TC)
    projected_velocity = sky_direction(RA, DEC) @ precession_matrix(TC) @ velocity
    uniform = projected_velocity * ttc / speed_of_light
    # first order: a uniform motion, whose Doppler factor is doppler_factor
    # (to the ~cm interpolation error, 1e-10 s)
    np.testing.assert_allclose(delay[ttc < 100], uniform[ttc < 100], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(doppler_factor(RA, DEC, TC),
                               1 - projected_velocity / speed_of_light)
    # the curvature: centripetal acceleration ~ 6 mm/s^2 for a day
    assert 1e-3 < np.abs(delay - uniform)[-1] < 0.1
    # without orbital motion the geocentre stays at the center
    np.testing.assert_array_equal(geocentre_delay(RA, DEC, TC, ttc, orbital_motion=False), 0)


def _response(ifo, frequencies, ttc, **kwargs):
    fps, _ = ifo.frequency_dependent_antenna_response(
        RA, DEC, TC, PSI, frequencies, TC - 1000, ttc, finite_size=False, **kwargs)
    return fps


def test_delay_follows_the_detector_position():
    """The response's delay is -n.(R_d(t_f) - C)/c, with R_d the
    barycentric position of the vertex."""
    ifo = InterferometerList(["CE"])[0]
    frequencies = np.geomspace(5.0, 1024.0, 400)
    ttc = calculate_time_to_merger_for_any_mode(frequencies, 1.45, 1.3, mode=2, safety=1)
    static = _response(ifo, frequencies, ttc, earth_rotation_time_delay=False,
                       orbital_motion=False)
    direction = sky_direction(RA, DEC)
    # the vertex relative to the geocentre, both at coalescence
    static_delay = -direction @ (detector_position(ifo.geometry.vertex, TC, axes_time=TC)
                                 - detector_position(np.zeros(3), TC, axes_time=TC)) / speed_of_light
    for center in [Center.geocenter(), Center.interferometer(ifo),
                   Center.geocenter(reference_time=TC - 3e4), Center.fixed([1e11, 2e10, -3e10])]:
        moving = _response(ifo, frequencies, ttc, center=center)
        position = detector_position(ifo.geometry.vertex, TC - ttc, axes_time=TC)
        delay = -(position - center.position(TC)) @ direction / speed_of_light
        expected = np.exp(-2j * np.pi * frequencies * (delay - static_delay))
        # 1e-7 rad: the rounding of the ~1e3 s * 1e3 Hz phases
        np.testing.assert_allclose(moving / static, expected, rtol=0, atol=1e-6)


def test_center_only_shifts_the_time():
    ifos = InterferometerList(["ET-EMR"])
    frequencies = np.geomspace(3.0, 2048.0, 300)
    ttc = calculate_time_to_merger_for_any_mode(frequencies, 1.45, 1.3, mode=2, safety=1)
    weighted = Center.weighted(ifos, {"ET-EMR1": 1.0, "ET-EMR2": 2.0, "ET-EMR3": 3.0})
    for center in [resolve_center("ET-EMR2", ifos), weighted]:
        shifts = []
        for ifo in ifos:
            ratio = (_response(ifo, frequencies, ttc, center=center)
                     / _response(ifo, frequencies, ttc))
            shift = -np.unwrap(np.angle(ratio)) / (2 * np.pi * frequencies)
            np.testing.assert_allclose(shift, shift[0], rtol=0, atol=1e-12)
            shifts.append(shift[0])
        # the same shift, direction . center / c, for all detectors
        expected = sky_direction(RA - _gmst(TC), DEC) @ center.offset / speed_of_light
        np.testing.assert_allclose(shifts, expected, rtol=0, atol=1e-12)
    # the arrival time at a detector's own vertex has no delay there
    ifo = ifos[1]
    delay = geocentre_delay(RA, DEC, TC, [0.0], center=Center.interferometer(ifo))
    vertex = sky_direction(RA - _gmst(TC), DEC) @ ifo.geometry.vertex / speed_of_light
    np.testing.assert_allclose(delay - vertex, 0, atol=1e-12)


def _gmst(time):
    from bilby_cython.geometry import greenwich_mean_sidereal_time
    return greenwich_mean_sidereal_time(time)


def test_centers():
    ifos = InterferometerList(["CE", "CE20"])
    assert resolve_center(None).label == "geocenter"
    assert resolve_center("geocenter").label == "geocenter"
    assert resolve_center("CE20", ifos).label == "CE20"
    with pytest.raises(ValueError):
        resolve_center("V1", ifos)
    np.testing.assert_allclose(
        Center.weighted(ifos, [1.0, 3.0]).offset,
        (ifos[0].geometry.vertex + 3 * ifos[1].geometry.vertex) / 4)
    # a center pinned at a reference time is a fixed barycentric point
    pinned = Center.interferometer(ifos[0], reference_time=TC - 1e4)
    fixed = resolve_center(precession_matrix(TC).T @ pinned.position(TC))
    for time in (TC, TC + 3e5):
        np.testing.assert_allclose(fixed.position(time), pinned.position(time), rtol=0,
                                   atol=1e-3)


def test_response_with_and_without_orbit():
    """Switching the orbit on multiplies the response by
    exp(-2 pi i f geocentre_delay), the same for every interferometer."""
    ifos = InterferometerList(["ET-EMR"])
    frequencies = np.geomspace(3.0, 2048.0, 300)
    parameters = dict(mass_1=1.45, mass_2=1.3, chi_1=0.0, chi_2=0.0, ra=RA, dec=DEC,
                      psi=PSI, geocent_time=TC, luminosity_distance=40.0)
    polarizations = {"plus": np.ones_like(frequencies, dtype=complex),
                     "cross": 1j * np.ones_like(frequencies)}
    ttc = calculate_time_to_merger_for_any_mode(frequencies, 1.45, 1.3, mode=2, safety=1)
    expected = np.exp(-2j * np.pi * frequencies * geocentre_delay(RA, DEC, TC, ttc))
    for ifo in ifos:
        kwargs = dict(waveform_polarizations=polarizations, parameters=parameters,
                      start_time=TC - 1000, frequencies=frequencies)
        on = ifo.get_detector_response_for_frequency_dependent_antenna_response(**kwargs)
        off = ifo.get_detector_response_for_frequency_dependent_antenna_response(
            orbital_motion=False, **kwargs)
        np.testing.assert_allclose(on / off, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("orbital_motion", [True, False])
def test_jax_geocentre_delay(orbital_motion):
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from bilby_xG.batched import _geocentre_delay

    jax.config.update("jax_enable_x64", True)
    ifo = InterferometerList(["CE"])[0]
    ttc = np.geomspace(1e-3, 5e4, 100)
    table = dict(default_ephemeris.table(TC - 86400, TC + 86400), precession=precession_matrix(TC))
    for center in [Center.geocenter(), Center.interferometer(ifo),
                   Center.fixed([1e11, 2e10, -3e10])]:
        if center.offset is not None:
            jax_center = dict(offset=center.offset)
        else:
            jax_center = dict(position=center.position(TC))
        delay = _geocentre_delay(jnp.float64(RA), jnp.float64(DEC), jnp.float64(TC),
                                 jnp.float64(_gmst(TC)), jnp.asarray(ttc), table, jax_center,
                                 orbital_motion)
        np.testing.assert_allclose(
            np.asarray(delay),
            geocentre_delay(RA, DEC, TC, ttc, center=center, orbital_motion=orbital_motion),
            rtol=0, atol=1e-12)


def test_uniform_part_is_a_doppler_rescaling():
    """With the orbital motion, the signal matches the one without it with
    masses and distance multiplied by doppler_factor (and not otherwise)."""
    pytest.importorskip("lalsimulation")
    import bilby
    from scipy.optimize import minimize_scalar

    ifo = InterferometerList(["CE"])[0]
    frequencies = np.geomspace(20.0, 512.0, 100_000)
    weights = np.gradient(frequencies) / \
        ifo.power_spectral_density.get_power_spectral_density_array(frequencies)
    ra, dec, tc = 1.0, 0.5, TC + 1e7

    def strain(factor, orbital_motion, dt=0.0):
        masses = dict(mass_1=1.48 * factor, mass_2=1.27 * factor,
                      luminosity_distance=40.0 * factor)
        pols = bilby.gw.source.binary_black_hole_frequency_sequence(
            frequencies, **masses, a_1=0.0, tilt_1=0.0, phi_12=0.0, a_2=0.0, tilt_2=0.0,
            phi_jl=0.0, theta_jn=0.5, phase=0.0, waveform_approximant="IMRPhenomD",
            reference_frequency=50.0, frequencies=frequencies)
        parameters = dict(masses, chi_1=0.0, chi_2=0.0, ra=ra, dec=dec, psi=0.3,
                          geocent_time=tc + dt)
        return ifo.get_detector_response_for_frequency_dependent_antenna_response(
            pols, parameters, tc - 1e3, frequencies, orbital_motion=orbital_motion)

    def mismatch(a, b):
        return 1 - np.abs(np.sum(weights * np.conj(a) * b)) / np.sqrt(
            np.sum(weights * np.abs(a) ** 2) * np.sum(weights * np.abs(b) ** 2))

    signal = strain(1.0, True)

    def best(factor):
        return minimize_scalar(lambda dt: mismatch(signal, strain(factor, False, dt * 1e-4)),
                               bracket=(-1, 1)).fun

    factor = doppler_factor(ra, dec, tc)
    assert best(factor) < 1e-6
    assert best(1.0) > 1e-3
    assert best(1 / factor) > 1e-3


def test_response_uses_the_waveform_time_to_merger():
    """A "time_to_merger" entry replaces the 2PN estimate, per mode."""
    ifo = InterferometerList(["ET_1L_IT"])[0]
    frequencies = np.geomspace(3.0, 2048.0, 300)
    parameters = dict(mass_1=1.45, mass_2=1.3, chi_1=0.0, chi_2=0.0, ra=RA, dec=DEC,
                      psi=PSI, geocent_time=TC, luminosity_distance=40.0)
    plus = np.ones_like(frequencies, dtype=complex)
    ttc = calculate_time_to_merger_for_any_mode(frequencies, 1.45, 1.3, mode=2, safety=1)
    own = 0.9 * ttc

    def response(polarizations, times_to_coalescence=None):
        if times_to_coalescence is None:
            return ifo.get_detector_response_for_frequency_dependent_antenna_response(
                polarizations, parameters, TC - 1000, frequencies)
        fps, fcs = ifo.frequency_dependent_antenna_response(
            RA, DEC, TC, PSI, frequencies, TC - 1000, times_to_coalescence)
        return fps + 1j * fcs

    expected = response(None, own)
    for polarizations in [{"plus": plus, "cross": 1j * plus, "time_to_merger": own},
                          {"2,2": {"plus": plus, "cross": 1j * plus, "time_to_merger": own}}]:
        np.testing.assert_allclose(response(polarizations), expected, rtol=1e-12)
    np.testing.assert_allclose(response({"plus": plus, "cross": 1j * plus}),
                               response(None, ttc), rtol=1e-12)
    # the same shared cache serves modes with the same m and different maps
    shared = {}
    both = ifo.get_detector_response_for_frequency_dependent_antenna_response(
        {"2,2": {"plus": plus, "cross": 1j * plus, "time_to_merger": own},
         "3,2": {"plus": plus, "cross": 1j * plus, "time_to_merger": ttc}},
        parameters, TC - 1000, frequencies, shared=shared)
    np.testing.assert_allclose(both, expected + response(None, ttc), rtol=1e-12)


def test_uniform_part_is_a_doppler_rescaling_with_the_waveform_time():
    """With the waveform's own time-frequency map (mlgw_bns, referred to the
    merger), the orbital motion of the detector is a Doppler rescaling of the
    masses and distance up to the curvature of the orbit: tiny above 30 Hz.
    The 2PN map, referred to the point-particle coalescence, is not."""
    pytest.importorskip("mlgw_bns")
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes

    ifo = InterferometerList(["ET_1L_IT"])[0]
    frequencies = np.geomspace(30.0, 1000.0, 10_000)
    ra, dec, tc = 1.0, 0.5, TC + 1e7
    factor = doppler_factor(ra, dec, tc)

    def strain(scale, orbital_motion, own_time=True):
        parameters, _ = convert_to_mlgw_bns_parameters(dict(
            chirp_mass=1.2 * scale, mass_ratio=0.9, chi_1=0.0, chi_2=0.0, lambda_1=400.0,
            lambda_2=600.0, luminosity_distance=40.0 * scale, theta_jn=0.5, phase=0.3,
            ra=ra, dec=dec, psi=0.3, geocent_time=tc))
        pols = mlgw_bns_individual_modes(frequencies, **parameters, mode_array=[[2, 2]])
        if not own_time:
            del pols["2,2"]["time_to_merger"]
        return ifo.get_detector_response_for_frequency_dependent_antenna_response(
            pols, parameters, tc, frequencies, orbital_motion=orbital_motion)

    def residual(own_time):
        ratio = strain(1.0, True, own_time) / strain(factor, False, own_time)
        return np.max(np.abs(np.log(ratio)))

    assert residual(True) < 1e-4
    assert residual(False) > 3e-4
