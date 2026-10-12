# Licensed under an MIT style license -- see LICENSE

"""Detector networks for next-generation interferometers.

Mirrors :mod:`bilby.gw.detector.networks` but returns
:class:`bilby_xG.interferometer.Interferometer` instances, whose antenna
response can be evaluated with the frequency-dependent, finite-size,
Earth-rotation-aware method needed for next-generation detectors.

Detector definitions shipped with bilby_xG (the next-generation ``CE`` and
``CE20``; the latest Einstein Telescope configurations ``ET-EMR``,
``ET_1L_IT`` and ``ET_1L_DE``) take precedence; any other name falls back
to the definitions shipped with bilby, loaded as bilby_xG interferometers.
Amplitude spectral densities referenced by the detector files are resolved
first from bilby_xG's ``data/noise_curves``, then from bilby's built-in
noise curves.

.. note:: bilby_xG ships an updated 40 km ``CE`` definition that takes
   precedence over bilby's built-in ``CE``. Use a different name if you need
   the original.

The Einstein Telescope definitions (``ET-EMR``, ``ET_1L_IT``, ``ET_1L_DE``)
default to the cryogenic HFLF baseline sensitivity; the high-frequency-only
PSDs are also shipped. See the README for how to switch between them.
"""
import os

import numpy as np

from bilby.core.utils import logger
from bilby.gw.detector import networks as _networks
from bilby.gw.detector.psd import PowerSpectralDensity as _PowerSpectralDensity
from bilby.gw.utils import get_vertex_position_geocentric

from .interferometer import Interferometer
from .utils import detector_file, noise_curve_file

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]


def _resolve_noise_curve(kwargs):
    """Resolve ``psd_file``/``asd_file`` names against bilby_xG's data.

    Bare file names that match a shipped noise curve are replaced by their
    absolute path; anything else is passed through unchanged so bilby's own
    file lookup applies.
    """
    resolved = dict(kwargs)
    for key in ("psd_file", "asd_file"):
        value = resolved.get(key)
        if value is not None and not os.path.isfile(value):
            ours = noise_curve_file(os.path.basename(value))
            if ours is not None:
                resolved[key] = ours
    return resolved


class PowerSpectralDensity(_PowerSpectralDensity):
    """As :class:`bilby.gw.detector.psd.PowerSpectralDensity`, but
    ``psd_file``/``asd_file`` names also resolve against the noise curves
    shipped with bilby_xG."""

    def __init__(self, **kwargs):
        super(PowerSpectralDensity, self).__init__(**_resolve_noise_curve(kwargs))


class TriangularInterferometer(_networks.TriangularInterferometer):
    """A triangular interferometer built from bilby_xG interferometers.

    Parameters are as in
    :class:`bilby.gw.detector.networks.TriangularInterferometer`, with the
    addition of ``colocated``: if True, the three channels share the same
    latitude, longitude and elevation rather than being placed at the
    triangle vertices.

    Each next channel is at the end of the current channel's x arm (a
    straight arm of ``length``, on the WGS84 ellipsoid), with both arms
    rotated by 120 degrees about the normal to the detector plane, so that
    every channel's arms lie along the edges of the triangle. This is the
    construction of LAL (LIGO-T1400308) and of bilby-dev/bilby#894; bilby up
    to at least 2.8 instead rotates the arms by 240 degrees, which leaves
    one arm of the second and third channels off the triangle.
    """

    def __init__(self, name, power_spectral_density, minimum_frequency,
                 maximum_frequency, length, latitude, longitude, elevation,
                 xarm_azimuth, yarm_azimuth, xarm_tilt=0., yarm_tilt=0.,
                 colocated=False):
        # Reimplements the upstream constructor so the three channels are
        # bilby_xG Interferometers with the frequency-dependent response.
        list.__init__(self)
        self.name = name
        if isinstance(power_spectral_density, _PowerSpectralDensity):
            power_spectral_density = [power_spectral_density] * 3
        if isinstance(minimum_frequency, (int, float)):
            minimum_frequency = [minimum_frequency] * 3
        if isinstance(maximum_frequency, (int, float)):
            maximum_frequency = [maximum_frequency] * 3

        for ii in range(3):
            self.append(Interferometer(
                "{}{}".format(name, ii + 1), power_spectral_density[ii],
                minimum_frequency[ii], maximum_frequency[ii], length, latitude,
                longitude, elevation, xarm_azimuth, yarm_azimuth, xarm_tilt,
                yarm_tilt))
            (latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
             xarm_tilt, yarm_tilt) = _next_vertex(self[-1], colocated)


def _next_vertex(ifo, colocated=False):
    """Position and arm orientation of the channel after ``ifo`` in a
    triangular detector: at the end of its x arm (or at the same vertex, if
    ``colocated``), with both arms rotated by 120 degrees about the normal
    to the detector plane.

    Returns latitude and longitude [deg], elevation [m], the x- and y-arm
    azimuths [deg] and tilts [rad], as taken by
    :class:`bilby_xG.interferometer.Interferometer`.
    """
    x_arm, y_arm = ifo.geometry.x, ifo.geometry.y
    if colocated:
        latitude, longitude, elevation = (
            ifo.latitude_radians, ifo.longitude_radians, ifo.elevation)
    else:
        vertex = get_vertex_position_geocentric(
            ifo.latitude_radians, ifo.longitude_radians, ifo.elevation)
        latitude, longitude, elevation = _geocentric_to_geodetic(
            vertex + ifo.length * 1e3 * x_arm)

    normal = np.cross(x_arm, y_arm)
    normal /= np.linalg.norm(normal)
    angle = 2 * np.pi / 3

    def rotate(arm):
        # Rodrigues' formula, for an arm orthogonal to the normal
        return arm * np.cos(angle) + np.cross(normal, arm) * np.sin(angle)

    # the local frame in which bilby defines azimuths and tilts
    east = np.array([-np.sin(longitude), np.cos(longitude), 0])
    north = np.array([-np.sin(latitude) * np.cos(longitude),
                      -np.sin(latitude) * np.sin(longitude), np.cos(latitude)])
    up = np.array([np.cos(latitude) * np.cos(longitude),
                   np.cos(latitude) * np.sin(longitude), np.sin(latitude)])
    azimuths, tilts = [], []
    for arm in (rotate(x_arm), rotate(y_arm)):
        azimuths.append(np.degrees(np.arctan2(arm @ north, arm @ east)))
        tilts.append(np.arcsin(arm @ up))
    return (np.degrees(latitude), np.degrees(longitude), elevation,
            *azimuths, *tilts)


def _geocentric_to_geodetic(position):
    """WGS84 latitude and longitude [rad] and elevation [m] of a geocentric
    ``position`` [m]: the exact inverse of
    :func:`bilby.gw.utils.get_vertex_position_geocentric`, by Borkowski's
    method (as in bilby-dev/bilby#894)."""
    semi_major_axis = 6378137
    semi_minor_axis = 6356752.314
    x_comp, y_comp, z_comp = position
    r = np.hypot(x_comp, y_comp)
    focal = semi_major_axis ** 2 - semi_minor_axis ** 2
    e = (semi_minor_axis * abs(z_comp) - focal) / (semi_major_axis * r)
    f = (semi_minor_axis * abs(z_comp) + focal) / (semi_major_axis * r)
    p = 4 / 3 * (e * f + 1)
    q = 2 * (e ** 2 - f ** 2)
    d = p ** 3 + q ** 2
    v = np.cbrt(np.sqrt(d) - q) - np.cbrt(np.sqrt(d) + q)
    g = (np.sqrt(e ** 2 + v) + e) / 2
    t = np.sqrt(g ** 2 + (f - v * g) / (2 * g - e)) - g
    latitude = np.sign(z_comp) * np.arctan(
        semi_major_axis * (1 - t ** 2) / (2 * semi_minor_axis * t))
    longitude = np.arctan2(y_comp, x_comp)
    elevation = ((r - semi_major_axis * t) * np.cos(latitude)
                 + (z_comp - semi_minor_axis * np.sign(z_comp)) * np.sin(latitude))
    return latitude, longitude, elevation


def load_interferometer(filename, **overrides):
    """Load a bilby_xG interferometer from a ``*.interferometer`` file.

    The file format is the same as bilby's; ``PowerSpectralDensity`` entries
    additionally resolve against the noise curves shipped with bilby_xG.
    ``overrides`` replace or add constructor arguments, e.g.
    ``colocated=True`` for a triangular detector.
    """
    parameters = dict()
    with open(filename, "r") as parameter_file:
        for line in parameter_file.readlines():
            if line[0] == "#" or line[0] == "\n":
                continue
            split_line = line.split("=")
            key = split_line[0].strip()
            value = eval(
                "=".join(split_line[1:]),
                {"PowerSpectralDensity": PowerSpectralDensity},
            )
            parameters[key] = value
    parameters.update(overrides)
    shape = parameters.pop("shape", "L")
    if shape.lower() in ["l", "ligo"]:
        ifo = Interferometer(**parameters)
        logger.debug("Assuming L shape for {}".format(parameters.get("name")))
    elif shape.lower() in ["triangular", "triangle"]:
        ifo = TriangularInterferometer(**parameters)
    else:
        raise IOError(
            "{} could not be loaded. Invalid parameter 'shape'.".format(filename))
    return ifo


def get_empty_interferometer(name, **overrides):
    """Get a bilby_xG interferometer with standard parameters by name.

    Detector definitions shipped with bilby_xG (``CE``, ``CE20``, ``ET-EMR``,
    ``ET_1L_IT``, ``ET_1L_DE``) take precedence; any other name (e.g. ``H1``,
    ``L1``, ``V1``, ``ET``) is loaded from bilby's built-in definitions as a
    bilby_xG interferometer.

    Parameters
    ==========
    name: str
        Interferometer identifier.
    overrides:
        Passed to :func:`load_interferometer`.

    Returns
    =======
    interferometer: bilby_xG.interferometer.Interferometer
        Interferometer instance
    """
    filename = detector_file(name)
    if filename is None:
        filename = os.path.join(
            os.path.dirname(_networks.__file__), "detectors",
            "{}.interferometer".format(name))
    try:
        return load_interferometer(filename, **overrides)
    except OSError:
        raise ValueError("Interferometer {} not implemented".format(name))


class InterferometerList(_networks.InterferometerList):
    """A list of bilby_xG Interferometer objects.

    As :class:`bilby.gw.detector.networks.InterferometerList`, but names are
    resolved through :func:`bilby_xG.networks.get_empty_interferometer`, so
    string entries load the bilby_xG detector definitions and return
    interferometers with the frequency-dependent antenna response.
    """

    def __init__(self, interferometers):
        list.__init__(self)
        if isinstance(interferometers, str):
            raise TypeError("Input must not be a string")
        for ifo in interferometers:
            if isinstance(ifo, str):
                ifo = get_empty_interferometer(ifo)
            if not isinstance(ifo, (_networks.Interferometer,
                                    _networks.TriangularInterferometer)):
                raise TypeError(
                    "Input list of interferometers are not all Interferometer objects")
            else:
                self.append(ifo)
        self._check_interferometers()
