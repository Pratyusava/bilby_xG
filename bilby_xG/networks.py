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

from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw.detector import networks as _networks
from bilby.gw.detector.psd import PowerSpectralDensity as _PowerSpectralDensity

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
    :class:`bilby.gw.detector.networks.TriangularInterferometer`.
    """

    def __init__(self, name, power_spectral_density, minimum_frequency,
                 maximum_frequency, length, latitude, longitude, elevation,
                 xarm_azimuth, yarm_azimuth, xarm_tilt=0., yarm_tilt=0.):
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

            xarm_azimuth += 240
            yarm_azimuth += 240
            latitude += np.arctan(
                length * np.sin(xarm_azimuth * np.pi / 180) * 1e3
                / utils.radius_of_earth) * 180 / np.pi
            longitude += np.arctan(
                length * np.cos(xarm_azimuth * np.pi / 180) * 1e3
                / utils.radius_of_earth) * 180 / np.pi


def load_interferometer(filename):
    """Load a bilby_xG interferometer from a ``*.interferometer`` file.

    The file format is the same as bilby's; ``PowerSpectralDensity`` entries
    additionally resolve against the noise curves shipped with bilby_xG.
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


def get_empty_interferometer(name):
    """Get a bilby_xG interferometer with standard parameters by name.

    Detector definitions shipped with bilby_xG (``CE``, ``CE20``, ``ET-EMR``,
    ``ET_1L_IT``, ``ET_1L_DE``) take precedence; any other name (e.g. ``H1``,
    ``L1``, ``V1``, ``ET``) is loaded from bilby's built-in definitions as a
    bilby_xG interferometer.

    Parameters
    ==========
    name: str
        Interferometer identifier.

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
        return load_interferometer(filename)
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
