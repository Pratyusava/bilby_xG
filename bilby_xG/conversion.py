# Licensed under an MIT style license -- see LICENSE

"""Parameter-conversion helpers.

Adds support for sampling in ``log10_luminosity_distance`` (base-10 log of the
luminosity distance in Mpc) on top of bilby's standard CBC conversions. Pass
these converters explicitly, e.g.::

    bilby.gw.WaveformGenerator(
        ...,
        parameter_conversion=bilby_xG.conversion.convert_to_lal_binary_black_hole_parameters,
    )
"""
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters as _convert_bbh,
    convert_to_lal_binary_neutron_star_parameters as _convert_bns,
)

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]


def _add_log10_luminosity_distance(parameters):
    """Return a copy with ``luminosity_distance`` set from its base-10 log.

    Only acts when ``log10_luminosity_distance`` is present and an explicit
    ``luminosity_distance`` is not; otherwise the parameters are returned
    unchanged.
    """
    if ("log10_luminosity_distance" in parameters
            and "luminosity_distance" not in parameters):
        parameters = parameters.copy()
        parameters["luminosity_distance"] = \
            10 ** parameters["log10_luminosity_distance"]
    return parameters


def convert_to_lal_binary_black_hole_parameters(parameters):
    """As bilby's BBH converter, but also accepts ``log10_luminosity_distance``."""
    return _convert_bbh(_add_log10_luminosity_distance(parameters))


def convert_to_lal_binary_neutron_star_parameters(parameters):
    """As bilby's BNS converter, but also accepts ``log10_luminosity_distance``."""
    return _convert_bns(_add_log10_luminosity_distance(parameters))
