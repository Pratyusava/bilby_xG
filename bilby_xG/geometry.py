# Licensed under an MIT style license -- see LICENSE

"""Detector geometry for next-generation interferometers.

Extends :class:`bilby.gw.detector.geometry.InterferometerGeometry` with the
symmetrised single-arm outer products ``0.5 * x_i x_j`` and ``0.5 * y_i y_j``
needed by the finite-size, frequency-dependent antenna response (Baral et al.
2023, arXiv:2304.09889). They are cheap 3x3 contractions, recomputed from the
current arm vectors on each access, so they always track changes to the
detector geometry without any cache-invalidation bookkeeping.
"""
import numpy as np

from bilby.gw.detector.geometry import (
    InterferometerGeometry as _InterferometerGeometry,
)

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]


class InterferometerGeometry(_InterferometerGeometry):
    """Interferometer geometry with per-arm detector tensors.

    Parameters are as in
    :class:`bilby.gw.detector.geometry.InterferometerGeometry`.
    """

    @property
    def xx(self):
        """``0.5 * einsum('i,j->ij', x, x)`` for the X arm unit vector."""
        return 0.5 * np.einsum("i,j->ij", self.x, self.x)

    @property
    def yy(self):
        """``0.5 * einsum('i,j->ij', y, y)`` for the Y arm unit vector."""
        return 0.5 * np.einsum("i,j->ij", self.y, self.y)
