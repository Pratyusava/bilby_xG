"""Greedy relative-binning bin selection trained on many test points.

The GETBINS/BISECTBINSEARCH algorithm of Leslie, Dai & Pratten 2021
(arXiv:2109.09872) bisects candidate bins until each one's log-likelihood
error is below ``eta / n_bins``; here the error of a bin is the maximum over
a set of training points, so that the bins hold for a whole region of
parameter space rather than one test point.

Training points lie where the phase-maximized network mismatch with the
fiducial waveform first reaches ``target_mismatch`` along segments from the
fiducial parameters toward prior draws. The maximization over a constant
orbital phase shift (mode (l, m) rotated by exp(i m delta)) keeps a pure
phase offset, which relative binning reproduces exactly, from setting the
distance of the points.

The data are modelled as noise free, d = h_fid, so the per-bin errors are
functionals of the fiducial and test waveforms alone and the data stream is
not needed: it only enters the likelihood's summary data, once the bins are
fixed. The noise term dropped this way, <n | h_test - h_RB>, is smaller than
the kept one by ~sqrt(2 SNR^2 mismatch).

Everything is evaluated on a geometric frequency grid, so integrals over any
[grid[a], grid[b]] come from prefix sums in O(1) and index bisection is
bisection in log f. The results do not depend on the grid size (checked from
15k to 480k points for a BNS in ET from 3 and 10 Hz).

Diagnostics, all against the same grid quadrature (zero noise):
  * bins and held-out validation error as a function of the number of
    training points K (nested subsets),
  * the log-likelihood error of the selected bins against ln L_max - ln L
    for points along a few prior-draw segments (geometric in beta, raw and
    phase-aligned),
  * a histogram of the bin edges.

Choice of eta. In the Gaussian (Fisher) picture, with d = ln L_max - ln L =
r^2 / 2 over D_eff effective dimensions, -d ln L / d ln X = 2 d / D_eff, so
nested sampling with n_live live points resolves likelihood steps of
Delta l(d) = 2 d / (D_eff n_live), and the posterior distribution of d is
Gamma(D_eff / 2, 1). The sampler only ever sees L_RB = L e^epsilon and nested
sampling estimates the integral of L_RB over the prior without bias however
the errors reorder the points, so the evidence bias from relative binning
(wrong likelihoods and the compression they induce together) is
ln E_post[e^epsilon]. eta is rescaled until this, and the cruder E_post|epsilon|,
estimated on points at stratified Gamma(D_eff / 2) quantiles of d, is below
``evidence_tolerance``. |epsilon| / Delta l(d), which bears on the run's
trajectory and the posterior rather than the evidence, is reported too.
"""
import json
import multiprocessing
import os
import time

import numpy as np
from bilby.core.prior import Prior
from bilby.core.utils import logger
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import gamma

from .motion import resolve_center

CHUNK_SIZE = 250_000


def delta_log_likelihood(model, h_test):
    """ln L - ln L_max = -||h - d||^2 / 2, computed directly (no cancellation)."""
    distance2 = (4 / model.psd) * np.abs(h_test.sum(axis=1) - model.d) ** 2
    return -0.5 * float(np.trapezoid(distance2, x=model.freqs).sum())


def nested_sampling_resolution(d, effective_dimension, n_live):
    """Gaussian-L(X) dead-point spacing in ln L at ln L_max - ln L = d."""
    return 2 * np.asarray(d) / (effective_dimension * n_live)


def sampled_parameter_count(priors):
    """Parameters of a PriorDict that are sampled: not fixed, not constraints."""
    from bilby.core.prior import Constraint, DeltaFunction
    return sum(isinstance(p, Prior) and not isinstance(p, (Constraint, DeltaFunction))
               for p in priors.values())


def evidence_bias(errors):
    """max(|ln mean e^epsilon|, mean |epsilon|) over posterior-distributed points."""
    errors = np.asarray(errors)
    return max(abs(float(logsumexp(errors) - np.log(len(errors)))),
               float(np.mean(np.abs(errors))))


def trapezoid_prefix(g, freqs):
    """P[..., k] = integral of g from freqs[0] to freqs[k] (trapezoid rule)."""
    segments = 0.5 * (g[..., 1:] + g[..., :-1]) * np.diff(freqs)
    out = np.zeros(g.shape, dtype=segments.dtype)
    np.cumsum(segments, axis=-1, out=out[..., 1:])
    return out


class Projector:
    """Per-mode detector strain at arbitrary frequencies, through the
    waveform generator's ``frequency_bin_edges`` path (as the mode-by-mode
    relative-binning likelihood evaluates its fiducial waveform)."""

    def __init__(self, interferometers, waveform_generator, earth_rotation_time_delay=True,
                 earth_rotation_beam_patterns=True, finite_size=True, orbital_motion=True,
                 center=None):
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.mode_keys = [f"{l},{m}" for l, m in waveform_generator.waveform_arguments["mode_array"]]
        self.m = np.array([int(key.split(",")[1]) for key in self.mode_keys])
        self.response_kwargs = dict(
            earth_rotation_time_delay=bool(earth_rotation_time_delay),
            earth_rotation_beam_patterns=bool(earth_rotation_beam_patterns),
            finite_size=bool(finite_size),
            orbital_motion=bool(orbital_motion),
            center=resolve_center(center, interferometers))

    def convert(self, parameters):
        parameters = dict(parameters, fiducial=1)
        converted, _ = self.waveform_generator.parameter_conversion(parameters)
        return converted

    def polarizations(self, converted, freqs):
        wfg = self.waveform_generator
        source_parameters = {key: value for key, value in converted.items()
                             if key in wfg.source_parameter_keys}
        source_parameters["fiducial"] = 0
        source_parameters.update(wfg.waveform_arguments)
        source_parameters["frequency_bin_edges"] = np.asarray(freqs)
        return wfg.frequency_domain_source_model(wfg.frequency_array, **source_parameters)

    def project(self, parameters, freqs):
        """Detector strain, shape (n_ifo, n_mode, n_freq)."""
        converted = self.convert(parameters)
        h = np.zeros((len(self.interferometers), len(self.mode_keys), len(freqs)), dtype=complex)
        for start in range(0, len(freqs), CHUNK_SIZE):
            sl = slice(start, start + CHUNK_SIZE)
            pols = self.polarizations(converted, freqs[sl])
            for i, ifo in enumerate(self.interferometers):
                for k, key in enumerate(self.mode_keys):
                    h[i, k, sl] = ifo.get_detector_response_for_frequency_dependent_antenna_response(
                        waveform_polarizations={key: pols[key]}, parameters=converted,
                        start_time=ifo.strain_data.start_time, frequencies=freqs[sl],
                        **self.response_kwargs)
        return h

    def maximum_frequency(self, parameters, minimum_frequency, maximum_frequency, n_probe=200_000):
        """Lowest over modes of the last frequency where the waveform is
        non-zero (the likelihood's fiducial maximum frequency), on a
        geometric probe grid."""
        probe = np.geomspace(minimum_frequency, maximum_frequency, n_probe)
        pols = self.polarizations(self.convert(parameters), probe)
        last = [probe[np.flatnonzero(pols[key]["plus"])[-1]] for key in self.mode_keys
                if np.any(pols[key]["plus"])]
        return float(min(last)) if last else float(maximum_frequency)


class GridModel:
    """Fiducial prefix sums on a geometric grid: the continuum analogue of
    the relative-binning summary data, with d = h_fid."""

    def __init__(self, projector, fiducial_parameters, freqs):
        self.projector = projector
        self.freqs = freqs
        self.psd = np.array([
            ifo.power_spectral_density.get_power_spectral_density_array(freqs)
            for ifo in projector.interferometers])
        self.h0 = projector.project(fiducial_parameters, freqs)
        self.d = self.h0.sum(axis=1)
        weight = 4 / self.psd[:, None, :]
        a = np.conj(self.h0) * self.d[:, None, :] * weight
        self.A0 = trapezoid_prefix(a, freqs)
        self.A1 = trapezoid_prefix(a * freqs, freqs)
        # b[i, k, kp] = conj(h0_kp) h0_k / S, as in the likelihood's summary data
        b = self.h0[:, :, None, :] * np.conj(self.h0[:, None, :, :]) * weight[:, None]
        self.B0 = trapezoid_prefix(b, freqs)
        self.B1 = trapezoid_prefix(b * freqs, freqs)
        self.norm_d = float(np.real(4 * np.trapezoid(np.abs(self.d) ** 2 / self.psd, x=freqs).sum()))

    def inner(self, a, b):
        """Network <a|b> (complex), summed over detectors (leading axis);
        the remaining axes of a and b broadcast."""
        psd = self.psd.reshape(self.psd.shape[:1] + (1,) * (a.ndim - 2) + self.psd.shape[1:])
        return 4 * np.trapezoid(np.conj(a) * b / psd, x=self.freqs, axis=-1).sum(axis=0)

    def index_of(self, frequencies):
        idx = np.clip(np.searchsorted(self.freqs, frequencies), 1, len(self.freqs) - 1)
        left_closer = (frequencies - self.freqs[idx - 1]) < (self.freqs[idx] - frequencies)
        return np.where(left_closer, idx - 1, idx)


class GridPoint:
    """One test point: its waveform ratio and exact per-bin log-likelihood."""

    def __init__(self, model, h_test):
        self.model = model
        with np.errstate(divide="ignore", invalid="ignore"):
            self.ratio = np.where(model.h0 != 0, h_test / model.h0, 0)
        h = h_test.sum(axis=1)
        integrand = (4 / model.psd) * (np.real(np.conj(h) * model.d) - 0.5 * np.abs(h) ** 2)
        self.E = trapezoid_prefix(integrand, model.freqs).sum(axis=0)
        self.delta_log_likelihood = delta_log_likelihood(model, h_test)

    def bin_errors(self, lo, hi):
        """Signed exact - relative-binning log-likelihood error for the bins
        [freqs[lo], freqs[hi]] (index arrays), with the same linear-ratio
        approximation (and the same dropped r1 r1* term) as the likelihood."""
        m = self.model
        f_lo, f_hi = m.freqs[lo], m.freqs[hi]
        fc = 0.5 * (f_lo + f_hi)
        r_lo, r_hi = self.ratio[..., lo], self.ratio[..., hi]
        r0 = 0.5 * (r_lo + r_hi)
        r1 = (r_hi - r_lo) / (f_hi - f_lo)
        A0 = m.A0[..., hi] - m.A0[..., lo]
        A1 = m.A1[..., hi] - m.A1[..., lo] - fc * A0
        B0 = m.B0[..., hi] - m.B0[..., lo]
        B1 = m.B1[..., hi] - m.B1[..., lo] - fc * B0
        dh = np.sum(A0 * np.conj(r0) + A1 * np.conj(r1), axis=1)
        hh = (np.einsum("ikpb,ikb,ipb->ib", B0, r0, np.conj(r0))
              + np.einsum("ikpb,ikb,ipb->ib", B1, r0, np.conj(r1))
              + np.einsum("ikpb,ipb,ikb->ib", B1, np.conj(r0), r1))
        approx = np.sum(np.real(dh) - 0.5 * np.real(hh), axis=0)
        return self.E[hi] - self.E[lo] - approx

    def total_error(self, edges):
        """Relative-binning minus exact log-likelihood ratio for bin edges
        (grid indices); relative binning is exact at the fiducial point."""
        return -float(np.sum(self.bin_errors(edges[:-1], edges[1:])))


def select_bins(points, eta, initial_number_of_bins=200, max_iterations=50):
    """GETBINS with a max over test points: bisect (in grid index, i.e.
    log f) until every bin's |error| at every point is <= eta / n_bins, then
    iterate n_bins to self-consistency. Returns grid indices of the edges."""
    n = len(points[0].model.freqs)
    target, previous, iterations = initial_number_of_bins, None, 0
    while previous != target and iterations < max_iterations:
        budget = eta / target
        accepted = []
        pending = [(0, n - 1)]
        while pending:
            lo, hi = np.array(pending).T
            error = np.max([np.abs(p.bin_errors(lo, hi)) for p in points], axis=0)
            split = (error > budget) & (hi - lo > 1)
            accepted.extend(zip(lo[~split], hi[~split]))
            mid = (lo[split] + hi[split]) // 2
            pending = list(zip(lo[split], mid)) + list(zip(mid, hi[split]))
        edges = np.unique(np.array(accepted).ravel())
        previous, target = target, len(edges) - 1
        iterations += 1
    return edges


class PointFinder:
    """Test points along the segment from the fiducial parameters toward a
    prior draw, theta(beta) = (1 - beta) theta_fid + beta theta_draw, with
    the orbital phase optionally aligned to maximize the match."""

    def __init__(self, model, fiducial_parameters, n_phase=720):
        self.model = model
        self.projector = model.projector
        self.fiducial = fiducial_parameters
        self.keys = [key for key, value in fiducial_parameters.items()
                     if key != "fiducial" and np.isscalar(value)
                     and np.issubdtype(type(value), np.number)]
        self.deltas = np.linspace(0, 2 * np.pi, n_phase, endpoint=False)

    def combine(self, draw, beta):
        return {key: (1 - beta) * self.fiducial[key] + beta * draw.get(key, self.fiducial[key])
                for key in self.keys}

    def phase_maximized(self, h):
        """(mismatch, delta) maximizing the network match over a constant
        orbital phase shift delta, mode (l, m) rotated by exp(i m delta)."""
        c = self.model.inner(self.model.d[:, None, :], h)
        g = self.model.inner(h[:, :, None, :], h[:, None, :, :])

        def match(deltas):
            rot = np.exp(1j * np.outer(np.atleast_1d(deltas), self.projector.m))
            overlap = np.real(rot @ c)
            norm = np.real(np.einsum("dk,kl,dl->d", np.conj(rot), g, rot))
            return overlap / np.sqrt(self.model.norm_d * norm)

        best = self.deltas[np.argmax(match(self.deltas))]
        step = self.deltas[1]
        res = minimize_scalar(lambda x: -match(x)[0], bounds=(best - step, best + step),
                              method="bounded", options=dict(xatol=1e-12))
        return 1 + res.fun, float(res.x)

    def plain_mismatch(self, h):
        h = h.sum(axis=1)
        return 1 - float(np.real(self.model.inner(self.model.d, h))) / np.sqrt(
            self.model.norm_d * float(np.real(self.model.inner(h, h))))

    def align(self, params, h, delta):
        """params and strain with ``phase`` shifted so that the waveform is
        the phase-maximized one; the sign relating ``phase`` to the mode
        rotation is chosen by the better match."""
        candidates = []
        for sign in (1, -1):
            p = dict(params, phase=(params["phase"] + sign * delta) % (2 * np.pi))
            hp = self.projector.project(p, self.model.freqs)
            candidates.append((self.plain_mismatch(hp), p, hp))
        return min(candidates, key=lambda c: c[0])[1:]

    def at_mismatch(self, draw, target_mismatch, tolerance=0.01):
        """First beta at which the phase-maximized mismatch reaches the
        target, phase-aligned there."""
        cache = {}

        def mismatch(beta):
            h = self.projector.project(self.combine(draw, beta), self.model.freqs)
            m, delta = self.phase_maximized(h)
            cache[beta] = (delta, h)
            return m

        found = self._first_crossing(mismatch, target_mismatch, tolerance)
        if found is None:
            return None
        beta, m = found
        delta, h = cache[beta]
        params, h = self.align(self.combine(draw, beta), h, delta)
        return dict(beta=float(beta), mismatch=float(m), parameters=params), h

    def at_delta_log_likelihood(self, draw, target, tolerance=0.01):
        """First beta at which the raw (not phase-aligned) ln L_max - ln L
        reaches the target."""
        cache = {}

        def distance(beta):
            h = self.projector.project(self.combine(draw, beta), self.model.freqs)
            cache[beta] = h
            return -delta_log_likelihood(self.model, h)

        found = self._first_crossing(distance, target, tolerance)
        if found is None:
            return None
        beta, d = found
        return dict(beta=float(beta), delta=float(d), parameters=self.combine(draw, beta)), \
            cache[beta]

    @staticmethod
    def _first_crossing(function, target, tolerance):
        """First beta at which the (increasing near beta = 0) function
        reaches the target: geometric scan (x1.5), then power-law
        interpolation inside the bracket (further out it oscillates).
        Returns (beta, value) or None."""
        lo = m_lo = hi = None
        beta = 1e-6
        while beta <= 1.0:
            m = function(beta)
            if m >= target:
                hi, m_hi = beta, m
                break
            lo, m_lo = beta, m
            beta *= 1.5
        if hi is None:
            return None
        for _ in range(20):
            if abs(m_hi - target) < tolerance * target:
                break
            if lo is None or m_lo <= 0:
                mid = hi / 1.5
            else:
                slope = np.log(m_hi / m_lo) / np.log(hi / lo)
                mid = lo * (target / m_lo) ** (1 / slope)
                mid = min(max(mid, lo * (hi / lo) ** 0.05), lo * (hi / lo) ** 0.95)
            m = function(mid)
            if m < target:
                lo, m_lo = mid, m
            else:
                hi, m_hi = mid, m
        return hi, m_hi

    def along(self, draw, beta, aligned):
        params = self.combine(draw, beta)
        h = self.projector.project(params, self.model.freqs)
        if aligned:
            _, delta = self.phase_maximized(h)
            params, h = self.align(params, h, delta)
        return params, h


# multiprocessing (fork): workers inherit the finder
_FINDER = None


def _training_point(args):
    draw, target = args
    found = _FINDER.at_mismatch(draw, target)
    return None if found is None else (found[0], found[1])


def _evidence_point(args):
    draw, target = args
    found = _FINDER.at_delta_log_likelihood(draw, target)
    return None if found is None else (found[0], found[1])


def _scan_point(args):
    draw, beta, aligned = args
    params, h = _FINDER.along(draw, beta, aligned)
    return params, h


def _map(function, tasks, npool):
    if npool <= 1:
        return [function(task) for task in tasks]
    with multiprocessing.get_context("fork").Pool(npool) as pool:
        return pool.map(function, tasks, chunksize=1)


def select_relative_binning_bins(
        interferometers, waveform_generator, fiducial_parameters, priors,
        minimum_frequency, maximum_frequency, eta=3.0, n_train=128, n_validation=64,
        target_mismatch=0.1, effective_dimension=None, n_live=1000, evidence_tolerance=1e-5,
        n_evidence=64, max_eta_iterations=8, n_grid=30_001, n_scan_draws=4,
        scan_betas=np.geomspace(1e-8, 1, 17), comparison_bins=None,
        earth_rotation_time_delay=True, earth_rotation_beam_patterns=True, finite_size=True,
        npool=1, outdir=None, orbital_motion=True, center=None):
    """Select relative-binning bin edges for the mode-by-mode likelihood.

    Parameters
    ----------
    interferometers: list
        Interferometers with PSDs and strain-data start times (the strain
        itself is not used).
    waveform_generator: WaveformGenerator
        The relative-binning (mode-by-mode, ``frequency_bin_edges``) generator.
    fiducial_parameters: dict
        The fiducial point, in the sampling basis.
    priors: PriorDict
        Directions of the training segments are prior draws.
    minimum_frequency, maximum_frequency: float
        The band; the upper edge is lowered to the fiducial maximum frequency.
    eta: float
        Initial budget for the total |log-likelihood error| at every training
        point; it is rescaled (error ~ eta) until the evidence bias estimate
        is within [evidence_tolerance / 2, evidence_tolerance], keeping the
        largest eta that meets the tolerance. eta = 3 at mismatch 0.1 gave
        ~1000 (10 Hz) / ~2000 (3 Hz) bins for a BNS in ET, with errors of
        <~1e-3 within ln L_max - ln L < 100.
    n_train, n_validation: int
        Training points (bins are selected on all of them) and held-out points.
    target_mismatch: float
        Phase-maximized network mismatch of the training/validation points.
    effective_dimension: float, optional
        D_eff of the Gaussian L(X) model: the parameters the data constrain
        (not e.g. calibration). Default: the number of sampled parameters in
        ``priors`` (neither fixed nor constraints).
    n_live: int
        Live points, for the nested-sampling resolution diagnostic.
    evidence_tolerance: float or None
        Target for max(|ln E_post[e^epsilon]|, E_post|epsilon|); None keeps
        eta fixed (the evidence bias is still reported).
    n_evidence: int
        Points at stratified Gamma(D_eff / 2) quantiles of ln L_max - ln L
        (raw, not phase-aligned) estimating the posterior expectations.
    max_eta_iterations: int
        Bin selections tried while tuning eta.
    n_grid: int
        Geometric grid size.
    n_scan_draws: int
        Prior-draw segments for the error-vs-likelihood diagnostic.
    comparison_bins: dict, optional
        {label: bin-edge frequencies, or a function of (minimum_frequency,
        maximum_frequency) returning them, called with the final band}
        evaluated in the diagnostics too (e.g. the closed-form epsilon bins).
    npool: int
        Worker processes (fork) for the training-point search.
    outdir: str, optional
        Where the bin edges, a JSON summary and the summary plots are written.

    Returns
    -------
    bin_freqs: ndarray
        Bin-edge frequencies on the geometric grid.
    """
    global _FINDER
    t_start = time.time()
    projector = Projector(interferometers, waveform_generator, earth_rotation_time_delay,
                          earth_rotation_beam_patterns, finite_size, orbital_motion, center)
    f_max = min(maximum_frequency, projector.maximum_frequency(
        fiducial_parameters, minimum_frequency, maximum_frequency))
    freqs = np.geomspace(minimum_frequency, f_max, n_grid)
    model = GridModel(projector, fiducial_parameters, freqs)
    _FINDER = PointFinder(model, fiducial_parameters)
    logger.info(f"Bin selection: geometric grid of {n_grid} points over "
                f"[{minimum_frequency:.4g}, {f_max:.4g}] Hz; searching {n_train} + "
                f"{n_validation} points at phase-maximized mismatch {target_mismatch} "
                f"({npool} processes)")

    n_found = n_train + n_validation
    n_drawn = n_found + n_scan_draws
    if effective_dimension is None:
        effective_dimension = sampled_parameter_count(priors)
    draws = [priors.sample() for _ in range(n_drawn + n_evidence)]
    t0 = time.time()
    found = _map(_training_point, [(d, target_mismatch) for d in draws[:n_found]], npool)
    missing = sum(f is None for f in found)
    if missing:
        logger.warning(f"Bin selection: mismatch {target_mismatch} not reached along "
                       f"{missing} segments; those points are dropped")
    train = [f for f in found[:n_train] if f is not None]
    validation = [f for f in found[n_train:] if f is not None]
    train_points = [GridPoint(model, h) for _, h in train]
    validation_points = [GridPoint(model, h) for _, h in validation]
    logger.info(f"Bin selection: {len(train)} training / {len(validation)} validation "
                f"points in {time.time() - t0:.0f} s")

    t0 = time.time()
    targets = gamma.ppf((np.arange(n_evidence) + 0.5) / n_evidence, effective_dimension / 2)
    found = _map(_evidence_point, list(zip(draws[n_drawn:], targets)), npool)
    if any(f is None for f in found):
        logger.warning(f"Bin selection: ln L_max - ln L target not reached along "
                       f"{sum(f is None for f in found)} evidence segments; dropped")
    evidence_points = [GridPoint(model, h) for f in found if f is not None for h in [f[1]]]
    evidence_d = np.array([-p.delta_log_likelihood for p in evidence_points])
    logger.info(f"Bin selection: {len(evidence_points)} evidence points at Gamma("
                f"{effective_dimension / 2:g}) quantiles of ln L_max - ln L in "
                f"{time.time() - t0:.0f} s")

    def errors(points, set_edges):
        return np.array([p.total_error(set_edges) for p in points])

    t0 = time.time()
    eta_history, best = [], None
    for _ in range(max_eta_iterations if evidence_tolerance is not None else 1):
        trial = select_bins(train_points, eta)
        bias = evidence_bias(errors(evidence_points, trial))
        eta_history.append(dict(eta=eta, n_bins=len(trial) - 1, evidence_bias=bias))
        logger.info(f"Bin selection: eta = {eta:.4g}: {len(trial) - 1} bins, evidence "
                    f"bias {bias:.3g} (tolerance {evidence_tolerance})")
        if evidence_tolerance is None:
            best = (eta, trial, bias)
            break
        if bias <= evidence_tolerance and (best is None or eta > best[0]):
            best = (eta, trial, bias)
        if 0.5 * evidence_tolerance <= bias <= evidence_tolerance:
            break
        eta *= 0.75 * evidence_tolerance / max(bias, 1e-300)
    if best is None:
        logger.warning(f"Bin selection: evidence tolerance {evidence_tolerance:g} not met "
                       f"in {max_eta_iterations} iterations; using the last eta")
        best = (eta_history[-1]["eta"], trial, eta_history[-1]["evidence_bias"])
    eta, edges, bias = best
    bin_freqs = freqs[edges]
    logger.info(f"Bin selection: {len(edges) - 1} bins (eta = {eta:.4g}, {len(train)} "
                f"training points, evidence bias {bias:.3g}) in {time.time() - t0:.0f} s")

    # nested training subsets K = 1, 2, 4, ...
    subsets = []
    k = 1
    while k < len(train_points):
        subsets.append((k, select_bins(train_points[:k], eta)))
        k *= 2
    subsets.append((len(train_points), edges))

    sets = {f"greedy, eta={eta:.3g}, K={len(train_points)}": edges}
    for label, f in (comparison_bins or {}).items():
        f = np.asarray(f(freqs[0], freqs[-1]) if callable(f) else f)
        f = f[(f >= freqs[0]) & (f <= freqs[-1])]
        sets[label] = np.unique(np.concatenate([[0], model.index_of(f), [n_grid - 1]]))

    scan_tasks = [(d, float(b), aligned) for d in draws[n_found:n_drawn]
                  for b in scan_betas for aligned in (False, True)]
    t0 = time.time()
    scan = _map(_scan_point, scan_tasks, npool)
    scan_points = [GridPoint(model, h) for _, h in scan]
    logger.info(f"Bin selection: {len(scan_points)} scan points in {time.time() - t0:.0f} s")

    summary = dict(
        eta=eta, eta_history=eta_history, evidence_bias=bias,
        evidence_tolerance=evidence_tolerance, effective_dimension=effective_dimension,
        n_live=n_live, evidence_delta_log_likelihood=(-evidence_d).tolist(),
        target_mismatch=target_mismatch, n_grid=n_grid,
        minimum_frequency=float(freqs[0]), maximum_frequency=float(freqs[-1]),
        n_bins=len(edges) - 1, n_train=len(train), n_validation=len(validation),
        training_beta=[t["beta"] for t, _ in train],
        subsets=[dict(k=k, n_bins=len(e) - 1, validation=np.abs(errors(validation_points, e)).tolist())
                 for k, e in subsets],
        sets={label: dict(n_bins=len(e) - 1, bin_freqs=freqs[e].tolist(),
                          validation=errors(validation_points, e).tolist(),
                          evidence=errors(evidence_points, e).tolist(),
                          scan=errors(scan_points, e).tolist())
              for label, e in sets.items()},
        scan=[dict(beta=b, aligned=a, delta_log_likelihood=p.delta_log_likelihood)
              for (_, b, a), p in zip(scan_tasks, scan_points)],
        wall_time=time.time() - t_start)
    scan_d = np.array([-p.delta_log_likelihood for p in scan_points])
    for label, s in summary["sets"].items():
        v = np.abs(s["validation"])
        # ratio to the nested-sampling resolution; d < 0.01 is left out, where
        # both are at the quadrature noise floor
        d = np.concatenate([evidence_d, scan_d])
        e = np.abs(np.concatenate([s["evidence"], s["scan"]]))
        keep = d > 0.01
        s["evidence_bias"] = evidence_bias(s["evidence"])
        s["max_resolution_ratio"] = float(np.max(
            e[keep] / nested_sampling_resolution(d[keep], effective_dimension, n_live)))
        logger.info(f"Bin selection: {label}: {s['n_bins']} bins, validation |error| "
                    f"max {v.max():.3g}, median {np.median(v):.3g}; evidence bias "
                    f"{s['evidence_bias']:.3g}; max |error| / NS resolution "
                    f"{s['max_resolution_ratio']:.3g}")
    ratio = summary["sets"][next(iter(sets))]["max_resolution_ratio"]
    if ratio > 1:
        logger.warning(f"Bin selection: the error exceeds the nested-sampling resolution "
                       f"2 d / (D_eff n_live) by up to {ratio:.3g}x (D_eff = "
                       f"{effective_dimension:g}, n_live = {n_live})")

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        np.savetxt(os.path.join(outdir, "bin_freqs.txt"), bin_freqs)
        with open(os.path.join(outdir, "bin_selection.json"), "w") as f:
            json.dump(summary, f)
        plot_bin_selection(summary, outdir)
        logger.info(f"Bin selection: summary and plots written to {outdir}")
    _FINDER = None
    return bin_freqs


def plot_bin_selection(summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eta = summary["eta"]
    labels = list(summary["sets"])
    subsets = summary["subsets"]
    ks = [s["k"] for s in subsets]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    ax.plot(ks, [s["n_bins"] for s in subsets], "o-", color="C0", lw=2)
    for i, label in enumerate(labels[1:], start=1):
        ax.axhline(summary["sets"][label]["n_bins"], color=f"C{i}", ls=":", lw=1.5, label=label)
    ax.set(xscale="log", yscale="log", xlabel="training points K", ylabel="bins selected",
           title=rf"Bins vs. training points ($\eta$ = {eta:.3g})")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(ks, [max(s["validation"]) for s in subsets], "o-", color="C3", lw=2,
            label="max over validation points")
    ax.plot(ks, [np.median(s["validation"]) for s in subsets], "s-", color="C0", lw=2,
            label="median")
    ax.axhline(eta, color="gray", ls="--", lw=1, label=rf"$\eta$ = {eta:.3g}")
    for i, label in enumerate(labels[1:], start=1):
        ax.axhline(np.max(np.abs(summary["sets"][label]["validation"])), color=f"C{i}",
                   ls=":", lw=1.5, label=f"{label}: max")
    ax.set(xscale="log", yscale="log", xlabel="training points K",
           ylabel=r"validation $|\Delta \ln L|$",
           title=f"Error at {summary['n_validation']} held-out points "
                 f"(mismatch {summary['target_mismatch']:g})")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    x = -np.array([s["delta_log_likelihood"] for s in summary["scan"]])
    aligned = np.array([s["aligned"] for s in summary["scan"]])
    positive = x > 0
    edges = np.logspace(np.floor(np.log10(x[positive].min())), np.ceil(np.log10(x.max())), 17)
    centres = np.sqrt(edges[1:] * edges[:-1])
    idx = np.digitize(x, edges) - 1
    for i, label in enumerate(labels):
        e = np.abs(summary["sets"][label]["scan"])
        ax.scatter(x[aligned], e[aligned], s=8, color=f"C{i}", alpha=0.4, lw=0)
        ax.scatter(x[~aligned], e[~aligned], s=10, facecolor="none", edgecolor=f"C{i}",
                   alpha=0.4, lw=0.6)
        mx = [e[idx == j].max() if np.any(idx == j) else np.nan for j in range(len(centres))]
        ax.plot(centres, mx, "-", color=f"C{i}", lw=1.8,
                label=f"{label} ({summary['sets'][label]['n_bins']} bins"
                      + (rf", $\delta \ln Z$ ~ {summary['sets'][label]['evidence_bias']:.1e}"
                         if "evidence_bias" in summary["sets"][label] else "") + ")")
        if "evidence" in summary["sets"][label]:
            ax.scatter(-np.array(summary["evidence_delta_log_likelihood"]),
                       np.abs(summary["sets"][label]["evidence"]), s=14, marker="^",
                       color=f"C{i}", alpha=0.6, lw=0)
    ax.axhline(eta, color="gray", ls="--", lw=1)
    if "effective_dimension" in summary:
        dim, n_live = summary["effective_dimension"], summary["n_live"]
        ax.plot(centres, nested_sampling_resolution(centres, dim, n_live), "k--", lw=1.5,
                label=rf"NS resolution $2d / (D_{{\rm eff}} n_{{\rm live}})$, "
                      rf"$D_{{\rm eff}}$ = {dim:g}, $n_{{\rm live}}$ = {n_live}")
        ax.axvspan(*gamma.ppf([0.01, 0.99], dim / 2), color="gray", alpha=0.12,
                   label=r"posterior 98% of $d$, Gamma($D_{\rm eff}$/2)")
    ax.set(xscale="log", yscale="log", xlabel=r"$d = \ln L_{\max} - \ln L$",
           ylabel=r"$|\ln L_{\rm RB} - \ln L|$",
           title="Error vs. distance from the peak (lines: max per bin; filled: phase-\n"
                 "aligned, open: raw, triangles: posterior-distributed; zero noise)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    lo, hi = summary["minimum_frequency"], summary["maximum_frequency"]
    hist_edges = np.geomspace(lo, hi, 41)
    for i, label in enumerate(labels):
        f = np.array(summary["sets"][label]["bin_freqs"])
        ax.hist(f, bins=hist_edges, histtype="step", lw=1.8, color=f"C{i}",
                label=f"{label} ({len(f) - 1} bins)")
    ax.set(xscale="log", yscale="log", xlabel="frequency [Hz]",
           ylabel="bin edges per log-spaced interval",
           title=f"Bin-edge histogram ({len(hist_edges) - 1} log-spaced intervals)")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.3)
    title = (f"Relative-binning bin selection: {summary['n_bins']} bins over "
             f"[{lo:.3g}, {hi:.4g}] Hz, {summary['n_train']} training points")
    if "evidence_bias" in summary:
        title += (rf", $\eta$ = {eta:.3g}: evidence bias {summary['evidence_bias']:.2g} "
                  f"(tolerance {summary['evidence_tolerance']})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bin_selection.png"), dpi=150)
    plt.close(fig)
