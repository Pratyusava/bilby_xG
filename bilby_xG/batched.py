# Licensed under an MIT style license -- see LICENSE

"""Batched (vectorised) evaluation of the mode-by-mode relative-binning
likelihood, for samplers that propose many points at once (e.g. nessai).

Prototype. :class:`BatchedRelativeBinningLikelihood` wraps an existing
:class:`~bilby_xG.likelihood.RelativeBinningGravitationalWaveTransientNextGenerationModebyMode`
built with :func:`~bilby_xG.source.mlgw_bns_individual_modes` or
:func:`~bilby_xG.source.mlgw_bns_precessing_individual_modes` (either
``precessing_scheme``), and reuses its bins, fiducial waveforms and summary
data. Its
:meth:`~BatchedRelativeBinningLikelihood.log_likelihood_ratio` takes a
dict of parameter *arrays* and evaluates all of them in a single jitted
JAX function: the mlgw_bns surrogate, evaluated for the whole batch at once
through its public batched interface
(:meth:`mlgw_bns.model.Model.jax_modes_amp_phase` and
:func:`mlgw_bns.batched.mode_polarizations`), then, ``vmap``-ed over the
batch, the frequency-dependent detector response and the summary-data
contraction.

Scope of the prototype: general relativity only (no ``vG`` / ``(a, A)``),
no distance/phase/time marginalisation, ``reference_frame="sky"`` with
geocentre time. The detector motion (:mod:`bilby_xG.motion`) is evaluated on
the ephemeris tabulated over the data segment, in the axes of date at its
start. The JAX surrogate agrees with the numpy one only to the
rounding of its kernel-ridge regressors (see :mod:`mlgw_bns.batched`); see
``benchmarks/et_mlgw_bns_batched.py`` for the resulting log-likelihood
differences and the speed-up.

:class:`BatchedBilbyModel` and :class:`BatchedNessai` connect it to nessai:
pass ``sampler="batched_nessai"`` to :func:`bilby.run_sampler` (a
``bilby.samplers`` entry point).
"""
import math

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.utils import logger, speed_of_light
from bilby_cython.geometry import greenwich_mean_sidereal_time

from .motion import default_ephemeris, hermite_interpolate, precession_matrix, resolve_center
from .utils import calculate_time_to_merger_for_any_mode

__author__ = ["Jacopo Tissino"]

_DAY = 24. * 60. * 60.


def mlgw_bns_jax_modes(model, mode_array):
    """A JAX function evaluating selected mlgw_bns modes for a batch.

    Returns ``modes(theta, total_mass, distance, inclination, phase,
    frequencies) -> (plus, cross, time_to_merger)``, each of shape
    ``(n, n_modes, n_freqs)``,
    with ``theta`` of shape ``(n, 5)``, rows ``[q >= 1, lambda_1, lambda_2,
    chi_1, chi_2]``, and the other parameters of shape ``(n,)``: the same
    per-mode polarisations as :func:`bilby_xG.source.mlgw_bns_individual_modes`
    (up to the rounding of the surrogate's regressors). Rows outside the
    surrogate's range are NaN.
    """
    import jax.numpy as jnp
    from mlgw_bns.batched import mode_polarizations

    predict = model.jax_modes_amp_phase(mode_array, return_tf=True)

    def modes(theta, total_mass, distance, inclination, phase, frequencies):
        amp, phi, times = predict(theta, total_mass, frequencies, distance)
        # pi/2 - phase: the azimuth convention of the TEOBResumS SPA models
        plus, cross = mode_polarizations(amp, phi, mode_array, inclination,
                                         jnp.pi / 2 - phase, xp=jnp)
        return plus, cross, -times

    return modes


def mlgw_bns_precessing_jax_modes(mode_array, frequencies, waveform_arguments, twist=False):
    """A JAX function evaluating selected precessing mlgw_bns modes for a
    batch, at fixed ``frequencies``.

    Returns ``modes(p) -> (plus, cross)``, or with ``twist`` ``modes(p) ->
    (coprecessing, c_plus, c_cross)`` (``plus = c_plus * coprecessing``),
    each of shape ``(n, n_modes, n_freqs)``, with ``p`` a dict of arrays of shape ``(n,)``: ``q >= 1``,
    ``lambda_1``, ``lambda_2``, ``total_mass``, ``distance`` and the LVK
    spins ``a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phase``. The
    same per-mode polarisations as
    :func:`bilby_xG.source.mlgw_bns_precessing_individual_modes` with the
    ``waveform_arguments`` (``reference_frequency``, ``minimum_frequency``,
    ``precession_n_steps``), including the start of the precession
    integration (up to the rounding of the surrogate's regressors). Rows
    outside the surrogate's range are NaN.
    """
    import jax.numpy as jnp
    from mlgw_bns.precessing_model import PrecessingModel
    from mlgw_bns.spin_conversion import lvk_to_precessing

    from .source import (MLGW_BNS_PRECESSION_STEPS, _mlgw_bns_model,
                         mlgw_bns_precessing_start_frequency)

    frequencies = np.asarray(frequencies, dtype=float)
    reference_frequency = float(waveform_arguments.get("reference_frequency", 50.0))
    n_steps = int(waveform_arguments.get("precession_n_steps", MLGW_BNS_PRECESSION_STEPS))
    start = mlgw_bns_precessing_start_frequency(
        frequencies, waveform_arguments.get("minimum_frequency"))
    in_band = frequencies >= start
    # the angles are integrated from the first frequency
    grid = np.concatenate([[start], frequencies[in_band]])
    predict = PrecessingModel(_mlgw_bns_model()).jax_predict_modes(mode_array, n_steps)

    def modes(p):
        q = p["q"]
        orientation = lvk_to_precessing(
            p["theta_jn"], p["phi_jl"], p["tilt_1"], p["tilt_2"], p["phi_12"], p["a_1"],
            p["a_2"], p["total_mass"] * q / (1 + q), p["total_mass"] / (1 + q),
            reference_frequency, p["phase"], xp=jnp)
        intrinsic = jnp.concatenate([
            jnp.stack([q, p["lambda_1"], p["lambda_2"]], axis=1),
            orientation.chi_1, orientation.chi_2], axis=1)
        coprecessing, c_plus, c_cross = predict(
            intrinsic, jnp.asarray(grid), p["total_mass"], p["distance"],
            orientation.inclination, orientation.azimuth, orientation.reference_phase,
            reference_frequency)
        shape = coprecessing.shape[:2] + (len(frequencies),)
        index = np.flatnonzero(in_band)

        def full(values):
            return jnp.zeros(shape, complex).at[..., index].set(values[..., 1:])

        # hx with LALSuite's sign, as the numpy source
        if twist:
            return full(coprecessing), full(c_plus), full(-c_cross)
        return full(c_plus * coprecessing), full(-c_cross * coprecessing)

    return modes


def _antenna_response(geometry, frame, frequencies, ifo_time, flags, geocentre_delay,
                      direction_at_coalescence, split=False):
    """``Interferometer.frequency_dependent_antenna_response`` (general
    relativity) in JAX, for one detector, given the wave frame, the
    detector-independent delay and the source direction in Earth-fixed axes
    at coalescence; with ``split``, ``(fps, fcs, delay_phase)``."""
    import jax.numpy as jnp

    earth_rotation_time_delay, earth_rotation_beam_patterns, finite_size = flags
    omegas, pol_plus, pol_cross = frame

    def constant(x):
        return x if earth_rotation_beam_patterns else jnp.full_like(x, x[-1])

    if finite_size:
        fpxx = constant(jnp.einsum('ij,ijk->k', geometry["xx"], pol_plus))
        fpyy = constant(jnp.einsum('ij,ijk->k', geometry["yy"], pol_plus))
        fcxx = constant(jnp.einsum('ij,ijk->k', geometry["xx"], pol_cross))
        fcyy = constant(jnp.einsum('ij,ijk->k', geometry["yy"], pol_cross))
        fl_over_c = frequencies * geometry["length"] * 1e3 / speed_of_light

        def arm(y):
            return 0.5 * (jnp.exp(-1j * math.pi * fl_over_c * (1 + y)) * jnp.sinc(fl_over_c * (1 - y))
                          + jnp.exp(1j * math.pi * fl_over_c * (1 - y)) * jnp.sinc(fl_over_c * (1 + y)))

        dxx = arm(-omegas.T @ geometry["x"])
        dyy = arm(-omegas.T @ geometry["y"])
        fps = fpxx * dxx - fpyy * dyy
        fcs = fcxx * dxx - fcyy * dyy
    else:
        fps = constant(jnp.einsum('ij,ijk->k', geometry["detector_tensor"], pol_plus))
        fcs = constant(jnp.einsum('ij,ijk->k', geometry["detector_tensor"], pol_cross))

    if earth_rotation_time_delay:
        vertex_distance = omegas.T @ geometry["vertex"]
    else:
        vertex_distance = direction_at_coalescence @ geometry["vertex"]
    ifo_times = ifo_time + geocentre_delay - vertex_distance / speed_of_light
    exp_fac = jnp.exp(-2j * math.pi * frequencies * ifo_times)
    if split:
        return fps, fcs, exp_fac
    return fps * exp_fac, fcs * exp_fac


def _geocentre_delay(ra, dec, geocent_time, gmst, times_to_coalescence, orbit, center,
                     orbital_motion):
    """:func:`bilby_xG.motion.geocentre_delay` in JAX, on the node table
    ``orbit`` (see :meth:`bilby_xG.motion.EarthEphemeris.table`, with its
    ``precession`` matrix). ``center`` is a dict with either ``offset``
    (Earth-fixed, at coalescence) or ``position`` (fixed, axes of date)."""
    import jax.numpy as jnp

    direction = jnp.stack([jnp.cos(dec) * jnp.cos(ra), jnp.cos(dec) * jnp.sin(ra), jnp.sin(dec)])
    icrs_direction = orbit["precession"].T @ direction
    times = geocent_time - times_to_coalescence if orbital_motion else geocent_time[None]
    times = jnp.append(times, geocent_time)
    distances = hermite_interpolate(times, orbit["t0"], orbit["step"], orbit["positions"],
                                    orbit["velocities"], xp=jnp) @ icrs_direction
    if "offset" in center:
        earth_fixed = jnp.stack([jnp.cos(dec) * jnp.cos(ra - gmst),
                                 jnp.cos(dec) * jnp.sin(ra - gmst), jnp.sin(dec)])
        center_distance = distances[-1] + earth_fixed @ center["offset"]
    else:
        center_distance = direction @ center["position"]
    delay = -(distances[:-1] - center_distance) / speed_of_light
    return delay if orbital_motion else jnp.full_like(times_to_coalescence, delay[0])


def _wave_frame(ra, dec, psi, gmsts):
    """:func:`bilby_xG.interferometer.compute_wave_frame` in JAX."""
    import jax.numpy as jnp

    theta, phi = jnp.pi / 2 - dec, ra - gmsts
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
    zero = jnp.zeros_like(phi)
    u = jnp.stack([cos_phi * cos_theta, cos_theta * sin_phi, -sin_theta * jnp.ones_like(phi)])
    v = jnp.stack([-sin_phi, cos_phi, zero])
    m = -u * jnp.sin(psi) - v * jnp.cos(psi)
    n = -u * jnp.cos(psi) + v * jnp.sin(psi)
    omegas = jnp.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta * jnp.ones_like(phi)])
    mn = jnp.einsum('ik,jk->ijk', m, n)
    pol_plus = jnp.einsum('ik,jk->ijk', m, m) - jnp.einsum('ik,jk->ijk', n, n)
    pol_cross = mn + jnp.transpose(mn, (1, 0, 2))
    return omegas, pol_plus, pol_cross


class BatchedRelativeBinningLikelihood(Likelihood):
    """Vectorised version of a mode-by-mode relative-binning likelihood.

    Parameters
    ----------
    likelihood : RelativeBinningGravitationalWaveTransientNextGenerationModebyMode
        A set-up likelihood whose source model is
        :func:`~bilby_xG.source.mlgw_bns_individual_modes`; its summary data,
        bins and fiducial waveforms are reused as they are.
    batch_size : int
        Points are evaluated in chunks of this many (the last one padded):
        each distinct chunk size is compiled once (tens of seconds), so
        only two shapes are ever used, this and 1 (for single points).
    """

    def __init__(self, likelihood, batch_size=256):
        import jax

        from .source import (_mlgw_bns_model, mlgw_bns_individual_modes,
                             mlgw_bns_precessing_individual_modes)

        super().__init__()
        self.likelihood = likelihood
        self.batch_size = int(batch_size)
        wfg = likelihood.waveform_generator
        source = wfg.frequency_domain_source_model
        if source not in (mlgw_bns_individual_modes, mlgw_bns_precessing_individual_modes):
            raise NotImplementedError("the batched likelihood supports mlgw_bns_individual_modes "
                                      "and mlgw_bns_precessing_individual_modes only")
        self.precessing = source is mlgw_bns_precessing_individual_modes
        if (likelihood.distance_marginalization or likelihood.phase_marginalization
                or likelihood.time_marginalization):
            raise NotImplementedError("marginalisation is not supported by the batched likelihood")
        if likelihood.reference_frame != "sky" or "geocent" not in likelihood.time_reference:
            raise NotImplementedError("only reference_frame='sky' with geocentre time is supported")

        self.mode_array = [tuple(int(x) for x in mode) for mode in likelihood.mode_array]
        keys = [f"{ell},{emm}" for ell, emm in self.mode_array]
        ifos = likelihood.interferometers
        self.frequencies = np.asarray(likelihood.bin_freqs, dtype=float)
        self.start_time = ifos[0].strain_data.start_time
        if any(ifo.strain_data.start_time != self.start_time for ifo in ifos):
            raise NotImplementedError("the interferometers must share a start time")

        summary = likelihood.summary_data
        stack = np.stack
        self._arrays = dict(
            a0=stack([stack([summary[ifo.name]["a0"][k] for k in keys]) for ifo in ifos]),
            a1=stack([stack([summary[ifo.name]["a1"][k] for k in keys]) for ifo in ifos]),
            b0=stack([stack([stack([summary[ifo.name]["b0"][k][kp] for kp in keys]) for k in keys])
                      for ifo in ifos]),
            b1=stack([stack([stack([summary[ifo.name]["b1"][k][kp] for kp in keys]) for k in keys])
                      for ifo in ifos]),
            reference=stack([stack([likelihood.per_detector_per_mode_fiducial_waveform_points[
                ifo.name][k] for k in keys]) for ifo in ifos]),
            bin_widths=np.asarray(likelihood.bin_widths, dtype=float),
            frequencies=self.frequencies,
        )
        for ifo in ifos:
            if (ifo.time_to_merger_model is not None or ifo.static_response_time
                    or ifo.fsr_phase_reference is not None):
                raise NotImplementedError(
                    f"{ifo.name}: the batched likelihood has only the default response "
                    "(time_to_merger_model, static_response_time, fsr_phase_reference)")
        self._geometries = [dict(
            xx=np.asarray(ifo.geometry.xx), yy=np.asarray(ifo.geometry.yy),
            detector_tensor=np.asarray(ifo.geometry.detector_tensor),
            x=np.asarray(ifo.geometry.x), y=np.asarray(ifo.geometry.y),
            vertex=np.asarray(ifo.geometry.vertex), length=float(ifo.geometry.length),
        ) for ifo in ifos]
        self._flags = (bool(likelihood.earth_rotation_time_delay),
                       bool(likelihood.earth_rotation_beam_patterns),
                       bool(likelihood.finite_size))
        self._orbital_motion = bool(likelihood.orbital_motion)
        # the ephemeris over the data segment, and a day either side
        table = default_ephemeris.table(self.start_time - _DAY,
                                        self.start_time + ifos.duration + _DAY)
        self._arrays["orbit"] = dict(
            t0=np.float64(table["t0"]), positions=table["positions"],
            velocities=table["velocities"], precession=precession_matrix(self.start_time))
        self._orbit_step = table["step"]
        center = resolve_center(likelihood.center, ifos)
        if center.fixed_position is None and center.reference_time is None:
            self._arrays["center"] = dict(offset=center.offset)
        else:
            self._arrays["center"] = dict(position=center.position(self.start_time))
        self.split = getattr(likelihood, "precessing_scheme", "ratio") == "split"
        if self.precessing:
            self._modes = mlgw_bns_precessing_jax_modes(
                self.mode_array, self.frequencies, wfg.waveform_arguments, twist=self.split)
        else:
            self._modes = mlgw_bns_jax_modes(_mlgw_bns_model(), self.mode_array)
        self._batched = jax.jit(self._batch)

    def _batch(self, p, arrays):
        """ln L ratio for a batch of parameter sets (a dict of arrays)."""
        import jax
        import jax.numpy as jnp

        if self.precessing:
            # no waveform t(f) from the precessing surrogate: the PN one, per m
            modes, time_to_merger = self._modes(p), None
        else:
            theta = jnp.stack([p["q"], p["lambda_1"], p["lambda_2"], p["chi_1"], p["chi_2"]],
                              axis=1)
            plus, cross, time_to_merger = self._modes(
                theta, p["total_mass"], p["distance"], p["inclination"], p["phase"],
                arrays["frequencies"])
            modes = (plus, cross)
        ln_l = jax.vmap(self._project, in_axes=(0, 0, 0, None))(
            p, modes, time_to_merger, arrays)
        # outside the surrogate's range: outside the support
        return jnp.where(jnp.isnan(ln_l), -jnp.inf, ln_l)

    def _project(self, p, modes, time_to_merger, arrays):
        """ln L ratio for one parameter set, given its modes ``(plus, cross)``
        (with ``precessing_scheme="split"``, ``(coprecessing, c_plus,
        c_cross)``), each ``(n_modes, n_freqs)``, and their times to merger
        (``None``: the post-Newtonian ones)."""
        import jax.numpy as jnp

        d_inner_h, h_inner_h = self._bin_inner_products(
            p, modes, time_to_merger, arrays)
        return jnp.sum(d_inner_h) - jnp.sum(h_inner_h) / 2

    def _bin_inner_products(self, p, modes, time_to_merger, arrays):
        """Re (d|h) and (h|h) of one parameter set in every bin, each
        ``(n_bins,)``, summed over detectors and modes."""
        import jax.numpy as jnp

        from .likelihood import split_scheme_coefficients

        f = arrays["frequencies"]
        ifo_time = p["geocent_time"] - self.start_time
        earth_rotation = self._flags[0] or self._flags[1]

        frames = {}
        ratios = []
        for geometry in self._geometries:
            strains = []
            for k, (_, emm) in enumerate(self.mode_array):
                # the waveform's own t(f) is per mode, the PN one per m
                key = emm if time_to_merger is None else k
                if key not in frames:
                    ttc = (calculate_time_to_merger_for_any_mode(
                        f, p["mass_1"], p["mass_2"], p["chi_1"], p["chi_2"], mode=emm, safety=1)
                        if time_to_merger is None else time_to_merger[k])
                    gmsts = (p["gmst"] - p["gmst_rate"] * ttc if earth_rotation
                             else jnp.full_like(f, p["gmst"]))
                    frames[key] = (_wave_frame(p["ra"], p["dec"], p["psi"], gmsts),
                                   _geocentre_delay(
                                       p["ra"], p["dec"], p["geocent_time"], p["gmst"], ttc,
                                       dict(arrays["orbit"], step=self._orbit_step),
                                       arrays["center"], self._orbital_motion))
                direction = jnp.stack([
                    jnp.cos(p["dec"]) * jnp.cos(p["ra"] - p["gmst"]),
                    jnp.cos(p["dec"]) * jnp.sin(p["ra"] - p["gmst"]), jnp.sin(p["dec"])])
                response = _antenna_response(geometry, frames[key][0], f, ifo_time, self._flags,
                                             frames[key][1], direction, split=self.split)
                if self.split:
                    coprecessing, c_plus, c_cross = modes
                    fps, fcs, delay_phase = response
                    strains.append(jnp.stack([delay_phase * coprecessing[k],
                                              c_plus[k] * fps + c_cross[k] * fcs]))
                else:
                    plus, cross = modes
                    fps, fcs = response
                    strains.append(plus[k] * fps + cross[k] * fcs)
            ratios.append(jnp.stack(strains))
        if self.split:
            factors = jnp.stack(ratios)  # (ifo, mode, fast/slow, edge)
            r0, r1 = split_scheme_coefficients(
                factors[:, :, 0] / arrays["reference"], factors[:, :, 1], arrays["bin_widths"])
        else:
            ratio = jnp.stack(ratios) / arrays["reference"]  # (ifo, mode, edge)
            r0 = 0.5 * (ratio[..., 1:] + ratio[..., :-1])
            r1 = (ratio[..., 1:] - ratio[..., :-1]) / arrays["bin_widths"]
        d_inner_h = jnp.sum(arrays["a0"] * jnp.conj(r0) + arrays["a1"] * jnp.conj(r1),
                            axis=(0, 1))
        h_inner_h = jnp.sum(
            arrays["b0"] * r0[:, :, None] * jnp.conj(r0[:, None, :])
            + arrays["b1"] * (r0[:, :, None] * jnp.conj(r1[:, None, :])
                              + jnp.conj(r0[:, None, :]) * r1[:, :, None]), axis=(0, 1, 2))
        return jnp.real(d_inner_h), jnp.real(h_inner_h)

    def _inputs(self, parameters):
        """Converted parameter arrays for the JAX function."""
        likelihood = self.likelihood
        for key in ("vG", "a", "A"):
            if key in parameters:
                raise NotImplementedError("the batched likelihood supports general relativity only")
        parameters = {key: np.asarray(value) for key, value in parameters.items()
                      if np.asarray(value).dtype.kind in "fiub"}
        size = max((value.size for value in parameters.values()), default=1)
        parameters = {key: np.broadcast_to(value, (size,)).astype(float)
                      for key, value in parameters.items()}
        parameters.update(likelihood.get_sky_frame_parameters(parameters))
        converted, _ = likelihood.waveform_generator.parameter_conversion(parameters)
        times = np.asarray(converted["geocent_time"], dtype=float)
        gmst = np.array([greenwich_mean_sidereal_time(t) for t in times])
        gmst_rate = (np.array([greenwich_mean_sidereal_time(t + _DAY) for t in times]) - gmst) / _DAY
        q = np.asarray(converted["q"], dtype=float)
        inputs = dict(
            q=np.where(q <= 1, 1 / q, q), lambda_1=converted["LambdaAl2"],
            lambda_2=converted["LambdaBl2"], total_mass=converted["M"],
            distance=converted["distance"], mass_1=converted["mass_1"],
            mass_2=converted["mass_2"], ra=converted["ra"], dec=converted["dec"],
            psi=converted["psi"], geocent_time=times, gmst=gmst, gmst_rate=gmst_rate)
        if self.precessing:
            inputs.update({key: converted[key] for key in (
                "chi_1", "chi_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
                "theta_jn", "phase")})
        else:
            inputs.update(chi_1=converted["chi1z"], chi_2=converted["chi2z"],
                          inclination=converted["inclination"],
                          phase=converted["coalescence_angle"])
        return {key: np.broadcast_to(np.asarray(value, dtype=float), (size,))
                for key, value in inputs.items()}, size

    def log_likelihood_ratio(self, parameters=None):
        """ln L ratio at ``parameters``, a dict of scalars (returns a float)
        or of equal-length arrays (returns an array)."""
        if parameters is None:
            parameters = self.parameters
        scalar = all(np.ndim(value) == 0 for value in parameters.values())
        inputs, size = self._inputs(parameters)
        out = np.empty(size)
        chunk = 1 if size == 1 else self.batch_size
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            batch = {key: np.pad(value[start:stop], (0, chunk - (stop - start)), mode="edge")
                     for key, value in inputs.items()}
            out[start:stop] = np.asarray(self._batched(batch, self._arrays))[:stop - start]
        return float(out[0]) if scalar else out

    def noise_log_likelihood(self):
        return self.likelihood.noise_log_likelihood()

    def log_likelihood(self, parameters=None):
        return self.log_likelihood_ratio(parameters) + self.noise_log_likelihood()


class BatchedTruncatedMixtureLikelihood(BatchedRelativeBinningLikelihood):
    r"""Mixture of the likelihoods of the data truncated at several
    frequencies, evaluated as :class:`BatchedRelativeBinningLikelihood`.

    With :math:`\ln L_i` the log-likelihood ratio of the data in the bins
    below the bin edge ``edges[i]`` (an index into ``bin_freqs``), and
    :math:`\ln Z_i` = ``log_evidences[i]``, this is

    .. math:: \ln L = \ln \sum_i L_i / Z_i,

    whose posterior is the mixture of the truncated-data posteriors, with
    weights proportional to the ratios of their true evidences to the
    :math:`Z_i` given (equal if those are exact). A nested-sampling run on it
    is then an importance sample for every truncated-data posterior: with the
    run's prior-volume elements :math:`\Delta X_j`, the weights
    :math:`L_i(\theta_j) \Delta X_j` have
    :math:`L_i / Z_i \leq L`, so each posterior gets about a fraction
    :math:`Z_i^\mathrm{true} / (Z_i \sum_k Z_k^\mathrm{true} / Z_k)` of the
    run's effective samples (early-warning localisation from one run).

    Parameters
    ----------
    likelihood, batch_size
        As :class:`BatchedRelativeBinningLikelihood`.
    edges : sequence of int
        Increasing bin-edge indices, from 1 to ``number_of_bins`` (the whole
        band).
    log_evidences : sequence of float
        :math:`\ln Z_i`, one per edge: estimates of the truncated-data log
        Bayes factors against noise.
    """

    def __init__(self, likelihood, edges, log_evidences, batch_size=256):
        super().__init__(likelihood, batch_size=batch_size)
        edges = np.asarray(edges, dtype=int)
        log_evidences = np.asarray(log_evidences, dtype=float)
        if edges.ndim != 1 or edges.shape != log_evidences.shape or edges.size == 0:
            raise ValueError("edges and log_evidences must be 1-d, of the same length")
        if np.any(np.diff(edges) <= 0) or edges[0] < 1 or edges[-1] > likelihood.number_of_bins:
            raise ValueError(f"edges must increase, from 1 to {likelihood.number_of_bins}")
        if not np.all(np.isfinite(log_evidences)):
            raise ValueError("log_evidences must be finite")
        self.edges = edges
        self.log_evidences = log_evidences

    def _project(self, p, modes, time_to_merger, arrays):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp

        d_inner_h, h_inner_h = self._bin_inner_products(
            p, modes, time_to_merger, arrays)
        cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_inner_h - h_inner_h / 2)])
        return logsumexp(cumulative[self.edges] - self.log_evidences)


def _batched_bilby_model_classes():
    from nessai_bilby.model import BilbyModel

    class BatchedBilbyModel(BilbyModel):
        """nessai model evaluating a :class:`BatchedRelativeBinningLikelihood`
        on whole batches of live points."""

        allow_vectorised = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # batch and single evaluations agree to rounding, not to the
            # 1e-15 nessai's automatic check demands
            self.vectorised_likelihood = True
            self.likelihood_chunksize = getattr(self.bilby_likelihood, "batch_size", None)

        def _theta(self, x):
            theta = dict(self.fixed_parameters)
            for name in self.names:
                theta[name] = np.asarray(x[name], dtype=float)
            return theta

        def log_likelihood(self, x):
            theta = self._theta(x)
            out = np.atleast_1d(self.bilby_log_likelihood_fn(theta))
            return out.reshape(np.shape(x)) if np.ndim(x) else out[0]

    class BatchedBilbyModelLikelihoodConstraint(BatchedBilbyModel):
        """As :class:`BatchedBilbyModel`, with the prior constraints in the
        likelihood."""

        def log_likelihood(self, x):
            theta = self._theta(x)
            allowed = np.atleast_1d(self.bilby_priors.evaluate_constraints(theta))
            out = np.atleast_1d(self.bilby_log_likelihood_fn(theta))
            out = np.where(allowed, out, -np.inf)
            return out.reshape(np.shape(x)) if np.ndim(x) else out[0]

        def log_prior(self, x):
            theta = {n: x[n] for n in self.names}
            return self.bilby_priors.ln_prob(theta, axis=0)

    return BatchedBilbyModel, BatchedBilbyModelLikelihoodConstraint


def __getattr__(name):
    # the nessai classes are built lazily, so nessai stays optional
    if name in ("BatchedBilbyModel", "BatchedBilbyModelLikelihoodConstraint"):
        return dict(zip(("BatchedBilbyModel", "BatchedBilbyModelLikelihoodConstraint"),
                        _batched_bilby_model_classes()))[name]
    if name == "BatchedNessai":
        return _batched_nessai_class()
    raise AttributeError(name)


_BATCHED_NESSAI = None


def _batched_nessai_class():
    global _BATCHED_NESSAI
    if _BATCHED_NESSAI is None:
        from nessai_bilby import plugin

        model, constrained = _batched_bilby_model_classes()

        class BatchedNessai(plugin.Nessai):
            """bilby's nessai sampler, with the likelihood evaluated on whole
            batches of points: ``bilby.run_sampler(likelihood=
            BatchedRelativeBinningLikelihood(...), sampler="batched_nessai",
            ...)``.
            """

            # bilby checks the install by importing this name, which
            # otherwise defaults to the lower-cased class name
            external_sampler_name = "nessai"

            def run_sampler(self):
                originals = plugin.BilbyModel, plugin.BilbyModelLikelihoodConstraint
                plugin.BilbyModel, plugin.BilbyModelLikelihoodConstraint = model, constrained
                try:
                    return super().run_sampler()
                finally:
                    plugin.BilbyModel, plugin.BilbyModelLikelihoodConstraint = originals

        _BATCHED_NESSAI = BatchedNessai
    return _BATCHED_NESSAI
