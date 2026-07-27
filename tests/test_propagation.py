"""Tests for the unified propagation models."""
import numpy as np
import pytest

from bilby_xG.propagation import (
    Propagation, SpeedOfGravity, ModifiedDispersion, build_propagation,
)

FREQS = np.linspace(10.0, 1024.0, 64)


def test_gr_is_trivial():
    gr = Propagation()
    assert np.allclose(gr.phase_velocity(FREQS), 1.0)
    assert np.allclose(gr.group_velocity(FREQS), 1.0)
    assert np.allclose(gr.propagation_phase(FREQS, mode=2), 0.0)


def test_speed_of_gravity_unity_reduces_to_gr():
    sog = SpeedOfGravity(vG=1.0)
    assert np.allclose(sog.phase_velocity(FREQS), 1.0)
    assert np.allclose(sog.group_velocity(FREQS), 1.0)
    assert np.allclose(sog.propagation_phase(FREQS), 0.0)


def test_speed_of_gravity_constant_offset():
    sog = SpeedOfGravity(vG=0.99)
    assert np.allclose(sog.phase_velocity(FREQS), 0.99)
    assert np.allclose(sog.group_velocity(FREQS), 0.99)


def test_modified_dispersion_small_amplitude_reduces_to_gr():
    md = ModifiedDispersion(a=3.0, A=1e-12, luminosity_distance=400.0,
                            mass_1=30.0, mass_2=30.0)
    assert np.allclose(md.phase_velocity(FREQS), 1.0, atol=1e-6)
    assert np.allclose(md.group_velocity(FREQS), 1.0, atol=1e-6)


def test_modified_dispersion_velocities_are_frequency_dependent():
    # The velocity deviation scales as A * E**(a-2) and is utterly negligible
    # for physical A (the *phase* is the observable; see the phase test). Use a
    # large A purely to exercise the velocity formula at representable values.
    md = ModifiedDispersion(a=3.0, A=1e18, luminosity_distance=400.0,
                            mass_1=30.0, mass_2=30.0)
    vp = md.phase_velocity(FREQS)
    vg = md.group_velocity(FREQS)
    # Phase velocity below c, group velocity above c for A > 0, a = 3.
    assert np.all(vp < 1.0)
    assert np.all(vg > 1.0)
    # Genuinely frequency dependent and distinct from each other.
    assert np.ptp(vp) > 0
    assert not np.allclose(vp, vg)


def test_modified_dispersion_phase_is_frequency_dependent():
    md = ModifiedDispersion(a=3.0, A=1.0, luminosity_distance=400.0,
                            mass_1=30.0, mass_2=30.0)
    phase = md.propagation_phase(FREQS, mode=2)
    assert phase.shape == FREQS.shape
    # The corrected (non-debug) implementation varies with frequency.
    assert np.ptp(phase) > 0


def test_modified_dispersion_phase_zero_without_distance():
    md = ModifiedDispersion(a=3.0, A=1.0)
    assert np.allclose(md.propagation_phase(FREQS), 0.0)


def test_modified_dispersion_a2_rejected():
    with pytest.raises(ValueError):
        ModifiedDispersion(a=2.0, A=1.0)


def test_build_propagation_selection():
    assert isinstance(build_propagation({}), Propagation)
    assert isinstance(build_propagation({"vG": 0.99}), SpeedOfGravity)
    md = build_propagation({"a": 3.0, "A": 1.0, "luminosity_distance": 400.0,
                            "mass_1": 30.0, "mass_2": 30.0})
    assert isinstance(md, ModifiedDispersion)
    # Dispersion takes precedence over vG when both present.
    both = build_propagation({"a": 3.0, "A": 1.0, "vG": 0.9})
    assert isinstance(both, ModifiedDispersion)
