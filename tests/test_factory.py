import numpy as np
import pytest

from vicsek import initialize_random_particles
from vicsek.models.particle import Particle


@pytest.mark.parametrize("n_dimensions", [1, 2, 3, 5])
def test_initialize_random_particles_geometry(n_dimensions):
    particles = initialize_random_particles(40, 7.5, 0.4, n_dimensions, seed=0)
    assert len(particles) == 40
    assert all(isinstance(p, Particle) for p in particles)
    positions = np.array([p.position for p in particles])
    velocities = np.array([p.velocity for p in particles])
    assert positions.shape == (40, n_dimensions)
    assert np.all((positions >= 0) & (positions < 7.5))
    # The documented contract: every particle moves at exactly the requested speed.
    assert np.allclose(np.linalg.norm(velocities, axis=1), 0.4)


def test_initialize_random_particles_names_and_types():
    particles = initialize_random_particles(5, 1.0, 1.0, 2, particle_type="prey", seed=0)
    assert [p.name for p in particles] == ["p0", "p1", "p2", "p3", "p4"]
    assert {p.type for p in particles} == {"prey"}
    assert initialize_random_particles(1, 1.0, 1.0, 2, seed=0)[0].type == "particle"


def test_initialize_random_particles_seed_reproducible():
    a = initialize_random_particles(10, 10.0, 0.1, 3, seed=123)
    b = initialize_random_particles(10, 10.0, 0.1, 3, seed=123)
    c = initialize_random_particles(10, 10.0, 0.1, 3, seed=124)
    for pa, pb in zip(a, b):
        assert np.array_equal(pa.position, pb.position)
        assert np.array_equal(pa.velocity, pb.velocity)
    assert not np.allclose([p.position for p in a], [p.position for p in c])


def test_initialize_random_particles_headings_are_not_biased():
    """Headings come from normalised Gaussian vectors, so they are isotropic:
    the mean heading of 4000 particles has length about 1/sqrt(4000) ~ 0.016.
    A bound of 0.06 is nearly four standard deviations (rejects uniform(-1, 1)^d
    velocities only if normalised, so this guards isotropy, not just speed)."""
    particles = initialize_random_particles(4000, 1.0, 1.0, 2, seed=9)
    headings = np.array([p.velocity for p in particles])
    assert np.linalg.norm(headings.mean(axis=0)) < 0.06
    # Roughly a quarter of headings in each quadrant (binomial sd ~ 27 of 1000).
    quadrant_counts = np.histogram(np.arctan2(headings[:, 1], headings[:, 0]),
                                   bins=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])[0]
    assert np.all(np.abs(quadrant_counts - 1000) < 110)


def test_zero_particles():
    assert initialize_random_particles(0, 1.0, 1.0, 2, seed=0) == []
