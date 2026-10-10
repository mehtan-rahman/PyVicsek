import numpy as np
import pytest

from vicsek.lyapunov.kernelized_vicsek import KernelizedVicsek
from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek
from vicsek.util.factory import initialize_random_particles


def make(particles, interaction_range=1.0, epsilon=0.1, noise=0.0, use_pbc=True, speed=0.1, **kw):
    return KernelizedVicsek(length=10.0, particles=particles, interaction_range=interaction_range,
                            speed=speed, noise_factor=noise, epsilon=epsilon, use_pbc=use_pbc, **kw)


def two_particles(separation, heading_b=(0.0, 1.0)):
    a = Particle(np.array([5.0, 5.0]), np.array([1.0, 0.0]), "a", "t")
    b = Particle(np.array([5.0 + separation, 5.0]), np.array(heading_b), "b", "t")
    return a, b


def test_is_a_vicsek_with_epsilon():
    a, b = two_particles(0.5)
    model = make([a, b], epsilon=0.05, seed=4)
    assert isinstance(model, Vicsek)
    assert model.epsilon == 0.05
    assert model._rng is not None


@pytest.mark.parametrize("interaction_range", [1.0, 2.0, 0.5])
def test_smooth_weight_shape(interaction_range):
    """The kernel is 1 up to r(1 - 2 eps), 0 beyond r, and a smooth monotone step in between."""
    eps = 0.1
    r = interaction_range
    a, b = two_particles(0.1)
    model = make([a, b], interaction_range=r, epsilon=eps)
    w = model._smooth_weight
    assert w(0.0) == 1.0
    assert w(r * (1 - 2 * eps) * 0.999) == 1.0
    assert np.isclose(w(r * (1 - 2 * eps)), 1.0)        # continuous at the inner edge
    assert np.isclose(w(r), 0.0)                        # continuous at the outer edge
    assert w(r * 1.001) == 0.0
    inner = np.linspace(r * (1 - 2 * eps), r, 200)
    values = np.array([w(d) for d in inner])
    assert np.all(np.diff(values) <= 1e-12)             # monotone decreasing
    assert np.all((values >= -1e-12) & (values <= 1 + 1e-12))   # round-off at the edges only
    assert np.isclose(w(r * (1 - eps)), 0.5)            # symmetric step: half weight at the midpoint
    # Derivative vanishes at both ends (C1 smoothness), by finite differences.
    h = 1e-6 * r
    assert abs(w(r * (1 - 2 * eps) + h) - 1.0) < 1e-9
    assert abs(w(r - h)) < 1e-9


def test_smooth_weight_uses_interaction_range():
    """With r = 2 a neighbour at distance 1.5 is well inside the range and must
    carry full weight (the original code compared the raw distance to 1)."""
    a, b = two_particles(1.5)
    model = make([a, b], interaction_range=2.0, epsilon=0.05)
    assert model._smooth_weight(1.5) == 1.0
    heading = model._compute_average_velocity(a)
    assert np.allclose(heading, np.array([1.0, 1.0]) / np.sqrt(2))


def test_average_velocity_without_neighbors_keeps_heading():
    a, b = two_particles(5.0)   # out of range
    model = make([a, b])
    assert np.allclose(model._compute_average_velocity(a), [1.0, 0.0])


def test_average_velocity_weights_fall_off_near_the_edge():
    """b at a distance where the kernel gives weight 1/2: the mean heading is
    (1 + 0, 0 + 0.5) / 1.5 normalised."""
    eps = 0.1
    a, b = two_particles(1 - eps)
    model = make([a, b], epsilon=eps)
    assert np.isclose(model._smooth_weight(1 - eps), 0.5)
    expected = np.array([1.0, 0.5])
    expected /= np.linalg.norm(expected)
    assert np.allclose(model._compute_average_velocity(a), expected)


def test_average_velocity_minimum_image_across_boundary():
    a = Particle(np.array([0.1, 5.0]), np.array([1.0, 0.0]), "a", "t")
    b = Particle(np.array([9.9, 5.0]), np.array([0.0, 1.0]), "b", "t")
    model = make([a, b], epsilon=0.1)
    # Through the wall the pair is 0.2 apart: full weight, mean heading (1, 1)/sqrt(2).
    assert np.allclose(model._compute_average_velocity(a), np.array([1.0, 1.0]) / np.sqrt(2))
    model_open = make([a, b], epsilon=0.1, use_pbc=False)
    assert np.allclose(model_open._compute_average_velocity(a), [1.0, 0.0])


def test_average_velocity_zero_mean_falls_back_to_random_heading():
    a, b = two_particles(0.2, heading_b=(-1.0, 0.0))
    model = make([a, b], seed=0)
    heading = model._compute_average_velocity(a)
    assert np.isclose(np.linalg.norm(heading), 1.0)


def test_step_is_synchronous_and_conserves_speed():
    a, b = two_particles(0.5)
    model = make([a, b], speed=0.1)
    model.step()
    expected = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(a.velocity / 0.1, expected) and np.allclose(b.velocity / 0.1, expected)
    assert np.allclose(a.position, [5.0, 5.0] + 0.1 * expected)


@pytest.mark.parametrize("use_pbc", [True, False])
def test_short_run_invariants(use_pbc):
    particles = initialize_random_particles(20, 10.0, 0.3, 2, seed=31)
    model = make(particles, interaction_range=1.5, epsilon=0.1, noise=0.2, use_pbc=use_pbc,
                 speed=0.3, seed=5)
    for _ in range(10):
        model.step()
        speeds = np.linalg.norm([p.velocity for p in particles], axis=1)
        assert np.allclose(speeds, 0.3)
        if use_pbc:
            pos = np.array([p.position for p in particles])
            assert np.all(pos >= 0) and np.all(pos < 10.0)
        assert 0.0 <= model.order_parameter() <= 1.0


def test_small_epsilon_matches_hard_cutoff_vicsek():
    """As epsilon -> 0 the kernel becomes the hard cutoff, so with zero noise the
    kernelized model and the plain Vicsek model follow identical trajectories."""
    def run(cls, **kw):
        particles = initialize_random_particles(30, 10.0, 0.2, 2, seed=32)
        model = cls(length=10.0, particles=particles, interaction_range=1.5, speed=0.2,
                    noise_factor=0.0, **kw)
        for _ in range(8):
            model.step()
        return np.array([p.position for p in particles])

    assert np.allclose(run(KernelizedVicsek, epsilon=1e-8), run(Vicsek), atol=1e-10)


@pytest.mark.parametrize("use_pbc", [True, False])
def test_step_matches_per_particle_reference(use_pbc):
    """The vectorised kernel step equals _compute_average_velocity applied to
    each particle with a replayed random stream, periodic and open."""
    from conftest import clone_rng_state
    from vicsek.util.linalg import _random_unit_vector

    particles = initialize_random_particles(120, 6.0, 0.2, 2, seed=33)
    particles[0].position = np.array([3.0, 3.0]); particles[0].velocity = np.array([1.0, 0.0])
    particles[1].position = np.array([3.3, 3.0]); particles[1].velocity = np.array([-1.0, 0.0])
    for p in particles[2:]:
        if np.linalg.norm(p.position - [3.0, 3.0]) < 1.5:
            p.position = (p.position + 3.0) % 6.0
    model = KernelizedVicsek(length=6.0, particles=particles, interaction_range=1.0, speed=0.2,
                             noise_factor=0.3, epsilon=0.2, use_pbc=use_pbc, seed=9)
    assert model._cell_list.get_neighbors(particles[0]) == [particles[1]]    # zero mean: fallback

    for _ in range(3):
        replay = clone_rng_state(model)
        gen = replay()
        model._rng = gen
        expected = []
        for p in model.particles:
            heading = model._compute_average_velocity(p)
            noisy = heading + model.mu * _random_unit_vector(2, gen)
            expected.append(model.v * noisy / np.linalg.norm(noisy))
        model._rng = replay()
        model.step()
        actual = np.array([p.velocity for p in model.particles])
        assert np.allclose(actual, np.array(expected), rtol=0, atol=1e-12)


def test_smooth_weights_vectorised_matches_scalar():
    a, b = two_particles(0.5)
    model = make([a, b], epsilon=0.1)
    d = np.linspace(0, 1.2, 50)
    assert np.array_equal(model._smooth_weights(d), np.array([model._smooth_weight(x) for x in d]))
