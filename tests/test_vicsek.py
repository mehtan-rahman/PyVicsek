import numpy as np
import pytest
from typing import List

from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek
from vicsek.util.factory import initialize_random_particles


def create_test_particles(n: int = 10, box_length: float = 10.0, speed: float = 1.0,
                          seed: int = 42) -> List[Particle]:
    """Particles with random positions and random headings at the given speed."""
    rng = np.random.default_rng(seed)
    particles = []
    for i in range(n):
        position = rng.uniform(0, box_length, size=2)
        velocity = rng.normal(0, 1, size=2)
        velocity = speed * velocity / np.linalg.norm(velocity)
        particles.append(Particle(position, velocity, f"p{i}", "test"))
    return particles


def headings(model: Vicsek) -> np.ndarray:
    return np.array([p.velocity for p in model.particles]) / model.v


def positions(model: Vicsek) -> np.ndarray:
    return np.array([p.position for p in model.particles])


def minimum_image(delta: np.ndarray, length: float) -> np.ndarray:
    return delta - length * np.round(delta / length)


def assert_invariants(model: Vicsek, use_pbc: bool = True):
    """Invariants every Vicsek state must satisfy."""
    speeds = np.linalg.norm([p.velocity for p in model.particles], axis=1)
    assert np.allclose(speeds, model.v, rtol=1e-12, atol=1e-12)
    if use_pbc:
        pos = positions(model)
        assert np.all(pos >= 0) and np.all(pos < model.length)
    assert 0.0 <= model.order_parameter() <= 1.0


@pytest.fixture
def model_params():
    """Fixture providing standard model parameters"""
    return {
        'box_length': 10.0,
        'interaction_range': 1.0,
        'speed': 0.03,
        'noise': 0.1,
        'n_particles': 10
    }


@pytest.fixture
def basic_model(model_params):
    """Fixture providing a basic Vicsek model instance"""
    particles = create_test_particles(
        model_params['n_particles'],
        model_params['box_length'],
        speed=model_params['speed'],
    )
    return Vicsek(
        length=model_params['box_length'],
        particles=particles,
        interaction_range=model_params['interaction_range'],
        speed=model_params['speed'],
        noise_factor=model_params['noise']
    )


@pytest.fixture
def aligned_model(model_params, rng):
    """Fixture providing a Vicsek model with perfectly aligned particles"""
    particles = []
    for i in range(model_params['n_particles']):
        position = rng.uniform(0, model_params['box_length'], size=2)
        velocity = np.array([1, 0]) * model_params['speed']
        particles.append(Particle(position, velocity, f"p{i}", "test"))

    return Vicsek(
        length=model_params['box_length'],
        particles=particles,
        interaction_range=model_params['interaction_range'],
        speed=model_params['speed'],
        noise_factor=0.0  # No noise for perfect alignment
    )


def test_model_initialization(basic_model, model_params):
    """Test proper initialization of model parameters"""
    assert basic_model.length == model_params['box_length']
    assert basic_model.interaction_range == model_params['interaction_range']
    assert basic_model.v == model_params['speed']
    assert basic_model.mu == model_params['noise']
    assert basic_model.delta_t == 1
    assert len(basic_model.particles) == model_params['n_particles']
    assert basic_model.dim == 2
    assert basic_model._cell_list.use_pbc is True
    assert Vicsek.__doc__ is not None and "Vicsek model" in Vicsek.__doc__


def test_initial_velocities_are_rescaled_to_speed():
    """The constructor enforces |v_i| == speed, keeping the heading."""
    particles = [
        Particle(np.array([1.0, 1.0]), np.array([3.0, 4.0]), "fast", "t"),   # |v| = 5
        Particle(np.array([2.0, 2.0]), np.array([0.0, -0.01]), "slow", "t"),  # |v| = 0.01
        Particle(np.array([3.0, 3.0]), np.array([0.0, 0.0]), "still", "t"),   # no heading
    ]
    model = Vicsek(length=10.0, particles=particles, interaction_range=1.0,
                   speed=0.5, noise_factor=0.0)
    assert np.allclose(particles[0].velocity, [0.3, 0.4])
    assert np.allclose(particles[1].velocity, [0.0, -0.5])
    assert np.isclose(np.linalg.norm(particles[2].velocity), 0.5)  # random heading, right speed
    assert 0.0 <= model.order_parameter() <= 1.0


@pytest.mark.parametrize("velocities,expected", [
    (np.array([[1, 0], [0, 1], [-1, 0]]), np.array([0.0, 1.0])),
    (np.array([[1, 1], [1, 1], [1, 1]]) / np.sqrt(2), np.array([1.0, 1.0]) / np.sqrt(2)),
    (np.array([[3, 0], [0, 4]]), np.array([0.6, 0.8])),  # magnitude-weighted mean, then normalised
])
def test_compute_average_velocity(basic_model, velocities, expected):
    """The mean heading is the normalised mean of the velocities."""
    avg_vel = basic_model._compute_average_velocity(velocities)
    assert np.allclose(avg_vel, expected)
    assert np.isclose(np.linalg.norm(avg_vel), 1.0)


def test_compute_average_velocity_zero_mean(basic_model):
    """Opposed velocities have no mean heading; the fallback is a random unit vector."""
    avg_vel = basic_model._compute_average_velocity(np.array([[1, 0], [-1, 0]]))
    assert avg_vel.shape == (2,)
    assert np.isclose(np.linalg.norm(avg_vel), 1.0)
    # Deterministic under the seeded RNG: two calls from the same seed agree.
    np.random.seed(0)
    first = basic_model._compute_average_velocity(np.zeros((3, 2)))
    np.random.seed(0)
    second = basic_model._compute_average_velocity(np.zeros((3, 2)))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("noise_factor", [0.1, 0.5, 1.0])
def test_apply_noise(model_params, noise_factor):
    """Noise is a vector of length mu added to the unit heading, then renormalised.
    The deviation angle therefore satisfies sin(theta) <= mu (theta <= 90 deg for mu = 1)."""
    particles = create_test_particles(1, model_params['box_length'])
    model = Vicsek(
        length=model_params['box_length'],
        particles=particles,
        interaction_range=model_params['interaction_range'],
        speed=model_params['speed'],
        noise_factor=noise_factor
    )

    initial_velocity = np.array([1.0, 0.0])
    for _ in range(50):
        noisy_vel = model._apply_noise(initial_velocity)
        assert np.isclose(np.linalg.norm(noisy_vel), 1.0)
        cos_theta = np.clip(np.dot(noisy_vel, initial_velocity), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        assert theta <= np.arcsin(min(noise_factor, 1.0)) + 1e-12


def test_apply_zero_noise_is_identity(model_params):
    particles = create_test_particles(1, model_params['box_length'])
    model = Vicsek(model_params['box_length'], particles, 1.0, 0.03, noise_factor=0.0)
    heading = np.array([0.6, 0.8])
    assert np.allclose(model._apply_noise(heading), heading)


def test_step_evolution(basic_model):
    """One step moves every particle by exactly speed * dt along its new heading
    and keeps every speed equal to the model speed."""
    initial_positions = positions(basic_model).copy()

    basic_model.step()

    final_positions = positions(basic_model)
    final_velocities = np.array([p.velocity for p in basic_model.particles])

    displacement = minimum_image(final_positions - initial_positions, basic_model.length)
    assert np.allclose(np.linalg.norm(displacement, axis=1), basic_model.v * basic_model.delta_t)
    # The displacement is along the updated velocity.
    assert np.allclose(displacement, final_velocities * basic_model.delta_t)
    assert_invariants(basic_model)


def test_timestep_scales_displacement():
    particles = create_test_particles(5, 10.0)
    model = Vicsek(10.0, particles, 1.0, speed=0.2, noise_factor=0.0, timestep=2.5)
    before = positions(model).copy()
    model.step()
    displacement = minimum_image(positions(model) - before, 10.0)
    assert np.allclose(np.linalg.norm(displacement, axis=1), 0.2 * 2.5)


@pytest.mark.parametrize("use_pbc", [True, False])
def test_boundary_conditions(model_params, use_pbc):
    """Test both periodic and non-periodic boundary conditions"""
    # Create particle near boundary
    particle = Particle(
        position=np.array([model_params['box_length'] - 0.1,
                           model_params['box_length'] - 0.1]),
        velocity=np.array([1, 1]) / np.sqrt(2),
        name="test",
        type="test"
    )

    model = Vicsek(
        length=model_params['box_length'],
        particles=[particle],
        interaction_range=model_params['interaction_range'],
        speed=model_params['speed'],
        noise_factor=0.0,
        use_pbc=use_pbc
    )

    # Each step moves the particle speed / sqrt(2) along each axis; take enough
    # steps to travel further than the 0.1 gap to the wall (0.03 * 10 / sqrt(2) ~ 0.21).
    n_steps = 10
    for _ in range(n_steps):
        model.step()

    travelled = n_steps * model_params['speed'] / np.sqrt(2)
    unwrapped = (model_params['box_length'] - 0.1) + travelled
    assert unwrapped > model_params['box_length']  # the test really crosses the wall

    if use_pbc:
        # Particle should wrap around to just inside the opposite wall
        expected = unwrapped % model_params['box_length']
        assert np.allclose(particle.position, expected)
        assert all(0 <= x < model_params['box_length'] for x in particle.position)
    else:
        # Particle should move beyond boundary
        assert np.allclose(particle.position, unwrapped)
        assert all(x > model_params['box_length'] for x in particle.position)


def test_isolated_particle_keeps_heading(model_params):
    """With no neighbors and no noise a particle moves in a straight line."""
    particle = Particle(np.array([5.0, 5.0]), np.array([0.0, 1.0]), "lone", "t")
    model = Vicsek(model_params['box_length'], [particle], 1.0, speed=0.1, noise_factor=0.0)
    for k in range(1, 6):
        model.step()
        assert np.allclose(particle.velocity, [0.0, 0.1])
        assert np.allclose(particle.position, [5.0, 5.0 + 0.1 * k])


def test_perfect_alignment_order(aligned_model):
    """Test order parameter for perfectly aligned particles"""
    assert np.isclose(aligned_model.order_parameter(), 1.0)
    # Zero noise keeps a perfectly aligned flock perfectly aligned.
    for _ in range(5):
        aligned_model.step()
        assert np.allclose(headings(aligned_model), [1.0, 0.0])
    assert aligned_model.order_parameter() == 1.0
    assert_invariants(aligned_model)


def test_random_alignment_order(basic_model):
    """Test order parameter for randomly aligned particles"""
    # The constructor rescales every velocity to |v| == speed, so the order
    # parameter |<v>| / v is the length of the mean unit heading and lies in [0, 1].
    speeds = np.linalg.norm([p.velocity for p in basic_model.particles], axis=1)
    assert np.allclose(speeds, basic_model.v)

    expected = np.linalg.norm(headings(basic_model).mean(axis=0))
    order_param = basic_model.order_parameter()
    assert np.isclose(order_param, expected)
    assert 0 <= order_param <= 1.0
    assert order_param < 1.0  # Should not be perfectly aligned


def test_order_parameter_of_opposed_pair_is_zero():
    particles = [
        Particle(np.array([1.0, 1.0]), np.array([1.0, 0.0]), "a", "t"),
        Particle(np.array([8.0, 8.0]), np.array([-1.0, 0.0]), "b", "t"),
    ]
    model = Vicsek(10.0, particles, 1.0, 0.03, 0.1)
    assert model.order_parameter() == 0.0


def test_zero_noise_converges_to_common_heading():
    """All particles within one interaction range and no noise: headings converge
    to a single common heading and the order parameter goes to 1."""
    particles = initialize_random_particles(30, 10.0, 0.5, 2, seed=3)
    model = Vicsek(length=10.0, particles=particles, interaction_range=20.0,
                   speed=0.5, noise_factor=0.0)
    assert model._cell_list.n_cells == 1  # everybody is everybody's neighbor
    assert model.order_parameter() < 0.5

    for _ in range(20):
        model.step()
        assert_invariants(model)

    h = headings(model)
    assert np.max(np.linalg.norm(h - h[0], axis=1)) < 1e-8
    assert np.isclose(model.order_parameter(), 1.0, atol=1e-12)


def test_equilibration_converges_for_aligned_flock(aligned_model):
    """An aligned flock without noise has zero variance, so equilibrate() returns
    at min_steps exactly, with zero variance."""
    converged, steps, variance = aligned_model.equilibrate(
        window_size=5,
        threshold=1e-6,
        max_steps=50,
        check_interval=1,
        min_steps=10,
        progress_bar=False
    )
    assert converged is True
    assert steps == 10
    assert variance == 0.0


def test_equilibration_stops_at_max_steps(aligned_model):
    """max_steps below min_steps can never satisfy the window, so it fails after max_steps."""
    converged, steps, variance = aligned_model.equilibrate(
        window_size=5, threshold=1e-6, max_steps=7, min_steps=10, progress_bar=False)
    assert converged is False
    assert steps == 7
    assert variance == 0.0


def test_equilibration_check_interval(aligned_model):
    """With check_interval=3 only every third step is recorded, so filling a window
    of 4 requires 10 steps (records at steps 0, 3, 6, 9)."""
    converged, steps, _ = aligned_model.equilibrate(
        window_size=4, threshold=1e-6, max_steps=50, check_interval=3,
        min_steps=1, progress_bar=False)
    assert converged is True
    assert steps == 10


def test_equilibration_with_progress_bar(aligned_model):
    """Both the converged and the exhausted paths close the tqdm bar cleanly."""
    converged, steps, _ = aligned_model.equilibrate(
        window_size=2, threshold=1e-6, max_steps=20, min_steps=2, progress_bar=True)
    assert (converged, steps) == (True, 2)
    converged, steps, _ = aligned_model.equilibrate(
        window_size=2, threshold=-1.0, max_steps=3, min_steps=1, progress_bar=True)
    assert (converged, steps) == (False, 3)


def test_equilibration_noisy_model_runs_to_threshold(basic_model):
    """A noisy model's order parameter has nonzero variance."""
    converged, steps, variance = basic_model.equilibrate(
        window_size=10, threshold=-1.0, max_steps=15, min_steps=10, progress_bar=False)
    assert converged is False
    assert steps == 15
    assert variance > 0.0
    assert_invariants(basic_model)


@pytest.mark.parametrize("n_steps", [1, 10, 50])
def test_order_parameter_evolution(basic_model, n_steps):
    """Test order parameter evolution over multiple timesteps"""
    evolution = basic_model.order_parameter_evolution(steps=n_steps)

    assert evolution.shape == (n_steps,)
    assert np.all((evolution >= 0) & (evolution <= 1.0))
    # The last recorded value is the current order parameter.
    assert np.isclose(evolution[-1], basic_model.order_parameter())


def test_run_advances_the_system(basic_model):
    before = positions(basic_model).copy()
    basic_model.run(iterations=3)
    after = positions(basic_model)
    # Three steps of length v each: no particle can be further than 3 v from where it started.
    displacement = np.linalg.norm(minimum_image(after - before, basic_model.length), axis=1)
    assert np.all(displacement <= 3 * basic_model.v + 1e-12)
    assert np.all(displacement > 0)
    assert_invariants(basic_model)


def test_model_persistence(basic_model):
    """Test that model maintains physical constraints over multiple steps"""
    for _ in range(10):
        basic_model.step()
        assert_invariants(basic_model)


def test_determinism_under_seed(model_params):
    """Two runs from the same seed produce identical trajectories."""
    def run():
        np.random.seed(7)
        particles = create_test_particles(8, 10.0, speed=0.1)
        model = Vicsek(10.0, particles, 1.0, 0.1, 0.3)
        for _ in range(5):
            model.step()
        return positions(model), headings(model)

    pos_a, head_a = run()
    pos_b, head_b = run()
    assert np.array_equal(pos_a, pos_b)
    assert np.array_equal(head_a, head_b)


def test_step_is_synchronous():
    """Vicsek 1995 updates every heading from the state at the start of the step.
    a heads +x and b heads +y within one range of each other and no noise: both must
    end up at (1, 1)/sqrt(2). A sequential sweep would give b the mean of a's
    already-updated heading and its own, (1, 1 + sqrt(2)) normalised, instead."""
    a = Particle(np.array([5.0, 5.0]), np.array([1.0, 0.0]), "a", "t")
    b = Particle(np.array([5.5, 5.0]), np.array([0.0, 1.0]), "b", "t")
    model = Vicsek(10.0, [a, b], 1.0, speed=0.1, noise_factor=0.0)
    model.step()
    expected = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(a.velocity / 0.1, expected)
    assert np.allclose(b.velocity / 0.1, expected)
    # Positions move with the new velocity.
    assert np.allclose(a.position, [5.0, 5.0] + 0.1 * expected)
    assert np.allclose(b.position, [5.5, 5.0] + 0.1 * expected)


def test_step_result_is_independent_of_particle_order():
    """Shuffling the particle list must not change any particle's trajectory.
    Zero noise so the only possible source of difference is the update order."""
    def final_state(order_seed):
        particles = create_test_particles(25, 10.0, speed=0.3, seed=11)
        np.random.default_rng(order_seed).shuffle(particles)
        model = Vicsek(10.0, particles, interaction_range=2.0, speed=0.3, noise_factor=0.0)
        for _ in range(5):
            model.step()
        return {p.name: (p.position.copy(), p.velocity.copy()) for p in particles}

    reference = final_state(0)
    for order_seed in (1, 2, 3):
        other = final_state(order_seed)
        for name, (pos, vel) in reference.items():
            assert np.allclose(other[name][0], pos, atol=1e-12)
            assert np.allclose(other[name][1], vel, atol=1e-12)


def test_model_seed_gives_reproducible_trajectories():
    """The seed argument makes a run reproducible without touching np.random."""
    def run(seed):
        np.random.seed(int(seed) + 1000)  # perturb the global RNG: it must not matter
        particles = create_test_particles(10, 10.0, speed=0.1, seed=1)
        model = Vicsek(10.0, particles, 1.0, 0.1, 0.4, seed=seed)
        for _ in range(5):
            model.step()
        return positions(model), headings(model)

    pos_a, head_a = run(3)
    np.random.seed(0)
    pos_b, head_b = run(3)
    assert np.array_equal(pos_a, pos_b) and np.array_equal(head_a, head_b)
    pos_c, _ = run(4)
    assert not np.allclose(pos_a, pos_c)


def test_model_accepts_generator_and_defaults_to_global_rng():
    particles = create_test_particles(3, 10.0)
    gen = np.random.default_rng(5)
    model = Vicsek(10.0, particles, 1.0, 0.1, 0.2, seed=gen)
    assert model._rng is gen
    model_default = Vicsek(10.0, create_test_particles(3, 10.0), 1.0, 0.1, 0.2)
    assert model_default._rng is None
    # seed=None draws from numpy's global RNG, so np.random.seed controls it.
    np.random.seed(1)
    first = model_default._apply_noise(np.array([1.0, 0.0]))
    np.random.seed(1)
    second = model_default._apply_noise(np.array([1.0, 0.0]))
    assert np.array_equal(first, second)
    # A seeded Generator is consumed independently of the global RNG.
    np.random.seed(1)
    noisy_a = model._apply_noise(np.array([1.0, 0.0]))
    np.random.seed(1)
    noisy_b = model._apply_noise(np.array([1.0, 0.0]))
    assert not np.array_equal(noisy_a, noisy_b)
