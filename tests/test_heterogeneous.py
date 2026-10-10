import numpy as np
import pytest
from typing import List, Dict, Tuple

from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek
from vicsek.models.heterogeneous import HeterogeneousVicsek


@pytest.fixture
def basic_particles() -> List[Particle]:
    """Create a basic set of test particles"""
    return [
        Particle(position=np.array([0.0, 0.0]), velocity=np.array([1.0, 0.0]), name="p1", type="A"),
        Particle(position=np.array([0.5, 0.5]), velocity=np.array([0.0, 1.0]), name="p2", type="B"),
    ]


@pytest.fixture
def weight_matrices() -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Create basic weight matrices for testing"""
    alignment_weights = {
        ("A", "A"): 1.0, ("A", "B"): 0.5,
        ("B", "A"): 0.5, ("B", "B"): 1.0
    }
    noise_weights = {
        ("A", "A"): 1.0, ("A", "B"): 1.0,
        ("B", "A"): 1.0, ("B", "B"): 1.0
    }
    return alignment_weights, noise_weights


@pytest.fixture
def basic_model(basic_particles, weight_matrices) -> HeterogeneousVicsek:
    """Create a basic heterogeneous Vicsek model for testing"""
    alignment_weights, noise_weights = weight_matrices
    return HeterogeneousVicsek(
        length=10.0,
        particles=basic_particles,
        interaction_range=1.0,
        speed=1.0,
        base_noise=0.1,
        alignment_weights=alignment_weights,
        noise_weights=noise_weights
    )


def make_model(particles, alignment, noise, base_noise=0.0, speed=1.0, interaction_range=1.0):
    return HeterogeneousVicsek(
        length=10.0, particles=particles, interaction_range=interaction_range, speed=speed,
        base_noise=base_noise, alignment_weights=alignment, noise_weights=noise)


def uniform_weights(types, value=1.0):
    return {(a, b): value for a in types for b in types}


def test_initialization(basic_model, basic_particles, weight_matrices):
    """Test proper initialization of the HeterogeneousVicsek model"""
    alignment_weights, noise_weights = weight_matrices

    assert isinstance(basic_model, Vicsek)  # Should inherit from Vicsek
    assert basic_model.particles == basic_particles
    assert basic_model.alignment_weights == alignment_weights
    assert basic_model.noise_weights == noise_weights
    # Sorted, so the key order of every returned dictionary is reproducible.
    assert basic_model.particle_types == ["A", "B"]
    assert basic_model.mu == 0.1


def test_initialization_missing_alignment_weight(basic_particles, weight_matrices):
    _, noise_weights = weight_matrices
    with pytest.raises(ValueError, match="alignment weight"):
        make_model(basic_particles, {("A", "A"): 1.0}, noise_weights)


def test_initialization_missing_noise_weight(basic_particles, weight_matrices):
    alignment_weights, _ = weight_matrices
    with pytest.raises(ValueError, match="noise weight"):
        make_model(basic_particles, alignment_weights, {("A", "A"): 1.0})


def test_compute_weighted_velocity(basic_model):
    """A: weight 1 on itself (1, 0), weight 0.5 on neighbor B (0, 1):
    mean = (1, 0.5) / 1.5, normalised."""
    particle = basic_model.particles[0]
    neighbors = [basic_model.particles[1]]

    result = basic_model._compute_weighted_velocity(particle, neighbors)
    expected = np.array([1.0, 0.5])
    expected /= np.linalg.norm(expected)
    assert np.allclose(result, expected)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_compute_weighted_velocity_no_neighbors(basic_model):
    """Test weighted velocity computation with no neighbors"""
    particle = basic_model.particles[0]
    result = basic_model._compute_weighted_velocity(particle, [])
    assert np.allclose(result, particle.velocity / np.linalg.norm(particle.velocity))


def test_anti_alignment_flips_neighbor_heading():
    """A negative weight makes a particle align against its neighbor's heading."""
    a = Particle(np.array([1.0, 1.0]), np.array([0.0, 1.0]), "a", "A")
    b = Particle(np.array([1.5, 1.0]), np.array([1.0, 0.0]), "b", "B")
    alignment = {("A", "A"): 1.0, ("A", "B"): -1.0, ("B", "A"): 1.0, ("B", "B"): 1.0}
    model = make_model([a, b], alignment, uniform_weights(["A", "B"]))
    # a weights itself +1 at (0, 1) and b at -1 so b's heading (1, 0) counts as (-1, 0).
    result = model._compute_weighted_velocity(a, [b])
    assert np.allclose(result, np.array([-1.0, 1.0]) / np.sqrt(2))
    # b (all positive weights) still averages towards a.
    result_b = model._compute_weighted_velocity(b, [a])
    assert np.allclose(result_b, np.array([1.0, 1.0]) / np.sqrt(2))


def test_weighted_velocity_zero_mean_falls_back_to_random_heading():
    a = Particle(np.array([1.0, 1.0]), np.array([1.0, 0.0]), "a", "A")
    b = Particle(np.array([1.5, 1.0]), np.array([-1.0, 0.0]), "b", "A")
    model = make_model([a, b], uniform_weights(["A"]), uniform_weights(["A"]))
    result = model._compute_weighted_velocity(a, [b])
    assert result.shape == (2,)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_compute_effective_noise(basic_model):
    """Effective noise is base noise times the mean of self and neighbor weights."""
    particle = basic_model.particles[0]
    neighbors = [basic_model.particles[1]]

    noise = basic_model._compute_effective_noise(particle, neighbors)
    assert isinstance(noise, float)
    assert np.isclose(noise, 0.1 * np.mean([1.0, 1.0]))

    basic_model.noise_weights[("A", "B")] = 3.0
    assert np.isclose(basic_model._compute_effective_noise(particle, neighbors), 0.1 * np.mean([3.0, 1.0]))
    # Without neighbors only the self weight counts.
    basic_model.noise_weights[("A", "A")] = 0.5
    assert np.isclose(basic_model._compute_effective_noise(particle, []), 0.1 * 0.5)


def test_get_type_specific_order(basic_model):
    """Each type here has a single particle, so each is perfectly ordered."""
    order_params = basic_model.get_type_specific_order()
    assert order_params == {"A": 1.0, "B": 1.0}


def test_get_type_specific_order_mixed():
    particles = [
        Particle(np.array([1.0, 1.0]), np.array([1.0, 0.0]), "a1", "A"),
        Particle(np.array([8.0, 8.0]), np.array([-1.0, 0.0]), "a2", "A"),   # opposed: A disordered
        Particle(np.array([3.0, 3.0]), np.array([0.0, 1.0]), "b1", "B"),
        Particle(np.array([6.0, 6.0]), np.array([1.0, 0.0]), "b2", "B"),    # perpendicular
    ]
    model = make_model(particles, uniform_weights(["A", "B"]), uniform_weights(["A", "B"]))
    order = model.get_type_specific_order()
    assert np.isclose(order["A"], 0.0)
    assert np.isclose(order["B"], np.sqrt(0.5))
    # Global order is the length of the mean of all four headings: (1, 1) / 4.
    assert np.isclose(model.order_parameter(), np.sqrt(2) / 4)


def test_compute_cross_correlations(basic_model):
    """A heads along x, B along y: self-correlations 1, cross-correlation 0, upper triangle only."""
    correlations = basic_model.compute_cross_correlations()
    assert set(correlations) == {("A", "A"), ("A", "B"), ("B", "B")}
    assert correlations[("A", "A")] == 1.0
    assert correlations[("B", "B")] == 1.0
    assert np.isclose(correlations[("A", "B")], 0.0)


def test_cross_correlation_values():
    particles = [
        Particle(np.array([1.0, 1.0]), np.array([1.0, 0.0]), "a1", "A"),
        Particle(np.array([8.0, 8.0]), np.array([-1.0, 0.0]), "a2", "A"),   # A has zero mean velocity
        Particle(np.array([3.0, 3.0]), np.array([1.0, 1.0]), "b1", "B"),
        Particle(np.array([6.0, 6.0]), np.array([1.0, 1.0]), "c1", "C"),
        Particle(np.array([6.0, 1.0]), np.array([-1.0, -1.0]), "c2", "C"),
        Particle(np.array([1.0, 6.0]), np.array([-1.0, -1.0]), "c3", "C"),  # C points opposite to B
    ]
    types = ["A", "B", "C"]
    model = make_model(particles, uniform_weights(types), uniform_weights(types))
    corr = model.compute_cross_correlations()
    assert corr[("A", "B")] == 0.0  # disordered type: defined as 0, not NaN
    assert corr[("A", "A")] == 0.0
    assert np.isclose(corr[("B", "C")], -1.0)
    assert corr[("B", "B")] == 1.0 and corr[("C", "C")] == 1.0
    assert all(-1.0 <= v <= 1.0 for v in corr.values())


@pytest.mark.parametrize("noise", [0.1, 0.5])
def test_step_evolution(basic_model, noise):
    """Each step moves every particle by exactly speed * dt and keeps speeds fixed."""
    basic_model.mu = noise
    initial_positions = np.array([p.position.copy() for p in basic_model.particles])

    basic_model.step()

    final_positions = np.array([p.position for p in basic_model.particles])
    displacement = final_positions - initial_positions
    displacement -= basic_model.length * np.round(displacement / basic_model.length)
    assert np.allclose(np.linalg.norm(displacement, axis=1), basic_model.v * basic_model.delta_t)
    speeds = np.linalg.norm([p.velocity for p in basic_model.particles], axis=1)
    assert np.allclose(speeds, basic_model.v)
    assert all(np.all(p.position >= 0) and np.all(p.position < basic_model.length)
               for p in basic_model.particles)  # Should respect boundaries


def test_step_without_pbc_lets_particles_leave():
    a = Particle(np.array([9.95, 5.0]), np.array([1.0, 0.0]), "a", "A")
    model = HeterogeneousVicsek(10.0, [a], 1.0, 0.1, 0.0, uniform_weights(["A"]),
                                uniform_weights(["A"]), use_pbc=False)
    model.step()
    assert np.allclose(a.position, [10.05, 5.0])


def test_zero_noise_cooperative_flock_aligns():
    """Positive weights, zero noise, everyone within range: both types converge to one heading."""
    rng = np.random.default_rng(5)
    particles = []
    for i in range(20):
        v = rng.normal(size=2)
        particles.append(Particle(rng.uniform(0, 10, 2), v / np.linalg.norm(v), f"p{i}", "A" if i % 2 else "B"))
    model = make_model(particles, uniform_weights(["A", "B"]), uniform_weights(["A", "B"]),
                       base_noise=0.0, interaction_range=20.0)
    for _ in range(20):
        model.step()
    assert np.isclose(model.order_parameter(), 1.0, atol=1e-10)
    corr = model.compute_cross_correlations()
    assert np.isclose(corr[("A", "B")], 1.0)
    assert all(np.isclose(v, 1.0) for v in model.get_type_specific_order().values())


def test_zero_noise_antagonistic_types_anti_align():
    """A aligns with A, B with B, and each anti-aligns with the other: the two types
    end up pointing in opposite directions (cross-correlation -1)."""
    rng = np.random.default_rng(6)
    particles = []
    for i in range(20):
        v = rng.normal(size=2)
        particles.append(Particle(rng.uniform(0, 10, 2), v / np.linalg.norm(v), f"p{i}", "A" if i % 2 else "B"))
    alignment = {("A", "A"): 1.0, ("B", "B"): 1.0, ("A", "B"): -1.0, ("B", "A"): -1.0}
    model = make_model(particles, alignment, uniform_weights(["A", "B"]),
                       base_noise=0.0, interaction_range=20.0)
    for _ in range(40):
        model.step()
    corr = model.compute_cross_correlations()
    assert np.isclose(corr[("A", "B")], -1.0, atol=1e-6)
    order = model.get_type_specific_order()
    assert np.isclose(order["A"], 1.0, atol=1e-6) and np.isclose(order["B"], 1.0, atol=1e-6)
    assert np.isclose(model.order_parameter(), 0.0, atol=1e-6)  # equal numbers cancel


def test_simulate_phase_transition(basic_model):
    """Test phase transition simulation"""
    noise_values = np.array([0.1, 0.2])
    results = basic_model.simulate_phase_transition(
        noise_values=noise_values,
        equilibration_steps=2,
        measurement_steps=2
    )

    global_order, global_fluct, type_orders, type_fluct, cross_corr = results

    assert global_order.shape == (2,)
    assert global_fluct.shape == (2,)
    assert set(type_orders) == {"A", "B"} and set(type_fluct) == {"A", "B"}
    assert set(cross_corr) == {(a, b) for a in "AB" for b in "AB"}
    for arr in list(type_orders.values()) + list(type_fluct.values()):
        assert arr.shape == (2,)
    # Only the upper triangle is ever measured; the lower-triangle keys stay empty.
    assert cross_corr[("A", "B")].shape == (2,)
    assert cross_corr[("B", "A")].shape == (0,)

    assert np.all((global_order >= 0) & (global_order <= 1))
    assert np.all(global_fluct >= 0)
    assert basic_model.mu == 0.2  # the last noise value was applied


def test_phase_transition_zero_noise_point_is_ordered():
    rng = np.random.default_rng(8)
    particles = []
    for i in range(12):
        v = rng.normal(size=2)
        particles.append(Particle(rng.uniform(0, 10, 2), v / np.linalg.norm(v), f"p{i}", "A" if i % 3 else "B"))
    model = make_model(particles, uniform_weights(["A", "B"]), uniform_weights(["A", "B"]),
                       base_noise=0.5, interaction_range=20.0)
    global_order, global_fluct, type_orders, type_fluct, cross_corr = model.simulate_phase_transition(
        noise_values=np.array([0.0, 2.0]), equilibration_steps=20, measurement_steps=5)
    assert np.isclose(global_order[0], 1.0, atol=1e-8)
    assert np.isclose(global_fluct[0], 0.0, atol=1e-8)
    assert np.isclose(cross_corr[("A", "B")][0], 1.0, atol=1e-8)
    assert global_order[1] < global_order[0]


def test_heterogeneous_step_is_synchronous():
    """Both particles see the other's old heading: with weight 1 everywhere and no
    noise both end at (1, 1)/sqrt(2)."""
    a = Particle(np.array([5.0, 5.0]), np.array([1.0, 0.0]), "a", "A")
    b = Particle(np.array([5.5, 5.0]), np.array([0.0, 1.0]), "b", "B")
    model = make_model([a, b], uniform_weights(["A", "B"]), uniform_weights(["A", "B"]))
    model.step()
    expected = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(a.velocity, expected) and np.allclose(b.velocity, expected)


def test_heterogeneous_seed_is_reproducible():
    def run(seed):
        rng = np.random.default_rng(2)
        particles = [Particle(rng.uniform(0, 10, 2), rng.normal(size=2), f"p{i}", "A" if i % 2 else "B")
                     for i in range(10)]
        np.random.seed(seed + 50)  # global RNG must be irrelevant when a seed is given
        model = HeterogeneousVicsek(10.0, particles, 1.5, 1.0, 0.4, uniform_weights(["A", "B"]),
                                    uniform_weights(["A", "B"]), seed=seed)
        for _ in range(4):
            model.step()
        return np.array([p.position for p in particles])

    assert np.array_equal(run(7), run(7))
    assert not np.allclose(run(7), run(8))


def test_step_matches_per_particle_reference():
    """The vectorised step equals the per-particle definition built from
    _compute_weighted_velocity and _compute_effective_noise with a replayed
    random stream: three types, negative and zero weights, unequal noise
    weights, lone particles and an opposed pair (random fallback)."""
    from conftest import clone_rng_state
    from vicsek.util.linalg import _random_unit_vector

    rng = np.random.default_rng(12)
    types = ["A", "B", "C"]
    particles = [Particle(rng.uniform(0, 8, 2), rng.normal(size=2), f"p{i}", types[i % 3]) for i in range(90)]
    def min_image_distance(a, b):
        delta = a - b
        return np.linalg.norm(delta - 8.0 * np.round(delta / 8.0))

    particles[0].position = np.array([4.0, 4.0]); particles[0].velocity = np.array([1.0, 0.0])
    particles[1].position = np.array([4.1, 4.0]); particles[1].velocity = np.array([-1.0, 0.0])
    particles[2].position = np.array([1.0, 1.0])                        # a lone particle
    for p in particles[3:]:
        if min_image_distance(p.position, [4.0, 4.0]) < 1.5:
            p.position = (p.position + 4.0) % 8.0
    for p in particles[3:]:
        if min_image_distance(p.position, [1.0, 1.0]) < 1.5:
            p.position = (p.position + [0.0, 4.0]) % 8.0
    alignment = {("A", "A"): 1.0, ("A", "B"): -0.5, ("A", "C"): 0.0,
                 ("B", "A"): 2.0, ("B", "B"): -1.0, ("B", "C"): 0.3,
                 ("C", "A"): 0.7, ("C", "B"): 1.5, ("C", "C"): 1.0}
    noise = {(a, b): 0.5 + 0.25 * (i % 4) for i, (a, b) in enumerate((a, b) for a in types for b in types)}
    model = HeterogeneousVicsek(8.0, particles, 1.0, 0.2, 0.3, alignment, noise, seed=4)
    assert model._cell_list.get_neighbors(particles[0]) == [particles[1]]
    assert model._cell_list.get_neighbors(particles[2]) == []

    for _ in range(3):
        replay = clone_rng_state(model)
        gen = replay()
        model._rng = gen            # reference draws from the replayed stream ...
        expected = []
        for p in model.particles:
            neighbors = model._cell_list.get_neighbors(p)
            heading = model._compute_weighted_velocity(p, neighbors)
            eff = model._compute_effective_noise(p, neighbors)
            noisy = heading + eff * _random_unit_vector(2, gen)
            expected.append(model.v * noisy / np.linalg.norm(noisy))
        model._rng = replay()       # ... and the model from an identical copy
        model.step()
        actual = np.array([p.velocity for p in model.particles])
        assert np.allclose(actual, np.array(expected), rtol=0, atol=1e-12)
