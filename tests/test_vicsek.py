import pytest
import numpy as np
from typing import List

from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek


def create_test_particles(n: int = 10, box_length: float = 10.0) -> List[Particle]:
    """Helper function to create test particles with random positions and velocities"""
    np.random.seed(42)  # For reproducibility
    particles = []
    for i in range(n):
        position = np.random.uniform(0, box_length, size=2)
        velocity = np.random.normal(0, 1, size=2)
        velocity = velocity / np.linalg.norm(velocity)
        particles.append(Particle(position, velocity, f"p{i}", "test"))
    return particles


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
        model_params['box_length']
    )
    return Vicsek(
        length=model_params['box_length'],
        particles=particles,
        interaction_range=model_params['interaction_range'],
        speed=model_params['speed'],
        noise_factor=model_params['noise']
    )


@pytest.fixture
def aligned_model(model_params):
    """Fixture providing a Vicsek model with perfectly aligned particles"""
    particles = []
    for i in range(model_params['n_particles']):
        position = np.random.uniform(0, model_params['box_length'], size=2)
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
    assert len(basic_model.particles) == model_params['n_particles']
    assert basic_model.dim == 2


@pytest.mark.parametrize("velocities,expected_norm", [
    (np.array([[1, 0], [0, 1], [-1, 0]]), 1.0),
    (np.array([[1, 1], [1, 1], [1, 1]]) / np.sqrt(2), 1.0),
    (np.zeros((3, 2)), 1.0),  # Edge case: zero velocities
])
def test_compute_average_velocity(basic_model, velocities, expected_norm):
    """Test average velocity computation with different input velocities"""
    avg_vel = basic_model._compute_average_velocity(velocities)
    assert np.isclose(np.linalg.norm(avg_vel), expected_norm)


@pytest.mark.parametrize("noise_factor", [0.1, 0.5, 1.0])
def test_apply_noise(model_params, noise_factor):
    """Test noise application with different noise factors"""
    particles = create_test_particles(1, model_params['box_length'])
    model = Vicsek(
        length=model_params['box_length'],
        particles=particles,
        interaction_range=model_params['interaction_range'],
        speed=model_params['speed'],
        noise_factor=noise_factor
    )

    initial_velocity = np.array([1, 0])
    noisy_vel = model._apply_noise(initial_velocity)

    assert np.isclose(np.linalg.norm(noisy_vel), 1.0)
    assert not np.allclose(initial_velocity, noisy_vel)


def test_step_evolution(basic_model):
    """Test system evolution during a single timestep"""
    initial_positions = np.array([p.position.copy() for p in basic_model.particles])
    initial_velocities = np.array([p.velocity.copy() for p in basic_model.particles])

    basic_model.step()

    final_positions = np.array([p.position for p in basic_model.particles])
    final_velocities = np.array([p.velocity for p in basic_model.particles])

    # Positions and velocities should change
    assert not np.allclose(initial_positions, final_positions)
    assert not np.allclose(initial_velocities, final_velocities)

    # All velocities should remain normalized
    speeds = np.linalg.norm(final_velocities, axis=1)
    assert np.allclose(speeds, basic_model.v)


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

    model.step()

    if use_pbc:
        # Particle should wrap around
        assert all(0 <= x <= model_params['box_length'] for x in particle.position)
    else:
        # Particle should move beyond boundary
        assert any(x > model_params['box_length'] for x in particle.position)


def test_perfect_alignment_order(aligned_model):
    """Test order parameter for perfectly aligned particles"""
    assert np.isclose(aligned_model.order_parameter(), 1.0)


def test_random_alignment_order(basic_model):
    """Test order parameter for randomly aligned particles"""
    order_param = basic_model.order_parameter()
    assert 0 <= order_param <= 1.0
    assert order_param < 1.0  # Should not be perfectly aligned


def test_equilibration(basic_model):
    """Test the equilibration process"""
    converged, steps, variance = basic_model.equilibrate(
        window_size=10,
        threshold=0.1,
        max_steps=100,
        check_interval=1,
        progress_bar=False
    )

    assert isinstance(converged, bool)
    assert isinstance(steps, int)
    assert isinstance(variance, float)
    assert steps > 0
    assert variance >= 0


@pytest.mark.parametrize("n_steps", [1, 10, 100])
def test_order_parameter_evolution(basic_model, n_steps):
    """Test order parameter evolution over multiple timesteps"""
    evolution = basic_model.order_parameter_evolution(steps=n_steps)

    assert len(evolution) == n_steps
    assert all(0 <= x <= 1.0 for x in evolution)


def test_model_persistence(basic_model):
    """Test that model maintains physical constraints over multiple steps"""
    for _ in range(10):
        basic_model.step()

        # Check velocity normalization
        velocities = np.array([p.velocity for p in basic_model.particles])
        speeds = np.linalg.norm(velocities, axis=1)
        assert np.allclose(speeds, basic_model.v)

        # Check particle containment with PBC
        positions = np.array([p.position for p in basic_model.particles])
        assert np.all((positions >= 0) & (positions <= basic_model.length))