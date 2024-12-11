import pytest
import numpy as np
from typing import List, Dict, Tuple
from unittest.mock import Mock, patch

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


def test_initialization(basic_model, basic_particles, weight_matrices):
    """Test proper initialization of the HeterogeneousVicsek model"""
    alignment_weights, noise_weights = weight_matrices

    assert isinstance(basic_model, Vicsek)  # Should inherit from Vicsek
    assert basic_model.particles == basic_particles
    assert basic_model.alignment_weights == alignment_weights
    assert basic_model.noise_weights == noise_weights
    assert set(basic_model.particle_types) == {"A", "B"}


def test_initialization_missing_weights():
    """Test initialization with missing weights raises ValueError"""
    particles = [
        Particle(position=np.array([0.0, 0.0]), velocity=np.array([1.0, 0.0]), name="p1", type="A"),
        Particle(position=np.array([0.5, 0.5]), velocity=np.array([0.0, 1.0]), name="p2", type="B"),
    ]

    incomplete_weights = {("A", "A"): 1.0}  # Missing other combinations

    with pytest.raises(ValueError):
        HeterogeneousVicsek(
            length=10.0,
            particles=particles,
            interaction_range=1.0,
            speed=1.0,
            base_noise=0.1,
            alignment_weights=incomplete_weights,
            noise_weights=incomplete_weights
        )


def test_compute_weighted_velocity(basic_model):
    """Test weighted velocity computation"""
    particle = basic_model.particles[0]
    neighbors = [basic_model.particles[1]]

    result = basic_model._compute_weighted_velocity(particle, neighbors)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.allclose(np.linalg.norm(result), 1.0)  # Should be normalized


def test_compute_weighted_velocity_no_neighbors(basic_model):
    """Test weighted velocity computation with no neighbors"""
    particle = basic_model.particles[0]
    result = basic_model._compute_weighted_velocity(particle, [])
    assert np.allclose(result, particle.velocity / np.linalg.norm(particle.velocity))


def test_compute_effective_noise(basic_model):
    """Test effective noise computation"""
    particle = basic_model.particles[0]
    neighbors = [basic_model.particles[1]]

    noise = basic_model._compute_effective_noise(particle, neighbors)
    assert isinstance(noise, float)
    assert noise >= 0.0
    assert noise <= basic_model.mu * max(basic_model.noise_weights.values())


def test_get_type_specific_order(basic_model):
    """Test type-specific order parameter calculation"""
    order_params = basic_model.get_type_specific_order()

    assert isinstance(order_params, dict)
    assert set(order_params.keys()) == {"A", "B"}
    assert all(0 <= v <= 1 for v in order_params.values())


def test_compute_cross_correlations(basic_model):
    """Test cross-correlation computation"""
    correlations = basic_model.compute_cross_correlations()

    assert isinstance(correlations, dict)
    assert ("A", "B") in correlations
    assert all(-1 <= v <= 1 for v in correlations.values())


@pytest.mark.parametrize("noise", [0.1, 0.5])
def test_step_evolution(basic_model, noise):
    """Test system evolution for different noise values"""
    basic_model.mu = noise
    initial_positions = np.array([p.position.copy() for p in basic_model.particles])

    basic_model.step()

    final_positions = np.array([p.position for p in basic_model.particles])
    assert not np.allclose(initial_positions, final_positions)  # Positions should change
    assert all(np.all(p.position >= 0) and np.all(p.position < basic_model.length)
               for p in basic_model.particles)  # Should respect boundaries


@patch('tqdm.tqdm')
def test_simulate_phase_transition(mock_tqdm, basic_model):
    """Test phase transition simulation"""
    noise_values = np.array([0.1, 0.2])
    results = basic_model.simulate_phase_transition(
        noise_values=noise_values,
        equilibration_steps=2,
        measurement_steps=2
    )

    global_order, global_fluct, type_orders, type_fluct, cross_corr = results

    assert isinstance(global_order, np.ndarray)
    assert isinstance(global_fluct, np.ndarray)
    assert isinstance(type_orders, dict)
    assert isinstance(type_fluct, dict)
    assert isinstance(cross_corr, dict)

    assert len(global_order) == len(noise_values)
    assert all(0 <= x <= 1 for x in global_order)