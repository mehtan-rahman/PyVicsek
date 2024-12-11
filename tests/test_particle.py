import pytest
import numpy as np
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal, assert_array_almost_equal

from vicsek.models.particle import Particle


@pytest.fixture
def basic_particle():
    """Create a basic particle for testing."""
    return Particle(
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.1, 0.2]),
        name="test_particle",
        type="test_type"
    )


def test_particle_initialization():
    """Test particle initialization with different input types."""
    # Test with numpy arrays
    p1 = Particle(
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.1, 0.2]),
        name="p1",
        type="type1"
    )
    assert_array_equal(p1.position, np.array([1.0, 2.0]))
    assert_array_equal(p1.velocity, np.array([0.1, 0.2]))
    assert p1.name == "p1"
    assert p1.type == "type1"

    # Test with lists
    p2 = Particle(
        position=[3.0, 4.0],
        velocity=[0.3, 0.4],
        name="p2",
        type="type2"
    )
    assert_array_equal(p2.position, np.array([3.0, 4.0]))
    assert_array_equal(p2.velocity, np.array([0.3, 0.4]))


def test_particle_properties(basic_particle):
    """Test particle property getters and setters."""
    # Test initial values
    assert_array_equal(basic_particle.position, np.array([1.0, 2.0]))
    assert_array_equal(basic_particle.velocity, np.array([0.1, 0.2]))

    # Test setters
    new_position = np.array([5.0, 6.0])
    new_velocity = np.array([0.5, 0.6])

    basic_particle.position = new_position
    basic_particle.velocity = new_velocity

    assert_array_equal(basic_particle.position, new_position)
    assert_array_equal(basic_particle.velocity, new_velocity)


def test_particle_subtraction(basic_particle):
    """Test particle distance calculation using subtraction operator."""
    other_particle = Particle(
        position=np.array([4.0, 6.0]),
        velocity=np.array([0.0, 0.0]),
        name="other",
        type="test_type"
    )

    distance = basic_particle - other_particle
    expected_distance = 5.0  # sqrt((4-1)^2 + (6-2)^2)
    assert np.abs(distance - expected_distance) < 1e-10


def test_particle_repr(basic_particle):
    """Test string representation of particle."""
    expected_repr = "Particle(name=test_particle, pos=[1. 2.])"
    assert repr(basic_particle) == expected_repr


def test_particle_visualization():
    """Test particle visualization method."""
    particle = Particle(
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.5, 0.5]),
        name="viz_test",
        type="test_type"
    )

    # Test basic visualization
    ax = particle.visualize()
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with existing axes
    fig, ax = plt.subplots()
    result_ax = particle.visualize(ax=ax)
    assert result_ax is ax
    plt.close()

    # Test with velocity arrow
    ax = particle.visualize(show_velocity=True)
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with label
    ax = particle.visualize(label=True)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_particle_visualization_customization():
    """Test particle visualization customization options."""
    particle = Particle(
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.5, 0.5]),
        name="viz_test",
        type="test_type"
    )

    # Test custom color
    ax = particle.visualize(color='red')
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test custom size
    ax = particle.visualize(size=100)
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test custom alpha
    ax = particle.visualize(alpha=0.5)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_edge_cases():
    """Test edge cases and potential error conditions."""
    # Test with zero velocity
    p = Particle(
        position=np.array([1.0, 2.0]),
        velocity=np.array([0.0, 0.0]),
        name="zero_vel",
        type="test_type"
    )
    ax = p.visualize(show_velocity=True)
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with very large numbers
    p = Particle(
        position=np.array([1e10, 1e10]),
        velocity=np.array([1e5, 1e5]),
        name="large_nums",
        type="test_type"
    )
    assert_array_equal(p.position, np.array([1e10, 1e10]))
    assert_array_equal(p.velocity, np.array([1e5, 1e5]))