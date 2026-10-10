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

def test_particle_becomes_a_view_of_the_model_state():
    """Once passed to a model, position and velocity read and write the model's
    (N, dim) arrays; the Particle API itself is unchanged."""
    from vicsek.models.vicsek import Vicsek

    particles = [
        Particle(np.array([1.0, 2.0]), np.array([1.0, 0.0]), "a", "t"),
        Particle([3, 4], [0, 1], "b", "t"),                # lists and ints are accepted
    ]
    model = Vicsek(length=10.0, particles=particles, interaction_range=1.0,
                   speed=0.5, noise_factor=0.0)
    assert model.positions.shape == (2, 2) and model.positions.dtype == float
    assert model.velocities.shape == (2, 2) and model.velocities.dtype == float
    assert model.particles is particles

    a, b = particles
    # Reads see the model's arrays (the velocity was rescaled to the model speed there).
    assert np.shares_memory(a.position, model.positions)
    assert np.shares_memory(b.velocity, model.velocities)
    assert_array_equal(b.velocity, [0.0, 0.5])

    # Writes through the particle land in the model's arrays, in place.
    a.position = np.array([5.0, 6.0])
    b.position += 1.0
    a.velocity[1] = 0.25
    b.velocity = [0.0, 1.0]
    assert_array_equal(model.positions, [[5.0, 6.0], [4.0, 5.0]])
    assert model.velocities[0, 1] == 0.25
    assert_array_equal(model.velocities[1], [0.0, 1.0])

    # Writes through the model are visible from the particles, and the arrays are
    # never reallocated, so views taken earlier stay valid.
    view = a.position
    model.positions = np.zeros((2, 2))
    model.velocities = np.full((2, 2), 0.5)
    assert_array_equal(view, [0.0, 0.0])
    assert_array_equal(a.position, [0.0, 0.0])
    assert_array_equal(b.velocity, [0.5, 0.5])
    model.positions[1] = [7.0, 8.0]
    assert_array_equal(b.position, [7.0, 8.0])
    assert repr(b) == "Particle(name=b, pos=[7. 8.])"

    # A shape mismatch is an error rather than a silent rebind.
    with pytest.raises(ValueError):
        a.position = np.array([1.0, 2.0, 3.0])


def test_unbound_particle_setters_rebind_the_array():
    """A free particle keeps the old behaviour: the setter replaces the array,
    so an int array can be replaced by a float one and the shape can change."""
    p = Particle(np.array([0, 0]), np.array([0, 0]), "p", "t")
    p.position = np.array([0.5, 0.5])
    p.velocity = np.array([0.1, 0.2, 0.3])
    assert p.position.dtype == float and p.position[0] == 0.5
    assert p.velocity.shape == (3,)
