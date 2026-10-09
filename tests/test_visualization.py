"""Plotting and the Vicsek phase-transition sweep, under the Agg backend."""
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek
from vicsek.util.factory import initialize_random_particles


@pytest.fixture
def model():
    particles = initialize_random_particles(12, 10.0, 0.3, 2, seed=41)
    for i, p in enumerate(particles):
        p.type = "A" if i % 3 else "B"
    return Vicsek(10.0, particles, 1.0, 0.3, 0.2, seed=6)


def test_visualize_creates_figure_and_axes(model):
    assert plt.get_fignums() == []
    ax = model.visualize()
    assert isinstance(ax, plt.Axes)
    assert len(plt.get_fignums()) == 1
    assert ax.get_xlim() == (0.0, 10.0) and ax.get_ylim() == (0.0, 10.0)
    assert ax.get_aspect() == 1.0
    assert len(ax.collections) == len(model.particles)    # one scatter per particle
    assert len(ax.patches) == 0                            # no velocity arrows by default
    assert len(ax.lines) == 0                              # no cell grid by default


def test_visualize_with_velocity_cells_and_legend(model):
    fig, ax = plt.subplots()
    legend = {"A": "red", "B": "blue"}
    result = model.visualize(ax=ax, show_velocity=True, show_cells=True, legend=legend)
    assert result is ax
    assert len(plt.get_fignums()) == 1                                   # drew on the given axes
    assert len(ax.lines) == 2 * (model._cell_list.n_cells + 1)           # the cell grid
    assert len(ax.patches) == len(model.particles)                       # one arrow each
    # Particles are drawn twice: once by the cell list (grey) and once coloured by type.
    assert len(ax.collections) == 2 * len(model.particles)
    colours = {tuple(c.get_facecolor()[0][:3]) for c in ax.collections[len(model.particles):]}
    from matplotlib.colors import to_rgb
    assert colours == {to_rgb("red"), to_rgb("blue")}


def test_visualize_default_legend_uses_fallback_colour(model):
    """legend=None used to raise AttributeError; now every particle is dimgrey."""
    from matplotlib.colors import to_rgb
    ax = model.visualize(legend=None)
    colours = {tuple(c.get_facecolor()[0][:3]) for c in ax.collections}
    assert colours == {to_rgb("dimgrey")}


def test_animate_builds_quivers_and_advances_the_model(model, tmp_path):
    anim = model.animate(frames=3, interval=10)
    assert isinstance(anim, FuncAnimation)
    fig = anim._fig
    ax = fig.axes[0]
    quivers = [c for c in ax.collections if type(c).__name__ == "Quiver"]
    assert len(quivers) == 2                               # one per particle type
    assert ax.get_legend() is not None
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"A", "B"}

    before = np.array([p.position for p in model.particles])
    anim.save(tmp_path / "anim.gif", writer=PillowWriter(fps=5))
    after = np.array([p.position for p in model.particles])
    assert (tmp_path / "anim.gif").stat().st_size > 0
    # Every frame drawn calls step() once. FuncAnimation.save draws the first frame
    # once more as the initial frame (no init_func), so frames=3 gives 4 steps, each
    # of length v * dt.
    delta = after - before
    delta -= 10.0 * np.round(delta / 10.0)
    distance = np.linalg.norm(delta, axis=1)
    assert np.all(distance <= 4 * model.v + 1e-9)
    assert np.all(distance > 3 * model.v - 0.6)     # at least 4 steps, allowing turning
    assert np.all(distance > 0)
    # The order-parameter text was updated on the last frame.
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("Order: ") for t in texts)
    shown = float([t for t in texts if t.startswith("Order: ")][0].split()[1])
    assert np.isclose(shown, model.order_parameter(), atol=5e-4)
    quiver_offsets = quivers[0].get_offsets()
    names = [p for p in model.particles if p.type == quivers[0].get_label()]
    assert np.allclose(quiver_offsets, [p.position for p in names])


def test_animate_with_explicit_legend_skips_absent_types(model):
    anim = model.animate(frames=1, legend={"A": "green", "B": "black", "C": "red"})
    ax = anim._fig.axes[0]
    quivers = [c for c in ax.collections if type(c).__name__ == "Quiver"]
    assert {q.get_label() for q in quivers} == {"A", "B"}   # no particles of type C


def test_simulate_phase_transition_applies_each_noise_value():
    """Everyone within range. At noise 0 the flock locks and the order parameter is 1
    with zero susceptibility; at noise 1.5 it is disordered; mu ends at the last value."""
    particles = initialize_random_particles(16, 10.0, 0.3, 2, seed=42)
    model = Vicsek(10.0, particles, interaction_range=20.0, speed=0.3, noise_factor=0.5, seed=7)
    noise_values = np.array([0.0, 1.5])
    order, susceptibility = model.simulate_phase_transition(
        noise_values, equilibration_steps=5, measurement_steps=10)
    assert order.shape == (2,) and susceptibility.shape == (2,)
    assert order[0] == 1.0
    assert susceptibility[0] == 0.0
    assert order[1] < 0.9
    assert susceptibility[1] > 0.0
    assert model.mu == 1.5


def test_simulate_phase_transition_susceptibility_is_n_times_variance():
    particles = initialize_random_particles(10, 10.0, 0.3, 2, seed=43)
    model = Vicsek(10.0, particles, 1.0, 0.3, 0.5, seed=8)
    # Run the sweep and recompute the susceptibility definition from a replay with the same seed.
    _, susceptibility = model.simulate_phase_transition(np.array([0.8]), equilibration_steps=2,
                                                        measurement_steps=6)
    particles = initialize_random_particles(10, 10.0, 0.3, 2, seed=43)
    replay = Vicsek(10.0, particles, 1.0, 0.3, 0.8, seed=8)
    replay.run(2)
    measurements = [replay.order_parameter_evolution(1)[0] for _ in range(6)]
    assert np.isclose(susceptibility[0], 10 * np.var(measurements))
