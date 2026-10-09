"""Short simulations in 3-D (and a 4-D smoke test) checking the model invariants."""
import numpy as np
import pytest

from vicsek.models.heterogeneous import HeterogeneousVicsek
from vicsek.models.vicsek import Vicsek
from vicsek.util.factory import initialize_random_particles


def check_invariants(model, use_pbc=True):
    velocities = np.array([p.velocity for p in model.particles])
    pos = np.array([p.position for p in model.particles])
    assert velocities.shape[1] == model.dim and pos.shape[1] == model.dim
    assert np.allclose(np.linalg.norm(velocities, axis=1), model.v, rtol=1e-12, atol=1e-12)
    if use_pbc:
        assert np.all(pos >= 0) and np.all(pos < model.length)
    assert 0.0 <= model.order_parameter() <= 1.0


@pytest.mark.parametrize("use_pbc", [True, False])
def test_three_dimensional_run(use_pbc):
    """30 particles in a 10^3 box, speed 0.3, noise 0.2, 25 steps. Checks per step:
    every particle's speed stays 0.3, every position stays in [0, 10) under PBC,
    each step displaces each particle by exactly v * dt, and the order parameter
    is in [0, 1]. Without PBC the particles that cross a wall simply leave."""
    particles = initialize_random_particles(30, 10.0, 0.3, 3, seed=21)
    model = Vicsek(length=10.0, particles=particles, interaction_range=1.5, speed=0.3,
                   noise_factor=0.2, use_pbc=use_pbc, seed=1)
    assert model.dim == 3
    assert model._cell_list.n_dimensions == 3
    assert model._cell_list.n_cells == 6
    assert len(model._cell_list.cells) == 6 ** 3
    check_invariants(model, use_pbc)

    for _ in range(25):
        before = np.array([p.position for p in model.particles])
        model.step()
        after = np.array([p.position for p in model.particles])
        delta = after - before
        if use_pbc:
            delta -= 10.0 * np.round(delta / 10.0)
        assert np.allclose(np.linalg.norm(delta, axis=1), 0.3)
        check_invariants(model, use_pbc)

    if not use_pbc:
        # Some particle started within 25 * 0.3 of a wall and has left; the cell
        # list keeps only those still inside.
        inside = sum(len(c) for c in model._cell_list.cells.values())
        assert inside < len(particles)


def test_three_dimensional_zero_noise_converges():
    """All particles within one interaction range, zero noise: headings converge to
    a single 3-vector and the order parameter reaches 1."""
    particles = initialize_random_particles(25, 10.0, 0.5, 3, seed=22)
    model = Vicsek(length=10.0, particles=particles, interaction_range=20.0, speed=0.5,
                   noise_factor=0.0)
    assert model._cell_list.n_cells == 1
    assert model.order_parameter() < 0.6
    # Synchronous update with everyone in range: one step aligns everyone exactly.
    model.step()
    headings = np.array([p.velocity for p in model.particles]) / 0.5
    assert np.max(np.linalg.norm(headings - headings[0], axis=1)) < 1e-12
    assert np.isclose(model.order_parameter(), 1.0, atol=1e-12)
    for _ in range(5):
        model.step()
        check_invariants(model)
    assert np.isclose(model.order_parameter(), 1.0, atol=1e-12)


def test_three_dimensional_heterogeneous_run():
    particles = initialize_random_particles(20, 8.0, 0.2, 3, seed=23)
    for i, p in enumerate(particles):
        p.type = "A" if i % 2 else "B"
    weights = {(a, b): 1.0 for a in "AB" for b in "AB"}
    model = HeterogeneousVicsek(8.0, particles, 1.0, 0.2, 0.3, weights, weights, seed=2)
    for _ in range(10):
        model.step()
        check_invariants(model)
    order = model.get_type_specific_order()
    assert set(order) == {"A", "B"} and all(0.0 <= v <= 1.0 for v in order.values())
    corr = model.compute_cross_correlations()
    assert all(-1.0 <= v <= 1.0 for v in corr.values())


def test_four_dimensional_smoke():
    """N = 4: a short run keeps all invariants and the cell list has 3^4 neighbour offsets."""
    particles = initialize_random_particles(15, 6.0, 0.25, 4, seed=24)
    model = Vicsek(length=6.0, particles=particles, interaction_range=1.0, speed=0.25,
                   noise_factor=0.3, seed=3)
    assert model.dim == 4
    assert model._cell_list.neighbor_offsets.shape == (81, 4)
    for _ in range(10):
        model.step()
        check_invariants(model)
    evolution = model.order_parameter_evolution(steps=5)
    assert evolution.shape == (5,) and np.all((evolution >= 0) & (evolution <= 1))
