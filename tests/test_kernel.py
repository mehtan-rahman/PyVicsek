"""The C++ kernel (vicsek._kernel) against the numpy cell list, and the switch
between them (vicsek.util.backend).

The kernel performs the same floating-point operations in the same order as
the numpy path, so every comparison here is exact (assert_array_equal), not
approximate: same pairs in the same order, same distances, same means, and
identical trajectories over many seeded steps.
"""
import importlib
import sys

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import vicsek
from vicsek import HeterogeneousVicsek, Vicsek, initialize_random_particles
from vicsek.lyapunov.kernelized_vicsek import KernelizedVicsek
from vicsek.util import backend
from vicsek.util.cell_list import CellList, neighbor_pairs

requires_kernel = pytest.mark.skipif(not backend.kernel_available(), reason="vicsek._kernel not built")


def pairs_both(positions, box_length, radius, use_pbc):
    with vicsek.use_kernel(False):
        reference = neighbor_pairs(positions, box_length, radius, use_pbc)
    with vicsek.use_kernel(True):
        fast = neighbor_pairs(positions, box_length, radius, use_pbc)
    return reference, fast


def mean_both(positions, velocities, box_length, radius, use_pbc, threads=1):
    cell_list = CellList([], box_length, radius, positions.shape[1], use_pbc)
    with vicsek.use_kernel(False):
        reference = cell_list.mean_neighbor_velocity(positions, velocities)
    with vicsek.use_kernel(True, threads=threads):
        fast = cell_list.mean_neighbor_velocity(positions, velocities)
    return reference, fast


def assert_same(reference, fast):
    assert len(reference) == len(fast)
    for a, b in zip(reference, fast):
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b)


def check_configuration(positions, box_length, radius, use_pbc, rng):
    velocities = rng.normal(size=positions.shape)
    reference, fast = pairs_both(positions, box_length, radius, use_pbc)
    assert fast[0].dtype == np.int64 and fast[1].dtype == np.int64 and fast[2].dtype == np.float64
    assert_same(reference, fast)
    assert_same(*mean_both(positions, velocities, box_length, radius, use_pbc))


# ---------------------------------------------------------------- equivalence

@requires_kernel
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("use_pbc", [True, False])
@pytest.mark.parametrize("box_length, radius", [
    (10.0, 1.0),     # 10 cells per axis
    (7.3, 1.0),      # cell wider than the radius (1.043)
    (3.0, 1.0),      # 3 cells: smallest grid with a full stencil
    (2.5, 1.0),      # 2 cells: -1 and +1 wrap onto the same cell under PBC
    (1.5, 1.0),      # 1 cell, radius above L / 2: the minimum image decides
    (2.0, 3.0),      # radius beyond the box: one cell, every pair within L sqrt(dim) / 2
    (12.0, 2.7),     # radius not dividing the box
])
def test_random_configurations_match(dim, use_pbc, box_length, radius, rng):
    n = 400 if dim == 2 else 600
    spill = 0.0 if use_pbc else 0.2 * box_length   # open box: some particles outside
    positions = rng.uniform(-spill, box_length + spill, size=(n, dim))
    check_configuration(positions, box_length, radius, use_pbc, rng)


@requires_kernel
@pytest.mark.parametrize("dim", [1, 4])
@pytest.mark.parametrize("use_pbc", [True, False])
def test_other_dimensions_match(dim, use_pbc, rng):
    box_length = 6.0 if dim == 4 else 30.0
    positions = rng.uniform(0, box_length, size=(500, dim))
    check_configuration(positions, box_length, 1.3, use_pbc, rng)


@requires_kernel
@pytest.mark.parametrize("use_pbc", [True, False])
def test_boundary_positions_match(use_pbc, rng):
    """Particles exactly on cell edges, on 0 and L, just inside L, outside the
    box (negative and beyond L), and pairs at exactly the interaction range,
    directly and across the periodic boundary."""
    box_length, radius = 4.0, 1.0
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nextafter(4.0, 0), -1e-17, -0.5, 4.5, 1e-300])
    grid = np.array(np.meshgrid(edges, edges)).reshape(2, -1).T
    exact_range = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5],   # distance exactly 1
                            [0.0, 2.0], [3.0, 2.0]])              # 3 apart, 1 by minimum image
    positions = np.vstack([grid, exact_range, rng.uniform(0, box_length, (50, 2))])
    reference, fast = pairs_both(positions, box_length, radius, use_pbc)
    assert_same(reference, fast)
    pairs = set(zip(fast[0].tolist(), fast[1].tolist()))
    base = len(grid)
    assert (base, base + 1) in pairs and (base, base + 2) in pairs
    assert ((base + 3, base + 4) in pairs) == use_pbc
    check_configuration(positions, box_length, radius, use_pbc, rng)


@requires_kernel
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("use_pbc", [True, False])
def test_empty_and_single_particle(dim, use_pbc, rng):
    for n in (0, 1):
        positions = rng.uniform(0, 5.0, size=(n, dim))
        check_configuration(positions, 5.0, 1.0, use_pbc, rng)
    i, j, d = vicsek._kernel.neighbor_pairs(np.empty((0, dim)), 5.0, 1.0)
    assert i.shape == j.shape == d.shape == (0,)
    mean, count = vicsek._kernel.mean_neighbor_velocity(np.empty((0, dim)), np.empty((0, dim)), 5.0, 1.0)
    assert mean.shape == (0, dim) and count.shape == (0,)
    one = np.array([[1.0] * dim])
    mean, count = vicsek._kernel.mean_neighbor_velocity(one, 2 * one, 5.0, 1.0)
    np.testing.assert_array_equal(mean, 2 * one)
    np.testing.assert_array_equal(count, [0])


@requires_kernel
@settings(derandomize=True, max_examples=60, deadline=None)
@given(
    n=st.integers(0, 80),
    dim=st.integers(1, 3),
    box_length=st.floats(0.5, 15.0),
    radius=st.floats(0.2, 4.0),
    use_pbc=st.booleans(),
    snap=st.booleans(),
    seed=st.integers(0, 2 ** 32 - 1),
)
def test_hypothesis_configurations_match(n, dim, box_length, radius, use_pbc, snap, seed):
    rng = np.random.default_rng(seed)
    spill = 0.0 if use_pbc else 0.25 * box_length
    positions = rng.uniform(-spill, box_length + spill, size=(n, dim))
    if snap:  # put coordinates on a lattice of half the radius: ties and cell edges
        positions = np.round(positions / (radius / 2)) * (radius / 2)
    check_configuration(positions, box_length, radius, use_pbc, rng)


@requires_kernel
@pytest.mark.parametrize("dim, box_length", [(2, 35.0), (3, 10.0)])
def test_threads_do_not_change_the_result(dim, box_length, rng):
    positions = rng.uniform(0, box_length, size=(5000, dim))
    velocities = rng.normal(size=positions.shape)
    serial = vicsek._kernel.mean_neighbor_velocity(positions, velocities, box_length, 1.0, True, 1)
    serial_pairs = vicsek._kernel.neighbor_pairs(positions, box_length, 1.0, True, 1)
    for threads in (2, 3, 8):
        assert_same(serial, vicsek._kernel.mean_neighbor_velocity(positions, velocities, box_length, 1.0, True, threads))
        assert_same(serial_pairs, vicsek._kernel.neighbor_pairs(positions, box_length, 1.0, True, threads))
    assert_same(*mean_both(positions, velocities, box_length, 1.0, True, threads=4))


@requires_kernel
def test_kernel_converts_inputs(rng):
    """Integer, float32 and non-contiguous inputs give the numpy path's answer."""
    positions = rng.uniform(0, 10, size=(300, 4))[:, ::2]      # non-contiguous view
    velocities = rng.integers(-3, 4, size=(300, 2))
    assert_same(*mean_both(positions, velocities, 10, 1, True))
    assert_same(*pairs_both(positions.astype(np.float32).astype(float), 10, 1, True))


@requires_kernel
def test_kernel_rejects_bad_input():
    k = vicsek._kernel
    p = np.zeros((3, 2))
    with pytest.raises(ValueError, match="same shape"):
        k.mean_neighbor_velocity(p, np.zeros((3, 3)), 5.0, 1.0)
    with pytest.raises(ValueError, match="n_threads"):
        k.neighbor_pairs(p, 5.0, 1.0, True, 0)
    with pytest.raises(ValueError, match="n_threads"):
        k.mean_neighbor_velocity(p, p, 5.0, 1.0, True, 0)
    with pytest.raises(ValueError, match="radius"):
        k.neighbor_pairs(p, 5.0, 0.0)
    with pytest.raises(ValueError, match="box_length"):
        k.neighbor_pairs(p, -1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        k.neighbor_pairs(np.array([[0.0, np.nan]]), 5.0, 1.0)
    with pytest.raises(ValueError, match="outside"):
        k.neighbor_pairs(np.array([[0.0, 1e300]]), 5.0, 1.0)
    with pytest.raises(ValueError, match="too many cells"):
        k.neighbor_pairs(p, 1e6, 1e-3)
    with pytest.raises(ValueError, match="at least one column"):
        k.neighbor_pairs(np.zeros((3, 0)), 5.0, 1.0)
    with pytest.raises(TypeError):
        k.neighbor_pairs(np.zeros(3), 5.0, 1.0)      # 1-D positions


# ------------------------------------------------------ trajectory equivalence

def _trajectory(make_model, steps, use):
    with vicsek.use_kernel(use):
        model = make_model()
        history = []
        for _ in range(steps):
            model.step()
            history.append((model.positions.copy(), model.velocities.copy()))
    return history


def _assert_same_trajectory(make_model, steps=25):
    reference = _trajectory(make_model, steps, False)
    fast = _trajectory(make_model, steps, True)
    for (p0, v0), (p1, v1) in zip(reference, fast):
        np.testing.assert_array_equal(p0, p1)
        np.testing.assert_array_equal(v0, v1)


@requires_kernel
@pytest.mark.parametrize("dim, n, box_length", [(2, 800, 14.0), (3, 600, 5.3), (2, 60, 2.5)])
@pytest.mark.parametrize("use_pbc", [True, False])
def test_vicsek_trajectory_matches(dim, n, box_length, use_pbc):
    def make():
        particles = initialize_random_particles(n, box_length, 0.3, dim, seed=7)
        return Vicsek(box_length, particles, 1.0, 0.3, 0.4, use_pbc=use_pbc, seed=11)
    _assert_same_trajectory(make)


@requires_kernel
def test_vicsek_trajectory_matches_with_global_rng():
    def make():
        np.random.seed(3)
        return Vicsek(10.0, initialize_random_particles(500, 10.0, 0.5, 2, seed=1), 1.0, 0.5, 0.6)
    _assert_same_trajectory(make)


@requires_kernel
@pytest.mark.parametrize("dim", [2, 3])
def test_heterogeneous_trajectory_matches(dim):
    def make():
        particles = (initialize_random_particles(300, 6.0, 0.2, dim, "A", seed=2)
                     + initialize_random_particles(300, 6.0, 0.2, dim, "B", seed=3))
        alignment = {("A", "A"): 1.0, ("A", "B"): -0.5, ("B", "A"): 0.3, ("B", "B"): 0.8}
        noise = {("A", "A"): 1.0, ("A", "B"): 1.5, ("B", "A"): 0.7, ("B", "B"): 1.2}
        return HeterogeneousVicsek(6.0, particles, 1.0, 0.2, 0.3, alignment, noise, seed=5)
    _assert_same_trajectory(make)


@requires_kernel
@pytest.mark.parametrize("use_pbc", [True, False])
def test_kernelized_trajectory_matches(use_pbc):
    def make():
        particles = initialize_random_particles(500, 9.0, 0.3, 2, seed=4)
        return KernelizedVicsek(9.0, particles, 1.0, 0.3, 0.4, epsilon=0.05, use_pbc=use_pbc, seed=6)
    _assert_same_trajectory(make)


@requires_kernel
def test_model_uses_kernel_when_enabled(monkeypatch):
    """The dispatch really reaches the kernel: a model step calls it once, and
    not at all with the kernel switched off."""
    calls = []
    real = vicsek._kernel.mean_neighbor_velocity

    class Spy:
        neighbor_pairs = staticmethod(vicsek._kernel.neighbor_pairs)

        @staticmethod
        def mean_neighbor_velocity(*args):
            calls.append(args[-1])
            return real(*args)

    monkeypatch.setattr(backend, "_kernel", Spy)
    model = Vicsek(10.0, initialize_random_particles(100, 10.0, 0.5, 2, seed=1), 1.0, 0.5, 0.2, seed=1)
    with vicsek.use_kernel(True, threads=2):
        model.step()
    assert calls == [2]
    with vicsek.use_kernel(False):
        model.step()
    assert calls == [2]


# ----------------------------------------------------------------- the switch

def test_use_kernel_switch_and_context(monkeypatch):
    monkeypatch.setattr(backend, "_kernel", object())   # pretend it is built
    monkeypatch.setattr(backend, "_enabled", True)
    monkeypatch.setattr(backend, "_threads", 1)
    vicsek.use_kernel(False)
    assert not vicsek.kernel_enabled() and backend.kernel() is None
    with vicsek.use_kernel(True, threads=3) as setting:
        assert isinstance(setting, vicsek.use_kernel)
        assert vicsek.kernel_enabled() and vicsek.kernel_threads() == 3
        assert backend.kernel() is backend._kernel
    assert not vicsek.kernel_enabled() and vicsek.kernel_threads() == 1
    vicsek.use_kernel(True)
    assert vicsek.kernel_enabled() and vicsek.kernel_threads() == 1
    with pytest.raises(ValueError, match="threads"):
        vicsek.use_kernel(True, threads=0)
    assert vicsek.kernel_enabled()


def test_use_kernel_without_kernel(monkeypatch):
    monkeypatch.setattr(backend, "_kernel", None)
    monkeypatch.setattr(backend, "_enabled", False)
    assert not vicsek.kernel_available()
    with pytest.raises(RuntimeError, match="not available"):
        vicsek.use_kernel(True)
    with vicsek.use_kernel(False):       # always allowed
        assert backend.kernel() is None


@pytest.fixture
def reload_backend():
    """Reload vicsek.util.backend after the test (and after monkeypatch has
    restored the environment), so later tests see the normal module state.
    Reloading updates the module object in place, which cell_list refers to."""
    yield
    importlib.reload(backend)


def test_environment_variable_disables_kernel(monkeypatch, reload_backend):
    with monkeypatch.context() as m:
        m.setenv("VICSEK_NO_KERNEL", "1")
        importlib.reload(backend)
        assert not backend.kernel_enabled() and backend.kernel() is None
        m.setenv("VICSEK_NO_KERNEL", "0")
        importlib.reload(backend)
        assert backend.kernel_enabled() == backend.kernel_available()


def test_missing_extension_falls_back_to_numpy(monkeypatch, reload_backend, rng):
    """Without a built vicsek._kernel the import fails quietly and numpy is used."""
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "vicsek._kernel", None)   # makes the import raise ImportError
        m.delattr(vicsek, "_kernel", raising=False)
        importlib.reload(backend)
        assert not backend.kernel_available() and not backend.kernel_enabled()
        positions = rng.uniform(0, 5, size=(50, 2))
        i, j, d = neighbor_pairs(positions, 5.0, 1.0)
        assert i.dtype == np.intp and np.all(d <= 1.0)
