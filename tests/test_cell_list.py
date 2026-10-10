import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from matplotlib import pyplot as plt

from vicsek.models.particle import Particle
from vicsek.util.cell_list import CellList


@pytest.fixture
def particles():
    return [
        Particle(position=np.array([0.5, 0.5]), velocity=np.array([0, 0]), name="A", type="A"),
        Particle(position=np.array([0.6, 0.6]), velocity=np.array([0, 0]), name="B", type="B"),
        Particle(position=np.array([1.5, 1.5]), velocity=np.array([0, 0]), name="C", type="C"),
        Particle(position=np.array([5.0, 5.0]), velocity=np.array([0, 0]), name="D", type="D")
    ]


def brute_force_neighbors(particles, box_length, interaction_range, use_pbc):
    """O(N^2) reference: for every particle, the names of all others within the
    interaction range. With PBC distances use the minimum-image convention; without
    PBC a particle outside [0, L) has no neighbors and is nobody's neighbor (the
    CellList contract)."""
    def inside(p):
        return bool(np.all((p.position >= 0) & (p.position < box_length)))

    result = {}
    for p in particles:
        if not use_pbc and not inside(p):
            result[p.name] = []
            continue
        names = []
        for q in particles:
            if q is p:
                continue
            if not use_pbc and not inside(q):
                continue
            delta = q.position - p.position
            if use_pbc:
                delta = delta - box_length * np.round(delta / box_length)
            if np.linalg.norm(delta) <= interaction_range:
                names.append(q.name)
        result[p.name] = sorted(names)
    return result


def random_particles(rng, n, box_length, n_dimensions, spill=0.0):
    """Random particles in [0 - spill, L + spill)^d."""
    return [
        Particle(
            position=rng.uniform(-spill, box_length + spill, size=n_dimensions),
            velocity=np.zeros(n_dimensions),
            name=f"p{i}",
            type="t",
        )
        for i in range(n)
    ]


def assert_matches_brute_force(particles, box_length, interaction_range, n_dimensions, use_pbc):
    cell_list = CellList(particles, box_length, interaction_range,
                         n_dimensions=n_dimensions, use_pbc=use_pbc)
    cell_list.build()
    reference = brute_force_neighbors(particles, box_length, interaction_range, use_pbc)
    for i, p in enumerate(particles):
        found = [q.name for q in cell_list.get_neighbors(p)]
        assert len(found) == len(set(found)), f"duplicate neighbors for {p.name}: {found}"
        assert sorted(found) == reference[p.name]
        # Lookup by index must agree with lookup by object.
        assert [q.name for q in cell_list.get_neighbors(i)] == found


def test_initialize(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    assert len(cell_list.particles) == 4
    assert cell_list.box_length == 10.0
    assert cell_list.interaction_range == 1.0
    assert cell_list.n_cells == 10
    assert cell_list.cell_size == 1.0
    assert cell_list.cells == {}  # nothing is binned until build()


@pytest.mark.parametrize("box_length,interaction_range,expected_cells,expected_size", [
    (10.0, 1.0, 10, 1.0),
    (10.0, 3.0, 3, 10.0 / 3),   # floor(10 / 3) cells, each wider than the range
    (1.5, 1.0, 1, 1.5),         # box smaller than two ranges collapses to one cell
    (0.5, 1.0, 1, 0.5),         # box smaller than the range still gives one cell
])
def test_cell_geometry(particles, box_length, interaction_range, expected_cells, expected_size):
    cell_list = CellList(particles, box_length, interaction_range)
    assert cell_list.n_cells == expected_cells
    assert np.isclose(cell_list.cell_size, expected_size)
    # The cell must never be narrower than the range, or neighbors could be missed.
    assert cell_list.cell_size >= interaction_range or cell_list.n_cells == 1


def test_build(particles):
    # cell_size == 1, so a particle at x lands in cell floor(x): A and B share
    # cell (0, 0), C is in (1, 1) and D in (5, 5).
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    assert len(cell_list.cells) == 100
    assert [p.name for p in cell_list.cells[(0, 0)]] == ["A", "B"]
    assert [p.name for p in cell_list.cells[(1, 1)]] == ["C"]
    assert [p.name for p in cell_list.cells[(5, 5)]] == ["D"]
    assert sum(len(c) for c in cell_list.cells.values()) == len(particles)


def test_cell_index_axis_order():
    # _hash_position stores indices in reversed axis order: a particle at
    # (x, y) = (7.5, 2.5) is in cells[(2, 7)]. Documented, not endorsed.
    p = Particle(position=np.array([7.5, 2.5]), velocity=np.zeros(2), name="p", type="t")
    cell_list = CellList([p], box_length=10.0, interaction_range=1.0)
    cell_list.build()
    assert cell_list.cells[(2, 7)] == [p]
    assert cell_list._hash_position(np.array([7.5, 2.5])) == (2, 7)


def test_update(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    particles[0].position = np.array([1.5, 1.5])
    cell_list.update()
    assert [p.name for p in cell_list.cells[(0, 0)]] == ["B"]
    assert sorted(p.name for p in cell_list.cells[(1, 1)]) == ["A", "C"]
    assert sum(len(c) for c in cell_list.cells.values()) == len(particles)


def test_neighbors_pbc(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0, use_pbc=True)
    cell_list.build()
    neighbors = cell_list.get_neighbors(particles[0])
    assert [n.name for n in neighbors] == ["B"]


def test_neighbors_pbc_across_boundary():
    # 0.1 and 9.9 are 0.2 apart through the periodic wall, 9.8 apart directly.
    a = Particle(position=np.array([0.1, 5.0]), velocity=np.zeros(2), name="a", type="t")
    b = Particle(position=np.array([9.9, 5.0]), velocity=np.zeros(2), name="b", type="t")
    for use_pbc, expected in [(True, ["b"]), (False, [])]:
        cell_list = CellList([a, b], box_length=10.0, interaction_range=1.0, use_pbc=use_pbc)
        cell_list.build()
        assert [n.name for n in cell_list.get_neighbors(a)] == expected
    assert np.isclose(cell_list._minimum_image_distance(a, b), 0.2)


def test_neighbors_no_pbc(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0, use_pbc=False)
    cell_list.build()
    neighbors = cell_list.get_neighbors(particles[0])
    assert [n.name for n in neighbors] == ["B"]

    # A particle that leaves the box is dropped from the cells and has no neighbors,
    # and is no longer a neighbor of anyone.
    particles[0].position = np.array([-0.5, -0.5])
    cell_list.update()
    assert sum(len(c) for c in cell_list.cells.values()) == len(particles) - 1
    assert cell_list.get_neighbors(particles[0]) == []
    assert cell_list.get_neighbors(particles[1]) == []


def test_build_skips_out_of_box_without_pbc():
    inside = Particle(position=np.array([5.0, 5.0]), velocity=np.zeros(2), name="in", type="t")
    outside = Particle(position=np.array([10.5, 5.0]), velocity=np.zeros(2), name="out", type="t")
    cell_list = CellList([inside, outside], box_length=10.0, interaction_range=1.0, use_pbc=False)
    cell_list.build()
    assert sum(len(c) for c in cell_list.cells.values()) == 1
    # With PBC the same particle is wrapped into the box instead of dropped.
    cell_list = CellList([inside, outside], box_length=10.0, interaction_range=1.0, use_pbc=True)
    cell_list.build()
    assert sum(len(c) for c in cell_list.cells.values()) == 2


def test_non_pbc_offsets_single_cell():
    # With one cell per axis and no PBC the only valid offset is the zero offset.
    p = Particle(position=np.array([0.2, 0.2]), velocity=np.zeros(2), name="p", type="t")
    cell_list = CellList([p], box_length=1.0, interaction_range=1.0, use_pbc=False)
    assert cell_list.neighbor_offsets.tolist() == [[0, 0]]
    cell_list = CellList([p], box_length=10.0, interaction_range=1.0, use_pbc=False)
    assert len(cell_list.neighbor_offsets) == 9


@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("use_pbc", [True, False])
@pytest.mark.parametrize("box_length,interaction_range", [
    (10.0, 1.0),   # 10 cells per axis
    (10.0, 3.0),   # 3 cells per axis: the smallest grid where offsets do not wrap onto each other
    (2.5, 1.0),    # 2 cells per axis: offsets -1 and +1 wrap onto the same cell under PBC
    (1.5, 1.0),    # 1 cell per axis
])
def test_neighbors_match_brute_force(rng, n_dimensions, use_pbc, box_length, interaction_range):
    n = 60 if n_dimensions == 2 else 80
    # Without PBC also scatter a few particles just outside the box.
    spill = 0.0 if use_pbc else 0.3 * interaction_range
    particles = random_particles(rng, n, box_length, n_dimensions, spill=spill)
    assert_matches_brute_force(particles, box_length, interaction_range, n_dimensions, use_pbc)


@settings(derandomize=True, max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_dimensions=st.sampled_from([2, 3]),
    use_pbc=st.booleans(),
    n_cells=st.integers(min_value=1, max_value=6),
    n_particles=st.integers(min_value=2, max_value=40),
)
def test_neighbors_match_brute_force_property(seed, n_dimensions, use_pbc, n_cells, n_particles):
    """Property version: random box sizes from 1 to 6 interaction ranges."""
    rng = np.random.default_rng(seed)
    interaction_range = 1.0
    box_length = interaction_range * (n_cells + rng.uniform(0.0, 0.99))
    particles = random_particles(rng, n_particles, box_length, n_dimensions)
    assert_matches_brute_force(particles, box_length, interaction_range, n_dimensions, use_pbc)


def test_neighbors_after_update_match_brute_force(rng):
    """Moving particles and calling update() must give the same answer as a fresh build()."""
    particles = random_particles(rng, 50, 10.0, 2)
    cell_list = CellList(particles, 10.0, 1.0, use_pbc=True)
    cell_list.build()
    for p in particles:
        p.position = (p.position + rng.uniform(-2, 2, size=2)) % 10.0
    cell_list.update()
    reference = brute_force_neighbors(particles, 10.0, 1.0, True)
    for p in particles:
        assert sorted(q.name for q in cell_list.get_neighbors(p)) == reference[p.name]


def test_visualize(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    ax = cell_list.visualize(show_cell_grid=True, label_cells=True, label_particles=True)
    assert ax.has_data()
    # 11 vertical + 11 horizontal grid lines for 10 cells per axis
    assert len(ax.lines) == 2 * (cell_list.n_cells + 1)
    # one scatter collection per particle
    assert len(ax.collections) == len(particles)
    # one annotation per cell plus one per particle
    assert len(ax.texts) == cell_list.n_cells ** 2 + len(particles)
    assert ax.get_xlim() == (0.0, 10.0) and ax.get_ylim() == (0.0, 10.0)
    assert len(plt.get_fignums()) == 1


def test_visualize_on_existing_axes_without_grid(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    fig, ax = plt.subplots()
    result = cell_list.visualize(ax=ax, show_cell_grid=False)
    assert result is ax
    assert len(ax.lines) == 0
    assert len(ax.texts) == 0
    assert len(ax.collections) == len(particles)
    assert len(plt.get_fignums()) == 1  # no extra figure was created


# --- vectorised routine: neighbor_pairs / mean_neighbor_velocity ---------------

def pairs_from_get_neighbors(cell_list, particles):
    """Reference: the ordered (i, j) pairs that per-particle get_neighbors returns."""
    index_of = {id(p): k for k, p in enumerate(particles)}
    pairs = set()
    for i, p in enumerate(particles):
        for q in cell_list.get_neighbors(p):
            pairs.add((i, index_of[id(q)]))
    return pairs


@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("use_pbc", [True, False])
@pytest.mark.parametrize("box_length,interaction_range", [
    (10.0, 1.0),   # 10 cells per axis
    (10.0, 3.0),   # 3 cells per axis
    (2.5, 1.0),    # 2 cells per axis: offsets -1 and +1 wrap onto the same cell under PBC
    (1.5, 1.0),    # 1 cell per axis
])
def test_neighbor_pairs_match_get_neighbors(rng, n_dimensions, use_pbc, box_length, interaction_range):
    """The vectorised neighbour search returns exactly the pairs the per-particle
    search does, with the same distances, for random configurations in 2-D and
    3-D, periodic and open (with particles spilled outside the open box)."""
    n = 120 if n_dimensions == 2 else 160
    spill = 0.0 if use_pbc else 0.3 * interaction_range
    particles = random_particles(rng, n, box_length, n_dimensions, spill=spill)
    cell_list = CellList(particles, box_length, interaction_range,
                         n_dimensions=n_dimensions, use_pbc=use_pbc)
    cell_list.build()
    positions = np.array([p.position for p in particles])

    i, j, dist = cell_list.neighbor_pairs(positions)
    assert i.shape == j.shape == dist.shape
    assert len(set(zip(i.tolist(), j.tolist()))) == len(i)          # no duplicate pairs
    assert set(zip(i.tolist(), j.tolist())) == pairs_from_get_neighbors(cell_list, particles)
    assert np.all(np.diff(i) >= 0)                                  # grouped by focal particle
    # Distances agree with the per-pair definition.
    delta = positions[j] - positions[i]
    if use_pbc:
        delta -= box_length * np.round(delta / box_length)
    assert np.allclose(dist, np.linalg.norm(delta, axis=1), rtol=0, atol=1e-12)
    assert np.all(dist <= interaction_range)
    # Default argument reads the particles' current positions.
    i2, j2, _ = cell_list.neighbor_pairs()
    assert np.array_equal(i, i2) and np.array_equal(j, j2)


@settings(derandomize=True, max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_dimensions=st.sampled_from([2, 3]),
    use_pbc=st.booleans(),
    n_cells=st.integers(min_value=1, max_value=6),
    n_particles=st.integers(min_value=1, max_value=40),
)
def test_neighbor_pairs_match_get_neighbors_property(seed, n_dimensions, use_pbc, n_cells, n_particles):
    rng = np.random.default_rng(seed)
    box_length = n_cells + rng.uniform(0.0, 0.99)
    spill = 0.0 if use_pbc else 0.4
    particles = random_particles(rng, n_particles, box_length, n_dimensions, spill=spill)
    cell_list = CellList(particles, box_length, 1.0, n_dimensions=n_dimensions, use_pbc=use_pbc)
    cell_list.build()
    i, j, _ = cell_list.neighbor_pairs()
    assert set(zip(i.tolist(), j.tolist())) == pairs_from_get_neighbors(cell_list, particles)


@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("use_pbc", [True, False])
def test_mean_neighbor_velocity_matches_per_particle_mean(rng, n_dimensions, use_pbc):
    """mean[i] is the plain average of particle i's velocity and its neighbours';
    a particle with no neighbours (or outside an open box) keeps its own velocity."""
    n = 150
    particles = random_particles(rng, n, 8.0, n_dimensions, spill=0.0 if use_pbc else 0.5)
    for p in particles:
        p.velocity = rng.normal(size=n_dimensions)
    cell_list = CellList(particles, 8.0, 1.0, n_dimensions=n_dimensions, use_pbc=use_pbc)
    cell_list.build()
    positions = np.array([p.position for p in particles])
    velocities = np.array([p.velocity for p in particles])

    mean, counts = cell_list.mean_neighbor_velocity(positions, velocities)
    assert mean.shape == (n, n_dimensions) and counts.shape == (n,)
    for i, p in enumerate(particles):
        neighbors = cell_list.get_neighbors(p)
        assert counts[i] == len(neighbors)
        expected = np.mean([p.velocity] + [q.velocity for q in neighbors], axis=0)
        assert np.allclose(mean[i], expected, rtol=0, atol=1e-13)
    assert counts.max() > 0                       # the configuration is not trivial


def test_neighbor_pairs_everyone_outside_open_box():
    particles = random_particles(np.random.default_rng(0), 5, 1.0, 2)
    for p in particles:
        p.position = p.position + 10.0           # all outside [0, 1)
    cell_list = CellList(particles, 1.0, 1.0, use_pbc=False)
    i, j, dist = cell_list.neighbor_pairs()
    assert i.size == j.size == dist.size == 0
    mean, counts = cell_list.mean_neighbor_velocity(
        np.array([p.position for p in particles]), np.ones((5, 2)))
    assert np.array_equal(mean, np.ones((5, 2))) and np.all(counts == 0)


def test_cells_are_rebuilt_lazily_after_invalidate(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    assert [p.name for p in cell_list.cells[(0, 0)]] == ["A", "B"]
    particles[0].position = np.array([1.5, 1.5])
    cell_list.invalidate()                        # nothing is rebuilt yet ...
    assert [p.name for p in cell_list._cells[(0, 0)]] == ["A", "B"]
    assert [p.name for p in cell_list.cells[(0, 0)]] == ["B"]   # ... until cells is read
    assert sorted(p.name for p in cell_list.cells[(1, 1)]) == ["A", "C"]
    particles[0].position = np.array([0.5, 0.5])
    cell_list.invalidate()
    assert [n.name for n in cell_list.get_neighbors(particles[1])] == ["A"]  # get_neighbors rebuilds too
