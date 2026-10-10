from typing import Dict, Tuple, List, Union, Optional
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from vicsek.models.particle import Particle


def _grid(box_length: float, interaction_range: float) -> Tuple[int, float]:
    """Cells per axis and cell width: as many cells as fit, each at least one range wide."""
    n_cells = max(1, int(np.floor(box_length / interaction_range)))
    return n_cells, box_length / n_cells


def _cell_offsets(n_dimensions: int, n_cells: int, use_pbc: bool) -> NDArray:
    """The distinct offsets to a cell's neighbours, shape (k, dim).

    Normally the 3^dim offsets in {-1, 0, 1}. Under periodic boundaries with
    one or two cells per axis, -1 and +1 (or all three) wrap onto the same
    cell, so only the distinct offsets modulo n_cells are kept and no cell is
    visited twice. (Without periodic boundaries out-of-grid cells are dropped
    by the caller instead.)
    """
    axis = np.array([-1, 0, 1])
    if use_pbc and n_cells < 3:
        axis = np.unique(axis % n_cells)
    return np.array(np.meshgrid(*[axis] * n_dimensions, indexing="ij")).reshape(n_dimensions, -1).T


def _bin_particles(positions: NDArray, n_cells: int, cell_size: float, use_pbc: bool
                   ) -> Tuple[NDArray, NDArray, NDArray]:
    """Sort particles into cells: (cell index per particle (N, dim), particle
    order sorted by flat cell id, cell start offsets (n_cells^dim + 1,))."""
    dim = positions.shape[1]
    cells = (positions / cell_size).astype(np.int64)      # truncation, as in _hash_position
    if use_pbc:
        cells %= n_cells
    else:
        np.clip(cells, 0, n_cells - 1, out=cells)
    strides = n_cells ** np.arange(dim - 1, -1, -1)
    flat = cells @ strides
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n_cells ** dim)
    starts = np.concatenate(([0], np.cumsum(counts)))
    return cells, order, starts


def _candidate_pairs(cells: NDArray, order: NDArray, starts: NDArray, n_cells: int, use_pbc: bool
                     ) -> Tuple[NDArray, NDArray]:
    """Every (focal, candidate) index pair with the candidate in one of the
    focal particle's neighbouring cells (the focal particle itself included)."""
    n, dim = cells.shape
    strides = n_cells ** np.arange(dim - 1, -1, -1)
    offsets = _cell_offsets(dim, n_cells, use_pbc)                     # (k, dim)
    nb_cells = cells[:, None, :] + offsets[None, :, :]                 # (N, k, dim)
    if use_pbc:
        nb_cells %= n_cells
        nb_flat = (nb_cells @ strides).ravel()                         # (N k,)
        owner = np.repeat(np.arange(n), len(offsets))
    else:
        valid = np.all((nb_cells >= 0) & (nb_cells < n_cells), axis=2).ravel()
        nb_flat = (nb_cells @ strides).ravel()[valid]
        owner = np.repeat(np.arange(n), len(offsets))[valid]
    seg_start = starts[nb_flat]
    seg_len = starts[nb_flat + 1] - seg_start
    focal = np.repeat(owner, seg_len)
    within = np.arange(int(seg_len.sum())) - np.repeat(np.cumsum(seg_len) - seg_len, seg_len)
    candidate = order[np.repeat(seg_start, seg_len) + within]
    return focal, candidate


def _pair_distances(positions: NDArray, focal: NDArray, candidate: NDArray, box_length: float,
                    use_pbc: bool) -> NDArray:
    """Distance of every candidate pair, minimum image under periodic boundaries."""
    delta = positions[candidate] - positions[focal]
    if use_pbc:
        delta -= box_length * np.rint(delta / box_length)
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))


def neighbor_pairs(positions: NDArray, box_length: float, interaction_range: float,
                   use_pbc: bool = True) -> Tuple[NDArray, NDArray, NDArray]:
    """All ordered neighbour pairs of a configuration, with no per-particle loop.

    Returns ``(i, j, distance)``: three arrays over every ordered pair with
    ``i != j`` and ``distance <= interaction_range``, where ``distance`` is the
    minimum-image distance under periodic boundaries and the plain Euclidean
    distance otherwise. Pairs are grouped by ``i`` in increasing order. Without
    periodic boundaries, particles outside ``[0, box_length)`` on any axis have
    no neighbours and are nobody's neighbour, exactly as ``CellList.get_neighbors``.

    The algorithm is the cell list of ``CellList`` in array form: hash each
    particle to a cell, sort by cell, expand each particle's 3^dim neighbouring
    cells into candidate pairs with ``np.repeat``, then keep the pairs within
    range. Cost is O(N) in time and memory (about 3^dim times the mean cell
    occupancy in candidates per particle).
    """
    positions = np.asarray(positions, dtype=float)
    n, dim = positions.shape
    n_cells, cell_size = _grid(box_length, interaction_range)

    if use_pbc:
        subset = None
    else:
        subset = np.flatnonzero(np.all((positions >= 0) & (positions < box_length), axis=1))
        positions = positions[subset]
        if positions.shape[0] == 0:
            empty = np.empty(0, dtype=np.intp)
            return empty, empty, np.empty(0)

    cells, order, starts = _bin_particles(positions, n_cells, cell_size, use_pbc)
    focal, candidate = _candidate_pairs(cells, order, starts, n_cells, use_pbc)
    distance = _pair_distances(positions, focal, candidate, box_length, use_pbc)
    keep = (distance <= interaction_range) & (focal != candidate)
    focal, candidate, distance = focal[keep], candidate[keep], distance[keep]
    if subset is not None:
        focal, candidate = subset[focal], subset[candidate]
    return focal, candidate, distance


class CellList:
    """
    Cell list data structure for efficient neighbor finding.

    Divides a simulation box into cells and sorts particles into cells based
    on their position. Allows efficient retrieval of neighboring particles within
    a certain interaction range.

    Attributes:
        particles (List[Particle]): List of particles.
        box_length (float): Length of the simulation box.
        interaction_range (float): Maximum interaction distance between particles.
        n_dimensions (int): Number of spatial dimensions (default 2).
        use_pbc (bool): Whether to use periodic boundary conditions (default True).
        n_cells (int): Number of cells along each dimension.
        cell_size (float): Size of each cell.
        neighbor_offsets (NDArray): Offsets to neighboring cells.
        cells (Dict[Tuple[int, ...], List[Particle]]): Dictionary mapping cell indices to lists of particles.

    Two ways to query it. ``get_neighbors`` answers for one particle from the
    ``cells`` dictionary (built by ``build``/``update``). ``neighbor_pairs`` and
    ``mean_neighbor_velocity`` answer for every particle at once from arrays of
    positions, with no per-particle Python loop and without the dictionary;
    the models' ``step`` uses these. After the models move particles they call
    ``invalidate`` rather than ``update``, and the dictionary is rebuilt on its
    next access (``cells`` or ``get_neighbors``).
    """

    __slots__ = ('particles', 'box_length', 'interaction_range', 'n_dimensions',
                 'use_pbc', 'n_cells', 'cell_size', 'neighbor_offsets', '_cells', '_stale')

    def __init__(
            self,
            particles: List[Particle],
            box_length: float,
            interaction_range: float,
            n_dimensions: int = 2,
            use_pbc: bool = True
    ):
        """
        Initialize a CellList.

        Args:
            particles: List of particles.
            box_length: Length of the simulation box.
            interaction_range: Maximum interaction distance between particles.
            n_dimensions: Number of spatial dimensions (default 2).
            use_pbc: Whether to use periodic boundary conditions (default True).
        """
        self.particles = particles
        self.box_length = box_length
        self.interaction_range = interaction_range
        self.n_dimensions = n_dimensions
        self.use_pbc = use_pbc

        self.n_cells, self.cell_size = _grid(box_length, interaction_range)

        if self.use_pbc:
            self.neighbor_offsets = np.array(
                np.meshgrid(*[[-1, 0, 1]] * self.n_dimensions)
            ).T.reshape(-1, self.n_dimensions)
        else:
            self._compute_non_pbc_neighbor_offsets()

        self._cells: Dict[Tuple[int, ...], List[Particle]] = {}
        self._stale = False

    @property
    def cells(self) -> Dict[Tuple[int, ...], List[Particle]]:
        """Cell index tuple -> particles in that cell. Empty until ``build``;
        rebuilt on access after ``invalidate``."""
        if self._stale:
            self.update()
        return self._cells

    def _compute_non_pbc_neighbor_offsets(self):
        """Compute neighbor offsets for non-periodic boundary conditions."""
        base_offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * self.n_dimensions)).T.reshape(-1, self.n_dimensions)
        valid_offsets = []

        for offset in base_offsets:
            def is_valid_offset(idx, offset):
                return 0 <= idx + offset < self.n_cells

            for cell_idx in np.ndindex((self.n_cells,) * self.n_dimensions):
                if all(is_valid_offset(idx, off) for idx, off in zip(cell_idx, offset)):
                    valid_offsets.append(offset)
                    break

        self.neighbor_offsets = np.array(valid_offsets)

    def _hash_position(self, position: NDArray) -> Tuple[int, ...]:
        """
        Hash a position to a cell index.

        Args:
            position: Position vector.

        Returns:
            Tuple of cell indices.
        """
        indices = (position / self.cell_size).astype(int)
        indices = np.flipud(indices)

        if self.use_pbc:
            cell_indices = tuple(idx % self.n_cells for idx in indices)
        else:
            cell_indices = tuple(
                np.clip(idx, 0, self.n_cells - 1) for idx in indices
            )

        return cell_indices

    def build(self) -> None:
        """Build the cell list by assigning particles to cells."""
        self._cells = {
            tuple(idx): []
            for idx in np.ndindex((self.n_cells,) * self.n_dimensions)
        }
        self._stale = False

        for particle in self.particles:
            if not self.use_pbc:
                if not all(0 <= p < self.box_length for p in particle.position):
                    continue
            cell_index = self._hash_position(particle.position)
            self._cells[cell_index].append(particle)

    def update(self) -> None:
        """Update the cell list after particles have moved."""
        self.build()

    def invalidate(self) -> None:
        """Record that particles have moved; ``cells`` is rebuilt on next access."""
        self._stale = True

    def neighbor_pairs(self, positions: Optional[NDArray] = None) -> Tuple[NDArray, NDArray, NDArray]:
        """All ordered neighbour pairs ``(i, j, distance)`` at once (see the
        module-level ``neighbor_pairs``). ``positions`` defaults to the current
        positions of ``self.particles``."""
        if positions is None:
            positions = np.array([p.position for p in self.particles], dtype=float)
        return neighbor_pairs(positions, self.box_length, self.interaction_range, self.use_pbc)

    def mean_neighbor_velocity(self, positions: NDArray, velocities: NDArray) -> Tuple[NDArray, NDArray]:
        """Mean velocity over each particle and its neighbours, for all particles at once.

        Returns ``(mean, n_neighbors)``: ``mean[i]`` is the average of
        ``velocities[i]`` and the velocities of the neighbours of ``i`` (so a
        particle with no neighbours keeps its own velocity), and
        ``n_neighbors[i]`` is the number of neighbours, excluding ``i`` itself.
        This is what ``Vicsek.step`` averages before normalising.
        """
        velocities = np.asarray(velocities, dtype=float)
        focal, neighbor, _ = self.neighbor_pairs(positions)
        n, dim = velocities.shape
        sums = velocities.copy()
        for k in range(dim):
            sums[:, k] += np.bincount(focal, weights=velocities[neighbor, k], minlength=n)
        n_neighbors = np.bincount(focal, minlength=n)
        return sums / (n_neighbors + 1)[:, None], n_neighbors

    def get_neighbors(self, particle: Union[Particle, int]) -> List[Particle]:
        """
        Get neighboring particles within the interaction range.

        Args:
            particle: Particle or index of particle to find neighbors for.

        Returns:
            List of neighboring particles.
        """
        if isinstance(particle, int):
            particle = self.particles[particle]

        if not self.use_pbc:
            if not all(0 <= p < self.box_length for p in particle.position):
                return []

        neighbors = []
        cell_index = self._hash_position(particle.position)
        visited = set()

        for offset in self.neighbor_offsets:
            if self.use_pbc:
                neighbor_cell = tuple(int(i) for i in (np.array(cell_index) + offset) % self.n_cells)
            else:
                neighbor_cell = tuple(int(i) for i in np.array(cell_index) + offset)
                if not all(0 <= idx < self.n_cells for idx in neighbor_cell):
                    continue
            # With fewer than three cells per axis several offsets wrap onto the
            # same cell; visit each cell once so no neighbor is counted twice.
            if neighbor_cell in visited:
                continue
            visited.add(neighbor_cell)
            neighbors.extend(self.cells[neighbor_cell])

        if self.use_pbc:
            neighbors = [
                p for p in neighbors
                if p is not particle and self._minimum_image_distance(p, particle) <= self.interaction_range
            ]
        else:
            neighbors = [
                p for p in neighbors
                if p is not particle and np.linalg.norm(p - particle) <= self.interaction_range
            ]

        return neighbors

    def _minimum_image_distance(self, p1: Particle, p2: Particle) -> float:
        """
        Compute minimum image distance between two particles.

        Args:
            p1: First particle.
            p2: Second particle.

        Returns:
            Minimum image distance.
        """
        delta = p1.position - p2.position
        delta = delta - self.box_length * np.round(delta / self.box_length)
        return np.linalg.norm(delta)

    def visualize(
            self,
            ax: plt.Axes = None,
            show_cell_grid: bool = True,
            label_cells: bool = False,
            label_particles: bool = False,
            color: str = 'dimgrey',
    ) -> plt.Axes:
        """
        Draw the cell grid and the particles it contains (first two axes only).

        Args:
            ax: Matplotlib axes to plot on. If None, a new figure is created.
            show_cell_grid: Whether to draw the cell boundaries.
            label_cells: Whether to annotate each cell with its index tuple.
            label_particles: Whether to annotate each particle with its name.
            color: Color of the particle markers.

        Returns:
            The matplotlib axes with the cell list drawn.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if show_cell_grid:
            edges = np.arange(self.n_cells + 1) * self.cell_size
            for edge in edges:
                ax.axvline(edge, color='lightgrey', linewidth=0.8, zorder=0)
                ax.axhline(edge, color='lightgrey', linewidth=0.8, zorder=0)

        if label_cells:
            for cell_index in self.cells:
                # Cell indices are stored in reversed axis order (see _hash_position).
                xy = np.flipud(np.array(cell_index))
                centre = (xy + 0.5) * self.cell_size
                ax.annotate(str(cell_index), (centre[0], centre[1]),
                            ha='center', va='center', fontsize=7, color='grey')

        for particle in self.particles:
            particle.visualize(ax=ax, label=label_particles, color=color, size=20)

        ax.set_xlim(0, self.box_length)
        ax.set_ylim(0, self.box_length)
        ax.set_aspect('equal')
        return ax