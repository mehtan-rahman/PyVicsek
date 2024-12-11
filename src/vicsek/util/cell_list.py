from typing import Dict, Tuple, List, Union
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from vicsek.models.particle import Particle


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
    """

    __slots__ = ('particles', 'box_length', 'interaction_range', 'n_dimensions',
                 'use_pbc', 'n_cells', 'cell_size', 'neighbor_offsets', 'cells')

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

        self.n_cells = max(1, int(np.floor(box_length / interaction_range)))
        self.cell_size = box_length / self.n_cells

        if self.use_pbc:
            self.neighbor_offsets = np.array(
                np.meshgrid(*[[-1, 0, 1]] * self.n_dimensions)
            ).T.reshape(-1, self.n_dimensions)
        else:
            self._compute_non_pbc_neighbor_offsets()

        self.cells: Dict[Tuple[int, ...], List[Particle]] = {}

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
        self.cells = {
            tuple(idx): []
            for idx in np.ndindex((self.n_cells,) * self.n_dimensions)
        }

        for particle in self.particles:
            if not self.use_pbc:
                if not all(0 <= p < self.box_length for p in particle.position):
                    continue
            cell_index = self._hash_position(particle.position)
            self.cells[cell_index].append(particle)

    def update(self) -> None:
        """Update the cell list after particles have moved."""
        for cell in self.cells.values():
            cell.clear()

        for particle in self.particles:
            if not self.use_pbc:
                if not all(0 <= p < self.box_length for p in particle.position):
                    continue
            cell_index = self._hash_position(particle.position)
            self.cells[cell_index].append(particle)

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

        for offset in self.neighbor_offsets:
            if self.use_pbc:
                neighbor_cell = tuple((np.array(cell_index) + offset) % self.n_cells)
            else:
                neighbor_cell = tuple(np.array(cell_index) + offset)
                if not all(0 <= idx < self.n_cells for idx in neighbor_cell):
                    continue
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