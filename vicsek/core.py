from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from shapely import Geometry, Point
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation



class CellList:
    """
    Cell List data structure with optional periodic boundary conditions
    """

    def __init__(
            self,
            particles: List[Particle],
            box_length: float,
            interaction_range: float,
            n_dimensions: int = 2,
            use_pbc: bool = True
    ):
        """Initialize the cell list structure.

        Args:
            box_length: Length of the simulation box
            interaction_range: Maximum distance for particle interactions
            n_dimensions: Number of spatial dimensions (2 or 3)
            use_pbc: Whether to use periodic boundary conditions
        """
        if n_dimensions not in (2, 3):
            raise ValueError("n_dimensions must be either 2 or 3")

        self.particles = particles
        self.box_length = box_length
        self.interaction_range = interaction_range
        self.n_dimensions = n_dimensions
        self.use_pbc = use_pbc

        # Ensure cell size is slightly larger than interaction range for efficiency
        self.n_cells = max(1, int(np.floor(box_length / interaction_range)))
        self.cell_size = box_length / self.n_cells

        # Pre-compute neighbor offsets for efficiency
        if self.use_pbc:
            # Use all neighboring cells for PBC
            self.neighbor_offsets = np.array(
                np.meshgrid(*[[-1, 0, 1]] * self.n_dimensions)
            ).T.reshape(-1, self.n_dimensions)
        else:
            # Only use neighboring cells within bounds for non-PBC
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
        indices = (position / self.cell_size).astype(int)
        # Swap x and y, and flip y coordinate
        y_idx = self.n_cells - 1 - indices[1]  # flip y to start from bottom
        x_idx = indices[0]  # x stays the same

        if self.use_pbc:
            cell_indices = (y_idx % self.n_cells, x_idx % self.n_cells)
        else:
            cell_indices = (
                np.clip(y_idx, 0, self.n_cells - 1),
                np.clip(x_idx, 0, self.n_cells - 1)
            )

        return cell_indices

    def build(self) -> None:
        """Rebuild the entire cell list structure."""
        # Initialize all possible cells
        self.cells = {
            tuple(idx): []
            for idx in np.ndindex((self.n_cells,) * self.n_dimensions)
        }

        # Assign particles to cells
        for particle in self.particles:
            # Skip particles outside bounds when not using PBC
            if not self.use_pbc:
                if not all(0 <= p < self.box_length for p in particle.position):
                    continue
            cell_index = self._hash_position(particle.position)
            self.cells[cell_index].append(particle)

    def update(self) -> None:
        """Update cell assignments without full rebuild if possible."""
        # Clear current assignments but keep cell structure
        for cell in self.cells.values():
            cell.clear()

        # Reassign particles
        for particle in self.particles:
            # Skip particles outside bounds when not using PBC
            if not self.use_pbc:
                if not all(0 <= p < self.box_length for p in particle.position):
                    continue
            cell_index = self._hash_position(particle.position)
            self.cells[cell_index].append(particle)

    def get_neighbors_by_particle(self, particle: Particle) -> List[Particle]:
        """Get all particles within interaction range of the given particle."""
        # Skip if particle is outside bounds when not using PBC
        if not self.use_pbc:
            if not all(0 <= p < self.box_length for p in particle.position):
                return []

        neighbors = []
        cell_index = self._hash_position(particle.position)

        # Use pre-computed offsets for neighbor cells
        for offset in self.neighbor_offsets:
            if self.use_pbc:
                neighbor_cell = tuple((np.array(cell_index) + offset) % self.n_cells)
            else:
                neighbor_cell = tuple(np.array(cell_index) + offset)
                # Skip cells outside bounds for non-PBC
                if not all(0 <= idx < self.n_cells for idx in neighbor_cell):
                    continue
            neighbors.extend(self.cells[neighbor_cell])

        # Filter by actual distance and remove self
        if self.use_pbc:
            # Use minimum image convention for PBC
            neighbors = [
                p for p in neighbors
                if p is not particle and self._minimum_image_distance(p, particle) <= self.interaction_range
            ]
        else:
            # Use direct distance for non-PBC
            neighbors = [
                p for p in neighbors
                if p is not particle and np.linalg.norm(p - particle) <= self.interaction_range
            ]

        return neighbors

    def _minimum_image_distance(self, p1: Particle, p2: Particle) -> float:
        """Calculate minimum image distance between two particles under PBC."""
        delta = p1.position - p2.position
        # Apply minimum image convention
        delta = delta - self.box_length * np.round(delta / self.box_length)
        return np.linalg.norm(delta)

    def get_neighbors_by_index(self, idx: int) -> List[Particle]:
        particle = self.particles[idx]
        return self.get_neighbors_by_particle(particle)

    def visualize(
            self,
            ax: plt.Axes = None,
            show_cell_grid: bool = False,
            label_cells: bool = False,
            label_particles: bool = False
    ) -> plt.Axes:
        """Visualize particles and optionally show cell grid and labels."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Set up the plot bounds
        ax.set_xlim(0, self.box_length)
        ax.set_ylim(0, self.box_length)

        # Draw cell grid and labels
        if show_cell_grid:
            for i in range(self.n_cells + 1):
                pos = i * self.cell_size
                ax.axvline(x=pos, color='black', alpha=0.3, linestyle='-')
                ax.axhline(y=pos, color='black', alpha=0.3, linestyle='-')

                # Add cell labels if requested
                if label_cells and i < self.n_cells:
                    for j in range(self.n_cells):
                        center_x = (j + 0.5) * self.cell_size
                        center_y = (i + 0.5) * self.cell_size
                        # Note: i represents y-coordinate from bottom up
                        ax.text(center_x, center_y, f'({self.n_cells - 1 - i},{j})',
                                ha='center', va='center', alpha=0.5)

        # Plot all particles
        for i, particle in enumerate(self.particles):
            ax.scatter(particle.position[0], particle.position[1],
                       c='dimgrey', s=50, alpha=0.7)
            if label_particles:
                ax.text(particle.position[0], particle.position[1], str(i),
                        ha='right', va='bottom')

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        return ax
