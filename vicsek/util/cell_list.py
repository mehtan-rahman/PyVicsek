from typing import Dict, Tuple, List, Union
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from vicsek.models.particle import Particle


class CellList:
    def __init__(
            self,
            particles: List[Particle],
            box_length: float,
            interaction_range: float,
            n_dimensions: int = 2,
            use_pbc: bool = True
    ):
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
        indices = np.flipud(indices)

        if self.use_pbc:
            cell_indices = tuple(idx % self.n_cells for idx in indices)
        else:
            cell_indices = tuple(
                np.clip(idx, 0, self.n_cells - 1) for idx in indices
            )

        return cell_indices

    def build(self) -> None:
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
        for cell in self.cells.values():
            cell.clear()

        for particle in self.particles:
            if not self.use_pbc:
                if not all(0 <= p < self.box_length for p in particle.position):
                    continue
            cell_index = self._hash_position(particle.position)
            self.cells[cell_index].append(particle)

    def get_neighbors(self, particle: Union[Particle, int]) -> List[Particle]:
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
        delta = p1.position - p2.position
        delta = delta - self.box_length * np.round(delta / self.box_length)
        return np.linalg.norm(delta)

    def visualize(
            self,
            ax: plt.Axes = None,
            show_cell_grid: bool = False,
            label_cells: bool = False,
            label_particles: bool = False
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlim(0, self.box_length)
        ax.set_ylim(0, self.box_length)

        if show_cell_grid and self.n_dimensions <= 2:
            for i in range(self.n_cells + 1):
                pos = i * self.cell_size
                ax.axvline(x=pos, color='black', alpha=0.3, linestyle='-')
                ax.axhline(y=pos, color='black', alpha=0.3, linestyle='-')

                if label_cells and i < self.n_cells:
                    for j in range(self.n_cells):
                        center_x = (j + 0.5) * self.cell_size
                        center_y = (i + 0.5) * self.cell_size
                        ax.text(center_x, center_y, f'({i},{j})',
                                ha='center', va='center', alpha=0.5)

        for i, particle in enumerate(self.particles):
            pos = particle.position[:2]
            ax.scatter(pos[0], pos[1], c='dimgrey', s=50, alpha=0.7)
            if label_particles:
                ax.text(pos[0], pos[1], str(i), ha='right', va='bottom')

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        return ax
