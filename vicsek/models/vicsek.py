from typing import List
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from vicsek.models.particle import Particle
from vicsek.util.cell_list import CellList


def _random_unit_vector(n_dimensions: int) -> NDArray:
    vector = np.random.normal(0, 1, n_dimensions)
    return vector / np.linalg.norm(vector)


class Vicsek:
    def __init__(
            self,
            length: float,
            particles: List[Particle],
            interaction_range: float,
            speed: float,
            noise_factor: float,
            timestep: float = 1,
            use_pbc: bool = True,
    ) -> None:
        self.length = length
        self.interaction_range = interaction_range
        self.v = speed
        self.mu = noise_factor
        self.delta_t = timestep

        self.particles = particles
        self.dim = len(particles[0].position)

        self._cell_list = CellList(
            particles=self.particles,
            box_length=length,
            interaction_range=interaction_range,
            n_dimensions=self.dim,
            use_pbc=use_pbc
        )
        self._cell_list.build()

    def _compute_average_velocity(self, velocities: NDArray) -> NDArray:
        mean_velocity = np.mean(velocities, axis=0)
        norm = np.linalg.norm(mean_velocity)
        return mean_velocity / norm if norm > 0 else _random_unit_vector(self.dim)

    def _apply_noise(self, velocity: NDArray) -> NDArray:
        noise = self.mu * _random_unit_vector(self.dim)
        noisy_velocity = velocity + noise
        return noisy_velocity / np.linalg.norm(noisy_velocity)

    def step(self):
        for particle in self.particles:
            neighbors = self._cell_list.get_neighbors(particle)
            all_particles = [particle] + neighbors

            velocities = np.array([p.velocity for p in all_particles])
            avg_direction = self._compute_average_velocity(velocities)

            noisy_direction = self._apply_noise(avg_direction)
            particle.velocity = self.v * noisy_direction

            new_position = particle.position + particle.velocity * self.delta_t

            if self._cell_list.use_pbc:
                new_position = new_position % self.length

            particle.position = new_position

        self._cell_list.update()

    def run(self, iterations: int = 10):
        for _ in range(iterations):
            self.step()

    def order_parameter(self) -> float:
        velocities = np.array([p.velocity for p in self.particles])
        return np.linalg.norm(np.mean(velocities, axis=0)) / self.v

    def visualize(
            self,
            ax: plt.Axes = None,
            show_velocity: bool = True,
            show_cells: bool = False
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if show_cells:
            self._cell_list.visualize(ax=ax, show_cell_grid=True)

        for particle in self.particles:
            particle.visualize(
                ax=ax,
                show_velocity=show_velocity,
                size=50,
                alpha=0.7
            )

        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.length)
        ax.set_aspect('equal')
        return ax
