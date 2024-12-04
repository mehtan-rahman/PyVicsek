from typing import List, Dict

import numpy as np
from numpy.typing import NDArray, ArrayLike
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from vicsek.models.particle import Particle
from vicsek.util.cell_list import CellList


def _random_unit_vector(n_dimensions: int) -> NDArray:
    vector = np.random.normal(0, 1, n_dimensions)
    return vector / np.linalg.norm(vector)


class Vicsek:
    __slots__ = ('length', 'interaction_range', 'v', 'mu', 'delta_t',
                 'particles', 'dim', '_cell_list')

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

    def run(self, iterations: int = 400):
        for _ in range(iterations):
            self.step()

    def order_parameter(self) -> float:
        velocities = np.array([p.velocity for p in self.particles])
        return np.linalg.norm(np.mean(velocities, axis=0)) / self.v

    def visualize(
            self,
            ax: plt.Axes = None,
            show_velocity: bool = False,
            show_cells: bool = False,
            legend: Dict[str, str] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if show_cells:
            self._cell_list.visualize(ax=ax, show_cell_grid=True)

        for particle in self.particles:
            color = legend.get(particle.type, 'dimgrey')

            particle.visualize(
                ax=ax,
                show_velocity=show_velocity,
                color=color,
                alpha=0.7
            )

        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.length)
        ax.set_aspect('equal')
        return ax

    def animate(
            self,
            frames: int = 200,
            interval: int = 50,
            particle_scale: float = 10,
            legend: Dict[str, str] = None,
    ) -> FuncAnimation:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.length)
        ax.set_aspect('equal')

        if legend is None:
            unique_types = set(p.type for p in self.particles)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
            legend = dict(zip(unique_types, colors))

        quivers = {}
        for p_type in legend.keys():
            type_particles = [p for p in self.particles if p.type == p_type]
            if not type_particles:
                continue

            positions = np.array([p.position for p in type_particles])
            orientations = np.arctan2([p.velocity[1] for p in type_particles],
                                      [p.velocity[0] for p in type_particles])

            qv = ax.quiver(positions[:, 0], positions[:, 1],
                           np.cos(orientations), np.sin(orientations),
                           color=legend[p_type],
                           scale=particle_scale,
                           scale_units='inches',
                           width=0.003,
                           headwidth=3,
                           headlength=5,
                           headaxislength=4.5,
                           pivot='mid',
                           minshaft=0,
                           label=p_type)

            quivers[p_type] = {
                'quiver': qv,
                'particles': type_particles
            }

        ax.legend(loc='upper right')
        order_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             verticalalignment='top')

        def animate(frame):
            self.step()
            artists = []

            for p_type, data in quivers.items():
                positions = np.array([p.position for p in data['particles']])
                orientations = np.arctan2([p.velocity[1] for p in data['particles']],
                                          [p.velocity[0] for p in data['particles']])

                data['quiver'].set_offsets(positions)
                data['quiver'].set_UVC(np.cos(orientations), np.sin(orientations))
                artists.append(data['quiver'])

            # Update order parameter
            order_param = self.order_parameter()
            order_text.set_text(f'Order: {order_param:.3f}')
            artists.append(order_text)

            return artists

        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)
        return anim

    def simulate_phase_transition(
            self,
            noise_values: ArrayLike,
            equilibration_steps: int = 400,
            measurement_steps: int = 300
    ):
        order_parameters = []
        order_fluctuations = []

        for noise in tqdm(noise_values, desc="Noise values", position=0):

            self.run(equilibration_steps)

            measurements = []
            for _ in tqdm(range(measurement_steps),
                          desc=f"Measuring η={noise:.3f}",
                          position=1,
                          leave=False):
                self.step()
                measurements.append(self.order_parameter())

            order_parameters.append(np.mean(measurements))
            order_fluctuations.append(len(self.particles) * np.var(measurements))  # Susceptibility

        return np.array(order_parameters), np.array(order_fluctuations)
