from typing import List, Dict, Tuple

import numpy as np
from numpy.typing import NDArray, ArrayLike
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from vicsek.models.particle import Particle
from vicsek.util.cell_list import CellList
from vicsek.util.linalg import _random_unit_vector


class Vicsek:
    __slots__ = ('length', 'interaction_range', 'v', 'mu', 'delta_t',
                 'particles', 'dim', '_cell_list')
"""
A class for simulating the Vicsek model of self-propelled particles.

Parameters
    length (float): Length of the simulation box.
    particles (List[Particle]): List of particles in the simulation.
    interaction_range (float): Range of interaction between particles.
    speed (float): Speed of the particles.
    noise_factor (float): Strength of the angular noise.
    timestep (float): Duration of a single timestep.
    use_pbc (bool): Whether to use periodic boundary conditions.

"""
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
        self._cell_list.build()  #Create a cell list for efficient neighbor search

    def _compute_average_velocity(self, velocities: NDArray) -> NDArray:
        """Computes normalized average velocity of particles. 
            If the norm is zero, a random direction is returned. """
        
        mean_velocity = np.mean(velocities, axis=0)
        norm = np.linalg.norm(mean_velocity)
        return mean_velocity / norm if norm > 0 else _random_unit_vector(self.dim)

    def _apply_noise(self, velocity: NDArray) -> NDArray:
    """ Adding noise to velocity"""
        noise = self.mu * _random_unit_vector(self.dim)
        noisy_velocity = velocity + noise
        return noisy_velocity / np.linalg.norm(noisy_velocity)

    def step(self):
        for particle in self.particles:
            neighbors = self._cell_list.get_neighbors(particle) #finding neighbors based on the cell list
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
        for _ in tqdm(range(iterations), f'Running {iterations} Steps'):
            self.step()

    def equilibrate(
            self,
            window_size: int = 100,
            threshold: float = 0.01,
            max_steps: int = 1000,
            check_interval: int = 1,
            min_steps: int = 250,
            progress_bar: bool = True
    ) -> Tuple[bool, int, float]:
        """ 
        Checks if order parameter stabilizes (system equilibrates) within specified window of steps 
        Parameters:
            window_size (int): The size of the window used to compute variance for checking stabilization (default is 100).
            threshold (float): The variance threshold below which the system is considered equilibrated (default is 0.01).
            max_steps (int): The maximum number of steps to simulate (default is 1000).
            check_interval (int): The interval at which to check the order parameter (default is 1).
            min_steps (int): The minimum number of steps before checking equilibration (default is 250).
            progress_bar (bool): Whether to show a progress bar (default is True).

        Returns:
            Tuple[bool, int, float]: A tuple containing:
                - A boolean indicating whether the system equilibrated.
                - The total number of steps taken.
                - The variance of the order parameter at the end of the equilibration.
        """
        
        order_params = []
        total_steps = 0

        iterator = range(max_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="Equilibrating")

        for step in iterator:
            self.step()
            total_steps += 1

            if step % check_interval == 0:
                order_params.append(self.order_parameter())

                if len(order_params) >= window_size and total_steps >= min_steps:
                    window = order_params[-window_size:]
                    variance = np.var(window)

                    if variance < threshold:
                        if progress_bar:
                            iterator.close()
                        return True, total_steps, variance

        if progress_bar:
            iterator.close()
        return False, total_steps, np.var(order_params[-window_size:])

    def order_parameter(self) -> float:
    """
    Computes the order parameter of the system, which measures the alignment of particle velocities.

    Returns:
        float: The order parameter value, a measure of the average alignment of particles, normalized by the speed.
    """
        velocities = np.array([p.velocity for p in self.particles])
        return np.linalg.norm(np.mean(velocities, axis=0)) / self.v

    def order_parameter_evolution(self, steps: int = 750) -> NDArray:
    """
    Tracks the evolution of the order parameter over a number of steps.

    Parameters:
        steps (int): The number of steps to track the evolution of the order parameter (default is 750).

    Returns:
        NDArray: An array containing the order parameter values over the specified number of steps.
    """
        
        order_params = []

        for _ in tqdm(range(steps), desc="Steps"):
            self.step()
            order_params.append(self.order_parameter())

        return np.array(order_params)


    def visualize(
            self,
            ax: plt.Axes = None,
            show_velocity: bool = False,
            show_cells: bool = False,
            legend: Dict[str, str] = None,
    ) -> plt.Axes:
    """ Visualizes the state of the system, including the positions and the velocities, and optionally the cell grid. """
       
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
        """ Creates an animation of particle motion over time. 
        
            Parameters:
                frames (int): The number of frames in the animation.
                interval (int): The interval (in milliseconds) between frames.
                particle_scale (float): The scaling factor for the particle arrows.
                legend (Dict[str, str]): A dictionary mapping particle types to colors for visualization.
        """
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
        """
    Simulates the phase transition of the system by running the simulation for different noise values and measuring the order parameter.

    Parameters:
        noise_values (ArrayLike): An array of noise values (η) to test during the phase transition simulation.
        equilibration_steps (int): The number of steps to equilibrate the system before measurements.
        measurement_steps (int): The number of steps to collect measurements of the order parameter.

    Returns:
        Tuple[NDArray, NDArray]: 
            - The average order parameter for each noise value.
            - The order parameter fluctuations (susceptibility) for each noise value.
    """
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
