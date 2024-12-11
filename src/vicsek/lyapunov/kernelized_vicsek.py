from typing import List

import numpy as np

from vicsek.models.vicsek import Vicsek
from vicsek.models.particle import Particle
from vicsek.util.linalg import _random_unit_vector


class KernelizedVicsek(Vicsek):
    """Vicsek model with smooth weighting function for Lyapunov analysis.

    This class extends the basic Vicsek model by introducing a smooth weighting
    function for the interaction between particles. The smooth weighting is
    necessary for computing Lyapunov exponents, which measure the chaotic
    behavior of the system.

    The smooth weighting function replaces the hard cutoff at the interaction
    range used in the original Vicsek model. It ensures that the interaction
    goes to zero continuously as the distance between particles approaches
    the interaction range.

    Attributes:
        epsilon (float): Smoothing parameter for the weighting function.
            Smaller values approximate the original Vicsek model more closely.
    """

    def __init__(
            self,
            length: float,
            particles: List[Particle],
            interaction_range: float,
            speed: float,
            noise_factor: float,
            epsilon: float = 1e-8,
            timestep: float = 1,
            use_pbc: bool = True,
    ):
        """Initialize a KernelizedVicsek simulation.

        Args:
            length (float): Length of the simulation box.
            particles (List[Particle]): List of particles in the simulation.
            interaction_range (float): Range of interaction between particles.
            speed (float): Speed of the particles.
            noise_factor (float): Strength of the angular noise.
            epsilon (float): Smoothing parameter for the weighting function.
            timestep (float): Duration of a single timestep.
            use_pbc (bool): Whether to use periodic boundary conditions.
        """
        super().__init__(
            length=length,
            particles=particles,
            interaction_range=interaction_range,
            speed=speed,
            noise_factor=noise_factor,
            timestep=timestep,
            use_pbc=use_pbc
        )
        self.epsilon = epsilon

    def _smooth_weight(self, distance: float) -> float:
        """Compute smooth weighting factor.

        The weighting factor is computed using a cubic function that smoothly
        decreases from 1 to 0 as the distance approaches the interaction range.

        Args:
            distance (float): Distance between two particles.

        Returns:
            float: The weighting factor.
        """
        if distance < 1 - 2 * self.epsilon:
            return 1.0
        elif distance > 1:
            return 0.0
        else:
            x = (distance - 1)
            return (x * x * (3 * self.epsilon + x)) / (4 * self.epsilon ** 3)

    def _compute_average_velocity(self, particle: Particle) -> np.ndarray:
        """Compute average velocity using smooth weights for a single particle.

        Args:
            particle (Particle): The particle for which to compute the average.

        Returns:
            np.ndarray: The average velocity vector.
        """
        # Get neighbors
        neighbors = self._cell_list.get_neighbors(particle)
        all_particles = [particle] + neighbors

        if not neighbors:  # If no neighbors, maintain current direction
            return particle.velocity / np.linalg.norm(particle.velocity)

        # Collect positions and velocities
        positions = np.array([p.position for p in all_particles])
        velocities = np.array([p.velocity for p in all_particles])

        # Compute distances from current particle to all others
        if self._cell_list.use_pbc:
            # Use minimum image convention
            delta = positions - positions[0]
            delta = delta - self.length * np.round(delta / self.length)
            distances = np.linalg.norm(delta, axis=1)
        else:
            distances = np.linalg.norm(positions - positions[0], axis=1)

        # Compute weights
        weights = np.array([self._smooth_weight(d) for d in distances])
        weights = weights / np.sum(weights)  # Normalize weights

        # Compute weighted average velocity
        mean_velocity = np.sum(velocities * weights[:, np.newaxis], axis=0)
        norm = np.linalg.norm(mean_velocity)

        if norm > 0:
            return mean_velocity / norm
        return _random_unit_vector(self.dim)

    def step(self):
        """Single time step evolution with smooth weighting.

        Each particle's velocity is updated based on the smoothly weighted
        average of its neighbors' velocities, with angular noise applied.
        The particle positions are then updated based on their new velocities.
        """
        for particle in self.particles:
            # Compute average direction using smooth weights
            avg_direction = self._compute_average_velocity(particle)

            # Apply noise and update
            noisy_direction = self._apply_noise(avg_direction)
            particle.velocity = self.v * noisy_direction

            # Update position
            new_position = particle.position + particle.velocity * self.delta_t
            if self._cell_list.use_pbc:
                new_position = new_position % self.length
            particle.position = new_position

        self._cell_list.update()