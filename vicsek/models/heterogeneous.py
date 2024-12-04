from typing import List, Dict, Tuple

import numpy as np

from vicsek.models.vicsek import Vicsek, _random_unit_vector
from vicsek.models.particle import Particle


class HeterogeneousVicsek(Vicsek):
    __slots__ = ('length', 'interaction_range', 'v', 'mu', 'delta_t',
                 'particles', 'dim', '_cell_list', 'alignment_weights',
                 'noise_weights')

    def __init__(
            self,
            length: float,
            particles: List[Particle],
            interaction_range: float,
            speed: float,
            base_noise: float,
            alignment_weights: Dict[Tuple[str, str], float],
            noise_weights: Dict[Tuple[str, str], float],
            timestep: float = 1,
            use_pbc: bool = True,
    ) -> None:

        super().__init__(
            length=length,
            particles=particles,
            interaction_range=interaction_range,
            speed=speed,
            noise_factor=base_noise,
            timestep=timestep,
            use_pbc=use_pbc
        )
        self.alignment_weights = alignment_weights
        self.noise_weights = noise_weights

        particle_types = {p.type for p in particles}
        for type1 in particle_types:
            for type2 in particle_types:
                if (type1, type2) not in alignment_weights:
                    raise ValueError(f"Missing alignment weight for types {type1} and {type2}")
                if (type1, type2) not in noise_weights:
                    raise ValueError(f"Missing noise weight for types {type1} and {type2}")

    def _compute_weighted_velocity(self, particle: Particle, neighbors: List[Particle]) -> np.ndarray:
        if not neighbors:
            return particle.velocity / np.linalg.norm(particle.velocity)

        weights = np.array([
            self.alignment_weights[(particle.type, neighbor.type)]
            for neighbor in neighbors
        ])

        velocities = np.array([neighbor.velocity for neighbor in neighbors])
        weights = np.append(weights, self.alignment_weights[(particle.type, particle.type)])
        velocities = np.vstack([velocities, particle.velocity])

        velocities[weights < 0] *= -1
        weights = np.abs(weights)

        weights = weights / np.sum(weights)

        mean_velocity = np.average(velocities, axis=0, weights=weights)
        norm = np.linalg.norm(mean_velocity)

        return mean_velocity / norm if norm > 0 else _random_unit_vector(self.dim)

    def _compute_effective_noise(self, particle: Particle, neighbors: List[Particle]) -> float:
        if not neighbors:
            return self.mu * self.noise_weights[(particle.type, particle.type)]

        neighbor_weights = [
            self.noise_weights[(particle.type, neighbor.type)]
            for neighbor in neighbors
        ]

        weights = neighbor_weights + [self.noise_weights[(particle.type, particle.type)]]
        effective_noise = self.mu * np.mean(weights)

        return effective_noise

    def step(self):
        for particle in self.particles:
            neighbors = self._cell_list.get_neighbors(particle)

            avg_direction = self._compute_weighted_velocity(particle, neighbors)

            effective_noise = self._compute_effective_noise(particle, neighbors)
            noise = effective_noise * _random_unit_vector(self.dim)
            noisy_direction = avg_direction + noise
            noisy_direction = noisy_direction / np.linalg.norm(noisy_direction)

            particle.velocity = self.v * noisy_direction
            new_position = particle.position + particle.velocity * self.delta_t

            if self._cell_list.use_pbc:
                new_position = new_position % self.length

            particle.position = new_position

        self._cell_list.update()

    def get_type_specific_order(self) -> Dict[str, float]:
        type_velocities = {}
        type_counts = {}

        for p in self.particles:
            if p.type not in type_velocities:
                type_velocities[p.type] = np.zeros(self.dim)
                type_counts[p.type] = 0

            type_velocities[p.type] += p.velocity
            type_counts[p.type] += 1

        return {
            ptype: np.linalg.norm(vel) / (self.v * type_counts[ptype])
            for ptype, vel in type_velocities.items()
            if type_counts[ptype] > 0
        }
