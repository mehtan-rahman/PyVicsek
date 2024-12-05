from typing import List, Dict, Tuple, NamedTuple

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from tqdm import tqdm

from vicsek.models.vicsek import Vicsek
from vicsek.util.linalg import _random_unit_vector
from vicsek.models.particle import Particle


class HeterogeneousVicsek(Vicsek):
    """Enhanced Vicsek model with heterogeneous particle interactions and phase analysis"""

    __slots__ = ('length', 'interaction_range', 'v', 'mu', 'delta_t',
                 'particles', 'dim', '_cell_list', 'alignment_weights',
                 'noise_weights', 'particle_types')

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
        self.particle_types = list({p.type for p in particles})

        # Validate weights matrices
        for type1 in self.particle_types:
            for type2 in self.particle_types:
                if (type1, type2) not in alignment_weights:
                    raise ValueError(f"Missing alignment weight for types {type1} and {type2}")
                if (type1, type2) not in noise_weights:
                    raise ValueError(f"Missing noise weight for types {type1} and {type2}")

    def _compute_weighted_velocity(self, particle: Particle, neighbors: List[Particle]) -> np.ndarray:
        """Compute weighted average velocity considering particle types"""
        if not neighbors:
            return particle.velocity / np.linalg.norm(particle.velocity)

        weights = np.array([
            self.alignment_weights[(particle.type, neighbor.type)]
            for neighbor in neighbors
        ])

        velocities = np.array([neighbor.velocity for neighbor in neighbors])
        weights = np.append(weights, self.alignment_weights[(particle.type, particle.type)])
        velocities = np.vstack([velocities, particle.velocity])

        # Handle negative alignment weights (anti-alignment)
        velocities[weights < 0] *= -1
        weights = np.abs(weights)
        weights = weights / np.sum(weights)

        mean_velocity = np.average(velocities, axis=0, weights=weights)
        norm = np.linalg.norm(mean_velocity)

        return mean_velocity / norm if norm > 0 else _random_unit_vector(self.dim)

    def _compute_effective_noise(self, particle: Particle, neighbors: List[Particle]) -> float:
        """Compute effective noise based on particle types"""
        if not neighbors:
            return self.mu * self.noise_weights[(particle.type, particle.type)]

        neighbor_weights = [
            self.noise_weights[(particle.type, neighbor.type)]
            for neighbor in neighbors
        ]

        weights = neighbor_weights + [self.noise_weights[(particle.type, particle.type)]]
        return self.mu * np.mean(weights)

    def step(self):
        """Execute one timestep of the simulation"""
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
        """Calculate order parameters for each particle type"""
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

    def compute_cross_correlations(self) -> Dict[Tuple[str, str], float]:
        correlations = {}
        for i, type1 in enumerate(self.particle_types):
            for type2 in self.particle_types[i:]:  # Upper triangle only
                type1_particles = [p for p in self.particles if p.type == type1]
                type2_particles = [p for p in self.particles if p.type == type2]

                if not type1_particles or not type2_particles:
                    correlations[(type1, type2)] = 0.0
                    continue

                v1_mean = np.mean([p.velocity for p in type1_particles], axis=0)
                v2_mean = np.mean([p.velocity for p in type2_particles], axis=0)

                corr = np.dot(v1_mean, v2_mean) / (np.linalg.norm(v1_mean) * np.linalg.norm(v2_mean))
                correlations[(type1, type2)] = corr

        return correlations

    def simulate_phase_transition(
            self,
            noise_values: NDArray,
            equilibration_steps: int = 400,
            measurement_steps: int = 300
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        global_order = []
        global_fluctuations = []
        type_orders = {ptype: [] for ptype in self.particle_types}
        type_fluctuations = {ptype: [] for ptype in self.particle_types}
        cross_correlations = {(t1, t2): [] for t1 in self.particle_types
                              for t2 in self.particle_types}

        # Iterate through noise values
        for noise in tqdm(noise_values, desc="Noise values", position=0):
            self.mu = noise  # Set base noise

            # Equilibration phase
            self.run(equilibration_steps)

            # Measurement phase
            global_measurements = []
            type_measurements = {ptype: [] for ptype in self.particle_types}
            correlation_measurements = {(t1, t2): [] for t1 in self.particle_types
                                        for t2 in self.particle_types}

            for _ in tqdm(range(measurement_steps),
                          desc=f"Measuring η={noise:.3f}"):
                self.step()

                # Global order parameter
                global_measurements.append(self.order_parameter())

                # Type-specific order parameters
                type_orders_current = self.get_type_specific_order()
                for ptype, order in type_orders_current.items():
                    type_measurements[ptype].append(order)

                # Cross-correlations between types
                correlations = self.compute_cross_correlations()
                for key, value in correlations.items():
                    correlation_measurements[key].append(value)

            # Store averaged measurements
            global_order.append(np.mean(global_measurements))
            global_fluctuations.append(len(self.particles) * np.var(global_measurements))

            # Store type-specific measurements
            for ptype in self.particle_types:
                if type_measurements[ptype]:  # Check if we have measurements
                    type_orders[ptype].append(np.mean(type_measurements[ptype]))
                    type_fluctuations[ptype].append(
                        len([p for p in self.particles if p.type == ptype]) *
                        np.var(type_measurements[ptype])
                    )

            # Store cross-correlation measurements
            for key in correlation_measurements:
                if correlation_measurements[key]:  # Check if we have measurements
                    cross_correlations[key].append(np.mean(correlation_measurements[key]))

        # Convert lists to numpy arrays
        global_order = np.array(global_order)
        global_fluctuations = np.array(global_fluctuations)
        type_orders = {k: np.array(v) for k, v in type_orders.items()}
        type_fluctuations = {k: np.array(v) for k, v in type_fluctuations.items()}
        cross_correlations = {k: np.array(v) for k, v in cross_correlations.items()}

        return global_order, global_fluctuations, type_orders, type_fluctuations, cross_correlations
