from typing import List, Dict, Tuple, NamedTuple
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from vicsek.models.vicsek import Vicsek, _random_unit_vector
from vicsek.models.particle import Particle


class PhaseTransitionResults(NamedTuple):
    """Container for phase transition analysis results"""
    noise_values: np.ndarray
    global_order: np.ndarray
    global_susceptibility: np.ndarray
    type_orders: Dict[str, np.ndarray]
    type_susceptibilities: Dict[str, np.ndarray]
    cross_correlations: Dict[Tuple[str, str], np.ndarray]


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
            noise_values: np.ndarray,
            equilibration_steps: int = 400,
            measurement_steps: int = 300
    ) -> PhaseTransitionResults:
        global_orders = []
        global_susceptibilities = []
        type_orders = {ptype: [] for ptype in self.particle_types}
        type_susceptibilities = {ptype: [] for ptype in self.particle_types}
        cross_correlations = {
            (t1, t2): []
            for i, t1 in enumerate(self.particle_types)
            for t2 in self.particle_types[i:]
        }

        original_noise = self.mu

        for noise in tqdm(noise_values, desc="Analyzing noise values"):
            # Set noise and equilibrate
            self.mu = noise
            self.run(equilibration_steps)

            # Initialize measurement arrays
            global_measurements = []
            type_measurements = {ptype: [] for ptype in self.particle_types}
            correlation_measurements = {k: [] for k in cross_correlations.keys()}

            # Collect measurements
            for _ in range(measurement_steps):
                self.step()

                # Global order
                global_order = self.order_parameter()
                global_measurements.append(global_order)

                # Type-specific orders
                type_orders_current = self.get_type_specific_order()
                for ptype in self.particle_types:
                    type_measurements[ptype].append(type_orders_current.get(ptype, 0.0))

                # Cross-correlations
                correlations = self.compute_cross_correlations()
                for key in correlation_measurements:
                    correlation_measurements[key].append(correlations[key])

            # Calculate statistics
            N = len(self.particles)

            # Global quantities
            global_orders.append(np.mean(global_measurements))
            global_susceptibilities.append(N * np.var(global_measurements))

            # Type-specific quantities
            for ptype in self.particle_types:
                type_orders[ptype].append(np.mean(type_measurements[ptype]))
                type_susceptibilities[ptype].append(N * np.var(type_measurements[ptype]))

            # Cross-correlations
            for key in cross_correlations:
                cross_correlations[key].append(np.mean(correlation_measurements[key]))

        # Restore original noise value
        self.mu = original_noise

        return PhaseTransitionResults(
            noise_values=noise_values,
            global_order=np.array(global_orders),
            global_susceptibility=np.array(global_susceptibilities),
            type_orders={k: np.array(v) for k, v in type_orders.items()},
            type_susceptibilities={k: np.array(v) for k, v in type_susceptibilities.items()},
            cross_correlations={k: np.array(v) for k, v in cross_correlations.items()}
        )

    def find_transition_points(self, results: PhaseTransitionResults) -> Dict[str, float]:
        transition_points = {}

        peak_idx = np.argmax(results.global_susceptibility)
        transition_points['global'] = results.noise_values[peak_idx]

        for ptype in self.particle_types:
            peak_idx = np.argmax(results.type_susceptibilities[ptype])
            transition_points[f'type_{ptype}'] = results.noise_values[peak_idx]

        return transition_points

    def plot_phase_diagram(self, results: PhaseTransitionResults) -> plt.Figure:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

        # Global order and susceptibility
        ax1.plot(results.noise_values, results.global_order, 'k-', label='Global Order')
        ax1.set_xlabel('Noise (η)')
        ax1.set_ylabel('Order Parameter (φ)')
        ax1.legend()

        ax2.plot(results.noise_values, results.global_susceptibility, 'r-',
                 label='Global Susceptibility')
        ax2.set_xlabel('Noise (η)')
        ax2.set_ylabel('Susceptibility (χ)')
        ax2.legend()

        # Type-specific orders
        for ptype, orders in results.type_orders.items():
            ax3.plot(results.noise_values, orders, label=f'Type {ptype}')
        ax3.set_xlabel('Noise (η)')
        ax3.set_ylabel('Type-Specific Order')
        ax3.legend()

        # Cross-correlations
        for (t1, t2), corr in results.cross_correlations.items():
            ax4.plot(results.noise_values, corr, label=f'{t1}-{t2}')
        ax4.set_xlabel('Noise (η)')
        ax4.set_ylabel('Cross-correlation')
        ax4.legend()

        plt.tight_layout()
        return fig
