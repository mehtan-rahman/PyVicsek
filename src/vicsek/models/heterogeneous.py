from typing import List, Dict, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from tqdm import tqdm

from vicsek.models.vicsek import Vicsek
from vicsek.util.linalg import _random_unit_vector
from vicsek.models.particle import Particle


class HeterogeneousVicsek(Vicsek):
    """
    Enhanced Vicsek model with heterogeneous particle interactions and phase analysis.

    This class extends the base Vicsek model to include type-specific alignment
    and noise weights, allowing for more complex collective behavior.

    Attributes:
        length (float): Length of the simulation box.
        interaction_range (float): Interaction range of particles.
        v (float): Speed of particles.
        mu (float): Base noise factor.
        delta_t (float): Time step size.
        particles (List[Particle]): List of particles in the simulation.
        dim (int): Number of dimensions.
        _cell_list (CellList): Cell list for efficient neighbor searching.
        alignment_weights (Dict[Tuple[str, str], float]): Alignment weights between particle types.
        noise_weights (Dict[Tuple[str, str], float]): Noise weights between particle types.
        particle_types (List[str]): Sorted list of the unique particle types in the simulation.
        types (NDArray[str]): Type of every particle, shape ``(N,)``, in particle order.

    Particle types are read once, at construction, into the ``types`` array
    (``particle.type`` stays a plain attribute); changing a particle's type
    afterwards does not affect the dynamics. The weight dictionaries are read
    on every step, so they may be edited between steps.
    """

    __slots__ = ('alignment_weights', 'noise_weights', 'particle_types', 'types', '_type_ids')

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
            seed: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """
        Initialize a HeterogeneousVicsek simulation.

        Args:
            length: Length of the simulation box.
            particles: List of particles in the simulation.
            interaction_range: Interaction range of particles.
            speed: Speed of particles.
            base_noise: Base strength of the vectorial noise (see Vicsek.noise_factor);
                scaled per particle by the mean of the relevant noise weights.
            alignment_weights: Dictionary of alignment weights between particle types.
            noise_weights: Dictionary of noise weights between particle types.
            timestep: Time step size (default 1).
            use_pbc: Whether to use periodic boundary conditions (default True).
            seed: Seed or Generator for the noise; None uses numpy's global RNG.
        """
        super().__init__(
            length=length,
            particles=particles,
            interaction_range=interaction_range,
            speed=speed,
            noise_factor=base_noise,
            timestep=timestep,
            use_pbc=use_pbc,
            seed=seed,
        )
        self.alignment_weights = alignment_weights
        self.noise_weights = noise_weights
        self.particle_types = sorted({p.type for p in particles})  # sorted: set order is hash-seed dependent
        self.types = np.array([p.type for p in particles])
        type_index = {t: k for k, t in enumerate(self.particle_types)}
        self._type_ids = np.array([type_index[p.type] for p in particles], dtype=np.intp)

        # Validate weights matrices
        for type1 in self.particle_types:
            for type2 in self.particle_types:
                if (type1, type2) not in alignment_weights:
                    raise ValueError(f"Missing alignment weight for types {type1} and {type2}")
                if (type1, type2) not in noise_weights:
                    raise ValueError(f"Missing noise weight for types {type1} and {type2}")

    def _compute_weighted_velocity(self, particle: Particle, neighbors: List[Particle]) -> np.ndarray:
        """
        Compute the weighted average velocity considering particle types.

        Args:
            particle: The focal particle.
            neighbors: List of neighboring particles.

        Returns:
            The weighted average velocity vector.
        """
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

        return mean_velocity / norm if norm > 0 else _random_unit_vector(self.dim, self._rng)

    def _compute_effective_noise(self, particle: Particle, neighbors: List[Particle]) -> float:
        """
        Compute the effective noise based on particle types.

        Args:
            particle: The focal particle.
            neighbors: List of neighboring particles.

        Returns:
            The effective noise value.
        """
        if not neighbors:
            return self.mu * self.noise_weights[(particle.type, particle.type)]

        neighbor_weights = [
            self.noise_weights[(particle.type, neighbor.type)]
            for neighbor in neighbors
        ]

        weights = neighbor_weights + [self.noise_weights[(particle.type, particle.type)]]
        return self.mu * np.mean(weights)

    def _weight_matrix(self, weights: Dict[Tuple[str, str], float]) -> NDArray:
        """The weight dictionary as an ``(n_types, n_types)`` matrix indexed by type id."""
        n_types = len(self.particle_types)
        matrix = np.empty((n_types, n_types))
        for a, type1 in enumerate(self.particle_types):
            for b, type2 in enumerate(self.particle_types):
                matrix[a, b] = weights[(type1, type2)]
        return matrix

    def step(self):
        """Execute one synchronous time step of the simulation (see Vicsek.step).

        Vectorised: the alignment and noise weights are gathered per neighbour
        pair from ``(n_types, n_types)`` matrices, the weighted velocity sums and
        the mean noise weights are accumulated with ``bincount``, and the result
        goes through ``Vicsek._noisy_headings``. Equivalent to calling
        ``_compute_weighted_velocity`` and ``_compute_effective_noise`` for every
        particle in turn.
        """
        alignment = self._weight_matrix(self.alignment_weights)
        noise = self._weight_matrix(self.noise_weights)
        types = self._type_ids
        n = len(types)

        focal, neighbor, _ = self._cell_list.neighbor_pairs(self._positions)
        n_neighbors = np.bincount(focal, minlength=n)

        # w * v flips the neighbour's heading when w < 0 and weights it by |w|.
        pair_weight = alignment[types[focal], types[neighbor]]
        weighted = alignment[types, types][:, None] * self._velocities
        for k in range(self.dim):
            weighted[:, k] += np.bincount(focal, weights=pair_weight * self._velocities[neighbor, k], minlength=n)
        # A particle without neighbours keeps its own heading (whatever its self weight).
        lone = n_neighbors == 0
        weighted[lone] = self._velocities[lone]

        # Effective noise: base noise times the mean noise weight over self and neighbours.
        noise_sum = noise[types, types] + np.bincount(focal, weights=noise[types[focal], types[neighbor]], minlength=n)
        effective_noise = self.mu * noise_sum / (n_neighbors + 1)

        new_headings = self._noisy_headings(weighted, effective_noise[:, None])
        self._apply_updates(self.v * new_headings)

    def get_type_specific_order(self) -> Dict[str, float]:
        """
        Calculate the order parameters for each particle type.

        Returns:
            Dictionary mapping particle types to their respective order parameters.
        """
        order = {}
        for k, ptype in enumerate(self.particle_types):
            mean_velocity = self._velocities[self._type_ids == k].mean(axis=0)
            order[ptype] = min(1.0, float(np.linalg.norm(mean_velocity) / self.v))
        return order

    def compute_cross_correlations(self) -> Dict[Tuple[str, str], float]:
        """
        Compute the cross-correlations between mean velocities of particle types.

        Returns:
            Dictionary mapping pairs of particle types to their velocity cross-correlation.
        """
        mean_velocities = [self._velocities[self._type_ids == k].mean(axis=0)
                           for k in range(len(self.particle_types))]
        correlations = {}
        for i, type1 in enumerate(self.particle_types):
            for j in range(i, len(self.particle_types)):  # Upper triangle only
                type2 = self.particle_types[j]
                v1_mean, v2_mean = mean_velocities[i], mean_velocities[j]

                denominator = np.linalg.norm(v1_mean) * np.linalg.norm(v2_mean)
                if denominator == 0:  # a fully disordered type has no mean heading
                    correlations[(type1, type2)] = 0.0
                    continue

                # Clip so floating-point round-off cannot push a cosine outside [-1, 1].
                corr = float(np.clip(np.dot(v1_mean, v2_mean) / denominator, -1.0, 1.0))
                correlations[(type1, type2)] = corr

        return correlations

    def simulate_phase_transition(
            self,
            noise_values: NDArray,
            equilibration_steps: int = 400,
            measurement_steps: int = 300
    ) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray], Dict[Tuple[str, str], NDArray]]:
        """
        Simulate a phase transition by varying the noise parameter.

        Args:
            noise_values: Array of noise values to simulate.
            equilibration_steps: Number of steps to equilibrate at each noise value (default 400).
            measurement_steps: Number of steps to take measurements at each noise value (default 300).

        Returns:
            A tuple containing:
            - Global order parameters
            - Global order parameter fluctuations (susceptibility)
            - Dictionary of type-specific order parameters
            - Dictionary of type-specific order parameter fluctuations
            - Dictionary of cross-correlations between particle types
        """
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
                        int(np.count_nonzero(self.types == ptype)) *
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
