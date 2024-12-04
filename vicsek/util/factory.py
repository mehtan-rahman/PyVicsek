import numpy as np
from vicsek.models.particle import Particle


def initialize_random_particles(
        n_particles: int,
        box_length: float,
        speed: float,
        n_dimensions: int,
        particle_type: str = 'particle',
        seed: int = None,
):
    particles = []
    rng = np.random.default_rng(seed)

    for i in range(n_particles):
        position = rng.uniform(0, box_length, size=n_dimensions)
        velocity = speed * rng.uniform(-1, 1, size=n_dimensions)

        particles.append(Particle(
            position=position,
            velocity=velocity,
            name=f"p{i}",
            type=particle_type
        ))
    return particles
