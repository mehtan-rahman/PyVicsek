import numpy as np
from vicsek.models.particle import Particle


def initialize_random_particles(n_particles: int, box_length: float, speed: float, n_dimensions: int):
    particles = []

    for i in range(n_particles):
        position = np.random.uniform(0, box_length, size=n_dimensions)
        velocity = speed * np.random.uniform(-1, 1, size=n_dimensions)

        particles.append(Particle(
            position=position,
            velocity=velocity,
            name=f"p{i}",
            type="standard"
        ))
    return particles
