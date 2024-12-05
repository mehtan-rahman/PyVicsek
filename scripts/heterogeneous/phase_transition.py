import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import vicsek as vs


mpl.use('TkAgg')

N = 1024
L = 16
v = 0.03

particles_A = vs.initialize_random_particles(
    n_particles=round(N/2),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='A'
)

particles_B = vs.initialize_random_particles(
    n_particles=round(N/2),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='B'
)

particles = particles_A + particles_B

alignment_weights = {
    ('A', 'A'): 1.0,
    ('B', 'B'): 1.0,
    ('A', 'B'): -0.5,
    ('B', 'A'): -0.5,
}

noise_weights = {
    ('A', 'A'): 1.0,
    ('B', 'B'): 1.0,
    ('A', 'B'): 2.0,
    ('B', 'A'): 2.0,
}

legend = {
    'A': 'red',
    'B': 'blue',
}


vicsek = vs.HeterogeneousVicsek(
    particles=particles,
    length=L,
    interaction_range=1.0,
    speed=v,
    base_noise=0.5,
    alignment_weights=alignment_weights,
    noise_weights=noise_weights
)

anim = vicsek.animate(frames=200, legend=legend)
anim.save(filename='test3.gif')
