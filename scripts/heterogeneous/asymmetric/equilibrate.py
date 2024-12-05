import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import vicsek as vs


mpl.use('TkAgg')

N = 1024
L = 16
v = 0.03

particles_A = vs.initialize_random_particles(
    n_particles=round(N * 0.25),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='predator'
)

particles_B = vs.initialize_random_particles(
    n_particles=round(N * 0.75),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='prey'
)

particles = particles_A + particles_B

alignment_weights = {
    ('predator', 'predator'): 1.0,  # predators align normally with each other
    ('prey', 'prey'): 1.0,  # prey align normally with each other
    ('predator', 'prey'): 2.0,  # predator strongly attracted to prey
    ('prey', 'predator'): -2.0,  # prey strongly repelled by predator
}

noise_matrix = np.array([
    [1, 1],
    [1, 1]
])

noise_weights = {
    ('predator', 'predator'): noise_matrix[0, 0],  # low noise for effective pursuit/evasion
    ('prey', 'prey'): noise_matrix[1, 1],
    ('predator', 'prey'): noise_matrix[0, 1],
    ('prey', 'predator'): noise_matrix[1, 0],
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

order_params = vicsek.order_parameter_evolution()

fig, ax = plt.subplots()
ax.plot(order_params, color='black')

ax.set_xlabel('Time Steps', fontsize=12)
ax.set_ylabel('Order Parameter φ', fontsize=12)
ax.set_title('Order Parameter Evolution Over Time', fontsize=14)
ax.grid(True, alpha=0.3)
plt.legend()

fig.tight_layout()
plt.savefig('equilibration.png')
plt.show()


