import matplotlib as mpl
import matplotlib.pyplot as plt
import vicsek as vs


mpl.use('TkAgg')

N = 1024
L = 16
v = 0.03

particles_A = vs.initialize_random_particles(
    n_particles=round(N * 0.5),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='A'
)

particles_B = vs.initialize_random_particles(
    n_particles=round(N * 0.5),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='B'
)

particles = particles_A + particles_B

alignment_weights = {
    ('A', 'A'): 1.0,   # Normal alignment within groups
    ('B', 'B'): 1.0,
    ('A', 'B'): 1.5,   # Enhanced alignment between groups
    ('B', 'A'): 1.5,
}

noise_weights = {
    ('A', 'A'): 1.0,   # Normal noise within groups
    ('B', 'B'): 1.0,
    ('A', 'B'): 0.5,   # Reduced noise between groups
    ('B', 'A'): 0.5,
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
