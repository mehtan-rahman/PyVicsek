import numpy as np
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


noise_values = np.arange(0, 2, 0.1)
(global_order, global_fluctuations,
 type_orders, type_fluctuations, cross_correlations) = vicsek.simulate_phase_transition(noise_values)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

# Plot order parameters
ax1.plot(noise_values, global_order, 'k-', label='Global', linewidth=2)
for ptype, orders in type_orders.items():
    ax1.plot(noise_values, orders, f'{legend[ptype]}-', label=f'Type {ptype}', linewidth=2)

ax1.set_xlabel('Noise')
ax1.set_ylabel('Order Parameter')
ax1.set_title('Order Parameters vs Noise')
ax1.legend()
ax1.grid(True)

# Plot fluctuations
ax2.plot(noise_values, global_fluctuations, 'k-', label='Global', linewidth=2)
for ptype, fluct in type_fluctuations.items():
    ax2.plot(noise_values, fluct, f'{legend[ptype]}-', label=f'Type {ptype}', linewidth=2)

ax2.set_xlabel('Noise')
ax2.set_ylabel('Susceptibility')
ax2.set_title('Fluctuations vs Noise')
ax2.legend()
ax2.grid(True)

# Plot cross-correlations
for pair, corr in cross_correlations.items():
    ax3.plot(noise_values, corr, '-',
            label=f'Types {pair[0]}-{pair[1]}',
            color='purple' if pair[0] != pair[1] else legend[pair[0]])

ax3.set_xlabel('Noise')
ax3.set_ylabel('Cross-correlation')
ax3.set_title('Velocity Cross-correlations vs Noise')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
