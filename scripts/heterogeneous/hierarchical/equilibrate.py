import matplotlib as mpl
import matplotlib.pyplot as plt
import vicsek as vs


mpl.use('TkAgg')

N = 1024
L = 16
v = 0.03

particles_A = vs.initialize_random_particles(
    n_particles=round(N * 0.1),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='leader'
)

particles_B = vs.initialize_random_particles(
    n_particles=round(N * 0.9),
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='follower'
)

particles = particles_A + particles_B

alignment_weights = {
    ('leader', 'leader'): -2.0,   # Leaders align against each other
    ('follower', 'follower'): 0.5,   # Followers align weakly with each other
    ('leader', 'follower'): 2.0,   # Leaders strongly influence followers
    ('follower', 'leader'): 0.2,   # Followers weakly influence leaders
}

noise_weights = {
    ('leader', 'leader'): 0.5,   # Leaders have very low noise
    ('follower', 'follower'): 1.0,   # Followers have normal noise
    ('leader', 'follower'): 0.5,   # Moderate noise when leading
    ('follower', 'leader'): 2.0,   # High noise when following
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

anim = vicsek.animate(200)
anim.save('animation_hierarchical.gif')
