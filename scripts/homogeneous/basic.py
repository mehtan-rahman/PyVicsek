import vicsek as vs

N = 1024
L = 16
v = 0.03

particles = vs.initialize_random_particles(
    n_particles=N,
    box_length=L,
    speed=v,
    n_dimensions=2,
    particle_type='standard'
)

vicsek = vs.Vicsek(
    particles=particles,
    length=L,
    interaction_range=1.0,
    speed=v,
    noise_factor=1.0
)

anim = vicsek.animate(frames=200)
anim.save(filename='example.gif')
