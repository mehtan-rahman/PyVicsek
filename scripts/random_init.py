import numpy as np
import matplotlib.pyplot as plt
from shapely import Point
from scipy.spatial.transform import Rotation

from vicsek.models.particle import Particle
from vicsek.models.vicsek import Vicsek

import matplotlib
matplotlib.use('TkAgg')


n_particles = 150
length = 25
dim = 2
v = np.array([1, 0])

particles = []

for i in range(n_particles):
    position = np.random.uniform(low=0, high=length, size=dim)
    geometry = Point(position)

    if dim == 2:
        theta = np.random.uniform(0, 2 * np.pi)
        rotation = Rotation.from_euler('z', theta)
        rotation_matrix = np.delete(np.delete(rotation.as_matrix(), 2, 0), 2, 1)
    else:
        rotation = Rotation.random()
        rotation_matrix = rotation.as_matrix()

    velocity = v @ rotation_matrix

    particle = Particle(
        position=position,
        velocity=velocity,
        type='particle',
        name=f'{i}'
    )

    particles.append(particle)

vicsek = Vicsek(
    length=length,
    particles=particles,
    interaction_range=10,
    speed=5,
    noise_factor=1,
    timestep=0.1,
    use_pbc=True
)

anim = vicsek.animate(frames=200)
plt.show()
