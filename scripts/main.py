"""
Simulating a 2D / 3D particle system using the Vicsek model by generating particles with random positions and velocities and assigning them to a cell list for efficient neighbor searching

Modules Used:
    - numpy: For numerical operations.
    - matplotlib.pyplot: For visualization.
    - shapely.Point: For geometric representations of particle positions.
    - scipy.spatial.transform.Rotation: For generating random rotations in 2D and 3D.
    - vicsek.models.particle.Particle: Custom particle class.
    - vicsek.util.cell_list.CellList: Custom CellList class for spatial partitioning.

Simulation Parameters:
    n_particles (int): Number of particles in the simulation.
    length (float): Length of the simulation box (assumed square or cubic).
    dim (int): Dimensionality of the simulation (2D or 3D).
    v (NDArray): Default velocity vector for particles.

Generated Objects:
    particles (List[Particle]): A list of Particle objects with random positions and velocities.
    cell_list (CellList): A CellList object to manage particles and their spatial organization.

Visualization:
    - Shows the particle positions in a 2D simulation box.
    - Optionally displays the cell grid, cell labels, and particle labels.
"""



import numpy as np
import matplotlib.pyplot as plt
from shapely import Point
from scipy.spatial.transform import Rotation

from vicsek.models.particle import Particle
from vicsek.util.cell_list import CellList


n_particles = 1024
length = 16
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


cell_list = CellList(particles, box_length=length, interaction_range=10, use_pbc=True)
cell_list.build()
cell_list.visualize(show_cell_grid=True, label_cells=True, label_particles=False)
plt.show()
