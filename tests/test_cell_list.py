import pytest
import numpy as np
from vicsek.models.particle import Particle
from src.vicsek.util.cell_list import CellList


@pytest.fixture
def particles():
    return [
        Particle(position=np.array([0.5, 0.5]), velocity=np.array([0, 0]), name="A", type="A"),
        Particle(position=np.array([0.6, 0.6]), velocity=np.array([0, 0]), name="B", type="B"),
        Particle(position=np.array([1.5, 1.5]), velocity=np.array([0, 0]), name="C", type="C"),
        Particle(position=np.array([5.0, 5.0]), velocity=np.array([0, 0]), name="D", type="D")
    ]


def test_initialize(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    assert len(cell_list.particles) == 4
    assert cell_list.box_length == 10.0
    assert cell_list.interaction_range == 1.0
    assert cell_list.n_cells == 10
    assert cell_list.cell_size == 1.0


def test_build(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    assert len(cell_list.cells) == 100
    assert len(cell_list.cells[(9, 9)]) == 2
    assert len(cell_list.cells[(8, 8)]) == 1


def test_update(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    particles[0].position = np.array([1.5, 1.5])
    cell_list.update()
    assert len(cell_list.cells[(9, 9)]) == 1
    assert len(cell_list.cells[(8, 8)]) == 2


def test_neighbors_pbc(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0, use_pbc=True)
    cell_list.build()
    neighbors = cell_list.get_neighbors(particles[0])
    assert len(neighbors) == 1
    assert neighbors[0].name == "B"


def test_neighbors_no_pbc(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0, use_pbc=False)
    cell_list.build()
    neighbors = cell_list.get_neighbors(particles[0])
    assert len(neighbors) == 1
    assert neighbors[0].name == "B"

    particles[0].position = np.array([-0.5, -0.5])
    cell_list.update()
    assert len(cell_list.get_neighbors(particles[0])) == 0


def test_visualize(particles):
    cell_list = CellList(particles, box_length=10.0, interaction_range=1.0)
    cell_list.build()
    ax = cell_list.visualize(show_cell_grid=True, label_cells=True, label_particles=True)
    assert ax.has_data()
