import numpy as np
from matplotlib import pyplot as plt
from shapely import Geometry
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class Particle:
    def __init__(
        self,
        position: NDArray,
        velocity: NDArray,
        rotation: Rotation,
        geometry: Geometry,
        type: str,
        name: str
    ):
        self._position = np.array(position) if not isinstance(position, np.ndarray) else position
        self._velocity = np.array(velocity) if not isinstance(velocity, np.ndarray) else velocity
        self.rotation = rotation
        self.geometry = geometry
        self.type = type
        self.name = name

    def __sub__(self, other):
        return np.linalg.norm(other.position - self.position)

    def __repr__(self):
        return f"Particle(name={self.name}, pos={self.position})"

    @property
    def direction(self):
        return np.linalg.norm(self._velocity)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._position = new_position

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity):
        self._velocity = new_velocity

    def visualize(
            self,
            ax: plt.Axes = None,
            label: bool = None,
            show_velocity: bool = False,
            color: str = 'dimgrey',
            size: int = 50,
            alpha: float = 1.0
    ) -> plt.Axes:
        """Visualize a particle with optional velocity arrow."""
        if ax is None:
            fig, ax = plt.subplots()

        # Plot particle position
        ax.scatter(self.position[0], self.position[1], c=color, s=size, alpha=alpha)
        if label:
            ax.annotate(self.name, (self.position[0], self.position[1]))

        # Optionally show velocity vector
        if show_velocity and np.any(self.velocity):
            ax.arrow(
                self.position[0],
                self.position[1],
                self.velocity[0],
                self.velocity[1],
                head_width=0.1,
                head_length=0.2,
                fc=color,
                ec=color,
                alpha=alpha
            )

        return ax