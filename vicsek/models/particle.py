import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


class Particle:
    """
    A class representing a particle in a simulation.

    Attributes:
        name (str): Name of the particle.
        type (str): Type of the particle.
    """
    __slots__ = ('_position', '_velocity', 'name', 'type')

    def __init__(
        self,
        position: NDArray[float],
        velocity: NDArray[float],
        name: str,
        type: str
    ):
        """
        Initialize a Particle instance.

        Args:
            position: Initial position of the particle.
            velocity: Initial velocity of the particle.
            name: Name of the particle.
            type: Type of the particle.
        """
        self._position = np.array(position) if not isinstance(position, np.ndarray) else position
        self._velocity = np.array(velocity) if not isinstance(velocity, np.ndarray) else velocity

        self.name = name
        self.type = type

    def __sub__(self, other: 'Particle') -> float:
        """
        Compute the Euclidean distance between this particle and another.

        Args:
            other: The other particle.

        Returns:
            The Euclidean distance between the particles.
        """
        return np.linalg.norm(other.position - self.position)

    def __repr__(self) -> str:
        """
        Return a string representation of the Particle.

        Returns:
            String representation of the Particle.
        """
        return f"Particle(name={self.name}, pos={self.position})"

    @property
    def position(self) -> NDArray[float]:
        """
        Get the current position of the particle.

        Returns:
            The current position.
        """
        return self._position

    @position.setter
    def position(self, new_position: NDArray[float]):
        """
        Set a new position for the particle.

        Args:
            new_position: The new position.
        """
        self._position = new_position

    @property
    def velocity(self) -> NDArray[float]:
        """
        Get the current velocity of the particle.

        Returns:
            The current velocity.
        """
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity: NDArray[float]):
        """
        Set a new velocity for the particle.

        Args:
            new_velocity: The new velocity.
        """
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
        """
        Visualize the particle with an optional velocity arrow.

        Args:
            ax: Matplotlib axes to plot on. If None, a new figure is created.
            label: Whether to label the particle with its name.
            show_velocity: Whether to show the velocity arrow.
            color: Color of the particle marker and arrow.
            size: Size of the particle marker.
            alpha: Opacity of the particle marker and arrow.

        Returns:
            The matplotlib axes with the particle plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(self.position[0], self.position[1], c=color, s=size, alpha=alpha)
        if label:
            ax.annotate(self.name, (self.position[0], self.position[1]))

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