import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


class Particle:
    """
    A class representing a particle in a simulation.

    A particle can be constructed on its own and used freely; its ``position``
    and ``velocity`` are then ordinary arrays it owns. Once the particle is
    handed to a model (``Vicsek`` and its subclasses) it becomes a *view*: the
    model stores every position and velocity in two contiguous ``(N, dim)``
    arrays (``model.positions`` and ``model.velocities``) and binds each
    particle to its own row, so ``particle.position`` reads the model's array
    and ``particle.position = x`` or ``particle.position += dx`` writes into
    it. There is a single copy of the state and the public ``Particle`` API is
    unchanged. A particle belongs to the model that most recently received it.

    Attributes:
        name (str): Name of the particle.
        type (str): Type of the particle.
    """
    __slots__ = ('_position', '_velocity', 'name', 'type', '_bound')

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
        self._bound = False

        self.name = name
        self.type = type

    def _bind(self, position_view: NDArray[float], velocity_view: NDArray[float]) -> None:
        """Make this particle a view onto one row of a model's state arrays.

        Called by the model; ``position_view`` and ``velocity_view`` are row
        views of the model's ``(N, dim)`` arrays. From then on the property
        setters write into those rows instead of rebinding the attributes.
        """
        self._position = position_view
        self._velocity = velocity_view
        self._bound = True

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
            The current position (a view of the model's array once the particle
            belongs to a model).
        """
        return self._position

    @position.setter
    def position(self, new_position: NDArray[float]):
        """
        Set a new position for the particle.

        Args:
            new_position: The new position.
        """
        if self._bound:
            self._position[...] = new_position
        else:
            self._position = new_position

    @property
    def velocity(self) -> NDArray[float]:
        """
        Get the current velocity of the particle.

        Returns:
            The current velocity (a view of the model's array once the particle
            belongs to a model).
        """
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity: NDArray[float]):
        """
        Set a new velocity for the particle.

        Args:
            new_velocity: The new velocity.
        """
        if self._bound:
            self._velocity[...] = new_velocity
        else:
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
