import numpy as np
from numpy.typing import NDArray


def _random_unit_vector(n_dimensions: int) -> NDArray:
    vector = np.random.normal(0, 1, n_dimensions)
    return vector / np.linalg.norm(vector)
