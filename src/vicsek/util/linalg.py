from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _random_unit_vector(n_dimensions: int, rng: Optional[np.random.Generator] = None) -> NDArray:
    """
    Draw a uniformly distributed unit vector.

    Args:
        n_dimensions: Dimension of the vector.
        rng: Generator to draw from. When None, numpy's global RNG is used
            (seed it with ``np.random.seed`` for reproducibility).
    """
    source = rng if rng is not None else np.random
    vector = source.normal(0, 1, n_dimensions)
    return vector / np.linalg.norm(vector)
