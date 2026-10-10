"""Shared fixtures: deterministic RNG and a headless matplotlib backend."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # must precede the first pyplot import

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest
from matplotlib import pyplot as plt


SEED = 20251009


@pytest.fixture(autouse=True)
def seed_numpy():
    """Seed numpy's global RNG, which vicsek.util.linalg._random_unit_vector uses,
    so every test is reproducible run to run."""
    np.random.seed(SEED)
    yield


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened so figures do not leak between tests."""
    yield
    plt.close("all")


@pytest.fixture
def rng():
    """An independent, seeded Generator for building test configurations."""
    return np.random.default_rng(SEED)


def clone_rng_state(model):
    """Snapshot the random source a model draws from (its Generator, or numpy's
    global RNG when seed=None) so a reference computation can replay the same
    stream. Returns a callable producing a Generator-like object positioned at
    the snapshot."""
    if model._rng is not None:
        state = model._rng.bit_generator.state

        def replay():
            gen = np.random.default_rng()
            gen.bit_generator.state = state
            return gen
    else:
        state = np.random.get_state()

        def replay():
            np.random.set_state(state)
            return None
    return replay
