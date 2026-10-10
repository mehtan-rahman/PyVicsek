# PyVicsek

[![CI](https://github.com/mehtan-rahman/PyVicsek/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mehtan-rahman/PyVicsek/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mehtan-rahman/PyVicsek/branch/main/graph/badge.svg)](https://codecov.io/gh/mehtan-rahman/PyVicsek)
[![PyPI](https://img.shields.io/pypi/v/pyvicsek)](https://pypi.org/project/pyvicsek/)

A small Python library for simulating the [Vicsek model](https://doi.org/10.1103/PhysRevLett.75.1226) of collective motion: self-propelled particles that align with the mean heading of their neighbours within a radius, subject to noise.

What it does:

- Periodic boundary conditions (default, `use_pbc=True`). With `use_pbc=False` the box is open: particles are not wrapped, and a particle that leaves `[0, L)` is simply excluded from neighbour searches. There are no reflecting walls.
- Any number of spatial dimensions (the examples below are 2-D and 3-D; animation is 2-D only).
- Heterogeneous particle types with per-pair alignment and noise weights (`HeterogeneousVicsek`).
- A cell list for neighbour finding, an order-parameter time series, an equilibration check, and a noise sweep (`simulate_phase_transition`).

Noise is applied as a random vector of length `noise_factor` added to the unit mean direction, then renormalised (vectorial noise, as in Gregoire and Chate 2004), not as the uniform angular noise of the 1995 paper. Updates are synchronous: every new heading is computed from the state at time t, then all are applied together. Models accept `seed=` (an int or a numpy Generator) for reproducible runs.

## Installation

```
pip install pyvicsek
```

The import name is `vicsek`. Requires Python 3.10+ and depends on numpy, matplotlib and tqdm.

Releases after 0.2.0 ship binary wheels on PyPI for CPython 3.10, 3.11 and 3.12 (3.12 wheels use the stable ABI and also install on later versions) on Linux x86_64, macOS x86_64 and arm64, and Windows x86_64. They include a small C++ extension, `vicsek._kernel`, that runs the neighbour search of every step several times faster than numpy (see Performance). Anywhere else pip builds from the sdist, which needs CMake (fetched automatically from PyPI) and a C++17 compiler; if no compiler is found the build prints a warning and installs the pure-Python package, which gives the same results on the numpy path. `vicsek.kernel_available()` says which you have.

For development (builds the extension in place with your compiler):

```
git clone https://github.com/mehtan-rahman/PyVicsek
cd PyVicsek
pip install -e ".[dev]"
```

To build without the extension: `pip install . -Ccmake.define.VICSEK_KERNEL=OFF`.

## Examples

A 2-D system at number density 4:

```python
import vicsek as vs

N, L, v = 400, 10.0, 0.03          # density rho = N / L^2 = 4

particles = vs.initialize_random_particles(
    n_particles=N, box_length=L, speed=v, n_dimensions=2, seed=0
)
model = vs.Vicsek(
    particles=particles, length=L, interaction_range=1.0, speed=v, noise_factor=0.5
)
model.run(200)                      # 200 time steps with a progress bar
print(f"order parameter: {model.order_parameter():.3f}")
```

A 3-D system, stepped by hand:

```python
import vicsek as vs

N, L, v = 500, 5.0, 0.03           # density rho = N / L^3 = 4

particles = vs.initialize_random_particles(
    n_particles=N, box_length=L, speed=v, n_dimensions=3, seed=0
)
model = vs.Vicsek(
    particles=particles, length=L, interaction_range=1.0, speed=v, noise_factor=0.5
)
for _ in range(100):
    model.step()
print(f"order parameter: {model.order_parameter():.3f}")
```

Two particle types that align within a type and anti-align across types:

```python
import vicsek as vs

L, v = 10.0, 0.03
particles = (
    vs.initialize_random_particles(200, L, v, n_dimensions=2, particle_type="A", seed=0)
    + vs.initialize_random_particles(200, L, v, n_dimensions=2, particle_type="B", seed=1)
)
model = vs.HeterogeneousVicsek(
    particles=particles, length=L, interaction_range=1.0, speed=v, base_noise=0.5,
    alignment_weights={("A", "A"): 1.0, ("B", "B"): 1.0, ("A", "B"): -1.0, ("B", "A"): -1.0},
    noise_weights={("A", "A"): 1.0, ("B", "B"): 1.0, ("A", "B"): 2.0, ("B", "A"): 2.0},
)
for _ in range(100):
    model.step()
print(model.get_type_specific_order())   # order parameter per type
```

To animate a 2-D run, `model.animate(frames=200).save("example.gif")`:

![](https://github.com/mehtan-rahman/PyVicsek/blob/main/scripts/homogeneous/example.gif)

More scripts are in `scripts/`, organised by system type.

## Validation

`scripts/validate_phase_transition.py` sweeps the noise amplitude from 0 to 5 at density 4, speed 0.03 and interaction radius 1, averages the order parameter phi = |sum v_i| / (N v) over the second half of each run and over 3 seeds, and estimates the critical noise as the phi = 0.5 crossing.

![Order parameter against noise amplitude for N = 400](scripts/phase_transition.png)

Quick mode (N = 400, 300 steps per run, 21 noise values, 3 seeds; 152 s on an Apple Silicon M5, Python 3.12, 2026-10-09) gives **eta_c = 1.01**. The order-disorder transition is reproduced, with phi falling to the 1/sqrt(N) = 0.05 noise floor on the disordered side. The value is lower than the 2 to 3 reported by Vicsek et al. for the same N, density and speed because this library uses vectorial rather than angular noise: here `noise_factor = 1` adds a random vector as long as the mean direction itself, so the two noise scales are not directly comparable. Full mode (`--full`: N = 4000, 2000 steps per run, same noise grid and seeds; 181 s on the same machine with the vectorised step) gives **eta_c = 0.85**, with the disordered-side floor at 1/sqrt(N) = 0.016. The downward shift from N = 400 to N = 4000 is the expected finite-size effect: the transition sharpens and moves to lower noise as N grows.

![Order parameter against noise amplitude for N = 4000](scripts/phase_transition_full.png)

## Performance

`Vicsek.step()` is array arithmetic on the model's `(N, dim)` position and velocity arrays: a cell-list neighbour search and heading average, one `normal(size=(N, dim))` noise draw, and the position update. There is no Python loop over particles; cost is O(N).

The neighbour search and heading average run in one of two implementations with bit-for-bit identical results (same pairs, same floating-point operations in the same order; the test suite asserts exact equality of whole trajectories):

- the **C++ kernel** `vicsek._kernel` (nanobind; counting sort into cells, 3^dim stencil, minimum image, GIL released), used whenever it is installed;
- the **numpy path** (`CellList.neighbor_pairs`: hash particles to cells, sort, expand each particle's 3^dim neighbouring cells into candidate pairs with `np.repeat`, keep the pairs within range, `bincount` sums), used when the kernel is not built, when `VICSEK_NO_KERNEL=1` is set before import, or after `vicsek.use_kernel(False)`.

```python
import vicsek as vs
vs.kernel_available(), vs.kernel_enabled()   # built? in use?
vs.use_kernel(True, threads=4)               # optional std::thread parallelism (default 1)
with vs.use_kernel(False):                   # numpy path inside the block only
    model.step()
```

Full `Vicsek.step()` at density 4, interaction radius 1, periodic boundaries, median of 5 x 20 steps (`benchmarks/profile_step.py --sizes 20000 --size-3d 20000`). Apple Silicon (M5, 10 cores), macOS 26, Python 3.12.15, numpy 2.5.3, Apple clang 21, 2026-10-10:

| N = 20000 | numpy path (ms) | kernel, 1 thread (ms) | kernel, 4 threads (ms) | speedup, 1 / 4 threads |
|---|---:|---:|---:|---:|
| 2-D | 17.4 | 5.2 | 2.3 | 3.4x / 7.6x |
| 3-D | 52.0 | 12.8 | 4.5 | 4.0x / 11.6x |

With the kernel the neighbour phase is 4.2 ms of the 5.2 ms 2-D step (0.21 us per particle, about 6 ns per candidate pair); normalisation, noise and the position update (about 1 ms, numpy) are the rest. `HeterogeneousVicsek` and `KernelizedVicsek` get their neighbour pairs from the kernel but weight them in numpy, so they gain less: 18.1 to 12.3 ms and 19.2 to 10.0 ms at N = 20000 in 2-D.

`benchmarks/bench_cell_list.py` compares the neighbour searches at fixed density 4 in 2-D, after checking that all of them return identical neighbour sets at N = 1000: a brute-force O(N^2) search (one vectorised numpy pass per particle, periodic minimum-image distances), the per-particle API `CellList.get_neighbors` called for every particle, and `CellList.neighbor_pairs` (every pair at once) on the numpy path and through the kernel. The speedup column is brute force over the faster `neighbor_pairs`. The last two columns are a full `Vicsek.step()` (median of 5) on each path. Same machine and date:

| N | L | brute force (s) | get_neighbors (s) | neighbor_pairs, numpy (s) | neighbor_pairs, kernel (s) | speedup | step, numpy (ms) | step, kernel (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 11.2 | 0.004 | 0.037 | 0.0002 | 0.0001 | 44x | - | - |
| 1000 | 15.8 | 0.013 | 0.081 | 0.0005 | 0.0002 | 58x | - | - |
| 2000 | 22.4 | 0.045 | 0.153 | 0.0008 | 0.0005 | 100x | - | - |
| 5000 | 35.4 | 0.255 | 0.375 | 0.0024 | 0.0012 | 217x | - | - |
| 10000 | 50.0 | 0.967 | 0.758 | 0.0048 | 0.0025 | 389x | 6.9 | 2.6 |
| 20000 | 70.7 | 3.776 | 1.521 | 0.0105 | 0.0050 | 750x | 14.3 | 5.2 |

For comparison, the 0.2.0 step (a Python loop over particles calling `get_neighbors`) measured 176 ms at N = 1000, 820 ms at N = 5000 and 3238 ms at N = 20000 in 2-D, and 2612 ms at N = 5000 in 3-D on the same machine (under more load than the runs above). `docs/profiling.md` has the phase-by-phase profiles: the original loop, the vectorised numpy step, and the kernel.

```
python benchmarks/bench_cell_list.py            # full table
python benchmarks/bench_cell_list.py --quick    # N = 500, 1000, 2000
python benchmarks/bench_cell_list.py --sizes 1000,4000 --step-sizes 4000
python benchmarks/profile_step.py --quick       # per-phase profile of step(), numpy and kernel
```

## Testing

```
pip install -e ".[dev]"
pytest --cov=vicsek                      # kernel path by default
VICSEK_NO_KERNEL=1 pytest --cov=vicsek   # numpy path by default
```

`tests/test_kernel.py` switches between the two paths itself and checks they agree exactly, so either run covers both. CI runs the suite on Python 3.10, 3.11 and 3.12, once with each path as the default, requires 100% line coverage of the Python code, installs once without the extension, and uploads coverage to Codecov.

## Roadmap

- Lyapunov analysis of phase dimensions (based on [this paper](https://doi.org/10.1103/PhysRevE.105.014213)).
- Geometry support for particles and boundaries using [shapely](https://github.com/shapely/shapely).
