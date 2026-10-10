# Changelog

## Unreleased

`Vicsek.step()` is now a handful of NumPy array operations instead of a Python loop over particles: about 100x faster at N = 20000 in 2-D (see the Performance section of the README and the "After vectorisation" section of `docs/profiling.md` for the measurements). The physics is unchanged, and so is the public API.

On top of that, an optional C++ extension, `vicsek._kernel` (nanobind), runs the neighbour search and heading average with results bit-for-bit identical to the numpy path: a further 3.4x (2-D) to 4x (3-D) at N = 20000 single-threaded, up to 7.6x / 11.6x with 4 threads (README Performance, "After the kernel" in `docs/profiling.md`). The package now builds with scikit-build-core and ships binary wheels; without a C++ compiler it installs as before, pure Python.

### Changed

- Build backend: setuptools replaced by scikit-build-core with nanobind and a `CMakeLists.txt`. The package layout (`src/vicsek`), metadata and version are unchanged. `pip install .` builds the extension when a C++17 compiler is found and otherwise warns and installs the pure-Python package; `-Ccmake.define.VICSEK_KERNEL=OFF` skips it on purpose. The sdist excludes `scripts/` (80 MB of figures and data) and `.github/`.
- Struct-of-arrays state. A model owns `positions` and `velocities`, two contiguous float64 arrays of shape `(N, dim)`, and the step works on them directly. The `Particle` objects given to the constructor are kept (`model.particles` is the same list) and each is bound to its row, so `particle.position` / `particle.velocity` are views: reading one sees the current state and assigning to one (or `+=`) writes into the model's array. A `Particle` that has not been given to a model behaves exactly as before. The arrays are updated in place and never reallocated, so views stay valid; `model.positions = x` also assigns in place.
- `Vicsek.step`, `HeterogeneousVicsek.step` and `KernelizedVicsek.step` use `CellList.neighbor_pairs` / `CellList.mean_neighbor_velocity` (every neighbour pair at once) and `Vicsek._noisy_headings` (normalisation and noise for all particles at once). The random numbers are drawn in the same order as before, particle by particle (a fallback direction when the mean heading is zero, then the noise vector), so a seeded run gives the same trajectory as 0.2.0 up to round-off (about 1e-13 after a few steps; the sums are accumulated in a different order). No test had to change for this.
- `HeterogeneousVicsek` reads the particle types once, at construction, into a `types` array (and integer ids); changing `particle.type` afterwards no longer affects the dynamics or the per-type measurements. The alignment and noise weight dictionaries are still read on every step, so they can be edited between steps. `get_type_specific_order` and `compute_cross_correlations` compute from the arrays (same values, same keys).
- `CellList.cells` (the cell -> particles dictionary behind `get_neighbors`) is rebuilt lazily: the models call `CellList.invalidate()` after moving particles and the dictionary is rebuilt on its next access (`cells`, `get_neighbors`). `update()` still rebuilds it immediately. The hot path never touches it.
- `CellList` without periodic boundaries no longer reduces the neighbour-cell offsets for grids with one or two cells per axis (that reduction is only valid when offsets wrap); `get_neighbors` was already correct there, this only concerns the new array routine.

### Added

- C++ kernel `vicsek._kernel` with `neighbor_pairs(positions, box_length, radius, use_pbc=True, n_threads=1)` and `mean_neighbor_velocity(positions, velocities, box_length, radius, use_pbc=True, n_threads=1)`: counting sort into cells, 3^dim stencil (any dimension), minimum image, GIL released, optional `std::thread` parallelism. Same pairs in the same order and the same floating-point operations as the numpy path, so results are identical to the bit (compiled with `-ffp-contract=off`). `CellList.mean_neighbor_velocity` and `neighbor_pairs` use it when it is available, so all three models do.
- `vicsek.use_kernel(enabled=True, threads=None)` (persistent, or a context manager restoring the previous setting), `vicsek.kernel_available()`, `vicsek.kernel_enabled()`, `vicsek.kernel_threads()`; the environment variable `VICSEK_NO_KERNEL=1` makes numpy the default at import.
- Wheels: `publish.yml` builds CPython 3.10-3.12 wheels (3.12 as stable ABI) with cibuildwheel on Linux x86_64, macOS x86_64 and arm64 and Windows AMD64, tests each one, builds the sdist, and publishes on a release.
- CI runs the suite twice per Python version (kernel and numpy as the default path), checks the kernel was built, enforces 100% line coverage, and installs once without the extension.
- `tests/test_kernel.py`: 59 tests comparing the two paths with exact equality (random and property-based configurations in 1-4 D, PBC and open box, 1 and 2 cells per axis, radius above L/2 and beyond L, particles on cell edges, on 0 and L and outside the box, pairs at exactly the range, N = 0 and 1, thread counts; 25-step trajectories of all three models), plus the switch, the environment variable and the missing-extension fallback. 225 tests, 100% line coverage of the Python code on either default path.
- Benchmarks time the kernel path (`bench_cell_list.py` columns, `profile_step.py` section 5, `--size-3d`, `--kernel-threads`).
- `CellList.neighbor_pairs(positions=None) -> (i, j, distance)`: every ordered neighbour pair with its (minimum-image) distance, grouped by `i`; `vicsek.util.cell_list.neighbor_pairs(positions, box_length, interaction_range, use_pbc)` is the module-level function behind it. Without periodic boundaries particles outside `[0, L)` are excluded, as in `get_neighbors`.
- `CellList.mean_neighbor_velocity(positions, velocities) -> (mean, n_neighbors)`: mean over each particle and its neighbours, for all particles at once.
- `CellList.invalidate()`.
- `Vicsek.positions` and `Vicsek.velocities` (`(N, dim)` arrays, assignable in place); `HeterogeneousVicsek.types`.
- `KernelizedVicsek._smooth_weights(distances)`: the kernel on an array.
- Tests: the array routines against `get_neighbors` (2-D and 3-D, periodic and open, 1 to 10 cells per axis, parametrised and property-based); each model's vectorised step against the per-particle definition with a replayed random stream, to 1e-12, including the zero-mean fallback, seeded and global RNG; Particle view semantics. 166 tests, 100% line coverage.
- `benchmarks/profile_step.py` now profiles the phases of the vectorised step and checks it against the per-particle definition; `benchmarks/bench_cell_list.py` reports both the per-particle and the vectorised neighbour search.

### Unchanged

- Constructors, `step`, `run`, `equilibrate`, `order_parameter`, `order_parameter_evolution`, `simulate_phase_transition`, `visualize`, `animate`; `Particle` construction and attributes; `initialize_random_particles` (still returns `Particle` objects); `CellList(particles, ...)`, `build`, `update`, `get_neighbors`, `visualize`, `cells`; the per-particle methods `_compute_average_velocity`, `_apply_noise`, `_compute_weighted_velocity`, `_compute_effective_noise`, `_smooth_weight`, which remain the reference definitions the array code is tested against.
- Open boundaries: particles outside `[0, L)` take part in no neighbour search and nothing is wrapped.
- No new runtime dependencies (nanobind and scikit-build-core are build-time only).
- `scripts/validate_phase_transition.py --full` result (N = 4000, eta_c = 0.85) and figure added to the README.

## 0.2.0 (2026-10-10)

First release since 0.1.2. The simulation code was audited against the Vicsek model, every bug found was fixed with a regression test, and the repository gained CI, a benchmark, and a physics validation.

### Fixed

- `Vicsek.step` (and the heterogeneous and kernelized models) updated particles sequentially, so each particle saw neighbours that had already moved. Updates are now synchronous: all new headings are computed from the state at time t and applied together. Results no longer depend on particle order.
- `CellList.get_neighbors` returned each neighbour two or three times when there were two or fewer cells per axis under periodic boundaries.
- `Vicsek.simulate_phase_transition` never applied the noise values it swept, so every point measured the same noise.
- `initialize_random_particles` produced velocities with the wrong magnitude; every particle now has exactly the requested speed.
- `HeterogeneousVicsek.compute_cross_correlations` returned keys in a hash-seed-dependent order and could divide by zero.
- `KernelizedVicsek` hard-coded an interaction range of 1 in its smoothing kernel.
- `Vicsek.visualize` crashed when `legend` was not given, and `show_cells=True` called a `CellList.visualize` that did not exist.
- `order_parameter` could exceed 1 by round-off; it is now capped at 1.0.

### Added

- `seed` argument (an int or a `numpy.random.Generator`) on all three models for reproducible runs. Without it the global NumPy RNG is used, as before.
- `CellList.visualize`.
- `benchmarks/bench_cell_list.py`: cell list versus brute-force neighbour search, with a correctness check.
- `scripts/validate_phase_transition.py`: reproduces the order-disorder transition and estimates the critical noise.
- GitHub Actions CI on Python 3.10, 3.11 and 3.12 with coverage reporting. The test suite went from 35 tests at 72% line coverage to 128 tests at 99%.

### Changed

- Documentation now states that the noise is vectorial (Gregoire and Chate 2004), not the angular noise of Vicsek 1995, and that updates are synchronous.
- `Vicsek.__init__` rescales initial velocities to the model speed.
- Packaging: setuptools `src` layout, `dev` extra, corrected repository URL, Python classifiers.

## 0.1.2 (2024-12-11)

Bug-fix release of the original course-project library.
