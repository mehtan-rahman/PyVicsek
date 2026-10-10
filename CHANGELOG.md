# Changelog

## Unreleased

`Vicsek.step()` is now a handful of NumPy array operations instead of a Python loop over particles: about 100x faster at N = 20000 in 2-D (see the Performance section of the README and the "After vectorisation" section of `docs/profiling.md` for the measurements). The physics is unchanged, and so is the public API.

### Changed

- Struct-of-arrays state. A model owns `positions` and `velocities`, two contiguous float64 arrays of shape `(N, dim)`, and the step works on them directly. The `Particle` objects given to the constructor are kept (`model.particles` is the same list) and each is bound to its row, so `particle.position` / `particle.velocity` are views: reading one sees the current state and assigning to one (or `+=`) writes into the model's array. A `Particle` that has not been given to a model behaves exactly as before. The arrays are updated in place and never reallocated, so views stay valid; `model.positions = x` also assigns in place.
- `Vicsek.step`, `HeterogeneousVicsek.step` and `KernelizedVicsek.step` use `CellList.neighbor_pairs` / `CellList.mean_neighbor_velocity` (every neighbour pair at once) and `Vicsek._noisy_headings` (normalisation and noise for all particles at once). The random numbers are drawn in the same order as before, particle by particle (a fallback direction when the mean heading is zero, then the noise vector), so a seeded run gives the same trajectory as 0.2.0 up to round-off (about 1e-13 after a few steps; the sums are accumulated in a different order). No test had to change for this.
- `HeterogeneousVicsek` reads the particle types once, at construction, into a `types` array (and integer ids); changing `particle.type` afterwards no longer affects the dynamics or the per-type measurements. The alignment and noise weight dictionaries are still read on every step, so they can be edited between steps. `get_type_specific_order` and `compute_cross_correlations` compute from the arrays (same values, same keys).
- `CellList.cells` (the cell -> particles dictionary behind `get_neighbors`) is rebuilt lazily: the models call `CellList.invalidate()` after moving particles and the dictionary is rebuilt on its next access (`cells`, `get_neighbors`). `update()` still rebuilds it immediately. The hot path never touches it.
- `CellList` without periodic boundaries no longer reduces the neighbour-cell offsets for grids with one or two cells per axis (that reduction is only valid when offsets wrap); `get_neighbors` was already correct there, this only concerns the new array routine.

### Added

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
- No new dependencies.
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
