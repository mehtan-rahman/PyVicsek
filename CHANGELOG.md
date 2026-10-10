# Changelog

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
