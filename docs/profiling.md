# Profiling `Vicsek.step()` and where C++ pays off

Measured with `benchmarks/profile_step.py` on an Apple M5 (Python 3.12.15, numpy 2.5.3) while a 4-core sweep was running; every timing is the median of 5 repeats (spread 2-10 %). Density 4, radius 1, periodic boundaries, 12.6 mean neighbours in 2-D (16.7 in 3-D).

## Where the time goes

Per-phase wall clock from timers wrapped around the library's own methods, no cProfile overhead:

| phase (function) | N=1000 2-D | N=5000 2-D | N=20000 2-D | N=5000 3-D |
|---|---:|---:|---:|---:|
| neighbour query (`CellList.get_neighbors`) | 89.8 % | 88.8 % | 88.8 % | 96.1 % |
| heading + noise (`_compute_average_velocity`, `_apply_noise`) | 5.9 % | 6.3 % | 6.2 % | 2.1 % |
| step loop rest (list concat, velocity gather, `np.array`) | 2.4 % | 2.8 % | 2.8 % | 1.1 % |
| cell-list rebuild (`CellList.update`) | 1.0 % | 1.2 % | 1.2 % | 0.4 % |
| position update + wrap (`_apply_updates`) | 0.8 % | 0.9 % | 0.9 % | 0.3 % |
| **step total (ms)** | **154** | **747** | **3087** | **2564** |

The neighbour query is the whole story. Inside it, `_minimum_image_distance` (cell_list.py:179-192) runs once per candidate from the list comprehension at cell_list.py:166-170: 37.0 candidates per particle in 2-D (9 cells x 4.1 particles), 135 in 3-D. Each call costs 3.5 us unprofiled: three numpy operations on a 2-element array (`-`, `round(/)`, `norm`) whose arithmetic is ~10 ns; `np.linalg.norm` on a 2-vector alone is 0.9 us.

cProfile over 20 steps at N=20000: 723 function calls per particle per step (2440 in 3-D). Internal time is 53 % vicsek Python code, 25 % numpy's Python wrappers (`norm`, `round`, `_wrapfunc`), 22 % C-level (`ndarray.dot`, `ndarray.round`, `numpy.array`), and the C-level share is itself per-call overhead on 2-element arrays. Effectively the whole step is interpreter and dispatch cost; floating-point work is below 1 %.

## Per-particle cost and scaling

| config | us/particle | candidates/particle | calls/particle |
|---|---:|---:|---:|
| N=1000 2-D | 154 | 40.3 | 778 |
| N=5000 2-D | 149 | 37.1 | 725 |
| N=20000 2-D | 154 | 37.0 | 723 |
| N=5000 3-D | 513 | 135.4 | 2440 |

Cost is O(N) and flat at ~150 us per particle in 2-D; the cell list does its job. 3-D costs 3.3x more per particle because the 27-cell stencil triples the candidates.

## Reference points at N=20000 (2-D)

| implementation | ms/step | us/particle | speedup vs current |
|---|---:|---:|---:|
| current library | 3087 | 154 | 1x |
| pure-numpy brute force (bench_cell_list.py, neighbour sets only, O(N^2)) | 6817 | 341 | 0.45x |
| vectorised numpy cell list (this script, full step) | 51.4 | 2.57 | 60x |
| C++ reference, single thread, -O2 (standalone, same algorithm) | 8.6 | 0.43 | 359x |

Brute force beats the library's query only at N=1000 (24 ms vs 139 ms).

The C++ reference is a plain struct-of-arrays implementation: counting sort into cells, 3^D stencil, minimum image with `std::round`, mean heading, `std::normal_distribution`, wrap. It gives 0.41-0.43 us/particle in 2-D and 1.08 in 3-D (5.4 ms/step at N=5000, 474x vs the library). Hoisting the periodic shift per neighbour cell and using float32 roughly halves that; 4 threads over the particle loop (embarrassingly parallel, GIL released) brings N=20000 under 3 ms. The kernel's ceiling is 300-1000x over today and 6-20x over the best numpy.

## Vectorised numpy: does it get most of the gain?

Yes: 60x of a possible ~360x, with no build step. `vectorized_mean_heading` in `benchmarks/profile_step.py`:

1. cell index `(P / cell_size).astype(int) % n_cells` (same hashing as `_hash_position`), `argsort`, `bincount` + `cumsum` for cell starts;
2. (N, 3^D) neighbour-cell ids for all offsets at once, candidate list expanded with `np.repeat` / `arange` (740 k candidates at N=20000);
3. minimum-image distance on the candidate array, mask `dist <= r`, `bincount(weights=)` per axis, normalise;
4. noise as one `rng.normal(size=(N, dim))` draw.

Verification against the library (seed 123): identical neighbour counts for every particle, max heading difference 8.8e-15, and after 3 seeded steps max velocity and position difference 8.9e-13 (`Generator.normal(size=(N, dim))` yields the same doubles as N sequential `normal(size=dim)` calls, so the noise stream matches).

| sub-phase | ms/step | % |
|---|---:|---:|
| bin + sort | 1.5 | 3 |
| gather candidates (repeat/arange) | 9.6 | 19 |
| distance + masked heading sum | 37.9 | 74 |
| noise + position update | 0.9 | 2 |
| **total** | **51.4** | 100 |

What remains is memory traffic: ~15 temporaries of 740 k x 2 doubles per step (~60 MB), 70 ns per candidate against the C++ reference's ~10 ns in cache. That 6x gap is the kernel's case.

## Recommended plan

**(a) Vectorised numpy path first.** Replace `Vicsek.step` (vicsek.py:116-138) and `_apply_updates` (140-152) with the array version operating on `(N, dim)` position and velocity arrays. `CellList.update`/`get_neighbors` stay for the API and tests but are no longer called from `step`. Gate on `n_cells >= 3` and `use_pbc`, falling back to the current path otherwise (it deduplicates wrapped cells and skips out-of-box particles). Expected: 3087 ms to ~60 ms at N=20000, and it lands the array layout the kernel needs.

**(b) nanobind kernel for the neighbour loop + heading sum.** One function, arrays in, arrays out, no Particle objects crossing the boundary:

```
mean_heading(positions: f64[N, D], velocities: f64[N, D], length: float,
             radius: float, n_cells: int, pbc: bool) -> (heading: f64[N, D], n_neighbors: i32[N])
```

Internals: counting sort, 3^D stencil, minimum image, sum, normalise; `nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig>` views, GIL released, `std::thread` over `i`. A second overload takes `type_id: i32[N]`, `alignment: f64[T, T]`, `noise: f64[T, T]` and returns `(heading, effective_noise)`, replacing `HeterogeneousVicsek._compute_weighted_velocity` and `_compute_effective_noise` (heterogeneous.py:87-140), whose per-neighbour dict lookups have the same shape. Zero-norm headings return a flag so Python draws the random direction and the RNG stream stays identical.

Build: switch `pyproject.toml` from setuptools to `scikit-build-core` with `nanobind` in `build-system.requires`, a 20-line `CMakeLists.txt` (`nanobind_add_module(_kernel src/vicsek/_kernel.cpp)`), `cibuildwheel` for macOS arm64/x86_64, Linux x86_64 and Windows wheels, and `try: from vicsek._kernel import mean_heading except ImportError:` falling back to the numpy path so the sdist works without a compiler.

**Keep in Python:** noise and renormalisation (0.9 ms at N=20000), position update and wrap, order parameter, equilibration, visualisation, the Particle API. They are vectorised or off the hot path; moving them buys nothing and complicates reproducibility.

## The Particle object design

Per-object layout is not the direct bottleneck: `Particle.position` property calls are 1.8 % of profiled time and the `np.array([p.velocity ...])` gathers 2 %. But it blocks the fix: the numpy path and the kernel both need contiguous `(N, dim)` arrays, and gathering them from 20000 objects costs 6.0 ms plus 5.4 ms to scatter back, 22 % of the vectorised step.

A struct-of-arrays refactor stores `positions`, `velocities` as `(N, dim)` arrays and `type_id` as `int32[N]` on the model, and makes `Particle` a handle whose `position`/`velocity` properties return row `i` as a view (`p.position += ...` still works). `__slots__` stays; `_position`/`_velocity` go. The public API is unchanged, `CellList` receives arrays, and the gather/scatter cost disappears. Do this as part of step (a).

## After vectorisation

Plan (a) is done: `Vicsek.step` (and the heterogeneous and kernelized steps) run on `(N, dim)` arrays through `CellList.neighbor_pairs` / `mean_neighbor_velocity` and `Vicsek._noisy_headings`, and the `Particle` objects are row views of the model's arrays. Measured with the updated `benchmarks/profile_step.py` on the same machine (Python 3.12.15, numpy 2.5.3, the 4-core sweep still running), medians of 5 repeats x 20 steps. The script also checks the array step against the per-particle definition (`get_neighbors` + `_compute_average_velocity` + `_apply_noise` with a replayed random stream): PASS at every size, zero neighbour-count mismatches, max velocity and position difference after 3 seeded steps 9.2e-14 at N = 20000 (1.8e-15 in 3-D).

### Phase table, N = 20000, 2-D

| phase | ms/step | % | us/particle |
|---|---:|---:|---:|
| bin + sort particles into cells (`_bin_particles`) | 1.32 | 5.0 | 0.07 |
| expand candidate pairs (`_candidate_pairs`) | 9.51 | 35.9 | 0.48 |
| pair distances, minimum image (`_pair_distances`) | 6.17 | 23.3 | 0.31 |
| mask pairs + heading sum (rest of `mean_neighbor_velocity`) | 8.18 | 30.9 | 0.41 |
| normalise + noise (`_noisy_headings`) | 1.43 | 5.4 | 0.07 |
| position update + wrap (`_apply_updates`) | 0.14 | 0.5 | 0.01 |
| **total step** | **26.45** | 100 | **1.32** |

N = 5000, 3-D: 16.5 ms (3.30 us/particle); candidates 34 %, distances 38 %, heading sum 23 %.

### Before and after, same machine, same hour (median of 5 steps)

| config | 0.2.0 step (ms) | vectorised step (ms) | speedup |
|---|---:|---:|---:|
| N=1000 2-D | 176 | 1.2 | 147x |
| N=5000 2-D | 820 | 6.2 | 132x |
| N=20000 2-D | 3238 | 26.0 | 125x |
| N=5000 3-D | 2612 | 16.7 | 156x |

The 0.2.0 numbers agree with the table at the top (154 / 747 / 3087 / 2564 ms), so the machine load was comparable. cProfile now counts about 3500 function calls per step in total (0.17 per particle at N = 1000, 0.01 at N = 20000) against 723 per particle before; the step is entirely inside numpy kernels.

### What changed against the prototype

The prototype measured 51.4 ms at N = 20000; the library step is 26 ms. Two differences: pair distances are computed one axis at a time (a 1-D column gather is several times cheaper than gathering `(k, 2)` rows, and the temporaries are half the size: 37.9 ms to 6.2 ms for this phase), and the pairs within range are selected with `flatnonzero` + `take` instead of boolean indexing three arrays (8 ms to 2 ms). The remaining time is spread over the candidate expansion (four `np.repeat` calls over 757 k candidates), the distance arithmetic and the `bincount` sums; all of it is memory traffic over the candidate arrays, roughly 35 ns per candidate. The C++ reference's 8.6 ms (10 ns per candidate, in cache) remains the next step, now 3x rather than 6x away, and the array layout it needs is in place: `positions`, `velocities` and (for the heterogeneous model) integer type ids are contiguous arrays on the model.

## After the kernel

Plan (b) is done in the smallest form that removes the Python-side neighbour phase: `src/vicsek/_kernel.cpp`, built as `vicsek._kernel` with nanobind (stable ABI, `NB_STATIC`) through scikit-build-core, exports two functions:

```
neighbor_pairs(positions: f64[N, D], box_length, radius, use_pbc=True, n_threads=1)
    -> (i: i64[P], j: i64[P], distance: f64[P])
mean_neighbor_velocity(positions: f64[N, D], velocities: f64[N, D], box_length, radius, use_pbc=True, n_threads=1)
    -> (mean: f64[N, D], n_neighbors: i64[N])
```

`CellList.mean_neighbor_velocity` (which `Vicsek.step` calls) and the module-level `neighbor_pairs` (which `HeterogeneousVicsek.step` and `KernelizedVicsek.step` call) dispatch to them through `vicsek.util.backend`; noise, normalisation, the position update and every per-pair weighting stay in numpy, so the random stream is untouched.

Differences from the plan above, and why:

- **Same semantics, to the bit.** The kernel reproduces the numpy path's arithmetic exactly rather than approximately: the same `n_cells = max(1, floor(L / r))` grid and truncating cell hash (floor modulo under PBC, clipped otherwise), the same distinct stencil offsets for one or two cells per axis, the same pair order (stencil offsets in order, each cell's particles by increasing index, which is what the stable argsort gives), `delta - L * rint(delta / L)` per axis, the squared sum in axis order, `sqrt` then `<= r`, the neighbour velocities summed from 0 in pair order (as `bincount` does) and then added to the particle's own velocity and divided by `n + 1`. It is compiled with `-ffp-contract=off` so no multiply-add is fused. Two shortcuts skip work without changing a bit: the division and `rint` are skipped when `|delta| < 0.49 L` (the quotient rounds to 0), and `sqrt` is skipped when the squared distance exceeds `r^2 (1 + 1e-12)`. The test suite therefore compares the two paths with exact equality, including 25-step trajectories of all three models.
- **No per-type overload.** `HeterogeneousVicsek` has one interaction radius for all types; types only weight the pairs. Its step takes the pairs from the kernel and keeps the weighting in numpy, which gives exact equality without a second C++ entry point, at the cost of a smaller speedup (below). A fused weighted kernel is the obvious next step if the heterogeneous model becomes the bottleneck.
- **Scalar box length, general dimension.** The library's box is cubic, so the kernel takes one `box_length`. Any dimension works (2 and 3 are compiled specialisations; tests cover 1 to 4).
- **Threads optional.** `std::thread` over contiguous blocks of focal particles (at least 256 per block), off by default (`vicsek.use_kernel(True, threads=k)`); each particle's sum is computed in the same order whatever the thread count, so results do not depend on it.
- **Fallback build.** `CMakeLists.txt` is `LANGUAGES NONE` and enables C++ only if `check_language(CXX)` finds a compiler, so the sdist installs as the pure-Python package without one. The import of `vicsek._kernel` is wrapped in `try/except ImportError` as planned.

Inputs the numpy path accepts silently but the kernel rejects with `ValueError`: non-finite positions, positions beyond about 4e18 cell widths from the box, non-positive radius or box length, and grids of more than 1e9 cells (where numpy would try to allocate the same `bincount` array). These never occur in a model step.

### Measurements

Same machine as above (Apple M5, 10 cores), macOS 26, Python 3.12.15, numpy 2.5.3, Apple clang 21 (`-O3` Release build via scikit-build-core), 2026-10-10, machine otherwise lightly loaded (load average about 1). `benchmarks/profile_step.py --sizes 20000 --size-3d 20000 --skip-cprofile --skip-brute --skip-vectorized`, medians of 5 repeats x 20 steps; section 5 checks 5 seeded steps bit-identical to the numpy path (PASS for both configurations).

| N = 20000 | numpy step (ms) | kernel step, 1 thread (ms) | of which C++ call (ms) | kernel step, 4 threads (ms) | of which C++ call (ms) |
|---|---:|---:|---:|---:|---:|
| 2-D | 17.4 | 5.18 | 4.24 | 2.29 | 1.33 |
| 3-D | 52.0 | 12.83 | 11.72 | 4.48 | 3.34 |

Speedup of the whole step: 3.4x (2-D) and 4.0x (3-D) single-threaded, 7.6x and 11.6x with 4 threads. The numpy step is faster here than in the "After vectorisation" table (17.4 against 26.0 ms in 2-D) because the earlier runs shared the machine with a 4-core sweep.

Phase split of the 1-thread 2-D kernel step: C++ neighbour phase 4.24 ms (82 %), normalise + noise 0.85 ms (16 %), position update 0.09 ms (2 %). The neighbour phase is 0.21 us per particle, about 5.7 ns per candidate pair (757 k candidates), against the standalone reference's 8.6 ms for the whole step; the two shortcuts above account for most of the difference (5.4 ms to 3.9 ms for the bare call in a quick before/after). With 4 threads the numpy remainder (noise draw and normalisation, about 1 ms) is 40 % of the step, so more threads or moving the noise into C++ would gain little more without changing how the random numbers are drawn.

Other models at N = 20000, 2-D, median of 5 x 10 steps:

| model | numpy (ms) | kernel (ms) |
|---|---:|---:|
| `HeterogeneousVicsek` (two types) | 18.1 | 12.3 |
| `KernelizedVicsek` (epsilon 0.05) | 19.2 | 10.0 |

Their remaining time is the per-pair numpy work (gathering type weights or evaluating the smooth kernel over 250 k pairs and `bincount`), not the neighbour search.

### Against the target in issue #4

The issue asked for one `step()` at N = 20000 "well under the current 1.65 s". The numpy step is 17.4 ms and the kernel step 5.2 ms (2.3 ms with 4 threads) on this machine, so the target is met by both paths; the kernel adds a further 3.4x to 7.6x in 2-D and 4x to 12x in 3-D.
