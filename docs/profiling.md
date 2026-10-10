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
