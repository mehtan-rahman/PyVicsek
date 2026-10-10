"""Profile one Vicsek.step() and prototype a vectorised numpy cell list.

For each configuration (N, dim) the box is scaled to number density
rho = N / L^dim = DENSITY, so the mean neighbour count is fixed
(rho * pi * r^2 = 12.6 in 2-D). The script:

1. Times `steps` consecutive Vicsek.step() calls `repeats` times with
   per-phase wall-clock timers (monkeypatched wrappers around the library's
   own methods, no source changes) and reports the median per-step time of
   each phase: cell-list rebuild, neighbour queries, heading + noise math,
   position update, and the remainder of the step loop (list building and
   the velocity gather in Vicsek.step itself).
2. Runs cProfile over `cprofile_steps` steps and prints the top functions by
   cumulative and by internal time, the number of Python function calls per
   particle per step, and the split of internal time between Python-level
   code (vicsek package) and C-level code (numpy, builtins).
3. Prototypes a fully vectorised numpy cell list (sort particles by cell,
   bincount/cumsum for cell starts, repeat/arange to expand candidate
   ranges, bincount to sum headings) that computes the same mean headings
   with no Python per-particle loop. It is checked against the library for
   a fixed seed (pre-noise headings, and a full seeded step including the
   noise stream) before it is timed.
4. Times the pure-numpy brute-force neighbour search from
   bench_cell_list.py as the other reference point.

Usage:
    python benchmarks/profile_step.py              # full run, ~12 min on an M-series laptop
    python benchmarks/profile_step.py --quick      # N = 1000 and 5000 only, fewer repeats
    python benchmarks/profile_step.py --sizes 20000 --no-3d --repeats 5 --steps 5
"""
from __future__ import annotations

import argparse
import cProfile
import io
import platform
import pstats
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import vicsek as vs
from vicsek.models.vicsek import Vicsek
from vicsek.util.cell_list import CellList

# bench_cell_list.py lives next to this script; python puts the script's
# directory first on sys.path, so the import works from any cwd.
from bench_cell_list import brute_force_neighbors  # noqa: E402

DENSITY = 4.0
RADIUS = 1.0
SPEED = 0.03
NOISE = 1.0
DEFAULT_SIZES = [1000, 5000, 20000]
QUICK_SIZES = [1000, 5000]
SIZE_3D = 5000

PHASES = ("cell_list_update", "neighbor_query", "heading_noise", "position_update", "step_loop_rest")
PHASE_LABEL = {
    "cell_list_update": "cell-list rebuild (CellList.update)",
    "neighbor_query": "neighbour query (CellList.get_neighbors)",
    "heading_noise": "heading + noise (_compute_average_velocity, _apply_noise)",
    "position_update": "position update + wrap (_apply_updates minus update)",
    "step_loop_rest": "step loop rest (list concat, velocity gather, append)",
}


def box_length(n: int, dim: int) -> float:
    return float((n / DENSITY) ** (1.0 / dim))


def make_model(n: int, dim: int, seed: int = 7, rng_seed: Optional[int] = 1) -> Vicsek:
    length = box_length(n, dim)
    particles = vs.initialize_random_particles(
        n_particles=n, box_length=length, speed=SPEED, n_dimensions=dim, seed=seed
    )
    return Vicsek(length=length, particles=particles, interaction_range=RADIUS,
                  speed=SPEED, noise_factor=NOISE, seed=rng_seed)


def median(xs: List[float]) -> float:
    return statistics.median(xs)


# --------------------------------------------------------------------------
# 1. Per-phase wall-clock timers via monkeypatched wrappers
# --------------------------------------------------------------------------

class PhaseTimers:
    """Wrap the library's methods so each phase accumulates perf_counter time.

    The wrappers add ~0.2 us per call; with 3 wrapped calls per particle per
    step that is < 1 us per particle against ~150 us measured, so it does not
    distort the breakdown.
    """

    def __init__(self) -> None:
        self.acc: Dict[str, float] = {p: 0.0 for p in PHASES}
        self._originals: Dict[Tuple[type, str], Callable] = {}

    def _wrap(self, cls: type, name: str, phase: str) -> None:
        original = getattr(cls, name)
        self._originals[(cls, name)] = original
        acc = self.acc
        perf = time.perf_counter

        def wrapper(*args, **kwargs):
            t0 = perf()
            try:
                return original(*args, **kwargs)
            finally:
                acc[phase] += perf() - t0

        setattr(cls, name, wrapper)

    def __enter__(self) -> "PhaseTimers":
        self._wrap(CellList, "get_neighbors", "neighbor_query")
        self._wrap(CellList, "update", "cell_list_update")
        self._wrap(Vicsek, "_compute_average_velocity", "heading_noise")
        self._wrap(Vicsek, "_apply_noise", "heading_noise")
        # _apply_updates includes CellList.update; subtract it afterwards.
        self._wrap(Vicsek, "_apply_updates", "position_update")
        return self

    def __exit__(self, *exc) -> None:
        for (cls, name), original in self._originals.items():
            setattr(cls, name, original)

    def reset(self) -> None:
        for p in PHASES:
            self.acc[p] = 0.0


def time_phases(n: int, dim: int, steps: int, repeats: int) -> Dict[str, float]:
    """Median per-step time (s) of each phase and of the whole step."""
    per_step: Dict[str, List[float]] = {p: [] for p in PHASES}
    totals: List[float] = []
    model = make_model(n, dim)
    with PhaseTimers() as timers:
        for _ in range(repeats):
            timers.reset()
            t0 = time.perf_counter()
            for _ in range(steps):
                model.step()
            total = (time.perf_counter() - t0) / steps
            acc = {p: timers.acc[p] / steps for p in PHASES}
            acc["position_update"] -= acc["cell_list_update"]
            acc["step_loop_rest"] = total - sum(acc[p] for p in PHASES if p != "step_loop_rest")
            for p in PHASES:
                per_step[p].append(acc[p])
            totals.append(total)
    out = {p: median(per_step[p]) for p in PHASES}
    out["total"] = median(totals)
    out["total_min"] = min(totals)
    out["total_max"] = max(totals)
    return out


# --------------------------------------------------------------------------
# 2. cProfile attribution
# --------------------------------------------------------------------------

@dataclass
class ProfileSummary:
    total_time: float
    calls: int
    prim_calls: int
    vicsek_tottime: float      # Python code in the vicsek package
    numpy_py_tottime: float    # numpy's own Python-level wrappers (norm, round, mean, ...)
    c_tottime: float           # builtins and C extension functions
    top_cum: str
    top_tot: str
    by_function: Dict[str, Tuple[int, float, float]] = field(default_factory=dict)  # name -> (ncalls, tottime, cumtime)


def profile_steps(n: int, dim: int, steps: int, top: int = 14) -> ProfileSummary:
    model = make_model(n, dim)
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(steps):
        model.step()
    prof.disable()

    stats = pstats.Stats(prof)
    vicsek_tot = 0.0
    numpy_py_tot = 0.0
    c_tot = 0.0
    by_function: Dict[str, Tuple[int, float, float]] = {}
    for (filename, line, funcname), (cc, nc, tt, ct, callers) in stats.stats.items():
        if filename == "~":                       # builtins and C extension functions
            c_tot += tt
        elif "/vicsek/" in filename.replace("\\", "/") or filename == __file__:
            vicsek_tot += tt
        else:                                     # numpy (and other) Python-level wrappers
            numpy_py_tot += tt
        by_function[f"{filename.rsplit('/', 1)[-1]}:{line}({funcname})"] = (nc, tt, ct)
    stats.strip_dirs()

    def table(sort_key: str) -> str:
        buf = io.StringIO()
        s = pstats.Stats(prof, stream=buf)
        s.strip_dirs().sort_stats(sort_key).print_stats(top)
        lines = buf.getvalue().splitlines()
        # keep the header row and the entries only
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("ncalls"))
        return "\n".join(lines[start:])

    return ProfileSummary(
        total_time=stats.total_tt,
        calls=stats.total_calls,
        prim_calls=stats.prim_calls,
        vicsek_tottime=vicsek_tot,
        numpy_py_tottime=numpy_py_tot,
        c_tottime=c_tot,
        top_cum=table("cumulative"),
        top_tot=table("tottime"),
        by_function=by_function,
    )


# --------------------------------------------------------------------------
# 3. Vectorised numpy cell list prototype
# --------------------------------------------------------------------------

def neighbor_offsets(dim: int) -> np.ndarray:
    return np.array(np.meshgrid(*[[-1, 0, 1]] * dim, indexing="ij")).reshape(dim, -1).T


def vectorized_mean_heading(positions: np.ndarray, velocities: np.ndarray, length: float,
                            radius: float, n_cells: int, timers: Optional[Dict[str, float]] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean heading (unit vector) of every particle over itself and its
    neighbours within `radius` under periodic minimum-image distance, with
    no Python per-particle loop. Also returns the neighbour count (excluding self).

    Requires n_cells >= 3 so that the 3^dim neighbouring cells are distinct
    (the library's CellList handles n_cells < 3 with a visited-set; for the
    benchmark sizes n_cells is 10 or more).
    """
    perf = time.perf_counter
    t0 = perf()
    n, dim = positions.shape
    cell_size = length / n_cells
    strides = n_cells ** np.arange(dim - 1, -1, -1)
    # same hashing as CellList._hash_position: truncate, then wrap
    cells = (positions / cell_size).astype(np.int64) % n_cells          # (N, dim)
    flat = cells @ strides                                                # (N,)
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n_cells ** dim)
    starts = np.concatenate(([0], np.cumsum(counts)))                     # (n_cells^dim + 1,)
    t1 = perf()

    offs = neighbor_offsets(dim)                                          # (3^dim, dim)
    nb_cells = ((cells[:, None, :] + offs[None, :, :]) % n_cells) @ strides  # (N, 3^dim)
    seg_start = starts[nb_cells].ravel()
    seg_len = (starts[nb_cells + 1] - starts[nb_cells]).ravel()
    total = int(seg_len.sum())
    focal = np.repeat(np.repeat(np.arange(n), offs.shape[0]), seg_len)   # i for every candidate
    within = np.arange(total) - np.repeat(np.cumsum(seg_len) - seg_len, seg_len)
    cand = order[np.repeat(seg_start, seg_len) + within]                  # j for every candidate
    t2 = perf()

    delta = positions[cand] - positions[focal]
    delta -= length * np.round(delta / length)
    dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    mask = dist <= radius                                                 # self has dist 0: included once, as in the library
    ii = focal[mask]
    jj = cand[mask]
    sums = np.empty((n, dim))
    for k in range(dim):
        sums[:, k] = np.bincount(ii, weights=velocities[jj, k], minlength=n)
    nb_count = np.bincount(ii, minlength=n)
    mean = sums / nb_count[:, None]
    norm = np.linalg.norm(mean, axis=1)
    if np.any(norm == 0):   # never happens for random data; keep the library's semantics explicit
        raise RuntimeError("zero mean heading; library would draw a random direction here")
    heading = mean / norm[:, None]
    t3 = perf()
    if timers is not None:
        timers["bin_sort"] = timers.get("bin_sort", 0.0) + (t1 - t0)
        timers["gather_candidates"] = timers.get("gather_candidates", 0.0) + (t2 - t1)
        timers["distance_sum"] = timers.get("distance_sum", 0.0) + (t3 - t2)
    return heading, nb_count - 1


def vectorized_step(positions: np.ndarray, velocities: np.ndarray, length: float, radius: float,
                    n_cells: int, speed: float, mu: float, dt: float, rng: np.random.Generator,
                    timers: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    heading, _ = vectorized_mean_heading(positions, velocities, length, radius, n_cells, timers)
    t0 = time.perf_counter()
    n, dim = positions.shape
    # Same stream as N sequential _random_unit_vector(dim, rng) calls:
    # Generator.normal(size=(N, dim)) yields the same doubles in row-major order.
    noise = rng.normal(0, 1, (n, dim))
    noise /= np.linalg.norm(noise, axis=1)[:, None]
    noisy = heading + mu * noise
    noisy /= np.linalg.norm(noisy, axis=1)[:, None]
    new_v = speed * noisy
    new_p = (positions + new_v * dt) % length
    if timers is not None:
        timers["noise_update"] = timers.get("noise_update", 0.0) + (time.perf_counter() - t0)
    return new_p, new_v


def state_arrays(model: Vicsek) -> Tuple[np.ndarray, np.ndarray]:
    return (np.array([p.position for p in model.particles], dtype=float),
            np.array([p.velocity for p in model.particles], dtype=float))


def verify_vectorized(n: int, dim: int, check_steps: int = 3) -> Dict[str, float]:
    """Compare the vectorised path with the library for a fixed seed.

    (a) pre-noise mean headings against the library's own neighbour lists,
    (b) a full seeded step sequence (velocities and positions) so the noise
    stream is checked too.
    """
    model = make_model(n, dim, seed=123, rng_seed=777)
    pos, vel = state_arrays(model)
    cl = model._cell_list

    ref = np.array([
        model._compute_average_velocity(np.array([p.velocity for p in [particle] + cl.get_neighbors(particle)]))
        for particle in model.particles
    ])
    ref_counts = np.array([len(cl.get_neighbors(p)) for p in model.particles])
    heading, counts = vectorized_mean_heading(pos, vel, model.length, RADIUS, cl.n_cells)
    out = {
        "max_abs_heading_diff": float(np.max(np.abs(heading - ref))),
        "neighbor_count_mismatches": int(np.sum(counts != ref_counts)),
        "mean_neighbors": float(ref_counts.mean()),
    }

    rng = np.random.default_rng(777)
    max_v = 0.0
    max_p = 0.0
    for _ in range(check_steps):
        pos, vel = vectorized_step(pos, vel, model.length, RADIUS, cl.n_cells, SPEED, NOISE, model.delta_t, rng)
        model.step()
        lib_pos, lib_vel = state_arrays(model)
        max_v = max(max_v, float(np.max(np.abs(vel - lib_vel))))
        max_p = max(max_p, float(np.max(np.abs(pos - lib_pos))))
    out["max_abs_velocity_diff_after_steps"] = max_v
    out["max_abs_position_diff_after_steps"] = max_p
    out["check_steps"] = check_steps
    return out


def time_vectorized(n: int, dim: int, steps: int, repeats: int) -> Dict[str, float]:
    model = make_model(n, dim)
    pos, vel = state_arrays(model)
    n_cells = model._cell_list.n_cells
    rng = np.random.default_rng(1)
    totals: List[float] = []
    phases: Dict[str, List[float]] = {}
    for _ in range(repeats):
        timers: Dict[str, float] = {}
        t0 = time.perf_counter()
        for _ in range(steps):
            pos, vel = vectorized_step(pos, vel, model.length, RADIUS, n_cells, SPEED, NOISE, 1.0, rng, timers)
        totals.append((time.perf_counter() - t0) / steps)
        for k, v in timers.items():
            phases.setdefault(k, []).append(v / steps)
    out = {k: median(v) for k, v in phases.items()}
    out["total"] = median(totals)
    return out


# --------------------------------------------------------------------------
# 4. Brute force reference (pure numpy, O(N^2), from bench_cell_list.py)
# --------------------------------------------------------------------------

def time_brute_force(n: int, dim: int, repeats: int) -> float:
    model = make_model(n, dim)
    pos, _ = state_arrays(model)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        brute_force_neighbors(pos, model.length, RADIUS)
        times.append(time.perf_counter() - t0)
    return median(times)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def fmt_phase_table(n: int, phases: Dict[str, float]) -> str:
    total = phases["total"]
    rows = ["| phase | ms/step | % | us/particle |", "|---|---:|---:|---:|"]
    for p in PHASES:
        t = phases[p]
        rows.append(f"| {PHASE_LABEL[p]} | {t * 1e3:.1f} | {100 * t / total:.1f} | {t / n * 1e6:.1f} |")
    rows.append(f"| **total step** | **{total * 1e3:.1f}** | 100 | **{total / n * 1e6:.1f}** |")
    return "\n".join(rows)


def run_config(n: int, dim: int, args: argparse.Namespace) -> Dict[str, object]:
    length = box_length(n, dim)
    n_cells = max(1, int(np.floor(length / RADIUS)))
    print("=" * 96)
    print(f"N = {n}, dim = {dim}, L = {length:.2f}, n_cells = {n_cells}^{dim}, cell size = {length / n_cells:.3f}, "
          f"rho = {DENSITY}, r = {RADIUS}")
    print("=" * 96)

    result: Dict[str, object] = {"n": n, "dim": dim}

    # -- phase timers
    phases = time_phases(n, dim, args.steps, args.repeats)
    result["phases"] = phases
    print(f"\n[1] Per-phase wall-clock, median of {args.repeats} repeats x {args.steps} steps "
          f"(step total min/median/max = {phases['total_min'] * 1e3:.0f}/{phases['total'] * 1e3:.0f}/{phases['total_max'] * 1e3:.0f} ms)")
    print(fmt_phase_table(n, phases))

    # -- cProfile
    if not args.skip_cprofile:
        prof = profile_steps(n, dim, args.cprofile_steps)
        result["profile"] = prof
        per_particle_calls = prof.calls / (n * args.cprofile_steps)
        tot = prof.vicsek_tottime + prof.numpy_py_tottime + prof.c_tottime
        print(f"\n[2] cProfile over {args.cprofile_steps} steps: {prof.total_time:.2f} s total "
              f"({prof.total_time / args.cprofile_steps / n * 1e6:.0f} us/particle/step under the profiler)")
        print(f"    {prof.calls:,} function calls ({prof.prim_calls:,} primitive; Python functions and C builtins) = "
              f"{per_particle_calls:.0f} calls per particle per step")
        print(f"    internal time: vicsek Python code {prof.vicsek_tottime:.2f} s ({100 * prof.vicsek_tottime / tot:.0f}%), "
              f"numpy Python wrappers {prof.numpy_py_tottime:.2f} s ({100 * prof.numpy_py_tottime / tot:.0f}%), "
              f"C-level (numpy kernels, builtins) {prof.c_tottime:.2f} s ({100 * prof.c_tottime / tot:.0f}%)")
        print("\n    top by cumulative time:")
        print("\n".join("    " + l for l in prof.top_cum.splitlines()))
        print("\n    top by internal time (tottime):")
        print("\n".join("    " + l for l in prof.top_tot.splitlines()))

    # -- vectorised prototype
    if not args.skip_vectorized:
        check = verify_vectorized(n, dim)
        result["vector_check"] = check
        ok = (check["max_abs_heading_diff"] < 1e-12 and check["neighbor_count_mismatches"] == 0
              and check["max_abs_velocity_diff_after_steps"] < 1e-12 and check["max_abs_position_diff_after_steps"] < 1e-9)
        print(f"\n[3] Vectorised numpy cell list, agreement with library (seed 123, noise seed 777): "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"    mean neighbours {check['mean_neighbors']:.2f} (expected {DENSITY * np.pi * RADIUS ** 2:.2f} in 2-D); "
              f"neighbour-count mismatches {check['neighbor_count_mismatches']}; "
              f"max |heading diff| {check['max_abs_heading_diff']:.1e}; "
              f"after {check['check_steps']} seeded steps max |v diff| {check['max_abs_velocity_diff_after_steps']:.1e}, "
              f"max |x diff| {check['max_abs_position_diff_after_steps']:.1e}")
        if not ok:
            raise SystemExit("vectorised path disagrees with the library; not timing it")
        vec = time_vectorized(n, dim, args.vector_steps, args.repeats)
        result["vector"] = vec
        print(f"    median per step over {args.repeats} x {args.vector_steps} steps: {vec['total'] * 1e3:.1f} ms "
              f"({vec['total'] / n * 1e6:.2f} us/particle), library step {phases['total'] * 1e3:.1f} ms -> "
              f"{phases['total'] / vec['total']:.1f}x faster")
        print("    | sub-phase | ms/step | % |")
        print("    |---|---:|---:|")
        for k in ("bin_sort", "gather_candidates", "distance_sum", "noise_update"):
            print(f"    | {k} | {vec[k] * 1e3:.2f} | {100 * vec[k] / vec['total']:.0f} |")

    # -- brute force
    if not args.skip_brute:
        bf = time_brute_force(n, dim, max(1, args.repeats if n <= 5000 else 3))
        result["brute"] = bf
        print(f"\n[4] Pure-numpy brute-force neighbour search (bench_cell_list.brute_force_neighbors): "
              f"{bf * 1e3:.0f} ms per pass ({bf / n * 1e6:.1f} us/particle); "
              f"library neighbour query {phases['neighbor_query'] * 1e3:.0f} ms")
    print()
    return result


def summary(results: List[Dict[str, object]]) -> None:
    print("=" * 96)
    print("SUMMARY (medians)")
    print("=" * 96)
    print("| N | dim | step ms | us/particle | rebuild % | neighbour % | heading+noise % | update % | loop rest % | calls/particle | vectorised ms | speedup | brute ms |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        n = r["n"]
        ph = r["phases"]
        tot = ph["total"]
        pct = {p: 100 * ph[p] / tot for p in PHASES}
        prof = r.get("profile")
        calls = f"{prof.calls / (n * cprofile_steps_used):.0f}" if prof else "-"
        vec = r.get("vector")
        vec_ms = f"{vec['total'] * 1e3:.1f}" if vec else "-"
        speed = f"{tot / vec['total']:.1f}x" if vec else "-"
        bf = r.get("brute")
        bf_ms = f"{bf * 1e3:.0f}" if bf is not None else "-"
        print(f"| {n} | {r['dim']} | {tot * 1e3:.0f} | {tot / n * 1e6:.0f} | {pct['cell_list_update']:.0f} | "
              f"{pct['neighbor_query']:.0f} | {pct['heading_noise']:.0f} | {pct['position_update']:.0f} | "
              f"{pct['step_loop_rest']:.0f} | {calls} | {vec_ms} | {speed} | {bf_ms} |")


cprofile_steps_used = 20


def main() -> None:
    global cprofile_steps_used
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=None,
                    help="comma-separated N for the 2-D runs (default 1000,5000,20000)")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3-D N=5000 configuration")
    ap.add_argument("--quick", action="store_true", help="N = 1000 and 5000 only, 3 repeats, no 3-D")
    ap.add_argument("--steps", type=int, default=5, help="steps per repeat for the phase timers (default 5)")
    ap.add_argument("--repeats", type=int, default=5, help="repeats for every timing; medians are reported (default 5)")
    ap.add_argument("--cprofile-steps", type=int, default=20, help="steps under cProfile (default 20)")
    ap.add_argument("--vector-steps", type=int, default=20, help="steps per repeat for the vectorised prototype (default 20)")
    ap.add_argument("--skip-cprofile", action="store_true")
    ap.add_argument("--skip-vectorized", action="store_true")
    ap.add_argument("--skip-brute", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.sizes = args.sizes or QUICK_SIZES
        args.repeats = min(args.repeats, 3)
        args.no_3d = True
    sizes = args.sizes or DEFAULT_SIZES
    cprofile_steps_used = args.cprofile_steps

    print(f"Machine: {platform.machine()} ({platform.system()} {platform.release()}), "
          f"Python {sys.version.split()[0]}, numpy {np.__version__}")
    print(f"Density rho = {DENSITY}, r = {RADIUS}, speed = {SPEED}, noise = {NOISE}, periodic boundaries. "
          f"Timings: median of {args.repeats} repeats.\n")

    configs = [(n, 2) for n in sizes]
    if not args.no_3d:
        configs.append((SIZE_3D, 3))
    results = [run_config(n, dim, args) for n, dim in configs]
    summary(results)


if __name__ == "__main__":
    main()
