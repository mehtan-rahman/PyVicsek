"""Profile one Vicsek.step() phase by phase.

For each configuration (N, dim) the box is scaled to number density
rho = N / L^dim = DENSITY, so the mean neighbour count is fixed
(rho * pi * r^2 = 12.6 in 2-D). The script:

1. Times `steps` consecutive Vicsek.step() calls `repeats` times with
   per-phase wall-clock timers (monkeypatched wrappers around the library's
   own functions, no source changes) and reports the median per-step time of
   each phase of the vectorised step: binning particles into cells, expanding
   candidate pairs, pair distances, the masked heading sum, heading
   normalisation + noise, and the position update.
2. Runs cProfile over `cprofile_steps` steps and prints the top functions by
   cumulative and by internal time, the number of Python function calls per
   particle per step, and the split of internal time between Python-level
   code (vicsek package) and C-level code (numpy, builtins).
3. Checks the vectorised step against the per-particle definition
   (CellList.get_neighbors + _compute_average_velocity + _apply_noise with a
   replayed random stream) for a fixed seed.
4. Times the pure-numpy brute-force neighbour search from
   bench_cell_list.py and the per-particle CellList.get_neighbors query as
   the two reference points.

Before 0.3.0 the step was a Python loop over particles (see docs/profiling.md
for that profile and the prototype this step grew out of).

Usage:
    python benchmarks/profile_step.py              # full run, a few minutes
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
from vicsek.util import cell_list as cell_list_module
from vicsek.util.cell_list import CellList
from vicsek.util.linalg import _random_unit_vector

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

PHASES = ("bin_sort", "candidate_pairs", "pair_distances", "heading_sum", "heading_noise",
          "position_update", "step_rest")
PHASE_LABEL = {
    "bin_sort": "bin + sort particles into cells (_bin_particles)",
    "candidate_pairs": "expand candidate pairs (_candidate_pairs)",
    "pair_distances": "pair distances, minimum image (_pair_distances)",
    "heading_sum": "mask pairs + heading sum (rest of mean_neighbor_velocity)",
    "heading_noise": "normalise + noise (_noisy_headings)",
    "position_update": "position update + wrap (_apply_updates)",
    "step_rest": "step rest",
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
    """Wrap the library's functions so each phase accumulates perf_counter time.

    Six wrapped calls per step, ~0.2 us each: negligible against a step of
    milliseconds. The sub-phases of the neighbour search are module-level
    functions in vicsek.util.cell_list, looked up through the module globals
    at call time, so replacing the module attribute is enough.
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
        self._wrap(cell_list_module, "_bin_particles", "bin_sort")
        self._wrap(cell_list_module, "_candidate_pairs", "candidate_pairs")
        self._wrap(cell_list_module, "_pair_distances", "pair_distances")
        # mean_neighbor_velocity includes the three above; subtract them afterwards.
        self._wrap(CellList, "mean_neighbor_velocity", "heading_sum")
        self._wrap(Vicsek, "_noisy_headings", "heading_noise")
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
            acc["heading_sum"] -= acc["bin_sort"] + acc["candidate_pairs"] + acc["pair_distances"]
            acc["step_rest"] = total - sum(acc[p] for p in PHASES if p != "step_rest")
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
# 3. Vectorised step versus the per-particle definition
# --------------------------------------------------------------------------

def reference_step(model: Vicsek, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """What the step must produce, computed one particle at a time from the
    per-particle API (as Vicsek.step did before 0.3.0), drawing from `rng`."""
    cl = model._cell_list
    new_v = []
    for p in model.particles:
        vel = np.array([q.velocity for q in [p] + cl.get_neighbors(p)])
        mean = vel.mean(axis=0)
        norm = np.linalg.norm(mean)
        heading = mean / norm if norm > 0 else _random_unit_vector(model.dim, rng)
        noisy = heading + model.mu * _random_unit_vector(model.dim, rng)
        new_v.append(model.v * noisy / np.linalg.norm(noisy))
    new_v = np.array(new_v)
    new_p = (model.positions + new_v * model.delta_t) % model.length
    return new_p, new_v


def verify_vectorized(n: int, dim: int, check_steps: int = 3) -> Dict[str, float]:
    """Compare the library's vectorised step with the per-particle definition
    for a fixed seed, over `check_steps` steps, replaying the same random stream."""
    model = make_model(n, dim, seed=123, rng_seed=777)
    counts = np.array([len(model._cell_list.get_neighbors(p)) for p in model.particles])
    _, vec_counts = model._cell_list.mean_neighbor_velocity(model.positions, model.velocities)
    out = {"mean_neighbors": float(counts.mean()),
           "neighbor_count_mismatches": int(np.sum(counts != vec_counts))}
    max_v = 0.0
    max_p = 0.0
    for _ in range(check_steps):
        replay = np.random.default_rng()
        replay.bit_generator.state = model._rng.bit_generator.state
        ref_p, ref_v = reference_step(model, replay)
        model.step()
        max_v = max(max_v, float(np.max(np.abs(model.velocities - ref_v))))
        max_p = max(max_p, float(np.max(np.abs(model.positions - ref_p))))
    out["max_abs_velocity_diff_after_steps"] = max_v
    out["max_abs_position_diff_after_steps"] = max_p
    out["check_steps"] = check_steps
    return out


def time_get_neighbors(n: int, dim: int, repeats: int) -> float:
    """Per-particle CellList.get_neighbors over every particle (the pre-0.3.0 query)."""
    model = make_model(n, dim)
    cl = model._cell_list
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for p in model.particles:
            cl.get_neighbors(p)
        times.append(time.perf_counter() - t0)
    return median(times)


# --------------------------------------------------------------------------
# 4. Brute force reference (pure numpy, O(N^2), from bench_cell_list.py)
# --------------------------------------------------------------------------

def time_brute_force(n: int, dim: int, repeats: int) -> float:
    model = make_model(n, dim)
    pos = model.positions
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
        rows.append(f"| {PHASE_LABEL[p]} | {t * 1e3:.2f} | {100 * t / total:.1f} | {t / n * 1e6:.2f} |")
    rows.append(f"| **total step** | **{total * 1e3:.2f}** | 100 | **{total / n * 1e6:.2f}** |")
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
              f"{per_particle_calls:.2f} calls per particle per step")
        print(f"    internal time: vicsek Python code {prof.vicsek_tottime:.2f} s ({100 * prof.vicsek_tottime / tot:.0f}%), "
              f"numpy Python wrappers {prof.numpy_py_tottime:.2f} s ({100 * prof.numpy_py_tottime / tot:.0f}%), "
              f"C-level (numpy kernels, builtins) {prof.c_tottime:.2f} s ({100 * prof.c_tottime / tot:.0f}%)")
        print("\n    top by cumulative time:")
        print("\n".join("    " + l for l in prof.top_cum.splitlines()))
        print("\n    top by internal time (tottime):")
        print("\n".join("    " + l for l in prof.top_tot.splitlines()))

    # -- vectorised step against the per-particle definition
    if not args.skip_vectorized:
        check = verify_vectorized(n, dim)
        result["vector_check"] = check
        ok = (check["neighbor_count_mismatches"] == 0
              and check["max_abs_velocity_diff_after_steps"] < 1e-12 and check["max_abs_position_diff_after_steps"] < 1e-9)
        print(f"\n[3] Vectorised step versus per-particle definition (seed 123, noise seed 777): "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"    mean neighbours {check['mean_neighbors']:.2f} (expected {DENSITY * np.pi * RADIUS ** 2:.2f} in 2-D); "
              f"neighbour-count mismatches {check['neighbor_count_mismatches']}; "
              f"after {check['check_steps']} seeded steps max |v diff| {check['max_abs_velocity_diff_after_steps']:.1e}, "
              f"max |x diff| {check['max_abs_position_diff_after_steps']:.1e}")
        if not ok:
            raise SystemExit("vectorised step disagrees with the per-particle definition")

    # -- reference points: brute force and the per-particle query
    if not args.skip_brute:
        reps = max(1, args.repeats if n <= 5000 else 3)
        bf = time_brute_force(n, dim, reps)
        gn = time_get_neighbors(n, dim, reps)
        result["brute"] = bf
        result["get_neighbors"] = gn
        print(f"\n[4] Reference points: pure-numpy brute-force neighbour search "
              f"{bf * 1e3:.0f} ms per pass ({bf / n * 1e6:.1f} us/particle); "
              f"per-particle CellList.get_neighbors over all particles {gn * 1e3:.0f} ms "
              f"({gn / n * 1e6:.1f} us/particle); vectorised step total {phases['total'] * 1e3:.1f} ms")
    print()
    return result


def summary(results: List[Dict[str, object]]) -> None:
    print("=" * 96)
    print("SUMMARY (medians)")
    print("=" * 96)
    print("| N | dim | step ms | us/particle | bin % | candidates % | distances % | heading sum % | noise % | update % | calls/particle | get_neighbors ms | brute ms |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        n = r["n"]
        ph = r["phases"]
        tot = ph["total"]
        pct = {p: 100 * ph[p] / tot for p in PHASES}
        prof = r.get("profile")
        calls = f"{prof.calls / (n * cprofile_steps_used):.2f}" if prof else "-"
        gn = r.get("get_neighbors")
        gn_ms = f"{gn * 1e3:.0f}" if gn is not None else "-"
        bf = r.get("brute")
        bf_ms = f"{bf * 1e3:.0f}" if bf is not None else "-"
        print(f"| {n} | {r['dim']} | {tot * 1e3:.1f} | {tot / n * 1e6:.2f} | {pct['bin_sort']:.0f} | "
              f"{pct['candidate_pairs']:.0f} | {pct['pair_distances']:.0f} | {pct['heading_sum']:.0f} | "
              f"{pct['heading_noise']:.0f} | {pct['position_update']:.0f} | {calls} | {gn_ms} | {bf_ms} |")


cprofile_steps_used = 20


def main() -> None:
    global cprofile_steps_used
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=None,
                    help="comma-separated N for the 2-D runs (default 1000,5000,20000)")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3-D N=5000 configuration")
    ap.add_argument("--quick", action="store_true", help="N = 1000 and 5000 only, 3 repeats, no 3-D")
    ap.add_argument("--steps", type=int, default=20, help="steps per repeat for the phase timers (default 20)")
    ap.add_argument("--repeats", type=int, default=5, help="repeats for every timing; medians are reported (default 5)")
    ap.add_argument("--cprofile-steps", type=int, default=20, help="steps under cProfile (default 20)")
    ap.add_argument("--skip-cprofile", action="store_true")
    ap.add_argument("--skip-vectorized", action="store_true", help="skip the check against the per-particle definition")
    ap.add_argument("--skip-brute", action="store_true", help="skip the brute-force and get_neighbors reference timings")
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
