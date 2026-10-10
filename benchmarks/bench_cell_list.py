"""Benchmark PyVicsek's CellList neighbour search against a brute-force O(N^2) search.

For each N the box is scaled so that the number density is fixed
(rho = N / L^2 = DENSITY), which keeps the mean neighbour count constant
(rho * pi * r^2). Both methods use periodic minimum-image distances and the
same `<= interaction_range` criterion. Before any timing the two methods are
checked to return identical neighbour sets at N = 1000.

Two cell-list columns are reported: the per-particle API (`CellList.build`
then `get_neighbors` for every particle) and the vectorised routine
(`CellList.neighbor_pairs`, every pair at once, which is what `Vicsek.step`
uses). The speedup column is brute force over the vectorised routine.

The script also times a full `Vicsek.step()` of the library at the sizes
given by --step-sizes (default 10000 and 20000); the median of 5 steps.

Usage:
    python benchmarks/bench_cell_list.py
    python benchmarks/bench_cell_list.py --quick
    python benchmarks/bench_cell_list.py --sizes 500,1000,2000 --step-sizes 2000
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from datetime import date
from typing import Dict, List, Set

import numpy as np

import vicsek as vs
from vicsek.util.cell_list import CellList

DENSITY = 4.0          # particles per unit area
RADIUS = 1.0           # interaction range
SPEED = 0.03
DEFAULT_SIZES = [500, 1000, 2000, 5000, 10000, 20000]
DEFAULT_STEP_SIZES = [10000, 20000]
QUICK_SIZES = [500, 1000, 2000]
QUICK_STEP_SIZES = [2000]
CHECK_N = 1000


def box_length(n: int) -> float:
    return float(np.sqrt(n / DENSITY))


def make_particles(n: int, seed: int = 0):
    return vs.initialize_random_particles(
        n_particles=n, box_length=box_length(n), speed=SPEED, n_dimensions=2, seed=seed
    )


def brute_force_neighbors(positions: np.ndarray, length: float, radius: float) -> List[Set[int]]:
    """O(N^2) neighbour search with periodic minimum-image distances.

    One vectorised numpy pass per particle over all other particles.
    """
    n = len(positions)
    result: List[Set[int]] = []
    for i in range(n):
        delta = positions - positions[i]
        delta -= length * np.round(delta / length)
        dist = np.linalg.norm(delta, axis=1)
        mask = dist <= radius
        mask[i] = False
        result.append(set(np.flatnonzero(mask).tolist()))
    return result


def cell_list_neighbors(particles, length: float, radius: float) -> List[Set[int]]:
    """Neighbour search with the library's CellList (build + query every particle)."""
    index_of: Dict[int, int] = {id(p): k for k, p in enumerate(particles)}
    cl = CellList(particles=particles, box_length=length, interaction_range=radius, n_dimensions=2, use_pbc=True)
    cl.build()
    return [set(index_of[id(q)] for q in cl.get_neighbors(p)) for p in particles]


def vectorized_pairs(positions: np.ndarray, length: float, radius: float):
    """Neighbour search with the library's vectorised routine (all pairs at once)."""
    return CellList(particles=[], box_length=length, interaction_range=radius, n_dimensions=2,
                    use_pbc=True).neighbor_pairs(positions)


def vectorized_neighbors(positions: np.ndarray, length: float, radius: float) -> List[Set[int]]:
    """The vectorised pairs as per-particle sets, for the agreement check (not timed)."""
    i, j, _ = vectorized_pairs(positions, length, radius)
    result: List[Set[int]] = [set() for _ in range(len(positions))]
    for a, b in zip(i.tolist(), j.tolist()):
        result[a].add(b)
    return result


def check_agreement(n: int = CHECK_N) -> None:
    particles = make_particles(n, seed=123)
    length = box_length(n)
    positions = np.array([p.position for p in particles])
    bf = brute_force_neighbors(positions, length, RADIUS)
    cl = cell_list_neighbors(particles, length, RADIUS)
    vec = vectorized_neighbors(positions, length, RADIUS)
    mismatches = sum(1 for a, b in zip(bf, cl) if a != b)
    mismatches_vec = sum(1 for a, b in zip(bf, vec) if a != b)
    mean_nb = np.mean([len(s) for s in bf])
    if mismatches or mismatches_vec:
        raise SystemExit(f"FAIL: {mismatches}/{n} (get_neighbors) and {mismatches_vec}/{n} (neighbor_pairs) particles "
                         f"have different neighbour sets from brute force")
    print(f"Neighbour sets identical for all {n} particles in all three methods (mean neighbours {mean_nb:.2f}, "
          f"expected rho*pi*r^2 = {DENSITY * np.pi * RADIUS**2:.2f}).")


def time_once(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def time_step(n: int, repeats: int = 5) -> float:
    """Median of `repeats` consecutive Vicsek.step() calls (after one warm-up step)."""
    particles = make_particles(n, seed=7)
    model = vs.Vicsek(length=box_length(n), particles=particles, interaction_range=RADIUS, speed=SPEED, noise_factor=1.0)
    model.step()
    return statistics.median(time_once(model.step) for _ in range(repeats))


def parse_sizes(text: str) -> List[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=parse_sizes, default=None, help="comma-separated N values for the neighbour-search comparison")
    ap.add_argument("--step-sizes", type=parse_sizes, default=None, help="comma-separated N values for timing one Vicsek.step()")
    ap.add_argument("--quick", action="store_true", help=f"small sizes only ({QUICK_SIZES}; step at {QUICK_STEP_SIZES})")
    ap.add_argument("--skip-check", action="store_true", help="skip the brute-force vs cell-list agreement check")
    args = ap.parse_args()

    sizes = args.sizes or (QUICK_SIZES if args.quick else DEFAULT_SIZES)
    step_sizes = args.step_sizes or (QUICK_STEP_SIZES if args.quick else DEFAULT_STEP_SIZES)

    print(f"Machine: {platform.machine()} ({platform.system()} {platform.release()}), "
          f"Python {sys.version.split()[0]}, numpy {np.__version__}, {date.today().isoformat()}")
    print(f"Density rho = {DENSITY}, interaction range r = {RADIUS}, 2-D, periodic boundaries.\n")

    if not args.skip_check:
        check_agreement()
        print()

    rows = []
    step_times = {n: time_step(n) for n in step_sizes}
    for n in sizes:
        particles = make_particles(n, seed=0)
        length = box_length(n)
        positions = np.array([p.position for p in particles])
        t_bf = time_once(brute_force_neighbors, positions, length, RADIUS)
        t_cl = time_once(cell_list_neighbors, particles, length, RADIUS)
        t_vec = statistics.median(time_once(vectorized_pairs, positions, length, RADIUS) for _ in range(5))
        step = step_times.get(n)
        if step is None and n in step_sizes:
            step = time_step(n)
        rows.append((n, length, t_bf, t_cl, t_vec, step))
        print(f"  N={n:>6}  brute {t_bf:8.3f}s  get_neighbors {t_cl:8.3f}s  neighbor_pairs {t_vec:8.4f}s", file=sys.stderr)
    for n in step_sizes:
        if n not in sizes:
            rows.append((n, box_length(n), None, None, None, step_times[n]))
    rows.sort(key=lambda r: r[0])

    print("\n| N | L | brute force (s) | cell list, get_neighbors (s) | cell list, neighbor_pairs (s) | speedup | Vicsek.step() (ms) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n, length, t_bf, t_cl, t_vec, step in rows:
        bf = f"{t_bf:.3f}" if t_bf is not None else "-"
        cl = f"{t_cl:.3f}" if t_cl is not None else "-"
        vec = f"{t_vec:.4f}" if t_vec is not None else "-"
        sp = f"{t_bf / t_vec:.0f}x" if (t_bf is not None and t_vec) else "-"
        st = f"{step * 1e3:.1f}" if step is not None else "-"
        print(f"| {n} | {length:.1f} | {bf} | {cl} | {vec} | {sp} | {st} |")


if __name__ == "__main__":
    main()
