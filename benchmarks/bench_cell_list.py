"""Benchmark PyVicsek's CellList neighbour search against a brute-force O(N^2) search.

For each N the box is scaled so that the number density is fixed
(rho = N / L^2 = DENSITY), which keeps the mean neighbour count constant
(rho * pi * r^2). Both methods use periodic minimum-image distances and the
same `<= interaction_range` criterion. Before any timing the two methods are
checked to return identical neighbour sets at N = 1000.

The script also times one full `Vicsek.step()` of the library at the sizes
given by --step-sizes (default 10000 and 20000).

Usage:
    python benchmarks/bench_cell_list.py
    python benchmarks/bench_cell_list.py --quick
    python benchmarks/bench_cell_list.py --sizes 500,1000,2000 --step-sizes 2000
"""
from __future__ import annotations

import argparse
import platform
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


def check_agreement(n: int = CHECK_N) -> None:
    particles = make_particles(n, seed=123)
    length = box_length(n)
    positions = np.array([p.position for p in particles])
    bf = brute_force_neighbors(positions, length, RADIUS)
    cl = cell_list_neighbors(particles, length, RADIUS)
    mismatches = sum(1 for a, b in zip(bf, cl) if a != b)
    mean_nb = np.mean([len(s) for s in bf])
    if mismatches:
        raise SystemExit(f"FAIL: {mismatches}/{n} particles have different neighbour sets between brute force and CellList")
    print(f"Neighbour sets identical for all {n} particles (mean neighbours {mean_nb:.2f}, expected rho*pi*r^2 = {DENSITY * np.pi * RADIUS**2:.2f}).")


def time_once(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def time_step(n: int) -> float:
    particles = make_particles(n, seed=7)
    model = vs.Vicsek(length=box_length(n), particles=particles, interaction_range=RADIUS, speed=SPEED, noise_factor=1.0)
    return time_once(model.step)


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
        step = step_times.get(n)
        if step is None and n in step_sizes:
            step = time_step(n)
        rows.append((n, length, t_bf, t_cl, step))
        print(f"  N={n:>6}  brute {t_bf:8.3f}s  cell-list {t_cl:8.3f}s", file=sys.stderr)
    for n in step_sizes:
        if n not in sizes:
            rows.append((n, box_length(n), None, None, step_times[n]))
    rows.sort(key=lambda r: r[0])

    print("\n| N | L | brute force (s) | cell list (s) | speedup | Vicsek.step() (s) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n, length, t_bf, t_cl, step in rows:
        bf = f"{t_bf:.3f}" if t_bf is not None else "-"
        cl = f"{t_cl:.3f}" if t_cl is not None else "-"
        sp = f"{t_bf / t_cl:.1f}x" if (t_bf is not None and t_cl) else "-"
        st = f"{step:.2f}" if step is not None else "-"
        print(f"| {n} | {length:.1f} | {bf} | {cl} | {sp} | {st} |")


if __name__ == "__main__":
    main()
