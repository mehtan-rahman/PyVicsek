"""Reproduce the Vicsek (1995) order-disorder transition with PyVicsek.

Sweeps the noise amplitude eta from 0 to 5 in steps of 0.25 at number density
rho = 4 (L = sqrt(N / rho)), speed v = 0.03 and interaction radius r = 1, and
measures the polar order parameter

    phi = | sum_i v_i | / (N v)

averaged over the second half of each run (the first half is discarded as
transient) and over 3 independent seeds. The critical noise eta_c is estimated
as the noise at which phi crosses 0.5 (linear interpolation between sweep
points).

Note on the noise model: this library adds a random *vector* of length eta to
the unit mean direction and renormalises ("vectorial noise", Gregoire & Chate
2004) rather than the uniform angular noise of Vicsek et al. 1995, and particle
velocities are updated sequentially within a step. The transition is still
present, but eta_c is not expected to match the 1995 value exactly.

Usage:
    python scripts/validate_phase_transition.py            # N = 400, quick
    python scripts/validate_phase_transition.py --full     # N = 4000
    python scripts/validate_phase_transition.py --steps 600 --workers 4

Outputs scripts/phase_transition.png and scripts/phase_transition.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np

import vicsek as vs

DENSITY = 4.0
SPEED = 0.03
RADIUS = 1.0
NOISES = np.arange(0.0, 5.0 + 1e-9, 0.25)
SEEDS = (0, 1, 2)
HERE = Path(__file__).resolve().parent


def run_one(args: Tuple[int, float, int, int]) -> Tuple[float, int, float, float]:
    """Run one (N, eta, seed) simulation; return (eta, seed, mean phi, std phi) over the second half."""
    n, eta, seed, steps = args
    length = float(np.sqrt(n / DENSITY))
    np.random.seed(seed)  # the library's noise draws from numpy's global RNG
    particles = vs.initialize_random_particles(
        n_particles=n, box_length=length, speed=SPEED, n_dimensions=2, seed=seed
    )
    model = vs.Vicsek(
        length=length, particles=particles, interaction_range=RADIUS, speed=SPEED, noise_factor=eta
    )
    discard = steps // 2
    for _ in range(discard):
        model.step()
    samples = []
    for _ in range(steps - discard):
        model.step()
        samples.append(model.order_parameter())
    return eta, seed, float(np.mean(samples)), float(np.std(samples))


def critical_noise(noises: np.ndarray, phi: np.ndarray, threshold: float = 0.5) -> float | None:
    """First noise at which phi drops below threshold, by linear interpolation."""
    for k in range(1, len(noises)):
        if phi[k - 1] >= threshold > phi[k]:
            x0, x1, y0, y1 = noises[k - 1], noises[k], phi[k - 1], phi[k]
            return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
    return None


def plot(noises, phi_mean, phi_err, eta_c, n, steps, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.errorbar(noises, phi_mean, yerr=phi_err, color=blue, lw=2, marker="o", ms=6,
                mfc="white", mew=2, capsize=3, ecolor=blue, elinewidth=1)
    if eta_c is not None:
        ax.axvline(eta_c, color=muted, lw=1, ls="--")
        ax.annotate(f"$\\eta_c \\approx {eta_c:.2f}$", xy=(eta_c, 0.5), xytext=(eta_c + 0.15, 0.55),
                    color=ink, fontsize=10)
    ax.axhline(0.5, color=grid, lw=1)
    ax.set_xlabel("noise amplitude $\\eta$", color=ink)
    ax.set_ylabel("order parameter $\\varphi$", color=ink)
    ax.set_title(f"Vicsek order-disorder transition  (N = {n}, $\\rho$ = {DENSITY:g}, v = {SPEED}, r = {RADIUS:g}, "
                 f"{steps} steps, {len(SEEDS)} seeds)", fontsize=10, color=ink, loc="left")
    ax.set_xlim(noises.min() - 0.1, noises.max() + 0.1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="N = 4000 instead of 400 (slow)")
    ap.add_argument("-n", type=int, default=None, help="override particle count")
    ap.add_argument("--steps", type=int, default=None,
                    help="steps per run (first half discarded). Default 300 quick, 2000 full")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel worker processes (one (eta, seed) run each)")
    ap.add_argument("--out-prefix", default=str(HERE / "phase_transition"), help="output path prefix for .png/.csv")
    args = ap.parse_args()

    n = args.n or (4000 if args.full else 400)
    steps = args.steps or (2000 if args.full else 300)
    out_png, out_csv = Path(args.out_prefix + ".png"), Path(args.out_prefix + ".csv")

    jobs = [(n, float(eta), seed, steps) for eta in NOISES for seed in SEEDS]
    print(f"N = {n}, L = {np.sqrt(n / DENSITY):.2f}, v = {SPEED}, r = {RADIUS}, {steps} steps/run "
          f"({steps // 2} discarded), {len(NOISES)} noise values x {len(SEEDS)} seeds, {args.workers} workers")
    t0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_one, jobs))
    else:
        results = [run_one(j) for j in jobs]
    elapsed = time.perf_counter() - t0

    # aggregate over seeds
    phi_mean, phi_err = [], []
    for eta in NOISES:
        vals = [r[2] for r in results if abs(r[0] - eta) < 1e-12]
        phi_mean.append(np.mean(vals))
        phi_err.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    phi_mean, phi_err = np.array(phi_mean), np.array(phi_err)
    eta_c = critical_noise(NOISES, phi_mean)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "eta", "phi_mean", "phi_std_over_seeds", "phi_seed0", "phi_seed1", "phi_seed2"])
        for eta, m, e in zip(NOISES, phi_mean, phi_err):
            per_seed = [r[2] for r in sorted(results, key=lambda r: r[1]) if abs(r[0] - eta) < 1e-12]
            w.writerow([n, f"{eta:.2f}", f"{m:.4f}", f"{e:.4f}", *[f"{v:.4f}" for v in per_seed]])
    print(f"Saved {out_csv}")

    print("\n  eta    phi (mean +- std over seeds)")
    for eta, m, e in zip(NOISES, phi_mean, phi_err):
        print(f"  {eta:4.2f}   {m:.3f} +- {e:.3f}")
    if eta_c is None:
        print("\nphi never crossed 0.5: no eta_c estimate.")
    else:
        print(f"\nEstimated critical noise eta_c = {eta_c:.2f} (phi = 0.5 crossing, linear interpolation)")
    print(f"Wall time: {elapsed:.0f} s")

    plot(NOISES, phi_mean, phi_err, eta_c, n, steps, out_png)


if __name__ == "__main__":
    main()
