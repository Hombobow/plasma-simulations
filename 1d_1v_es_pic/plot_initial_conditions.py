#!/usr/bin/env python3
"""
Plot the initial phase-space (x, vx) from the first particle snapshot.

Reads:  <run_dir>/particles/particle_0.csv
        <run_dir>/fields/fields_0.csv   (optional; used for dx / L)
Writes: figures/initial_phase_space.png (or --output path)

Examples:
  python plot_initial_conditions.py
  python plot_initial_conditions.py output/landaudamping
  python plot_initial_conditions.py output/twostream -o figures/twostream/initial_phase_space.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUN_DIR = "output/twostream"
DEFAULT_OUT     = "figures/initial_phase_space.png"
V_DRIFT         = 0.5    # expected beam centers for twostream (match initialization.cpp)
DELTA_N         = 0.001  # density seed amplitude (match initialization.cpp)
WAVELENGTH      = 2.0    # in units of 2π → k = 1/WAVELENGTH (match initialization.cpp)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot initial phase space from particle_0.csv."
    )
    p.add_argument(
        "run_dir",
        nargs="?",
        default=DEFAULT_RUN_DIR,
        help=f"run directory with particles/ and fields/ (default: {DEFAULT_RUN_DIR})",
    )
    p.add_argument(
        "-o", "--output",
        default=DEFAULT_OUT,
        help=f"output PNG path (default: {DEFAULT_OUT})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    part_file = run_dir / "particles" / "particle_0.csv"
    field_file = run_dir / "fields" / "fields_0.csv"
    out_file = Path(args.output)

    if not part_file.is_file():
        raise SystemExit(f"No {part_file} — run the sim first so output index 0 exists.")

    data = np.loadtxt(part_file, delimiter=",", skiprows=1)
    x, vx = data[:, 0], data[:, 1]

    # Bin width = cell size dx from the field grid (one row per cell in fields_*.csv).
    # Falls back to the same formula as initialization.cpp if fields are missing.
    if field_file.is_file():
        fields = np.loadtxt(field_file, delimiter=",", skiprows=1)
        n_cells = len(fields)
        dx = float(fields[1, 0] - fields[0, 0]) if n_cells > 1 else float(2.0 * fields[0, 0])
        L = n_cells * dx
    else:
        L = float(np.max(x)) if x.size else 2 * np.pi
        n_cells = 32 * int(round(L / (2 * np.pi)))
        dx = L / n_cells

    vpad = 0.15 * max(1.0, float(np.max(np.abs(vx))))
    vmin = float(np.min(vx) - vpad)
    vmax = float(np.max(vx) + vpad)

    # Split by sign of velocity so opposite beams are easy to see
    pos = vx >= 0.0
    neg = ~pos

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [2.2, 1.0],
                                          "width_ratios": [3.0, 1.0]})
    ax_ps, ax_vh = axes[0, 0], axes[0, 1]
    ax_xh, ax_empty = axes[1, 0], axes[1, 1]
    ax_empty.axis("off")

    # ---- phase space ----
    ax_ps.scatter(x[neg], vx[neg], s=2, alpha=0.35, color="tab:blue",
                  label=fr"$v_x < 0$  ({neg.sum()} particles)", rasterized=True)
    ax_ps.scatter(x[pos], vx[pos], s=2, alpha=0.35, color="tab:orange",
                  label=fr"$v_x \geq 0$  ({pos.sum()} particles)", rasterized=True)
    ax_ps.axhline(+V_DRIFT, color="k", ls="--", lw=1.0, alpha=0.6,
                  label=fr"$\pm v_{{\rm drift}} = \pm {V_DRIFT}$")
    ax_ps.axhline(-V_DRIFT, color="k", ls="--", lw=1.0, alpha=0.6)
    ax_ps.set_xlim(0, L)
    ax_ps.set_ylim(vmin, vmax)
    ax_ps.set_xlabel("x")
    ax_ps.set_ylabel(r"$v_x$")
    ax_ps.set_title(f"Initial phase space   ($N = {len(x)}$,  $L = {L:.3f}$)")
    ax_ps.grid(True, ls="--", alpha=0.3)
    ax_ps.legend(loc="upper right", fontsize=9)

    # ---- vx histogram (right) ----
    bins_v = max(40, min(200, int(np.sqrt(len(vx)))))
    ax_vh.hist(vx, bins=bins_v, orientation="horizontal",
               color="steelblue", edgecolor="none", alpha=0.75)
    ax_vh.axhline(+V_DRIFT, color="k", ls="--", lw=1.0, alpha=0.6)
    ax_vh.axhline(-V_DRIFT, color="k", ls="--", lw=1.0, alpha=0.6)
    ax_vh.set_ylim(vmin, vmax)
    ax_vh.set_xlabel("count")
    ax_vh.set_title(r"$v_x$ histogram")
    ax_vh.grid(True, ls="--", alpha=0.3)

    # ---- x histogram (bottom): one bin per cell (width = dx), zoomed on δn sin(kx) ----
    k = 1.0 / WAVELENGTH
    edges = np.linspace(0.0, L, n_cells + 1)
    counts, _ = np.histogram(x, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_count = float(np.mean(counts))

    n_analytic = 1.0 + DELTA_N * np.sin(k * centers)
    expected = (len(x) / n_cells) * n_analytic

    ax_xh.plot(centers, counts, ".-", color="steelblue", ms=4, lw=1.2,
               label="particle count / cell")
    ax_xh.plot(centers, expected, "k--", lw=1.8,
               label=fr"$N/n_{{\rm cells}}\,[1 + {DELTA_N}\sin(kx)]$")
    ax_xh.axhline(mean_count, color="gray", ls=":", lw=1.0, alpha=0.8,
                  label=fr"mean = {mean_count:.1f}")

    ax_xh.set_xlim(0, L)
    ax_xh.set_xlabel("x")
    ax_xh.set_ylabel("count / cell")
    ax_xh.set_title(fr"Initial $x$ distribution  ({n_cells} cells, $\mathrm{{d}}x={dx:.4f}$)")
    ax_xh.grid(True, ls="--", alpha=0.3)
    ax_xh.legend(loc="upper right", fontsize=8)

    mean_neg = float(np.mean(vx[neg])) if neg.any() else float("nan")
    mean_pos = float(np.mean(vx[pos])) if pos.any() else float("nan")
    std_neg  = float(np.std(vx[neg]))  if neg.any() else float("nan")
    std_pos  = float(np.std(vx[pos]))  if pos.any() else float("nan")
    print(f"N = {len(x)}  (neg={neg.sum()}, pos={pos.sum()})")
    print(f"grid: n_cells = {n_cells},  dx = {dx:.6f},  L = {L:.6f}")
    print(f"<vx>_neg = {mean_neg:.6f}  ± {std_neg:.6f}")
    print(f"<vx>_pos = {mean_pos:.6f}  ± {std_pos:.6f}")
    print(f"vx range = [{vx.min():.6f}, {vx.max():.6f}]")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
