#!/usr/bin/env python3
"""
Plot the initial phase-space (x, vx) from the first particle snapshot.

Reads:  output/particles/particle_0.csv   (header "x,vx" + one row per particle)
Writes: figures/initial_phase_space.png
Run from the directory that contains the `output/` folder.

Use this to check two-stream / Landau IC before looking at the full animation:
cold beams should sit on two horizontal lines at ±v_drift; a density seed
appears as a weak modulation in the x-histogram.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PART_FILE  = "output/particles/particle_0.csv"
FIELD_FILE = "output/fields/fields_0.csv"
OUT_FILE   = "figures/initial_phase_space.png"
V_DRIFT    = 0.5    # expected beam centers for twostream (match initialization.cpp)
DELTA_N    = 0.001  # density seed amplitude (match initialization.cpp)
WAVELENGTH = 2.0   # in units of 2π → k = 1/WAVELENGTH (match initialization.cpp)

if not os.path.isfile(PART_FILE):
    raise SystemExit(f"No {PART_FILE} — run the sim first so output index 0 exists.")

data = np.loadtxt(PART_FILE, delimiter=",", skiprows=1)
x, vx = data[:, 0], data[:, 1]

# Bin width = cell size dx from the field grid (one row per cell in fields_*.csv).
# Falls back to the same formula as initialization.cpp if fields are missing.
if os.path.isfile(FIELD_FILE):
    fields = np.loadtxt(FIELD_FILE, delimiter=",", skiprows=1)
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
# With δn ~ 0.001, variation is ~0.1% of the mean — a 0→max y-axis hides it completely.
k = 1.0 / WAVELENGTH
edges = np.linspace(0.0, L, n_cells + 1)          # cell faces: 0, dx, 2dx, ..., L
counts, _ = np.histogram(x, bins=edges)
centers = 0.5 * (edges[:-1] + edges[1:])          # cell centers (same as fields x)
mean_count = float(np.mean(counts))

# Analytic loading: n(x) ∝ 1 + δn sin(kx). Scale to expected count/cell:
# mean particles per cell = N * (dx / L) = N / n_cells.
n_analytic = 1.0 + DELTA_N * np.sin(k * centers)
expected = (len(x) / n_cells) * n_analytic

ax_xh.plot(centers, counts, ".-", color="steelblue", ms=4, lw=1.2,
           label="particle count / cell")
ax_xh.plot(centers, expected, "k--", lw=1.8,
           label=fr"$N/n_{{\rm cells}}\,[1 + {DELTA_N}\sin(kx)]$")
ax_xh.axhline(mean_count, color="gray", ls=":", lw=1.0, alpha=0.8,
              label=fr"mean = {mean_count:.1f}")

# Zoom y-axis on the modulation (not 0 → max). Pad by the larger of
# expected amplitude or a few stddevs of Poisson noise.
amp = max(DELTA_N * mean_count, 3.0 * np.sqrt(max(mean_count, 1.0)))
pad = 2.5 * amp
# ax_xh.set_ylim(mean_count - pad, mean_count + pad)
ax_xh.set_xlim(0, L)
ax_xh.set_xlabel("x")
ax_xh.set_ylabel("count / cell")
ax_xh.set_title(fr"Initial $x$ distribution  ({n_cells} cells, $\mathrm{{d}}x={dx:.4f}$)")
ax_xh.grid(True, ls="--", alpha=0.3)
ax_xh.legend(loc="upper right", fontsize=8)

# quick beam diagnostics
mean_neg = float(np.mean(vx[neg])) if neg.any() else float("nan")
mean_pos = float(np.mean(vx[pos])) if pos.any() else float("nan")
std_neg  = float(np.std(vx[neg]))  if neg.any() else float("nan")
std_pos  = float(np.std(vx[pos]))  if pos.any() else float("nan")
print(f"N = {len(x)}  (neg={neg.sum()}, pos={pos.sum()})")
print(f"grid: n_cells = {n_cells},  dx = {dx:.6f},  L = {L:.6f}")
print(f"<vx>_neg = {mean_neg:.6f}  ± {std_neg:.6f}")
print(f"<vx>_pos = {mean_pos:.6f}  ± {std_pos:.6f}")
print(f"vx range = [{vx.min():.6f}, {vx.max():.6f}]")

os.makedirs("figures", exist_ok=True)
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=140)
print(f"saved {OUT_FILE}")
