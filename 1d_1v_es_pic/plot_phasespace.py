#!/usr/bin/env python3
"""
Build a phase-space (x, vx) animation from the PIC particle output.

Reads:  output/particles/particle_<step>.csv   (header "x,vx" + one row per particle)
Writes: figures/phase_space.gif
Run from the directory that contains the `output/` folder.
"""

import glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- time per saved snapshot ---------------------------------------------
# run_loop saves every OUTPUT_EVERY steps and names files by output index,
# so consecutive files are OUTPUT_EVERY*dt apart in time.
DT_SIM       = 0.04 / 1     # dt in initialization.cpp (dt = 0.04 / scaling)
OUTPUT_EVERY = 1            # the "nt % scaling == 0" cadence in run_loop
FRAME_DT     = DT_SIM * OUTPUT_EVERY
FPS          = 12

# ---- collect and load snapshots, sorted by timestep ----
paths = sorted(glob.glob("output/particles/particle_*.csv"),
               key=lambda p: int(re.search(r"_(\d+)", p).group(1)))
if not paths:
    raise SystemExit("No particle files in output/particles/ — run the sim first.")
stride = max(1, len(paths) // 80)     # aim for ~80 frames
paths = paths[::stride]

frames = []
xmin, xmax = np.inf, -np.inf
vmin, vmax = np.inf, -np.inf
for p in paths:
    step = int(re.search(r"_(\d+)", p).group(1))
    data = np.loadtxt(p, delimiter=",", skiprows=1)
    x, v = data[:, 0], data[:, 1]
    frames.append((step, x, v))
    xmin = min(xmin, float(np.min(x)))
    xmax = max(xmax, float(np.max(x)))
    vmin = min(vmin, float(np.min(v)))
    vmax = max(vmax, float(np.max(v)))

# Prefer domain length from the field grid (avoids hardcoding L = 2π).
field0 = "output/fields/fields_0.csv"
if os.path.isfile(field0):
    fields = np.loadtxt(field0, delimiter=",", skiprows=1)
    n_cells = len(fields)
    dx = float(fields[1, 0] - fields[0, 0]) if n_cells > 1 else float(2.0 * fields[0, 0])
    L = n_cells * dx
else:
    L = xmax  # particles live in [0, L)

vpad = 0.15 * max(1.0, abs(vmin), abs(vmax))

# ---- animate ----
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter([], [], s=1, alpha=0.25, color="navy")
ax.set_xlim(0, L)
ax.set_ylim(vmin - vpad, vmax + vpad)
ax.set_xlabel("x")
ax.set_ylabel(r"$v_x$")
ax.grid(True, ls="--", alpha=0.3)
title = ax.set_title("")

def update(k):
    step, x, v = frames[k]
    sc.set_offsets(np.column_stack([x, v]))
    title.set_text(fr"phase space   $\tilde t$ = {step * FRAME_DT:.2f}")
    return sc, title

anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
os.makedirs("figures", exist_ok=True)
out = "figures/phase_space.gif"
anim.save(out, writer=PillowWriter(fps=FPS))
print(f"saved {out}  ({len(frames)} frames,  L = {L:.4f})")
