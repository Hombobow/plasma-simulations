#!/usr/bin/env python3
"""
Build a phase-space (x, vx) animation from the PIC particle output.

Reads:  <particles_dir>/particle_<step>.csv   (header "x,vx" + one row per particle)
Writes: figures/phase_space.gif (or --output path)

Examples:
  python plot_phasespace.py
  python plot_phasespace.py output/landaudamping/particles
  python plot_phasespace.py output/twostream/particles -o figures/twostream_phase.gif
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- time per saved snapshot ---------------------------------------------
# run_loop saves every OUTPUT_EVERY steps and names files by output index,
# so consecutive files are OUTPUT_EVERY*dt apart in time.
DT_SIM       = 0.04 / 1     # dt in initialization.cpp (0.04 / scaling)
OUTPUT_EVERY = 1            # matches scaling: nt % scaling == 0
FRAME_DT     = DT_SIM * OUTPUT_EVERY
FPS          = 12
L            = 2 * np.pi
DEFAULT_DIR  = "output/landaudamping/particles"
DEFAULT_OUT  = "figures/phase_space.gif"


def parse_args():
    p = argparse.ArgumentParser(description="Make a phase-space GIF from particle CSVs.")
    p.add_argument(
        "particles_dir",
        nargs="?",
        default=DEFAULT_DIR,
        help=f"directory with particle_*.csv (default: {DEFAULT_DIR})",
    )
    p.add_argument(
        "-o", "--output",
        default=DEFAULT_OUT,
        help=f"output GIF path (default: {DEFAULT_OUT})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    particles_dir = Path(args.particles_dir)
    out_gif = Path(args.output)

    # ---- collect and load snapshots, sorted by timestep ----
    pattern = str(particles_dir / "particle_*.csv")
    paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_(\d+)", p).group(1)),
    )
    if not paths:
        raise SystemExit(f"No particle files in {particles_dir}/ — run the sim first.")

    stride = max(1, len(paths) // 80)     # aim for ~80 frames
    paths = paths[::stride]

    frames = []
    for p in paths:
        step = int(re.search(r"_(\d+)", p).group(1))
        data = np.loadtxt(p, delimiter=",", skiprows=1)
        frames.append((step, data[:, 0], data[:, 1]))

    # ---- animate ----
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter([], [], s=1, alpha=0.25, color="navy")
    ax.set_xlim(0, L)
    ax.set_ylim(-5, 5)
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
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer=PillowWriter(fps=FPS))
    print(f"saved {out_gif}  ({len(frames)} frames from {particles_dir})")


if __name__ == "__main__":
    main()
