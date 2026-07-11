#!/usr/bin/env python3
"""
Plot kinetic and electrostatic energy vs time from the PIC scalar output,
and fit the Landau damping rate from the electrostatic-energy decay.

Reads:  output/scalars/scalars_<step>.csv   (each: header "ES_energy,KE" + one row)
Assumes the filename index is the timestep number, so time = step * DT.
Run from the same directory that contains the `output/` folder.
"""

import glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")               # save to file (no display needed)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DT = 0.04          # time between saved snapshots (= dt * output stride). Both scalars and particles
                   # are written with the same output index, so this DT works for both.
GAMMA = 1.7        # analytic linear damping rate for the reference line  E0 * exp(-GAMMA * t)
                   # (ES ~ exp(-2*gamma_field*t), so use 2*|gamma_field|, e.g. ~1.7 for k=1)
SCALAR_DIR = "output/scalars"
PART_DIR   = "output/particles"

# ---- load all scalar files, sorted by timestep ----
rows = []
for path in glob.glob(os.path.join(SCALAR_DIR, "scalars_*.csv")):
    m = re.search(r"scalars_(\d+)\.csv", os.path.basename(path))
    if not m:
        continue
    step = int(m.group(1))
    with open(path) as f:
        f.readline()                              # skip header
        es, ke = (float(x) for x in f.readline().split(","))
    rows.append((step, es, ke))

if not rows:
    raise SystemExit(f"No scalar files found in {SCALAR_DIR}/ — run the sim first.")

rows.sort()
step = np.array([r[0] for r in rows])
t    = step * DT
ES   = np.array([r[1] for r in rows])
KE   = np.array([r[2] for r in rows])

# ---- fit the Landau damping rate from the ES decay ----
# ES ~ exp(-2*gamma*t), so slope of ln(ES) = -2*gamma. Fit only the initial
# decaying region, before ES flattens out at the particle-noise floor.
plateau = np.median(ES[t > t.max() * 0.5])        # noise floor (late-time median)
thresh  = 5.0 * plateau
end = np.argmax(ES < thresh)                       # first index that drops into the floor
if end < 3:                                        # safety: need a few points to fit
    end = max(3, np.searchsorted(t, 2.0))
tf, ESf = t[:end], ES[:end]
slope, intercept = np.polyfit(tf, np.log(ESf), 1)
gamma = -slope / 2.0

# ---- plot ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

# kinetic energy (what you asked for)
ax1.plot(t, KE, color="tab:blue")
ax1.set_ylabel("Kinetic energy")
ax1.set_title("Kinetic energy vs time")
ax1.grid(True, ls="--", alpha=0.5)

# electrostatic energy on a log axis + damping-rate fit
ax2.semilogy(t, ES, ".-", color="tab:red", label="electrostatic energy")

# analytic linear damping reference: starts at the peak energy E0, decays as exp(-GAMMA t)
E0 = ES.max()
t_ref = np.linspace(0, 5, 200)
ax2.semilogy(t_ref, E0 * np.exp(-GAMMA * t_ref), "k--", lw=2,
             label="analytic linear damping rate")

ax2.set_ylabel("Electrostatic energy (log)")
ax2.set_title("Electrostatic energy vs time  (Landau damping)")
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.legend()

# total energy = kinetic + electrostatic
TOT = KE + ES
ax3.plot(t, TOT, color="tab:green")
ax3.set_ylim(0, TOT.max()+1)          # 0 -> peak, so the ~flat total shows conservation
ax3.set_xlabel("time  " + r"$\tilde t = \omega_{pe} t$")
ax3.set_ylabel("Total energy")
ax3.set_title("Total energy (KE + ES) vs time")
ax3.grid(True, ls="--", alpha=0.5)

fig.tight_layout()
fig.savefig("energy_vs_time.png", dpi=130)
print(f"measured Landau damping rate  gamma = {gamma:.4f}")
print("saved energy_vs_time.png")

"""
# ================= phase-space animation (x, vx) =================
ppaths = sorted(glob.glob(os.path.join(PART_DIR, "particle_*.csv")),
                key=lambda p: int(re.search(r"_(\d+)", p).group(1)))
if not ppaths:
    print(f"no particle files in {PART_DIR}/ — skipping phase_space.gif")
else:
    stride = max(1, len(ppaths) // 80)          # aim for ~80 frames
    ppaths = ppaths[::stride]
    frames = []
    for p in ppaths:
        k = int(re.search(r"_(\d+)", p).group(1))
        d = np.loadtxt(p, delimiter=",", skiprows=1)
        frames.append((k, d[:, 0], d[:, 1]))

    figp, axp = plt.subplots(figsize=(8, 5))
    sc = axp.scatter([], [], s=1, alpha=0.25, color="navy")
    axp.set_xlim(0, 2 * np.pi)
    axp.set_ylim(-5, 5)
    axp.set_xlabel("x")
    axp.set_ylabel(r"$v_x$")
    axp.grid(True, ls="--", alpha=0.3)
    titlep = axp.set_title("")

    def _update(i):
        k, x, v = frames[i]
        sc.set_offsets(np.column_stack([x, v]))
        titlep.set_text(fr"phase space   $\tilde t$ = {k * DT:.2f}")
        return sc, titlep

    anim = FuncAnimation(figp, _update, frames=len(frames), blit=False)
    anim.save("phase_space.gif", writer=PillowWriter(fps=12))
    print(f"saved phase_space.gif  ({len(frames)} frames)")

"""