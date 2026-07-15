#!/usr/bin/env python3
"""
Plot kinetic / electrostatic / total energy vs time for the two-stream run,
and estimate the linear growth rate from the early ES exponential rise.

Reads:  output/scalars/scalars_<step>.csv   (header "ES_energy,KE" + one row)
Assumes filename index k corresponds to time t = k * DT.
Run from the same directory that contains the `output/` folder.
"""

import glob, os, re
import numpy as np
import matplotlib.pyplot as plt

# Match initialization.cpp: dt = 0.04 / scaling, output every `scaling` steps.
# Output file index advances by 1 each save → Δt between files = dt * scaling = 0.04.
DT = 0.04
SCALAR_DIR = "output/scalars"
OUT_FILE = "figures/energy_vs_time_twostream.png"

rows = []
for path in glob.glob(os.path.join(SCALAR_DIR, "scalars_*.csv")):
    m = re.search(r"scalars_(\d+)\.csv", os.path.basename(path))
    if not m:
        continue
    step = int(m.group(1))
    with open(path) as f:
        f.readline()
        es, ke = (float(x) for x in f.readline().split(","))
    rows.append((step, es, ke))

if not rows:
    raise SystemExit(f"No scalar files found in {SCALAR_DIR}/ — run the sim first.")

rows.sort()
step = np.array([r[0] for r in rows])
t    = step * DT
ES   = np.array([r[1] for r in rows])
KE   = np.array([r[2] for r in rows])
TOT  = KE + ES

# ---- fit linear growth rate: ES ~ exp(2 γ t) while still growing ----
# Fit between the first point above a few× the seed and the first local max
# (onset of saturation). Drop points where ES is non-positive.
i_seed = 0
while i_seed + 1 < len(ES) and ES[i_seed + 1] > ES[i_seed]:
    # find start of clear rise: first ES that exceeds 3× initial seed
    if ES[i_seed] > 3.0 * max(ES[0], 1e-30):
        break
    i_seed += 1
# peak before saturation (global max is fine for cold two-stream)
i_peak = int(np.argmax(ES))
i0 = max(1, int(np.searchsorted(ES, 3.0 * max(ES[0], 1e-30))))
i1 = i_peak
if i1 <= i0 + 2:
    i0, i1 = 1, min(len(ES) - 1, max(5, len(ES) // 4))

mask = (np.arange(len(ES)) >= i0) & (np.arange(len(ES)) <= i1) & (ES > 0)
tf, ESf = t[mask], ES[mask]
if len(tf) >= 3:
    slope, intercept = np.polyfit(tf, np.log(ESf), 1)
    gamma = slope / 2.0          # ES ∝ exp(2 γ t)
else:
    slope, gamma = np.nan, np.nan

# ---- plot ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

ax1.plot(t, KE, color="tab:blue")
ax1.set_ylabel("Kinetic energy")
ax1.set_title("Kinetic energy vs time")
ax1.grid(True, ls="--", alpha=0.5)

ax2.semilogy(t, np.maximum(ES, 1e-30), ".-", color="tab:red",
             label="electrostatic energy")
if np.isfinite(gamma) and len(tf) >= 3:
    t_fit = np.linspace(tf[0], tf[-1], 200)
    ax2.semilogy(t_fit, np.exp(intercept + slope * t_fit), "k--", lw=2,
                 label=fr"fit  $ES \propto e^{{2\gamma t}}$,  $\gamma \approx {gamma:.4f}$")
ax2.set_ylabel("Electrostatic energy (log)")
ax2.set_title("Electrostatic energy vs time  (two-stream)")
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.legend()

ax3.plot(t, TOT, color="tab:green", label="total")
ax3.plot(t, KE, color="tab:blue", alpha=0.5, label="KE")
ax3.plot(t, ES, color="tab:red", alpha=0.5, label="ES")
# Zoom near initial total so the seed isn't crushed by late-time scale if energy
# is poorly conserved; still show full trace.
ax3.set_xlabel(r"time  $\tilde t = \omega_{pe} t$")
ax3.set_ylabel("Energy")
ax3.set_title("Total energy (KE + ES) vs time")
ax3.grid(True, ls="--", alpha=0.5)
ax3.legend(loc="best")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT_FILE, dpi=130)
print(f"t=0:  ES = {ES[0]:.6g},  KE = {KE[0]:.6g},  total = {TOT[0]:.6g}")
if np.isfinite(gamma):
    print(f"measured two-stream growth rate  gamma ≈ {gamma:.4f}  (fit window t∈[{tf[0]:.2f},{tf[-1]:.2f}])")
print(f"saved {OUT_FILE}")
