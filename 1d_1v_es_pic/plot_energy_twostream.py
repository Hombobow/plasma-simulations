#!/usr/bin/env python3
"""
Plot kinetic / electrostatic / total energy vs time for the two-stream run,
and estimate the linear growth rate from the early ES exponential rise.

Usage:
  python3 plot_energy_twostream.py                  # ./output
  python3 plot_energy_twostream.py figures/ncells=64
  for d in figures/ncells=*; do python3 plot_energy_twostream.py "$d"; done

Reads scalars_*.csv from <run_dir>/scalars/ (or <run_dir>/output/scalars/).
Assumes filename index k corresponds to time t = k * DT.
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

# Match initialization.cpp: dt = 0.04 / scaling, output every `scaling` steps.
# Output file index advances by 1 each save → Δt between files = dt * scaling = 0.04.
DT = 0.04
DEFAULT_RUN_DIR = "output"
DEFAULT_OUT_FILE = "figures/energy_vs_time_twostream.png"
# Auto-fit window as fractions of peak ES (skip early transient bump).
FIT_START_FRAC_OF_PEAK = 1e-2
FIT_FRAC_OF_PEAK = 0.15


def find_scalar_dir(run_dir: str) -> str:
    candidates = [
        os.path.join(run_dir, "output", "scalars"),
        os.path.join(run_dir, "scalars"),
        run_dir if run_dir.endswith("scalars") else "",
        "output/scalars" if run_dir in (".", "output") else "",
    ]
    for c in candidates:
        if c and glob.glob(os.path.join(c, "scalars_*.csv")):
            return c
    raise SystemExit(
        f"No scalars_*.csv under {run_dir!r}. "
        "Expected <run_dir>/scalars/ or <run_dir>/output/scalars/."
    )


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    scalar_dir = find_scalar_dir(run_dir)

    # Save into the run folder when analyzing a saved run; else default path.
    if run_dir in (".", "output", DEFAULT_RUN_DIR):
        out_file = DEFAULT_OUT_FILE
    else:
        out_file = os.path.join(run_dir, "energy_vs_time_twostream.png")

    rows = []
    for path in glob.glob(os.path.join(scalar_dir, "scalars_*.csv")):
        m = re.search(r"scalars_(\d+)\.csv", os.path.basename(path))
        if not m:
            continue
        step = int(m.group(1))
        with open(path) as f:
            f.readline()
            es, ke = (float(x) for x in f.readline().split(","))
        rows.append((step, es, ke))

    if not rows:
        raise SystemExit(f"No scalar files found in {scalar_dir}/ — run the sim first.")

    rows.sort()
    step = np.array([r[0] for r in rows])
    t = step * DT
    ES = np.array([r[1] for r in rows])
    KE = np.array([r[2] for r in rows])
    TOT = KE + ES

    # ---- fit linear growth rate: ES ~ exp(2 γ t) ----
    # Peak-relative window: start after early bump, end before saturation.
    i_peak = int(np.argmax(ES))
    es_peak = max(float(ES[i_peak]), 1e-30)
    slope = intercept = gamma = np.nan
    tf = t[:0]
    ESf = ES[:0]

    if es_peak >= 50.0 * max(float(ES[0]), 1e-30):
        es_start = FIT_START_FRAC_OF_PEAK * es_peak
        es_end = FIT_FRAC_OF_PEAK * es_peak

        below = np.where((np.arange(len(ES)) < i_peak) & (ES < es_start))[0]
        if len(below) > 0:
            i0 = int(below[-1]) + 1
        else:
            i0 = 1
        i0 = max(1, min(i0, i_peak - 3))

        i1 = i0
        while i1 < i_peak and ES[i1] < es_end:
            i1 += 1
        if i1 <= i0 + 2:
            i1 = max(i0 + 3, min(i_peak, len(ES) - 1))

        mask = (np.arange(len(ES)) >= i0) & (np.arange(len(ES)) <= i1) & (ES > 0)
        tf, ESf = t[mask], ES[mask]
        if len(tf) >= 3:
            slope, intercept = np.polyfit(tf, np.log(ESf), 1)
            gamma = slope / 2.0  # ES ∝ exp(2 γ t)
    else:
        print("ES has not grown enough yet to auto-fit γ (run longer).")

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
    ax3.set_xlabel(r"time  $\tilde t = \omega_{pe} t$")
    ax3.set_ylabel("Energy")
    ax3.set_title("Total energy (KE + ES) vs time")
    ax3.grid(True, ls="--", alpha=0.5)
    ax3.legend(loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, dpi=130)
    plt.close(fig)

    print(f"run: {run_dir}")
    print(f"t=0:  ES = {ES[0]:.6g},  KE = {KE[0]:.6g},  total = {TOT[0]:.6g}")
    if np.isfinite(gamma):
        print(
            f"measured two-stream growth rate  gamma ≈ {gamma:.4f}  "
            f"(fit window t∈[{tf[0]:.2f},{tf[-1]:.2f}])"
        )
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
