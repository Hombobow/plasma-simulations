#!/usr/bin/env python3
"""
Plot kinetic / electrostatic / total energy vs time for the two-stream run,
fit the measured linear growth rate, and overlay the cold-beam theoretical γ.

Usage:
  python3 plot_energy_twostream.py
  python3 plot_energy_twostream.py output/twostream
  python3 plot_energy_twostream.py output/twostream --k 0.5 --v-drift 0.5

Reads scalars_*.csv from <run_dir>/scalars/ (or <run_dir>/output/scalars/).
Always writes under figures/.
Assumes filename index corresponds to time t = index * DT.

Theory (equal cold beams, PDF §6.3.2): streams at ±V with ω_pe,1² each,
  γ = [√(4 k² V² ω_pe,1² + ω_pe,1⁴) − (k² V² + ω_pe,1²)]^{1/2}
  unstable when k V < √2 ω_pe,1.
In normalized units with total n0 = 1 → ω_pe² = 1, each stream has ω_pe,1² = 1/2.
ES energy grows as exp(2 γ t).
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# Match initialization.cpp: dt = 0.04 / scaling, output every `scaling` steps.
# Output file index advances by 1 each save → Δt between files = dt * scaling = 0.04.
DT = 0.04
DEFAULT_RUN_DIR = "output/twostream"
DEFAULT_OUT_FILE = "figures/energy_vs_time_twostream.png"
# Auto-fit window as fractions of peak ES (skip early transient bump).
FIT_START_FRAC_OF_PEAK = 1e-2
FIT_FRAC_OF_PEAK = 0.15

# --- theoretical growth rate: tune to match initialization.cpp ---
# wavelength = 2 (units of 2π) → k = 1/wavelength = 0.5; v_drift = 0.5
K = 1/2
V_DRIFT = 0.8
# Each stream has n = n0/2 → ω_pe,1² = 1/2 when total ω_pe² = 1
OMEGA_PE1_SQ = 0.5


def theoretical_growth_rate(k: float, V: float, omega_pe1_sq: float = OMEGA_PE1_SQ) -> float:
    """Cold equal-beam two-stream growth rate γ (PDF Eq. 6.2 / §6.3.2)."""
    kv2 = (k * V) ** 2
    w2 = omega_pe1_sq
    inside = np.sqrt(4.0 * kv2 * w2 + w2 * w2) - (kv2 + w2)
    if inside <= 0.0:
        return 0.0  # stable (k V >= √2 ω_pe,1)
    return float(np.sqrt(inside))


def find_scalar_dir(run_dir: str) -> str:
    candidates = [
        os.path.join(run_dir, "output", "scalars"),
        os.path.join(run_dir, "scalars"),
        run_dir if run_dir.endswith("scalars") else "",
        "output/twostream/scalars" if run_dir in (".", "output") else "",
    ]
    for c in candidates:
        if c and glob.glob(os.path.join(c, "scalars_*.csv")):
            return c
    raise SystemExit(
        f"No scalars_*.csv under {run_dir!r}. "
        "Expected <run_dir>/scalars/ or <run_dir>/output/scalars/."
    )


def out_path_for(run_dir: str) -> str:
    """Always save under figures/, naming by run folder when not the default."""
    name = os.path.basename(os.path.normpath(run_dir))
    if run_dir in (".", "output", DEFAULT_RUN_DIR) or name in ("twostream", "scalars"):
        return DEFAULT_OUT_FILE
    return os.path.join("figures", f"energy_vs_time_{name}.png")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot two-stream energies and compare measured vs theoretical γ."
    )
    p.add_argument(
        "run_dir",
        nargs="?",
        default=DEFAULT_RUN_DIR,
        help=f"run directory (default: {DEFAULT_RUN_DIR})",
    )
    p.add_argument(
        "--k",
        type=float,
        default=K,
        help=f"wavenumber k (default: {K}, matches wavelength=2 → k=0.5)",
    )
    p.add_argument(
        "--v-drift",
        type=float,
        default=V_DRIFT,
        dest="v_drift",
        help=f"beam drift speed V (default: {V_DRIFT})",
    )
    p.add_argument(
        "--omega-pe1-sq",
        type=float,
        default=OMEGA_PE1_SQ,
        dest="omega_pe1_sq",
        help=f"ω_pe,1² per stream (default: {OMEGA_PE1_SQ} for n1=n2=n0/2)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    k = args.k
    V = args.v_drift
    omega_pe1_sq = args.omega_pe1_sq

    scalar_dir = find_scalar_dir(run_dir)
    out_file = out_path_for(run_dir)

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

    gamma_th = theoretical_growth_rate(k, V, omega_pe1_sq)
    omega_pe1 = np.sqrt(omega_pe1_sq)
    kv = k * V
    unstable = kv < np.sqrt(2.0) * omega_pe1

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

    # Theoretical ES ~ exp(2 γ_th t), anchored in the linear window (or at t=0).
    if gamma_th > 0.0:
        if len(tf) >= 1:
            t0, E0 = float(tf[0]), float(ESf[0])
            t_end = float(tf[-1])
        else:
            t0, E0 = float(t[0]), float(max(ES[0], 1e-30))
            t_end = float(t[min(len(t) - 1, max(1, int(0.3 * len(t))))])
        t_th = np.linspace(t0, t_end, 200)
        ax2.semilogy(
            t_th,
            E0 * np.exp(2.0 * gamma_th * (t_th - t0)),
            color="tab:green",
            ls=":",
            lw=2.5,
            label=fr"theory  $\gamma = {gamma_th:.4f}$  ($k={k:g}$, $V={V:g}$)",
        )

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
    print(
        f"theory:  k={k:g},  V={V:g},  ω_pe,1²={omega_pe1_sq:g}  →  "
        f"γ_theory = {gamma_th:.4f}  "
        f"({'unstable' if unstable else 'STABLE'} since kV={kv:.4f} "
        f"{'<' if unstable else '>='} √2 ω_pe,1={np.sqrt(2)*omega_pe1:.4f})"
    )
    if np.isfinite(gamma):
        print(
            f"measured: γ ≈ {gamma:.4f}  "
            f"(fit window t∈[{tf[0]:.2f},{tf[-1]:.2f}])  "
            f"  |γ − γ_theory| = {abs(gamma - gamma_th):.4f}"
        )
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
