#!/usr/bin/env python3
"""
Two-stream electrostatic energy growth overlay.

Loads ES(t) from a run, fits the linear-growth window
  ln(ES) ≈ intercept + (2 γ_sim) t,
and overlays:
  - the measured exponential (slope from the fit)
  - the theoretical exponential (slope 2 γ_theory), anchored at the
    same (t0, ES(t0)) so the two slopes are easy to compare by eye.

Edit RUN_DIR below (or pass a path as argv[1]).

Expected layout:
  <run_dir>/scalars/scalars_*.csv   (or <run_dir>/output/scalars/)
  <run_dir>/conditions.txt          (optional; used for k, v_drift)

Time between scalar files: DT = dt * scaling = 0.04.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Defaults (match initialization.cpp twostream block)
# ---------------------------------------------------------------------------
DT = 0.04
K_DEFAULT = 0.5
V_DEFAULT = 0.5
OMEGA_PE1_SQ = 0.5

# Which run to analyze. Override with:  python plot_es_growth_twostream.py <run_dir>
RUN_DIR = "output"

# Optional manual fit window in simulation time. None → auto.
FIT_WINDOW: Optional[tuple[float, float]] = None
# Auto-fit window as fractions of peak ES (skip early transient bump).
FIT_START_FRAC_OF_PEAK = 1e-2
FIT_FRAC_OF_PEAK = 0.15

OUT_DIR = "figures"
OUT_FILE = os.path.join(OUT_DIR, "es_growth_overlay_twostream.png")


def gamma_theory(k: float, V: float, omega_pe1_sq: float = OMEGA_PE1_SQ) -> float:
    """Cold equal-density two-stream growth rate (amplitude ~ e^{γ t})."""
    kv2 = (k * V) ** 2
    disc = np.sqrt(4.0 * kv2 * omega_pe1_sq + omega_pe1_sq**2)
    inside = disc - (kv2 + omega_pe1_sq)
    if inside <= 0:
        return 0.0
    return float(np.sqrt(inside))


def parse_conditions(path: str) -> dict:
    out: dict = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.search(r"(\w+)\s*=\s*([-+0-9.eE]+)", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            try:
                out[key] = float(val) if ("." in val or "e" in val.lower()) else int(val)
            except ValueError:
                continue
    return out


def find_scalar_dir(run_dir: str) -> str:
    candidates = [
        os.path.join(run_dir, "output", "scalars"),
        os.path.join(run_dir, "output_mid_res", "scalars"),
        os.path.join(run_dir, "scalars"),
        run_dir,
    ]
    for c in candidates:
        if glob.glob(os.path.join(c, "scalars_*.csv")):
            return c
    raise FileNotFoundError(
        f"No scalars_*.csv under {run_dir!r}. "
        "Save the run's output/ folder into the run directory first."
    )


def load_es(scalar_dir: str, dt: float = DT):
    rows = []
    for path in glob.glob(os.path.join(scalar_dir, "scalars_*.csv")):
        m = re.search(r"scalars_(\d+)\.csv", os.path.basename(path))
        if not m:
            continue
        step = int(m.group(1))
        with open(path) as f:
            f.readline()
            es, _ke = (float(x) for x in f.readline().split(","))
        rows.append((step, es))
    if not rows:
        raise FileNotFoundError(f"Empty scalar dir: {scalar_dir}")
    rows.sort()
    step = np.array([r[0] for r in rows], dtype=float)
    ES = np.array([r[1] for r in rows], dtype=float)
    return step * dt, ES


def fit_window_mask(
    t: np.ndarray,
    ES: np.ndarray,
    fit_window: Optional[tuple[float, float]] = FIT_WINDOW,
    frac_start: float = FIT_START_FRAC_OF_PEAK,
    frac_end: float = FIT_FRAC_OF_PEAK,
) -> np.ndarray:
    if fit_window is not None:
        t0, t1 = fit_window
        return (t >= t0) & (t <= t1) & (ES > 0)

    # Peak-relative window: start after early bump, end before saturation.
    i_peak = int(np.argmax(ES))
    es_peak = max(float(ES[i_peak]), 1e-30)
    if es_peak < 50.0 * max(float(ES[0]), 1e-30):
        raise RuntimeError(
            "ES never grows far above the seed — cannot auto-pick a linear window "
            "(run longer, or set FIT_WINDOW manually)."
        )

    es_start = frac_start * es_peak
    es_end = frac_end * es_peak

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

    return (np.arange(len(ES)) >= i0) & (np.arange(len(ES)) <= i1) & (ES > 0)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR

    cond = parse_conditions(os.path.join(run_dir, "conditions.txt"))
    # also check parent / run_dir/output layouts
    if not cond:
        cond = parse_conditions(os.path.join(os.path.dirname(run_dir.rstrip("/")), "conditions.txt"))
    if not cond and os.path.isdir(run_dir):
        for cand in (
            os.path.join(run_dir, "conditions.txt"),
            "conditions.txt",
        ):
            cond = parse_conditions(cand)
            if cond:
                break

    k = float(cond.get("k", K_DEFAULT))
    V = float(cond.get("v_drift", V_DEFAULT))
    g_th = gamma_theory(k, V)

    scalar_dir = find_scalar_dir(run_dir)
    t, ES = load_es(scalar_dir)
    mask = fit_window_mask(t, ES)
    tf, ESf = t[mask], ES[mask]
    if len(tf) < 3:
        raise SystemExit("Need ≥3 points in the fit window to estimate γ.")

    slope_sim, intercept = np.polyfit(tf, np.log(ESf), 1)
    g_sim = float(slope_sim / 2.0)
    t0, t1 = float(tf[0]), float(tf[-1])
    es0 = float(ESf[0])  # anchor both overlays at the start of the fit window

    # ES ∝ e^{2 γ t}  →  slopes on ln(ES) are 2γ
    slope_th = 2.0 * g_th
    # theory line through (t0, es0):  ln ES = ln(es0) + 2 γ_th (t - t0)
    intercept_th = np.log(es0) - slope_th * t0

    print(f"run: {run_dir}")
    print(f"  k={k}, V={V}  →  γ_theory = {g_th:.5f}")
    print(f"  γ_sim    = {g_sim:.5f}   (fit t∈[{t0:.2f},{t1:.2f}])")
    print(f"  |Δγ|     = {abs(g_sim - g_th):.5f}")
    print(f"  ln(ES) slopes:  sim={slope_sim:.5f}  theory={slope_th:.5f}")

    t_fit = np.linspace(t0, t1, 200)
    ES_fit = np.exp(intercept + slope_sim * t_fit)
    ES_th = np.exp(intercept_th + slope_th * t_fit)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(t, np.maximum(ES, 1e-30), ".-", color="tab:red", ms=3,
                label="electrostatic energy")
    ax.axvspan(t0, t1, color="0.85", alpha=0.6, label="fit window", zorder=0)
    ax.semilogy(
        t_fit, ES_fit, "k--", lw=2.2,
        label=rf"fit  $ES \propto e^{{2\gamma_{{\rm sim}} t}}$,  "
              rf"$\gamma_{{\rm sim}}={g_sim:.4f}$",
    )
    ax.semilogy(
        t_fit, ES_th, "b-.", lw=2.2,
        label=rf"theory  $ES \propto e^{{2\gamma_{{\rm th}} t}}$,  "
              rf"$\gamma_{{\rm th}}={g_th:.4f}$",
    )

    ax.set_xlabel(r"time  $\tilde t = \omega_{pe} t$")
    ax.set_ylabel("Electrostatic energy (log)")
    ax.set_title("Two-stream ES growth: measured vs theoretical slope")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    # if analyzing a named run folder, tag the output filename
    out = OUT_FILE
    tag = os.path.basename(os.path.normpath(run_dir))
    if tag and tag not in ("output", ".", ""):
        stem, ext = os.path.splitext(OUT_FILE)
        out = f"{stem}_{tag}{ext}"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
