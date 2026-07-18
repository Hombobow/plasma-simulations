#!/usr/bin/env python3
"""
Two-stream growth-rate error analysis.

For each saved run:
  1. load ES(t) from scalars_*.csv
  2. fit ln(ES) ≈ intercept + slope * t  over the linear-growth window
  3. gamma_sim = slope / 2     (because ES ∝ e^{2 γ t} while the field ~ e^{γ t})
  4. gamma_theory from the cold symmetric two-stream formula
  5. abs_error = |gamma_sim - gamma_theory|

Two sweeps (edit the path lists below after you save each run):
  - NPPC_SWEEP:     vary N_ppc at fixed scaling
  - NCELLS_SWEEP:   vary N_cells
  - SCALING_SWEEP:  vary scaling at fixed N_ppc

Expected layout for each run directory:
  <run_dir>/scalars/scalars_*.csv
  <run_dir>/conditions.txt          (optional but preferred)

Default single-run check uses output/twostream/.
Plots always write under figures/.

Time between scalar files is always DT = dt * scaling = 0.04 (see initialization.cpp).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physics defaults (match initialization.cpp twostream block)
# ---------------------------------------------------------------------------
DT = 0.04  # time between successive scalars_<n>.csv files
K_DEFAULT = 0.5  # k = 1 / wavelength, wavelength = 2
V_DEFAULT = 0.5  # v_drift
# each beam has n_j = n0/2 → ω_{pe,1}^2 = 0.5 in units where total ω_pe = 1
OMEGA_PE1_SQ = 0.5

# Optional manual fit window in simulation time. None → auto (linear flank → near peak).
# Example: FIT_WINDOW = (10.0, 25.0)
FIT_WINDOW: Optional[tuple[float, float]] = None

# Auto-fit window as fractions of peak ES. Start high enough to skip the early
# transient bump; end before saturation.
FIT_START_FRAC_OF_PEAK = 1e-2
FIT_FRAC_OF_PEAK = 0.15

OUT_DIR = "figures"
OUT_NPPC = os.path.join(OUT_DIR, "error_vs_nppc_twostream.png")
OUT_NCELLS = os.path.join(OUT_DIR, "error_vs_ncells_twostream.png")
OUT_SCALING = os.path.join(OUT_DIR, "error_vs_scaling_twostream.png")

DEFAULT_RUN_DIR = "output/twostream"

# High-resolution reference run (shown as a horizontal line, not a sweep point).
HIGH_RES_DIR = "output/high_res nppc=128000 ncells=1024"

# ---------------------------------------------------------------------------
# Sweep run directories — fill these in after each PIC run.
# Each entry is (parameter_value, path_to_run_dir).
# Layout: <run_dir>/scalars/scalars_*.csv
# Folder name output/nppc={factor} uses N_ppc = 1000 * 2^factor
# (see initialization.cpp twostream block).
# ---------------------------------------------------------------------------
NPPC_SWEEP: list[tuple[float, str]] = [
    (1000,  "output/nppc=0"),
    (2000,  "output/nppc=1"),
    (4000,  "output/nppc=2"),
    (8000,  "output/nppc=3"),
    (16000, "output/nppc=4"),
    (32000, "output/nppc=5"),
    (64000, "output/nppc=6"),
]

# N_cells sweep (folder name = cell count).
NCELLS_SWEEP: list[tuple[float, str]] = [
    (64,   "output/ncells=64"),
    (128,  "output/ncells=128"),
    (256,  "output/ncells=256"),
    (384,  "output/ncells=384"),
    (512,  "output/ncells=512"),
    (640,  "output/ncells=640"),
    (768,  "output/ncells=768"),
    (896,  "output/ncells=896"),
    (1024, "output/ncells=1024"),
]

# Scaling sweep at fixed N_ppc. High-res (scaling=16) is NOT listed
# here — it is used only as the γ_highres reference line via HIGH_RES_DIR.
SCALING_SWEEP: list[tuple[float, str]] = [
    # add scalings as you run them, e.g. (1, "output/scaling=1"), ...
]


# ---------------------------------------------------------------------------
# Theory + I/O helpers
# ---------------------------------------------------------------------------
def gamma_theory(k: float, V: float, omega_pe1_sq: float = OMEGA_PE1_SQ) -> float:
    """
    Cold, equal-density two-stream growth rate (amplitude ~ e^{γ t}).

    Unstable branch of
      ω² = (k²V² + ω_pe1²) ± √(4 k² V² ω_pe1² + ω_pe1⁴)
    when k V < √2 ω_pe1:
      γ = [√(4 k² V² ω_pe1² + ω_pe1⁴) - (k² V² + ω_pe1²)]^{1/2}
    """
    kv2 = (k * V) ** 2
    disc = np.sqrt(4.0 * kv2 * omega_pe1_sq + omega_pe1_sq**2)
    inside = disc - (kv2 + omega_pe1_sq)
    if inside <= 0:
        return 0.0
    return float(np.sqrt(inside))


def parse_conditions(path: str) -> dict:
    """Parse key = value lines from a conditions.txt dump."""
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
    """Locate scalars_*.csv under a run directory (several naming layouts)."""
    candidates = [
        os.path.join(run_dir, "scalars"),
        os.path.join(run_dir, "output", "scalars"),
        run_dir if run_dir.endswith("scalars") else "",
        run_dir,
    ]
    for c in candidates:
        if c and glob.glob(os.path.join(c, "scalars_*.csv")):
            return c
    raise FileNotFoundError(
        f"No scalars_*.csv under {run_dir!r}. "
        "Expected <run_dir>/scalars/ (e.g. output/twostream/scalars/)."
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


def auto_fit_indices(
    ES: np.ndarray,
    frac_start: float = FIT_START_FRAC_OF_PEAK,
    frac_end: float = FIT_FRAC_OF_PEAK,
) -> tuple[int, int]:
    """
    Indices [i0, i1] covering the linear exponential rise into the main peak.

    Uses peak-relative thresholds so early transient bumps (above the seed but
    still << peak) are skipped. Start = last time ES is below frac_start*peak
    before the peak; end = first time ES reaches frac_end*peak on that flank.
    """
    i_peak = int(np.argmax(ES))
    es_peak = max(float(ES[i_peak]), 1e-30)
    # Incomplete / noise-only runs: peak never leaves the seed floor.
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
    return i0, i1


def fit_gamma_sim(
    t: np.ndarray,
    ES: np.ndarray,
    fit_window: Optional[tuple[float, float]] = FIT_WINDOW,
    frac_of_peak: float = FIT_FRAC_OF_PEAK,
    frac_start: float = FIT_START_FRAC_OF_PEAK,
) -> tuple[float, float, float]:
    """
    Return (gamma_sim, t_fit_start, t_fit_end).

    ES ∝ exp(2 γ t)  →  gamma_sim = (1/2) d/dt ln(ES).
    """
    if fit_window is not None:
        t0, t1 = fit_window
        mask = (t >= t0) & (t <= t1) & (ES > 0)
    else:
        i0, i1 = auto_fit_indices(ES, frac_start=frac_start, frac_end=frac_of_peak)
        mask = (np.arange(len(ES)) >= i0) & (np.arange(len(ES)) <= i1) & (ES > 0)

    tf, ESf = t[mask], ES[mask]
    if len(tf) < 3:
        raise RuntimeError("Need ≥3 points in the fit window to estimate γ.")

    slope, _intercept = np.polyfit(tf, np.log(ESf), 1)
    return float(slope / 2.0), float(tf[0]), float(tf[-1])


def analyze_run(run_dir: str) -> dict:
    cond = parse_conditions(os.path.join(run_dir, "conditions.txt"))
    k = float(cond.get("k", K_DEFAULT))
    V = float(cond.get("v_drift", V_DEFAULT))
    nppc = cond.get("N_ppc", None)
    scaling = cond.get("scaling", None)

    scalar_dir = find_scalar_dir(run_dir)
    t, ES = load_es(scalar_dir)
    g_sim, t0, t1 = fit_gamma_sim(t, ES)
    g_th = gamma_theory(k, V)
    return {
        "run_dir": run_dir,
        "nppc": nppc,
        "scaling": scaling,
        "k": k,
        "V": V,
        "gamma_sim": g_sim,
        "gamma_theory": g_th,
        "abs_error": abs(g_sim - g_th),
        "fit_window": (t0, t1),
    }


def analyze_sweep(sweep: list[tuple[float, str]], param_name: str) -> list[dict]:
    results = []
    for param, run_dir in sweep:
        if not os.path.isdir(run_dir):
            print(f"  skip missing dir: {run_dir}")
            continue
        try:
            r = analyze_run(run_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  skip {run_dir}: {exc}")
            continue
        r["param"] = param
        r["param_name"] = param_name
        results.append(r)
        print(
            f"  {param_name}={param:<8g}  γ_sim={r['gamma_sim']:.5f}  "
            f"γ_th={r['gamma_theory']:.5f}  |Δγ|={r['abs_error']:.5f}  "
            f"fit t∈[{r['fit_window'][0]:.2f},{r['fit_window'][1]:.2f}]  "
            f"({os.path.basename(run_dir)})"
        )
    return results


def fit_error_power_law(
    x: np.ndarray,
    err: np.ndarray,
) -> Optional[tuple[float, float]]:
    """
    Fit |error| ∝ x^slope  via log-log least squares.

    Returns (slope, intercept) for log(err) = slope * log(x) + intercept,
    or None if fewer than 2 positive points.
    """
    mask = (x > 0) & (err > 0) & np.isfinite(x) & np.isfinite(err)
    if np.count_nonzero(mask) < 2:
        return None
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(err[mask]), 1)
    return float(slope), float(intercept)


def plot_error(
    results: list[dict],
    param_name: str,
    xlabel: str,
    out_path: str,
    gamma_highres: Optional[float] = None,
) -> Optional[float]:
    """
    Plot γ and |Δγ| vs the sweep parameter.

    For the absolute-error panel, also fit a power law on log-log axes
    (|error| ∝ param^slope) and return that slope (theory error).
    """
    if not results:
        print(f"  no results to plot for {param_name}")
        return None

    results = sorted(results, key=lambda r: r["param"])
    x = np.array([r["param"] for r in results], dtype=float)
    err = np.array([r["abs_error"] for r in results], dtype=float)
    g_sim = np.array([r["gamma_sim"] for r in results], dtype=float)
    g_th = results[0]["gamma_theory"]
    # xlabel may be mathtext (e.g. $N_{\mathrm{ppc}}$); strip $ for embedding in fit labels
    param_tex = xlabel.strip("$") if "$" in xlabel else rf"\mathrm{{{xlabel}}}"

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax0.plot(x, g_sim, "o-", color="tab:red", label=r"$\gamma_{\rm sim}$")
    ax0.axhline(g_th, color="k", ls="--", lw=1.5,
                label=r"$\gamma_{\rm theory}=%.4f$" % g_th)
    if gamma_highres is not None and np.isfinite(gamma_highres):
        ax0.axhline(
            gamma_highres, color="tab:green", ls="-.", lw=1.8,
            label=r"$\gamma_{\rm highres}=%.4f$" % gamma_highres
            + r"  ($N_{\rm ppc}{=}128000$, $N_{\rm cells}{=}1024$)",
        )
    ax0.set_ylabel(r"growth rate $\gamma$")
    ax0.set_title(rf"Two-stream growth rate vs ${param_tex}$")
    ax0.grid(True, ls="--", alpha=0.5)
    ax0.legend()

    ax1.plot(x, err, "s-", color="tab:blue", label=r"$|\gamma_{\rm sim}-\gamma_{\rm theory}|$")

    slope_theory = None
    fit_th = fit_error_power_law(x, err)
    if fit_th is not None:
        slope_theory, intercept_th = fit_th
        x_fit = np.linspace(x.min(), x.max(), 200)
        err_fit = np.exp(intercept_th) * x_fit**slope_theory
        ax1.plot(
            x_fit, err_fit, ":", color="tab:blue", lw=1.8,
            label=rf"fit $\propto {param_tex}^{{{slope_theory:.3f}}}$",
        )
        print(f"  log-log fit |Δγ|_theory vs {param_name}:  slope = {slope_theory:.4f}")

    if gamma_highres is not None and np.isfinite(gamma_highres):
        err_hr = np.abs(g_sim - gamma_highres)
        ax1.plot(x, err_hr, "D--", color="tab:green",
                 label=r"$|\gamma_{\rm sim}-\gamma_{\rm highres}|$")
        fit_hr = fit_error_power_law(x, err_hr)
        if fit_hr is not None:
            slope_hr, intercept_hr = fit_hr
            x_fit = np.linspace(x.min(), x.max(), 200)
            err_fit = np.exp(intercept_hr) * x_fit**slope_hr
            ax1.plot(
                x_fit, err_fit, ":", color="tab:green", lw=1.8,
                label=rf"fit $\propto {param_tex}^{{{slope_hr:.3f}}}$",
            )
            print(f"  log-log fit |Δγ|_highres vs {param_name}: slope = {slope_hr:.4f}")

    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("absolute error")
    ax1.set_title("Absolute error (convergence)")
    ax1.grid(True, ls="--", alpha=0.5)
    use_log = np.all(x > 0) and len(x) >= 2 and (np.max(x) / np.min(x) >= 4)
    if use_log:
        ax1.set_xscale("log")
        ax0.set_xscale("log")
        # log-log makes the power-law fit a straight line
        if np.any(err > 0):
            ax1.set_yscale("log")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"  saved {out_path}")
    plt.close(fig)
    return slope_theory


def main():
    g0 = gamma_theory(K_DEFAULT, V_DEFAULT)
    print(f"γ_theory (k={K_DEFAULT}, V={V_DEFAULT}, ω_pe1²={OMEGA_PE1_SQ}) = {g0:.6f}")
    print("Note: ES ∝ e^{2γt}, so γ_sim = (1/2) * slope of ln(ES).\n")

    gamma_highres = None
    if os.path.isdir(HIGH_RES_DIR):
        try:
            hr = analyze_run(HIGH_RES_DIR)
            gamma_highres = hr["gamma_sim"]
            print(
                f"γ_highres = {gamma_highres:.5f}  "
                f"(from {HIGH_RES_DIR}, fit t∈[{hr['fit_window'][0]:.2f},{hr['fit_window'][1]:.2f}])\n"
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"could not load HIGH_RES_DIR ({HIGH_RES_DIR}): {exc}\n")

    if not NPPC_SWEEP and not NCELLS_SWEEP and not SCALING_SWEEP:
        print(
            "No sweeps configured.\n"
            "Edit NPPC_SWEEP / NCELLS_SWEEP / SCALING_SWEEP at the top of this file to point at\n"
            "run directories that contain scalars/ (and preferably conditions.txt).\n"
            f"\nQuick single-run check of {DEFAULT_RUN_DIR}:"
        )
        if glob.glob(os.path.join(DEFAULT_RUN_DIR, "scalars", "scalars_*.csv")):
            r = analyze_run(DEFAULT_RUN_DIR)
            print(
                f"  γ_sim={r['gamma_sim']:.5f}  γ_th={r['gamma_theory']:.5f}  "
                f"|Δγ|={r['abs_error']:.5f}  "
                f"fit t∈[{r['fit_window'][0]:.2f},{r['fit_window'][1]:.2f}]"
            )
        else:
            print(f"  (no {DEFAULT_RUN_DIR}/scalars found)")
        return

    if NPPC_SWEEP:
        print("--- N_ppc sweep (fixed scaling) ---")
        print("    (needs scalars_*.csv under each run dir; skips missing dirs)")
        res = analyze_sweep(NPPC_SWEEP, "N_ppc")
        if res:
            slope = plot_error(
                res, "N_ppc", r"$N_{\mathrm{ppc}}$", OUT_NPPC, gamma_highres
            )
            if slope is not None:
                print(f"  → returned slope (theory abs error vs N_ppc) = {slope:.4f}")
        else:
            print(
                "  → no nppc points loaded. Save each run under output/nppc=* "
                "(or update NPPC_SWEEP), then re-run.\n"
                f"  → expected plot path once data exists: {OUT_NPPC}"
            )

    if NCELLS_SWEEP:
        print("--- N_cells sweep ---")
        print("    (needs scalars_*.csv under each run dir; skips missing dirs)")
        res = analyze_sweep(NCELLS_SWEEP, "N_cells")
        if res:
            slope = plot_error(
                res, "N_cells", r"$N_{\mathrm{cells}}$", OUT_NCELLS, gamma_highres
            )
            if slope is not None:
                print(f"  → returned slope (theory abs error vs N_cells) = {slope:.4f}")
        else:
            print(
                "  → no ncells points loaded. Save each run under output/ncells=* "
                "(or update NCELLS_SWEEP), then re-run.\n"
                f"  → expected plot path once data exists: {OUT_NCELLS}"
            )

    if SCALING_SWEEP:
        print("--- scaling sweep (fixed N_ppc) ---")
        res = analyze_sweep(SCALING_SWEEP, "scaling")
        slope = plot_error(res, "scaling", "scaling", OUT_SCALING, gamma_highres)
        if slope is not None:
            print(f"  → returned slope (theory abs error vs scaling) = {slope:.4f}")


if __name__ == "__main__":
    main()