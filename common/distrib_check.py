"""distrib_check.py — per-prompt ACF analysis of Lambda(t).

Diagnostics:
  1. What fraction of prompts yield a well-defined ACF decay time?
  2. Distribution of tau per model (median, IQR, histogram-style bins).
  3. Comparison: Qwen2.5-3B vs Mamba2-1.3B.

The fraction of prompts where tau is *undefined* (ACF decays too fast
for a single-exponential fit) is itself the primary signal: it
separates "noisy residual stream" (Qwen) from "smooth residual stream"
(Mamba).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SM_DIR    = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")
CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")

MAX_LAG = 200
ACF_THRESHOLD = 0.02
MIN_POINTS_FOR_FIT = 4


# ----------------------------------------------------------------------
# Core routines
# ----------------------------------------------------------------------
def acf(x, max_lag=MAX_LAG):
    n = len(x)
    if n < 4:
        return None
    x = x - x.mean()
    if (x * x).sum() <= 0:
        return None
    f = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(f * np.conj(f))[:n]
    if acf_full[0] <= 0:
        return None
    acf_norm = acf_full / acf_full[0]
    return acf_norm[:min(max_lag + 1, n)]


def fit_tau(acf_vals, threshold=ACF_THRESHOLD,
            min_pts=MIN_POINTS_FOR_FIT):
    """Single-exponential fit of ACF. Returns (tau, r2, n_pts) or None."""
    if acf_vals is None or len(acf_vals) < min_pts:
        return None
    k = np.arange(len(acf_vals))
    mask = acf_vals > threshold
    if mask.sum() < min_pts:
        return None
    kk = k[mask].astype(float)
    yy = np.log(acf_vals[mask])
    A = np.vstack([kk, np.ones_like(kk)]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    slope, intercept = coef
    if slope >= 0:
        return None
    tau = -1.0 / slope
    pred = A @ coef
    ss_res = ((yy - pred) ** 2).sum()
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return tau, r2, int(mask.sum())


def fit_tau_double(acf_vals, threshold=ACF_THRESHOLD):
    """Two-timescale fit by coarse grid. Returns (t1, t2, a, r2) or None."""
    if acf_vals is None or len(acf_vals) < 8:
        return None
    k = np.arange(len(acf_vals)).astype(float)
    y = acf_vals
    mask = y > threshold
    if mask.sum() < 6:
        return None
    k = k[mask]
    y = y[mask]
    best = (None, None, None, -np.inf)
    t1_grid = np.array([1, 2, 4, 8, 16])
    t2_grid = np.array([16, 32, 64, 128, 256, 512])
    a_grid = np.array([0.3, 0.5, 0.7, 0.9])
    for t1 in t1_grid:
        for t2 in t2_grid:
            if t2 <= t1:
                continue
            e1 = np.exp(-k / t1)
            e2 = np.exp(-k / t2)
            for a in a_grid:
                pred = a * e1 + (1 - a) * e2
                ss_res = ((y - pred) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                if r2 > best[3]:
                    best = (t1, t2, a, r2)
    return best


# ----------------------------------------------------------------------
# Per-model analysis
# ----------------------------------------------------------------------
def load_lambdas(model):
    sfx = "" if model == "qwen" else f"_{model}"
    for p in [
        SM_DIR / f"L0{sfx}_traj.npz",
        CACHE_DIR / f"sm_L0{sfx}.npz",
    ]:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "lambdas" in d:
                return ([np.asarray(x, dtype=float) for x in d["lambdas"]],
                        p.name)
    return None, None


def analyze(model):
    lambdas, src = load_lambdas(model)
    if lambdas is None:
        print(f"{model}: no data")
        return None

    n_total = 0
    n_single_ok = 0
    n_double_ok = 0
    taus = []
    tau_r2s = []
    double_r2s = []
    n_pts_list = []

    for lam in lambdas:
        if len(lam) < 64:
            continue
        n_total += 1
        a = acf(lam)
        if a is None:
            continue
        single = fit_tau(a)
        if single is not None:
            tau, r2, npts = single
            taus.append(tau)
            tau_r2s.append(r2)
            n_pts_list.append(npts)
            n_single_ok += 1
        dbl = fit_tau_double(a)
        if dbl is not None:
            double_r2s.append(dbl[3])
            n_double_ok += 1

    return {
        "model": model,
        "source": src,
        "n_total": n_total,
        "n_single_fit": n_single_ok,
        "n_double_fit": n_double_ok,
        "frac_single": n_single_ok / max(1, n_total),
        "taus": taus,
        "single_r2": tau_r2s,
        "double_r2": double_r2s,
        "n_points": n_pts_list,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def summarize(r):
    if r is None:
        return
    taus = np.array(r["taus"])
    print(f"\n=== {r['model']}  ({r['source']}) ===")
    print(f"  prompts analysed:      {r['n_total']}")
    print(f"  single-exp fit ok:     {r['n_single_fit']} "
          f"({100*r['frac_single']:.0f}%)")
    print(f"  double-exp fit ok:     {r['n_double_fit']}")

    if len(taus) == 0:
        print("  no tau values — residual stream too noisy for ACF fit")
        return

    print(f"  tau values (sorted):   "
          f"{[round(float(t),1) for t in np.sort(taus)]}")
    print(f"  tau median:            {np.median(taus):.1f}")
    print(f"  tau mean ± std:        {taus.mean():.1f} ± {taus.std():.1f}")
    print(f"  tau IQR:               "
          f"{np.percentile(taus,75)-np.percentile(taus,25):.1f}")
    print(f"  tau min/max:           {taus.min():.1f} / {taus.max():.1f}")

    # histogram-style bins
    bins = [(0, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, 1e9)]
    print("  tau distribution:")
    for lo, hi in bins:
        n = int(((taus >= lo) & (taus < hi)).sum())
        if n > 0:
            print(f"    [{lo:>4.0f}, {hi:>5.0f}): {n} "
                  f"({'#'*n})")

    if r["single_r2"]:
        r2 = np.array(r["single_r2"])
        print(f"  single-exp R² median:  {np.median(r2):.3f}")
    if r["double_r2"]:
        r2 = np.array(r["double_r2"])
        print(f"  double-exp R² median:  {np.median(r2):.3f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"max_lag={MAX_LAG}  acf_threshold={ACF_THRESHOLD}")
    results = {}
    for model in ["qwen", "mamba2"]:
        r = analyze(model)
        summarize(r)
        if r is not None:
            results[model] = {
                "n_total": r["n_total"],
                "n_single_fit": r["n_single_fit"],
                "frac_single": r["frac_single"],
                "tau_median": float(np.median(r["taus"]))
                              if r["taus"] else None,
                "tau_mean": float(np.mean(r["taus"]))
                            if r["taus"] else None,
                "tau_std": float(np.std(r["taus"]))
                           if r["taus"] else None,
                "taus": [float(t) for t in r["taus"]],
            }

    # Verdict
    print("\n=== VERDICT ===")
    q, m = results.get("qwen"), results.get("mamba2")
    if q and m:
        frac_q = q["frac_single"]
        frac_m = m["frac_single"]
        print(f"  ACF-fit success rate: Qwen {100*frac_q:.0f}%  "
              f"Mamba {100*frac_m:.0f}%")
        if frac_m > frac_q + 0.2:
            print("  -> Mamba residual stream is systematically smoother "
                  "(ACF well-defined for a larger fraction of prompts)")
        elif frac_q > frac_m + 0.2:
            print("  -> Qwen residual stream is smoother "
                  "(unexpected direction)")
        else:
            print("  -> both models comparable in ACF-fit rate")

        if q["tau_median"] is not None and m["tau_median"] is not None:
            ratio = m["tau_median"] / q["tau_median"]
            print(f"  tau_median: Qwen {q['tau_median']:.1f}  "
                  f"Mamba {m['tau_median']:.1f}  ratio={ratio:.2f}")

    # Save
    import json
    out = SM_DIR.parent / "spectral" / "distrib_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()