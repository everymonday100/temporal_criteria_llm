"""spectral_analysis.py — test H1' on Lambda(t) spectra and autocorrelation.

Hypothesis H1' from the paper:
    SSM input-modulation is bounded and linear (Delta_t = softplus,
    B_t, C_t linear in u_t), so the residual stream is a stationary
    low-pass filter output. Transformer input-modulation is unbounded
    and nonlinear (softmax over context), so the residual stream is
    non-stationary.

Predictions:
    P1. Mamba PSD falls as 1/f^alpha with high alpha (>1.5);
        Qwen PSD is flatter (alpha < 1) or multi-peaked.
    P2. Mamba ACF is well fit by a single exponential;
        Qwen ACF needs a sum of exponentials or a power law.
    P3. Mamba spectrum shape is stable across prompts (fixed kernel);
        Qwen spectrum shape varies.

Outputs:
    data/stage1/spectral/spectra.json
    data/stage1/spectral/summary.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")
SM_DIR    = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")
OUT_DIR   = Path(r"E:\OllamaModels\temporal_llm\data\stage1\spectral")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _load_lambdas(path):
    """Load list of Lambda(t) arrays from a traj.npz or sm_L0.npz."""
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    if "lambdas" not in d:
        return None
    return [np.asarray(x, dtype=np.float64) for x in d["lambdas"]]


def _sfx(model):
    return "" if model == "qwen" else f"_{model}"


def _load_model_l0(model):
    """L0 baselines: prefer traj file, fall back to raw cache."""
    sfx = _sfx(model)
    for p in [
        SM_DIR / f"L0{sfx}_traj.npz",
        CACHE_DIR / f"sm_L0{sfx}.npz",
    ]:
        l = _load_lambdas(p)
        if l is not None:
            return l, p.name
    return None, None


# ----------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------
def welch_psd(x, nperseg=None, overlap=0.5):
    """Simple Welch PSD without scipy. Returns (freqs, power)."""
    n = len(x)
    if nperseg is None:
        nperseg = min(128, n // 2)
    if nperseg < 8 or n < nperseg:
        return None, None
    step = max(1, int(nperseg * (1 - overlap)))
    win = np.hanning(nperseg)
    win_norm = (win ** 2).sum()
    segs = []
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg] * win
        seg = seg - seg.mean()
        segs.append(np.abs(np.fft.rfft(seg)) ** 2 / win_norm)
    if not segs:
        return None, None
    psd = np.mean(segs, axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1.0)
    return freqs, psd


def power_law_fit(freqs, psd, fmin=0.02, fmax=0.4):
    """Fit log(psd) = -alpha * log(f) + c over the given band."""
    mask = (freqs >= fmin) & (freqs <= fmax) & (psd > 0)
    if mask.sum() < 4:
        return None, None, None
    lf = np.log(freqs[mask])
    lp = np.log(psd[mask])
    # least squares
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, res, *_ = np.linalg.lstsq(A, lp, rcond=None)
    slope, intercept = coef
    pred = A @ coef
    ss_res = ((lp - pred) ** 2).sum()
    ss_tot = ((lp - lp.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return -slope, intercept, r2


def acf(x, max_lag=None):
    """Unbiased-ish ACF via FFT."""
    n = len(x)
    x = x - x.mean()
    if max_lag is None:
        max_lag = min(64, n // 3)
    # full autocorrelation
    f = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(f * np.conj(f))[:n]
    # normalize by variance (using 0-lag)
    if acf_full[0] <= 0:
        return None
    acf_norm = acf_full / acf_full[0]
    return acf_norm[:max_lag + 1]


def fit_exp_decay(acf_vals):
    """Fit acf[k] = exp(-k / tau). Returns tau (in tokens) and R²."""
    if acf_vals is None or len(acf_vals) < 4:
        return None, None
    k = np.arange(len(acf_vals))
    mask = acf_vals > 0.05
    if mask.sum() < 4:
        return None, None
    kk = k[mask]
    yy = np.log(acf_vals[mask])
    A = np.vstack([kk, np.ones_like(kk)]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    slope, intercept = coef
    tau = -1.0 / slope if slope < 0 else np.inf
    pred = A @ coef
    ss_res = ((yy - pred) ** 2).sum()
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return tau, r2


def fit_double_exp(acf_vals):
    """Fit acf[k] = a*exp(-k/t1) + (1-a)*exp(-k/t2) by grid search on t1, t2.

    Returns (t1, t2, a, r2). Coarse grid — enough for qualitative comparison.
    """
    if acf_vals is None or len(acf_vals) < 8:
        return None
    k = np.arange(len(acf_vals)).astype(float)
    y = acf_vals
    best = (None, None, None, -np.inf)
    t1_grid = np.array([1, 2, 4, 8, 16])
    t2_grid = np.array([16, 32, 64, 128, 256])
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
def analyze_lambdas(lambdas, label, max_lag=48):
    """Return per-prompt PSD, ACF, alpha, and tau."""
    alphas, taus, exp_r2s, dbl_r2s = [], [], [], []
    spectra, acfs = [], []
    for lam in lambdas:
        if len(lam) < 64:
            continue
        f, p = welch_psd(lam, nperseg=64)
        if f is None:
            continue
        a, c, r2 = power_law_fit(f, p)
        if a is not None:
            alphas.append(a)
        a_vals = acf(lam, max_lag=max_lag)
        if a_vals is not None and len(a_vals) > 4:
            acfs.append(a_vals)
            tau, r2e = fit_exp_decay(a_vals)
            if tau is not None:
                taus.append(tau)
                exp_r2s.append(r2e)
            dbl = fit_double_exp(a_vals)
            if dbl is not None:
                dbl_r2s.append(dbl[3])

    alphas = np.array(alphas)
    taus = np.array(taus)
    exp_r2s = np.array(exp_r2s)
    dbl_r2s = np.array(dbl_r2s)

    # Aggregate spectra: mean log-PSD over prompts
    if spectra:
        # all have same freqs (nperseg=64)
        pass

    mean_acf = np.mean(acfs, axis=0) if acfs else None

    return {
        "label": label,
        "n_prompts_used": len(alphas),
        "alpha_mean":   float(alphas.mean()) if len(alphas) else None,
        "alpha_std":    float(alphas.std())  if len(alphas) else None,
        "tau_mean":     float(taus.mean())   if len(taus) else None,
        "tau_std":      float(taus.std())    if len(taus) else None,
        "exp_fit_r2_mean": float(exp_r2s.mean()) if len(exp_r2s) else None,
        "double_exp_fit_r2_mean": float(dbl_r2s.mean()) if len(dbl_r2s) else None,
        "acf_mean": mean_acf.tolist() if mean_acf is not None else None,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="L0",
                        help="Which interface to analyze (L0, L1a, ...)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model in ["qwen", "mamba2"]:
        sfx = _sfx(model)
        if args.iface == "L0":
            lambdas, src = _load_model_l0(model)
        else:
            p = SM_DIR / f"{args.iface}{sfx}_traj.npz"
            lambdas = _load_lambdas(p)
            src = p.name if lambdas else None

        if lambdas is None:
            print(f"[{model}] {args.iface} — not found")
            continue

        r = analyze_lambdas(lambdas, f"{model}/{args.iface}")
        r["source"] = src
        results[model] = r
        print(f"\n=== {model} / {args.iface} ({src}) ===")
        print(f"  prompts used:        {r['n_prompts_used']}")
        print(f"  alpha (PSD slope):   {r['alpha_mean']:.3f} ± {r['alpha_std']:.3f}"
              if r['alpha_mean'] is not None else "  alpha: n/a")
        print(f"  tau   (ACF exp):     {r['tau_mean']:.2f} ± {r['tau_std']:.2f} tokens"
              if r['tau_mean'] is not None else "  tau: n/a")
        print(f"  exp fit R²:          {r['exp_fit_r2_mean']:.3f}"
              if r['exp_fit_r2_mean'] is not None else "  exp fit: n/a")
        print(f"  double-exp fit R²:   {r['double_exp_fit_r2_mean']:.3f}"
              if r['double_exp_fit_r2_mean'] is not None else "  double: n/a")

    # Verdict against H1'
    print("\n=== VERDICT (H1' predictions) ===")
    q = results.get("qwen", {})
    m = results.get("mamba2", {})
    if q and m:
        # P1: alpha_mamba > alpha_qwen
        a_q = q.get("alpha_mean")
        a_m = m.get("alpha_mean")
        if a_q is not None and a_m is not None:
            passed = a_m > a_q
            print(f"  P1 (alpha_mamba > alpha_qwen):        "
                  f"{'PASS' if passed else 'fail'}  "
                  f"alpha_qwen={a_q:.2f}  alpha_mamba={a_m:.2f}")
        # P2: single-exp R² lower for Qwen
        r_q = q.get("exp_fit_r2_mean")
        r_m = m.get("exp_fit_r2_mean")
        if r_q is not None and r_m is not None:
            passed = r_q < r_m
            print(f"  P2 (single-exp fits Mamba better):    "
                  f"{'PASS' if passed else 'fail'}  "
                  f"R2_qwen={r_q:.3f}  R2_mamba={r_m:.3f}")
        # P2b: double-exp helps Qwen more than Mamba
        r_q2 = q.get("double_exp_fit_r2_mean")
        r_m2 = m.get("double_exp_fit_r2_mean")
        if r_q2 is not None and r_m2 is not None:
            gain_q = r_q2 - (r_q if r_q is not None else 0)
            gain_m = r_m2 - (r_m if r_m is not None else 0)
            passed = gain_q > gain_m
            print(f"  P2b (double-exp helps Qwen more):     "
                  f"{'PASS' if passed else 'fail'}  "
                  f"gain_qwen={gain_q:+.3f}  gain_mamba={gain_m:+.3f}")

    out_path = OUT_DIR / f"spectra_{args.iface}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()