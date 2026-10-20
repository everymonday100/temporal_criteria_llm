"""partial_correlation.py — separate marginal vs conditional level effect.

Figure 4 shows marginal dependence of future_std on current mean.
Ridge ablation shows no conditional dependence for Mamba. These are
compatible if the marginal U-shape is fully explained by current std.

Test:
  1. Marginal correlation: corr(mean, future_std)
  2. Conditional: corr(mean, residual | future_std ~ std)
     i.e. partial correlation of mean and future_std given std
  3. Residual plot: bin by mean, plot mean of residual (future_std
     after regression on std). Flat → mean is redundant.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.generation import to_tau

SM_DIR    = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")
CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")
OUT_DIR   = Path(r"E:\OllamaModels\temporal_llm\data\stage1\spectral")

K_FIX = 4
H_FIX = 2
W_WIN = 8
N_BINS = 10


def _load(model):
    sfx = "" if model == "qwen" else f"_{model}"
    for p in [SM_DIR / f"L0{sfx}_traj.npz",
              CACHE_DIR / f"sm_L0{sfx}.npz"]:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "lambdas" in d:
                return [np.asarray(x, dtype=float) for x in d["lambdas"]]
    return None


def zscore(x):
    m, s = x.mean(), x.std()
    return (x - m) / s if s > 1e-9 else np.zeros_like(x)


def collect_within(lambdas):
    """Per-prompt z-scored means, stds, targets."""
    means, stds, targets = [], [], []
    for lam in lambdas:
        _, v = to_tau(lam)
        if len(v) < K_FIX + H_FIX + W_WIN + 2:
            continue
        m_arr, s_arr, t_arr = [], [], []
        for i in range(K_FIX, len(v) - H_FIX - W_WIN):
            w = v[i - K_FIX:i]
            m_arr.append(w.mean())
            s_arr.append(w.std())
            t_arr.append(v[i + H_FIX:i + H_FIX + W_WIN].std())
        if len(m_arr) < 10:
            continue
        means.append(zscore(np.array(m_arr)))
        stds.append(zscore(np.array(s_arr)))
        targets.append(zscore(np.array(t_arr)))
    return (np.concatenate(means),
            np.concatenate(stds),
            np.concatenate(targets))


def binned_residual(x, resid, n_bins=N_BINS):
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    cx, cy, cs = [], [], []
    for k in range(n_bins):
        lo, hi = qs[k], qs[k + 1]
        m = (x >= lo) & (x < hi)
        if m.sum() < 5:
            continue
        cx.append((lo + hi) / 2)
        cy.append(resid[m].mean())
        cs.append(resid[m].std() / np.sqrt(m.sum()))
    return np.array(cx), np.array(cy), np.array(cs)


def analyze(model):
    lambdas = _load(model)
    if lambdas is None:
        return None
    m, s, y = collect_within(lambdas)
    if len(m) < 100:
        return None

    # marginal correlations
    corr_marg = float(np.corrcoef(m, y)[0, 1])
    corr_std = float(np.corrcoef(s, y)[0, 1])

    # partial correlation of mean and y given std
    # regress y ~ std, get residual; regress m ~ std, get residual;
    # corr of residuals
    def residual(target, predictor):
        A = np.vstack([predictor, np.ones_like(predictor)]).T
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        return target - A @ coef

    r_y_s = residual(y, s)
    r_m_s = residual(m, s)
    corr_partial = float(np.corrcoef(r_m_s, r_y_s)[0, 1])

    # residual plot
    cx, cy, cs = binned_residual(m, r_y_s)

    return {
        "model": model, "n": len(m),
        "corr_marginal_mean_y":   corr_marg,
        "corr_std_y":             corr_std,
        "corr_partial_mean_y|std": corr_partial,
        "residual_binned_x": cx.tolist(),
        "residual_binned_y": cy.tolist(),
        "residual_binned_se": cs.tolist(),
    }


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    results = {}
    for ax, model in zip(axes, ["qwen", "mamba2"]):
        r = analyze(model)
        if r is None:
            ax.set_title(f"{model} (no data)")
            continue
        results[model] = r
        cx = np.array(r["residual_binned_x"])
        cy = np.array(r["residual_binned_y"])
        cs = np.array(r["residual_binned_se"])
        ax.errorbar(cx, cy, yerr=cs, marker="o", capsize=3, lw=1.5)
        slope = np.polyfit(cx, cy, 1)[0]
        ax.set_title(f"{model}  partial_corr={r['corr_partial_mean_y|std']:+.3f}\n"
                     f"residual slope={slope:+.3f}")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("within-prompt z(mean Λ)")
        ax.set_ylabel("residual future std (given std)")
        ax.grid(alpha=0.3)

        print(f"\n=== {model} ===")
        print(f"  n samples:                {r['n']}")
        print(f"  corr(mean, future_std):   {r['corr_marginal_mean_y']:+.4f}")
        print(f"  corr(std, future_std):    {r['corr_std_y']:+.4f}")
        print(f"  partial corr (|std):      {r['corr_partial_mean_y|std']:+.4f}")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig4_partial_correlation.png"
    plt.savefig(out, dpi=140)
    print(f"\n[saved] {out}")

    # Verdict
    print("\n=== VERDICT ===")
    q = results.get("qwen")
    m_r = results.get("mamba2")
    if q and m_r:
        pc_q = q["corr_partial_mean_y|std"]
        pc_m = m_r["corr_partial_mean_y|std"]
        print(f"  partial corr qwen:  {pc_q:+.4f}")
        print(f"  partial corr mamba: {pc_m:+.4f}")
        if pc_q > 0.15 and abs(pc_m) < 0.05:
            print("  -> Qwen has conditional level effect; Mamba does not.")
            print("     Figure 4 U-shape on Mamba is fully explained by std.")
        elif abs(pc_q - pc_m) < 0.05:
            print("  -> both models show comparable conditional level effect.")
        else:
            print("  -> mixed; inspect partial correlation values.")

    import json
    with open(OUT_DIR / "partial_correlation.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()