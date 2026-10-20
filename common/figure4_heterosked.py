"""figure4_v2.py — level-dependent conditional variance, per-prompt z-scored.

Fixes the between-prompt pooling artifact in the first version:
each prompt is z-scored independently before pooling.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def per_prompt_zscore(x):
    m, s = x.mean(), x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - m) / s


def collect_within(lambdas, K, H, W):
    """Collect samples where both features and target are z-scored
    within each prompt (removes between-prompt effects)."""
    xs, ys = [], []
    for lam in lambdas:
        _, v = to_tau(lam)
        if len(v) < K + H + W + 2:
            continue
        means, targets = [], []
        idxs = []
        for i in range(K, len(v) - H - W):
            means.append(v[i - K:i].mean())
            targets.append(v[i + H:i + H + W].std())
            idxs.append(i)
        if len(means) < 10:
            continue
        m_arr = np.array(means)
        t_arr = np.array(targets)
        # z-score within prompt
        mz = per_prompt_zscore(m_arr)
        tz = per_prompt_zscore(t_arr)
        xs.append(mz)
        ys.append(tz)
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def binned(x, y, n_bins=N_BINS):
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    cx, cy, cs, cn = [], [], [], []
    for k in range(n_bins):
        lo, hi = qs[k], qs[k + 1]
        m = (x >= lo) & (x < hi)
        if m.sum() < 5:
            continue
        cx.append((lo + hi) / 2)
        cy.append(y[m].mean())
        cs.append(y[m].std() / np.sqrt(m.sum()))
        cn.append(int(m.sum()))
    return map(np.array, (cx, cy, cs, cn))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model in zip(axes, ["qwen", "mamba2"]):
        lambdas = _load(model)
        if lambdas is None:
            ax.set_title(f"{model} (no data)")
            continue
        x, y = collect_within(lambdas, K_FIX, H_FIX, W_WIN)
        if len(x) < 100:
            ax.set_title(f"{model} (too few)")
            continue
        cx, cy, cstd, cn = binned(x, y)
        ax.errorbar(cx, cy, yerr=cstd, marker="o", capsize=3, lw=1.5)
        slope = np.polyfit(x, y, 1)[0]
        ax.set_title(f"{model}  slope={slope:+.3f}  n={len(x)}")
        ax.set_xlabel("within-prompt z(mean Λ)")
        ax.set_ylabel("within-prompt z(future std)")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig4_within_prompt.png"
    plt.savefig(out, dpi=140)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()