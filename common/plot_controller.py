"""plot_controller.py — figures for Paper 3.

Reads per_problem_{model}_gsmhard_n{N}.npz (written by
eval_controller.py) and produces:

  fig1_effort_dial.png       — ||c_t|| vs difficulty, main figure
  fig2_accuracy_bars.png     — A/B/C/D accuracies with bootstrap CIs
  fig3_bootstrap_dist.png    — distribution of paired bootstrap deltas

Also prints summary numbers to console for cross-checking against
the JSON produced by eval_controller.py.

Usage:
  python common/plot_controller.py \
      --npz data/stage1/controller/per_problem_qwen_gsmhard_n300.npz \
      --out data/stage1/controller
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--out", type=str, default="data/stage1/controller")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap_B", type=int, default=2000)
    return p.parse_args()


def paired_bootstrap(a, b, B=2000, seed=42):
    """a, b: arrays of per-problem correctness (0/1). Returns
    (mean_delta, lo, hi, all_deltas)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    deltas = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, n)
        deltas[k] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi), deltas


def plot_effort_dial(c_norm, difficulty, out_path, seed=42):
    """Main figure: scatter of c_norm vs difficulty with jitter + fit."""
    rng = np.random.default_rng(seed)

    # Jitter on x: difficulty is binary 0/1, add small horizontal noise.
    x = difficulty + rng.uniform(-0.06, 0.06, size=len(difficulty))
    y = c_norm

    # Compute correlation (Pearson)
    r = float(np.corrcoef(c_norm, difficulty)[0, 1])
    from scipy.stats import pearsonr
    _, p_val = pearsonr(c_norm, difficulty)

    fig, ax = plt.subplots(figsize=(6, 4.2))

    # Split by difficulty for color coding
    mask_easy = difficulty == 0
    mask_hard = difficulty == 1

    ax.scatter(x[mask_easy], y[mask_easy],
               alpha=0.45, s=14, color="#3b7dd8",
               label=f"easy (base correct), n={mask_easy.sum()}")
    ax.scatter(x[mask_hard], y[mask_hard],
               alpha=0.45, s=14, color="#d83b3b",
               label=f"hard (base incorrect), n={mask_hard.sum()}")

    # Means per group
    ax.hlines(y[mask_easy].mean(), -0.15, 0.15,
              colors="#1e3d6d", linewidth=2.2)
    ax.hlines(y[mask_hard].mean(), 0.85, 1.15,
              colors="#6d1e1e", linewidth=2.2)

    ax.set_xlabel("Difficulty (1 = frozen base answers incorrectly)")
    ax.set_ylabel("Mean intervention magnitude  ||c_t||")
    ax.set_title(f"Effort dial:  r = {r:+.3f},  p = {p_val:.1e}")
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")
    return r, p_val


def plot_accuracy_bars(A, B, C, D, deltas, out_path):
    """Bar chart of A/B/C/D accuracy with bootstrap CIs on deltas."""
    labels = ["A\nfrozen", "B\nLoRA", "C\nrandom", "D\ntrained"]
    accs = [A.mean(), B.mean(), C.mean(), D.mean()]
    colors = ["#888", "#3b7dd8", "#888", "#d83b3b"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(labels, accs, color=colors, alpha=0.85)

    # Annotate with accuracy values
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                acc + 0.004, f"{acc:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy on 300 held-out GSM-HARD")
    ax.set_ylim(0.30, 0.42)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Four-condition evaluation")

    # Add delta annotations between key pairs
    def annotate(x1, x2, y, text):
        ax.annotate("", xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle="-", color="black",
                                    lw=0.8))
        ax.text((x1 + x2) / 2, y + 0.003, text,
                ha="center", va="bottom", fontsize=8)

    annotate(0, 1, 0.395, f"B−A = {deltas['BA'][0]:+.3f}")
    annotate(1, 3, 0.405, f"D−B = {deltas['DB'][0]:+.3f}")
    annotate(2, 3, 0.395, f"D−C = {deltas['DC'][0]:+.3f}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def plot_bootstrap_dists(deltas, out_path):
    """Histogram of the three paired bootstrap distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    pairs = [
        ("B − A", deltas["BA"], "#3b7dd8"),
        ("D − B", deltas["DB"], "#d83b3b"),
        ("D − C", deltas["DC"], "#3ba85b"),
    ]

    for ax, (title, dist, color) in zip(axes, pairs):
        ax.hist(dist, bins=40, color=color, alpha=0.75)
        ax.axvline(0, color="black", linestyle="--", lw=1)
        lo, hi = np.percentile(dist, [2.5, 97.5])
        ax.axvline(lo, color="gray", linestyle=":", lw=1)
        ax.axvline(hi, color="gray", linestyle=":", lw=1)
        mean = float(dist.mean())
        ax.set_title(f"{title}:  {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        ax.set_xlabel("Δ accuracy")
        ax.set_yticks([])
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"missing: {npz_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    d = np.load(npz_path, allow_pickle=True)
    A = d["A"]
    B = d["B"]
    C = d["C"]
    D = d["D"]
    difficulty = d["difficulty"]
    c_norm = d["c_norm_D"]

    print(f"[plot] n={len(A)} problems")
    print(f"[plot] acc: A={A.mean():.4f} B={B.mean():.4f} "
          f"C={C.mean():.4f} D={D.mean():.4f}")
    print(f"[plot] c_norm: mean={c_norm.mean():.4f} "
          f"std={c_norm.std():.4f} CV={c_norm.std()/c_norm.mean():.4f}")
    print(f"[plot] difficulty: mean={difficulty.mean():.4f}")

    # Paired bootstraps
    BA = paired_bootstrap(B, A, B=args.bootstrap_B, seed=args.seed)
    DB = paired_bootstrap(D, B, B=args.bootstrap_B, seed=args.seed)
    DC = paired_bootstrap(D, C, B=args.bootstrap_B, seed=args.seed)

    deltas = {
        "BA": (BA[0], BA[1], BA[2], BA[3]),
        "DB": (DB[0], DB[1], DB[2], DB[3]),
        "DC": (DC[0], DC[1], DC[2], DC[3]),
    }

    print(f"[plot] Δ B−A = {BA[0]:+.4f}  [{BA[1]:+.4f}, {BA[2]:+.4f}]")
    print(f"[plot] Δ D−B = {DB[0]:+.4f}  [{DB[1]:+.4f}, {DB[2]:+.4f}]")
    print(f"[plot] Δ D−C = {DC[0]:+.4f}  [{DC[1]:+.4f}, {DC[2]:+.4f}]")

    # Figure 1: effort dial
    r, p_val = plot_effort_dial(c_norm, difficulty,
                                out_dir / "fig1_effort_dial.png",
                                seed=args.seed)
    print(f"[plot] effort dial correlation r = {r:+.4f}, p = {p_val:.3e}")

    # Figure 2: accuracy bars
    plot_accuracy_bars(A, B, C, D, deltas,
                       out_dir / "fig2_accuracy_bars.png")

    # Figure 3: bootstrap distributions — pass only the distribution
    # arrays, not the (mean, lo, hi, dist) tuples.
    dists = {k: v[3] for k, v in deltas.items()}
    plot_bootstrap_dists(dists, out_dir / "fig3_bootstrap_dist.png")

    print(f"\n[plot] all figures saved to {out_dir}")


if __name__ == "__main__":
    main()