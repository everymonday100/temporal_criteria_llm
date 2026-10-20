"""plot_progressive_n.py — форма кривой cosine_within(N).

Запускается после завершения всех четырёх оценок.
Строит два графика:
  1. Форма кривой: cosine_within и cosine_between vs N
  2. Δaccuracy vs N

Вердикт по форме:
  - Линейное падение → incremental
  - Ступенчатое падение → фазовый переход
  - Плато → гипотеза B
  - Немонотонное → шум или interference
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\progressive_n")
N_VALUES = [1, 2, 5, 10]


def main():
    results = {}
    for N in N_VALUES:
        path = OUT_DIR / f"progN_eval_N{N}_qwen.json"
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue
        with open(path) as f:
            results[N] = json.load(f)

    if not results:
        print("No results found. Run eval first.")
        return

    Ns = sorted(results.keys())

    # --- Данные для графиков ---
    within = [results[N]["mean_cosine_within"] for N in Ns]
    between = [results[N].get("mean_cosine_between", np.nan)
               for N in Ns]
    delta_acc = [results[N]["delta_D_minus_B"] for N in Ns]
    cv_c = [results[N]["cv_c_norm"] for N in Ns]
    corr_diff = [results[N]["corr_c_difficulty"] for N in Ns]

    # --- График 1: форма кривой косинусов ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(Ns, within, 'o-', color='blue', linewidth=2,
            markersize=8, label='cosine_within')
    if any(not np.isnan(b) for b in between):
        valid_Ns = [N for N, b in zip(Ns, between)
                    if not np.isnan(b)]
        valid_between = [b for b in between if not np.isnan(b)]
        ax.plot(valid_Ns, valid_between, 's--', color='orange',
                linewidth=2, markersize=8, label='cosine_between')
    ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.5,
               label='threshold 0.9')
    ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5,
               label='threshold 0.7')
    ax.set_xlabel('N (injection points)')
    ax.set_ylabel('Mean pairwise cosine')
    ax.set_title('Direction variance vs N\n(main result)')
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Аннотация формы
    if len(within) >= 2:
        drop = within[0] - within[-1]
        if drop < 0.05:
            shape = "PLATEAU → B"
        elif all(within[i] >= within[i+1]
                 for i in range(len(within)-1)):
            max_drop = max(within[i] - within[i+1]
                          for i in range(len(within)-1))
            if max_drop > 0.15:
                shape = "STEP → phase transition"
            else:
                shape = "LINEAR → incremental"
        else:
            shape = "NON-MONOTONE → noise/interference"
        ax.annotate(shape, xy=(Ns[-1], within[-1]),
                    xytext=(Ns[-1] - 1, within[-1] - 0.05),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax = axes[1]
    ax.bar([str(N) for N in Ns], delta_acc, color='steelblue',
           alpha=0.8)
    ax.axhline(y=0.02, color='green', linestyle='--', alpha=0.5,
               label='Δ=+2pp threshold')
    ax.set_xlabel('N (injection points)')
    ax.set_ylabel('Δ accuracy (D − B)')
    ax.set_title('Accuracy gain vs N')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(Ns, cv_c, 'o-', color='purple', linewidth=2,
            markersize=8, label='CV of ||c||')
    ax2.plot(Ns, corr_diff, 's--', color='red', linewidth=2,
             markersize=8, label='corr(||c||, difficulty)')
    ax.set_xlabel('N (injection points)')
    ax.set_ylabel('CV of ||c||', color='purple')
    ax2.set_ylabel('corr(||c||, difficulty)', color='red')
    ax.set_title('Selectivity diagnostics vs N')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig_path = OUT_DIR / "progressive_N_shape.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"[saved] {fig_path}")

    # --- Текстовая сводка ---
    print(f"\n{'='*70}")
    print(f"PROGRESSIVE-N SUMMARY")
    print(f"{'='*70}")
    print(f"{'N':>4}  {'cos_within':>10}  {'cos_between':>11}  "
          f"{'Δacc':>8}  {'CV':>6}  {'corr':>8}  {'shape_tag':>14}  "
          f"{'core_verdict':>16}")
    for N in Ns:
        r = results[N]
        cb = r.get("mean_cosine_between", float("nan"))
        print(f"{N:>4}  {r['mean_cosine_within']:>10.4f}  "
              f"{cb:>11.4f}  {r['delta_D_minus_B']:>+8.3f}  "
              f"{r['cv_c_norm']:>6.3f}  "
              f"{r['corr_c_difficulty']:>+8.4f}  "
              f"{r['shape_tag']:>14}  {r['core_verdict']:>16}")

    print(f"\n{'='*70}")
    print(f"INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print(f"cosine_within < 0.7  → content-specific directions (A)")
    print(f"cosine_within > 0.9  → constant direction (B)")
    print(f"cos_between low, cos_within high → positional specialization")
    print(f"cos_between low, cos_within low  → content-specificity ✓")
    print(f"\nCore justified iff: cos_within < 0.7 AND Δacc < +2pp")
    print(f"Core NOT needed iff: cos_within < 0.7 AND Δacc ≥ +2pp")
    print(f"Core futile iff:     cos_within > 0.9 (any Δacc)")


if __name__ == "__main__":
    main()