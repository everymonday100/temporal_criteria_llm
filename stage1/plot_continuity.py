import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\continuity")

# Data from your run
metrics = {
    "Frozen\n(Always CoT)": {"acc": 0.4747, "f1": 0.2146},
    "Global\nMean-Optimal": {"acc": 0.4747, "f1": 0.2146},
    "Shuffled\nContext": {"acc": 0.5253, "f1": 0.4162},
    "Argmax\n(Context)": {"acc": 0.5253, "f1": 0.4691},
    "Cost-Aware\n(Context)": {"acc": 0.5758, "f1": 0.5242},
}

labels = list(metrics.keys())
f1_scores = [m["f1"] for m in metrics.values()]
accs = [m["acc"] for m in metrics.values()]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, f1_scores, width, label='Macro F1', color='#1f77b4')
rects2 = ax.bar(x + width/2, accs, width, label='Accuracy', color='#ff7f0e', alpha=0.7)

ax.set_ylabel('Score')
ax.set_title('Continuity Track: Decision-Rule Dependence (Qwen2.5-3B)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 0.7)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
out_path = OUT_DIR / "fig_continuity.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"[saved] {out_path}")
plt.show()