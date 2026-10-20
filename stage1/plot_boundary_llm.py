"""plot_boundary_llm.py — Bar chart for LLM boundary results."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\boundary")

def load_auc(design, control="none"):
    path = OUT_DIR / f"fm_results_{design}_F_{control}.json"
    if not path.exists():
        return 0.0
    with open(path) as f:
        data = json.load(f)
    return data["18"]["auc"]  # Layer 18

# Data
conditions = ["Reactive\nBaseline", "Reactive\nC1 (Random W)", "Reactive\nC7 (Random A)", "Reactive\nC8 (Delayed A)", 
              "Observation\nBaseline", "Observation\nC1 (Random W)"]
aucs = [
    load_auc("R", "none"),
    load_auc("R", "C1"),
    load_auc("R", "C7"),
    load_auc("R", "C8"),
    load_auc("O", "none"),
    load_auc("O", "C1")
]

# Colors: R = grey, O = blue
colors = ['#7f7f7f', '#7f7f7f', '#7f7f7f', '#7f7f7f', '#1f77b4', '#1f77b4']

fig, ax = plt.subplots(figsize=(10, 5))

# Bars
bars = ax.bar(conditions, aucs, color=colors, edgecolor='black', alpha=0.8)

# Threshold lines
ax.axhline(0.85, color='#d62728', linestyle='--', linewidth=1.5, label='Sharp boundary threshold (0.85)')
ax.axhline(0.75, color='#ff7f0e', linestyle=':', linewidth=1.5, label='Reactive limit (0.75)')

# Annotations
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Aesthetics
ax.set_ylim(0.45, 1.05)
ax.set_ylabel('AUC (Forward-model boundary)', fontsize=12)
ax.set_title('Boundary as a function of interface (Qwen2.5-3B, L18)', fontsize=13, pad=15)
ax.legend(loc='upper left', framealpha=0.9)

plt.xticks(rotation=15, ha='right', fontsize=10)
plt.tight_layout()

# Save
out_path = OUT_DIR / "boundary_llm_plot.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
print(f"[saved] {out_path}")
plt.show()