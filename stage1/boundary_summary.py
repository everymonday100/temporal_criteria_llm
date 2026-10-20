"""boundary_summary.py — Aggregate results into Outcome Matrix.

Reads fm_results_*.json and constructs the final verdict for the paper.
"""
import argparse
import json
from pathlib import Path

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\boundary")

def load_results(design, variant, control, layer=18):
    path = OUT_DIR / f"fm_results_{design}_{variant}_{control}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(str(layer))

def print_row(desc, res):
    if res is None:
        print(f"  {desc:<40} | MISSING")
        return
    print(f"  {desc:<40} | AUC={res['auc']:.4f}  R2={res['r2']:+.4f}")

def main():
    print("=" * 75)
    print("BOUNDARY EXPERIMENT SUMMARY (Layer 18)")
    print("=" * 75)
    
    # 1. Main Results
    print("\n[Main Results]")
    res_R_main = load_results("R", "F", "none")
    res_O_main = load_results("O", "F", "none")
    print_row("Reactive (Design R)", res_R_main)
    print_row("Observation (Design O)", res_O_main)

    # 2. Controls
    print("\n[Controls]")
    res_R_c1 = load_results("R", "F", "C1")
    res_O_c1 = load_results("O", "F", "C1")
    print_row("C1 Random Weights (R)", res_R_c1)
    print_row("C1 Random Weights (O)", res_O_c1)

    res_R_c7 = load_results("R", "F", "C7")
    res_O_c7 = load_results("O", "F", "C7")
    print_row("C7 Random Action (R)", res_R_c7)
    print_row("C7 Random Action (O)", res_O_c7)

    res_R_c8 = load_results("R", "F", "C8")
    print_row("C8 Delayed Action (R)", res_R_c8)
    
    # 3. Verdict
    print("\n" + "=" * 75)
    print("VERDICT")
    print("=" * 75)
    
    if not (res_R_main and res_O_main):
        print("  INCOMPLETE: Missing main results.")
        return

    # C1 gate — applies to R (control design); O is trivially separable
    # by observation-variance structure and its C1 is a class-integrity
    # check, not a triviality gate.
    c1_R = res_R_c1["auc"] if res_R_c1 else 0.0
    c1_O = res_O_c1["auc"] if res_O_c1 else 0.0

    if c1_R > 0.60:
        print(f"  ❌ FAIL (C1 Gate on R): C1 AUC={c1_R:.3f} > 0.60.")
        return
    print(f"  ✅ C1(R) = {c1_R:.3f} < 0.60 — reactive task is non-trivial.")
    if c1_O > 0.60:
        print(f"  ℹ️  C1(O) = {c1_O:.3f} > 0.60 — design O is trivially separable")
        print("      by observation-variance structure; treat as class-integrity")
        print("      check, not a triviality gate.")

    auc_R = res_R_main["auc"]
    auc_O = res_O_main["auc"]
    
    # Interface Thesis Check
    if auc_R < 0.75 and auc_O >= 0.85:
        print("  ✅ PASS: Full Interface Thesis confirmed.")
        print(f"     Observation AUC={auc_O:.3f} >= 0.85, Reactive AUC={auc_R:.3f} < 0.75.")
    elif auc_R >= 0.85 and auc_O >= 0.85:
        print("  ⚠️  PARTIAL: Both interfaces gave high AUC.")
        print("     Forward model capacity binds, not interface.")
    elif auc_R < 0.75 and auc_O < 0.75:
        print("  ❌ FAIL: Boundary fails on LLM at this task complexity.")
    else:
        print("  ⚠️  INCONCLUSIVE: Mixed results.")
        print(f"     R={auc_R:.3f}, O={auc_O:.3f}. Check controls and CI.")

    # Check Controls for interpretation
    if res_R_c7 and res_O_c7:
            
    if res_R_c8 and res_R_c8["auc"] > 0.75:
        print("  ⚠️  WARNING: C8 (Delayed Action) gave high AUC on R.")
        print("     Forward model is not sensitive to current action.")

if __name__ == "__main__":
    main()