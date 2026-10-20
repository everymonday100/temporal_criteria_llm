"""audit_2x2_matrix.py — Полный аудит 2×2: (model) × (LoRA frozen/trainable).
Исправлено: использует sys.executable для запуска дочерних процессов в том же
виртуальном окружении, что и текущий скрипт.
"""
import subprocess, json, sys
from pathlib import Path

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\controller")

def run_audit(model, lora_frozen):
    """Run audit_loss_anisotropy_v2.py with given parameters."""
    # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: sys.executable вместо "python"
    cmd = [sys.executable, "stage1/audit_loss_anisotropy_v2.py",
           "--model", model, "--n", "64", "--layer", "17"]
    if lora_frozen:
        cmd.append("--lora_frozen")
    print(f"\n{'='*62}")
    print(f"Running: {' '.join(cmd)}")
    print('='*62)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    # Parse output file
    out_path = OUT_DIR / f"audit_{model}_lora_frozen={lora_frozen}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return None

def main():
    conditions = [
        ("qwen", False, "Qwen + LoRA trainable"),
        ("qwen", True, "Qwen + LoRA frozen"),
        ("xlam", False, "xLAM + LoRA trainable"),
        ("xlam", True, "xLAM + LoRA frozen (v1 condition)"),
    ]
    
    results = {}
    for model, frozen, label in conditions:
        stats = run_audit(model, frozen)
        if stats:
            results[label] = stats
    
    print("\n" + "="*62)
    print("2×2 AUDIT MATRIX")
    print("="*62)
    print(f"{'Condition':<35} {'v1':>6} {'eff_rank':>8} {'β':>6}")
    print("-"*62)
    for label, stats in results.items():
        v1 = stats.get('v1', float('nan'))
        er = stats.get('eff_rank', float('nan'))
        beta = stats.get('beta', float('nan'))
        print(f"{label:<35} {v1:>6.3f} {er:>8.2f} {beta:>6.3f}")
    
    # Interpretation
    print("\n[INTERPRETATION]")
    v1_values = [s.get('v1', 1) for s in results.values()]
    beta_values = [s.get('beta', 0) for s in results.values() if 'beta' in s]
    
    if all(v > 0.9 for v in v1_values):
        print("  ❌ All conditions show ray collapse (v1 > 0.9)")
        print("     → Gradient is essentially 1D everywhere")
    elif all(v < 0.5 for v in v1_values):
        print("  ✅ All conditions show multi-directional gradient (v1 < 0.5)")
        print("     → Ray collapse is NOT about gradient geometry")
    else:
        print(f"  ⚠️  Mixed results: v1 ranges from {min(v1_values):.3f} to {max(v1_values):.3f}")
    
    if beta_values and max(beta_values) < 0.1:
        print(f"  ⚠️  β < 0.1 in all conditions: between-class signal is weak")
        print("     → Contrastive hinge REQUIRED to boost β_eff")
    elif beta_values and min(beta_values) > 0.3:
        print(f"  ✅ β > 0.3 in all conditions: between-class signal is strong")
    
    # Save matrix
    with open(OUT_DIR / "audit_2x2_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT_DIR / 'audit_2x2_matrix.json'}")

if __name__ == "__main__":
    main()