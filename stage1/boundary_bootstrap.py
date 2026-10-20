"""boundary_bootstrap.py — Stream-level bootstrap for AUC confidence intervals."""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\boundary")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--design", choices=["R", "O"], required=True)
    p.add_argument("--variant", choices=["F", "L"], default="F")
    p.add_argument("--control", default="none")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    preds_file = OUT_DIR / f"preds_{args.design}_{args.variant}_{args.control}_L{args.layer}.npz"
    if not preds_file.exists():
        print(f"[Error] File {preds_file} not found. Run boundary_train_fm.py first.")
        return

    data = np.load(preds_file)
    sids = data["stream_id"]
    residuals = data["residuals"]
    is_ext = data["is_ext"]

    rng = np.random.default_rng(args.seed)
    unique_streams = np.unique(sids)
    auc_scores = []

    print(f"[Bootstrap] Resampling {len(unique_streams)} streams {args.n_boot} times...")
    
    for _ in range(args.n_boot):
        # Сэмплируем стримы с возвращением
        sampled_sids = rng.choice(unique_streams, size=len(unique_streams), replace=True)
        
        # Собираем индексы шагов, принадлежащих сэмплированным стримам
        mask = np.isin(sids, sampled_sids)
        
        # Проверяем, что есть оба класса (self и external)
        if len(np.unique(is_ext[mask])) < 2:
            continue
            
        auc = roc_auc_score(is_ext[mask], residuals[mask])
        auc_scores.append(auc)

    if not auc_scores:
        print("[Error] Not enough class diversity in bootstrap samples.")
        return

    mean_auc = np.mean(auc_scores)
    ci_low, ci_high = np.percentile(auc_scores, [2.5, 97.5])
    
    print(f"\n[Bootstrap Result] Design={args.design} Variant={args.variant} Control={args.control} L={args.layer}")
    print(f"  Mean AUC: {mean_auc:.4f}")
    print(f"  95% CI:   [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Проверка критериев
    if args.design == "O" and ci_low >= 0.85:
        print("  VERDICT: PASS (Observation AUC > 0.85)")
    elif args.design == "R" and ci_high < 0.75:
        print("  VERDICT: PASS (Reactive AUC < 0.75)")
    else:
        print("  VERDICT: INCONCLUSIVE / FAIL")

if __name__ == "__main__":
    main()