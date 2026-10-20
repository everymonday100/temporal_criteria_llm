"""continuity_probe_m3.py — Probes for M3a (Jensen artifact) and M3c (Decision-rule dependence)."""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\continuity")

# Utility Matrix (from Reviewer)
# 0: Direct, 1: CoT, 2: PAL
U_SUCCESS = np.array([1.0, 0.6, 1.0])  # Direct=1.0, CoT=0.6, PAL=1.0
U_FAIL    = np.array([-0.5, 0.6, -0.2]) # Direct=-0.5, CoT=0.6, PAL=-0.2

def main():
    print("[M3] Loading features...")
    data = np.load(OUT_DIR / "continuity_features.npz", allow_pickle=True)
    
    # Features of N-1 predict strategy of N
    H = data["h_18"]          # shape (200, 2048)
    strategies_str = data["strategies"] # shape (200,)
    
    # Map strings to ints
    strat_map = {"Direct": 0, "CoT": 1, "PAL": 2}
    strat_int = np.array([strat_map[s] for s in strategies_str])
    
    X = H[:-1]
    y = strat_int[1:]
    
    # Chronological split: 100 train, 99 test
    split = 100
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Standardize features (crucial for Ridge/LogReg with h_18)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Shuffled control
    np.random.seed(42)
    y_train_shuffled = np.random.permutation(y_train)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ==========================================
    # M3a: Jensen Artifact Baseline
    # ==========================================
    print("\n--- M3a: Baseline & Global Optimum ---")
    
    # 1. Frozen Default (Always CoT)
    frozen_pred = np.full_like(y_test, 1)
    frozen_acc = accuracy_score(y_test, frozen_pred)
    frozen_f1 = f1_score(y_test, frozen_pred, average="macro")
    
    # 2. Global Mean-Optimal (Most frequent in train)
    global_opt_strat = np.bincount(y_train).argmax()
    global_pred = np.full_like(y_test, global_opt_strat)
    global_acc = accuracy_score(y_test, global_pred)
    global_f1 = f1_score(y_test, global_pred, average="macro")
    
    # 3. Trained Classifier (Context)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    clf_acc = accuracy_score(y_test, preds)
    clf_f1 = f1_score(y_test, preds, average="macro")
    
    # 4. Shuffled Context Control
    clf_shuf = LogisticRegression(max_iter=1000, C=1.0)
    clf_shuf.fit(X_train, y_train_shuffled)
    preds_shuf = clf_shuf.predict(X_test)
    shuf_acc = accuracy_score(y_test, preds_shuf)
    shuf_f1 = f1_score(y_test, preds_shuf, average="macro")

    print(f"Frozen Default (Always CoT): Acc={frozen_acc:.4f}, F1={frozen_f1:.4f}")
    print(f"Global Mean-Optimal        : Acc={global_acc:.4f}, F1={global_f1:.4f}")
    print(f"Context Classifier (M3a)   : Acc={clf_acc:.4f}, F1={clf_f1:.4f}")
    print(f"Shuffled Context           : Acc={shuf_acc:.4f}, F1={shuf_f1:.4f}")
    
    if clf_f1 <= global_f1:
        print("  ⚠️ M3a Result: JENSEN ARTIFACT. Context does not beat global optimum.")
    else:
        print("  ✅ M3a Result: Context beats global optimum! (Signal exists)")

    # ==========================================
    # M3c: Decision-Rule Dependence
    # ==========================================
    print("\n--- M3c: Argmax vs Cost-Aware ---")
    
    probs = clf.predict_proba(X_test)
    
    # Rule 1: Argmax
    argmax_preds = np.argmax(probs, axis=1)
    argmax_acc = accuracy_score(y_test, argmax_preds)
    argmax_f1 = f1_score(y_test, argmax_preds, average="macro")
    
    # Rule 2: Cost-Aware Expected F1 (Utility Matrix)
    # EU(a) = P(a) * U_success(a) + (1 - P(a)) * U_fail(a)
    expected_utilities = probs * U_SUCCESS + (1 - probs) * U_FAIL
    cost_aware_preds = np.argmax(expected_utilities, axis=1)
    cost_aware_acc = accuracy_score(y_test, cost_aware_preds)
    cost_aware_f1 = f1_score(y_test, cost_aware_preds, average="macro")
    
    print(f"Argmax Policy       : Acc={argmax_acc:.4f}, F1={argmax_f1:.4f}")
    print(f"Cost-Aware Policy   : Acc={cost_aware_acc:.4f}, F1={cost_aware_f1:.4f}")
    
    # Check how often Cost-Aware falls back to CoT (Safe Default)
    safe_default_frac = np.mean(cost_aware_preds == 1)
    print(f"Cost-Aware collapse to CoT: {safe_default_frac*100:.1f}% of cases")
    
    if argmax_f1 != cost_aware_f1:
        print("  ✅ M3c Result: DECISION-RULE DEPENDENCE CONFIRMED.")
        if cost_aware_f1 > argmax_f1:
            print("     (Inverted polarity: Cost-aware acts as a regularizer against classifier overconfidence, improving F1).")
        else:
            print("     (Standard polarity: Cost-aware collapsed to safe default, degrading F1).")
    else:
        print("  ⚠️ M3c Result: No difference between rules. Check utility matrix.")

if __name__ == "__main__":
    main()