"""run_probe_scan.py — scan layers for a transferable linear correctness probe.

Phase A of the external steering pipeline, extended to multiple layers
in a single forward pass per problem.

Probe: RidgeClassifierCV with leave-one-out CV over alpha in
[1e-3, 1e3]. This is the standard configuration for the p >> n regime
of mechanistic interpretability.

Controls (v3):
  - L0 baseline: same probe on the embedding-layer hidden state.
    If L0 AUC matches deeper layers, the signal is a prompt-level
    property (length, question type, difficulty), not correctness.
  - Label-shuffle control: 10 shuffles of y_train, train and evaluate
    on y_test. If shuffled AUC matches true AUC, the signal is an
    artifact of the split structure, not of the labels.

For each layer L in --layers:
    train RidgeClassifierCV on (h_L, y_train)
    report train AUC, held-out AUC with bootstrap CI, chosen alpha,
    and shuffled-labels AUC (mean over 10 shuffles).

L0 is always evaluated as a reference; pass or omit from --layers.

Verdict rules:
  candidate_layer_L* : held-out AUC >= threshold AND shuffle AUC < 0.55
                       AND L0 AUC < 0.55
  confounded_*       : held-out AUC >= threshold but shuffle AUC >= 0.55
                       or L0 AUC >= 0.55
  marginal           : held-out AUC in [0.55, threshold) with clean controls
  no_transferable_axis : all layers below 0.55

Output: data/stage1/steer/probe_scan_{model}_n{n}_T{T}.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import extract_answer
from common.datasets import load_gsm8k, gsm8k_prompt

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\steer")
BOOTSTRAP_B = 500
SHUFFLE_REPS = 10
ALPHAS = np.logspace(-3, 3, 13)  # 0.001 … 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--layers", type=int, nargs="+",
                   default=[6, 12, 18, 24, 30])
    p.add_argument("--max_new", type=int, default=290)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--auc_threshold", type=float, default=0.65)
    p.add_argument("--shuffle_threshold", type=float, default=0.55,
                   help="Shuffled-labels AUC above which a layer's "
                        "signal is treated as a split artifact.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------
def is_valid(text, min_words=15):
    if not text:
        return False
    if len(text.split()) < min_words:
        return False
    if not re.search(r"\d", text) and "=" not in text:
        return False
    words = text.split()
    if len(words) > 15 and len(set(words[-20:])) < 5:
        return False
    return True


@torch.inference_mode()
def extract_hidden_all(model, tok, prompt, layer_ids):
    """Extract hidden state at each requested layer + L0 (embedding),
    last prompt position."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states  # tuple (n_layers+1) of (1, T, H)
    all_ids = sorted(set([0] + list(layer_ids)))
    return {L: hs[L][0, -1, :].float().cpu().numpy()
            for L in all_ids if L < len(hs)}


@torch.inference_mode()
def generate_baseline(model, tok, prompt, max_new, seed, temperature):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text, is_valid(text)


# ----------------------------------------------------------------------
def bootstrap_auc(y_true, scores, B=BOOTSTRAP_B, seed=42):
    y = np.asarray(y_true)
    s = np.asarray(scores)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(y, s))
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        try:
            boots.append(roc_auc_score(yb, sb))
        except ValueError:
            continue
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def shuffle_control(h_tr_z, h_te_z, y_tr, y_te, seed):
    """Train probe on shuffled y_tr, evaluate on true y_te. Return
    (mean, max) AUC across SHUFFLE_REPS permutations."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(SHUFFLE_REPS):
        y_shuf = rng.permutation(y_tr)
        probe_s = RidgeClassifierCV(alphas=ALPHAS).fit(h_tr_z, y_shuf)
        s_te = probe_s.decision_function(h_te_z)
        try:
            aucs.append(float(roc_auc_score(y_te, s_te)))
        except ValueError:
            continue
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.max(aucs))


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    problems = load_gsm8k(args.n)
    print(f"[scan] n={len(problems)}  model={args.model}  "
          f"layers={args.layers}  T={args.temperature}")
    print(f"[scan] probe = RidgeClassifierCV, "
          f"alphas = logspace(-3, 3, 13)")
    print(f"[scan] controls: L0 baseline + {SHUFFLE_REPS}-rep shuffle")

    tok, model = load_model(args.model)

    # ---------- Collect ----------
    cache = []
    t0 = time.time()
    for i, p in enumerate(problems):
        prompt = gsm8k_prompt(p["question"])
        h_by_layer = extract_hidden_all(model, tok, prompt, args.layers)
        text, valid = generate_baseline(
            model, tok, prompt,
            max_new=args.max_new,
            seed=args.seed + p["i"],
            temperature=args.temperature,
        )
        pred = extract_answer(text)
        ok = int(valid and pred == p["reference"])
        cache.append({
            "i": p["i"],
            "h": h_by_layer,
            "correct": ok,
            "valid": int(valid),
            "pred": pred,
            "ref": p["reference"],
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(problems):
            elapsed = time.time() - t0
            n_ok = sum(c["correct"] for c in cache)
            n_v = sum(c["valid"] for c in cache)
            print(f"  [{i+1:3d}/{len(problems)}]  "
                  f"valid={n_v}/{i+1}  correct={n_ok}  "
                  f"({elapsed:.0f}s)")

    # ---------- Split ----------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cache))
    n_train = len(cache) // 2
    train = [cache[i] for i in perm[:n_train]]
    test  = [cache[i] for i in perm[n_train:]]
    print(f"\n[scan] split: train={len(train)}  test={len(test)}")

    y_tr = np.array([c["correct"] for c in train])
    y_te = np.array([c["correct"] for c in test])
    n_pos_tr = int(y_tr.sum())
    n_neg_tr = int(len(y_tr) - n_pos_tr)
    n_pos_te = int(y_te.sum())
    n_neg_te = int(len(y_te) - n_pos_te)
    print(f"[scan] train: {n_pos_tr} correct, {n_neg_tr} incorrect")
    print(f"[scan] test:  {n_pos_te} correct, {n_neg_te} incorrect")

    if n_pos_tr < 5 or n_neg_tr < 5:
        print("[scan] too few examples in one train class. Aborting.")
        return

    # ---------- Per-layer probe with controls ----------
    print(f"\n=== PROBE SCAN (RidgeClassifierCV) ===")
    header = (f"  {'layer':>5}  {'train':>7}  {'test':>7}  "
              f"{'95% CI':>17}  {'alpha':>10}  "
              f"{'shuf_mean':>10}  {'shuf_max':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    # Ensure L0 is included for reference
    scan_layers = sorted(set([0] + args.layers))

    for L in scan_layers:
        if L not in cache[0]["h"]:
            continue
        h_tr = np.array([c["h"][L] for c in train])
        h_te = np.array([c["h"][L] for c in test])

        mu = h_tr.mean(axis=0)
        sd = h_tr.std(axis=0) + 1e-6
        h_tr_z = (h_tr - mu) / sd
        h_te_z = (h_te - mu) / sd

        probe = RidgeClassifierCV(alphas=ALPHAS).fit(h_tr_z, y_tr)

        s_tr = probe.decision_function(h_tr_z)
        try:
            auc_tr = float(roc_auc_score(y_tr, s_tr))
        except ValueError:
            auc_tr = float("nan")

        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            auc_te, lo, hi = float("nan"), float("nan"), float("nan")
        else:
            s_te = probe.decision_function(h_te_z)
            auc_te, lo, hi = bootstrap_auc(y_te, s_te, seed=args.seed)

        # Shuffle control
        sh_mean, sh_max = shuffle_control(
            h_tr_z, h_te_z, y_tr, y_te, seed=args.seed + 7777 + L)

        chosen_alpha = float(probe.alpha_)
        results[L] = {
            "train_auc": auc_tr,
            "test_auc": auc_te,
            "test_auc_ci_low": lo,
            "test_auc_ci_high": hi,
            "alpha": chosen_alpha,
            "shuffle_auc_mean": sh_mean,
            "shuffle_auc_max": sh_max,
            "n_train": len(train),
            "n_test": len(test),
        }
        ci_str = f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "n/a"
        marker = "  <-- L0" if L == 0 else ""
        print(f"  {L:>5}  {auc_tr:>7.3f}  {auc_te:>7.3f}  "
              f"{ci_str:>17}  {chosen_alpha:>10.3f}  "
              f"{sh_mean:>10.3f}  {sh_max:>9.3f}{marker}")

    # ---------- Verdict ----------
    def _safe(L):
        a = results[L]["test_auc"]
        return a if not np.isnan(a) else -1

    # Only consider requested layers (not L0) for "best"
    requested = [L for L in args.layers if L in results]
    if not requested:
        print("\n[scan] no requested layers were collected. Aborting.")
        return

    l0_auc = results.get(0, {}).get("test_auc", float("nan"))
    print(f"\n[scan] L0 embedding baseline AUC = {l0_auc:.3f}")

    best_layer = max(requested, key=_safe)
    best = results[best_layer]
    best_auc = best["test_auc"]
    best_shuf = best["shuffle_auc_mean"]

    print(f"[scan] best requested layer: L={best_layer}  "
          f"test_AUC={best_auc:.3f}  "
          f"shuffle_AUC={best_shuf:.3f}  "
          f"L0_AUC={l0_auc:.3f}")

    # Verdict logic
    shuffle_mean_ok = best_shuf < args.shuffle_threshold
    shuffle_max_ok = best["shuffle_auc_max"] < 0.60
    l0_ok = (np.isnan(l0_auc) or l0_auc < args.shuffle_threshold)

    if (best_auc >= args.auc_threshold
            and shuffle_mean_ok and shuffle_max_ok and l0_ok):
        tag = f"candidate_layer_L{best_layer}"
        msg = (f"Layer {best_layer} AUC {best_auc:.3f} >= {args.auc_threshold}, "
               f"shuffle mean/max {best_shuf:.3f}/{best['shuffle_auc_max']:.3f} "
               f"< {args.shuffle_threshold}/0.60, "
               f"L0 AUC {l0_auc:.3f} < {args.shuffle_threshold}. "
               f"Signal is not a split or length artifact. "
               f"Proceed to causality smoke on L{best_layer}.")
    elif best_auc >= args.auc_threshold and not shuffle_mean_ok:
        tag = "confounded_by_split"
        msg = (f"Layer {best_layer} AUC {best_auc:.3f} exceeds threshold, but "
               f"shuffle mean AUC {best_shuf:.3f} >= {args.shuffle_threshold}. "
               f"The apparent signal is a split-structure artifact. "
               f"Do not proceed to steering.")
    elif best_auc >= args.auc_threshold and not shuffle_max_ok:
        tag = "unstable_signal"
        msg = (f"Layer {best_layer} AUC {best_auc:.3f} exceeds threshold and "
               f"shuffle mean AUC {best_shuf:.3f} is clean, but shuffle max "
               f"{best['shuffle_auc_max']:.3f} >= 0.60. Signal is unstable "
               f"relative to label permutation at this n. Increase --n to "
               f"300+ before proceeding.")
    elif best_auc >= args.auc_threshold and not l0_ok:
        tag = "confounded_by_prompt"
        msg = (f"Layer {best_layer} AUC {best_auc:.3f} exceeds threshold, but "
               f"L0 embedding baseline AUC {l0_auc:.3f} >= "
               f"{args.shuffle_threshold}. The signal is a prompt-level "
               f"property (length, question type, difficulty), not "
               f"correctness. Do not proceed to steering.")
    elif best_auc >= 0.55:
        tag = "marginal"
        msg = (f"Best AUC {best_auc:.3f} is marginal (0.55–{args.auc_threshold}) "
               f"with clean controls. Increase --n and re-test; if AUC is "
               f"stable, treat as weak signal.")
    else:
        tag = "no_transferable_axis"
        msg = (f"Best AUC {best_auc:.3f} is at or below chance across all "
               f"layers tested. External linear steering does not have a "
               f"transferable direction on this model and task.")

    print(f"  TAG: {tag}")
    print(f"  {msg}")

    # ---------- Save ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (OUT_DIR /
                f"probe_scan_{args.model}_n{args.n}_T{args.temperature}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "n": len(cache),
            "n_train": len(train),
            "n_test": len(test),
            "temperature": args.temperature,
            "layers": scan_layers,
            "auc_threshold": args.auc_threshold,
            "shuffle_threshold": args.shuffle_threshold,
            "shuffle_reps": SHUFFLE_REPS,
            "probe": "RidgeClassifierCV",
            "alphas_grid": [float(a) for a in ALPHAS],
            "results": {
                str(L): {
                    "train_auc": results[L]["train_auc"],
                    "test_auc": results[L]["test_auc"],
                    "test_auc_ci_low": results[L]["test_auc_ci_low"],
                    "test_auc_ci_high": results[L]["test_auc_ci_high"],
                    "alpha": results[L]["alpha"],
                    "shuffle_auc_mean": results[L]["shuffle_auc_mean"],
                    "shuffle_auc_max": results[L]["shuffle_auc_max"],
                    "n_train": results[L]["n_train"],
                    "n_test": results[L]["n_test"],
                }
                for L in scan_layers if L in results
            },
            "l0_auc": float(l0_auc) if not np.isnan(l0_auc) else None,
            "best_layer": best_layer,
            "best_auc": float(best_auc),
            "tag": tag,
            "message": msg,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()