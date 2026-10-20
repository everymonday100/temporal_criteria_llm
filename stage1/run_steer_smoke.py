"""run_steer_smoke.py — Phase 1 smoke v4: latent steering on correctness.

Changes from v3:
  1. Probe target = correctness (binary), not confidence.
     At T=0.2 the confidence range was 0.855–1.000, which made the
     v3 probe degenerate (r=0.998 on train, ~0.0 on held-out).
     At T=0.7 accuracy variance is larger and correctness is the
     variable we actually want to influence.
  2. Baseline temperature = 0.7 (matched temperature in test).
  3. Alphas = [0.1, 0.2, 0.3] (soft steering, 10–30% of ||h||).
     Literature (Rimsky et al. CAA, Turner et al. activation addition)
     uses 10–30% of hidden norm; v3 used 50–200% and mostly broke
     generation.
  4. n default = 150 (test split ~75).
  5. No PCA. Ridge / logistic regression with L2 already regularizes
     and preserves the axes in which a semantic direction might live.
  6. Probe AUC reported on the held-out half, not in-sample.
  7. diagnostic: correlation between probe prediction and correctness.

Protocol (n=150, alphas=[0.1, 0.2, 0.3], T=0.7):
  Phase A — extract h at end-of-prompt, generate baseline at T=0.7,
            label by correctness.
  Phase B — 50/50 split. Logistic regression probe on h → correct.
            Direction d = normalize(coef).
  Phase C — on test half, four conditions per alpha:
              baseline (cached), +alpha*d, -alpha*d, +alpha*d_rand.
  Report — accuracy on valid generations, valid rate, verdict.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import extract_answer
from common.datasets import load_gsm8k, gsm8k_prompt

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\steer")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--max_new", type=int, default=290)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Baseline and test temperature. Same for all "
                        "conditions in Phase C.")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------
# Layer access
# ----------------------------------------------------------------------
def _get_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise RuntimeError(f"Cannot access layer {layer_idx} on this model")


# ----------------------------------------------------------------------
# Validity
# ----------------------------------------------------------------------
def is_valid(text, min_words=15):
    if not text:
        return False
    if len(text.split()) < min_words:
        return False
    if not re.search(r"\d", text) and "=" not in text:
        return False
    words = text.split()
    if len(words) > 15:
        tail = words[-20:]
        if len(set(tail)) < 5:
            return False
    return True


# ----------------------------------------------------------------------
# Forward / generation
# ----------------------------------------------------------------------
@torch.inference_mode()
def extract_hidden(model, tok, prompt, layer_idx):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()


@torch.inference_mode()
def generate(model, tok, prompt, max_new, seed, temperature,
             layer_idx=None, direction_np=None, alpha=0.0):
    """Unified generate. If direction_np is not None, inject
    alpha * direction at layer_idx, last prompt position."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    handle = None
    if direction_np is not None:
        layer = _get_layer(model, layer_idx)
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        d_t = torch.from_numpy(direction_np).to(
            device=model_device, dtype=torch.float32)

        def pre_hook(module, args):
            if not args:
                return args
            hs = args[0]
            if hs.shape[1] != prompt_len:
                return args
            h = hs.clone()
            inject = (alpha * d_t).to(dtype=h.dtype, device=h.device)
            h[:, -1, :] = h[:, -1, :] + inject
            return (h,) + args[1:]
        handle = layer.register_forward_pre_hook(pre_hook)

    try:
        torch.manual_seed(seed)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()

    text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text, is_valid(text)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    problems = load_gsm8k(args.n)
    print(f"[steer v4] n={len(problems)}  model={args.model}  "
          f"layer={args.layer}  T={args.temperature}  "
          f"alphas={args.alphas}")

    tok, model = load_model(args.model)
    _get_layer(model, args.layer)

    # ---------- Phase A ----------
    cache = []
    t0 = time.time()
    for i, p in enumerate(problems):
        prompt = gsm8k_prompt(p["question"])
        h = extract_hidden(model, tok, prompt, args.layer)
        text, valid = generate(
            model, tok, prompt,
            max_new=args.max_new,
            seed=args.seed + p["i"],
            temperature=args.temperature,
        )
        pred = extract_answer(text)
        ok = int(valid and pred == p["reference"])
        cache.append({
            "i": p["i"],
            "h": h,
            "correct": ok,
            "valid": int(valid),
            "pred": pred,
            "ref": p["reference"],
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
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
    print(f"\n[steer v4] split: train={len(train)}  test={len(test)}")

    h_tr = np.array([c["h"] for c in train])
    y_tr = np.array([c["correct"] for c in train])
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    print(f"[steer v4] train labels: {n_pos} correct  {n_neg} incorrect")

    if n_pos < 5 or n_neg < 5:
        print("[steer v4] too few examples in one class. "
              "Increase --n or lower --temperature.")
        return

    # ---------- Probe ----------
    # Scale features for numerical stability of logistic regression.
    feat_mean = h_tr.mean(axis=0)
    feat_std = h_tr.std(axis=0) + 1e-6
    h_tr_z = (h_tr - feat_mean) / feat_std

    probe = LogisticRegression(
        max_iter=3000, C=1.0, solver="lbfgs").fit(h_tr_z, y_tr)
    p_tr = probe.predict_proba(h_tr_z)[:, 1]
    try:
        auc_tr = roc_auc_score(y_tr, p_tr)
    except ValueError:
        auc_tr = float("nan")
    print(f"[steer v4] probe train AUC (in-sample): {auc_tr:.3f}")

    # Held-out probe AUC on test half — the honest diagnostic.
    h_te = np.array([c["h"] for c in test])
    y_te = np.array([c["correct"] for c in test])
    h_te_z = (h_te - feat_mean) / feat_std
    if y_te.sum() > 0 and y_te.sum() < len(y_te):
        p_te = probe.predict_proba(h_te_z)[:, 1]
        auc_te = roc_auc_score(y_te, p_te)
        print(f"[steer v4] probe test AUC (held-out):   {auc_te:.3f}")
    else:
        auc_te = float("nan")
        print("[steer v4] test half has one class only; AUC undefined")

    # ---------- Direction ----------
    d_raw = probe.coef_[0].astype(np.float64)
    # Undo z-scoring to express direction in original space
    d_raw = d_raw / feat_std
    d_raw_norm = np.linalg.norm(d_raw)
    if d_raw_norm < 1e-9:
        print("[steer v4] degenerate direction. Aborting.")
        return
    d = (d_raw / d_raw_norm).astype(np.float32)

    h_norms = np.array([np.linalg.norm(c["h"]) for c in cache])
    h_scale = float(h_norms.mean())
    print(f"[steer v4] ||d_raw|| = {d_raw_norm:.4f}  "
          f"mean ||h|| = {h_scale:.2f}")

    # Random control
    rng2 = np.random.default_rng(args.seed + 999)
    d_rand = rng2.standard_normal(d.shape).astype(np.float32)
    d_rand = d_rand / np.linalg.norm(d_rand)

    # ---------- Phase C ----------
    conditions = ["baseline", "pos", "neg", "random"]
    results = {c: {a: {"valid": 0, "correct": 0, "n": 0}
                   for a in args.alphas} for c in conditions}

    n_test = len(test)
    print(f"\n[steer v4] test-phase: {len(args.alphas)} alphas × "
          f"4 conditions × {n_test} problems")

    t1 = time.time()
    for k, c in enumerate(test):
        p = next(pp for pp in problems if pp["i"] == c["i"])
        prompt = gsm8k_prompt(p["question"])
        ref = p["reference"]
        seed_i = args.seed + 10_000 + p["i"]

        base_valid = c["valid"]
        base_ok = c["correct"] if base_valid else 0

        for a in args.alphas:
            alpha_val = a * h_scale

            results["baseline"][a]["valid"] += base_valid
            results["baseline"][a]["correct"] += base_ok
            results["baseline"][a]["n"] += 1

            t_p, v_p = generate(
                model, tok, prompt,
                max_new=args.max_new, seed=seed_i,
                temperature=args.temperature,
                layer_idx=args.layer, direction_np=d, alpha=alpha_val)
            ok_p = int(v_p and extract_answer(t_p) == ref)
            results["pos"][a]["valid"] += int(v_p)
            results["pos"][a]["correct"] += ok_p
            results["pos"][a]["n"] += 1

            t_n, v_n = generate(
                model, tok, prompt,
                max_new=args.max_new, seed=seed_i,
                temperature=args.temperature,
                layer_idx=args.layer, direction_np=d, alpha=-alpha_val)
            ok_n = int(v_n and extract_answer(t_n) == ref)
            results["neg"][a]["valid"] += int(v_n)
            results["neg"][a]["correct"] += ok_n
            results["neg"][a]["n"] += 1

            t_r, v_r = generate(
                model, tok, prompt,
                max_new=args.max_new, seed=seed_i,
                temperature=args.temperature,
                layer_idx=args.layer, direction_np=d_rand,
                alpha=alpha_val)
            ok_r = int(v_r and extract_answer(t_r) == ref)
            results["random"][a]["valid"] += int(v_r)
            results["random"][a]["correct"] += ok_r
            results["random"][a]["n"] += 1

        if (k + 1) % 5 == 0 or (k + 1) == n_test:
            print(f"  [{k+1:3d}/{n_test}]  "
                  f"({time.time()-t1:.0f}s)")

    # ---------- Summary ----------
    def _rates(r):
        n = max(1, r["n"])
        v = r["valid"]
        acc = r["correct"] / v if v > 0 else 0.0
        return v / n, acc

    print(f"\n=== STEER SMOKE v4 (n_test={n_test}, "
          f"T={args.temperature}) ===")
    header = (f"  {'alpha':>5}  {'cond':<9}  "
              f"{'valid%':>7}  {'acc|valid':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for a in args.alphas:
        for cond in conditions:
            vr, acc = _rates(results[cond][a])
            print(f"  {a:>5.1f}  {cond:<9}  "
                  f"{vr:>6.1%}  {acc:>10.2%}")
        print("  " + "-" * (len(header) - 2))

    # ---------- Verdict ----------
    print("\n[verdict per alpha]")
    verdicts = {}
    for a in args.alphas:
        base_vr, base_acc = _rates(results["baseline"][a])
        pos_vr,  pos_acc  = _rates(results["pos"][a])
        neg_vr,  neg_acc  = _rates(results["neg"][a])
        rnd_vr,  rnd_acc  = _rates(results["random"][a])

        pos_up  = pos_acc - base_acc
        neg_dn  = base_acc - neg_acc
        rnd_eff = rnd_acc - base_acc

        valid_ok = pos_vr > 0.80 * base_vr
        rnd_ok = abs(rnd_eff) < 0.05

        if (pos_up > 0.05 and neg_dn > 0.05
                and rnd_ok and valid_ok):
            tag = "directional_effect"
        elif not valid_ok:
            tag = "generation_broken"
        elif abs(rnd_eff) >= abs(pos_up) and abs(pos_up) > 0.02:
            tag = "non_specific_perturbation"
        elif abs(pos_up) + abs(neg_dn) + abs(rnd_eff) < 0.05:
            tag = "no_effect"
        else:
            tag = "inconclusive"

        verdicts[a] = {
            "pos_up": float(pos_up),
            "neg_dn": float(neg_dn),
            "rnd_eff": float(rnd_eff),
            "tag": tag,
            "valid_rate_pos": float(pos_vr),
            "valid_rate_base": float(base_vr),
            "acc_pos": float(pos_acc),
            "acc_base": float(base_acc),
        }
        print(f"  alpha={a:<4.1f}  "
              f"Δpos={pos_up:+.3f}  Δneg={-neg_dn:+.3f}  "
              f"Δrnd={rnd_eff:+.3f}  "
              f"valid={pos_vr:.2f}/{base_vr:.2f}  →  {tag}")

    tags = [verdicts[a]["tag"] for a in args.alphas]
    n_dir = sum(1 for t in tags if t == "directional_effect")
    if n_dir == len(tags):
        cross = "directional_effect (all alphas)"
    elif n_dir >= 1 and all(t in ("directional_effect", "inconclusive")
                            for t in tags):
        cross = f"partial (directional at {n_dir}/{len(tags)} alphas)"
    elif all(t == "no_effect" for t in tags):
        cross = "no_effect (all alphas)"
    elif all(t == "generation_broken" for t in tags):
        cross = "generation_broken (all alphas)"
    else:
        cross = "inconclusive"
    print(f"\n[cross-alpha] {cross}")

    # ---------- Save ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (OUT_DIR /
                f"v4_{args.model}_L{args.layer}_"
                f"T{args.temperature}_n{args.n}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "layer": args.layer,
            "temperature": args.temperature,
            "n_total": len(cache),
            "n_test": n_test,
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
            "probe_auc_train": float(auc_tr),
            "probe_auc_test": float(auc_te),
            "h_scale": h_scale,
            "alphas": args.alphas,
            "results": {
                cond: {
                    str(a): {
                        "valid_rate": float(_rates(results[cond][a])[0]),
                        "acc_on_valid": float(_rates(results[cond][a])[1]),
                        "n": results[cond][a]["n"],
                        "n_valid": results[cond][a]["valid"],
                        "n_correct": results[cond][a]["correct"],
                    } for a in args.alphas
                } for cond in conditions
            },
            "verdicts": {str(a): verdicts[a] for a in args.alphas},
            "cross_alpha": cross,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()