"""run_causality_smoke.py — inject probe.coef_ direction into Qwen at L6.

Takes the direction found by run_probe_scan at the best layer,
injects it at alphas [0.1, 0.2, 0.3, 0.5] during generation at T=0.7,
reports accuracy per condition. Baseline is cached from the scan
(same seed, same T, same n).

Protocol:
  1. n=300 problems, split 150/150.
  2. Train RidgeClassifierCV probe on L6 hidden states (train half).
  3. Extract normalized direction d from probe.coef_ (undo z-scoring).
  4. On test half, generate 5 conditions at matched seed:
       baseline (cached), +a*d, -a*d, +a*d_rand, for a in alphas.
  5. Report accuracy, valid_rate, verdict per alpha.

Interpretation:
  - If +d raises accuracy and -d lowers it (and random does nothing):
    direction is causal. Proceed to halt gate design with this
    direction as a candidate init.
  - If random has the same effect: direction is not causal,
    steering is non-specific at this amplitude.
  - If all conditions equal baseline: direction is not causal even
    if predictive. Phase 1 negative.

Output: data/stage1/steer/causality_L6_n300.json
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
ALPHAS = np.logspace(-3, 3, 13)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--max_new", type=int, default=290)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--inject_alphas", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


def _get_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise RuntimeError(f"Cannot access layer {layer_idx}")


@torch.inference_mode()
def extract_hidden(model, tok, prompt, layer_idx):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()


@torch.inference_mode()
def generate(model, tok, prompt, max_new, seed, temperature,
             layer_idx=None, direction_np=None, alpha=0.0):
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


def main():
    args = parse_args()
    problems = load_gsm8k(args.n)
    print(f"[causal] n={len(problems)}  layer={args.layer}  "
          f"T={args.temperature}  inject_alphas={args.inject_alphas}")

    tok, model = load_model(args.model)

    # ---------- Phase A: h + baseline ----------
    cache = []
    t0 = time.time()
    for i, p in enumerate(problems):
        prompt = gsm8k_prompt(p["question"])
        h = extract_hidden(model, tok, prompt, args.layer)
        text, valid = generate(
            model, tok, prompt,
            max_new=args.max_new, seed=args.seed + p["i"],
            temperature=args.temperature,
        )
        pred = extract_answer(text)
        ok = int(valid and pred == p["reference"])
        cache.append({
            "i": p["i"], "h": h, "correct": ok,
            "valid": int(valid), "pred": pred, "ref": p["reference"],
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(problems):
            print(f"  [{i+1:3d}/{len(problems)}]  "
                  f"({time.time()-t0:.0f}s)")

    # ---------- Split ----------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cache))
    n_train = len(cache) // 2
    train = [cache[i] for i in perm[:n_train]]
    test  = [cache[i] for i in perm[n_train:]]
    print(f"\n[causal] split: train={len(train)}  test={len(test)}")

    h_tr = np.array([c["h"] for c in train])
    y_tr = np.array([c["correct"] for c in train])
    print(f"[causal] train: {int(y_tr.sum())} correct, "
          f"{int(len(y_tr)-y_tr.sum())} incorrect")

    # ---------- Probe ----------
    mu = h_tr.mean(axis=0)
    sd = h_tr.std(axis=0) + 1e-6
    h_tr_z = (h_tr - mu) / sd

    probe = RidgeClassifierCV(alphas=ALPHAS).fit(h_tr_z, y_tr)

    # Held-out AUC (verify scan reproduced)
    h_te = np.array([c["h"] for c in test])
    y_te = np.array([c["correct"] for c in test])
    h_te_z = (h_te - mu) / sd
    s_te = probe.decision_function(h_te_z)
    try:
        auc_te = float(roc_auc_score(y_te, s_te))
    except ValueError:
        auc_te = float("nan")
    print(f"[causal] held-out probe AUC = {auc_te:.3f}")

    # ---------- Direction ----------
    d_raw = probe.coef_[0].astype(np.float64) / sd
    d_norm = np.linalg.norm(d_raw)
    if d_norm < 1e-9:
        print("[causal] degenerate direction.")
        return
    d = (d_raw / d_norm).astype(np.float32)

    h_norms = np.array([np.linalg.norm(c["h"]) for c in cache])
    h_scale = float(h_norms.mean())
    print(f"[causal] ||d_raw|| = {d_norm:.4f}  mean ||h|| = {h_scale:.2f}")

    # Random control
    rng2 = np.random.default_rng(args.seed + 999)
    d_rand = rng2.standard_normal(d.shape).astype(np.float32)
    d_rand = d_rand / np.linalg.norm(d_rand)

    # ---------- Phase C ----------
    conditions = ["baseline", "pos", "neg", "random"]
    results = {c: {a: {"valid": 0, "correct": 0, "n": 0}
                   for a in args.inject_alphas} for c in conditions}

    n_test = len(test)
    print(f"\n[causal] test: {len(args.inject_alphas)} alphas × "
          f"4 cond × {n_test}")

    t1 = time.time()
    for k, c in enumerate(test):
        p = next(pp for pp in problems if pp["i"] == c["i"])
        prompt = gsm8k_prompt(p["question"])
        ref = p["reference"]
        seed_i = args.seed + 10_000 + p["i"]

        for a in args.inject_alphas:
            alpha_val = a * h_scale

            results["baseline"][a]["valid"] += c["valid"]
            results["baseline"][a]["correct"] += c["correct"]
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

        if (k + 1) % 10 == 0 or (k + 1) == n_test:
            print(f"  [{k+1:3d}/{n_test}]  ({time.time()-t1:.0f}s)")

    # ---------- Summary ----------
    def _rates(r):
        n = max(1, r["n"]); v = r["valid"]
        return v / n, (r["correct"] / v if v > 0 else 0.0)

    print(f"\n=== CAUSALITY SMOKE (L={args.layer}, n_test={n_test}) ===")
    print(f"  {'alpha':>5}  {'cond':<9}  {'valid%':>7}  {'acc|valid':>10}")
    print("  " + "-" * 42)
    for a in args.inject_alphas:
        for cond in conditions:
            vr, acc = _rates(results[cond][a])
            print(f"  {a:>5.1f}  {cond:<9}  {vr:>6.1%}  {acc:>10.2%}")
        print("  " + "-" * 42)

    # ---------- Verdict ----------
    print("\n[verdict per alpha]")
    verdicts = {}
    for a in args.inject_alphas:
        _, base_acc = _rates(results["baseline"][a])
        _, pos_acc  = _rates(results["pos"][a])
        _, neg_acc  = _rates(results["neg"][a])
        _, rnd_acc  = _rates(results["random"][a])
        pos_vr, _ = _rates(results["pos"][a])
        base_vr, _ = _rates(results["baseline"][a])

        pos_up = pos_acc - base_acc
        neg_dn = base_acc - neg_acc
        rnd_eff = rnd_acc - base_acc
        valid_ok = pos_vr > 0.85 * base_vr

        if (pos_up > 0.05 and neg_dn > 0.05
                and abs(rnd_eff) < 0.04 and valid_ok):
            tag = "causal_direction"
        elif not valid_ok:
            tag = "generation_broken"
        elif abs(rnd_eff) >= abs(pos_up) and abs(pos_up) > 0.02:
            tag = "non_specific"
        elif abs(pos_up) + abs(neg_dn) + abs(rnd_eff) < 0.04:
            tag = "no_effect"
        else:
            tag = "inconclusive"
        verdicts[a] = {"pos_up": float(pos_up), "neg_dn": float(neg_dn),
                       "rnd_eff": float(rnd_eff), "tag": tag}
        print(f"  a={a:<4.1f}  Δpos={pos_up:+.3f}  Δneg={-neg_dn:+.3f}  "
              f"Δrnd={rnd_eff:+.3f}  →  {tag}")

    tags = [verdicts[a]["tag"] for a in args.inject_alphas]
    if all(t == "causal_direction" for t in tags):
        cross = "causal_direction (all alphas)"
    elif any(t == "causal_direction" for t in tags):
        cross = "partial"
    elif all(t == "no_effect" for t in tags):
        cross = "no_effect (all alphas)"
    else:
        cross = "inconclusive"
    print(f"\n[cross-alpha] {cross}")

    # ---------- Save ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"causality_L{args.layer}_n{args.n}_T{args.temperature}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "layer": args.layer,
            "n": len(cache), "n_test": n_test,
            "temperature": args.temperature,
            "held_out_probe_auc": float(auc_te),
            "h_scale": h_scale,
            "inject_alphas": args.inject_alphas,
            "results": {
                c: {str(a): {
                    "valid_rate": float(_rates(results[c][a])[0]),
                    "acc_on_valid": float(_rates(results[c][a])[1]),
                } for a in args.inject_alphas}
                for c in conditions
            },
            "verdicts": {str(a): verdicts[a] for a in args.inject_alphas},
            "cross_alpha": cross,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()