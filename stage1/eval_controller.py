"""eval_controller.py — Four-condition evaluation (v4, gsm-hard).

Patches vs v3:
  - Per-problem correctness saved for A/B/C/D. Required for paired
    bootstrap on Δ(D−B) and Δ(D−C).
  - Per-problem ||c_t|| saved for condition D. Required for scatter
    plot against difficulty.
  - Paired bootstrap CI printed at the end (B−A, D−B, D−C).
  - All arrays saved to data/stage1/controller/per_problem_*.npz.
  - Difficulty computed once and reused (was computed twice: once in
    eval, once at the end for correlation).

Usage:
  python stage1/eval_controller.py --model qwen --n_test 300 --n_train_skip 700
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import extract_answer
from common.datasets import load_gsm_hard, gsm_hard_prompt
from common.controller_model import ControllerModel
from common.stats import paired_bootstrap

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\controller")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--n_train_skip", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new", type=int, default=290)
    return p.parse_args()


@torch.inference_mode()
def generate_greedy(model, tok, prompt, max_new, seed):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


@torch.inference_mode()
def get_c_norm(ctrl_model, tok, prompt, device):
    inputs = tok(prompt, return_tensors="pt").to(device)
    _ = ctrl_model(inputs["input_ids"],
                   attention_mask=inputs["attention_mask"],
                   use_cache=False)
    if ctrl_model._last_c_norms is not None:
        return float(ctrl_model._last_c_norms.mean().item())
    return 0.0


def main():
    args = parse_args()
    print(f"[ctrl eval] dataset=gsm-hard model={args.model} "
          f"n_test={args.n_test}")

    tok, base = load_model(args.model)
    base.config.use_cache = True
    device = base.device

    all_problems = load_gsm_hard()
    test_problems = all_problems[args.n_train_skip:
                                  args.n_train_skip + args.n_test]
    print(f"[ctrl eval] {len(test_problems)} test problems "
          f"(from index {args.n_train_skip})")

    # ---------- A: frozen base ----------
    print("\n[eval] A: frozen base")
    per_A = []
    per_A_text = []
    for k, p in enumerate(test_problems):
        prompt = gsm_hard_prompt(p["question"])
        text = generate_greedy(base, tok, prompt, args.max_new,
                               args.seed + p["i"])
        pred = extract_answer(text)
        ok = int(pred == p["reference"])
        per_A.append(ok)
        per_A_text.append(pred)
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(test_problems)}]  "
                  f"acc={sum(per_A)/len(per_A):.3f}")
    acc_A = sum(per_A) / len(per_A)
    print(f"  A = {acc_A:.3f}")

    # ---------- Difficulty: reuse per_A. Base fails => hard. ----------
    difficulty = np.array([1.0 - float(ok) for ok in per_A])

    # ---------- Controller model ----------
    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(device)
    ctrl_model.controller.to(device)
    ctrl_model.w_in.to(device)

    ckpt_path = OUT_DIR / f"ctrl_{args.model}_gsmhard_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Train first: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # ---------- B: LoRA-only ----------
    print("\n[eval] B: LoRA-only")
    ctrl_model.model.load_state_dict(state["lora_state"], strict=False)
    ctrl_model.w_in.weight.data.zero_()
    per_B = []
    per_B_text = []
    for k, p in enumerate(test_problems):
        prompt = gsm_hard_prompt(p["question"])
        text = generate_greedy(ctrl_model.model, tok, prompt,
                               args.max_new, args.seed + p["i"])
        pred = extract_answer(text)
        ok = int(pred == p["reference"])
        per_B.append(ok)
        per_B_text.append(pred)
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(test_problems)}]  "
                  f"acc={sum(per_B)/len(per_B):.3f}")
    acc_B = sum(per_B) / len(per_B)
    print(f"  B = {acc_B:.3f}")

    # ---------- C: random controller ----------
    print("\n[eval] C: random controller")
    torch.manual_seed(args.seed + 999)
    for p in ctrl_model.controller.parameters():
        torch.nn.init.normal_(p, std=0.02)
    torch.nn.init.normal_(ctrl_model.w_in.weight, std=0.02)
    per_C = []
    for k, p in enumerate(test_problems):
        prompt = gsm_hard_prompt(p["question"])
        text = generate_greedy(ctrl_model.model, tok, prompt,
                               args.max_new, args.seed + p["i"])
        pred = extract_answer(text)
        ok = int(pred == p["reference"])
        per_C.append(ok)
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(test_problems)}]  "
                  f"acc={sum(per_C)/len(per_C):.3f}")
    acc_C = sum(per_C) / len(per_C)
    print(f"  C = {acc_C:.3f}")

    # ---------- D: trained controller ----------
    print("\n[eval] D: trained controller")
    ctrl_model.controller.load_state_dict(state["controller_state"])
    ctrl_model.w_in.load_state_dict(state["w_in_state"])
    per_D = []
    per_D_text = []
    c_norm_D = []
    for k, p in enumerate(test_problems):
        prompt = gsm_hard_prompt(p["question"])
        text = generate_greedy(ctrl_model.model, tok, prompt,
                               args.max_new, args.seed + p["i"])
        pred = extract_answer(text)
        ok = int(pred == p["reference"])
        per_D.append(ok)
        per_D_text.append(pred)
        c_norm_D.append(get_c_norm(ctrl_model, tok, prompt, device))
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(test_problems)}]  "
                  f"acc={sum(per_D)/len(per_D):.3f}")
    acc_D = sum(per_D) / len(per_D)
    print(f"  D = {acc_D:.3f}")

    # ---------- Diagnostics ----------
    c_norm_D = np.array(c_norm_D)
    mean_c = float(c_norm_D.mean())
    std_c = float(c_norm_D.std())
    cv_c = std_c / (mean_c + 1e-9)

    if np.std(c_norm_D) > 1e-9 and 0.0 < difficulty.mean() < 1.0:
        corr = float(np.corrcoef(c_norm_D, difficulty)[0, 1])
        # Two-tailed p-value via t-distribution approximation
        from scipy.stats import pearsonr
        _, p_val = pearsonr(c_norm_D, difficulty)
    else:
        corr, p_val = 0.0, 1.0

    # ---------- Paired bootstrap ----------
    per_A_arr = np.array(per_A, dtype=float)
    per_B_arr = np.array(per_B, dtype=float)
    per_C_arr = np.array(per_C, dtype=float)
    per_D_arr = np.array(per_D, dtype=float)

    d_ba, lo_ba, hi_ba = paired_bootstrap(per_B_arr, per_A_arr)
    d_db, lo_db, hi_db = paired_bootstrap(per_D_arr, per_B_arr)
    d_dc, lo_dc, hi_dc = paired_bootstrap(per_D_arr, per_C_arr)

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("CONTROLLER EVALUATION (GSM-HARD)")
    print("=" * 70)
    print(f"  A (frozen):        {acc_A:.3f}")
    print(f"  B (LoRA-only):     {acc_B:.3f}")
    print(f"  C (random ctrl):   {acc_C:.3f}")
    print(f"  D (trained ctrl):  {acc_D:.3f}")
    print()
    print(f"  Paired bootstrap on test half (n={len(per_A)}):")
    print(f"    Δ B−A = {d_ba:+.4f}  [{lo_ba:+.4f}, {hi_ba:+.4f}]")
    print(f"    Δ D−B = {d_db:+.4f}  [{lo_db:+.4f}, {hi_db:+.4f}]")
    print(f"    Δ D−C = {d_dc:+.4f}  [{lo_dc:+.4f}, {hi_dc:+.4f}]")
    print()
    print(f"  mean ||c_t||:      {mean_c:.4f}")
    print(f"  std  ||c_t||:      {std_c:.4f}")
    print(f"  CV of ||c_t||:     {cv_c:.3f}")
    print(f"  corr(||c||, diff): {corr:+.3f} (p={p_val:.3e})")

    # ---------- Verdict ----------
    sig_db = lo_db > 0.0
    sig_dc = lo_dc > 0.0

    if sig_db and sig_dc and acc_D - acc_B >= 0.02:
        tag = "controller_contributes"
    elif sig_db and sig_dc:
        tag = "controller_contributes_marginal"
    elif sig_db or sig_dc:
        tag = "partial_significance"
    elif cv_c > 0.5 and mean_c > 0.01:
        tag = "active_but_no_gain"
    elif mean_c < 0.01:
        tag = "no_op"
    else:
        tag = "inconclusive"

    print(f"\n[verdict]")
    print(f"  Significant Δ(D−B): {sig_db}")
    print(f"  Significant Δ(D−C): {sig_dc}")
    print(f"  TAG: {tag}")

    # ---------- Save ----------
    out_path = OUT_DIR / f"ctrl_eval_{args.model}_gsmhard_n{args.n_test}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": "gsm-hard",
            "model": args.model,
            "n_test": args.n_test,
            "n_train_skip": args.n_train_skip,
            "acc_frozen": acc_A,
            "acc_lora": acc_B,
            "acc_random_ctrl": acc_C,
            "acc_trained_ctrl": acc_D,
            "delta_D_vs_B": acc_D - acc_B,
            "delta_D_vs_C": acc_D - acc_C,
            "bootstrap": {
                "B_minus_A": {"mean": d_ba, "ci_low": lo_ba, "ci_high": hi_ba},
                "D_minus_B": {"mean": d_db, "ci_low": lo_db, "ci_high": hi_db},
                "D_minus_C": {"mean": d_dc, "ci_low": lo_dc, "ci_high": hi_dc},
            },
            "mean_c_norm": mean_c,
            "std_c_norm": std_c,
            "cv_c_norm": cv_c,
            "corr_c_difficulty": corr,
            "corr_p_value": float(p_val),
            "tag": tag,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")

    # Per-problem arrays
    per_path = OUT_DIR / f"per_problem_{args.model}_gsmhard_n{args.n_test}.npz"
    np.savez(
        per_path,
        A=per_A_arr,
        B=per_B_arr,
        C=per_C_arr,
        D=per_D_arr,
        difficulty=difficulty,
        c_norm_D=c_norm_D,
        A_text=np.array(per_A_text, dtype=object),
        B_text=np.array(per_B_text, dtype=object),
        D_text=np.array(per_D_text, dtype=object),
        references=np.array([p["reference"] for p in test_problems],
                            dtype=object),
    )
    print(f"[saved] {per_path}")


if __name__ == "__main__":
    main()