"""eval_progressive_n.py — оценка с раздельным анализом косинусов.

Патченная версия (v2):
  - Использует gsm-hard вместо gsm8k (patch 4)
  - Difficulty берётся из per-problem frozen-base correctness, не из D (patch 5)
  - cosine_between считает по task_id-pairing, не по индексу (patch 6)
  - Явная работа с device, убраны лишние .to(tok.device)

Для каждого N ∈ {1, 2, 5, 10}:
  1. Генерация на 300 тестовых задачах
  2. Запись векторов контроллера на каждой точке инъекции
  3. Раздельный анализ:
     - cosine_within(p): попарный косинус на одной позиции
       по разным задачам → контент-зависимость
     - cosine_between(p,q): попарный косинус на разных
       позициях одной задачи → позиционная специализация
  4. Accuracy для каждой конфигурации
"""
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import extract_answer
from common.datasets import load_gsm_hard, gsm_hard_prompt
from common.segmented_controller import SegmentedModel
from common.stats import paired_bootstrap

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\progressive_n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--n_injections", type=int, required=True,
                   choices=[1, 2, 5, 10])
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--n_train_skip", type=int, default=700)
    p.add_argument("--max_new", type=int, default=290)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap_B", type=int, default=2000)
    return p.parse_args()


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosines(vectors):
    n = len(vectors)
    if n < 2:
        return []
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            c = cosine(vectors[i], vectors[j])
            if c is not None:
                out.append(c)
    return out


@torch.inference_mode()
def generate_with_recording(seg_model, tok, prompt, max_new, seed):
    device = next(seg_model.model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Reset hook state for eval mode
    seg_model._in_training_forward = False
    seg_model._c_vectors = []
    seg_model._inject_positions = []
    seg_model._h_buffer = None
    seg_model._gen_step = 0
    seg_model._decode_positions = seg_model._compute_decode_positions(max_new)

    torch.manual_seed(seed)
    out = seg_model.model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)

    c_vecs = [c.detach().cpu().numpy() for c in seg_model._c_vectors]
    inject_pos = list(seg_model._inject_positions)

    return text, c_vecs, inject_pos


def main():
    args = parse_args()
    N = args.n_injections
    print(f"[progN eval] N={N}  n_test={args.n_test}  model={args.model}")

    tok, base = load_model(args.model)
    base.config.use_cache = False
    device = base.device

    # Загружаем обученную модель
    seg_model = SegmentedModel(base, n_injections=N, lambda_reg=0.0)
    seg_model.model.to(device)
    seg_model.controller.to(device)
    seg_model.w_in.to(device)

    ckpt_path = OUT_DIR / f"seg_N{N}_{args.model}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Train first: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    seg_model.model.load_state_dict(state["lora_state"], strict=False)
    seg_model.controller.load_state_dict(state["controller_state"])
    seg_model.w_in.load_state_dict(state["w_in_state"])
    seg_model.model.eval()

    all_problems = load_gsm_hard()
    test_problems = all_problems[
        args.n_train_skip:args.n_train_skip + args.n_test]
    print(f"[progN eval] {len(test_problems)} test problems "
          f"(from index {args.n_train_skip})")

    # ---- A: frozen base ----
    print(f"\n[eval] A: frozen base")
    per_A = []  # per-problem correctness
    for k, p in enumerate(test_problems):
        prompt = gsm_hard_prompt(p["question"])
        inputs = tok(prompt, return_tensors="pt").to(device)
        plen = inputs["input_ids"].shape[1]
        torch.manual_seed(args.seed + p["i"])
        out = base.generate(
            **inputs, max_new_tokens=args.max_new, do_sample=False,
            pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][plen:], skip_special_tokens=True)
        ok = int(extract_answer(text) == p["reference"])
        per_A.append(ok)
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(test_problems)}]  "
                  f"acc={sum(per_A)/len(per_A):.3f}")
    acc_A = sum(per_A) / len(per_A)
    print(f"  A = {acc_A:.3f}")

    # ---- B: LoRA-only (controller off) ----
    print(f"\n[eval] B: LoRA-only")
    original_w_in = seg_model.w_in.weight.data.clone()
    seg_model.w_in.weight.data.zero_()

    per_B = []
    for k, p in enumerate(test_problems):
        prompt = gsm_hard_prompt(p["question"])
        inputs = tok(prompt, return_tensors="pt").to(device)
        plen = inputs["input_ids"].shape[1]
        torch.manual_seed(args.seed + p["i"])
        out = seg_model.model.generate(
            **inputs, max_new_tokens=args.max_new,
            do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][plen:], skip_special_tokens=True)
        ok = int(extract_answer(text) == p["reference"])
        per_B.append(ok)
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(test_problems)}]  "
                  f"acc={sum(per_B)/len(per_B):.3f}")
    acc_B = sum(per_B) / len(per_B)
    print(f"  B = {acc_B:.3f}")

    # Restore
    seg_model.w_in.weight.data.copy_(original_w_in)

    # ---- D: trained segmented controller ----
    print(f"\n[eval] D: trained controller (N={N})")
    per_D = []
    # Записи: (task_i, position, vector)
    c_records = []
    per_problem_norms = []  # (task_i, mean_norm)

    for k, p in enumerate(test_problems):
        prompt = gsm_hard_prompt(p["question"])
        text, c_vecs, inject_pos = generate_with_recording(
            seg_model, tok, prompt, args.max_new, args.seed + p["i"])
        ok = int(extract_answer(text) == p["reference"])
        per_D.append(ok)

        for pos, cv in zip(inject_pos, c_vecs):
            c_records.append((p["i"], pos, cv))

        norms = [float(np.linalg.norm(cv)) for cv in c_vecs]
        per_problem_norms.append(
            (p["i"], float(np.mean(norms)) if norms else 0.0, ok))

        if (k + 1) % 50 == 0:
            n_positions = len(set(pos for _, pos, _ in c_records
                                  if _ == p["i"]))
            print(f"  [{k+1}/{len(test_problems)}]  "
                  f"acc={sum(per_D)/len(per_D):.3f}  "
                  f"last_n_positions={n_positions}")
    acc_D = sum(per_D) / len(per_D)
    print(f"  D = {acc_D:.3f}")

    # ---- Paired bootstrap ----
    print(f"\n[eval] Paired bootstrap (n={len(per_A)}):")
    d_db, lo_db, hi_db = paired_bootstrap(
        np.array(per_D, dtype=float),
        np.array(per_B, dtype=float),
        B=args.bootstrap_B, seed=args.seed)
    d_ba, lo_ba, hi_ba = paired_bootstrap(
        np.array(per_B, dtype=float),
        np.array(per_A, dtype=float),
        B=args.bootstrap_B, seed=args.seed)
    print(f"  Δ B−A = {d_ba:+.4f}  [{lo_ba:+.4f}, {hi_ba:+.4f}]")
    print(f"  Δ D−B = {d_db:+.4f}  [{lo_db:+.4f}, {hi_db:+.4f}]")

    # ---- Cosine analysis ----
    print(f"\n{'='*60}")
    print(f"COSINE ANALYSIS (N={N})")
    print(f"{'='*60}")

    # Группируем по позиции
    by_position = defaultdict(list)
    by_task = defaultdict(dict)
    for ti, pos, v in c_records:
        by_position[pos].append(v)
        by_task[ti][pos] = v

    sorted_positions = sorted(by_position.keys())

    # cosine_within(p): попарный косинус на одной позиции по задачам
    cosine_within = {}
    for pos in sorted_positions:
        cosines = pairwise_cosines(by_position[pos])
        if cosines:
            cosine_within[pos] = {
                "mean": float(np.mean(cosines)),
                "std": float(np.std(cosines)),
                "n_pairs": len(cosines),
                "n_vectors": len(by_position[pos]),
            }
            print(f"  within(pos={pos:>4}): "
                  f"mean={cosine_within[pos]['mean']:.4f}  "
                  f"std={cosine_within[pos]['std']:.4f}  "
                  f"n_vecs={cosine_within[pos]['n_vectors']}")

    # cosine_between(p, q): попарный косинус на разных позициях
    # одной и той же задачи (правильное pairing)
    cosine_between = {}
    if len(sorted_positions) >= 2:
        for idx in range(len(sorted_positions) - 1):
            p1 = sorted_positions[idx]
            p2 = sorted_positions[idx + 1]
            cosines = []
            for ti, pos_map in by_task.items():
                if p1 in pos_map and p2 in pos_map:
                    c = cosine(pos_map[p1], pos_map[p2])
                    if c is not None:
                        cosines.append(c)
            if cosines:
                key = f"{p1}-{p2}"
                cosine_between[key] = {
                    "mean": float(np.mean(cosines)),
                    "std": float(np.std(cosines)),
                    "n_pairs": len(cosines),
                }
                print(f"  between(pos={p1:>4}↔{p2:>4}): "
                      f"mean={cosine_between[key]['mean']:.4f}  "
                      f"std={cosine_between[key]['std']:.4f}  "
                      f"n_tasks={len(cosines)}")

    # ---- Diagnostics ----
    within_means = [v["mean"] for v in cosine_within.values()]
    mean_within = float(np.mean(within_means)) if within_means else 1.0

    between_means = [v["mean"] for v in cosine_between.values()]
    mean_between = float(np.mean(between_means)) if between_means else 1.0

    norms_arr = np.array([n for _, n, _ in per_problem_norms])
    cv_c = float(np.std(norms_arr) / (np.mean(norms_arr) + 1e-9)) \
        if len(norms_arr) else 0.0

    # difficulty from A (frozen), not from D
    difficulties = np.array([1 - c for c in per_A], dtype=float)
    if len(set(difficulties)) > 1 and np.std(norms_arr) > 1e-9:
        corr_diff, p_diff = pearsonr(norms_arr, difficulties)
    else:
        corr_diff, p_diff = 0.0, 1.0

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY (N={N})")
    print(f"{'='*60}")
    print(f"  A (frozen):     {acc_A:.3f}")
    print(f"  B (LoRA-only):  {acc_B:.3f}")
    print(f"  D (seg N={N}):   {acc_D:.3f}")
    print(f"  Δ(D−B):         {acc_D - acc_B:+.4f}  "
          f"[{lo_db:+.4f}, {hi_db:+.4f}]")
    print(f"  mean cosine_within:  {mean_within:.4f}")
    print(f"  mean cosine_between: {mean_between:.4f}")
    print(f"  CV of ||c||:         {cv_c:.4f}")
    print(f"  corr(||c||, diff):   {corr_diff:+.4f} (p={p_diff:.2e})")

    # Verdict
    if mean_within > 0.9:
        shape_tag = "PLATEAU"
    elif mean_within < 0.7:
        shape_tag = "CONTENT_SPECIFIC"
    else:
        shape_tag = "INTERMEDIATE"

    delta_D_B = acc_D - acc_B
    lo_db_excl_zero = lo_db > 0.0

    if mean_within < 0.7 and delta_D_B >= 0.02 and lo_db_excl_zero:
        core_verdict = "CORE_NOT_NEEDED"
    elif mean_within < 0.7 and (delta_D_B < 0.02 or not lo_db_excl_zero):
        core_verdict = "CORE_JUSTIFIED"
    elif mean_within > 0.9:
        core_verdict = "CORE_FUTILE"
    else:
        core_verdict = "INCONCLUSIVE"

    print(f"\n  SHAPE TAG:    {shape_tag}")
    print(f"  CORE VERDICT: {core_verdict}")

    # ---- Save ----
    result = {
        "model": args.model,
        "n_injections": N,
        "n_test": args.n_test,
        "acc_frozen": acc_A,
        "acc_lora": acc_B,
        "acc_segmented": acc_D,
        "delta_D_minus_B": delta_D_B,
        "bootstrap_D_minus_B": {
            "mean": d_db, "ci_low": lo_db, "ci_high": hi_db,
        },
        "bootstrap_B_minus_A": {
            "mean": d_ba, "ci_low": lo_ba, "ci_high": hi_ba,
        },
        "cosine_within": {str(k): v
                          for k, v in cosine_within.items()},
        "cosine_between": {str(k): v
                           for k, v in cosine_between.items()},
        "mean_cosine_within": mean_within,
        "mean_cosine_between": mean_between,
        "cv_c_norm": cv_c,
        "corr_c_difficulty": float(corr_diff),
        "corr_p_value": float(p_diff),
        "shape_tag": shape_tag,
        "core_verdict": core_verdict,
    }
    out_path = OUT_DIR / f"progN_eval_N{N}_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {out_path}")

    # Save raw vectors
    npz_payload = {
        f"pos_{pos}": np.array(vecs)
        for pos, vecs in by_position.items()
    }
    npz_payload["per_problem"] = np.array(
        [[n, int(ok)] for _, n, ok in per_problem_norms])
    npz_payload["per_A_correct"] = np.array(per_A, dtype=int)
    npz_payload["per_B_correct"] = np.array(per_B, dtype=int)
    npz_payload["per_D_correct"] = np.array(per_D, dtype=int)
    npz_path = OUT_DIR / f"progN_vectors_N{N}_{args.model}.npz"
    np.savez(npz_path, **npz_payload)
    print(f"[saved] {npz_path}")


if __name__ == "__main__":
    main()