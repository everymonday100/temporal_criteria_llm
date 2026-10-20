"""run_l0.py — L0 baselines for all four properties.

Model-aware: writes {name}_L0{suffix}.npz where suffix is "" for qwen
and "_mamba2" for mamba2.

Outputs:
    data/stage1/cache/sm_L0[_mamba2].npz
    data/stage1/cache/ag_L0[_mamba2].npz
    data/stage1/cache/bd_L0[_mamba2].npz
    data/stage1/cache/ct_L0[_mamba2].npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import (
    generate_segment, compute_lambda, warmup_signals,
    extract_answer, majority,
)
from common.datasets import (
    load_sm_prompts, load_gsm8k, gsm8k_prompt,
    load_bd_tasks, bd_prompt, load_ct_tasks,
)
from common.io import ensure_dir

CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--which", default="all",
                   choices=["all", "sm", "ag", "bd", "ct"])
    p.add_argument("--model", type=str, default="qwen",
                   choices=["qwen", "mamba2"])
    p.add_argument("--n_sm", type=int, default=25)
    p.add_argument("--n_ag", type=int, default=400)
    p.add_argument("--n_bd", type=int, default=300)
    p.add_argument("--n_ct", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _sfx(args) -> str:
    return "" if args.model == "qwen" else f"_{args.model}"


# ----------------------------------------------------------------------
# SM: 25 prompts × 6 segments × 50 tokens
# ----------------------------------------------------------------------
def run_sm(model, tok, args):
    sfx = _sfx(args)
    path = CACHE_DIR / f"sm_L0{sfx}.npz"
    if path.exists() and not args.overwrite:
        print(f"[sm{sfx}] cache exists, skipping")
        return
    prompts = load_sm_prompts(args.n_sm)
    print(f"[sm{sfx}] {len(prompts)} prompts × 6 segments × 50 tokens")

    per_lambda, per_text = [], []
    for i, prompt in enumerate(prompts):
        seg_lams, seg_texts = [], []
        gen = ""
        for seg in range(6):
            out = generate_segment(model, tok, prompt + gen,
                                    n_tokens=50, T=0.55,
                                    seed=args.seed + i * 1000 + seg,
                                    return_hidden=True)
            seg_lams.append(compute_lambda(out["hidden"]))
            seg_texts.append(out["text"])
            gen += out["text"]
            del out["hidden"]
        per_lambda.append(np.concatenate(seg_lams))
        per_text.append(seg_texts)
        print(f"  [{i+1:2d}/{len(prompts)}] lam_mean={per_lambda[-1].mean():.0f}"
              f"  lam_std={per_lambda[-1].std():.0f}")

    ensure_dir(CACHE_DIR)
    np.savez(path,
             n_prompts=len(prompts),
             lambdas=np.array(per_lambda, dtype=object),
             texts=np.array(per_text, dtype=object),
             prompts=np.array(prompts, dtype=object))
    print(f"[sm{sfx}] saved {path}")


# ----------------------------------------------------------------------
# AG: 400 GSM8K problems, first + 3 retries + warmup signals
# ----------------------------------------------------------------------
def run_ag(model, tok, args):
    sfx = _sfx(args)
    path = CACHE_DIR / f"ag_L0{sfx}.npz"
    if path.exists() and not args.overwrite:
        print(f"[ag{sfx}] cache exists, skipping")
        return
    problems = load_gsm8k(args.n_ag)
    print(f"[ag{sfx}] {len(problems)} GSM8K problems")

    warm_sig, first_correct, multi_correct = [], [], []
    first_pred, multi_pred, refs = [], [], []

    for k, p in enumerate(problems):
        prompt = gsm8k_prompt(p["question"])
        s = warmup_signals(model, tok, prompt, warmup=20, T=0.2,
                           seed=args.seed + p["i"], need_attn=True)
        warm_sig.append(s)

        first = generate_segment(model, tok, prompt, n_tokens=290,
                                 T=0.2, seed=args.seed + p["i"] + 1,
                                 return_hidden=False, return_attn=False)
        first_ans = extract_answer(first["text"])
        first_ok = int(first_ans == p["reference"])

        retries = []
        for j in range(3):
            r = generate_segment(model, tok, prompt, n_tokens=290,
                                 T=0.6,
                                 seed=args.seed + p["i"] + 100 + j,
                                 return_hidden=False, return_attn=False)
            retries.append(extract_answer(r["text"]))
        maj = majority([first_ans] + retries)
        multi_ok = int(maj == p["reference"])

        first_correct.append(first_ok)
        multi_correct.append(multi_ok)
        first_pred.append(first_ans)
        multi_pred.append(maj)
        refs.append(p["reference"])

        print(f"  [{k+1:3d}/{len(problems)}] "
              f"P1={s['P1_mean']:.3f} A={s['A_attn_mean']:.2f}  "
              f"first={'v' if first_ok else 'x'}  "
              f"multi={'v' if multi_ok else 'x'}")

    ensure_dir(CACHE_DIR)
    keys = list(warm_sig[0].keys())
    data = {f"sig_{k}": np.array([s[k] for s in warm_sig]) for k in keys}
    data.update({
        "first_correct": np.array(first_correct),
        "multi_correct": np.array(multi_correct),
        "first_pred": np.array(first_pred, dtype=object),
        "multi_pred": np.array(multi_pred, dtype=object),
        "reference": np.array(refs, dtype=object),
    })
    np.savez(path, **data)
    print(f"[ag{sfx}] saved {path}")


# ----------------------------------------------------------------------
# BD: synthetic episodes — warmup signals + observation y
# ----------------------------------------------------------------------
def run_bd(model, tok, args):
    sfx = _sfx(args)
    path = CACHE_DIR / f"bd_L0{sfx}.npz"
    if path.exists() and not args.overwrite:
        print(f"[bd{sfx}] cache exists, skipping")
        return
    tasks = load_bd_tasks(args.n_bd, seed=args.seed)
    print(f"[bd{sfx}] {len(tasks)} synthetic episodes")

    warm_sig, y_obs = [], []
    for k, t in enumerate(tasks):
        prompt = bd_prompt(t)
        s = warmup_signals(model, tok, prompt, warmup=20, T=0.2,
                           seed=args.seed + t["i"], need_attn=True)
        out = generate_segment(model, tok, prompt, n_tokens=96,
                               T=0.5, seed=args.seed + t["i"] + 1,
                               return_hidden=False, return_attn=False)
        y = (float(np.mean(out["ids"][-20:]))
             if len(out["ids"]) >= 20 else 0.0)
        warm_sig.append(s)
        y_obs.append(y)
        print(f"  [{k+1:3d}/{len(tasks)}] y={y:.1f}  "
              f"P1={s['P1_mean']:.3f}  A={s['A_attn_mean']:.2f}")

    ensure_dir(CACHE_DIR)
    keys = list(warm_sig[0].keys())
    data = {f"sig_{k}": np.array([s[k] for s in warm_sig]) for k in keys}
    data["y_obs"] = np.array(y_obs)
    np.savez(path, **data)
    print(f"[bd{sfx}] saved {path}")


# ----------------------------------------------------------------------
# CT: 50 trajectories × 3 episodes
# ----------------------------------------------------------------------
def run_ct(model, tok, args):
    sfx = _sfx(args)
    path = CACHE_DIR / f"ct_L0{sfx}.npz"
    if path.exists() and not args.overwrite:
        print(f"[ct{sfx}] cache exists, skipping")
        return
    traj = load_ct_tasks(args.n_ct, 3, seed=args.seed)
    print(f"[ct{sfx}] {len(traj)} trajectories × 3 episodes")

    warm_by_traj = []
    first_correct = np.zeros((len(traj), 3), dtype=int)
    refs = np.empty((len(traj), 3), dtype=object)

    for k, tr in enumerate(traj):
        sigs = []
        for ep_idx, p in enumerate(tr["episodes"]):
            prompt = gsm8k_prompt(p["question"])
            s = warmup_signals(model, tok, prompt, warmup=20, T=0.2,
                               seed=args.seed + k * 100 + ep_idx,
                               need_attn=True)
            out = generate_segment(model, tok, prompt, n_tokens=290,
                                    T=0.2,
                                    seed=args.seed + k * 100 + ep_idx + 1,
                                    return_hidden=False, return_attn=False)
            first_correct[k, ep_idx] = int(
                extract_answer(out["text"]) == p["reference"])
            refs[k, ep_idx] = p["reference"]
            sigs.append(s)
        warm_by_traj.append(sigs)
        print(f"  [{k+1:3d}/{len(traj)}] topic={tr['topic']}")

    ensure_dir(CACHE_DIR)
    keys = list(warm_by_traj[0][0].keys())
    data = {}
    for key in keys:
        arr = np.zeros((len(traj), 3), dtype=float)
        for kk in range(len(traj)):
            for ep in range(3):
                arr[kk, ep] = warm_by_traj[kk][ep][key]
        data[f"sig_{key}"] = arr
    data["first_correct"] = first_correct
    data["reference"] = refs
    data["topics"] = np.array([tr["topic"] for tr in traj], dtype=object)
    np.savez(path, **data)
    print(f"[ct{sfx}] saved {path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dir(CACHE_DIR)
    tok, model = load_model(args.model)
    t0 = time.time()
    try:
        if args.which in ("all", "sm"):
            print(f"\n=== SM baseline ({args.model}) ===")
            run_sm(model, tok, args)
        if args.which in ("all", "ag"):
            print(f"\n=== AG baseline ({args.model}) ===")
            run_ag(model, tok, args)
        if args.which in ("all", "bd"):
            print(f"\n=== BD baseline ({args.model}) ===")
            run_bd(model, tok, args)
        if args.which in ("all", "ct"):
            print(f"\n=== CT baseline ({args.model}) ===")
            run_ct(model, tok, args)
    finally:
        print(f"\n[total] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()