"""run_halt_gates.py — smoke, train, eval for halt gates on Qwen2.5-3B.

Modes:
  smoke : oracle depth analysis across temperatures. No training.
  train : train gates + adapters on GSM8K with soft halting.
  eval  : autoregressive generation with halt gating; compare against
          full-stack baseline; report accuracy delta and depth.
  all   : smoke -> train -> eval.

Data:
  GSM8K train split (first --n_train problems) for training.
  GSM8K test split (first --n_test problems) for eval.

Loss:
  CE(out_logits, labels) over response tokens
  + lambda_depth * mean(expected_depth)
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import extract_answer
from common.datasets import load_gsm8k, gsm8k_prompt
from common.halt_model import HaltModel, EXIT_LAYERS

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\halt")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="all",
                   choices=["smoke", "train", "eval", "all"])
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_depth", type=float, default=0.005)
    p.add_argument("--normalize_depth", action="store_true",
                   help="If set, normalize expected_depth to [0, 1] "
                        "before adding to loss.")
    p.add_argument("--max_seq", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_n", type=int, default=100)
    p.add_argument("--smoke_temps", type=float, nargs="+",
                   default=[0.2, 0.7])
    return p.parse_args()


# ----------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------
def build_example(tok, problem, max_seq):
    prompt = gsm8k_prompt(problem["question"])
    response = problem["answer"].strip()
    if not response.endswith("#### " + problem["reference"]):
        response = response + "\n#### " + problem["reference"]

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_text = prompt + response
    full_ids = tok(full_text, add_special_tokens=False,
                   truncation=True, max_length=max_seq)["input_ids"]

    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100
    return full_ids, labels


# ----------------------------------------------------------------------
# SMOKE
# ----------------------------------------------------------------------
@torch.inference_mode()
def smoke(args):
    print(f"[halt smoke] n={args.smoke_n}  exit_layers={EXIT_LAYERS}  "
          f"temps={args.smoke_temps}")
    tok, model = load_model(args.model)

    problems = load_gsm8k(args.smoke_n)

    results = []
    t0 = time.time()
    for k, p in enumerate(problems):
        prompt = gsm8k_prompt(p["question"])
        ref = p["reference"]

        per_temp_results = []
        for temp in args.smoke_temps:
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            plen = inputs["input_ids"].shape[1]

            torch.manual_seed(args.seed + p["i"] + int(temp * 1000))
            gen = model.generate(
                **inputs,
                max_new_tokens=290,
                do_sample=(temp > 0.0),
                temperature=max(temp, 1e-6),
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
            gen_ids = gen[0][plen:]
            pred = extract_answer(tok.decode(gen_ids, skip_special_tokens=True))
            correct = int(pred == ref)

            fwd = model(gen, output_hidden_states=True, use_cache=False)
            hs = fwd.hidden_states
            lm_head = model.lm_head

            agree = {}
            for L in EXIT_LAYERS:
                head_dtype = lm_head.weight.dtype
                h = hs[L][0, plen:-1, :].to(head_dtype)
                logits = lm_head(h).float()
                top1 = logits.argmax(dim=-1)
                target = gen[0, plen + 1:]
                n = min(top1.shape[0], target.shape[0])
                agree[L] = float((top1[:n] == target[:n]).float().mean().item()) \
                    if n > 0 else 0.0

            min_layer_95 = None
            for L in EXIT_LAYERS:
                if agree[L] >= 0.95:
                    min_layer_95 = L
                    break

            per_temp_results.append({
                "temp": temp, "correct": correct,
                "agree": agree, "min_layer_95": min_layer_95,
            })

        results.append({"i": p["i"], "per_temp": per_temp_results})
        if (k + 1) % 10 == 0 or (k + 1) == len(problems):
            print(f"  [{k+1:3d}/{len(problems)}]  ({time.time()-t0:.0f}s)")

    # Aggregate over correct trajectories
    correct_records = []
    for r in results:
        for tr in r["per_temp"]:
            if tr["correct"] == 1:
                correct_records.append(tr)

    n_total = len(results) * len(args.smoke_temps)
    print(f"\n[halt smoke] correct trajectories: "
          f"{len(correct_records)}/{n_total}")

    if not correct_records:
        print("[halt smoke] no correct trajectories; inconclusive.")
        return

    print(f"\n[halt smoke] agreement to next token, "
          f"mean over correct trajectories:")
    print(f"  {'layer':>5}  {'agree_mean':>12}  {'pct_>=0.95':>12}")
    agree_mean = {}
    pct_95 = {}
    for L in EXIT_LAYERS:
        vals = np.array([r["agree"][L] for r in correct_records])
        agree_mean[L] = float(vals.mean())
        pct_95[L] = float((vals >= 0.95).mean())
        print(f"  {L:>5}  {agree_mean[L]:>12.3f}  {pct_95[L]:>12.2%}")

    min_layers = [r["min_layer_95"] for r in correct_records
                  if r["min_layer_95"] is not None]
    hist = {}
    verdict = "inconclusive"
    if min_layers:
        print(f"\n[halt smoke] min layer where per-problem "
              f"agreement >= 0.95:")
        print(f"  n with min_layer: {len(min_layers)}/{len(correct_records)}")
        print(f"  mean: {np.mean(min_layers):.1f}  "
              f"median: {np.median(min_layers):.1f}  "
              f"min: {min(min_layers)}  max: {max(min_layers)}")
        for L in EXIT_LAYERS:
            hist[L] = int(sum(1 for m in min_layers if m == L))
        print(f"  distribution:")
        for L in EXIT_LAYERS:
            n = hist[L]
            bar = "#" * n
            print(f"    L={L:>3}: {n:>3}  {bar}")
        mean_min = float(np.mean(min_layers))
        verdict = "halt_viable" if mean_min <= 24 else "halt_low_room"
        print(f"\n[halt smoke] VERDICT: {verdict}")
    else:
        print("\n[halt smoke] no trajectory reached agreement >= 0.95 "
              "below the final exit; halt gates have no room.")
        verdict = "halt_low_room"

    out = OUT_DIR / f"halt_smoke_{args.model}_n{args.smoke_n}.json"
    with open(out, "w") as f:
        json.dump({
            "n": len(results),
            "n_trajectories": n_total,
            "n_correct_trajectories": len(correct_records),
            "agreement_mean": {str(L): agree_mean[L] for L in EXIT_LAYERS},
            "pct_agreement_95": {str(L): pct_95[L] for L in EXIT_LAYERS},
            "min_layer_distribution": {
                str(L): hist.get(L, 0) for L in EXIT_LAYERS
            },
            "verdict": verdict,
        }, f, indent=2)
    print(f"[saved] {out}")


# ----------------------------------------------------------------------
# TRAIN
# ----------------------------------------------------------------------
def train(args):
    print(f"[halt train] n_train={args.n_train}  epochs={args.epochs}  "
          f"lambda_depth={args.lambda_depth}  lr={args.lr}  "
          f"normalize_depth={args.normalize_depth}")
    tok, base = load_model(args.model)
    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    halt = HaltModel(base, normalize_depth=args.normalize_depth).to(base.device)
    halt.train()

    n_trainable = sum(p.numel() for p in halt.trainable_parameters())
    print(f"[halt train] trainable params: {n_trainable/1e6:.2f} M")

    problems = load_gsm8k(args.n_train)
    print(f"[halt train] {len(problems)} problems")

    optim = torch.optim.AdamW(halt.trainable_parameters(),
                              lr=args.lr, weight_decay=0.0)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = np.random.default_rng(args.seed + epoch).permutation(
            len(problems))
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_depth = 0.0
        n_items = 0
        optim.zero_grad()

        for idx in perm:
            p = problems[int(idx)]
            ids, labels = build_example(tok, p, args.max_seq)
            input_ids = torch.tensor([ids], device=base.device)
            label_ids = torch.tensor([labels], device=base.device)

            out = halt(input_ids)
            logits = out["logits"]
            exp_depth = out["expected_depth"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = label_ids[:, 1:].contiguous()
            mask = shift_labels != -100

            if mask.sum() == 0:
                continue

            ce = F.cross_entropy(shift_logits[mask], shift_labels[mask])
            depth_loss = exp_depth[:, :-1][mask].mean()
            loss = ce + args.lambda_depth * depth_loss

            (loss / args.grad_accum).backward()

            epoch_loss += float(loss.item())
            epoch_ce += float(ce.item())
            epoch_depth += float(depth_loss.item())
            n_items += 1
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    halt.trainable_parameters(), 1.0)
                optim.step()
                optim.zero_grad()

            if step % 20 == 0:
                print(f"  epoch {epoch}  step {step}  "
                      f"ce={epoch_ce/max(1,n_items):.3f}  "
                      f"depth={epoch_depth/max(1,n_items):.1f}  "
                      f"({time.time()-t0:.0f}s)")

        optim.step()
        optim.zero_grad()
        print(f"[halt train] epoch {epoch}: "
              f"ce={epoch_ce/max(1,n_items):.3f}  "
              f"depth={epoch_depth/max(1,n_items):.1f}")

    save_path = OUT_DIR / f"halt_{args.model}_n{args.n_train}.pt"
    state = {
        "gates": halt.gates.state_dict(),
        "adapters": halt.adapters.state_dict()
                    if halt.adapters is not None else None,
        "exit_layers": EXIT_LAYERS,
        "lambda_depth": args.lambda_depth,
        "normalize_depth": args.normalize_depth,
    }
    torch.save(state, save_path)
    print(f"[halt train] saved {save_path}")


# ----------------------------------------------------------------------
# EVAL — autoregressive generation with halt gating
# ----------------------------------------------------------------------
@torch.inference_mode()
def eval_halt(args):
    print(f"[halt eval] n_test={args.n_test}  mode=autoregressive")
    tok, base = load_model(args.model)

    halt = HaltModel(base, normalize_depth=False).to(base.device)
    save_path = OUT_DIR / f"halt_{args.model}_n{args.n_train}.pt"
    if not save_path.exists():
        raise FileNotFoundError(f"train first: {save_path}")
    state = torch.load(save_path, map_location=base.device)
    halt.gates.load_state_dict(state["gates"])
    if halt.adapters is not None and state["adapters"] is not None:
        halt.adapters.load_state_dict(state["adapters"])
    halt.eval()

    problems = load_gsm8k(args.n_test)
    device = base.device

    correct_base = 0
    correct_halt = 0
    depth_sum = 0.0
    n = 0
    t0 = time.time()

    for k, p in enumerate(problems):
        prompt = gsm8k_prompt(p["question"])
        ref = p["reference"]
        inputs = tok(prompt, return_tensors="pt").to(device)
        plen = inputs["input_ids"].shape[1]

        # ----- Baseline full-stack greedy -----
        torch.manual_seed(args.seed + p["i"])
        gen = base.generate(
            **inputs, max_new_tokens=290, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        pred_b = extract_answer(
            tok.decode(gen[0][plen:], skip_special_tokens=True))
        if pred_b == ref:
            correct_base += 1

        # ----- Halt model: manual autoregressive loop -----
        cur_ids = inputs["input_ids"]
        generated = []
        depth_used_sum = 0.0
        step_count = 0
        stop = False

        for step in range(290):
            out = halt(cur_ids)
            exp_depth = out["expected_depth"][0, -1].item()
            depth_used_sum += exp_depth
            step_count += 1

            last_logits = out["logits"][0, -1, :]
            next_id = int(last_logits.argmax().item())
            generated.append(next_id)

            if next_id == tok.eos_token_id:
                stop = True
                break

            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[next_id]], device=device)],
                dim=1)

            # Truncate to keep memory bounded: keep prompt + last window
            if cur_ids.shape[1] > args.max_seq:
                keep_tail = args.max_seq - plen
                cur_ids = torch.cat(
                    [cur_ids[:, :plen], cur_ids[:, -keep_tail:]],
                    dim=1)

        pred_h = extract_answer(
            tok.decode(torch.tensor(generated), skip_special_tokens=True))
        if pred_h == ref:
            correct_halt += 1

        mean_depth = depth_used_sum / max(1, step_count)
        depth_sum += mean_depth
        n += 1

        if (k + 1) % 5 == 0 or (k + 1) == len(problems):
            print(f"  [{k+1:3d}/{len(problems)}]  "
                  f"base={correct_base}  halt={correct_halt}  "
                  f"depth={depth_sum/n:.1f}  ({time.time()-t0:.0f}s)")

    print(f"\n[halt eval] full-stack accuracy: {correct_base/n:.3f}")
    print(f"[halt eval] halt-gated accuracy: {correct_halt/n:.3f}")
    print(f"[halt eval] mean expected depth: {depth_sum/n:.1f}")
    print(f"[halt eval] accuracy delta:     "
          f"{(correct_halt - correct_base)/n:+.3f}")
    print(f"[halt eval] compute savings:    "
          f"{(36 - depth_sum/n)/36:.1%}")

    out = OUT_DIR / f"halt_eval_{args.model}_n{args.n_test}.json"
    with open(out, "w") as f:
        json.dump({
            "n_test": n,
            "full_stack_accuracy": correct_base / n,
            "halt_gated_accuracy": correct_halt / n,
            "mean_expected_depth": depth_sum / n,
            "accuracy_delta": (correct_halt - correct_base) / n,
            "compute_savings": (36 - depth_sum / n) / 36,
        }, f, indent=2)
    print(f"[saved] {out}")


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    if args.mode in ("smoke", "all"):
        smoke(args)
    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("eval", "all"):
        eval_halt(args)


if __name__ == "__main__":
    main()