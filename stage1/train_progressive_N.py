"""train_progressive_n.py — обучение segmented controller для данного N.

Патчи относительно v2:
  - gsm-hard вместо gsm8k.
  - weighted CE + ce_raw logging.
  - L1 регуляризация (в SegmentedModel).
  - Real-time cosine_within diagnostic с rolling buffer.
  - Early stop при постоянном направлении (cosine > threshold).
  - set_problem_id перед forward для корректной агрегации.

Запуск:
  python stage1/train_progressive_n.py --n_injections 1 --n_train 600 \
      --n_val 100 --epochs 2 --warmup 30 --ramp 30 --log_every 50 \
      --lambda_reg 0.02 --c_max 3.0 \
      --diagnose_cosine --diagnose_every 50 --early_stop_cosine 0.95
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.datasets import load_gsm_hard, gsm_hard_prompt
from common.segmented_controller import SegmentedModel

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\progressive_n")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--n_injections", type=int, required=True,
                   choices=[1, 2, 5, 10])
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--lr_ctrl", type=float, default=1e-3)
    p.add_argument("--lambda_reg", type=float, default=0.02)
    p.add_argument("--c_max", type=float, default=3.0)
    p.add_argument("--max_seq", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--ramp", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    # Diagnostic flags
    p.add_argument("--diagnose_cosine", action="store_true",
                   help="Compute real-time cosine_within on a rolling "
                        "buffer of recent problems.")
    p.add_argument("--diagnose_every", type=int, default=50,
                   help="Log cosine diagnostic every N training steps.")
    p.add_argument("--early_stop_cosine", type=float, default=None,
                   help="Stop training if mean cosine_within exceeds "
                        "this threshold at any diagnostic step.")
    return p.parse_args()


class GSMHardDataset(Dataset):
    def __init__(self, tok, problems, max_seq):
        self.tok = tok
        self.problems = problems
        self.max_seq = max_seq

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        p = self.problems[idx]
        prompt = gsm_hard_prompt(p["question"])
        response = p.get("code", "").strip()
        if not response:
            response = "\n#### " + p["reference"]
        if not response.endswith("#### " + p["reference"]):
            response += "\n#### " + p["reference"]

        full_text = prompt + response
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(
            full_text, add_special_tokens=False,
            truncation=True, max_length=self.max_seq,
        )["input_ids"]

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "problem_id": p["i"],
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id,
                           dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    problem_ids = []
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1
        problem_ids.append(item["problem_id"])
    return input_ids, attention_mask, labels, problem_ids


def main():
    args = parse_args()
    N = args.n_injections
    print(f"[progN train] N={N}  model={args.model}  "
          f"n_train={args.n_train}  epochs={args.epochs}")
    print(f"[progN train] lambda_reg={args.lambda_reg}  "
          f"c_max={args.c_max}  warmup={args.warmup}  ramp={args.ramp}")

    tok, base = load_model(args.model)
    base.config.use_cache = False

    seg_model = SegmentedModel(base, n_injections=N, lambda_reg=0.0,
                                c_max=args.c_max)
    seg_model.model.to(base.device)
    seg_model.controller.to(base.device)
    seg_model.w_in.to(base.device)

    if args.diagnose_cosine:
        seg_model.enable_cosine_diagnostic(buffer_size=20)
        print(f"[progN train] cosine diagnostic enabled, "
              f"logging every {args.diagnose_every} steps")

    n_lora = sum(p.numel() for p in seg_model.lora_parameters())
    n_ctrl = sum(p.numel() for p in seg_model.controller_parameters())
    print(f"[progN train] LoRA: {n_lora/1e6:.2f}M  "
          f"Controller: {n_ctrl/1e6:.2f}M")

    all_problems = load_gsm_hard()
    train_problems = all_problems[:args.n_train]
    val_problems = all_problems[args.n_train:args.n_train + args.n_val]

    train_ds = GSMHardDataset(tok, train_problems, args.max_seq)
    val_ds = GSMHardDataset(tok, val_problems, args.max_seq)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id or 0))
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id or 0))

    optimizer = torch.optim.AdamW([
        {"params": seg_model.lora_parameters(), "lr": args.lr_lora},
        {"params": seg_model.controller_parameters(), "lr": args.lr_ctrl},
    ], weight_decay=0.0)

    device = base.device
    step = 0
    t0 = time.time()
    best_val_ce = float("inf")
    history = []
    cosine_history = []
    early_stopped = False

    for epoch in range(args.epochs):
        if early_stopped:
            break

        seg_model.model.train()
        epoch_ce = 0.0
        epoch_c_norm = 0.0
        n_items = 0
        optimizer.zero_grad()

        for input_ids, attn_mask, labels, pids in train_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            pid = int(pids[0])

            # Ramp
            if step < args.warmup:
                lam = 0.0
            elif step < args.warmup + args.ramp:
                lam = args.lambda_reg * (step - args.warmup) / args.ramp
            else:
                lam = args.lambda_reg
            seg_model.lambda_reg = lam

            seg_model.set_problem_id(pid)
            out = seg_model(input_ids, attention_mask=attn_mask,
                            labels=labels)
            loss = out["loss"] / args.grad_accum
            loss.backward()

            ce_val = float(out["ce_loss"].item())
            ce_raw_val = float(out["ce_raw"].item()) \
                if out["ce_raw"] is not None else 0.0
            if out.get("c_norms"):
                c_step = float(np.mean(out["c_norms"]))
            else:
                c_step = 0.0

            epoch_ce += ce_val
            epoch_c_norm += c_step
            n_items += 1
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    seg_model.lora_parameters() +
                    seg_model.controller_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                print(f"  N={N} epoch {epoch} step {step}  "
                      f"lam={lam:.5f}  "
                      f"ce_raw={ce_raw_val:.3f}  "
                      f"ce_w={ce_val:.3f}  "
                      f"||c||={c_step:.4f}  "
                      f"({time.time()-t0:.0f}s)")

            # Real-time cosine diagnostic
            if (args.diagnose_cosine and
                    step % args.diagnose_every == 0):
                cw = seg_model.compute_cosine_within()
                if cw:
                    mean_all = float(np.mean(
                        [v["mean"] for v in cw.values()]))
                    per_pos_str = "  ".join(
                        f"p{p}={cw[p]['mean']:.4f}"
                        for p in sorted(cw.keys()))
                    print(f"  [cosine] step={step}  "
                          f"cosine_within_mean={mean_all:.4f}  "
                          f"({per_pos_str})")

                    cosine_history.append({
                        "step": step,
                        "epoch": epoch,
                        "cosine_within_mean": mean_all,
                        "per_position": {
                            str(p): cw[p] for p in cw.keys()
                        },
                    })

                    if (args.early_stop_cosine is not None and
                            mean_all > args.early_stop_cosine):
                        print(f"  [early_stop] cosine_within "
                              f"{mean_all:.4f} > "
                              f"{args.early_stop_cosine:.4f} "
                              f"at step {step}.")
                        print(f"  Interpretation: constant direction "
                              f"under cap. Content-specificity not "
                              f"emerging.")
                        early_stopped = True
                        break

        if early_stopped:
            break

        # Validation
        seg_model.model.eval()
        val_ce = 0.0
        val_n = 0
        with torch.no_grad():
            for input_ids, attn_mask, labels, pids in val_loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                out = seg_model(input_ids, attention_mask=attn_mask,
                                labels=labels)
                val_ce += float(out["ce_raw"].item()) \
                    if out["ce_raw"] is not None else 0.0
                val_n += 1
        val_ce /= max(1, val_n)
        print(f"[progN train] N={N} epoch {epoch}: val_ce={val_ce:.3f}")

        history.append({
            "epoch": epoch,
            "train_ce": epoch_ce / max(1, n_items),
            "val_ce": val_ce,
            "mean_c_norm": epoch_c_norm / max(1, n_items),
        })

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            save_path = OUT_DIR / f"seg_N{N}_{args.model}_best.pt"
            torch.save({
                "lora_state": {k: v for k, v in
                               seg_model.model.state_dict().items()
                               if "lora_" in k},
                "controller_state": seg_model.controller.state_dict(),
                "w_in_state": seg_model.w_in.state_dict(),
                "n_injections": N,
                "epoch": epoch,
                "val_ce": val_ce,
                "c_max": args.c_max,
                "lambda_reg": args.lambda_reg,
                "early_stopped": early_stopped,
            }, save_path)
            print(f"  [saved] {save_path}")

    hist_path = OUT_DIR / f"seg_N{N}_{args.model}_history.json"
    with open(hist_path, "w") as f:
        json.dump({
            "history": history,
            "cosine_history": cosine_history,
            "early_stopped": early_stopped,
            "config": {
                "n_injections": N,
                "lambda_reg": args.lambda_reg,
                "c_max": args.c_max,
                "warmup": args.warmup,
                "ramp": args.ramp,
                "epochs_run": len(history),
            },
        }, f, indent=2)
    print(f"[progN train] N={N} done. best_val_ce={best_val_ce:.3f}  "
          f"early_stopped={early_stopped}")


if __name__ == "__main__":
    main()