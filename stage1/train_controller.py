"""train_controller.py — Training loop for latent controller (v3, gsm-hard).

Dataset: reasoning-machines/gsm-hard (1319 problems, larger numbers).
Qwen2.5-3B accuracy ~30-50%, CE 1.5-3.0, giving a strong training signal.

Patches vs v2:
  - Dataset swapped from GSM8K (contaminated, CE 0.3-0.6) to GSM-HARD
    (held-out numbers, CE 1.5-3.0).
  - Controller init fixed (non-zero on both ends of bottleneck).
  - Lambda warmup: first 300 steps with lambda=0, then ramp.
  - Per-step ||c|| logging.

Usage:
  python stage1/train_controller.py --n_train 600 --n_val 100 --epochs 3 \
      --lr_lora 5e-5 --lr_ctrl 1e-3 --lambda_reg 0.001
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
from common.controller_model import ControllerModel

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\controller")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--lr_ctrl", type=float, default=1e-3)
    p.add_argument("--lambda_reg", type=float, default=0.001)
    p.add_argument("--lambda_warmup", type=int, default=300)
    p.add_argument("--lambda_ramp", type=int, default=200)
    p.add_argument("--max_seq", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def current_lambda(step, base, warmup, ramp):
    if step < warmup:
        return 0.0
    if step < warmup + ramp:
        return base * float(step - warmup) / float(ramp)
    return base


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
        # Response = Python code + final answer line.
        # The `code` field contains the PAL-style solution.
        response = p["code"].strip()
        if not response.endswith("#### " + p["reference"]):
            response += "\n#### " + p["reference"]

        full_text = prompt + response
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(
            full_text, add_special_tokens=False,
            truncation=True, max_length=self.max_seq
        )["input_ids"]

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id,
                           dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1

    return input_ids, attention_mask, labels


def main():
    args = parse_args()
    print(f"[ctrl train] dataset=gsm-hard model={args.model} "
          f"n_train={args.n_train} epochs={args.epochs}")
    print(f"[ctrl train] lr_lora={args.lr_lora} lr_ctrl={args.lr_ctrl} "
          f"lambda={args.lambda_reg} warmup={args.lambda_warmup} "
          f"ramp={args.lambda_ramp}")

    tok, base = load_model(args.model)
    base.config.use_cache = False

    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(base.device)
    ctrl_model.controller.to(base.device)
    ctrl_model.w_in.to(base.device)

    n_trainable = sum(p.numel() for p in ctrl_model.trainable_parameters())
    n_lora = sum(p.numel() for p in ctrl_model.lora_parameters())
    n_ctrl = sum(p.numel() for p in ctrl_model.controller_parameters())
    print(f"[ctrl train] trainable: {n_trainable/1e6:.2f}M "
          f"(LoRA: {n_lora/1e6:.2f}M, Controller: {n_ctrl/1e6:.2f}M)")

    # Load gsm-hard, split train/val
    total = args.n_train + args.n_val
    all_problems = load_gsm_hard(n=total)
    train_problems = all_problems[:args.n_train]
    val_problems = all_problems[args.n_train:total]
    print(f"[ctrl train] loaded {len(all_problems)} problems "
          f"(train={len(train_problems)} val={len(val_problems)})")

    train_ds = GSMHardDataset(tok, train_problems, args.max_seq)
    val_ds = GSMHardDataset(tok, val_problems, args.max_seq)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id or 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id or 0),
    )

    optimizer = torch.optim.AdamW([
        {"params": ctrl_model.lora_parameters(), "lr": args.lr_lora},
        {"params": ctrl_model.controller_parameters(), "lr": args.lr_ctrl},
    ], weight_decay=0.0)

    device = base.device
    step = 0
    t0 = time.time()
    best_val_ce = float("inf")
    history = []

    for epoch in range(args.epochs):
        ctrl_model.model.train()
        epoch_ce = 0.0
        epoch_reg = 0.0
        epoch_c = 0.0
        n_items = 0
        optimizer.zero_grad()

        for input_ids, attn_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            lam = current_lambda(step, args.lambda_reg,
                                 args.lambda_warmup, args.lambda_ramp)
            ctrl_model.set_lambda(lam)

            out = ctrl_model(input_ids, attention_mask=attn_mask,
                             labels=labels, use_cache=False)
            loss = out["loss"] / args.grad_accum
            loss.backward()

            ce_val = float(out["ce_loss"].item()) if out["ce_loss"] is not None else 0.0
            reg_val = float(out["reg_loss"].item()) if out["reg_loss"] is not None else 0.0
            c_step = float(out["c_norms"].mean().item()) if out["c_norms"] is not None else 0.0

            epoch_ce += ce_val
            epoch_reg += reg_val
            epoch_c += c_step
            n_items += 1
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    ctrl_model.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                avg_ce = epoch_ce / max(1, n_items)
                avg_c = epoch_c / max(1, n_items)
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {step}  "
                      f"lam={lam:.5f}  "
                      f"ce(step)={ce_val:.3f}  "
                      f"ce(epoch)={avg_ce:.3f}  "
                      f"||c||(step)={c_step:.4f}  "
                      f"||c||(epoch)={avg_c:.4f}  "
                      f"({elapsed:.0f}s)")

                if step > args.lambda_warmup + args.lambda_ramp + 100:
                    if c_step < 0.005:
                        print(f"  [WARN] ||c||(step)={c_step:.5f} < 0.005 "
                              f"after warmup. Reduce lambda_reg or "
                              f"increase lr_ctrl.")

        # Validation
        ctrl_model.model.eval()
        val_ce = 0.0
        val_n = 0
        with torch.no_grad():
            for input_ids, attn_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                ctrl_model.set_lambda(0.0)
                out = ctrl_model(input_ids, attention_mask=attn_mask,
                                 labels=labels, use_cache=False)
                if out["ce_loss"] is not None:
                    val_ce += float(out["ce_loss"].item())
                    val_n += 1

        val_ce /= max(1, val_n)
        print(f"[ctrl train] epoch {epoch}: val_ce={val_ce:.3f}")

        history.append({
            "epoch": epoch,
            "train_ce": epoch_ce / max(1, n_items),
            "val_ce": val_ce,
            "mean_c": epoch_c / max(1, n_items),
        })

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            save_path = OUT_DIR / f"ctrl_{args.model}_gsmhard_best.pt"
            torch.save({
                "lora_state": {k: v for k, v in
                               ctrl_model.model.state_dict().items()
                               if "lora_" in k},
                "controller_state": ctrl_model.controller.state_dict(),
                "w_in_state": ctrl_model.w_in.state_dict(),
                "epoch": epoch,
                "val_ce": val_ce,
                "lambda_reg": args.lambda_reg,
            }, save_path)
            print(f"  [saved best] {save_path}")

    hist_path = OUT_DIR / f"ctrl_{args.model}_gsmhard_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[ctrl train] done. best_val_ce={best_val_ce:.3f}")
    print(f"[saved] {hist_path}")


if __name__ == "__main__":
    main()