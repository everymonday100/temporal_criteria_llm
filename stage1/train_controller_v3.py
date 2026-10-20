"""train_controller_v3_contrastive.py — H2: Contrastive Loss on mixed regimes.

Patched:
  1. ctrl_grad_norm measured BEFORE optimizer.zero_grad()
  2. One-time grad_fn / requires_grad diagnostic at step 1
  3. ctrl_param_L1 tracked per epoch (verifies weights actually move)

Запуск:
  python stage1/train_controller_v3_contrastive.py --n_train 400 --epochs 3 --lambda_contrast 5.0 --margin 0.3
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
from datasets import load_dataset

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\controller")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--lr_ctrl", type=float, default=1e-3)
    p.add_argument("--lambda_reg", type=float, default=0.001)
    p.add_argument("--lambda_contrast", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--lambda_warmup", type=int, default=300)
    p.add_argument("--max_seq", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def current_lambda(step, base, warmup):
    if step < warmup:
        return 0.0
    return base


class MixedDataset(Dataset):
    """Пары (easy, hard) из GSM8K и GSM-HARD."""

    def __init__(self, tok, n_pairs, max_seq, offset=0):
        self.tok = tok
        self.max_seq = max_seq

        print(f"[Data] Loading GSM8K ({n_pairs}) and GSM-HARD ({n_pairs})...")
        gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
        gsm8k_raw = gsm8k_raw.select(range(offset, offset + n_pairs))
        gsm_hard_raw = load_gsm_hard(n=offset + n_pairs)

        self.pairs = []
        for i in range(n_pairs):
            q_easy = gsm8k_raw[i]["question"]
            a_easy = gsm8k_raw[i]["answer"].split("####")[-1].strip()
            resp_easy = f"#### {a_easy}"

            idx_hard = offset + i
            q_hard = gsm_hard_raw[idx_hard]["question"]
            resp_hard = (gsm_hard_raw[idx_hard]["code"].strip() +
                         "\n#### " + gsm_hard_raw[idx_hard]["reference"])

            self.pairs.append({
                "easy": {"q": q_easy, "r": resp_easy},
                "hard": {"q": q_hard, "r": resp_hard}
            })

    def _tokenize(self, prompt, response):
        full_text = prompt + response
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(
            full_text, add_special_tokens=False,
            truncation=True, max_length=self.max_seq
        )["input_ids"]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return (torch.tensor(full_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        p_easy = f"Question: {pair['easy']['q']}\nAnswer:"
        p_hard = f"Question: {pair['hard']['q']}\nAnswer:"
        ids_e, lab_e = self._tokenize(p_easy, pair["easy"]["r"])
        ids_h, lab_h = self._tokenize(p_hard, pair["hard"]["r"])
        return {"easy": (ids_e, lab_e), "hard": (ids_h, lab_h)}


def collate_mixed(batch, pad_token_id=0):
    return batch[0]


def extract_c_t(out):
    """Извлечь c_t из выхода модели с проверкой размерности."""
    if "c_t" not in out or out["c_t"] is None:
        return None
    c = out["c_t"]
    if isinstance(c, list):
        if len(c) == 0:
            return None
        c = c[-1]
    if c.dim() == 1:
        c = c.unsqueeze(0)
    return c


def compute_contrastive_loss(c_e, c_h, margin, lambda_contrast):
    """Hinge loss: penalize when cos(c_e, c_h) > -margin."""
    c_e_norm = F.normalize(c_e, p=2, dim=-1)
    c_h_norm = F.normalize(c_h, p=2, dim=-1)
    cos_sim = torch.mm(c_e_norm, c_h_norm.t()).squeeze()
    contrast_loss = F.relu(cos_sim + margin) * lambda_contrast
    return contrast_loss, cos_sim.item()


def ctrl_grad_norm(ctrl_model):
    """Норма градиентов контроллера (вызывать ДО zero_grad!)."""
    total_sq = 0.0
    n_with_grad = 0
    for p in ctrl_model.controller_parameters():
        if p.grad is not None:
            total_sq += p.grad.norm().item() ** 2
            n_with_grad += 1
    return total_sq ** 0.5, n_with_grad


def ctrl_param_L1(ctrl_model):
    """L1-норма параметров контроллера (для отслеживания изменений)."""
    total = 0.0
    for p in ctrl_model.controller_parameters():
        total += p.data.abs().sum().item()
    return total


def diagnose_c_t_graph(c_e, c_h, ctrl_model):
    """Однократная диагностика: есть ли градиентная связь у c_t."""
    print("\n[DIAG] --- c_t gradient graph check ---")
    if c_e is None:
        print("[DIAG] c_e is None — hook not triggered!")
        return
    print(f"[DIAG] c_e.shape={tuple(c_e.shape)}  "
          f"requires_grad={c_e.requires_grad}  "
          f"grad_fn={c_e.grad_fn}")
    if c_h is not None:
        print(f"[DIAG] c_h.shape={tuple(c_h.shape)}  "
              f"requires_grad={c_h.requires_grad}  "
              f"grad_fn={c_h.grad_fn}")

    if c_e.requires_grad and c_e.grad_fn is not None:
        # Попробовать вычислить градиент вручную
        test_loss = c_e.sum()
        grads = torch.autograd.grad(
            test_loss, ctrl_model.controller_parameters(),
            retain_graph=True, allow_unused=True)
        n_ok = sum(1 for g in grads if g is not None and g.abs().sum() > 0)
        n_total = len(grads)
        print(f"[DIAG] manual grad check: {n_ok}/{n_total} params "
              f"have non-zero gradient")
        if n_ok == 0:
            print("[DIAG] ⚠️  c_t is in graph but gradient does NOT "
                  "reach controller params!")
        else:
            print("[DIAG] ✅ Gradient flows from c_t to controller params.")
    else:
        print("[DIAG] ❌ c_t NOT in computation graph. "
              "Contrastive loss CANNOT train controller.")
    print("[DIAG] ----------------------------------\n")


def main():
    args = parse_args()
    print(f"[ctrl train] v3 (Contrastive) lambda_c={args.lambda_contrast} "
          f"margin={args.margin}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok, base = load_model(args.model)
    base.config.use_cache = False

    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(base.device)
    ctrl_model.controller.to(base.device)
    ctrl_model.w_in.to(base.device)

    n_ctrl = sum(p.numel() for p in ctrl_model.controller_parameters())
    n_lora = sum(p.numel() for p in ctrl_model.lora_parameters())
    print(f"[ctrl train] LoRA: {n_lora/1e6:.2f}M  "
          f"Controller: {n_ctrl/1e6:.2f}M")

    train_ds = MixedDataset(tok, args.n_train, args.max_seq, offset=0)
    val_ds = MixedDataset(tok, args.n_val, args.max_seq,
                          offset=args.n_train + 700)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=lambda b: collate_mixed(b, tok.pad_token_id or 0))
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: collate_mixed(b, tok.pad_token_id or 0))

    optimizer = torch.optim.AdamW([
        {"params": ctrl_model.lora_parameters(), "lr": args.lr_lora},
        {"params": ctrl_model.controller_parameters(), "lr": args.lr_ctrl},
    ], weight_decay=0.0)

    device = base.device
    step = 0
    t0 = time.time()
    history = []
    diagnosed = False
    param_L1_start = ctrl_param_L1(ctrl_model)
    print(f"[ctrl train] initial ctrl_param_L1 = {param_L1_start:.4f}")

    for epoch in range(args.epochs):
        ctrl_model.model.train()
        epoch_ce = 0.0
        epoch_contrast = 0.0
        epoch_cos = []
        epoch_ctrl_grad = []
        n_items = 0
        optimizer.zero_grad()

        for batch in train_loader:
            ids_e = batch["easy"][0].unsqueeze(0).to(device)
            lab_e = batch["easy"][1].unsqueeze(0).to(device)
            ids_h = batch["hard"][0].unsqueeze(0).to(device)
            lab_h = batch["hard"][1].unsqueeze(0).to(device)

            lam = current_lambda(step, args.lambda_reg, args.lambda_warmup)
            ctrl_model.set_lambda(lam)

            out_e = ctrl_model(ids_e, labels=lab_e, use_cache=False)
            out_h = ctrl_model(ids_h, labels=lab_h, use_cache=False)

            loss_ce = out_e["loss"] + out_h["loss"]

            # --- Contrastive Loss ---
            contrast_loss = torch.tensor(0.0, device=device)
            cos_sim_val = 0.0

            c_e = extract_c_t(out_e)
            c_h = extract_c_t(out_h)

            # Однократная диагностика на шаге 1
            if not diagnosed and step == 0:
                diagnose_c_t_graph(c_e, c_h, ctrl_model)
                diagnosed = True

            if c_e is not None and c_h is not None:
                contrast_loss, cos_sim_val = compute_contrastive_loss(
                    c_e, c_h, args.margin, args.lambda_contrast)
                epoch_cos.append(cos_sim_val)

            total_loss = (loss_ce + contrast_loss) / 2.0
            total_loss.backward()

            # --- ФИКС: измеряем градиент ДО обнуления ---
            cgn, n_g = ctrl_grad_norm(ctrl_model)
            epoch_ctrl_grad.append(cgn)

            epoch_ce += loss_ce.item() / 2.0
            if isinstance(contrast_loss, torch.Tensor):
                epoch_contrast += contrast_loss.item()
            n_items += 1
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    ctrl_model.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                avg_ce = epoch_ce / max(1, n_items)
                avg_contrast = epoch_contrast / max(1, n_items)
                avg_cos = np.mean(epoch_cos) if epoch_cos else 0.0
                avg_cgn = np.mean(epoch_ctrl_grad[-args.log_every:])
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {step:>4d}  "
                      f"ce={avg_ce:.3f}  "
                      f"contrast={avg_contrast:.4f}  "
                      f"cos(e,h)={avg_cos:.4f}  "
                      f"ctrl_grad={avg_cgn:.2e}  "
                      f"({elapsed:.0f}s)")

        # --- Конец эпохи ---
        avg_cos_epoch = np.mean(epoch_cos) if epoch_cos else float("nan")
        avg_cgn_epoch = np.mean(epoch_ctrl_grad) if epoch_ctrl_grad else 0.0
        param_L1_now = ctrl_param_L1(ctrl_model)
        param_delta = param_L1_now - param_L1_start

        print(f"\n[epoch {epoch}] SUMMARY:")
        print(f"  train_ce         = {epoch_ce / max(1, n_items):.4f}")
        print(f"  contrast_loss    = {epoch_contrast / max(1, n_items):.4f}")
        print(f"  cos(easy,hard)   = {avg_cos_epoch:.4f}")
        print(f"  ctrl_grad_mean   = {avg_cgn_epoch:.2e}")
        print(f"  ctrl_param_L1    = {param_L1_now:.4f}  "
              f"(Δ from start: {param_delta:+.4f})")

        # --- Валидация ---
        ctrl_model.model.eval()
        val_ce = 0.0
        val_cos_list = []
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ids_e = batch["easy"][0].unsqueeze(0).to(device)
                lab_e = batch["easy"][1].unsqueeze(0).to(device)
                ids_h = batch["hard"][0].unsqueeze(0).to(device)
                lab_h = batch["hard"][1].unsqueeze(0).to(device)

                ctrl_model.set_lambda(0.0)
                out_e = ctrl_model(ids_e, labels=lab_e, use_cache=False)
                out_h = ctrl_model(ids_h, labels=lab_h, use_cache=False)

                val_ce += (out_e["loss"].item() +
                           out_h["loss"].item()) / 2.0
                val_n += 1

                c_e = extract_c_t(out_e)
                c_h = extract_c_t(out_h)
                if c_e is not None and c_h is not None:
                    c_e_n = F.normalize(c_e, p=2, dim=-1)
                    c_h_n = F.normalize(c_h, p=2, dim=-1)
                    cv = torch.mm(c_e_n, c_h_n.t()).squeeze().item()
                    val_cos_list.append(cv)

        val_ce /= max(1, val_n)
        val_cos_mean = (np.mean(val_cos_list)
                        if val_cos_list else float("nan"))
        print(f"  val_ce           = {val_ce:.4f}")
        print(f"  val_cos(e,h)     = {val_cos_mean:.4f}  "
              f"(n={len(val_cos_list)})")

        history.append({
            "epoch": epoch,
            "train_ce": epoch_ce / max(1, n_items),
            "contrast_loss": epoch_contrast / max(1, n_items),
            "train_cos": avg_cos_epoch,
            "ctrl_grad_mean": avg_cgn_epoch,
            "ctrl_param_L1": param_L1_now,
            "ctrl_param_delta": param_delta,
            "val_ce": val_ce,
            "val_cos": val_cos_mean,
        })
        param_L1_start = param_L1_now

    # --- Сохранение ---
    save_path = OUT_DIR / f"ctrl_{args.model}_contrastive.pt"
    torch.save({
        "lora_state": {k: v for k, v in
                       ctrl_model.model.state_dict().items()
                       if "lora_" in k},
        "controller_state": ctrl_model.controller.state_dict(),
        "w_in_state": ctrl_model.w_in.state_dict(),
        "lambda_contrast": args.lambda_contrast,
        "margin": args.margin,
        "epochs": args.epochs,
    }, save_path)
    print(f"\n[saved] {save_path}")

    hist_path = OUT_DIR / f"ctrl_{args.model}_contrastive_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[saved] {hist_path}")

    # --- Финальный вердикт ---
    if history:
        final_cos = history[-1].get("train_cos", float("nan"))
        initial_cos = history[0].get("train_cos", float("nan"))
        final_grad = history[-1].get("ctrl_grad_mean", 0.0)
        print(f"\n{'='*60}")
        print(f"CONTRASTIVE RESULT")
        print(f"{'='*60}")
        print(f"  Initial cos(easy,hard): {initial_cos:.4f}")
        print(f"  Final cos(easy,hard):   {final_cos:.4f}")
        print(f"  Final ctrl_grad_mean:   {final_grad:.2e}")
        if final_grad < 1e-8:
            print(f"  ❌ GRADIENT DEAD: contrastive loss cannot train "
                  f"controller.")
            print(f"     Check controller_model.py hook for .detach() "
                  f"or torch.no_grad().")
        elif final_cos < 0.7:
            print(f"  ✅ SUCCESS: directions separated "
                  f"({initial_cos:.3f} → {final_cos:.3f})")
        elif final_cos < 0.9:
            print(f"  ⚠️  PARTIAL: some separation "
                  f"({initial_cos:.3f} → {final_cos:.3f})")
        else:
            print(f"  ❌ FAIL: directions still constant "
                  f"({initial_cos:.3f} → {final_cos:.3f})")
            print(f"     CE gradient dominates. Try lambda_contrast=10.0.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()