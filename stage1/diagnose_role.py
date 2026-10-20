"""diagnose_role.py — does the controller have a role in the current loss?

Compares ||∇_ctrl CE|| at the trained state vs at a random-init state.
Both use the same trained LoRA. If the trained controller's gradient is
much larger, training has moved it to a region where CE cares about it
(the controller has a role). If comparable to random, the controller
is not sensitive to CE and no amount of λ tuning will help.

Usage:
    python common/diagnose_role.py --n_injections 1 --n_samples 20
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.datasets import load_gsm_hard, gsm_hard_prompt
from common.segmented_controller import SegmentedModel

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\progressive_n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--n_injections", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--max_seq", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["trained", "random", "both"],
                   default="both")
    return p.parse_args()


def build_batch(tok, problem, max_seq):
    prompt = gsm_hard_prompt(problem["question"])
    response = problem.get("code", "").strip()
    if not response:
        response = "\n#### " + problem["reference"]
    if not response.endswith("#### " + problem["reference"]):
        response += "\n#### " + problem["reference"]

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(prompt + response, add_special_tokens=False,
                   truncation=True, max_length=max_seq)["input_ids"]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100
    return full_ids, labels


@torch.enable_grad()
def grad_norm_wrt_ce(seg_model, ids, labels, device):
    """Compute ||∂CE_weighted / ∂θ_ctrl|| for one example."""
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    label_ids = torch.tensor([labels], dtype=torch.long, device=device)

    ctrl_params = list(seg_model.controller_parameters())
    for p in ctrl_params:
        if p.grad is not None:
            p.grad = None

    out = seg_model(input_ids, attention_mask=attention_mask,
                    labels=label_ids)
    ce = out["ce_loss"]  # weighted CE, same as training signal

    grads = torch.autograd.grad(ce, ctrl_params,
                                 retain_graph=False, create_graph=False,
                                 allow_unused=True)
    total_sq = 0.0
    for g in grads:
        if g is None:
            continue
        total_sq += float((g ** 2).sum().item())
    return float(np.sqrt(total_sq))


def run_diagnostic(args, seg, tok, problems, mode, seed):
    """Run grad_norm over problems with controller in given mode.

    seg is reused between calls — no new SegmentedModel, no VRAM doubling.
    """
    device = next(seg.model.parameters()).device

    # Load LoRA (idempotent — same weights both times)
    ckpt = OUT_DIR / f"seg_N{args.n_injections}_{args.model}_best.pt"
    if not ckpt.exists():
        print(f"[diagnose] WARN no checkpoint {ckpt}")
        return None, None
    st = torch.load(ckpt, map_location=device)
    seg.model.load_state_dict(st["lora_state"], strict=False)

    if mode == "trained":
        seg.controller.load_state_dict(st["controller_state"])
        seg.w_in.load_state_dict(st["w_in_state"])
        print(f"[diagnose] controller: trained")
    else:
        # Re-init controller in place
        torch.manual_seed(seed + 999)
        for p in seg.controller.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.02)
            else:
                torch.nn.init.zeros_(p)
        torch.nn.init.normal_(seg.w_in.weight, std=0.02)
        print(f"[diagnose] controller: random-init")

    seg.model.eval()
    seg.controller.eval()
    seg.w_in.eval()

    norms = []
    for k, p in enumerate(problems):
        ids, labels = build_batch(tok, p, args.max_seq)
        n = grad_norm_wrt_ce(seg, ids, labels, device)
        norms.append(n)
        print(f"  [{k+1:2d}/{len(problems)}]  grad_norm={n:.4e}")

    return np.array(norms), None


def main():
    args = parse_args()
    print(f"[diagnose] N={args.n_injections}  n_samples={args.n_samples}  "
          f"mode={args.mode}")

    tok, base = load_model(args.model)
    base.config.use_cache = False

    # Single SegmentedModel, reused across modes
    seg = SegmentedModel(base, n_injections=args.n_injections,
                          lambda_reg=0.0)
    seg.model.to(base.device)
    seg.controller.to(base.device)
    seg.w_in.to(base.device)

    all_problems = load_gsm_hard()
    problems = all_problems[:args.n_samples]

    results = {}

    if args.mode in ("trained", "both"):
        print(f"\n=== TRAINED CONTROLLER ===")
        n_tr, _ = run_diagnostic(args, seg, tok, problems, "trained",
                                  args.seed)
        torch.cuda.empty_cache()
        if n_tr is not None:
            results["trained"] = {
                "grad_norm_mean": float(n_tr.mean()),
                "grad_norm_std": float(n_tr.std()),
                "grad_norm_median": float(np.median(n_tr)),
            }
            print(f"[trained]  mean={n_tr.mean():.4e}  "
                  f"median={np.median(n_tr):.4e}  std={n_tr.std():.4e}")

    if args.mode in ("random", "both"):
        print(f"\n=== RANDOM CONTROLLER ===")
        n_rd, _ = run_diagnostic(args, seg, tok, problems, "random",
                                  args.seed)
        torch.cuda.empty_cache()
        if n_rd is not None:
            results["random"] = {
                "grad_norm_mean": float(n_rd.mean()),
                "grad_norm_std": float(n_rd.std()),
                "grad_norm_median": float(np.median(n_rd)),
            }
            print(f"[random]   mean={n_rd.mean():.4e}  "
                  f"median={np.median(n_rd):.4e}  std={n_rd.std():.4e}")

    if "trained" in results and "random" in results:
        ratio = results["trained"]["grad_norm_mean"] / (
            results["random"]["grad_norm_mean"] + 1e-12)
        print(f"\n[ratio]  trained / random = {ratio:.3f}")
        if ratio > 5.0:
            verdict = "STRONG_ROLE"
        elif ratio > 1.5:
            verdict = "WEAK_ROLE"
        elif ratio > 0.5:
            verdict = "NO_ROLE"
        else:
            verdict = "RANDOM_BETTER"
        print(f"[verdict] {verdict}")
        results["ratio"] = ratio
        results["verdict"] = verdict

    # Save
    out_path = OUT_DIR / f"diagnose_N{args.n_injections}_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()