"""diagnose_gx.py — measure g(x) = -∂CE/∂||c|| as a function of ||c||.

Fixes the controller direction u at unit length, scales the magnitude
by x, and measures the derivative of weighted cross-entropy with
respect to x. This determines the shape of the CE landscape as a
function of intervention magnitude.

Predictions:
  - Monotonically decreasing g(x) → saturation, no bifurcation
  - U-shaped g(x) → S-shaped landscape, L1 can find equilibrium
  - Monotonically increasing g(x) → no equilibrium, architectural fix

For L2 regularization (reg = λ·x²), equilibrium requires g(x*) = 2λx*.
For L1 (reg = λ·x), equilibrium requires g(x*) = λ.
For monotonic g(x), the existence of an interior equilibrium depends
only on whether a horizontal line at λ intersects g(x).

Usage:
    python stage1/diagnose_gx.py --n_injections 1 --n_samples 20
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
X_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--n_injections", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--max_seq", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_batch(tok, problem, max_seq):
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
def measure_one(seg, tok, problem, x_value, max_seq, device):
    """Return g(x) = -∂CE/∂x for one problem, or None on failure."""
    full_ids, labels = make_batch(tok, problem, max_seq)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    label_ids = torch.tensor([labels], dtype=torch.long, device=device)

    x_t = torch.tensor(float(x_value), dtype=torch.float32,
                       device=device, requires_grad=True)
    seg.set_diag_x(x_t)
    seg.lambda_reg = 0.0

    try:
        out = seg(input_ids, attention_mask=attention_mask,
                  labels=label_ids)
        ce = out.get("ce_loss")
        if ce is None or not ce.requires_grad:
            return None
        grads = torch.autograd.grad(
            ce, [x_t], retain_graph=False, allow_unused=False)
        if grads[0] is None:
            return None
        return -float(grads[0].item())
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"    measure_one error at x={x_value}: "
              f"{type(e).__name__}: {e}")
        return None
    finally:
        seg.set_diag_x(None)


def main():
    args = parse_args()
    print(f"[gx] N={args.n_injections}  n_samples={args.n_samples}  "
          f"x_grid={X_GRID}")

    tok, base = load_model(args.model)
    base.config.use_cache = False
    device = base.device

    seg = SegmentedModel(base, n_injections=args.n_injections,
                          lambda_reg=0.0)
    seg.model.to(device)
    seg.controller.to(device)
    seg.w_in.to(device)

    ckpt = OUT_DIR / f"seg_N{args.n_injections}_{args.model}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    st = torch.load(ckpt, map_location=device)
    seg.model.load_state_dict(st["lora_state"], strict=False)
    if "controller_state" in st:
        seg.controller.load_state_dict(st["controller_state"])
        seg.w_in.load_state_dict(st["w_in_state"])
    print(f"[gx] loaded {ckpt.name}")

    seg.model.eval()
    seg.controller.eval()
    seg.w_in.eval()

    all_problems = load_gsm_hard()
    problems = all_problems[:args.n_samples]

    curve = {}
    for x in X_GRID:
        print(f"\n[gx] measuring at x={x}...")
        g_vals = []
        for k, p in enumerate(problems):
            g = measure_one(seg, tok, p, x, args.max_seq, device)
            if g is not None and np.isfinite(g):
                g_vals.append(g)
            if (k + 1) % 5 == 0:
                print(f"  [{k+1}/{len(problems)}]  "
                      f"collected {len(g_vals)} valid")

        if g_vals:
            curve[str(x)] = {
                "mean": float(np.mean(g_vals)),
                "std": float(np.std(g_vals)),
                "median": float(np.median(g_vals)),
                "n": len(g_vals),
            }
            print(f"  x={x:6.2f}  g_mean={curve[str(x)]['mean']:+.4e}  "
                  f"std={curve[str(x)]['std']:.2e}  n={len(g_vals)}")
        else:
            curve[str(x)] = {"mean": None, "n": 0}
            print(f"  x={x:6.2f}  no valid measurements")

    # ---------- Shape analysis ----------
    valid_x = [x for x in X_GRID if curve[str(x)]["n"] > 0]
    means = np.array([curve[str(x)]["mean"] for x in valid_x])
    stds = np.array([curve[str(x)]["std"] for x in valid_x])

    print(f"\n{'='*70}")
    print(f"g(x) SHAPE ANALYSIS")
    print(f"{'='*70}")
    print(f"{'x':>8}  {'g(x) mean':>14}  {'g(x) std':>12}  {'n':>4}")
    for x in valid_x:
        r = curve[str(x)]
        print(f"{x:>8.2f}  {r['mean']:>+14.4e}  {r['std']:>12.2e}  "
              f"{r['n']:>4}")

    if len(valid_x) < 3:
        verdict = "insufficient_data"
        msg = "Too few valid measurements to determine shape."
    else:
        diffs = np.diff(means)
        if np.all(diffs > 0):
            verdict = "monotonic_increasing"
            msg = ("g(x) increases with ||c||. No equilibrium for any "
                   "λ in the tested range. L1 will not help — g(x) "
                   "crosses λ from below and runs away.")
        elif np.all(diffs < 0):
            verdict = "monotonic_decreasing"
            msg = ("g(x) decreases with ||c||. Saturation behavior. "
                   "But observed runaway at λ=0.001 is inconsistent "
                   "with saturation at trained direction. Check if "
                   "direction u is problem-specific or constant.")
        else:
            min_idx = int(np.argmin(means))
            if 0 < min_idx < len(means) - 1:
                verdict = "U_shaped"
                msg = (f"Minimum at x≈{valid_x[min_idx]:.2f}, "
                       f"g_min={means[min_idx]:+.3e}. S-shaped landscape. "
                       f"L1 with λ_L1 ≈ {abs(means[min_idx]):.3e} will "
                       f"find a stable interior equilibrium at x > "
                       f"{valid_x[min_idx]:.2f}. "
                       f"L2 will not — its gradient 2λx grows with x "
                       f"and skips over the minimum.")
            else:
                verdict = "non_monotonic_edge"
                msg = "Non-monotonic but minimum at edge. Ambiguous."
    print(f"\n[verdict] {verdict}")
    print(f"  {msg}")

    # Recommendation for L1 lambda
    if verdict == "U_shaped":
        min_idx = int(np.argmin(means))
        lam_rec = abs(means[min_idx])
        print(f"\n[recommend] L1 λ ≈ {lam_rec:.3e}")
    elif verdict == "monotonic_decreasing":
        print(f"\n[recommend] Check direction u. If constant, "
              f"the landscape is not what it appears.")

    out = OUT_DIR / f"diagnose_gx_N{args.n_injections}_{args.model}.json"
    with open(out, "w") as f:
        json.dump({
            "n_injections": args.n_injections,
            "n_samples": len(problems),
            "x_grid": X_GRID,
            "curve": curve,
            "valid_x": valid_x,
            "means": means.tolist(),
            "stds": stds.tolist(),
            "verdict": verdict,
            "message": msg,
        }, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()