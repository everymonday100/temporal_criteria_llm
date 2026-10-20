"""check_direction_variance.py — diagnose whether the trained controller
produces problem-specific directions or a constant direction.

Reads the trained checkpoint, forwards N prompts through base+controller,
extracts the control vector c_t at the last prompt position, and computes:

  1. Pairwise cosine similarity between c_t vectors across problems
     (on the 64-dim c_t and on the 2048-dim injected vector W_in @ c_t).
  2. Cosine similarity split by difficulty (correct vs incorrect).
  3. PCA on c_t across problems: explained variance ratio of first
     few components. If PC1 > 0.9, the controller is essentially 1D
     (magnitude only). If PC1 < 0.5, the direction is problem-specific.

Verdict:
  mean_cos > 0.90  → direction constant, segmented injection won't help
  mean_cos < 0.70  → direction varies, segmented justified
  otherwise        → ambiguous, inspect PCA and cross-difficulty split

Usage:
  python common/check_direction_variance.py \
      --model qwen \
      --n 30 \
      --n_train_skip 700 \
      --layer 18
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
from common.generation import extract_answer
from common.controller_model import ControllerModel


OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\controller")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--n_train_skip", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def extract_control(model_wrapper, tok, prompt, device):
    """Forward prompt, return (c_t vector, injected vector) at last pos.

    c_t is (64,) at last prompt position.
    injected = W_in @ c_t is (2048,).
    Returns None if hook didn't fire.
    """
    inputs = tok(prompt, return_tensors="pt").to(device)
    model_wrapper._last_c_norms = None
    # Save intermediate c_t via a small monkey-patch on the controller:
    # we run controller forward once more on the last position to get
    # the raw c_t vector (the hook only saves norms, not the vectors).
    _ = model_wrapper(inputs["input_ids"],
                      attention_mask=inputs["attention_mask"],
                      use_cache=False)

    # Recompute c_t explicitly at the last prompt position
    # We need the hidden state at layer 18 at the last prompt position.
    # The controller_model already has access to base.model.model.layers.
    from common.controller_model import CONTROLLER_LAYER

    # Forward base with hidden states to get h at controller layer
    base = model_wrapper.model.base_model.model
    h_layer = base.model.layers[CONTROLLER_LAYER]

    # Capture h at CONTROLLER_LAYER by running forward manually
    captured = {}

    def capture_hook(module, inp, out):
        # out is the hidden state entering the next layer
        if isinstance(out, tuple):
            captured["h"] = out[0].detach()
        else:
            captured["h"] = out.detach()

    handle = h_layer.register_forward_hook(capture_hook)
    try:
        _ = model_wrapper.model.base_model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
    finally:
        handle.remove()

    if "h" not in captured:
        return None

    h = captured["h"][0, -1, :].float()  # (2048,) at last position
    c_t = model_wrapper.controller(h.unsqueeze(0)).squeeze(0)  # (64,)
    c_t = c_t.detach().float().cpu().numpy()

    w_in = model_wrapper.w_in.weight.detach().float().cpu()  # (2048, 64)
    injected = (w_in @ torch.from_numpy(c_t)).numpy()  # (2048,)

    return c_t, injected


def cosine(a, b):
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosines(vecs):
    n = len(vecs)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(cosine(vecs[i], vecs[j]))
    return np.array(out)


def main():
    args = parse_args()
    print(f"[dir-var] model={args.model} n={args.n} "
          f"n_train_skip={args.n_train_skip}")

    tok, base = load_model(args.model)
    base.config.use_cache = False
    device = base.device

    ctrl = ControllerModel(base, lambda_reg=0.0)
    ctrl.model.to(device)
    ctrl.controller.to(device)
    ctrl.w_in.to(device)

    ckpt_path = OUT_DIR / f"ctrl_{args.model}_gsmhard_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Train first: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    ctrl.model.load_state_dict(state["lora_state"], strict=False)
    ctrl.controller.load_state_dict(state["controller_state"])
    ctrl.w_in.load_state_dict(state["w_in_state"])
    ctrl.model.eval()
    ctrl.controller.eval()
    ctrl.w_in.eval()

    all_problems = load_gsm_hard()
    problems = all_problems[args.n_train_skip:
                            args.n_train_skip + args.n]

    c_vecs = []
    inj_vecs = []
    norms = []
    correctness = []

    # We need per-problem correctness to split by difficulty. Use a
    # frozen run to label each problem as easy/hard.
    for k, p in enumerate(problems):
        prompt = gsm_hard_prompt(p["question"])

        # Frozen correctness
        inputs = tok(prompt, return_tensors="pt").to(device)
        plen = inputs["input_ids"].shape[1]
        torch.manual_seed(args.seed + p["i"])
        gen = base.generate(
            **inputs, max_new_tokens=290, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(gen[0][plen:], skip_special_tokens=True)
        pred = extract_answer(text)
        correct = int(pred == p["reference"])
        correctness.append(correct)

        # Control vector at last prompt position
        res = extract_control(ctrl, tok, prompt, device)
        if res is None:
            print(f"  [{k+1:2d}/{args.n}]  hook did not fire — skip")
            continue
        c_t, injected = res
        c_vecs.append(c_t)
        inj_vecs.append(injected)
        norms.append(float(np.linalg.norm(c_t)))

        print(f"  [{k+1:2d}/{args.n}]  correct={correct}  "
              f"||c_t||={norms[-1]:.4f}  "
              f"||injected||={np.linalg.norm(injected):.4f}")

    n_used = len(c_vecs)
    if n_used < 5:
        print("[dir-var] too few problems; aborting.")
        return

    c_vecs = np.array(c_vecs)
    inj_vecs = np.array(inj_vecs)
    norms = np.array(norms)
    correctness = np.array(correctness)

    # ---------- Pairwise cosines on c_t (64-dim) ----------
    cos_c = pairwise_cosines(c_vecs)
    cos_inj = pairwise_cosines(inj_vecs)

    print(f"\n[dir-var] pairwise cosine on c_t (64-dim, n={n_used}):")
    print(f"  mean   = {cos_c.mean():+.4f}")
    print(f"  median = {np.median(cos_c):+.4f}")
    print(f"  std    = {cos_c.std():.4f}")
    print(f"  p10    = {np.percentile(cos_c, 10):+.4f}")
    print(f"  p90    = {np.percentile(cos_c, 90):+.4f}")

    print(f"\n[dir-var] pairwise cosine on injected (2048-dim):")
    print(f"  mean   = {cos_inj.mean():+.4f}")
    print(f"  median = {np.median(cos_inj):+.4f}")
    print(f"  std    = {cos_inj.std():.4f}")

    # ---------- Split by difficulty ----------
    easy_idx = np.where(correctness == 1)[0]
    hard_idx = np.where(correctness == 0)[0]

    def split_cosines(vecs, idx_a, idx_b):
        out = []
        for i in idx_a:
            for j in idx_b:
                out.append(cosine(vecs[i], vecs[j]))
        return np.array(out) if out else np.array([np.nan])

    within_easy = split_cosines(c_vecs, easy_idx, easy_idx)
    within_hard = split_cosines(c_vecs, hard_idx, hard_idx)
    cross = split_cosines(c_vecs, easy_idx, hard_idx)

    print(f"\n[dir-var] cosine within/between difficulty groups:")
    print(f"  within easy  (n={len(easy_idx)}):  "
          f"{np.nanmean(within_easy):+.4f}")
    print(f"  within hard  (n={len(hard_idx)}):  "
          f"{np.nanmean(within_hard):+.4f}")
    print(f"  cross easy↔hard:               "
          f"{np.nanmean(cross):+.4f}")

    # ---------- PCA on c_t ----------
    # Center vectors, do SVD
    c_centered = c_vecs - c_vecs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(c_centered, full_matrices=False)
    expl_var = (S ** 2) / (S ** 2).sum()
    print(f"\n[dir-var] PCA on c_t:")
    print(f"  PC1 explains {expl_var[0]:.3f}")
    print(f"  PC2 explains {expl_var[1]:.3f}")
    print(f"  PC3 explains {expl_var[2]:.3f}")
    print(f"  PC1+PC2    = {expl_var[:2].sum():.3f}")
    print(f"  PC1..PC4   = {expl_var[:4].sum():.3f}")

    # ---------- Verdict ----------
    mean_cos = float(cos_c.mean())
    pc1 = float(expl_var[0])

    print(f"\n[verdict]")
    print(f"  mean pairwise cosine (c_t): {mean_cos:+.4f}")
    print(f"  PC1 explained variance:     {pc1:.4f}")

    # Use raw cosine similarity as primary signal. PCA on centered
    # vectors is dominated by tiny deviations once the mean direction
    # is subtracted, and can show low PC1 even when directions are
    # nearly identical.
    if mean_cos > 0.95:
        tag = "constant_direction"
    elif mean_cos > 0.90:
        tag = "mostly_constant"
    elif mean_cos > 0.70:
        tag = "partial_variation"
    else:
        tag = "varying_direction"
        msg = ("Controller emits nearly identical directions across "
               "problems. Segmented injection will not help — the "
               "controller has only one axis to modulate.")
    elif mean_cos < 0.70:
        tag = "varying_direction"
        msg = ("Directions vary substantially across problems. "
               "Segmented injection is justified: the controller has "
               "content to act on.")
    else:
        tag = "ambiguous"
        msg = ("Directions partially overlap. Check PCA and cross-"
               "difficulty split; the top PC may dominate but with "
               "meaningful residual variation.")

    print(f"  TAG: {tag}")
    print(f"  {msg}")

    # ---------- Save ----------
    out_path = OUT_DIR / f"dir_var_{args.model}_n{n_used}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_used": n_used,
            "cos_c_t": {
                "mean": float(cos_c.mean()),
                "median": float(np.median(cos_c)),
                "std": float(cos_c.std()),
                "p10": float(np.percentile(cos_c, 10)),
                "p90": float(np.percentile(cos_c, 90)),
            },
            "cos_injected": {
                "mean": float(cos_inj.mean()),
                "median": float(np.median(cos_inj)),
                "std": float(cos_inj.std()),
            },
            "by_difficulty": {
                "within_easy": float(np.nanmean(within_easy)),
                "within_hard": float(np.nanmean(within_hard)),
                "cross": float(np.nanmean(cross)),
                "n_easy": int(len(easy_idx)),
                "n_hard": int(len(hard_idx)),
            },
            "pca_explained_variance": expl_var[:8].tolist(),
            "verdict": {
                "mean_cos": mean_cos,
                "pc1": pc1,
                "tag": tag,
                "message": msg,
            },
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()