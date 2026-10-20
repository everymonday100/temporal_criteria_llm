"""generation.py — generation, residual norms, warmup/segment signals.

Supports both Qwen-style transformers (get_output_embeddings) and
Mamba2 (lm_head fallback). The LL_norm channel degrades to 0.0 only
if both are unavailable.

Notes on tensor handling:
    Hidden states stay on GPU. .clone() escapes inference-mode tagging
    so downstream matmul (LL_norm) works inside @torch.inference_mode().
    .cpu() is applied only where numpy conversion is needed
    (compute_lambda output).
"""
from collections import Counter

import numpy as np
import torch


SIGNAL_NAMES = [
    "P1_mean", "P1_std", "H_mean", "H_std", "H_slope",
    "Lambda_mean", "Lambda_std", "v_tau_std",
    "A_attn_mean", "LL_norm_mean", "LL_norm_std",
]


# ----------------------------------------------------------------------
# Core generation
# ----------------------------------------------------------------------
@torch.inference_mode()
def generate_segment(
    model, tok, full_prompt: str,
    n_tokens: int, T: float, seed: int,
    top_p: float = 0.95,
    return_hidden: bool = False,
    return_attn: bool = False,
    return_scores: bool = True,
):
    """Generate continuation after full_prompt."""
    inputs = tok(full_prompt, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    torch.manual_seed(seed)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=float(T),
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        output_hidden_states=return_hidden,
        output_attentions=return_attn,
        output_scores=return_scores,
        return_dict_in_generate=True,
    )
    out = model.generate(**gen_kwargs)
    gen_ids = out.sequences[0][prompt_len:]

    result = {
        "text": tok.decode(gen_ids, skip_special_tokens=True),
        "ids": gen_ids.tolist(),
        "prompt_len": prompt_len,
    }
    result["scores"] = (out.scores
                        if return_scores and getattr(out, "scores", None)
                        is not None else None)

    if return_hidden or return_attn:
        fwd = model(out.sequences,
                    output_hidden_states=True,
                    output_attentions=return_attn)
        if return_hidden and getattr(fwd, "hidden_states", None) is not None:
            hs = fwd.hidden_states
            # Stay on GPU. .clone() escapes inference-mode tag so LL_norm
            # matmul works inside @torch.inference_mode().
            result["hidden"] = tuple(
                h[0, prompt_len:, :].float().clone() for h in hs
            )
        else:
            result["hidden"] = None
        if return_attn and getattr(fwd, "attentions", None) is not None:
            at = fwd.attentions
            result["attentions"] = tuple(
                a[0, :, prompt_len:, :].float().clone() for a in at
            )
        else:
            result["attentions"] = None
        del fwd

    return result


def compute_lambda(hidden_states) -> np.ndarray:
    """Per-token sum over layers of ||h_{L+1} - h_L||.

    hidden_states is expected on GPU; .cpu() applied at the end.
    """
    n_layers = len(hidden_states) - 1
    diffs = torch.stack(
        [(hidden_states[l + 1] - hidden_states[l]).norm(dim=-1)
         for l in range(n_layers)],
        dim=0,
    )
    return diffs.sum(dim=0).cpu().numpy().astype(np.float64)


def to_tau(lam: np.ndarray, delta_tau: float = None):
    """Arc-length resampling.

    If delta_tau is None, choose adaptively so all models get ~30
    samples per token regardless of Lambda scale. Makes Qwen
    (Lambda ~1500) and Mamba2 (Lambda ~10000) comparable.
    """
    if delta_tau is None:
        if len(lam) == 0 or lam.mean() < 1e-6:
            return np.array([]), np.array([])
        delta_tau = float(lam.mean()) / 30.0
    tau = np.cumsum(lam)
    if tau[-1] < 2 * delta_tau:
        return np.array([]), np.array([])
    n = int(tau[-1] // delta_tau)
    tau_grid = (np.arange(1, n + 1) * delta_tau).astype(np.float64)
    v = np.interp(tau_grid, tau, lam)
    return tau_grid, v


# ----------------------------------------------------------------------
# Helpers: attention entropy, LL norm
# ----------------------------------------------------------------------
def _attn_entropy_mean(attentions, tail):
    if not attentions:
        return 0.0
    last = attentions[-1]
    gen_slice = last[:, -tail:, :].float()
    p = gen_slice.clamp_min(1e-10)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    ent = -(p * p.log()).sum(dim=-1)
    ent = torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)
    return float(ent.mean().item())


def _get_unembedding_weight(model):
    """Return the unembedding matrix (on GPU), or None if unavailable.

    Chain:
      1. get_output_embeddings() (Qwen, standard HF)
      2. lm_head.weight (Mamba2 HF checkpoints)
      3. tied input embeddings if tie_word_embeddings=True
    """
    if hasattr(model, "get_output_embeddings"):
        try:
            emb = model.get_output_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight
        except Exception:
            pass
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if (emb is not None and hasattr(emb, "weight") and
                    getattr(model.config, "tie_word_embeddings", False)):
                return emb.weight
        except Exception:
            pass
    return None


def _ll_norm_stats(hidden, tail, model):
    """LL_norm_mean, LL_norm_std. Matmul stays on GPU."""
    if hidden is None or model is None:
        return 0.0, 0.0
    try:
        W_U = _get_unembedding_weight(model)
        if W_U is None:
            return 0.0, 0.0
        h_last = hidden[-1][-tail:].float().clone()
        ll_logits = h_last @ W_U.float().T
        norms = ll_logits.norm(dim=-1)
        mean = float(norms.mean().item())
        std = float(norms.std().item())
        del ll_logits, norms
        return mean, std
    except Exception as e:
        print(f"[warn] LL norm failed: {type(e).__name__}: {e}")
        return 0.0, 0.0


# ----------------------------------------------------------------------
# Signals from a generated segment
# ----------------------------------------------------------------------
def signals_from_segment(out, model=None, need_attn=False):
    """Extract all 11 Stage-1 signals from a generate_segment() result."""
    G = len(out.get("ids", []))
    if G == 0:
        return {k: 0.0 for k in SIGNAL_NAMES}

    scores = out.get("scores")
    p1_vals, h_vals = [], []
    if scores is not None:
        for s in scores:
            probs = torch.softmax(s[0].float(), dim=-1)
            p1_vals.append(float(probs.max().item()))
            p = probs.clamp_min(1e-12)
            h_vals.append(float(-(p * p.log()).sum().item()))
    p1_vals = np.array(p1_vals) if p1_vals else np.array([0.0])
    h_vals = np.array(h_vals) if h_vals else np.array([0.0])
    x = np.arange(len(h_vals))
    h_slope = float(np.polyfit(x, h_vals, 1)[0]) if len(h_vals) > 1 else 0.0

    hidden = out.get("hidden")
    lam = compute_lambda(hidden) if hidden is not None else np.array([0.0])
    _, v = to_tau(lam)
    v_tau_std = float(v.std()) if len(v) > 1 else 0.0

    ll_mean, ll_std = _ll_norm_stats(hidden, len(lam), model)

    a_mean = 0.0
    if need_attn and out.get("attentions"):
        a_mean = _attn_entropy_mean(out["attentions"], len(lam))

    return {
        "P1_mean":      float(p1_vals.mean()),
        "P1_std":       float(p1_vals.std()),
        "H_mean":       float(h_vals.mean()),
        "H_std":        float(h_vals.std()),
        "H_slope":      float(h_slope),
        "Lambda_mean":  float(lam.mean()),
        "Lambda_std":   float(lam.std()),
        "v_tau_std":    v_tau_std,
        "A_attn_mean":  a_mean,
        "LL_norm_mean": ll_mean,
        "LL_norm_std":  ll_std,
    }


# ----------------------------------------------------------------------
# Warmup signals — used for L0 cache, AG, BD, CT
# ----------------------------------------------------------------------
@torch.inference_mode()
def warmup_signals(
    model, tok, prompt: str,
    warmup: int = 20,
    T: float = 0.2,
    seed: int = 42,
    need_attn: bool = False,
) -> dict:
    """Generate `warmup` tokens, extract all 11 signals."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=warmup,
        do_sample=True,
        temperature=float(T),
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
        output_scores=True,
        output_hidden_states=True,
        output_attentions=need_attn,
        return_dict_in_generate=True,
    )

    # ---- per-step output stats
    p1_vals, h_vals = [], []
    for s in out.scores[:warmup]:
        probs = torch.softmax(s[0].float(), dim=-1)
        p1_vals.append(float(probs.max().item()))
        p = probs.clamp_min(1e-12)
        h_vals.append(float(-(p * p.log()).sum().item()))
    p1_vals = np.array(p1_vals)
    h_vals = np.array(h_vals)
    x = np.arange(len(h_vals))
    h_slope = float(np.polyfit(x, h_vals, 1)[0]) if len(h_vals) > 1 else 0.0

    # ---- forward for hidden / attention
    fwd = model(out.sequences, output_hidden_states=True,
                output_attentions=need_attn)
    hs = fwd.hidden_states
    n_layers = len(hs) - 1
    diffs = torch.stack(
        [(hs[l + 1][0] - hs[l][0]).norm(dim=-1).float()
         for l in range(n_layers)],
        dim=0,
    )
    lam_gen = diffs.sum(dim=0).cpu().numpy().astype(np.float64)[prompt_len:]

    _, v = to_tau(lam_gen)
    v_tau_std = float(v.std()) if len(v) > 1 else 0.0

    # ---- attention entropy (only if need_attn)
    a_mean = 0.0
    if need_attn and getattr(fwd, "attentions", None) is not None \
            and len(fwd.attentions) > 0:
        last = fwd.attentions[-1][0]              # (n_heads, T, T)
        gen_slice = last[:, prompt_len:, :]
        if gen_slice.numel() > 0:
            a_mean = _attn_entropy_mean(
                (gen_slice.unsqueeze(0),), gen_slice.shape[1]
            )

    # ---- LL norm (GPU, .clone() to escape inference-mode tag)
    ll_mean, ll_std = 0.0, 0.0
    try:
        W_U = _get_unembedding_weight(model)
        if W_U is not None:
            h_last = hs[-1][0, prompt_len:, :].float().clone()
            ll_logits = h_last @ W_U.float().T
            ll_norms = ll_logits.norm(dim=-1)
            ll_mean = float(ll_norms.mean().item())
            ll_std = float(ll_norms.std().item())
            del ll_logits, ll_norms
    except Exception as e:
        print(f"[warn] LL norm (warmup) failed: "
              f"{type(e).__name__}: {e}")

    del fwd
    torch.cuda.empty_cache()

    return {
        "P1_mean":      float(p1_vals.mean()),
        "P1_std":       float(p1_vals.std()),
        "H_mean":       float(h_vals.mean()),
        "H_std":        float(h_vals.std()),
        "H_slope":      float(h_slope),
        "Lambda_mean":  float(lam_gen.mean()),
        "Lambda_std":   float(lam_gen.std()),
        "v_tau_std":    v_tau_std,
        "A_attn_mean":  a_mean,
        "LL_norm_mean": ll_mean,
        "LL_norm_std":  ll_std,
    }


# ----------------------------------------------------------------------
# GSM8K helpers
# ----------------------------------------------------------------------
import re
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _normalize_number(s):
    """Normalize a numeric string for comparison.

    Handles: '200.0' -> '200', '13919358.399999999' -> '13919358.4',
    '1,234' -> '1234', '0.5' -> '0.5'. Non-numeric strings pass through.
    """
    if s is None:
        return ""
    s = str(s).replace(",", "").strip()
    try:
        f = float(s)
    except (ValueError, OverflowError):
        return s
    # If the value is effectively an integer, return it as int string.
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    # Otherwise format to 6 decimal places (enough for any GSM-style
    # answer) and strip trailing zeros.
    return f"{f:.6f}".rstrip("0").rstrip(".")

def extract_answer(text: str) -> str:
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        nums = _NUM_RE.findall(m.group(1))
        if nums:
            return _normalize_number(nums[-1])
    nums = _NUM_RE.findall(text.replace(",", ""))
    return _normalize_number(nums[-1]) if nums else ""


def majority(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]