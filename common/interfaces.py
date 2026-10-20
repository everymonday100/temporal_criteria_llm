"""interfaces.py — Stage-1 interface specifications and state-block builder.

Key design: segmented interfaces extract signals from the TAIL of the
previous segment (not from the full accumulated trajectory). This makes
generation cheap and matches the semantics of warmup_signals.
"""
from typing import Optional

from .generation import (
    generate_segment, signals_from_segment, SIGNAL_NAMES,
)


INTERFACE_SPECS = {
    "L0":  None,
    "L1a": ("Lambda_mean",),
    "L1b": ("H_mean",),
    "L1c": ("P1_mean",),
    "L1d": ("v_tau_std",),
    "L1e": ("A_attn_mean",),
    "L1f": ("LL_norm_mean",),
    "L2a": ("Lambda_mean", "H_mean", "P1_mean", "v_tau_std"),
    "L2b": ("Lambda_mean", "H_mean", "P1_mean",
            "v_tau_std", "A_attn_mean", "LL_norm_mean"),
}

ALL_INTERFACES = list(INTERFACE_SPECS.keys())
NONZERO_INTERFACES = [k for k in ALL_INTERFACES if k != "L0"]


def channels_of(iface: str) -> Optional[tuple]:
    return INTERFACE_SPECS.get(iface)


def needs_attention(iface: str) -> bool:
    ch = INTERFACE_SPECS.get(iface)
    return ch is not None and "A_attn_mean" in ch


# ----------------------------------------------------------------------
# State block
# ----------------------------------------------------------------------
def _fmt_value(v):
    try:
        f = float(v)
    except Exception:
        return "nan"
    if abs(f) >= 1000:
        return f"{f:.0f}"
    return f"{f:.2f}"


def build_state_block(
    iface: str,
    signals: dict,
    seg_idx: int = 0,
    n_segments: int = 1,
    include_seg_label: bool = False,
) -> str:
    """Return [STATE] block. Empty string for L0."""
    ch = INTERFACE_SPECS.get(iface)
    if ch is None:
        return ""
    lines = ["[STATE]"]
    if include_seg_label:
        lines.append(f"seg {seg_idx + 1}/{n_segments}")
    for name in ch:
        lines.append(f"{name} {_fmt_value(signals.get(name, 0.0))}")
    lines.append("[/STATE]")
    return "\n" + "\n".join(lines) + "\n\n"


# ----------------------------------------------------------------------
# Segmented application (SM, CT)
# ----------------------------------------------------------------------
def apply_interface_segmented(
    model, tok, prompt: str, iface: str,
    n_segments: int = 6,
    tokens_per_seg: int = 50,
    T: float = 0.55,
    seed: int = 42,
) -> dict:
    """Segment loop: state block built from previous segment's signals.

    Returns:
        iface
        per_seg_lambda   — list of per-segment Λ arrays (as lists)
        per_seg_text     — list of per-segment decoded texts
        per_seg_signals  — list of signal dicts (signals from that segment)
        full_lambda      — concatenated Λ trajectory over all segments
    """
    ch = INTERFACE_SPECS.get(iface)
    need_attn = needs_attention(iface)

    per_seg_lambda = []
    per_seg_text = []
    per_seg_signals = []

    gen_so_far = ""
    prev_signals = None

    for seg in range(n_segments):
        if ch is None:
            state_block = ""
        elif seg == 0:
            state_block = build_state_block(
                iface, {c: 0.0 for c in ch},
                seg_idx=0, n_segments=n_segments,
                include_seg_label=True,
            )
        else:
            state_block = build_state_block(
                iface, prev_signals,
                seg_idx=seg, n_segments=n_segments,
                include_seg_label=True,
            )

        full_prompt = prompt + gen_so_far + state_block
        out = generate_segment(
            model, tok, full_prompt,
            n_tokens=tokens_per_seg, T=T,
            seed=seed + seg * 17,
            return_hidden=True,
            return_attn=need_attn,
            return_scores=True,
        )

        from .generation import compute_lambda
        lam = compute_lambda(out["hidden"])
        per_seg_lambda.append(lam.tolist())
        per_seg_text.append(out["text"])

        prev_signals = signals_from_segment(out, model=model,
                                             need_attn=need_attn)
        per_seg_signals.append(prev_signals)

        # cleanup
        out["hidden"] = None
        out["attentions"] = None
        out["scores"] = None

        gen_so_far = gen_so_far + state_block + out["text"]

    import numpy as np
    full_lam = np.concatenate([np.array(x) for x in per_seg_lambda])

    return {
        "iface": iface,
        "per_seg_lambda": per_seg_lambda,
        "per_seg_text": per_seg_text,
        "per_seg_signals": per_seg_signals,
        "full_lambda": full_lam.tolist(),
    }