"""convert_ag_cache.py — convert old agency_retry cache to run_ag format.

Old format (data/agency_retry_cache.npz):
    i, P1_mean_20, H_mean_20, H_std_20, H_slope_20,
    first_correct, multi_correct, first_pred, multi_pred, reference

New format (data/stage1/cache/ag_L0.npz):
    sig_P1_mean_20, sig_H_mean_20, sig_H_std_20, sig_H_slope_20,
    sig_Lambda_mean, sig_Lambda_std, sig_v_tau_std,
    sig_A_attn_mean, sig_LL_norm_mean, sig_LL_norm_std,
    sig_P1_std (all missing signals filled with NaN),
    first_correct, multi_correct, first_pred, multi_pred,
    reference, model, n_problems

Missing signals (Lambda_*, v_tau_std, A_attn_mean, LL_norm_*)
are filled with NaN; run_ag skips NaN in AUC computation, and
P1_mean_20 (the primary signal) is fully preserved.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = Path(r"E:\OllamaModels\temporal_llm\data\agency_retry_cache.npz")
DST = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache\ag_L0.npz")


def main():
    if not SRC.exists():
        print(f"[convert] source not found: {SRC}")
        print(f"[convert] run agency_retry.py first, or run:")
        print(f"  python stage1/run_ag.py --model qwen --mode gen")
        return

    src = np.load(SRC, allow_pickle=True)
    n = len(src["first_correct"])

    # Mapping: new key -> old key
    mapping = {
        "sig_P1_mean_20":  "P1_mean_20",
        "sig_H_mean_20":   "H_mean_20",
        "sig_H_std_20":    "H_std_20",
        "sig_H_slope_20":  "H_slope_20",
    }
    # Signals not present in old cache (fill with NaN)
    missing = [
        "sig_P1_std",
        "sig_Lambda_mean", "sig_Lambda_std",
        "sig_v_tau_std",
        "sig_A_attn_mean",
        "sig_LL_norm_mean", "sig_LL_norm_std",
    ]

    out = {}
    for new_key, old_key in mapping.items():
        if old_key in src.files:
            out[new_key] = src[old_key]
            print(f"  {new_key:<20} <- {old_key}  "
                  f"(n={len(src[old_key])}, "
                  f"range=[{float(np.nanmin(src[old_key])):.3f}, "
                  f"{float(np.nanmax(src[old_key])):.3f}])")
        else:
            out[new_key] = np.full(n, np.nan)
            print(f"  {new_key:<20} <- MISSING, filled NaN")

    for key in missing:
        out[key] = np.full(n, np.nan)
    print(f"  + {len(missing)} extra channels filled with NaN")

    # Copy pass-through fields
    for key in ["first_correct", "multi_correct",
                "first_pred", "multi_pred", "reference"]:
        if key in src.files:
            out[key] = src[key]
        else:
            print(f"[warn] {key} missing in source")

    out["model"] = np.array(["qwen"], dtype=object)
    out["n_problems"] = n

    DST.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DST, **out)
    print(f"\n[convert] saved {DST}")
    print(f"[convert] n_problems={n}")
    print(f"[convert] first_correct.mean()={out['first_correct'].mean():.4f}")


if __name__ == "__main__":
    main()