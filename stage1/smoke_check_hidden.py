"""smoke_check_hidden.py — Sanity check on extracted hidden states before FM training."""
import numpy as np
from pathlib import Path

import sys
FILE = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\OllamaModels\temporal_llm\data\stage1\boundary\hidden_train_streams_F.npz")


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def cos_rows(a, b):
    return np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def main():
    print(f"Loading {FILE}...")
    d = np.load(FILE, allow_pickle=True)
    print(f"Keys: {sorted(d.keys())}")
    n = len(d["y_t"])
    print(f"Total steps: {n}")

    # --- 1. Shapes, dtypes, NaN/INF ---
    section("1. SHAPES, DTYPES, NaN/INF")
    for k in sorted(d.keys()):
        arr = d[k]
        s = f"  {k:12s}  shape={str(arr.shape):18s}  dtype={str(arr.dtype):10s}"
        if arr.dtype.kind in "fc":
            s += f"  NaN={int(np.isnan(arr).sum())}  Inf={int(np.isinf(arr).sum())}"
        print(s)

    # --- 2. Design & event split ---
    section("2. DESIGN & EVENT SPLIT")
    designs = d["design"]
    is_ext = d["is_external"].astype(bool)
    sids = d["stream_id"]
    for dd in ["R", "O"]:
        m = designs == dd
        print(f"  Design {dd}: {int(m.sum()):5d} steps, "
              f"{len(np.unique(sids[m])):3d} streams  "
              f"({int((m & ~is_ext).sum()):5d} self, {int((m & is_ext).sum()):4d} ext)")

    # --- 3. Action & target distributions ---
    section("3. ACTION & TARGET DISTRIBUTIONS")
    for dd in ["R", "O"]:
        m = designs == dd
        acts, cnts = np.unique(d["action"][m], return_counts=True)
        print(f"\n  Design {dd}:")
        print(f"    actions: {dict(zip(acts.tolist(), cnts.tolist()))}")
        y, dy = d["y_t"][m], d["delta_y"][m]
        print(f"    y_t: mean={y.mean():+.4f} std={y.std():.4f} "
              f"[{y.min():+.3f}, {y.max():+.3f}]")
        print(f"    dy:  mean={dy.mean():+.4f} std={dy.std():.4f} "
              f"[{dy.min():+.3f}, {dy.max():+.3f}]")
        for label, mm in [("self", m & ~is_ext), ("ext", m & is_ext)]:
            if mm.sum():
                s = d["delta_y"][mm]
                print(f"    dy[{label:4s}]: n={int(mm.sum()):5d} "
                      f"mean={s.mean():+.4f} std={s.std():.4f}")

    # --- 4. Hidden state health ---
    section("4. HIDDEN STATE HEALTH")
    layers = ["h_12", "h_18", "h_24"]
    for k in layers:
        arr = d[k]
        norms = np.linalg.norm(arr, axis=1)
        perdim = arr.std(axis=0)
        dead = int((perdim < 1e-4).sum())
        print(f"  {k}: ‖h‖ mean={norms.mean():.2f} std={norms.std():.2f}  "
              f"per-dim std median={np.median(perdim):.4f}  "
              f"dead dims={dead}/{arr.shape[1]}")

    section("5. LAYER DIVERSITY (should be << 1.0)")
    h12, h18, h24 = d["h_12"], d["h_18"], d["h_24"]
    print(f"  cos(h12,h18) mean = {cos_rows(h12, h18).mean():.4f}")
    print(f"  cos(h12,h24) mean = {cos_rows(h12, h24).mean():.4f}")
    print(f"  cos(h18,h24) mean = {cos_rows(h18, h24).mean():.4f}")

    section("6. WITHIN- vs ACROSS-STREAM VARIATION (h_18)")
    uniq = np.unique(sids)
    arr = d["h_18"]
    within = np.mean([arr[sids == s].std(axis=0).mean() for s in uniq])
    means = np.stack([arr[sids == s].mean(axis=0) for s in uniq])
    across = means.std(axis=0).mean()
    print(f"  within-stream per-dim std  = {within:.4f}")
    print(f"  across-stream (of means)   = {across:.4f}")
    print(f"  ratio within/across        = {within / (across + 1e-9):.3f}")

    # --- 7. Linear probe: h_18 -> delta_y ---
    section("7. LINEAR PROBE h_18 -> delta_y (per design)")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    for dd in ["R", "O"]:
        m = designs == dd
        H, Y, S = d["h_18"][m], d["delta_y"][m], sids[m]
        u = np.unique(S)
        tr = np.array([s in set(u[: len(u) // 2].tolist()) for s in S])
        te = ~tr
        mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-6
        Htr, Hte = (H[tr] - mu) / sd, (H[te] - mu) / sd
        r = Ridge(alpha=10.0).fit(Htr, Y[tr])
        print(f"  Design {dd}: Train R²={r2_score(Y[tr], r.predict(Htr)):+.4f}  "
              f"Test R²={r2_score(Y[te], r.predict(Hte)):+.4f}")

    # --- 8. Action signal in design O self-events ---
    section("8. ACTION SIGNAL (design O, self-events)")
    m = (designs == "O") & ~is_ext
    a, dy = d["action"][m], d["delta_y"][m]
    for av in np.unique(a):
        mm = a == av
        print(f"  action={av:+d}: n={int(mm.sum()):4d}  "
              f"dy mean={dy[mm].mean():+.4f}  std={dy[mm].std():.4f}")
    across_a = np.var([dy[a == av].mean() for av in np.unique(a)])
    within_a = np.mean([dy[a == av].var() for av in np.unique(a)])
    print(f"  across-action var of means = {across_a:.4f}")
    print(f"  mean within-action var    = {within_a:.4f}")
    print(f"  SNR (across/within)       = {across_a / (within_a + 1e-9):.3f}")

    # --- 9. Trivial self-vs-ext separation baseline (design O) ---
    section("9. TRIVIAL SEPARATION BASELINE (design O)")
    from sklearn.metrics import roc_auc_score
    m = designs == "O"
    aO, dyO, extO = d["action"][m], d["delta_y"][m], is_ext[m]
    self_m = ~extO
    per_act_mean = {av: dyO[self_m & (aO == av)].mean()
                    for av in np.unique(aO[self_m])}
    pred = np.array([per_act_mean.get(av, 0.0) for av in aO])
    resid = np.abs(dyO - pred)
    auc = roc_auc_score(extO.astype(int), resid)
    print(f"  AUC(ext vs self | |residual| from per-action mean) = {auc:.4f}")
    print("  Интерпретация: ~0.5 — FM должен учиться; 0.8–0.99 — сигнал есть,")
    print("                 >0.99 — тривиально разделимо (проверь confound).")

    section("DONE")


if __name__ == "__main__":
    main()