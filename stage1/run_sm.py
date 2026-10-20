"""run_sm.py — self-model grid for Stage-1 interfaces.

Test (matching Paper 2's token-coordinate negative test):
    Features: 5 features of prefix window of Lambda(t)
    Target:   std(Lambda) over future window [i+H : i+H+W]
    Grid:     K in {2,4,8,16}, H in {2,4}, W = 16

Protocol per interface:
    25 prompts x 6 segments x 50 tokens with segmented state injection.
    Extract full Lambda trajectory per prompt.
    Split prompts 17/8 (train/test).
    Fit ridge. Report R2, centered, null, shuffled.

Model-aware:
    --model qwen     → uses Qwen2.5-3B, all interfaces (L1a..L2b)
    --model mamba2   → uses Mamba2-1.3b, excludes attention-based
                       interfaces (L1e, L2b)

Outputs (suffix _mamba2 for Mamba):
    data/stage1/sm/{iface}[_mamba2].json
    data/stage1/sm/{iface}[_mamba2]_traj.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.interfaces import (
    apply_interface_segmented, INTERFACE_SPECS, NONZERO_INTERFACES,
)
from common.datasets import load_sm_prompts
from common.io import save_json, load_npz, ensure_dir

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")
CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")

K_GRID = (2, 4, 8, 16)
H_GRID = (2, 4)
W_WIN = 16
RIDGE_ALPHA = 1.0
TRAIN_FRAC = 0.7
SEED = 42
BOOTSTRAP_B = 1000

MAMBA_EXCLUDE = {"L1e", "L2b"}  # attention-based; unavailable on Mamba2


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", type=str, default="all",
                   help="'all' or specific interface (L1a, L2a, ...)")
    p.add_argument("--model", type=str, default="qwen",
                   choices=["qwen", "mamba2"])
    p.add_argument("--n_prompts", type=int, default=25)
    p.add_argument("--tokens_per_seg", type=int, default=50)
    p.add_argument("--n_segments", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _sfx(args) -> str:
    return "" if args.model == "qwen" else f"_{args.model}"


# ----------------------------------------------------------------------
# Feature / target builders
# ----------------------------------------------------------------------
def make_features(lam: np.ndarray, K: int):
    T = len(lam)
    if T <= K:
        return np.zeros((0, 5)), np.zeros(0, dtype=int)
    grid = np.arange(K)
    gc = grid - grid.mean()
    denom = (gc ** 2).sum()
    rows, idxs = [], []
    for i in range(K, T):
        w = lam[i - K:i]
        dw = np.diff(w)
        adw = np.abs(dw)
        wm = w.mean()
        s = float((gc * (w - wm)).sum() / denom) if denom > 0 else 0.0
        rows.append([
            wm,
            w.std(),
            adw.mean() if len(adw) else 0.0,
            adw.std() if len(adw) else 0.0,
            s,
        ])
        idxs.append(i)
    return np.array(rows, dtype=np.float64), np.array(idxs, dtype=int)


def make_target(lam: np.ndarray, idxs, H: int, W: int):
    T = len(lam)
    ys, keep = [], []
    for j, i in enumerate(idxs):
        lo, hi = i + H, i + H + W
        if hi <= T:
            ys.append(lam[lo:hi].std())
            keep.append(j)
    return np.array(ys, dtype=np.float64), np.array(keep, dtype=int)


def build_dataset(trajectories, K, H, W):
    Xs, ys, ps = [], [], []
    for pid, lam in enumerate(trajectories):
        X, idxs = make_features(lam, K)
        if len(idxs) == 0:
            continue
        y, keep = make_target(lam, idxs, H, W)
        if len(keep) == 0:
            continue
        Xs.append(X[keep])
        ys.append(y)
        ps.append(np.full(len(y), pid))
    if not Xs:
        return (np.zeros((0, 5)), np.zeros(0), np.zeros(0, dtype=int))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)


def _center_within(y, pids):
    yc = y.copy()
    for pid in np.unique(pids):
        m = pids == pid
        yc[m] = y[m] - y[m].mean()
    return yc


def eval_grid_point(traj_train, traj_test, K, H, W, seed):
    Xtr, ytr, ptr = build_dataset(traj_train, K, H, W)
    Xte, yte, pte = build_dataset(traj_test, K, H, W)
    if len(Xtr) < 20 or len(Xte) < 10:
        return None

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    m = Ridge(alpha=RIDGE_ALPHA).fit(Xtr_s, ytr)
    yhat = m.predict(Xte_s)
    r2 = float(r2_score(yte, yhat))

    ytr_c = _center_within(ytr, ptr)
    yte_c = _center_within(yte, pte)
    mc = Ridge(alpha=RIDGE_ALPHA).fit(Xtr_s, ytr_c)
    r2_c = float(r2_score(yte_c, mc.predict(Xte_s)))

    rng = np.random.default_rng(seed + 7)
    ytr_n = ytr.copy()
    for pid in np.unique(ptr):
        idx = np.where(ptr == pid)[0]
        ytr_n[idx] = ytr_n[rng.permutation(idx)]
    mn = Ridge(alpha=RIDGE_ALPHA).fit(Xtr_s, ytr_n)
    r2_n = float(r2_score(yte, mn.predict(Xte_s)))

    perm = rng.permutation(len(Xtr_s))
    ms = Ridge(alpha=RIDGE_ALPHA).fit(Xtr_s[perm], ytr)
    r2_s = float(r2_score(yte, ms.predict(Xte_s)))

    return {
        "r2": r2, "r2_centered": r2_c,
        "r2_null": r2_n, "r2_shuffled": r2_s,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "yhat": yhat.tolist(), "yte": yte.tolist(), "pte": pte.tolist(),
    }


def bootstrap_ci_from_pairs(yhat, yte, pte, B=BOOTSTRAP_B, seed=SEED):
    unique = np.unique(pte)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(B):
        sample = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(pte, sample)
        if mask.sum() < 3:
            continue
        scores.append(r2_score(np.asarray(yte)[mask],
                                np.asarray(yhat)[mask]))
    if not scores:
        return 0.0, 0.0
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def verdict_sm(r2, centered, null_r2):
    if r2 >= 0.5 and centered >= 0.4 and null_r2 <= 0.1:
        return "positive"
    if null_r2 > 0.1:
        return "null_leak"
    if r2 < 0.1:
        return "T1_signal_weak"
    return "T3_below_threshold"


# ----------------------------------------------------------------------
# Per-interface runner
# ----------------------------------------------------------------------
def run_one_interface(iface, args, tok, model, prompts):
    sfx = _sfx(args)
    out_path = OUT_DIR / f"{iface}{sfx}.json"
    traj_path = OUT_DIR / f"{iface}{sfx}_traj.npz"

    if out_path.exists() and not args.overwrite:
        print(f"[{iface}{sfx}] exists, skipping")
        return

    print(f"\n[{iface}{sfx}] channels={INTERFACE_SPECS[iface]}")
    t0 = time.time()

    traj = []
    seg_texts_all = []
    seg_signals_all = []

    for i, prompt in enumerate(prompts):
        res = apply_interface_segmented(
            model, tok, prompt, iface,
            n_segments=args.n_segments,
            tokens_per_seg=args.tokens_per_seg,
            T=0.55,
            seed=args.seed + i * 1000,
        )
        traj.append(np.array(res["full_lambda"]))
        seg_texts_all.append(res["per_seg_text"])
        seg_signals_all.append(res["per_seg_signals"])
        print(f"  [{i+1:2d}/{len(prompts)}] len={len(traj[-1])}  "
              f"lam_mean={traj[-1].mean():.0f}  "
              f"lam_std={traj[-1].std():.0f}  "
              f"({time.time()-t0:.0f}s)")

    # --- Split prompts
    pids = np.arange(len(prompts))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pids))
    n_train = int(len(pids) * TRAIN_FRAC)
    train_ids = sorted(pids[perm[:n_train]].tolist())
    test_ids = sorted(pids[perm[n_train:]].tolist())
    traj_train = [traj[i] for i in train_ids]
    traj_test = [traj[i] for i in test_ids]

    # --- Save trajectories
    ensure_dir(OUT_DIR)
    try:
        np.savez(
            traj_path,
            iface=iface,
            model=args.model,
            lambdas=np.array(traj, dtype=object),
            train_ids=np.array(train_ids),
            test_ids=np.array(test_ids),
            prompts=np.array(prompts, dtype=object),
            seg_texts=np.array(seg_texts_all, dtype=object),
        )
        print(f"[{iface}{sfx}] trajectories saved {traj_path}")
    except Exception as e:
        print(f"[{iface}{sfx}] WARN could not save traj: "
              f"{type(e).__name__}: {e}")

    # --- Grid
    grid = {}
    best = None
    for K in K_GRID:
        for H in H_GRID:
            r = eval_grid_point(traj_train, traj_test, K, H, W_WIN,
                                seed=SEED)
            if r is None:
                continue
            lo, hi = bootstrap_ci_from_pairs(r["yhat"], r["yte"], r["pte"])
            tag = f"K{K}_H{H}"
            grid[tag] = {
                "K": K, "H": H,
                "r2": r["r2"], "r2_centered": r["r2_centered"],
                "r2_null": r["r2_null"], "r2_shuffled": r["r2_shuffled"],
                "ci_low": lo, "ci_high": hi,
                "n_train": r["n_train"], "n_test": r["n_test"],
            }
            print(f"  K={K:2d} H={H}  R2={r['r2']:+.4f}  "
                  f"centered={r['r2_centered']:+.4f}  "
                  f"null={r['r2_null']:+.4f}")
            if best is None or r["r2"] > best["r2"]:
                best = grid[tag]

    verdict = verdict_sm(best["r2"], best["r2_centered"],
                         best["r2_null"]) if best else "no_fit"

    out = {
        "iface": iface,
        "model": args.model,
        "channels": list(INTERFACE_SPECS[iface])
                    if INTERFACE_SPECS[iface] else [],
        "n_prompts": len(prompts),
        "n_segments": args.n_segments,
        "tokens_per_seg": args.tokens_per_seg,
        "split": {"train": train_ids, "test": test_ids},
        "grid": grid,
        "best": best,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
        "per_seg_text": seg_texts_all,
        "per_seg_signals": seg_signals_all,
    }
    save_json(out_path, out)
    print(f"[{iface}{sfx}] verdict={verdict}  saved {out_path}")


# ----------------------------------------------------------------------
# L0 reference (from cached baseline)
# ----------------------------------------------------------------------
def run_l0_reference(args):
    sfx = _sfx(args)
    l0_path = CACHE_DIR / f"sm_L0{sfx}.npz"
    if not l0_path.exists():
        print(f"[L0{sfx}] no cached baseline — run "
              f"stage1/run_l0.py --which sm --model {args.model}")
        return

    print(f"\n[L0{sfx}] using cached baseline ({l0_path})")
    data = load_npz(l0_path)
    traj = [np.asarray(x, dtype=np.float64) for x in data["lambdas"]]
    texts = data.get("texts", None)
    prompts = data.get("prompts", None)

    pids = np.arange(len(traj))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pids))
    n_train = int(len(pids) * TRAIN_FRAC)
    train_ids = sorted(pids[perm[:n_train]].tolist())
    test_ids = sorted(pids[perm[n_train:]].tolist())

    l0_traj_path = OUT_DIR / f"L0{sfx}_traj.npz"
    try:
        np.savez(
            l0_traj_path,
            iface="L0",
            model=args.model,
            lambdas=np.array(traj, dtype=object),
            train_ids=np.array(train_ids),
            test_ids=np.array(test_ids),
            prompts=prompts if prompts is not None
                    else np.array([], dtype=object),
            seg_texts=texts if texts is not None
                      else np.array([], dtype=object),
        )
        print(f"[L0{sfx}] trajectories saved {l0_traj_path}")
    except Exception as e:
        print(f"[L0{sfx}] WARN could not save traj: {e}")

    grid = {}
    best = None
    for K in K_GRID:
        for H in H_GRID:
            r = eval_grid_point(
                [traj[i] for i in train_ids],
                [traj[i] for i in test_ids],
                K, H, W_WIN, seed=SEED)
            if r is None:
                continue
            lo, hi = bootstrap_ci_from_pairs(r["yhat"], r["yte"], r["pte"])
            tag = f"K{K}_H{H}"
            grid[tag] = {
                "K": K, "H": H,
                "r2": r["r2"],
                "r2_centered": r["r2_centered"],
                "r2_null": r["r2_null"],
                "r2_shuffled": r["r2_shuffled"],
                "ci_low": lo, "ci_high": hi,
            }
            if best is None or r["r2"] > best["r2"]:
                best = grid[tag]

    verdict = verdict_sm(best["r2"], best["r2_centered"],
                         best["r2_null"]) if best else "no_fit"
    out = {
        "iface": "L0",
        "model": args.model,
        "channels": [],
        "n_prompts": len(traj),
        "split": {"train": train_ids, "test": test_ids},
        "grid": grid,
        "best": best,
        "verdict": verdict,
        "source": f"cached sm_L0{sfx}.npz",
    }
    save_json(OUT_DIR / f"L0{sfx}.json", out)
    print(f"[L0{sfx}] verdict={verdict}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dir(OUT_DIR)
    tok, model = load_model(args.model)
    prompts = load_sm_prompts(args.n_prompts)

    if args.iface == "all":
        ifaces = list(NONZERO_INTERFACES)
    else:
        ifaces = [args.iface]

    # Exclude attention-based interfaces for Mamba
    if args.model == "mamba2":
        before = len(ifaces)
        ifaces = [i for i in ifaces if i not in MAMBA_EXCLUDE]
        dropped = before - len(ifaces)
        if dropped:
            print(f"[mamba2] excluded {dropped} attention-based "
                  f"interface(s): {MAMBA_EXCLUDE}")

    for iface in ifaces:
        run_one_interface(iface, args, tok, model, prompts)

    if args.iface == "all":
        run_l0_reference(args)


if __name__ == "__main__":
    main()