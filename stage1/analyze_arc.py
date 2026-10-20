"""analyze_arc.py — apply Paper-2 arc-length analysis to Stage-1 trajectories.

Reads {iface}_traj.npz files written by run_sm.py and evaluates the
battery's self-model test in either coordinate:
  --coord arc    (default) arc-length tau
  --coord token  raw token index (matches the negative result in Paper 2)
  --coord both   run both and compare side by side

Model-aware (--model qwen | mamba2) selects file suffix.

Grid: K in {4,8,16}, H in {2,4}, W=8. In arc coord these are tau units;
in token coord they are token counts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.generation import to_tau

CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")
SM_DIR    = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")

TAU_GRID = [(4, 2), (4, 4), (8, 2), (8, 4), (16, 2), (16, 4)]
W_WIN    = 8
RIDGE_ALPHA = 1.0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="qwen",
                   choices=["qwen", "mamba2"])
    p.add_argument("--coord", type=str, default="arc",
                   choices=["arc", "token", "both"])
    return p.parse_args()


def _sfx(args):
    return "" if args.model == "qwen" else f"_{args.model}"


# ----------------------------------------------------------------------
# Feature / target builders
# ----------------------------------------------------------------------
def make_features(x, K):
    """Level-only features: [mean, std] over trailing window of K."""
    T = len(x)
    if T <= K:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    rows, idxs = [], []
    for i in range(K, T):
        w = x[i - K:i]
        rows.append([w.mean(), w.std()])
        idxs.append(i)
    return np.array(rows, dtype=np.float64), np.array(idxs, dtype=int)


def make_target(x, idxs, H, W):
    T = len(x)
    ys, keep = [], []
    for j, i in enumerate(idxs):
        if i + H + W <= T:
            ys.append(x[i + H:i + H + W].std())
            keep.append(j)
    return np.array(ys, dtype=np.float64), np.array(keep, dtype=int)


def build(trajs, K, H, W, coord="arc"):
    Xs, ys, ps = [], [], []
    for pid, lam in enumerate(trajs):
        if coord == "arc":
            _, v = to_tau(lam)
        else:
            v = np.asarray(lam, dtype=np.float64)
        if len(v) < K + H + W + 2:
            continue
        X, idxs = make_features(v, K)
        if len(idxs) == 0:
            continue
        y, keep = make_target(v, idxs, H, W)
        if len(keep) == 0:
            continue
        Xs.append(X[keep])
        ys.append(y)
        ps.append(np.full(len(y), pid))
    if not Xs:
        return None
    return (np.concatenate(Xs),
            np.concatenate(ys),
            np.concatenate(ps))


def eval_grid(traj_train, traj_test, K, H, W, coord="arc"):
    tr = build(traj_train, K, H, W, coord=coord)
    te = build(traj_test, K, H, W, coord=coord)
    if tr is None or te is None:
        return None
    Xtr, ytr, ptr = tr
    Xte, yte, pte = te
    if len(Xtr) < 20 or len(Xte) < 10:
        return None

    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(Xtr), ytr)
    yhat = m.predict(sc.transform(Xte))
    r2 = float(r2_score(yte, yhat))

    # per-prompt centering
    ytr_c = ytr.copy()
    yte_c = yte.copy()
    for pid in np.unique(ptr):
        mask = ptr == pid
        ytr_c[mask] -= ytr_c[mask].mean()
    for pid in np.unique(pte):
        mask = pte == pid
        yte_c[mask] -= yte_c[mask].mean()
    mc = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(Xtr), ytr_c)
    r2_c = float(r2_score(yte_c, mc.predict(sc.transform(Xte))))

    # null: shuffle target within each train prompt
    rng = np.random.default_rng(7)
    ytr_n = ytr.copy()
    for pid in np.unique(ptr):
        idx = np.where(ptr == pid)[0]
        ytr_n[idx] = ytr_n[rng.permutation(idx)]
    mn = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(Xtr), ytr_n)
    r2_n = float(r2_score(yte, mn.predict(sc.transform(Xte))))

    # AR(1): single-feature (std only) baseline
    x_tr_std = sc.transform(Xtr)[:, 1:2]
    x_te_std = sc.transform(Xte)[:, 1:2]
    m_ar1 = Ridge(alpha=RIDGE_ALPHA).fit(x_tr_std, ytr)
    r2_ar1 = float(r2_score(yte, m_ar1.predict(x_te_std)))

    return {
        "r2": r2,
        "r2_centered": r2_c,
        "r2_null": r2_n,
        "r2_ar1": r2_ar1,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }


# ----------------------------------------------------------------------
# Per-file analysis
# ----------------------------------------------------------------------
def analyze_one(name, path, coords=("arc",)):
    if not path.exists():
        if name == "L0":
            cache_alt = CACHE_DIR / \
                f"sm_L0{'_mamba2' if 'mamba2' in path.name else ''}.npz"
            if cache_alt.exists():
                print(f"\n[{name}] {path.name} not found, "
                      f"falling back to {cache_alt.name}")
                path = cache_alt
            else:
                print(f"\n[{name}] {path.name} not found — skipped")
                return None
        else:
            print(f"\n[{name}] {path.name} not found — skipped")
            return None

    d = np.load(path, allow_pickle=True)
    if "lambdas" not in d:
        print(f"\n[{name}] no 'lambdas' key — skipped")
        return None

    lambdas = [np.asarray(x, dtype=float) for x in d["lambdas"]]
    if "train_ids" in d and "test_ids" in d:
        train_ids = d["train_ids"].tolist()
        test_ids = d["test_ids"].tolist()
    else:
        train_ids = list(range(17))
        test_ids = list(range(17, len(lambdas)))

    traj_train = [lambdas[i] for i in train_ids if len(lambdas[i]) > 0]
    traj_test = [lambdas[i] for i in test_ids if len(lambdas[i]) > 0]

    out = {}
    for c in coords:
        print(f"\n[{name}] coord={c}  "
              f"train={len(traj_train)}  test={len(traj_test)}")
        print(f"  {'K':>3} {'H':>3} "
              f"{'R2':>9} {'centered':>10} {'null':>9} {'AR(1)':>9}")
        results = {}
        for K, H in TAU_GRID:
            r = eval_grid(traj_train, traj_test, K, H, W_WIN, coord=c)
            if r is None:
                print(f"  {K:>3} {H:>3}  (skipped — not enough samples)")
                continue
            tag = f"K{K}_H{H}"
            results[tag] = r
            print(f"  {K:>3} {H:>3} "
                  f"{r['r2']:>+9.4f} {r['r2_centered']:>+10.4f} "
                  f"{r['r2_null']:>+9.4f} {r['r2_ar1']:>+9.4f}")
        out[c] = results
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    sfx = _sfx(args)

    candidates = [
        ("L0",  SM_DIR / f"L0{sfx}_traj.npz"),
        ("L1a", SM_DIR / f"L1a{sfx}_traj.npz"),
        ("L1b", SM_DIR / f"L1b{sfx}_traj.npz"),
        ("L1c", SM_DIR / f"L1c{sfx}_traj.npz"),
        ("L1d", SM_DIR / f"L1d{sfx}_traj.npz"),
        ("L1e", SM_DIR / f"L1e{sfx}_traj.npz"),
        ("L1f", SM_DIR / f"L1f{sfx}_traj.npz"),
        ("L2a", SM_DIR / f"L2a{sfx}_traj.npz"),
        ("L2b", SM_DIR / f"L2b{sfx}_traj.npz"),
    ]

    coords = ("arc", "token") if args.coord == "both" \
             else (args.coord,)

    print(f"=== analyze_arc: model={args.model}  "
          f"coord={args.coord} ===")
    if "arc" in coords:
        print(f"    delta_tau adaptive (mean(Lambda)/30)")
    print(f"    grid: K={[k for k, _ in TAU_GRID]}  "
          f"H={sorted(set(h for _, h in TAU_GRID))}  W={W_WIN}")

    summary = {}
    for name, path in candidates:
        res = analyze_one(name, path, coords=coords)
        if res is None:
            continue
        entry = {}
        for c, grid in res.items():
            if not grid:
                continue
            best = max(grid.values(), key=lambda x: x["r2"])
            entry[c] = {
                "best_r2":       best["r2"],
                "best_centered": best["r2_centered"],
                "best_null":     best["r2_null"],
                "best_ar1":      best["r2_ar1"],
                "grid":          grid,
            }
        if entry:
            summary[name] = entry

    # Summary table
    if summary:
        print(f"\n=== SUMMARY ({args.model}) ===")
        for c in coords:
            rows = [(n, summary[n][c]) for n in
                    ["L0", "L1a", "L1b", "L1c",
                     "L1d", "L1e", "L1f", "L2a", "L2b"]
                    if n in summary and c in summary[n]]
            if not rows:
                continue
            print(f"\n  coord={c}")
            print(f"    {'iface':<6} {'R2':>9} {'centered':>10} "
                  f"{'null':>9} {'AR(1)':>9} {'gap':>9}")
            for name, s in rows:
                gap = s["best_r2"] - s["best_ar1"]
                print(f"    {name:<6} "
                      f"{s['best_r2']:>+9.4f} "
                      f"{s['best_centered']:>+10.4f} "
                      f"{s['best_null']:>+9.4f} "
                      f"{s['best_ar1']:>+9.4f} "
                      f"{gap:>+9.4f}")

    # Save JSON
    suffix = "" if args.coord == "arc" else f"_{args.coord}"
    out = SM_DIR / f"arc_summary{sfx}{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()