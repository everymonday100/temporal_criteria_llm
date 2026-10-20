"""feature_ablation.py — localize the residual gap over AR(1).

Feature set (6): mean, std, mean|dv|, std|dv|, slope, mean²

For each model in {qwen, mamba2} and coord in {arc, token}:
  - AR(1) baseline: std-only
  - full: all 6 features
  - leave-one-in, std+each, leave-one-out
  - a focused quadratic test: {std} vs {std,mean} vs {std,mean,mean²}

Motivation: Figure 4 shows Qwen's future std is monotone in current
mean (slope +0.70), Mamba's is U-shaped (linear slope -0.14). A linear
feature set cannot capture the U-shape. Adding mean² should close the
gap on Mamba while leaving Qwen unchanged.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.generation import to_tau

SM_DIR    = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")
CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")

K_FIX = 4
H_FIX = 2
W_WIN = 8
RIDGE_ALPHA = 1.0

FEATURE_NAMES = ["mean", "std", "mean|dv|", "std|dv|", "slope", "mean²"]
IDX_MEAN    = 0
IDX_STD     = 1
IDX_MEANDV  = 2
IDX_STDDV   = 3
IDX_SLOPE   = 4
IDX_MEAN2   = 5


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def _load_trajs(model):
    sfx = "" if model == "qwen" else f"_{model}"
    for p in [SM_DIR / f"L0{sfx}_traj.npz",
              CACHE_DIR / f"sm_L0{sfx}.npz"]:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "lambdas" in d:
                lambdas = [np.asarray(x, dtype=float)
                           for x in d["lambdas"]]
                if "train_ids" in d and "test_ids" in d:
                    tr = d["train_ids"].tolist()
                    te = d["test_ids"].tolist()
                else:
                    n = len(lambdas)
                    n_tr = int(n * 0.7)
                    tr = list(range(n_tr))
                    te = list(range(n_tr, n))
                return lambdas, tr, te, p.name
    return None, None, None, None


# ----------------------------------------------------------------------
# Feature / target builders
# ----------------------------------------------------------------------
def build_features(x, K):
    T = len(x)
    if T <= K:
        return np.zeros((0, 6)), np.zeros(0, dtype=int)
    grid = np.arange(K)
    gc = grid - grid.mean()
    denom = (gc ** 2).sum()
    rows, idxs = [], []
    for i in range(K, T):
        w = x[i - K:i]
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
            wm * wm,
        ])
        idxs.append(i)
    return np.array(rows, dtype=np.float64), np.array(idxs, dtype=int)


def build_target(x, idxs, H, W):
    T = len(x)
    ys, keep = [], []
    for j, i in enumerate(idxs):
        if i + H + W <= T:
            ys.append(x[i + H:i + H + W].std())
            keep.append(j)
    return np.array(ys, dtype=np.float64), np.array(keep, dtype=int)


def make_dataset(trajs, ids, coord, K, H, W):
    Xs, ys, ps = [], [], []
    for pid_local, i in enumerate(ids):
        lam = trajs[i]
        if coord == "arc":
            _, v = to_tau(lam)
        else:
            v = lam
        if len(v) < K + H + W + 2:
            continue
        X, idxs = build_features(v, K)
        if len(idxs) == 0:
            continue
        y, keep = build_target(v, idxs, H, W)
        if len(keep) == 0:
            continue
        Xs.append(X[keep])
        ys.append(y)
        ps.append(np.full(len(y), pid_local))
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(trajs, train_ids, test_ids, coord, feat_idx, K, H, W):
    tr = make_dataset(trajs, train_ids, coord, K, H, W)
    te = make_dataset(trajs, test_ids, coord, K, H, W)
    if tr is None or te is None:
        return None
    Xtr, ytr, _ = tr
    Xte, yte, _ = te
    if len(Xtr) < 20 or len(Xte) < 10:
        return None
    if isinstance(feat_idx, int):
        feat_idx = [feat_idx]
    Xtr = Xtr[:, feat_idx]
    Xte = Xte[:, feat_idx]
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(Xtr), ytr)
    return float(r2_score(yte, m.predict(sc.transform(Xte))))


# ----------------------------------------------------------------------
# Per-model analysis
# ----------------------------------------------------------------------
def run_model(model, coord):
    trajs, tr, te, src = _load_trajs(model)
    if trajs is None:
        return None

    base_std = evaluate(trajs, tr, te, coord, [IDX_STD], K_FIX, H_FIX, W_WIN)
    full     = evaluate(trajs, tr, te, coord,
                        [IDX_MEAN, IDX_STD, IDX_MEANDV, IDX_STDDV,
                         IDX_SLOPE, IDX_MEAN2],
                        K_FIX, H_FIX, W_WIN)
    if base_std is None or full is None:
        return None
    gap = full - base_std

    # leave-one-in
    singles = {}
    for k, name in enumerate(FEATURE_NAMES):
        r2 = evaluate(trajs, tr, te, coord, [k], K_FIX, H_FIX, W_WIN)
        singles[name] = r2

    # std + each other
    pairs = {}
    for k, name in enumerate(FEATURE_NAMES):
        if k == IDX_STD:
            continue
        r2 = evaluate(trajs, tr, te, coord,
                      [IDX_STD, k], K_FIX, H_FIX, W_WIN)
        pairs[name] = r2

    # leave-one-out from full
    loos = {}
    for k, name in enumerate(FEATURE_NAMES):
        rest = [j for j in range(len(FEATURE_NAMES)) if j != k]
        r2 = evaluate(trajs, tr, te, coord, rest, K_FIX, H_FIX, W_WIN)
        loos[name] = r2

    # quadratic focused test
    quad = {
        "std":              evaluate(trajs, tr, te, coord,
                                    [IDX_STD], K_FIX, H_FIX, W_WIN),
        "std+mean":         evaluate(trajs, tr, te, coord,
                                    [IDX_STD, IDX_MEAN], K_FIX, H_FIX, W_WIN),
        "std+mean+mean²":   evaluate(trajs, tr, te, coord,
                                    [IDX_STD, IDX_MEAN, IDX_MEAN2],
                                    K_FIX, H_FIX, W_WIN),
        "std+mean²":        evaluate(trajs, tr, te, coord,
                                    [IDX_STD, IDX_MEAN2],
                                    K_FIX, H_FIX, W_WIN),
    }

    # best pairs / singles
    pair_gain = {n: (pairs[n] - base_std)
                 for n in pairs if pairs[n] is not None}
    best_pair = max(pair_gain.items(), key=lambda kv: kv[1]) \
        if pair_gain else (None, 0.0)
    single_gain = {n: (r - base_std) if r is not None else -1
                   for n, r in singles.items()}
    best_single = max(single_gain.items(), key=lambda kv: kv[1]) \
        if single_gain else (None, 0.0)

    return {
        "model": model, "coord": coord, "source": src,
        "std_only": base_std, "full": full, "gap": gap,
        "singles": singles, "pairs": pairs, "loos": loos,
        "quad": quad,
        "best_single": best_single, "best_pair": best_pair,
        "pair_gain": pair_gain, "single_gain": single_gain,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _fmt(x):
    return f"{x:+.4f}" if x is not None else "  n/a "


def print_result(r):
    print(f"\n--- {r['model']}  coord={r['coord']}  ({r['source']}) ---")
    print(f"  AR(1) std-only:       {_fmt(r['std_only'])}")
    print(f"  full 6-feature:       {_fmt(r['full'])}")
    print(f"  gap:                  {_fmt(r['gap'])}")

    print("  single features:")
    for name in FEATURE_NAMES:
        v = r["singles"].get(name)
        g = r["single_gain"].get(name)
        print(f"    {name:<10} R2={_fmt(v)}  gain_over_std={_fmt(g)}")

    print("  pairs with std:")
    for name in FEATURE_NAMES:
        if name == "std":
            continue
        v = r["pairs"].get(name)
        g = r["pair_gain"].get(name)
        print(f"    std+{name:<10} R2={_fmt(v)}  gap_recovered={_fmt(g)}")

    print("  leave-one-out (full minus feature):")
    for name in FEATURE_NAMES:
        v = r["loos"].get(name)
        loss = (r["full"] - v) if v is not None else None
        print(f"    -{name:<10} R2={_fmt(v)}  drop={_fmt(loss)}")

    print("  QUADRATIC TEST:")
    for key in ["std", "std+mean", "std+mean+mean²", "std+mean²"]:
        v = r["quad"].get(key)
        print(f"    {key:<20} R2={_fmt(v)}")

    bs_name, bs_gain = r["best_single"]
    bp_name, bp_gain = r["best_pair"]
    print(f"  best single:    {bs_name}  gain={bs_gain:+.4f}")
    print(f"  best pair:      std+{bp_name}  gain={bp_gain:+.4f}")


def verdict(r):
    """Verdict based on quadratic test."""
    gap = r["gap"]
    if gap is None or abs(gap) < 0.02:
        return "no_gap", f"gap={gap:+.4f}"
    q = r["quad"]
    std_r2   = q.get("std")
    lin_r2   = q.get("std+mean")
    quad_r2  = q.get("std+mean+mean²")
    if None in (std_r2, lin_r2, quad_r2):
        return "inconclusive", "missing quadratic values"

    lin_gain  = lin_r2  - std_r2
    quad_gain = quad_r2 - lin_r2

    if quad_gain >= 0.5 * gap:
        return "quadratic_driver", (
            f"adding mean² recovers {quad_gain/gap*100:.0f}% of gap on "
            f"top of std+mean; U-shape is the mechanism")
    if lin_gain >= 0.8 * gap:
        return "linear_driver", (
            f"std+mean recovers {lin_gain/gap*100:.0f}% of gap; "
            f"monotone level dependence")
    if lin_gain + quad_gain >= 0.8 * gap:
        return "mixed_driver", (
            f"linear part {lin_gain/gap*100:.0f}%, quadratic part "
            f"{quad_gain/gap*100:.0f}%")
    return "inconclusive", (
        f"linear gain {lin_gain/gap*100:.0f}%, quadratic gain "
        f"{quad_gain/gap*100:.0f}%, neither dominant")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    all_results = {}
    for model in ["qwen", "mamba2"]:
        for coord in ["arc", "token"]:
            r = run_model(model, coord)
            if r is None:
                print(f"\n[{model}/{coord}] no data")
                continue
            print_result(r)
            label, expl = verdict(r)
            print(f"  VERDICT: {label} — {expl}")
            all_results[f"{model}_{coord}"] = {
                "model": model, "coord": coord,
                "std_only": r["std_only"], "full": r["full"],
                "gap": r["gap"],
                "singles": r["singles"], "pairs": r["pairs"],
                "loos": r["loos"], "quad": r["quad"],
                "best_single": list(r["best_single"]),
                "best_pair": list(r["best_pair"]),
                "verdict": label, "explanation": expl,
            }

    print("\n=== CROSS-MODEL SUMMARY (arc) ===")
    for model in ["qwen", "mamba2"]:
        r = all_results.get(f"{model}_arc")
        if not r:
            continue
        q = r["quad"]
        print(f"\n  {model}:")
        print(f"    gap             = {r['gap']:+.4f}")
        print(f"    std             = {q['std']:+.4f}")
        print(f"    std+mean        = {q['std+mean']:+.4f}")
        print(f"    std+mean+mean²  = {q['std+mean+mean²']:+.4f}")
        print(f"    verdict         = {r['verdict']}")

    import json
    out = SM_DIR.parent / "spectral" / "feature_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()