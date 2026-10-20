"""ar_p_baseline.py — AR(p) persistence baselines.

Extends the paper's AR(1) baseline to AR(3) and AR(5) in the std
coordinate. Compares against the full 5-feature ridge.

Interpretation:
    - If transformer gap survives AR(5), claim is robust.
    - If AR(5) closes transformer gap, residual structure is trivial
      long-memory and paper must be revised.
    - SSM: AR(1) and AR(5) should coincide (no structure beyond
      smoothness).

Design: each of p lags uses a K=4 sub-window (matching paper's K).
AR(p) total context = p * 4.
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

SUB_WIN = 4         # each lag's window size
H_FIX   = 2
W_WIN   = 8
RIDGE_ALPHA = 1.0


def _load(model):
    sfx = "" if model == "qwen" else f"_{model}"
    for p in [SM_DIR / f"L0{sfx}_traj.npz",
              CACHE_DIR / f"sm_L0{sfx}.npz"]:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "lambdas" in d:
                lambdas = [np.asarray(x, dtype=float) for x in d["lambdas"]]
                if "train_ids" in d and "test_ids" in d:
                    return lambdas, d["train_ids"].tolist(), d["test_ids"].tolist()
                n = len(lambdas)
                n_tr = int(n * 0.7)
                return lambdas, list(range(n_tr)), list(range(n_tr, n))
    return None, None, None


def ar_p_features(x, p):
    """Features: p lagged window-stds, each over SUB_WIN samples."""
    K_total = p * SUB_WIN
    T = len(x)
    if T <= K_total:
        return np.zeros((0, p)), np.zeros(0, dtype=int)
    rows, idxs = [], []
    for i in range(K_total, T):
        feats = []
        for j in range(p):
            lo = i - K_total + j * SUB_WIN
            w = x[lo:lo + SUB_WIN]
            feats.append(w.std() if len(w) > 1 else 0.0)
        rows.append(feats)
        idxs.append(i)
    return np.array(rows), np.array(idxs, dtype=int)


def full_features(x, K=4):
    """Paper's 5-feature set."""
    T = len(x)
    if T <= K:
        return np.zeros((0, 5)), np.zeros(0, dtype=int)
    grid = np.arange(K)
    gc = grid - grid.mean()
    denom = (gc ** 2).sum()
    rows, idxs = [], []
    for i in range(K, T):
        w = x[i - K:i]
        dw = np.diff(w); adw = np.abs(dw)
        wm = w.mean()
        s = float((gc * (w - wm)).sum() / denom) if denom > 0 else 0.0
        rows.append([wm, w.std(),
                     adw.mean() if len(adw) else 0.0,
                     adw.std() if len(adw) else 0.0, s])
        idxs.append(i)
    return np.array(rows), np.array(idxs, dtype=int)


def make_target(x, idxs):
    T = len(x)
    ys, keep = [], []
    for j, i in enumerate(idxs):
        if i + H_FIX + W_WIN <= T:
            ys.append(x[i + H_FIX:i + H_FIX + W_WIN].std())
            keep.append(j)
    return np.array(ys), np.array(keep, dtype=int)


def eval_features(trajs, tr, te, feat_fn):
    def _build(ids):
        Xs, ys = [], []
        for i in ids:
            _, v = to_tau(trajs[i])
            if len(v) < 32:
                continue
            X, idxs = feat_fn(v)
            if len(idxs) == 0:
                continue
            y, keep = make_target(v, idxs)
            if len(keep) == 0:
                continue
            Xs.append(X[keep]); ys.append(y)
        if not Xs:
            return None
        return np.concatenate(Xs), np.concatenate(ys)

    trd = _build(tr)
    ted = _build(te)
    if trd is None or ted is None:
        return None
    Xtr, ytr = trd
    Xte, yte = ted
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(Xtr), ytr)
    return float(r2_score(yte, m.predict(sc.transform(Xte))))


def analyze(model):
    trajs, tr, te = _load(model)
    if trajs is None:
        return None

    out = {"model": model}
    for p in (1, 3, 5, 10):
        r2 = eval_features(trajs, tr, te, lambda x, p=p: ar_p_features(x, p))
        out[f"AR({p})"] = r2
    out["full_5feat"] = eval_features(
        trajs, tr, te, lambda x: full_features(x, K=4))
    return out


def main():
    results = {}
    for model in ("qwen", "mamba2"):
        r = analyze(model)
        if r is None:
            print(f"[{model}] no data")
            continue
        results[model] = r
        print(f"\n=== {model} ===")
        print(f"  AR(1)       R² = {r['AR(1)']:+.4f}")
        print(f"  AR(3)       R² = {r['AR(3)']:+.4f}")
        print(f"  AR(5)       R² = {r['AR(5)']:+.4f}")
        print(f"  AR(10)      R² = {r['AR(10)']:+.4f}")
        print(f"  full 5-feat R² = {r['full_5feat']:+.4f}")
        print(f"  gap over AR(1)  = {r['full_5feat'] - r['AR(1)']:+.4f}")
        print(f"  gap over AR(5)  = {r['full_5feat'] - r['AR(5)']:+.4f}")
        print(f"  gap over AR(10) = {r['full_5feat'] - r['AR(10)']:+.4f}")

    import json
    out_path = SM_DIR.parent / "spectral" / "ar_p_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()