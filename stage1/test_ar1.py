import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")
SM_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")


def make_features(x, K):
    rows, idxs = [], []
    for i in range(K, len(x)):
        w = x[i-K:i]
        rows.append([w.mean(), w.std()])
        idxs.append(i)
    return np.array(rows), np.array(idxs)


def make_target(x, idxs, H, W):
    ys, keep = [], []
    for j, i in enumerate(idxs):
        if i + H + W <= len(x):
            ys.append(x[i+H:i+H+W].std())
            keep.append(j)
    return np.array(ys), np.array(keep)


def to_tau_grid(lam, delta=50.0):
    tau = np.cumsum(lam)
    if tau[-1] < 2 * delta:
        return np.array([])
    n = int(tau[-1] // delta)
    grid = (np.arange(1, n + 1) * delta).astype(float)
    return np.interp(grid, tau, lam)


def ar1_on(lambdas, train_ids, test_ids, K, H, W):
    Xtr_list, ytr_list = [], []
    for i in train_ids:
        v = to_tau_grid(np.asarray(lambdas[i], dtype=float))
        if len(v) < K + H + W:
            continue
        X, idxs = make_features(v, K)
        y, keep = make_target(v, idxs, H, W)
        Xtr_list.append(X[keep]); ytr_list.append(y)
    Xte_list, yte_list = [], []
    for i in test_ids:
        v = to_tau_grid(np.asarray(lambdas[i], dtype=float))
        if len(v) < K + H + W:
            continue
        X, idxs = make_features(v, K)
        y, keep = make_target(v, idxs, H, W)
        Xte_list.append(X[keep]); yte_list.append(y)
    if not Xtr_list or not Xte_list:
        return None
    Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list)
    Xte = np.concatenate(Xte_list); yte = np.concatenate(yte_list)
    sc = StandardScaler().fit(Xtr)
    Xtr_std = sc.transform(Xtr)[:, 1:2]
    Xte_std = sc.transform(Xte)[:, 1:2]
    m = Ridge(alpha=1.0).fit(Xtr_std, ytr)
    return r2_score(yte, m.predict(Xte_std))


# L0
print("=== L0 (cached) ===")
d = np.load(CACHE_DIR / "sm_L0.npz", allow_pickle=True)
lambdas = d["lambdas"]
train_ids = list(range(17))
test_ids = list(range(17, 25))
for K, H in [(4, 2), (4, 4), (8, 2), (8, 4)]:
    r2 = ar1_on(lambdas, train_ids, test_ids, K, H, 8)
    print(f"  K={K:2d} H={H}  AR(1)_R2={r2:+.4f}")

# L1a, L1d
for iface in ["L1a", "L1d"]:
    path = SM_DIR / f"{iface}_traj.npz"
    if not path.exists():
        print(f"\n=== {iface} — no traj.npz ===")
        continue
    print(f"\n=== {iface} ===")
    d = np.load(path, allow_pickle=True)
    lambdas = d["lambdas"]
    train_ids = d["train_ids"].tolist()
    test_ids = d["test_ids"].tolist()
    for K, H in [(4, 2), (4, 4), (8, 2), (8, 4)]:
        r2 = ar1_on(lambdas, train_ids, test_ids, K, H, 8)
        print(f"  K={K:2d} H={H}  AR(1)_R2={r2:+.4f}")
        
# добавляется в test_ar1.py
print("\n=== L0 shuffled (within-prompt) ===")
rng = np.random.default_rng(42)
lambdas_shuf = []
for lam in d["lambdas"]:
    arr = np.asarray(lam, dtype=float).copy()
    rng.shuffle(arr)
    lambdas_shuf.append(arr)
for K, H in [(4, 2), (4, 4), (8, 2), (8, 4)]:
    r2 = ar1_on(lambdas_shuf, list(range(17)), list(range(17, 25)), K, H, 8)
    print(f"  K={K:2d} H={H}  AR(1)_shuffled_R2={r2:+.4f}")