"""stats.py — bootstrap, AUC, verdict rules."""
import numpy as np
from sklearn.metrics import roc_auc_score


SEED = 42
BOOTSTRAP_B = 2000


def paired_bootstrap(a, b, B=BOOTSTRAP_B, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, n)
        diffs[k] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def mean_ci(x, B=BOOTSTRAP_B, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    boots = np.empty(B)
    for k in range(B):
        boots[k] = x[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def auc_de_long(y_true, scores, B=BOOTSTRAP_B, seed=SEED):
    """AUC with a simple bootstrap CI (approximation to DeLong)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, sb = y_true[idx], scores[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        boots.append(roc_auc_score(yb, sb))
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


# ----------------------------------------------------------------------
# Verdict rules
# ----------------------------------------------------------------------
def verdict_sm(r2, centered, null_r2):
    """Self-model verdict."""
    if r2 >= 0.5 and centered >= 0.4 and null_r2 <= 0.1:
        return "positive"
    if r2 < 0.1 and null_r2 <= 0.1:
        return "T1_signal_weak"
    if null_r2 > 0.1:
        return "null_leak"
    return "T3_below_threshold"


def verdict_ag(auc, delta_signal_random, ci_low_sr,
               delta_signal_single, ci_low_ss,
               hindsight_gap):
    """Agency verdict."""
    if hindsight_gap < 0.05:
        return "T4_task_limited"
    h1 = auc >= 0.65
    h2 = (delta_signal_random > 0) and (ci_low_sr > 0)
    h3 = (delta_signal_single >= 0.02) and (ci_low_ss > 0)
    if h1 and h2 and h3:
        return "positive"
    if not h1:
        return "T1_signal_weak"
    if h1 and not h2:
        return "T2_not_actionable"
    if h2 and not h3:
        return "T3_below_threshold"
    return "null"


def verdict_bd(auc, ci_low, threshold=0.85):
    if auc >= threshold and ci_low > threshold - 0.05:
        return "positive"
    if auc < 0.65:
        return "T1_signal_weak"
    return "T3_below_threshold"


def verdict_ct(delta_vs_features, ci_low_df,
               delta_vs_shuffled, ci_low_ds,
               threshold=0.02):
    if (delta_vs_features >= threshold and ci_low_df > 0 and
            delta_vs_shuffled >= threshold and ci_low_ds > 0):
        return "positive"
    if delta_vs_features < 0.005 and delta_vs_shuffled < 0.005:
        return "T1_signal_weak"
    if (delta_vs_features > 0 and delta_vs_shuffled > 0 and
            (ci_low_df <= 0 or ci_low_ds <= 0)):
        return "T2_not_actionable"
    if delta_vs_features > 0 and delta_vs_features < threshold:
        return "T3_below_threshold"
    return "null"