"""run_ag.py — agency via retry/abstention on GSM8K, both models.

Modes:
  gen      — run model on GSM8K, generate first + N retries, cache
  analyze  — load cache, compute AUC, policies, bootstrap CIs, save JSON
  compare  — load both model caches, print side-by-side summary
  all      — gen + analyze (per model)

Cache layout:
  data/stage1/cache/ag_L0.npz          (Qwen)
  data/stage1/cache/ag_L0_mamba2.npz   (Mamba2)

Output:
  data/stage1/ag/qwen.json
  data/stage1/ag/mamba2.json
  data/stage1/ag/comparison.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import (
    generate_segment, warmup_signals, extract_answer, majority,
)
from common.datasets import load_gsm8k, gsm8k_prompt
from common.stats import paired_bootstrap, auc_de_long
from common.io import ensure_dir

CACHE_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")
OUT_DIR   = Path(r"E:\OllamaModels\temporal_llm\data\stage1\ag")

N_DEFAULT   = 400
WARMUP      = 20
MAX_NEW     = 290
T_FIRST     = 0.2
T_RETRY     = 0.6
N_RETRY     = 3
SEED        = 42
BOOTSTRAP_B = 2000

RETRY_FRACS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gen", "analyze", "compare", "all"],
                   default="all")
    p.add_argument("--model", choices=["qwen", "mamba2"], default="qwen")
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _sfx(model):
    return "" if model == "qwen" else f"_{model}"


# ----------------------------------------------------------------------
# GEN
# ----------------------------------------------------------------------
def run_gen(args):
    sfx = _sfx(args.model)
    cache_path = CACHE_DIR / f"ag_L0{sfx}.npz"
    ckpt_path = CACHE_DIR / f"ag_L0{sfx}.jsonl"

    if cache_path.exists() and not args.overwrite:
        print(f"[gen {args.model}] cache exists, skipping: {cache_path}")
        return

    # --- Load checkpoint if present
    done = {}   # i -> record
    if ckpt_path.exists() and not args.overwrite:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "error" not in r and "i" in r:
                        done[r["i"]] = r
                except Exception:
                    continue
        print(f"[gen {args.model}] resuming from {len(done)} cached problems")

    problems = load_gsm8k(args.n)
    print(f"[gen {args.model}] {len(problems)} GSM8K problems")

    tok, model = load_model(args.model)
    need_attn = (args.model == "qwen")

    warm_keys = None
    warm_rows = []
    first_correct = []
    multi_correct = []
    first_pred = []
    multi_pred = []
    refs = []

    t0 = time.time()
    n_new = 0
    for k, p in enumerate(problems):
        i = p["i"]

        # --- Skip if already cached
        if i in done:
            r = done[i]
            s = r["warm_signals"]
            if warm_keys is None:
                warm_keys = list(s.keys())
            warm_rows.append(s)
            first_correct.append(r["first_correct"])
            multi_correct.append(r["multi_correct"])
            first_pred.append(r["first_pred"])
            multi_pred.append(r["multi_pred"])
            refs.append(r["reference"])
            continue

        n_new += 1
        prompt = gsm8k_prompt(p["question"])

        s = warmup_signals(
            model, tok, prompt,
            warmup=WARMUP, T=T_FIRST,
            seed=args.seed + i,
            need_attn=need_attn,
        )
        if warm_keys is None:
            warm_keys = list(s.keys())

        first = generate_segment(
            model, tok, prompt,
            n_tokens=MAX_NEW, T=T_FIRST,
            seed=args.seed + i + 1,
            return_hidden=False, return_attn=False, return_scores=False,
        )
        first_ans = extract_answer(first["text"])
        first_ok = int(first_ans == p["reference"])

        retries = []
        for j in range(N_RETRY):
            r = generate_segment(
                model, tok, prompt,
                n_tokens=MAX_NEW, T=T_RETRY,
                seed=args.seed + i + 100 + j,
                return_hidden=False, return_attn=False, return_scores=False,
            )
            retries.append(extract_answer(r["text"]))
        maj = majority([first_ans] + retries)
        multi_ok = int(maj == p["reference"])

        warm_rows.append(s)
        first_correct.append(first_ok)
        multi_correct.append(multi_ok)
        first_pred.append(first_ans)
        multi_pred.append(maj)
        refs.append(p["reference"])

        # --- Save this problem to checkpoint immediately
        rec = {
            "i": i,
            "warm_signals": {k2: float(v) for k2, v in s.items()},
            "first_correct": int(first_ok),
            "multi_correct": int(multi_ok),
            "first_pred": str(first_ans),
            "multi_pred": str(maj),
            "reference": str(p["reference"]),
        }
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        elapsed = time.time() - t0
        print(f"  [{k+1:3d}/{len(problems)}] "
              f"P1={s['P1_mean']:.3f}  "
              f"first={'v' if first_ok else 'x'}  "
              f"multi={'v' if multi_ok else 'x'}  "
              f"({elapsed:.0f}s, {elapsed/max(1,n_new):.1f}s/prob new)")

    ensure_dir(CACHE_DIR)
    data = {f"sig_{key}": np.array([s[key] for s in warm_rows])
            for key in warm_keys}
    data.update({
        "first_correct":  np.array(first_correct),
        "multi_correct":  np.array(multi_correct),
        "first_pred":     np.array(first_pred, dtype=object),
        "multi_pred":     np.array(multi_pred, dtype=object),
        "reference":      np.array(refs, dtype=object),
        "model":          np.array([args.model], dtype=object),
        "n_problems":     len(problems),
    })
    np.savez(cache_path, **data)
    print(f"[gen {args.model}] saved {cache_path}")
    print(f"[gen {args.model}] problems processed fresh this run: {n_new}")


# ----------------------------------------------------------------------
# Policy helpers
# ----------------------------------------------------------------------
def apply_signal_route(conf, first_correct, multi_correct, frac):
    """Retry bottom `frac` of conf (lowest confidence first)."""
    n = len(conf)
    k = max(1, int(round(frac * n)))
    order = np.argsort(conf)  # lowest confidence first
    retry_mask = np.zeros(n, dtype=bool)
    retry_mask[order[:k]] = True
    out = first_correct.copy()
    out[retry_mask] = multi_correct[retry_mask]
    return out, retry_mask


def apply_random_route(first_correct, multi_correct, frac, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(first_correct)
    k = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    retry_mask = np.zeros(n, dtype=bool)
    retry_mask[idx] = True
    out = first_correct.copy()
    out[retry_mask] = multi_correct[retry_mask]
    return out, retry_mask


# ----------------------------------------------------------------------
# ANALYZE
# ----------------------------------------------------------------------
def _orient_auc(x, y_wrong, name):
    """AUC for predicting 'wrong'. Oriented so that lower P1 -> wrong."""
    if np.isnan(x).any() or y_wrong.sum() == 0 or y_wrong.sum() == len(y_wrong):
        return None
    xa = -x if "P1" in name else x
    auc, lo, hi = auc_de_long(y_wrong, xa)
    return {"auc": auc, "ci_low": lo, "ci_high": hi}


def run_analyze(args):
    sfx = _sfx(args.model)
    cache_path = CACHE_DIR / f"ag_L0{sfx}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Run --mode gen first: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    n = len(data["first_correct"])
    valid = (data["first_correct"] >= 0) & (data["multi_correct"] >= 0)
    idx = np.where(valid)[0]
    print(f"[analyze {args.model}] {len(idx)} valid problems of {n}")

    # Pull signals
    sig_keys = [k for k in data.files if k.startswith("sig_")]
    sig_keys.sort()
    signals = {k.replace("sig_", ""): data[k][idx] for k in sig_keys}

    first = data["first_correct"][idx].astype(int)
    multi = data["multi_correct"][idx].astype(int)
    wrong = 1 - first

    # ---- Fixed baselines
    acc_first = float(first.mean())
    acc_multi = float(multi.mean())
    oracle = np.where(first == 1, first, multi)
    acc_oracle = float(oracle.mean())

    print(f"\n[fixed baselines {args.model}]")
    print(f"  fixed_single  acc={acc_first:.4f}")
    print(f"  fixed_multi   acc={acc_multi:.4f}")
    print(f"  oracle_route  acc={acc_oracle:.4f}")

    # ---- AUC per signal
    print(f"\n[signal AUC for 'first answer wrong']")
    aucs = {}
    for name, x in signals.items():
        r = _orient_auc(x, wrong, name)
        if r is None:
            continue
        aucs[name] = r
        print(f"  {name:<15}  AUC={r['auc']:.4f} "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    # ---- Pick threshold on calib split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(idx))
    n_calib = len(idx) // 2
    calib_idx = perm[:n_calib]
    test_idx  = perm[n_calib:]
    print(f"\n[split] calib={len(calib_idx)}  test={len(test_idx)}")

    # ---- Use P1_mean_20 as primary policy signal
    primary = None
    if "P1_mean_20" in signals:
        primary = "P1_mean_20"
    else:
        # Fall back to any signal with non-trivial variance,
        # preferring P1-like channels
        candidates = []
        for name, x in signals.items():
            if np.std(x) < 1e-9:
                continue  # constant channel, skip
            auc = aucs.get(name, {}).get("auc", 0.5)
            score = abs(auc - 0.5)
            candidates.append((score, name))
        if candidates:
            candidates.sort(reverse=True)
            primary = candidates[0][1]
    if primary is None:
        print("[warn] no usable primary signal (all constant)")
        return
    print(f"[primary signal] {primary}  "
          f"(non-constant, |AUC-0.5|={abs(aucs.get(primary,{}).get('auc',0.5)-0.5):.3f})")
    sig = signals[primary]

    sig_cal = sig[calib_idx]
    first_cal = first[calib_idx]
    multi_cal = multi[calib_idx]

    best_frac, best_acc = None, -1.0
    for frac in RETRY_FRACS:
        rout, _ = apply_signal_route(sig_cal, first_cal, multi_cal, frac)
        a = rout.mean()
        if a > best_acc:
            best_acc = a
            best_frac = frac
    print(f"[threshold picked on calibration]  frac={best_frac}  "
          f"calib_acc={best_acc:.4f}")

    # ---- Test evaluation
    sig_te = sig[test_idx]
    first_te = first[test_idx]
    multi_te = multi[test_idx]

    acc_first_te = float(first_te.mean())
    acc_multi_te = float(multi_te.mean())
    acc_oracle_te = float(np.where(first_te == 1, first_te, multi_te).mean())

    signal_correct, signal_mask = apply_signal_route(
        sig_te, first_te, multi_te, best_frac)
    random_correct, random_mask = apply_random_route(
        first_te, multi_te, best_frac)

    acc_signal = float(signal_correct.mean())
    acc_random = float(random_correct.mean())

    print(f"\n[test evaluation {args.model}]")
    print(f"  fixed_single     acc={acc_first_te:.4f}")
    print(f"  fixed_multi      acc={acc_multi_te:.4f}")
    print(f"  oracle_route     acc={acc_oracle_te:.4f}")
    print(f"  signal_route     acc={acc_signal:.4f}  "
          f"(retry rate {best_frac:.2f})")
    print(f"  random_route     acc={acc_random:.4f}  "
          f"(retry rate {best_frac:.2f})")

    d1, lo1, hi1 = paired_bootstrap(signal_correct, first_te)
    d2, lo2, hi2 = paired_bootstrap(signal_correct, random_correct)
    d3, lo3, hi3 = paired_bootstrap(signal_correct, multi_te)
    print(f"\n[paired bootstrap on test]")
    print(f"  signal_route vs fixed_single   Δ={d1:+.4f}  "
          f"[{lo1:+.4f},{hi1:+.4f}]")
    print(f"  signal_route vs random_route   Δ={d2:+.4f}  "
          f"[{lo2:+.4f},{hi2:+.4f}]")
    print(f"  signal_route vs fixed_multi    Δ={d3:+.4f}  "
          f"[{lo3:+.4f},{hi3:+.4f}]")

    # ---- Primary AUC on full data
    auc_primary = aucs.get(primary, {}).get("auc", float("nan"))
    auc_primary_lo = aucs.get(primary, {}).get("ci_low", float("nan"))

    # ---- Verdict
    h1 = auc_primary >= 0.65
    h2 = (d2 > 0) and (lo2 > 0)
    h3 = (d1 >= 0.02) and (lo1 > 0)
    print(f"\n=== VERDICT ({args.model}) ===")
    print(f"  H1 signal AUC >= 0.65:            "
          f"{'PASS' if h1 else 'fail'}  "
          f"AUC={auc_primary:.4f}")
    print(f"  H2 signal > random at same rate:  "
          f"{'PASS' if h2 else 'fail'}  "
          f"Δ={d2:+.4f} [{lo2:+.4f},{hi2:+.4f}]")
    print(f"  H3 signal > fixed_single by 0.02: "
          f"{'PASS' if h3 else 'fail'}  "
          f"Δ={d1:+.4f} [{lo1:+.4f},{hi1:+.4f}]")

    if h3 and h2:
        tag = "retry_agency_positive"
    elif h2 and not h3:
        tag = "signal_actionable_but_weak"
    elif h1 and not h2:
        tag = "signal_has_direction_not_selective"
    else:
        tag = "no_agency_through_retry"
    print(f"\n  TAG: {tag}")

    out = {
        "model": args.model,
        "n_total": int(len(idx)),
        "n_calib": int(len(calib_idx)),
        "n_test":  int(len(test_idx)),
        "primary_signal": primary,
        "retry_frac": float(best_frac),
        "aucs": {k: {kk: float(vv) for kk, vv in v.items()}
                 for k, v in aucs.items()},
        "test": {
            "fixed_single": acc_first_te,
            "fixed_multi":  acc_multi_te,
            "oracle_route": acc_oracle_te,
            "signal_route": acc_signal,
            "random_route": acc_random,
        },
        "deltas": {
            "signal_vs_single": {"mean": d1, "ci_low": lo1, "ci_high": hi1},
            "signal_vs_random": {"mean": d2, "ci_low": lo2, "ci_high": hi2},
            "signal_vs_multi":  {"mean": d3, "ci_low": lo3, "ci_high": hi3},
        },
        "checks": {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3)},
        "tag": tag,
    }
    ensure_dir(OUT_DIR)
    out_path = OUT_DIR / f"{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


# ----------------------------------------------------------------------
# COMPARE
# ----------------------------------------------------------------------
def run_compare(args):
    results = {}
    for model in ("qwen", "mamba2"):
        p = OUT_DIR / f"{model}.json"
        if not p.exists():
            print(f"[compare] missing {p} — run --mode analyze --model {model}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            results[model] = json.load(f)

    if len(results) < 2:
        print("[compare] need both qwen and mamba2 analyses")
        return

    q = results["qwen"]
    m = results["mamba2"]

    print("\n=== AGENCY COMPARISON ===")
    print(f"  {'metric':<30} {'qwen':>12} {'mamba2':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for key in ("n_total", "n_test"):
        print(f"  {key:<30} {q[key]:>12} {m[key]:>12}")

    # baseline accuracies
    for key in ("fixed_single", "fixed_multi", "oracle_route",
                "signal_route", "random_route"):
        print(f"  {key:<30} "
              f"{q['test'][key]:>12.4f} {m['test'][key]:>12.4f}")

    # primary AUC
    q_auc = q["aucs"][q["primary_signal"]]["auc"]
    m_auc = m["aucs"][m["primary_signal"]]["auc"]
    print(f"  {'AUC(primary)':<30} {q_auc:>12.4f} {m_auc:>12.4f}")

    # deltas
    for key in ("signal_vs_single", "signal_vs_random"):
        qd = q["deltas"][key]["mean"]
        md = m["deltas"][key]["mean"]
        print(f"  {key + ' (Δ)':<30} {qd:>+12.4f} {md:>+12.4f}")

    print(f"\n  tags: qwen={q['tag']}  mamba2={m['tag']}")

    out = {"qwen": q, "mamba2": m}
    cmp_path = OUT_DIR / "comparison.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {cmp_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dir(CACHE_DIR)
    ensure_dir(OUT_DIR)

    if args.mode == "compare":
        run_compare(args)
        return

    if args.mode in ("gen", "all"):
        run_gen(args)
    if args.mode in ("analyze", "all"):
        run_analyze(args)


if __name__ == "__main__":
    main()