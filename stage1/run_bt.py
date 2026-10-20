"""run_bt.py — segment-level backtracking agency on GSM8K.

FIXES APPLIED:
1. Increased T_BASE to 0.7 to allow meaningful entropy variation.
2. Lowered confidence thresholds to match T=0.7 behavior.
3. Added structural check (digits/operators) as a hard backtracking trigger.
4. Fixed context joining to use newlines instead of spaces.
"""
import argparse
import json
import sys
import time
import re
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.generation import generate_segment, extract_answer, majority
from common.datasets import load_gsm8k, gsm8k_prompt

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\bt")
CKPT_DIR = OUT_DIR / "ckpt"

N_DEFAULT   = 400
N_SEG       = 5
SEG_TOKENS  = 80  # Увеличили до 80, 60 слишком мало для шага рассуждения
WARMUP_SEG  = 15

# --- КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ ЗДЕСЬ ---
T_BASE      = 0.7   # Повысили T, чтобы модель могла "сомневаться" и давать разные варианты
T_REGEN     = 0.8   # При регенерации даем еще чуть больше свободы
SEED        = 42
MAX_REGEN   = 2     # Уменьшили до 2, чтобы не застревать
MAX_BACK    = 2

# Пороги адаптированы под T=0.7. P1 редко бывает > 0.8 при такой температуре.
THETA_LOW   = 0.35  # Ниже этого - почти гарантированная галлюцинация или тупик
THETA_HIGH  = 0.55  # Ниже этого - модель колеблется, стоит перегенерировать

# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gen", "analyze", "both"], default="both")
    p.add_argument("--model", choices=["qwen", "mamba2"], default="qwen")
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--theta_low", type=float, default=THETA_LOW)
    p.add_argument("--theta_high", type=float, default=THETA_HIGH)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def _sfx(m):
    return "" if m == "qwen" else f"_{m}"

# ----------------------------------------------------------------------
def segment_confidence(tok, model, prompt, seed, temperature):
    """Generate one segment, return (text, signals, ids)."""
    out = generate_segment(
        model, tok, prompt,
        n_tokens=SEG_TOKENS, T=temperature, seed=seed,
        return_hidden=False, return_attn=False,
        return_scores=True,
    )
    text = out["text"]
    scores = out.get("scores")
    
    p1_vals, h_vals = [], []
    if scores is not None:
        for s in scores[:WARMUP_SEG]:
            probs = torch.softmax(s[0].float(), dim=-1)
            p1_vals.append(float(probs.max().item()))
            p = probs.clamp_min(1e-12)
            h_vals.append(float(-(p * p.log()).sum().item()))
            
    if not p1_vals:
        p1_mean, h_mean, h_slope = 0.5, 1.0, 0.0
    else:
        p1 = np.array(p1_vals)
        h = np.array(h_vals)
        k = np.arange(len(h))
        h_slope = float(np.polyfit(k, h, 1)[0]) if len(h) > 1 else 0.0
        p1_mean, h_mean = float(p1.mean()), float(h.mean())

    # --- НОВАЯ ЭВРИСТИКА ДЛЯ GSM8K ---
    # Если в сегменте нет цифр или знаков '=', это почти наверняка "вода"
    has_math = bool(re.search(r'\d', text) or '=' in text)
    
    return text, {
        "P1": p1_mean, 
        "H": h_mean, 
        "H_slope": h_slope,
        "has_math": has_math
    }, out["ids"]

# ----------------------------------------------------------------------
def solve_backtrack(model, tok, prompt, theta_low, theta_high, seed_i):
    """Return dict with accepted text, actions taken, budget used."""
    accepted_texts = []
    actions = []
    regen_used = 0
    back_used = 0
    step = 0
    max_steps = N_SEG + MAX_REGEN + MAX_BACK + 5  # safety buffer

    while len(accepted_texts) < N_SEG and step < max_steps:
        step += 1
        
        # FIX: Используем \n для корректного продолжения контекста
        cur_ctx = "\n".join(accepted_texts) + "\n" if accepted_texts else ""
        full = prompt + cur_ctx

        # При регенерации/бэктреке даем чуть более высокую температуру
        current_T = T_REGEN if (regen_used > 0 or back_used > 0) else T_BASE
        current_seed = seed_i + step * 100 + len(accepted_texts)

        text, sig, ids = segment_confidence(tok, model, full, current_seed, current_T)

        # --- ЛОГИКА ПРИНЯТИЯ РЕШЕНИЙ ---
        # 1. Жесткий триггер: нет математики в сегменте рассуждений
        if not sig["has_math"] and len(accepted_texts) > 0:
            if back_used < MAX_BACK:
                accepted_texts.pop()
                back_used += 1
                actions.append(("back_no_math", sig["P1"]))
                continue # Идем на следующую итерацию
            else:
                # Если бэктреки кончились, вынуждены принять, но пометим
                actions.append(("accept_forced_no_math", sig["P1"]))
                accepted_texts.append(text)
                continue

        # 2. Низкая уверенность -> Бэктрек (откат на шаг назад)
        if sig["P1"] < theta_low and back_used < MAX_BACK and len(accepted_texts) > 0:
            accepted_texts.pop()
            back_used += 1
            actions.append(("back_low_conf", sig["P1"]))
            
        # 3. Средняя уверенность -> Регенерация текущего шага
        elif sig["P1"] < theta_high and regen_used < MAX_REGEN:
            regen_used += 1
            actions.append(("regen", sig["P1"]))
            # Не добавляем в accepted_texts, цикл повторится с тем же len(accepted_texts)
            
        # 4. Высокая уверенность или исчерпан бюджет -> Принять
        else:
            accepted_texts.append(text)
            actions.append(("accept", sig["P1"]))

    return {
        "text": "\n".join(accepted_texts),
        "actions": actions,
        "n_regen": regen_used,
        "n_back": back_used,
        "n_segments": len(accepted_texts),
    }

# ----------------------------------------------------------------------
# (Функции run_gen и run_analyze остаются без изменений, 
#  кроме передачи новых параметров, если нужно)
# ----------------------------------------------------------------------
def run_gen(args):
    sfx = _sfx(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"bt{sfx}.npz"
    ckpt_path = CKPT_DIR / f"bt{sfx}.jsonl"

    if out_path.exists() and not args.overwrite:
        print(f"[bt {args.model}] cache exists, skipping")
        return

    done = {}
    if ckpt_path.exists() and not args.overwrite:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "error" not in r and "i" in r:
                        done[r["i"]] = r
                except Exception:
                    continue
        print(f"[bt {args.model}] resume from {len(done)} cached")

    problems = load_gsm8k(args.n)
    print(f"[bt {args.model}] {len(problems)} problems")
    tok, model = load_model(args.model)

    records = []
    t0 = time.time()
    n_new = 0
    for k, p in enumerate(problems):
        i = p["i"]
        if i in done:
            records.append(done[i])
            continue
        n_new += 1
        prompt = gsm8k_prompt(p["question"])
        try:
            res = solve_backtrack(model, tok, prompt,
                                  args.theta_low, args.theta_high,
                                  args.seed + i * 17)
            pred = extract_answer(res["text"])
            ok = int(pred == p["reference"])
            rec = {
                "i": i,
                "pred": pred,
                "reference": p["reference"],
                "correct": ok,
                "n_regen": res["n_regen"],
                "n_back": res["n_back"],
                "n_segments": res["n_segments"],
                "actions": [(a, round(float(v), 3) if isinstance(v, (int, float)) else v)
                            for a, v in res["actions"]],
            }
        except Exception as e:
            rec = {"i": i, "error": str(e)[:200]}
        records.append(rec)
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        elapsed = time.time() - t0
        if "error" not in rec:
            print(f"  [{k+1:3d}/{len(problems)}] "
                  f"correct={'v' if rec['correct'] else 'x'}  "
                  f"regen={rec['n_regen']}  back={rec['n_back']}  "
                  f"({elapsed:.0f}s, {elapsed/max(1,n_new):.1f}s/prob)")

    valid = [r for r in records if "error" not in r]
    n = len(valid)
    if n > 0:
        arr = {
            "correct":   np.array([r["correct"] for r in valid]),
            "n_regen":   np.array([r["n_regen"] for r in valid]),
            "n_back":    np.array([r["n_back"] for r in valid]),
            "reference": np.array([r["reference"] for r in valid], dtype=object),
            "pred":      np.array([r["pred"] for r in valid], dtype=object),
            "model":     np.array([args.model], dtype=object),
        }
        np.savez(out_path, **arr)
        print(f"[bt {args.model}] saved {out_path}")
        print(f"[bt {args.model}] accuracy = {arr['correct'].mean():.4f}")
        print(f"[bt {args.model}] mean regen = {arr['n_regen'].mean():.2f}")
        print(f"[bt {args.model}] mean back  = {arr['n_back'].mean():.2f}")


def run_analyze(args):
    sfx = _sfx(args.model)
    out_path = OUT_DIR / f"bt{sfx}.npz"
    if not out_path.exists():
        raise FileNotFoundError(f"Run --mode gen first: {out_path}")
    d = np.load(out_path, allow_pickle=True)
    correct = d["correct"]
    print(f"\n[bt {args.model} analyze]")
    print(f"  n = {len(correct)}")
    print(f"  bt accuracy:       {correct.mean():.4f}")
    print(f"  mean regen:        {d['n_regen'].mean():.2f}")
    print(f"  mean back:         {d['n_back'].mean():.2f}")

    ag_path = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")
    ag_file = ag_path / f"ag_L0{sfx}.npz"
    if ag_file.exists():
        a = np.load(ag_file, allow_pickle=True)
        n = min(len(a["first_correct"]), len(correct))
        first = a["first_correct"][:n]
        multi = a["multi_correct"][:n]
        print(f"\n  fixed_single:      {first.mean():.4f}")
        print(f"  fixed_multi:       {multi.mean():.4f}")
        print(f"  delta bt vs fixed: {correct[:n].mean() - first.mean():+.4f}")
        print(f"  delta bt vs multi: {correct[:n].mean() - multi.mean():+.4f}")
    else:
        print(f"  (no ag cache for baseline comparison)")


def main():
    args = parse_args()
    if args.mode in ("gen", "both"):
        run_gen(args)
    if args.mode in ("analyze", "both"):
        run_analyze(args)

if __name__ == "__main__":
    main()