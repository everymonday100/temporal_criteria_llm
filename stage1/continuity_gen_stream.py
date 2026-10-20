"""continuity_gen_stream.py — Generate stream with hidden regimes and run Oracle Pass."""
import os
import re
import io
import pickle
import argparse
import contextlib
import traceback
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\continuity")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_EPISODES = 200
K_REGIME_SHIFT = 10

PROMPT_TEMPLATES = {
    "Direct": "{question}\n\nAnswer with a single number.\n####",
    "CoT": "{question}\n\nSolve step-by-step. End with the final number after '####'.\n####",
    "PAL": "{question}\n\nWrite a Python program to solve this. Only output the Python code in a ```python``` block."
}
MAX_NEW_TOKENS = {"Direct": 15, "CoT": 256, "PAL": 256}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--n_episodes", type=int, default=N_EPISODES)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def extract_number(text):
    """Extract the last number from a string."""
    if not text:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', str(text))
    if m:
        return float(m.group(1))
    # Fallback: last number in text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', str(text))
    if numbers:
        return float(numbers[-1])
    return None

def extract_code_block(text):
    """Extract the content of the first ```python ... ``` block."""
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    # Fallback: if no python block, try generic code block
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    return None

def execute_python(code_str):
    """Execute Python code and return the last printed number."""
    if not code_str:
        return None
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            exec(code_str, {"__builtins__": __builtins__})
        output = f.getvalue().strip()
        return extract_number(output)
    except Exception:
        return None

def get_question(row):
    for key in ["question", "prompt", "input"]:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"No question field. Keys: {list(row.keys())}")

def get_reference(row):
    if "target" in row and row["target"] is not None:
        try:
            return float(row["target"])
        except (ValueError, TypeError):
            pass
    if "answer" in row and row["answer"] is not None:
        return extract_number(row["answer"])
    return None

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"[Stream] Loading model...")
    tok = AutoTokenizer.from_pretrained("E:\\OllamaModels\\Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "E:\\OllamaModels\\Qwen2.5-3B", 
        dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = model.device

    print("[Stream] Loading datasets...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    try:
        gsm_hard = load_dataset("reasoning-machines/gsm-hard", split="train")
    except Exception as e:
        print(f"Fallback to socratic: {e}")
        gsm_hard = load_dataset("openai/gsm8k", "socratic", split="test")
    
    problems = []
    for i in range(args.n_episodes):
        regime = (i // K_REGIME_SHIFT) % 3
        if regime == 0:
            idx = i % len(gsm8k)
            problems.append({"id": i, "regime": "Direct-friendly", "question": get_question(gsm8k[idx]), "reference": get_reference(gsm8k[idx])})
        elif regime == 1:
            idx = i % len(gsm_hard)
            problems.append({"id": i, "regime": "CoT-friendly", "question": get_question(gsm_hard[idx]), "reference": get_reference(gsm_hard[idx])})
        else:
            idx = (i + 50) % len(gsm_hard)
            problems.append({"id": i, "regime": "PAL-friendly", "question": get_question(gsm_hard[idx]), "reference": get_reference(gsm_hard[idx])})

    print(f"[Stream] Running Oracle Pass for {len(problems)} episodes...")
    results = []
    
    for i, ep in enumerate(problems):
        oracle_results = {}
        for strategy, template in PROMPT_TEMPLATES.items():
            prompt = template.format(question=ep["question"])
            inputs = tok(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS[strategy],
                    do_sample=False,
                    pad_token_id=tok.eos_token_id
                )
            
            response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Special handling for PAL: execute code!
            if strategy == "PAL":
                code = extract_code_block(response)
                pred = execute_python(code)
            else:
                pred = extract_number(response)
                
            correct = (pred is not None) and (ep["reference"] is not None) and (abs(pred - ep["reference"]) < 1e-4)
            oracle_results[strategy] = {
                "response": response,
                "pred": pred,
                "correct": correct
            }
        
        # Determine Oracle Best Strategy
        # Priority: Direct > PAL > CoT
        if oracle_results["Direct"]["correct"]:
            best_strategy = "Direct"
        elif oracle_results["PAL"]["correct"]:
            best_strategy = "PAL"
        else:
            best_strategy = "CoT"
            
        ep["oracle_best_strategy"] = best_strategy
        ep["oracle_results"] = oracle_results
        results.append(ep)
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(problems)}] done.")

    out_path = OUT_DIR / "continuity_stream.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"[Stream] Saved {len(results)} episodes to {out_path}")
    
    from collections import Counter
    strategies = Counter([r["oracle_best_strategy"] for r in results])
    print("\n[Stream] Oracle Best Strategy Distribution:")
    for s, count in strategies.items():
        print(f"  {s}: {count} ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()