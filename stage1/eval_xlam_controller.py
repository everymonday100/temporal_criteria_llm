"""eval_xlam_controller.py — Evaluate contrastive controller on xLAM tool-calling (FIXED)."""
import pickle
import torch
import json
import re
import numpy as np
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.controller_model import ControllerModel, LatentController
import torch.nn as nn

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")
MODEL_PATH = r"E:\OllamaModels\xLAM-2-3b-fc-r"
CKPT_PATH = OUT_DIR / "ctrl_xlam_contrast.pt"

def parse_tool_call(text):
    """Extract tool name from generated response."""
    # Try JSON array format first
    try:
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if isinstance(data, list) and len(data) > 0 and "name" in data[0]:
                return data[0]["name"]
    except:
        pass
    # Try JSON object
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if "name" in data:
                return data["name"]
    except:
        pass
    # Try plain text mentions
    text_lower = text.lower()
    if "weather" in text_lower or "get_weather" in text_lower:
        return "get_weather"
    if "calculate" in text_lower:
        return "calculate"
    if "search" in text_lower:
        return "search_web"
    return None

def load_contrastive_controller(ctrl_model, device):
    """Load the contrastive-trained controller."""
    ckpt = torch.load(CKPT_PATH, map_location=device)
    ctrl_model.controller.load_state_dict(ckpt["controller_state"])
    ctrl_model.w_in.load_state_dict(ckpt["w_in_state"])
    print(f"[Load] Contrastive controller loaded from {CKPT_PATH}")

def randomize_controller(ctrl_model, device):
    """Reinitialize controller with random weights."""
    ctrl_model.controller = LatentController(ctrl_model.hidden).to(device)
    ctrl_model.w_in = nn.Linear(64, ctrl_model.hidden, bias=False).to(device)
    nn.init.normal_(ctrl_model.w_in.weight, std=0.02)
    print("[Load] Random controller initialized")

def run_condition(name, ctrl_model, tok, test_data, device, record_c=False, debug_first_n=3):
    """Run evaluation for one condition."""
    correct_list = []  # Храним per-example результаты
    total = len(test_data)
    tool_c_vectors = {"get_weather": [], "calculate": [], "search_web": []}
    
    for i, item in enumerate(test_data):
        messages = [
            {"role": "system", 
             "content": "You are a helpful assistant that can use tools. "
                       "You are developed by Salesforce xLAM team."},
            {"role": "user", "content": item["prompt"]}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, 
                                        add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            out = ctrl_model.generate(
                input_ids,
                max_new_tokens=50, 
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        response = tok.decode(out[0][input_ids.shape[1]:], 
                             skip_special_tokens=True)
        
        # Debug first N examples
        if i < debug_first_n:
            print(f"\n  [DEBUG {name} example {i+1}]")
            print(f"    Tool: {item['tool']}")
            print(f"    Response: {response[:200]}...")
            print(f"    Parsed: {parse_tool_call(response)}")
        
        pred = parse_tool_call(response)
        is_correct = (pred == item["tool"])
        correct_list.append(int(is_correct))
        
        if record_c and ctrl_model._last_c_t is not None:
            c_vec = ctrl_model._last_c_t.detach().cpu().numpy().squeeze()
            tool_c_vectors[item["tool"]].append(c_vec)
        
        if (i + 1) % 20 == 0:
            print(f"  [{name}] {i+1}/{total}  acc={sum(correct_list)/(i+1):.3f}")
    
    acc = sum(correct_list) / total
    print(f"[{name}] FINAL acc = {acc:.4f} ({sum(correct_list)}/{total})")
    return {
        "acc": acc, 
        "correct": correct_list,  # Просто возвращаем список, который мы уже посчитали
        "tool_c_vectors": tool_c_vectors if record_c else None
    }

def bootstrap_diff(a, b, n_boot=2000, seed=42):
    """Paired bootstrap for difference in means."""
    rng = np.random.default_rng(seed)
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    return (float(a.mean() - b.mean()), 
            float(np.percentile(diffs, 2.5)), 
            float(np.percentile(diffs, 97.5)))

def direction_analysis(tool_vectors):
    """Compute pairwise cosine similarities between tool directions."""
    means = {}
    for tool, vecs in tool_vectors.items():
        if vecs:
            v = np.stack(vecs)
            m = v.mean(axis=0)
            means[tool] = m / (np.linalg.norm(m) + 1e-8)
    
    if len(means) == 3:
        tools = list(means.keys())
        print("\n[Direction Analysis]")
        for i in range(len(tools)):
            for j in range(i+1, len(tools)):
                cos = np.dot(means[tools[i]], means[tools[j]])
                print(f"  cos({tools[i][:8]:8s}, {tools[j][:8]:8s}) = {cos:.4f}")
        return means
    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto", 
        trust_remote_code=True
    )
    base.config.use_cache = True
    
    print("[Init] Wrapping with ControllerModel...")
    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(device)
    ctrl_model.controller.to(device)
    ctrl_model.w_in.to(device)
    
    with open(OUT_DIR / "test.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    print(f"[Eval] Running on {len(test_data)} test examples...")
    
    # === Condition A: Frozen xLAM (no controller) ===
    print("\n=== A: Frozen xLAM ===")
    ctrl_model.remove_hook()
    res_A = run_condition("A_frozen", ctrl_model, tok, test_data, device, 
                         record_c=False, debug_first_n=3)
    
    # === Condition B: Contrastive Controller ===
    print("\n=== B: Contrastive Controller ===")
    load_contrastive_controller(ctrl_model, device)
    ctrl_model.reinstall_hook()
    ctrl_model.controller.eval()
    ctrl_model.w_in.eval()
    res_B = run_condition("B_contrast", ctrl_model, tok, test_data, device,
                         record_c=True, debug_first_n=3)
    
    # === Condition C: Random Controller ===
    print("\n=== C: Random Controller ===")
    randomize_controller(ctrl_model, device)
    ctrl_model.reinstall_hook()
    res_C = run_condition("C_random", ctrl_model, tok, test_data, device,
                         record_c=False, debug_first_n=3)
    
    # === Direction Analysis ===
    if res_B["tool_c_vectors"]:
        means = direction_analysis(res_B["tool_c_vectors"])
    
    # === Paired Bootstrap ===
    print("\n=== Paired Bootstrap (B=2000) ===")
    comparisons = [("B", res_B), ("C", res_C)]
    for name_x, res_x in comparisons:
        for name_y, res_y in [("A", res_A)]:
            d, lo, hi = bootstrap_diff(res_x["correct"], res_y["correct"])
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            print(f"  {name_x} - {name_y}: {d:+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}")
    
    # === Final Verdict ===
    print("\n" + "="*62)
    print("X-LAM CONTRASTIVE EVALUATION")
    print("="*62)
    print(f"  A (frozen):   {res_A['acc']:.4f}")
    print(f"  B (contrast): {res_B['acc']:.4f}")
    print(f"  C (random):   {res_C['acc']:.4f}")
    
    if res_B['acc'] > res_A['acc']:
        d, lo, hi = bootstrap_diff(res_B["correct"], res_A["correct"])
        if lo > 0:
            print(f"  ✅ Contrastive controller SIGNIFICANTLY improved accuracy!")
            print(f"     Δ = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        else:
            print(f"  ⚠️  Improvement not statistically significant")
    elif res_B['acc'] < res_A['acc']:
        print(f"  ❌ Contrastive controller degraded accuracy")
    else:
        print(f"  ⚠️  No change in accuracy (epiphenomenal separation)")
    print("="*62)
    
    # Save results
    summary = {
        "A_frozen": res_A["acc"],
        "B_contrast": res_B["acc"],
        "C_random": res_C["acc"],
        "n_test": len(test_data),
    }
    with open(OUT_DIR / "eval_contrast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {OUT_DIR / 'eval_contrast_summary.json'}")

if __name__ == "__main__":
    main()