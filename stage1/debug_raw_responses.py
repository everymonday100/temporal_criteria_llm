"""debug_raw_responses.py — Проверка сырых ответов для A, B и D."""
import torch
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.datasets import load_gsm_hard, gsm_hard_prompt
from common.controller_model import ControllerModel

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\controller")

def extract_number(text):
    if not text: return None
    m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', str(text))
    if m: return float(m.group(1))
    numbers = re.findall(r'-?\d+(?:\.\d+)?', str(text))
    if numbers: return float(numbers[-1])
    return None

def load_weights(ctrl_model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ctrl_model.controller.load_state_dict(ckpt["controller_state"])
    ctrl_model.w_in.load_state_dict(ckpt["w_in_state"])
    model_dict = ctrl_model.model.state_dict()
    for k, v in ckpt["lora_state"].items():
        if k in model_dict:
            model_dict[k] = v
    ctrl_model.model.load_state_dict(model_dict)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, base = load_model("qwen")
    base.config.use_cache = True
    
    problems = load_gsm_hard(n=1200)[1000:1003]  # Берем всего 3 задачи для теста
    
    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(device)
    ctrl_model.controller.to(device)
    ctrl_model.w_in.to(device)
    ctrl_model.controller.eval()
    ctrl_model.w_in.eval()

    print("=" * 80)
    print("=== CONDITION A: Frozen Base (gsm_hard_prompt) ===")
    print("=" * 80)
    for p in problems:
        prompt = gsm_hard_prompt(p["question"])
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = base.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Ref: {p['reference']} | Pred: {extract_number(response)}")
        print(f"Raw: {response[:300]}...\n")

    print("=" * 80)
    print("=== CONDITION B: LoRA-only (Question...\\nAnswer:) ===")
    print("=" * 80)
    load_weights(ctrl_model, OUT_DIR / "ctrl_qwen_contrastive.pt", device)
    ctrl_model.remove_hook()
    for p in problems:
        prompt = f"Question: {p['question']}\nAnswer:"
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = ctrl_model.model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Ref: {p['reference']} | Pred: {extract_number(response)}")
        print(f"Raw: {response[:300]}...\n")

    print("=" * 80)
    print("=== CONDITION D: Contrastive Controller (Question...\\nAnswer:) ===")
    print("=" * 80)
    ctrl_model.reinstall_hook()
    for p in problems:
        prompt = f"Question: {p['question']}\nAnswer:"
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = ctrl_model.model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Ref: {p['reference']} | Pred: {extract_number(response)}")
        print(f"Raw: {response[:300]}...\n")

if __name__ == "__main__":
    main()