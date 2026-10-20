"""eval_quick_D.py — Quick eval of D on the SHORT prompt."""
import re
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model import load_model
from common.datasets import load_gsm_hard
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

    print("[Eval] Loading 200 held-out GSM-HARD problems...")
    problems = load_gsm_hard(n=1200)[1000:1200]

    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(device)
    ctrl_model.controller.to(device)
    ctrl_model.w_in.to(device)
    ctrl_model.controller.eval()
    ctrl_model.w_in.eval()

    load_weights(ctrl_model, OUT_DIR / "ctrl_qwen_contrastive.pt", device)
    ctrl_model.reinstall_hook()

    correct = 0
    per_problem = []
    c_vectors = []

    print("[Eval] Running D on SHORT prompt...")
    for i, p in enumerate(problems):
        prompt = f"Question: {p['question']}\nAnswer:"
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = ctrl_model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        if ctrl_model._last_c_t is not None:
            c_vectors.append(ctrl_model._last_c_t.detach().cpu().numpy().squeeze())
            
        pred = extract_number(response)
        ok = pred is not None and abs(pred - float(p["reference"])) < 1e-4
        correct += int(ok)
        per_problem.append(int(ok))
        
        if (i + 1) % 40 == 0:
            print(f"  [D] {i+1}/200  acc={correct/(i+1):.3f}")

    acc = correct / 200
    print(f"\n[D_contrastive] FINAL acc = {acc:.4f} ({correct}/200)")

    if len(c_vectors) >= 10:
        c_arr = np.stack(c_vectors)
        norms = np.linalg.norm(c_arr, axis=1, keepdims=True)
        c_normed = c_arr / (norms + 1e-8)
        cos_matrix = c_normed @ c_normed.T
        mask = ~np.eye(len(cos_matrix), dtype=bool)
        mean_cos = float(cos_matrix[mask].mean())
        print(f"[diag] mean pairwise cosine of c_t: {mean_cos:.4f}")

if __name__ == "__main__":
    main()