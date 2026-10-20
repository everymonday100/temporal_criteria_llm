"""eval_contrastive.py — Four-condition evaluation with c_t extraction."""
import re
import json
import torch
import numpy as np
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


def run_condition(name, model_to_gen, tok, problems, device, 
                  use_gsm_hard_prompt=False, record_c=False, ctrl_model_ref=None):
    correct = 0
    per_problem = []
    c_vectors = []
    
    for i, p in enumerate(problems):
        # КРИТИЧЕСКИ ВАЖНО: Разные промпты для разных условий
        if use_gsm_hard_prompt:
            prompt = gsm_hard_prompt(p["question"])
        else:
            prompt = f"Question: {p['question']}\nAnswer:"
        
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_to_gen.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Извлекаем c_t только если передана ссылка на ctrl_model и включен record_c
        c_t = None
        if record_c and ctrl_model_ref is not None and hasattr(ctrl_model_ref, '_last_c_t') and ctrl_model_ref._last_c_t is not None:
            c_t = ctrl_model_ref._last_c_t.detach().cpu().numpy().squeeze()
            
        pred = extract_number(response)
        ok = pred is not None and abs(pred - float(p["reference"])) < 1e-4
        correct += int(ok)
        per_problem.append(int(ok))
        
        if c_t is not None:
            c_vectors.append(c_t)
            
        if (i + 1) % 40 == 0:
            print(f"  [{name}] {i+1}/{len(problems)}  acc={correct/(i+1):.3f}")
            
    acc = correct / len(problems)
    print(f"[{name}] FINAL acc = {acc:.4f} ({correct}/{len(problems)})")
    return {"name": name, "acc": acc, "per_problem": per_problem, "c_vectors": c_vectors}


def bootstrap_diff(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, base = load_model("qwen")
    base.config.use_cache = True

    print("[Eval] Loading 200 held-out GSM-HARD problems [1000:1200]...")
    problems = load_gsm_hard(n=1200)[1000:1200]

    # Инициализируем ctrl_model ТОЛЬКО для условий B и D
    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(device)
    ctrl_model.controller.to(device)
    ctrl_model.w_in.to(device)
    ctrl_model.controller.eval()
    ctrl_model.w_in.eval()

    # =========================================================================
    # === Condition A: frozen base (ЧИСТАЯ модель, БЕЗ LoRA-обертки) ===
    # =========================================================================
    print("\n=== A: frozen base ===")
    res_A = run_condition(
        "A_frozen", 
        base,  # <-- ПЕРЕДАЕМ ЧИСТУЮ base, а не ctrl_model!
        tok, problems, device, 
        use_gsm_hard_prompt=True  # <-- Правильный промпт для замороженной модели
    )

    # =========================================================================
    # === Condition B: LoRA-only (contrastive LoRA, controller disabled) ===
    # =========================================================================
    print("\n=== B: LoRA-only (contrastive LoRA, controller disabled) ===")
    load_weights(ctrl_model, OUT_DIR / "ctrl_qwen_contrastive.pt", device)
    ctrl_model.remove_hook() # Отключаем инъекцию контроллера
    
    res_B = run_condition(
        "B_lora_only", 
        ctrl_model.model,  # <-- Используем модель с загруженными весами LoRA
        tok, problems, device, 
        use_gsm_hard_prompt=False # <-- Промпт, на котором обучалась LoRA
    )

    # =========================================================================
    # === Condition D: contrastive trained controller ===
    # =========================================================================
    print("\n=== D: contrastive controller ===")
    ctrl_model.reinstall_hook() # Включаем инъекцию контроллера
    
    res_D = run_condition(
        "D_contrastive", 
        ctrl_model.model,  # <-- Модель с LoRA + активный контроллер
        tok, problems, device, 
        use_gsm_hard_prompt=False, 
        record_c=True, 
        ctrl_model_ref=ctrl_model # <-- Ссылка для извлечения _last_c_t
    )

    # =========================================================================
    # === c_t analysis ===
    # =========================================================================
    print("\n=== c_t diagnostics ===")
    c_vectors = res_D["c_vectors"]
    if len(c_vectors) >= 10:
        c_arr = np.stack(c_vectors)
        np.save(OUT_DIR / "contrastive_c_vectors.npy", c_arr)
        norms = np.linalg.norm(c_arr, axis=1, keepdims=True)
        c_normed = c_arr / (norms + 1e-8)
        cos_matrix = c_normed @ c_normed.T
        mask = ~np.eye(len(cos_matrix), dtype=bool)
        mean_cos = float(cos_matrix[mask].mean())
        print(f"[diag] mean pairwise cosine of c_t: {mean_cos:.4f}")

        corrects = np.array(res_D["per_problem"], dtype=bool)
        if corrects.sum() >= 2 and (~corrects).sum() >= 2:
            c_correct = c_normed[corrects]
            c_wrong = c_normed[~corrects]
            within_correct = (c_correct @ c_correct.T)[~np.eye(len(c_correct), dtype=bool)].mean()
            within_wrong = (c_wrong @ c_wrong.T)[~np.eye(len(c_wrong), dtype=bool)].mean()
            cross = (c_correct @ c_wrong.T).mean()
            print(f"[diag] within-correct: {within_correct:.4f}")
            print(f"[diag] within-wrong:   {within_wrong:.4f}")
            print(f"[diag] cross:          {cross:.4f}")

    # =========================================================================
    # === Paired bootstrap ===
    # =========================================================================
    print("\n=== Paired bootstrap (B=2000) ===")
    for name_x, res_x in [("D", res_D), ("B", res_B), ("A", res_A)]:
        for name_y, res_y in [("A", res_A), ("B", res_B)]:
            if name_x == name_y: continue
            d, lo, hi = bootstrap_diff(res_x["per_problem"], res_y["per_problem"])
            sig = "✓" if lo > 0 or hi < 0 else " "
            print(f"  {name_x} - {name_y}: {d:+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig}")

    summary = {
        "A_frozen": res_A["acc"],
        "B_lora_only": res_B["acc"],
        "D_contrastive": res_D["acc"],
        "n_test": len(problems),
    }
    with open(OUT_DIR / "eval_contrastive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {OUT_DIR / 'eval_contrastive_summary.json'}")

if __name__ == "__main__":
    main()