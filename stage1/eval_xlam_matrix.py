"""eval_xlam_matrix.py — fair matrix: A frozen / B contrast-ctrl / C rand-ctrl /
D LoRA-only. Two metrics: exact coarse-name match and canonicalized semantic
match (discloses the frozen-model vocab-mismatch artifact)."""
import pickle, json, re
import torch, numpy as np, torch.nn as nn
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.controller_model import ControllerModel, LatentController

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")
MODEL_PATH = r"E:\OllamaModels\xLAM-2-3b-fc-r"
SYS = ("You are a helpful assistant that can use tools. "
       "You are developed by Salesforce xLAM team.")
CANON = {
    "multiply": "calculate", "add": "calculate", "subtract": "calculate",
    "divide": "calculate", "pow": "calculate", "sqrt": "calculate",
    "sum": "calculate", "mean": "calculate",
    "get_web_search_results": "search_web", "web_search": "search_web",
    "browse": "search_web",
    "get_weather": "get_weather", "weather": "get_weather",
}

def parse_name(text):
    for pat in (r'\[.*?\]', r'\{.*?\}'):
        m = re.search(pat, text, re.DOTALL)
        if not m: continue
        try:
            d = json.loads(m.group(0))
            if isinstance(d, list) and d and "name" in d[0]: return d[0]["name"]
            if isinstance(d, dict) and "name" in d: return d["name"]
        except Exception:
            continue
    return None

def load_lora(cm, path, device):
    st = torch.load(path, map_location=device)
    md = cm.model.state_dict()
    for k, v in st.items():
        if k in md: md[k] = v
    cm.model.load_state_dict(md)

def run(name, cm, gen_model, tok, data, device):
    ex, cn = [], []
    for i, item in enumerate(data):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]
        with torch.no_grad():
            out = gen_model.generate(ids, max_new_tokens=50, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        raw = parse_name(resp)
        ex.append(int(raw == item["tool"]))
        cn.append(int(CANON.get(raw, raw) == item["tool"]))
        if (i + 1) % 30 == 0:
            print(f"  [{name}] {i+1}/{len(data)} exact={np.mean(ex):.3f} "
                  f"canon={np.mean(cn):.3f}")
    print(f"[{name}] exact={np.mean(ex):.4f}  canon={np.mean(cn):.4f}")
    return {"exact": ex, "canon": cn}

def boot(a, b, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.array(a, float), np.array(b, float)
    d = [a[rng.integers(0, len(a), len(a))].mean() -
         b[rng.integers(0, len(b), len(b))].mean() for _ in range(B)]
    return (float(a.mean()-b.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))

def main():
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    base.config.use_cache = True
    cm = ControllerModel(base, lambda_reg=0.0)
    cm.model.to(device); cm.controller.to(device); cm.w_in.to(device)
    cm.controller.eval(); cm.w_in.eval()
    with open(OUT_DIR / "test.pkl", "rb") as f:
        data = pickle.load(f)

    print("=== A: frozen (untrained LoRA is identity) ===")
    cm.remove_hook()
    rA = run("A", cm, cm.model, tok, data, device)

    print("=== B: frozen LoRA + contrastive controller ===")
    ck = torch.load(OUT_DIR / "ctrl_xlam_contrast.pt", map_location=device)
    cm.controller.load_state_dict(ck["controller_state"])
    cm.w_in.load_state_dict(ck["w_in_state"])
    cm.reinstall_hook()
    rB = run("B", cm, cm.model, tok, data, device)

    print("=== C: frozen LoRA + random controller ===")
    cm.controller = LatentController(cm.hidden).to(device)
    cm.w_in = nn.Linear(64, cm.hidden, bias=False).to(device)
    nn.init.normal_(cm.w_in.weight, std=0.02)
    rC = run("C", cm, cm.model, tok, data, device)

    print("=== D: LoRA-only trained, no controller ===")
    cm.remove_hook()
    load_lora(cm, OUT_DIR / "xlam_lora_only.pt", device)
    rD = run("D", cm, cm.model, tok, data, device)

    print("\n=== Paired bootstrap (exact match) ===")
    res = {}
    for xn, rx, yn, ry in [("B","rB",rB,"rA"), ("D","rD",rD,"rA"),
                           ("B","rB",rB,"rD")]:
        pass
    for (xn, rx), (yn, ry) in [(("B", rB), ("A", rA)), (("D", rD), ("A", rA)),
                              (("B", rB), ("D", rD))]:
        for metric in ("exact", "canon"):
            d, lo, hi = boot(rx[metric], ry[metric])
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            print(f"  {xn}-{yn} [{metric}]: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
            res[f"{xn}-{yn}_{metric}"] = [d, lo, hi]
    res.update({k: float(np.mean(v["exact"])) for k, v in
                [("A", rA), ("B", rB), ("C", rC), ("D", rD)]})
    with open(OUT_DIR / "eval_matrix_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {OUT_DIR / 'eval_matrix_summary.json'}")

if __name__ == "__main__":
    main()