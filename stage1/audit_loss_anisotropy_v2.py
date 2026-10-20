"""audit_loss_anisotropy_v2.py — Loss-gradient audit с β (between-class share).
Закрывает дыру в механизме статьи: измеряет, какая доля вариации градиента CE
приходится на межклассовые различия (β = tr(B)/tr(C)).

Для xLAM: классы = tool identity (get_weather, calculate, search_web)
Для Qwen: классы = correctness (easy/hard по frozen-base)
"""
import argparse, pickle, json
import torch, numpy as np
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model import load_model
from common.datasets import load_gsm_hard, gsm_hard_prompt

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\controller")
XLAM_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen", choices=["qwen", "xlam"])
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--lora_frozen", action="store_true",
                   help="If set, freeze LoRA before gradient computation")
    return p.parse_args()

def build_hook(layer, P, D):
    """Build a forward hook that adds P to layer output at positions 1:."""
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        inj = P.to(h.dtype).view(1, 1, D).expand(h.shape[0], h.shape[1]-1, D)
        h2 = torch.cat([h[:, :1, :], h[:, 1:, :] + inj], dim=1)
        return (h2,) + o[1:] if isinstance(o, tuple) else h2
    return hook

def collect_gradients(base, tok, samples, layer, device):
    """Collect ∇_w L for each sample."""
    D = base.config.hidden_size
    P = torch.zeros(D, device=device, requires_grad=True)
    handle = base.model.layers[layer].register_forward_hook(
        build_hook(layer, P, D))
    
    G, cls = [], []
    try:
        for fid, lab, c in samples:
            ids = torch.tensor(fid, device=device).unsqueeze(0)
            lab_t = torch.tensor(lab, device=device).unsqueeze(0)
            if P.grad is not None: P.grad = None
            out = base(input_ids=ids, labels=lab_t)
            out.loss.backward()
            if P.grad is not None:
                G.append(P.grad.detach().cpu().numpy().copy())
                cls.append(c)
            else:
                print(f"  ⚠️  no grad for sample with class {c}")
    finally:
        handle.remove()
    
    return np.stack(G), cls

def compute_spectral_stats(G, cls):
    """Compute v1, effective rank, β, between-class top-2 share."""
    N, D = G.shape
    Gc = G - G.mean(0)
    s = np.linalg.svd(Gc, compute_uv=False) ** 2
    s_norm = s / s.sum()
    v1 = float(s_norm[0])
    cum5 = float(s_norm[:5].sum())
    eff_rank = float(np.exp(-(s_norm[s_norm > 0] * np.log(s_norm[s_norm > 0])).sum()))
    
    stats = {
        "v1": v1,
        "cum_top5": cum5,
        "eff_rank": eff_rank,
    }
    
    if cls[0] is not None and len(set(cls)) > 1:
        names = sorted(set(cls))
        means = np.stack([G[np.array(cls) == k].mean(0) for k in names])
        Gbar = G.mean(0)
        B = (means - Gbar)
        trB = np.sum(B ** 2)
        trC = np.sum(Gc ** 2)
        beta = trB / trC if trC > 1e-12 else 0.0
        
        sb = np.linalg.svd(B, compute_uv=False) ** 2
        sb_norm = sb / sb.sum() if sb.sum() > 1e-12 else sb
        beta_top2 = float(sb_norm[:2].sum()) if len(sb_norm) >= 2 else float(sb_norm.sum())
        
        stats.update({
            "beta": float(beta),
            "beta_top2": beta_top2,
            "n_classes": len(names),
        })
    
    return stats

def load_samples_qwen(tok, n, device):
    """Load GSM-HARD samples with correctness as pseudo-class."""
    problems = load_gsm_hard(n=n)
    samples = []
    for p in problems:
        prompt = gsm_hard_prompt(p["question"])
        resp = p["code"].strip() + "\n#### " + p["reference"]
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        fid = tok(prompt + resp, add_special_tokens=False,
                  truncation=True, max_length=512)["input_ids"]
        lab = list(fid)
        for j in range(min(len(pid), len(lab))): lab[j] = -100
        samples.append((fid, lab, None))  # correctness unknown yet
    
    # Compute correctness on frozen base
    print(f"[audit] Computing correctness on frozen base for {n} samples...")
    tok_q, base_q = load_model("qwen")
    base_q.eval()
    correctness = []
    for i, (fid, lab, _) in enumerate(samples):
        prompt = gsm_hard_prompt(problems[i]["question"])
        inputs = tok_q(prompt, return_tensors="pt").to(base_q.device)
        with torch.no_grad():
            out = base_q.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok_q.eos_token_id)
        resp = tok_q.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        import re
        m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', resp)
        pred = float(m.group(1)) if m else None
        ref = float(problems[i]["reference"])
        ok = pred is not None and abs(pred - ref) < 1e-4
        correctness.append(1 if ok else 0)
    
    samples_with_cls = [(fid, lab, c) for (fid, lab, _), c in zip(samples, correctness)]
    return samples_with_cls

def load_samples_xlam(tok, n, device):
    """Load xLAM tool-calling samples with tool identity as class."""
    with open(XLAM_DIR / "train.pkl", "rb") as f:
        data = pickle.load(f)[:n]
    
    samples = []
    for it in data:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."},
             {"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        full = prompt + it["response"]
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        fid = tok(full, add_special_tokens=False, truncation=True, max_length=256)["input_ids"]
        lab = list(fid)
        for j in range(min(len(pid), len(lab))): lab[j] = -100
        samples.append((fid, lab, it["tool"]))
    return samples

def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    if args.model == "xlam":
        print("[audit] Loading xLAM-2-3b-fc-r...")
        tok = AutoTokenizer.from_pretrained(r"E:\OllamaModels\xLAM-2-3b-fc-r",
                                           trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            r"E:\OllamaModels\xLAM-2-3b-fc-r", dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
        samples = load_samples_xlam(tok, args.n, device)
    else:
        print("[audit] Loading Qwen2.5-3B...")
        tok, base = load_model("qwen")
        samples = load_samples_qwen(tok, args.n, device)
    
    base.config.use_cache = False
    
    # Freeze LoRA if requested
    if args.lora_frozen:
        print("[audit] Freezing LoRA parameters (xLAM v1 condition)...")
        for name, p in base.named_parameters():
            if "lora_" in name:
                p.requires_grad = False
    
    base.eval()
    
    print(f"[audit] Collecting gradients for {len(samples)} samples...")
    G, cls = collect_gradients(base, tok, samples, args.layer, device)
    
    print(f"[audit] Computing spectral statistics...")
    stats = compute_spectral_stats(G, cls)
    
    print("\n" + "="*62)
    print(f"AUDIT RESULTS: {args.model.upper()} (layer {args.layer})")
    print(f"  LoRA frozen: {args.lora_frozen}")
    print("="*62)
    print(f"  v1 (top-1 variance share):     {stats['v1']:.4f}")
    print(f"  cum_top5:                      {stats['cum_top5']:.4f}")
    print(f"  effective rank:                {stats['eff_rank']:.2f}")
    if "beta" in stats:
        print(f"  β (between-class share):       {stats['beta']:.4f}")
        print(f"  β_top2 (between-class top-2):  {stats['beta_top2']:.4f}")
        print(f"  n_classes:                     {stats['n_classes']}")
    
    # Verdict
    print("\n[VERDICT]")
    if stats['v1'] > 0.9:
        print("  ❌ RAY COLLAPSE: gradient is essentially 1D")
    elif stats['v1'] < 0.5:
        print("  ✅ Gradient is multi-directional (v1 < 0.5)")
    
    if "beta" in stats:
        if stats['beta'] < 0.1:
            print(f"  ⚠️  β = {stats['beta']:.3f} is LOW: between-class signal "
                  f"is < 10% of total gradient variance")
            print("     → Controller cannot discriminate classes with CE alone")
            print("     → Contrastive hinge required to boost β_eff")
        elif stats['beta'] < 0.3:
            print(f"  ⚠️  β = {stats['beta']:.3f} is MODERATE")
        else:
            print(f"  ✅ β = {stats['beta']:.3f} is HIGH: between-class signal "
                  f"is well-separated")
    
    print("="*62)
    
    # Save
    out_path = OUT_DIR / f"audit_{args.model}_lora_frozen={args.lora_frozen}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[saved] {out_path}")

if __name__ == "__main__":
    main()