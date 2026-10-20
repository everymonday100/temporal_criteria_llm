"""audit_loss_anisotropy.py — pre-registered prerequisite test A2/A3."""
import argparse, pickle, torch, numpy as np
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model import load_model
from common.datasets import load_gsm_hard, gsm_hard_prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlam", choices=["qwen", "xlam"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--layer", type=int, default=17)
    a = ap.parse_args()
    device = torch.device("cuda")
    if a.model == "xlam":
        tok, base = load_model("xlam")
    else:
        tok, base = load_model("qwen")
    D = base.config.hidden_size
    P = torch.zeros(D, device=device, requires_grad=True)
    handle = base.model.layers[a.layer].register_forward_hook(
        lambda m, i, o: (o[0] + P.to(o[0].dtype).view(1, 1, D).expand(
            o[0].shape[0], o[0].shape[1] - 1, D).pad((0,0,1,0)))
        if isinstance(o, tuple) else o)
    # simpler explicit hook:
    handle.remove()
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        inj = P.to(h.dtype).view(1, 1, D).expand(h.shape[0], h.shape[1]-1, D)
        h2 = torch.cat([h[:, :1, :], h[:, 1:, :] + inj], dim=1)
        return (h2,) + o[1:] if isinstance(o, tuple) else h2
    handle = base.model.layers[a.layer].register_forward_hook(hook)

    # samples: xLAM tools (classes) or GSM-HARD (no classes -> total anisotropy)
    samples = []
    if a.model == "xlam":
        with open(Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")
                  / "train.pkl", "rb") as f:
            data = pickle.load(f)[:a.n]
        for it in data:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."},
                 {"role": "user", "content": it["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            full = prompt + it["response"]
            pid = tok(prompt, add_special_tokens=False)["input_ids"]
            fid = tok(full, add_special_tokens=False, truncation=True,
                      max_length=256)["input_ids"]
            lab = list(fid)
            for j in range(min(len(pid), len(lab))): lab[j] = -100
            samples.append((fid, lab, it["tool"]))
    else:
        for p in load_gsm_hard(n=a.n):
            prompt = gsm_hard_prompt(p["question"])
            resp = p["code"].strip() + "\n#### " + p["reference"]
            pid = tok(prompt, add_special_tokens=False)["input_ids"]
            fid = tok(prompt + resp, add_special_tokens=False,
                      truncation=True, max_length=256)["input_ids"]
            lab = list(fid)
            for j in range(min(len(pid), len(lab))): lab[j] = -100
            samples.append((fid, lab, None))

    G, cls = [], []
    for fid, lab, c in samples:
        ids = torch.tensor(fid, device=device).unsqueeze(0)
        lab_t = torch.tensor(lab, device=device).unsqueeze(0)
        if P.grad is not None: P.grad = None
        out = base(input_ids=ids, labels=lab_t)
        out.loss.backward()
        G.append(P.grad.detach().cpu().numpy().copy())
        cls.append(c)
    G = np.stack(G)                      # (N, D)
    Gc = G - G.mean(0)
    s = np.linalg.svd(Gc, compute_uv=False) ** 2
    v = s / s.sum()
    v1 = float(v[0]); cum5 = float(v[:5].sum())
    eff_rank = float(np.exp(-(v[v > 0] * np.log(v[v > 0])).sum()))
    print(f"[audit] v1={v1:.4f}  cum_top5={cum5:.4f}  eff_rank={eff_rank:.1f}")
    if cls[0] is not None:
        names = sorted(set(cls))
        means = np.stack([G[np.array(cls) == k].mean(0) for k in names])
        B = (means - G.mean(0))
        sb = np.linalg.svd(B, compute_uv=False) ** 2
        sb = sb / sb.sum()
        print(f"[audit] between-class: top1={sb[0]:.4f} top2={sb[:2].sum():.4f}")
        verdict = sb[:2].sum() >= 0.5
    else:
        verdict = v1 <= 0.9
    print("[audit] VERDICT:",
          "directional gradient PRESENT (agency trainable)"
          if verdict else
          "RAY-COLLAPSE regime: CE gives scalar-only gradient "
          "(predict cos(c)≈1; contrastive term required)")

if __name__ == "__main__":
    main()