"""train_xlam_controller_v2_contrast.py — xLAM + frozen LoRA + contrastive directions.
v2 adds:
  1. Pairwise contrastive hinge on pooled c_t: cross-tool cos pushed below
     m_cross, within-tool cos pulled above m_within (SupCon-style, in-batch).
  2. Magnitude control WITHOUT touching controller_model.py:
     L1 on c_norms + hinge penalty relu(||c|| - c_max).
  3. Balanced round-robin batches (per_tool samples from each tool) so every
     batch contains cross-tool pairs.
  4. Test-set direction table at the end (generalization of separation).
"""
import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.controller_model import ControllerModel

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = r"E:\OllamaModels\xLAM-2-3b-fc-r"
TOOL2ID = {"get_weather": 0, "calculate": 1, "search_web": 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_tool", type=int, default=2, help="samples per tool per batch")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_contrast", type=float, default=1.0)
    p.add_argument("--m_cross", type=float, default=0.5, help="push cross-tool cos below this")
    p.add_argument("--m_within", type=float, default=0.9, help="pull within-tool cos above this")
    p.add_argument("--w_within", type=float, default=0.5)
    p.add_argument("--lambda_l1", type=float, default=2e-2)
    p.add_argument("--lambda_cap", type=float, default=1.0)
    p.add_argument("--c_max", type=float, default=3.0)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class ToolDataset:
    def __init__(self, tok, data, max_len):
        self.tok, self.max_len = tok, max_len
        self.data = data
        self.by_tool = {0: [], 1: [], 2: []}
        for i, item in enumerate(data):
            self.by_tool[TOOL2ID[item["tool"]]].append(i)

    def encode(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that can use tools. "
                        "You are developed by Salesforce xLAM team."},
            {"role": "user", "content": item["prompt"]},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        full_text = prompt + item["response"]
        pids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        fids = self.tok(full_text, add_special_tokens=False,
                        truncation=True, max_length=self.max_len)["input_ids"]
        labels = list(fids)
        for i in range(min(len(pids), len(labels))):
            labels[i] = -100
        return (torch.tensor(fids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                TOOL2ID[item["tool"]])

    def batches(self, per_tool, rng):
        lists = {t: list(v) for t, v in self.by_tool.items()}
        for t in lists:
            rng.shuffle(lists[t])
        n = min(len(v) // per_tool for v in lists.values())
        for b in range(n):
            idxs = []
            for t in (0, 1, 2):
                idxs += lists[t][b * per_tool:(b + 1) * per_tool]
            rng.shuffle(idxs)
            yield [self.encode(i) for i in idxs]


def pad(batch, pad_id):
    L = max(x[0].shape[0] for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (fi, la, _) in enumerate(batch):
        n = fi.shape[0]
        ids[i, :n] = fi
        lab[i, :n] = la
    tools = torch.tensor([x[2] for x in batch])
    return ids, lab, tools


def contrast_loss(c, tools, m_cross, m_within, w_within):
    """Pairwise hinge: cross-tool cos -> below m_cross; within-tool -> above m_within."""
    cn = F.normalize(c, p=2, dim=-1)
    cos = cn @ cn.t()
    t = tools.unsqueeze(0) == tools.unsqueeze(1)
    eye = torch.eye(len(cos), dtype=torch.bool, device=cos.device)
    cross = ~t & ~eye
    same = t & ~eye
    loss = torch.tensor(0.0, device=c.device)
    stats = {}
    if cross.any():
        stats["cross_cos"] = float(cos[cross].detach().mean())
        loss = loss + F.relu(cos[cross] - m_cross).mean()
    if same.any():
        stats["within_cos"] = float(cos[same].mean())
        loss = loss + w_within * F.relu(m_within - cos[same]).mean()
    return loss, stats


def direction_table(cm, ds, device, tag):
    """Per-tool mean directions on a dataset; print pairwise cosines."""
    cm.model.eval()
    acc = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for i in range(0, min(len(ds.data), 120)):
            fi, la, t = ds.encode(i)
            out = cm(fi.unsqueeze(0).to(device))
            if out["c_t"] is not None:
                acc[t].append(out["c_t"].squeeze(0).cpu())
    means = {}
    for t, v in acc.items():
        if v:
            m = torch.stack(v).mean(0)
            means[t] = m / (m.norm() + 1e-8)
    if len(means) == 3:
        c01 = torch.dot(means[0], means[1]).item()
        c02 = torch.dot(means[0], means[2]).item()
        c12 = torch.dot(means[1], means[2]).item()
        print(f"[{tag}] cos(w,c)={c01:.4f}  cos(w,s)={c02:.4f}  cos(c,s)={c12:.4f}")
        return {"cos_wc": c01, "cos_ws": c02, "cos_cs": c12}
    return {}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    base.config.use_cache = False

    cm = ControllerModel(base, lambda_reg=0.0)
    cm.model.to(device); cm.controller.to(device); cm.w_in.to(device)
    for name, p in cm.model.named_parameters():
        p.requires_grad = False  # freeze LoRA + base entirely
    cm.controller.train(); cm.w_in.train()

    with open(OUT_DIR / "train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(OUT_DIR / "test.pkl", "rb") as f:
        test_data = pickle.load(f)
    train_ds = ToolDataset(tok, train_data, args.max_len)
    test_ds = ToolDataset(tok, test_data, args.max_len)

    opt = torch.optim.AdamW(cm.controller_parameters(), lr=args.lr,
                            weight_decay=0.0)

    print("[Train] v2 contrastive. Baseline (v1) had cos~0.998, ||c||~20.")
    step = 0
    for epoch in range(args.epochs):
        sums = {"ce": 0.0, "cn": 0.0, "cc": 0.0, "n": 0}
        opt.zero_grad()
        for batch in train_ds.batches(args.per_tool, rng):
            ids, lab, tools = pad(batch, tok.pad_token_id)
            ids, lab, tools = ids.to(device), lab.to(device), tools.to(device)

            out = cm(ids, labels=lab)
            c, norms = out["c_t"], out["c_norms"]

            # Magnitude control from outside the hook:
            mag = (args.lambda_l1 * norms.mean()
                   + args.lambda_cap * F.relu(norms - args.c_max).mean())
            cl, stats = contrast_loss(c, tools, args.m_cross, args.m_within,
                                      args.w_within)
            loss = out["ce_loss"] + mag + args.lambda_contrast * cl
            (loss / args.grad_accum).backward()

            sums["ce"] += float(out["ce_loss"].item())
            sums["cn"] += float(norms.mean().item())
            sums["cc"] += stats.get("cross_cos", 0.0)
            sums["n"] += 1
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(cm.controller_parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if step % args.log_every == 0:
                n = sums["n"]
                print(f"  ep {epoch} step {step:>4d}  ce={sums['ce']/n:.4f}  "
                      f"||c||={sums['cn']/n:.3f}  cross_cos={sums['cc']/n:.4f}")

        tab = direction_table(cm, test_ds, device, f"epoch {epoch} TEST")

    save_path = OUT_DIR / "ctrl_xlam_contrast.pt"
    torch.save({"controller_state": cm.controller.state_dict(),
                "w_in_state": cm.w_in.state_dict(),
                "args": vars(args)}, save_path)
    print(f"[saved] {save_path}")

    print("\n" + "=" * 62)
    print("CONTRASTIVE VERDICT (xLAM, frozen LoRA)")
    print("=" * 62)
    mx = max(tab.get("cos_wc", 1), tab.get("cos_ws", 1), tab.get("cos_cs", 1))
    if mx < 0.5:
        print("  ✅ Directions SEPARATED across tools on held-out data.")
    elif mx < 0.9:
        print("  ⚠️  PARTIAL separation; raise lambda_contrast or lower m_cross.")
    else:
        print("  ❌ Still constant: contrastive hinge lost to CE. "
              "Try lambda_contrast=5.0.")
    print("=" * 62)


if __name__ == "__main__":
    main()