"""train_xlam_lora_only.py — Condition D control: LoRA-only, same tool data,
same weighted CE as the controller run, no controller at all.
Purpose: separate "controller steers behavior" from "fine-tuning aligns vocab".
"""
import argparse, pickle
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")
MODEL_PATH = r"E:\OllamaModels\xLAM-2-3b-fc-r"
SYS = ("You are a helpful assistant that can use tools. "
       "You are developed by Salesforce xLAM team.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

class ToolDS(Dataset):
    def __init__(self, tok, data, max_len):
        self.tok, self.max_len, self.data = tok, max_len, data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        full = prompt + item["response"]
        pids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        fids = self.tok(full, add_special_tokens=False,
                        truncation=True, max_length=self.max_len)["input_ids"]
        lab = list(fids)
        for j in range(min(len(pids), len(lab))):
            lab[j] = -100
        return (torch.tensor(fids, dtype=torch.long),
                torch.tensor(lab, dtype=torch.long))

def collate(b, pad):
    L = max(x[0].shape[0] for x in b)
    ids = torch.full((len(b), L), pad, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    for i, (f, l) in enumerate(b):
        n = f.shape[0]; ids[i, :n] = f; lab[i, :n] = l
    return ids, lab

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    base.config.use_cache = False

    # Same LoRA config as ControllerModel uses -> comparable parameter budget
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                     target_modules=["q_proj","k_proj","v_proj","o_proj",
                                     "gate_proj","up_proj","down_proj"],
                     lora_dropout=0.05)
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    model.to(base.device)

    with open(OUT_DIR / "train.pkl", "rb") as f:
        data = pickle.load(f)
    ds = ToolDS(tok, data, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    step = 0
    for ep in range(args.epochs):
        model.train(); tot = 0.0; n = 0
        opt.zero_grad()
        for ids, lab in dl:
            ids, lab = ids.to(base.device), lab.to(base.device)
            out = model(input_ids=ids, labels=None)
            sl = out.logits[:, :-1, :].contiguous()
            sb = lab[:, 1:].contiguous()
            pt = F.cross_entropy(sl.view(-1, sl.size(-1)), sb.view(-1),
                                 ignore_index=-100, reduction="none")
            with torch.no_grad():
                w = 1.0 + pt.detach().clamp(0.0, 10.0); ws = w.sum()
            loss = (pt * w).sum() / ws.clamp_min(1e-9) / args.grad_accum
            loss.backward()
            tot += loss.item() * args.grad_accum; n += 1; step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if step % 20 == 0:
                print(f"  ep {ep} step {step:>4d}  ce={tot/n:.4f}")
        print(f"[epoch {ep}] train_ce={tot/n:.4f}")

    save = OUT_DIR / "xlam_lora_only.pt"
    torch.save({k: v for k, v in model.state_dict().items() if "lora_" in k},
               save)
    print(f"[saved] {save}")

if __name__ == "__main__":
    main()