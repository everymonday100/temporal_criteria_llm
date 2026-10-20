"""train_xlam_controller.py — Train controller on tool-calling dataset (LoRA disabled)."""
import argparse
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.controller_model import ControllerModel

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")
MODEL_PATH = r"E:\OllamaModels\xLAM-2-3b-fc-r"

class ToolDataset(Dataset):
    def __init__(self, tok, data, max_len=256):
        self.tok = tok
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."},
            {"role": "user", "content": item["prompt"]}
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt + item["response"]
        
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        
        labels = list(full_ids)
        for i in range(len(prompt_ids)):
            labels[i] = -100
            
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "tool_id": {"get_weather": 0, "calculate": 1, "search_web": 2}[item["tool"]]
        }

def collate_fn(batch, pad_token_id):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    tool_ids = []
    for i, x in enumerate(batch):
        l = x["input_ids"].shape[0]
        input_ids[i, :l] = x["input_ids"]
        labels[i, :l] = x["labels"]
        tool_ids.append(x["tool_id"])
    return input_ids, labels, torch.tensor(tool_ids)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    base.config.use_cache = False
    
    ctrl_model = ControllerModel(base, lambda_reg=0.0)
    ctrl_model.model.to(device)
    ctrl_model.controller.to(device)
    ctrl_model.w_in.to(device)
    
    # --- CRITICAL: Disable LoRA training ---
    print("[Setup] Freezing LoRA weights. Only controller will train.")
    for name, p in ctrl_model.model.named_parameters():
        if "lora_" in name:
            p.requires_grad = False
    # --------------------------------------
    
    print("[Data] Loading...")
    with open(OUT_DIR / "train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(OUT_DIR / "test.pkl", "rb") as f:
        test_data = pickle.load(f)
        
    train_ds = ToolDataset(tok, train_data)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))
    
    # Optimizer only updates controller parameters
    optimizer = torch.optim.AdamW([
        {"params": ctrl_model.controller_parameters(), "lr": 1e-3}
    ], weight_decay=0.0)
    
    print("[Train] Starting...")
    epochs = 3
    step = 0
    
    # We will store c_t vectors for each tool to check directions
    tool_vectors = {0: [], 1: [], 2: []}
    
    for epoch in range(epochs):
        ctrl_model.model.train()
        epoch_ce = 0.0
        n_batches = 0
        
        for input_ids, labels, tool_ids in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Lambda is kept at 0 to avoid collapse
            ctrl_model.set_lambda(0.0)
            
            out = ctrl_model(input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()
            
            # Store c_t for each tool
            if out["c_t"] is not None:
                c_t_batch = out["c_t"].detach().cpu()  # [batch, 64]
                for i, t_id in enumerate(tool_ids):
                    tool_vectors[t_id.item()].append(c_t_batch[i])
            
            if (step + 1) % 2 == 0: # grad accum
                torch.nn.utils.clip_grad_norm_(ctrl_model.controller_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_ce += out["ce_loss"].item()
            n_batches += 1
            step += 1
            
            if step % 20 == 0:
                print(f"  Epoch {epoch} Step {step}  CE={epoch_ce/n_batches:.4f}  ||c||={out['c_norms'].mean().item():.4f}")
        
        # End of epoch: check cosine between tools
        print(f"\n[Epoch {epoch}] --- Direction Cosine Similarities ---")
        means = {}
        for t_id, vecs in tool_vectors.items():
            if vecs:
                v = torch.stack(vecs).mean(dim=0)  # [64]
                means[t_id] = v / (v.norm() + 1e-8)
        
        if len(means) == 3:
            print(f"  cos(weather, calculate) = {torch.dot(means[0], means[1]).item():.4f}")
            print(f"  cos(weather, search)    = {torch.dot(means[0], means[2]).item():.4f}")
            print(f"  cos(calculate, search)  = {torch.dot(means[1], means[2]).item():.4f}")
            
        # Clear for next epoch
        tool_vectors = {0: [], 1: [], 2: []}

    # Save
    save_path = OUT_DIR / "ctrl_xlam_frozen_lora.pt"
    torch.save({
        "controller_state": ctrl_model.controller.state_dict(),
        "w_in_state": ctrl_model.w_in.state_dict(),
    }, save_path)
    print(f"\n[saved] {save_path}")

if __name__ == "__main__":
    main()