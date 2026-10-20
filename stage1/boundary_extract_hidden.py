"""boundary_extract_hidden.py — Extract h_L from LLM for forward model training."""
import argparse
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model import load_model
from common.boundary_task import build_prompt

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\boundary")
LAYERS = [12, 18, 24]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--streams", required=True)
    p.add_argument("--variant", choices=["F", "L"], default="F")
    p.add_argument("--model", default="qwen")
    return p.parse_args()

@torch.inference_mode()
def extract_variant(args):
    print(f"[extract] Loading streams {args.streams}...")
    with open(OUT_DIR / args.streams, "rb") as f:
        streams = pickle.load(f)
    
    tok, model = load_model(args.model)
    if args.variant == "L":
        lora_path = OUT_DIR / "boundary_lora_best.pt"
        if lora_path.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"[extract] Loaded LoRA from {lora_path}")
        else:
            print("[extract] WARNING: No LoRA found, using frozen model.")
    model.eval()

    dataset = []
    t0 = time.time()

    for i, stream in enumerate(streams):
        history_steps = []
        
        for step in stream.steps:
            # 1. Строим префикс через единую функцию (гарантия совпадения)
            prompt = build_prompt(stream.design, step.y_t, history_steps)
            
            # 2. Строка действия
            if stream.design == "R":
                action_str = f" {step.action:+d}\nFramework:"
            else:
                action_str = f" [SET: {step.action:+d}]\nFramework:"
                
            full_text = prompt + action_str
            inputs = tok(full_text, return_tensors="pt").to(model.device)
            
            # 3. Forward pass
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            last_idx = inputs["input_ids"].shape[1] - 1
            
            h_dict = {f"h_{L}": hs[L][0, last_idx, :].float().cpu().numpy() for L in LAYERS}
            
            dataset.append({
                "stream_id": stream.stream_id,
                "step": step.t,
                "design": stream.design,
                "y_t": step.y_t,
                "action": step.action if step.action is not None else 0,
                "delta_y": step.delta_y,
                "is_external": step.is_external,
                "ext_type": step.external_type or "none",
                **h_dict
            })
            
            # 4. Сохраняем шаг для следующей итерации
            history_steps.append(step)

        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(streams)}] extracted {len(dataset)} steps ({time.time()-t0:.0f}s)")

    out_path = OUT_DIR / f"hidden_{args.streams.replace('.pkl', '')}_{args.variant}.npz"
    save_dict = {}
    for k in dataset[0].keys():
        if isinstance(dataset[0][k], np.ndarray):
            save_dict[k] = np.stack([d[k] for d in dataset])
        else:
            save_dict[k] = np.array([d[k] for d in dataset])
    np.savez(out_path, **save_dict)
    print(f"[saved] {out_path} ({len(dataset)} steps)")

if __name__ == "__main__":
    extract_variant(parse_args())