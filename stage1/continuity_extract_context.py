"""continuity_extract_context.py — Extract h_18 features for each episode."""
import pickle
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\continuity")
LAYER = 18

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    print("[Context] Loading stream...")
    with open(OUT_DIR / "continuity_stream.pkl", "rb") as f:
        stream = pickle.load(f)
        
    print("[Context] Loading model...")
    tok = AutoTokenizer.from_pretrained("E:\\OllamaModels\\Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "E:\\OllamaModels\\Qwen2.5-3B", 
        dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = model.device

    features = []
    
    print(f"[Context] Extracting h_{LAYER} for {len(stream)} episodes...")
    for i, ep in enumerate(stream):
        # We use a neutral prompt to extract the representation of the problem
        # without biasing it towards a specific strategy.
        prompt = f"{ep['question']}\n\nAnswer:"
        inputs = tok(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            
        # Get h_18 at the last token
        h_18 = out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()
        
        features.append({
            "id": ep["id"],
            "regime": ep["regime"],
            "oracle_best_strategy": ep["oracle_best_strategy"],
            "h_18": h_18,
            "correct": {k: v["correct"] for k, v in ep["oracle_results"].items()}
        })
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(stream)}] extracted.")

    # Save features
    out_path = OUT_DIR / "continuity_features.npz"
    np.savez(
        out_path,
        ids=np.array([f["id"] for f in features]),
        regimes=np.array([f["regime"] for f in features]),
        strategies=np.array([f["oracle_best_strategy"] for f in features]),
        h_18=np.stack([f["h_18"] for f in features]),
        correct_direct=np.array([f["correct"]["Direct"] for f in features]),
        correct_cot=np.array([f["correct"]["CoT"] for f in features]),
        correct_pal=np.array([f["correct"]["PAL"] for f in features])
    )
    print(f"[Context] Saved to {out_path}")

if __name__ == "__main__":
    main()