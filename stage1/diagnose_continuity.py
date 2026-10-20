import pickle
from pathlib import Path

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\continuity")

def main():
    with open(OUT_DIR / "continuity_stream.pkl", "rb") as f:
        stream = pickle.load(f)
    
    print("Inspecting PAL attempts (where PAL failed but maybe should have worked):")
    count = 0
    for ep in stream:
        res = ep["oracle_results"]["PAL"]
        if not res["correct"] and count < 3:
            print(f"\n--- Episode {ep['id']} (Regime: {ep['regime']}) ---")
            print(f"Reference: {ep['reference']}")
            print(f"Predicted: {res['pred']}")
            print(f"Raw Response:\n{res['response'][:500]}...")
            count += 1
            
    print("\n" + "="*50)
    print("Inspecting Direct attempts (where Direct failed):")
    count = 0
    for ep in stream:
        res = ep["oracle_results"]["Direct"]
        if not res["correct"] and count < 3:
            print(f"\n--- Episode {ep['id']} (Regime: {ep['regime']}) ---")
            print(f"Reference: {ep['reference']}")
            print(f"Predicted: {res['pred']}")
            print(f"Raw Response:\n{res['response'][:200]}...")
            count += 1

if __name__ == "__main__":
    main()