"""gen_tool_dataset.py — Synthetic tool-calling dataset for xLAM."""
import json
import random
import pickle
from pathlib import Path

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\xlam_agency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOOLS = {
    "get_weather": {
        "cities": ["Paris", "London", "Tokyo", "New York", "Berlin", "Moscow"],
        "template": 'What is the weather in {city}?',
        "response": '[{{"name": "get_weather", "arguments": {{"location": "{city}"}}}}]'
    },
    "calculate": {
        "exprs": ["25 * 4", "100 / 5", "18 + 24", "7 * 8", "144 / 12"],
        "template": 'What is {expr}?',
        "response": '[{{"name": "calculate", "arguments": {{"expression": "{expr}"}}}}]'
    },
    "search_web": {
        "queries": ["latest AI news", "quantum computing", "climate change", "Mars rover"],
        "template": 'Search the web for {query}.',
        "response": '[{{"name": "search_web", "arguments": {{"query": "{query}"}}}}]'
    }
}

def generate(n_per_tool=200):
    dataset = []
    for tool_name, data in TOOLS.items():
        for i in range(n_per_tool):
            if tool_name == "get_weather":
                arg = random.choice(data["cities"])
                prompt = data["template"].format(city=arg)
                resp = data["response"].format(city=arg)
            elif tool_name == "calculate":
                arg = random.choice(data["exprs"])
                prompt = data["template"].format(expr=arg)
                resp = data["response"].format(expr=arg)
            else:
                arg = random.choice(data["queries"])
                prompt = data["template"].format(query=arg)
                resp = data["response"].format(query=arg)
            
            dataset.append({"prompt": prompt, "response": resp, "tool": tool_name})
    
    random.shuffle(dataset)
    
    # Split 80/20
    split = int(len(dataset) * 0.8)
    train_data = dataset[:split]
    test_data = dataset[split:]
    
    with open(OUT_DIR / "train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(OUT_DIR / "test.pkl", "wb") as f:
        pickle.dump(test_data, f)
        
    print(f"[Data] Saved {len(train_data)} train, {len(test_data)} test to {OUT_DIR}")
    
    # Verify response format
    print("\n[Samples]")
    for d in train_data[:3]:
        print(f"Tool: {d['tool']}")
        print(f"Prompt: {d['prompt']}")
        print(f"Response: {d['response']}\n")

if __name__ == "__main__":
    generate()