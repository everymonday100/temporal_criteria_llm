"""test_xlam.py — Sanity check for xLAM-2-3b-fc-r."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"E:\OllamaModels\xLAM-2-3b-fc-r"

print("[Init] Loading tokenizer and model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)
model.eval()

print(f"[Info] Hidden size: {model.config.hidden_size}")
print(f"[Info] Num layers: {model.config.num_hidden_layers}")
print(f"[Info] Vocab size: {model.config.vocab_size}")

# Test generation
messages = [
    {"role": "user", "content": "What is the weather like in Paris right now?"}
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

print("\n[Prompt]")
print(prompt)

print("\n[Generation]")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)