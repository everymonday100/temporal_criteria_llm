"""test_xlam_controller.py — Shape check for xLAM with ControllerModel."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.controller_model import ControllerModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"E:\OllamaModels\xLAM-2-3b-fc-r"

print("[Init] Loading xLAM...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True
)

print("[Init] Wrapping with ControllerModel...")
ctrl_model = ControllerModel(base, lambda_reg=0.0)
ctrl_model.model.to(base.device)
ctrl_model.controller.to(base.device)
ctrl_model.w_in.to(base.device)

# Prepare a sample batch
messages = [
    {"role": "system", "content": "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."},
    {"role": "user", "content": "What is the weather in Paris?"}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(base.device)
labels = inputs["input_ids"].clone()

print("[Forward] Running with controller hook active...")
out = ctrl_model(inputs["input_ids"], labels=labels)

print(f"CE Loss: {out['ce_loss'].item():.4f}")
print(f"c_norms shape: {out['c_norms'].shape if out['c_norms'] is not None else None}")
print(f"c_t shape: {out['c_t'].shape if out['c_t'] is not None else None}")

# Check generation
print("\n[Generate] Testing generation with controller...")
gen_out = ctrl_model.generate(**inputs, max_new_tokens=50)
print(tok.decode(gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))