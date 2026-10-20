# smoke_mamba.py
import sys
sys.path.insert(0, '.')
from common.model import load_model
from common.generation import warmup_signals

tok, m = load_model("mamba2")
s = warmup_signals(m, tok, "What is 2+2?", warmup=10,
                   need_attn=False)
for k, v in s.items():
    print(f"  {k:15s} {v:>12.3f}")