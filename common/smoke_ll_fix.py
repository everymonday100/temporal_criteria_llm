import sys
sys.path.insert(0, '.')
from common.model import load_model
from common.generation import warmup_signals

tok, m = load_model("qwen")
s = warmup_signals(m, tok, "What is 2+2?", warmup=10)
print("LL_norm_mean:", round(s["LL_norm_mean"], 2))
print("LL_norm_std: ", round(s["LL_norm_std"], 2))