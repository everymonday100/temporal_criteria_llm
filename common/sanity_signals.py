import sys
sys.path.insert(0, '.')
from common.model import load_model
from common.generation import warmup_signals

tok, m = load_model()
prompts = [
    'What is 2+2?',
    'Explain the P=NP problem in one paragraph.',
    'List five random colors.',
]
for q in prompts:
    s = warmup_signals(m, tok, q, warmup=20, seed=42, need_attn=True)
    print(f"{q[:38]:<40} "
          f"P1={s['P1_mean']:.3f}  "
          f"H={s['H_mean']:.2f}  "
          f"Lam={s['Lambda_mean']:.0f}  "
          f"v_tau={s['v_tau_std']:.0f}  "
          f"A={s['A_attn_mean']:.2f}  "
          f"LL={s['LL_norm_mean']:.0f}")