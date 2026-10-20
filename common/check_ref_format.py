import sys
sys.path.insert(0, '.')
from common.datasets import load_gsm_hard

probs = load_gsm_hard()
print(f"Total: {len(probs)}")
for p in probs[700:715]:
    print(f"  i={p['i']:4d}  ref={p['reference']!r}")

n_int = sum(1 for p in probs
            if p["reference"].lstrip("-").isdigit())
n_dec = sum(1 for p in probs
            if not p["reference"].lstrip("-").isdigit()
            and "." in p["reference"])
print(f"\nIntegers: {n_int}/{len(probs)}")
print(f"Decimals: {n_dec}/{len(probs)}")