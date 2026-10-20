# check_gsmhard_field.py
import sys
sys.path.insert(0, '.')
from common.datasets import load_gsm_hard

probs = load_gsm_hard()
for p in probs[:3]:
    print(f"--- i={p['i']} ---")
    print(f"question: {p['question'][:100]}")
    print(f"code[:300]: {p['code'][:300]!r}")
    print(f"reference: {p['reference']!r}")
    print()